use crate::models::{find_model, ArgStyle};
use crate::paths;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, State};

/// Maps a run token to the currently-running child process for that run
/// (either the `cargo build` step or the training binary itself, whichever
/// is active), so `stop_run` can kill it.
pub type ProcessMap = Arc<Mutex<HashMap<String, Arc<Mutex<Child>>>>>;

#[derive(Debug, Clone, Serialize)]
pub struct LogLine {
    pub run_token: String,
    pub stream: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunFinished {
    pub run_token: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StartedRun {
    pub run_token: String,
    pub started_unix_secs: u64,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StartRunArgs {
    pub model_id: String,
    pub config_path: Option<String>,
    pub arch_path: Option<String>,
    pub step: bool,
    pub run_name: Option<String>,
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn emit_line(app: &AppHandle, token: &str, stream: &str, line: impl Into<String>) {
    let _ = app.emit(
        "training-log",
        LogLine {
            run_token: token.to_string(),
            stream: stream.to_string(),
            line: line.into(),
        },
    );
}

/// Spawns `cmd`, registers the child under `token` (replacing whatever was
/// there before, e.g. a prior build step), streams its stdout/stderr as
/// `training-log` events line-by-line, and blocks until it exits.
fn run_and_stream(
    app: &AppHandle,
    map: &ProcessMap,
    token: &str,
    mut cmd: Command,
) -> std::io::Result<std::process::ExitStatus> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let shared = Arc::new(Mutex::new(child));
    map.lock().unwrap().insert(token.to_string(), shared.clone());

    let mut handles = Vec::new();
    if let Some(stdout) = stdout {
        let app = app.clone();
        let token = token.to_string();
        handles.push(std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                emit_line(&app, &token, "stdout", line);
            }
        }));
    }
    if let Some(stderr) = stderr {
        let app = app.clone();
        let token = token.to_string();
        handles.push(std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                emit_line(&app, &token, "stderr", line);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    let status = shared.lock().unwrap().wait()?;
    Ok(status)
}

fn build_argv(model: &crate::models::ModelDescriptor, args: &StartRunArgs) -> Vec<String> {
    let mut argv = Vec::new();
    let config_path = args
        .config_path
        .clone()
        .or_else(|| model.default_config_path.map(|s| s.to_string()));

    match model.arg_style {
        ArgStyle::Positional => {
            if let Some(cfg) = &config_path {
                argv.push(cfg.clone());
            }
        }
        ArgStyle::ConfigFlag => {
            if let Some(cfg) = &config_path {
                argv.push("--config".to_string());
                argv.push(cfg.clone());
            }
        }
    }

    if let Some(default_arch) = model.default_arch_path {
        let arch = args.arch_path.clone().unwrap_or_else(|| default_arch.to_string());
        argv.push("--arch".to_string());
        argv.push(arch);
    }

    if args.step && model.supports_step {
        argv.push("--step".to_string());
    }

    if model.supports_registry_flags {
        if let Some(name) = &args.run_name {
            argv.push("--run-name".to_string());
            argv.push(name.clone());
        }
    }

    argv
}

#[tauri::command]
pub fn start_run(
    app: AppHandle,
    state: State<ProcessMap>,
    args: StartRunArgs,
) -> Result<StartedRun, String> {
    let model = find_model(&args.model_id).ok_or_else(|| format!("Unknown model id: {}", args.model_id))?;
    let root = paths::project_root();
    let bin_name = model.bin_name.to_string();
    let token = format!(
        "{}-{}",
        bin_name,
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    );
    let argv = build_argv(model, &args);
    let started_unix_secs = now_unix_secs();

    let map = state.inner().clone();
    let app_bg = app.clone();
    let token_bg = token.clone();
    let root_bg = root.clone();
    let argv_bg = argv.clone();

    std::thread::spawn(move || {
        emit_line(&app_bg, &token_bg, "system", format!("Building {bin_name} (release)..."));
        let mut build_cmd = Command::new("cargo");
        build_cmd
            .current_dir(&root_bg)
            .args(["build", "--release", "--bin", &bin_name]);
        let build_status = match run_and_stream(&app_bg, &map, &token_bg, build_cmd) {
            Ok(s) => s,
            Err(e) => {
                emit_line(&app_bg, &token_bg, "system", format!("Failed to start build: {e}"));
                let _ = app_bg.emit(
                    "training-finished",
                    RunFinished { run_token: token_bg.clone(), success: false, message: e.to_string() },
                );
                map.lock().unwrap().remove(&token_bg);
                return;
            }
        };

        if !build_status.success() {
            emit_line(&app_bg, &token_bg, "system", "Build failed.");
            let _ = app_bg.emit(
                "training-finished",
                RunFinished {
                    run_token: token_bg.clone(),
                    success: false,
                    message: "cargo build failed".to_string(),
                },
            );
            map.lock().unwrap().remove(&token_bg);
            return;
        }

        emit_line(&app_bg, &token_bg, "system", "Build finished. Starting run...");

        let exe = paths::release_dir().join(&bin_name);
        let mut run_cmd = Command::new(exe);
        run_cmd.current_dir(&root_bg).args(&argv_bg);

        let run_status = match run_and_stream(&app_bg, &map, &token_bg, run_cmd) {
            Ok(s) => s,
            Err(e) => {
                emit_line(&app_bg, &token_bg, "system", format!("Failed to start run: {e}"));
                let _ = app_bg.emit(
                    "training-finished",
                    RunFinished { run_token: token_bg.clone(), success: false, message: e.to_string() },
                );
                map.lock().unwrap().remove(&token_bg);
                return;
            }
        };

        map.lock().unwrap().remove(&token_bg);
        let success = run_status.success();
        emit_line(
            &app_bg,
            &token_bg,
            "system",
            if success { "Run finished." } else { "Run exited with an error." },
        );
        let _ = app_bg.emit(
            "training-finished",
            RunFinished {
                run_token: token_bg.clone(),
                success,
                message: format!("exit status: {run_status}"),
            },
        );
    });

    Ok(StartedRun { run_token: token, started_unix_secs, argv })
}

#[tauri::command]
pub fn stop_run(state: State<ProcessMap>, run_token: String) -> Result<(), String> {
    let map = state.inner().clone();
    let child = map.lock().unwrap().remove(&run_token);
    match child {
        Some(child) => {
            let mut child = child.lock().unwrap();
            child.kill().map_err(|e| e.to_string())?;
            Ok(())
        }
        None => Err("No running process for that run token (it may have already finished).".to_string()),
    }
}
