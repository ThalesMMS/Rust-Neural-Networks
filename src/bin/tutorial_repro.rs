//! Tutorial reproducibility runner.
//!
//! Reads `tutorial_repro_manifest.json` and runs the configured smoke checks.
//!
//! Usage:
//!   cargo run --bin tutorial_repro -- list
//!   cargo run --bin tutorial_repro -- run --path beginner
//!   cargo run --bin tutorial_repro -- run --all
//!   cargo run --bin tutorial_repro -- run --path intermediate --dry-run

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_MANIFEST_PATH: &str = "tutorial_repro_manifest.json";
const DEFAULT_ARTIFACTS_DIR: &str = "target/tutorial_runs";

#[derive(Debug, Deserialize)]
struct Manifest {
    version: u64,
    paths: BTreeMap<String, PathEntry>,
}

#[derive(Debug, Deserialize)]
struct PathEntry {
    title: String,
    checks: Vec<Check>,
}

#[derive(Debug, Deserialize)]
struct Check {
    id: String,
    description: String,
    command: Vec<String>,
    workdir: String,
    timeout_seconds: u64,
    expect: Option<Expect>,
    #[serde(default)]
    require: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Expect {
    stdout_contains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy)]
enum Outcome {
    Passed,
    Failed,
}

fn usage() -> String {
    let bin = "tutorial_repro";
    format!(
        "Tutorial reproducibility runner\n\nUsage:\n  {bin} list [--manifest <path>]\n  {bin} run (--all | --path <name>) [--dry-run] [--manifest <path>]\n\nOptions:\n  --manifest <path>     Path to manifest JSON (default: {DEFAULT_MANIFEST_PATH})\n  --dry-run             Print resolved commands without executing\n  --include-gpu         Include GPU/accelerator checks (manifested as require: ['gpu'])\n  --include-wasm        Include WASM checks (manifested as require: ['wasm'])\n  --artifacts-dir <p>   Directory for runner-created outputs (default: {DEFAULT_ARTIFACTS_DIR})\n\nExit codes:\n  0 success\n  2 CLI/manifest error\n  3 at least one check failed\n"
    )
}

fn main() {
    let mut args = std::env::args().skip(1);

    let Some(subcmd) = args.next() else {
        eprintln!("{}", usage());
        std::process::exit(2);
    };

    let mut manifest_path: PathBuf = PathBuf::from(DEFAULT_MANIFEST_PATH);
    let mut dry_run = false;
    let mut include_gpu = false;
    let mut include_wasm = false;
    let mut artifacts_dir: PathBuf = PathBuf::from(DEFAULT_ARTIFACTS_DIR);

    // Parse flags that can appear anywhere after the subcommand.
    let mut positional: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => {
                let Some(p) = args.next() else {
                    eprintln!("--manifest requires a value\n\n{}", usage());
                    std::process::exit(2);
                };
                manifest_path = PathBuf::from(p);
            }
            "--dry-run" => dry_run = true,
            "--include-gpu" => include_gpu = true,
            "--include-wasm" => include_wasm = true,
            "--artifacts-dir" => {
                let Some(p) = args.next() else {
                    eprintln!("--artifacts-dir requires a value\n\n{}", usage());
                    std::process::exit(2);
                };
                artifacts_dir = PathBuf::from(p);
            }
            "-h" | "--help" => {
                eprintln!("{}", usage());
                std::process::exit(0);
            }
            _ => positional.push(arg),
        }
    }

    let manifest = match load_manifest(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load manifest {}: {e}", manifest_path.display());
            std::process::exit(2);
        }
    };

    match subcmd.as_str() {
        "list" => {
            if !positional.is_empty() {
                eprintln!("Unexpected arguments for list\n\n{}", usage());
                std::process::exit(2);
            }
            list_paths(&manifest);
        }
        "run" => {
            let (run_all, path) = match parse_run_args(&positional) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}\n\n{}", usage());
                    std::process::exit(2);
                }
            };

            let failed = if run_all {
                run_all_paths(
                    &manifest,
                    dry_run,
                    include_gpu,
                    include_wasm,
                    &artifacts_dir,
                )
            } else {
                let path = path.expect("path required when not --all");
                run_one_path(
                    &manifest,
                    &path,
                    dry_run,
                    include_gpu,
                    include_wasm,
                    &artifacts_dir,
                )
            };

            if failed {
                std::process::exit(3);
            }
        }
        "-h" | "--help" => {
            eprintln!("{}", usage());
        }
        _ => {
            eprintln!("Unknown subcommand: {subcmd}\n\n{}", usage());
            std::process::exit(2);
        }
    }
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let data = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    if manifest.version != 1 {
        return Err(format!(
            "Unsupported manifest version {} (expected 1)",
            manifest.version
        ));
    }
    Ok(manifest)
}

fn list_paths(manifest: &Manifest) {
    println!("Available learning paths:\n");
    for (id, entry) in &manifest.paths {
        println!("- {id}: {} ({} checks)", entry.title, entry.checks.len());
    }
}

fn parse_run_args(args: &[String]) -> Result<(bool, Option<String>), String> {
    let mut run_all = false;
    let mut path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => {
                run_all = true;
                i += 1;
            }
            "--path" => {
                let Some(p) = args.get(i + 1) else {
                    return Err("--path requires a value".to_string());
                };
                path = Some(p.to_string());
                i += 2;
            }
            other => return Err(format!("Unknown argument for run: {other}")),
        }
    }

    if run_all == path.is_some() {
        return Err("Provide exactly one of: --all or --path <name>".to_string());
    }

    Ok((run_all, path))
}

fn run_all_paths(
    manifest: &Manifest,
    dry_run: bool,
    include_gpu: bool,
    include_wasm: bool,
    artifacts_dir: &Path,
) -> bool {
    let mut any_failed = false;
    for (id, _) in &manifest.paths {
        if run_one_path(
            manifest,
            id,
            dry_run,
            include_gpu,
            include_wasm,
            artifacts_dir,
        ) {
            any_failed = true;
        }
    }
    any_failed
}

fn run_one_path(
    manifest: &Manifest,
    path_id: &str,
    dry_run: bool,
    include_gpu: bool,
    include_wasm: bool,
    artifacts_dir: &Path,
) -> bool {
    let Some(path) = manifest.paths.get(path_id) else {
        eprintln!("Unknown path: {path_id}");
        return true;
    };

    println!("== Path: {path_id} ({}) ==", path.title);

    let mut any_failed = false;
    for check in &path.checks {
        match run_check(
            check,
            dry_run,
            include_gpu,
            include_wasm,
            path_id,
            artifacts_dir,
        ) {
            Outcome::Passed => {}
            Outcome::Failed => any_failed = true,
        }
    }

    if any_failed {
        eprintln!("== Path result: {path_id}: FAILED ==");
    } else {
        println!("== Path result: {path_id}: PASSED ==");
    }

    any_failed
}

fn run_check(
    check: &Check,
    dry_run: bool,
    include_gpu: bool,
    include_wasm: bool,
    path_id: &str,
    artifacts_dir: &Path,
) -> Outcome {
    if should_skip_check(check, include_gpu, include_wasm) {
        println!("\n--- Check: {} ---\n{}", check.id, check.description);
        println!(
            "Result: SKIPPED (missing prerequisites: {:?})",
            check.require
        );
        return Outcome::Passed;
    }

    println!("\n--- Check: {} ---\n{}", check.id, check.description);

    let (program, argv) = match check.command.split_first() {
        Some(v) => v,
        None => {
            eprintln!("Empty command for check {}", check.id);
            return Outcome::Failed;
        }
    };

    let workdir = PathBuf::from(&check.workdir);

    let check_artifacts_dir = artifacts_dir.join(path_id).join(&check.id);

    if dry_run {
        println!("[dry-run] workdir: {}", workdir.display());
        println!("[dry-run] artifacts-dir: {}", check_artifacts_dir.display());
        println!("[dry-run] command: {} {}", program, argv.join(" "));
        return Outcome::Passed;
    }

    if let Err(e) = fs::create_dir_all(&check_artifacts_dir) {
        eprintln!(
            "Failed to create artifacts dir {}: {e}",
            check_artifacts_dir.display()
        );
        return Outcome::Failed;
    }

    let timeout = Duration::from_secs(check.timeout_seconds);

    let start = Instant::now();
    let output =
        match run_command_with_timeout(program, argv, &workdir, timeout, &check_artifacts_dir) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Check {} failed to run: {e}", check.id);
                return Outcome::Failed;
            }
        };

    let elapsed = start.elapsed();
    let status = output.status;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !status.success() {
        eprintln!("Exit status: {}", display_exit_status(status));
        if !stdout.trim().is_empty() {
            eprintln!("--- stdout ---\n{stdout}");
        }
        if !stderr.trim().is_empty() {
            eprintln!("--- stderr ---\n{stderr}");
        }
        return Outcome::Failed;
    }

    if let Some(expect) = &check.expect {
        if let Some(needles) = &expect.stdout_contains {
            for needle in needles {
                if !stdout.contains(needle) {
                    eprintln!("Expected stdout to contain {needle:?} but it did not.");
                    eprintln!("--- stdout ---\n{stdout}");
                    return Outcome::Failed;
                }
            }
        }
    }

    println!("Result: PASSED in {:.2?}", elapsed);
    Outcome::Passed
}

fn should_skip_check(check: &Check, include_gpu: bool, include_wasm: bool) -> bool {
    for req in &check.require {
        match req.as_str() {
            "gpu" if !include_gpu => return true,
            "wasm" if !include_wasm => return true,
            _ => {}
        }
    }
    false
}

fn display_exit_status(status: ExitStatus) -> String {
    match status.code() {
        Some(code) => code.to_string(),
        None => "terminated by signal".to_string(),
    }
}

fn run_command_with_timeout(
    program: &str,
    argv: &[String],
    workdir: &Path,
    timeout: Duration,
    artifacts_dir: &Path,
) -> Result<std::process::Output, String> {
    let mut child = Command::new(program)
        .args(argv)
        .current_dir(workdir)
        .env("TUTORIAL_REPRO_ARTIFACTS_DIR", artifacts_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    let start = Instant::now();
    loop {
        if let Some(_status) = child.try_wait().map_err(|e| e.to_string())? {
            return child.wait_with_output().map_err(|e| e.to_string());
        }

        if start.elapsed() > timeout {
            let _ = child.kill();
            return Err(format!("timed out after {:?}", timeout));
        }

        std::thread::sleep(Duration::from_millis(50));
    }
}
