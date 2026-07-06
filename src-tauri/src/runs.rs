use crate::{models, paths};
use rust_neural_networks::experiment_registry::{load_run_record, RunRecord, RunStatus};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Lists runs from the experiment registry (`runs/<id>/run.json`). Only
/// `mnist_mlp`, `mnist_cnn`, and the hyperparameter sweep tool populate this
/// today; other models show up via [`list_all_logs`] instead.
#[tauri::command]
pub fn list_runs() -> Vec<RunRecord> {
    let dir = paths::runs_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut runs: Vec<RunRecord> = entries
        .flatten()
        .filter_map(|e| {
            let run_json = e.path().join("run.json");
            run_json.is_file().then(|| load_run_record(&run_json).ok()).flatten()
        })
        .collect();
    runs.sort_by(|a, b| b.timestamp_start.cmp(&a.timestamp_start));
    runs
}

const AUX_LOG_EXCLUDE: [&str; 4] = ["gradient", "latent", "sample", "attention_maps"];
const DEFAULT_PREVIEW_BYTES: usize = 24_000;
const MAX_PREVIEW_BYTES: usize = 128_000;

#[derive(Debug, Clone, Serialize)]
pub struct LogSummary {
    pub file_name: String,
    pub relative_path: String,
    pub row_count: usize,
    pub last_epoch: Option<usize>,
    pub last_train_loss: Option<f64>,
    pub last_val_loss: Option<f64>,
    pub last_val_accuracy: Option<f64>,
    pub modified_unix_secs: u64,
}

fn parse_last_row(contents: &str) -> (usize, Option<usize>, Option<f64>, Option<f64>, Option<f64>) {
    let parsed = parse_training_series_contents(contents);
    let Some(last) = parsed.rows.last() else {
        return (0, None, None, None, None);
    };
    (
        parsed.rows.len(),
        Some(last.epoch),
        Some(last.train_loss),
        Some(last.val_loss),
        Some(last.val_accuracy),
    )
}

/// Scans `logs/` directly (excluding auxiliary logs) so every training
/// binary shows up in the history view, even the ones that don't write to
/// the `runs/` registry.
///
/// Most binaries write `.csv`, but `mnist_mlp` writes a `.txt` file with the
/// same 5-column CSV contents — both extensions are treated as candidates.
/// Files that don't parse as a single valid metrics row are skipped, which
/// also filters out unrelated free-text `.txt` notes that happen to live in
/// `logs/`.
#[tauri::command]
pub fn list_all_logs() -> Vec<LogSummary> {
    let dir = paths::logs_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "csv" && e != "txt") {
            continue;
        }
        let Some(file_name) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };
        if AUX_LOG_EXCLUDE.iter().any(|kw| file_name.contains(kw)) {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        let (row_count, last_epoch, last_train_loss, last_val_loss, last_val_accuracy) =
            parse_last_row(&contents);
        if row_count == 0 || last_epoch.is_none() {
            continue;
        }
        let modified_unix_secs = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        out.push(LogSummary {
            file_name: file_name.clone(),
            relative_path: format!("logs/{file_name}"),
            row_count,
            last_epoch,
            last_train_loss,
            last_val_loss,
            last_val_accuracy,
            modified_unix_secs,
        });
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.modified_unix_secs));
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct CsvRow {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_time: f64,
    pub val_loss: f64,
    pub val_accuracy: f64,
}

/// Reads the full row set of a `logs/*.csv` file for charting. Tolerates
/// extra trailing columns (e.g. `resnet_cifar10`'s `learning_rate` column).
#[tauri::command]
pub fn read_log_csv(relative_path: String) -> Result<Vec<CsvRow>, String> {
    Ok(read_training_series(relative_path)?
        .into_iter()
        .map(|row| CsvRow {
            epoch: row.epoch,
            train_loss: row.train_loss,
            train_time: row.train_time,
            val_loss: row.val_loss,
            val_accuracy: row.val_accuracy,
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentSource {
    Registry,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Completed,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnvironmentSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_dirty: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crate_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    pub key: String,
    pub source: ExperimentSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub model_type: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_end: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_unix_secs: Option<u64>,
    pub status: ExperimentStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs_completed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_train_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_val_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_val_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_val_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_val_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_training_time_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_epoch_time_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_log_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_log_path: Option<String>,
    pub checkpoints: Vec<String>,
    pub plots: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_raw: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_parsed: Option<Value>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingPoint {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_time: f64,
    pub val_loss: f64,
    pub val_accuracy: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GradientPoint {
    pub epoch: usize,
    pub layer_name: String,
    pub grad_norm_weights: f64,
    pub grad_norm_biases: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactPreview {
    pub relative_path: String,
    pub kind: String,
    pub content: String,
    pub truncated: bool,
    pub bytes_total: usize,
    pub bytes_returned: usize,
}

#[derive(Debug, Clone, Default)]
struct ParsedTrainingSeries {
    rows: Vec<TrainingPoint>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct MetricRollup {
    epochs_completed: Option<usize>,
    final_train_loss: Option<f64>,
    final_val_loss: Option<f64>,
    final_val_accuracy: Option<f64>,
    best_val_accuracy: Option<f64>,
    best_val_loss: Option<f64>,
    total_training_time_seconds: Option<f64>,
    average_epoch_time_seconds: Option<f64>,
}

#[tauri::command]
pub fn list_experiments() -> Vec<ExperimentSummary> {
    collect_experiments(list_runs(), list_all_logs(), true)
}

#[tauri::command]
pub fn read_training_series(relative_path: String) -> Result<Vec<TrainingPoint>, String> {
    let normalized = normalize_relative_path(&relative_path)
        .ok_or_else(|| "training log path must not be empty".to_string())?;
    let path = resolve_project_relative(&normalized)?;
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {normalized}: {e}"))?;
    Ok(parse_training_series_contents(&contents).rows)
}

#[tauri::command]
pub fn read_gradient_csv(relative_path: String) -> Result<Vec<GradientPoint>, String> {
    let normalized = normalize_relative_path(&relative_path)
        .ok_or_else(|| "gradient log path must not be empty".to_string())?;
    let path = resolve_project_relative(&normalized)?;
    let contents = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {normalized}: {e}"))?;
    Ok(parse_gradient_csv_contents(&contents))
}

#[tauri::command]
pub fn read_artifact_preview(
    relative_path: String,
    max_bytes: Option<usize>,
) -> Result<ArtifactPreview, String> {
    let normalized = safe_preview_relative_path(&relative_path)?;
    let bytes = fs::read(resolve_project_relative(&normalized)?)
        .map_err(|e| format!("Failed to read {normalized}: {e}"))?;
    Ok(build_artifact_preview(
        &normalized,
        &bytes,
        max_bytes.unwrap_or(DEFAULT_PREVIEW_BYTES),
    ))
}

fn collect_experiments(
    runs: Vec<RunRecord>,
    logs: Vec<LogSummary>,
    read_artifacts: bool,
) -> Vec<ExperimentSummary> {
    let mut claimed_logs = HashSet::new();
    let mut out: Vec<ExperimentSummary> = runs
        .into_iter()
        .map(|run| registry_experiment(run, &mut claimed_logs, read_artifacts))
        .collect();

    out.extend(
        logs.into_iter()
            .filter(|log| !claimed_logs.contains(&log.relative_path))
            .map(|log| log_experiment(log, read_artifacts)),
    );

    out.sort_by(|a, b| experiment_sort_key(b).cmp(&experiment_sort_key(a)));
    out
}

fn registry_experiment(
    run: RunRecord,
    claimed_logs: &mut HashSet<String>,
    read_artifacts: bool,
) -> ExperimentSummary {
    let artifacts = run.artifacts.as_ref();
    let training_log_path = artifacts
        .and_then(|a| a.training_log_csv.as_deref())
        .and_then(normalize_relative_path);
    if let Some(path) = &training_log_path {
        claimed_logs.insert(path.clone());
    }

    let gradient_log_path = artifacts
        .and_then(|a| a.extra.as_ref())
        .and_then(gradient_path_from_extra)
        .or_else(|| infer_existing_gradient_path(&run.model_type));

    let checkpoints = artifacts
        .map(|a| a.checkpoints.iter().filter_map(|p| normalize_relative_path(p)).collect())
        .unwrap_or_default();
    let plots = artifacts
        .map(|a| a.plots.iter().filter_map(|p| normalize_relative_path(p)).collect())
        .unwrap_or_default();
    let model_id = known_model_id(&run.model_type);
    let label = run
        .run_name
        .as_ref()
        .map(|name| format!("{} ({name})", run.model_type))
        .unwrap_or_else(|| run.model_type.clone());

    let mut warnings = Vec::new();
    let rollup = if read_artifacts {
        rollup_for_log(training_log_path.as_deref(), &mut warnings)
    } else {
        MetricRollup::default()
    };
    warn_if_missing("training log", training_log_path.as_deref(), &mut warnings);
    warn_if_missing("gradient log", gradient_log_path.as_deref(), &mut warnings);

    let metrics = run.metrics.as_ref();
    let epochs = metrics.map(|m| m.epochs_completed).or(rollup.epochs_completed);
    let total_time = metrics
        .and_then(|m| m.total_training_time_seconds)
        .or(rollup.total_training_time_seconds);

    ExperimentSummary {
        key: format!("registry:{}", run.run_id),
        source: ExperimentSource::Registry,
        run_id: Some(run.run_id),
        model_id,
        model_type: run.model_type,
        label,
        timestamp_start: Some(run.timestamp_start),
        timestamp_end: run.timestamp_end,
        modified_unix_secs: None,
        status: experiment_status(&run.status),
        config_path: run.config.config_path.and_then(|p| normalize_relative_path(&p)),
        dataset_name: run.dataset.as_ref().map(|d| d.name.clone()),
        epochs_completed: epochs,
        final_train_loss: metrics.and_then(|m| m.final_train_loss).or(rollup.final_train_loss),
        final_val_loss: metrics.and_then(|m| m.final_val_loss).or(rollup.final_val_loss),
        final_val_accuracy: metrics
            .and_then(|m| m.final_val_accuracy)
            .or(rollup.final_val_accuracy),
        best_val_accuracy: rollup
            .best_val_accuracy
            .or_else(|| metrics.and_then(|m| m.final_val_accuracy)),
        best_val_loss: rollup.best_val_loss.or_else(|| metrics.and_then(|m| m.final_val_loss)),
        total_training_time_seconds: total_time,
        average_epoch_time_seconds: rollup
            .average_epoch_time_seconds
            .or_else(|| average_epoch_time(total_time, epochs)),
        training_log_path,
        gradient_log_path,
        checkpoints,
        plots,
        command: run.command,
        seed: Some(run.seed),
        environment: run.environment.map(|env| EnvironmentSummary {
            git_commit: env.git.as_ref().and_then(|g| g.commit.clone()),
            git_dirty: env.git.as_ref().and_then(|g| g.dirty),
            crate_version: env.crate_version,
            rustc_version: env.rustc_version,
            os: env.os,
        }),
        config_raw: run.config.raw,
        config_parsed: run.config.parsed,
        warnings,
    }
}

fn log_experiment(log: LogSummary, read_artifacts: bool) -> ExperimentSummary {
    let (model_id, model_type) = infer_model_from_log(&log.relative_path);
    let dataset_name = dataset_for_model(&model_type);
    let mut warnings = Vec::new();
    let rollup = if read_artifacts {
        rollup_for_log(Some(&log.relative_path), &mut warnings)
    } else {
        MetricRollup::default()
    };
    let gradient_log_path = infer_existing_gradient_path(&model_type);
    warn_if_missing("gradient log", gradient_log_path.as_deref(), &mut warnings);

    ExperimentSummary {
        key: format!("log:{}", log.relative_path),
        source: ExperimentSource::Log,
        run_id: None,
        model_id,
        model_type,
        label: log.file_name.trim_end_matches(".csv").trim_end_matches(".txt").to_string(),
        timestamp_start: None,
        timestamp_end: None,
        modified_unix_secs: Some(log.modified_unix_secs),
        status: ExperimentStatus::Unknown,
        config_path: None,
        dataset_name,
        epochs_completed: rollup.epochs_completed.or(log.last_epoch),
        final_train_loss: rollup.final_train_loss.or(log.last_train_loss),
        final_val_loss: rollup.final_val_loss.or(log.last_val_loss),
        final_val_accuracy: rollup.final_val_accuracy.or(log.last_val_accuracy),
        best_val_accuracy: rollup.best_val_accuracy.or(log.last_val_accuracy),
        best_val_loss: rollup.best_val_loss.or(log.last_val_loss),
        total_training_time_seconds: rollup.total_training_time_seconds,
        average_epoch_time_seconds: rollup.average_epoch_time_seconds,
        training_log_path: Some(log.relative_path),
        gradient_log_path,
        checkpoints: Vec::new(),
        plots: Vec::new(),
        command: None,
        seed: None,
        environment: None,
        config_raw: None,
        config_parsed: None,
        warnings,
    }
}

fn rollup_for_log(path: Option<&str>, warnings: &mut Vec<String>) -> MetricRollup {
    let Some(path) = path else {
        return MetricRollup::default();
    };
    let normalized = match normalize_relative_path(path) {
        Some(p) => p,
        None => return MetricRollup::default(),
    };
    let resolved = match resolve_project_relative(&normalized) {
        Ok(path) => path,
        Err(e) => {
            warnings.push(e);
            return MetricRollup::default();
        }
    };
    match fs::read_to_string(resolved) {
        Ok(contents) => {
            let parsed = parse_training_series_contents(&contents);
            warnings.extend(parsed.warnings);
            summarize_training(&parsed.rows)
        }
        Err(e) => {
            warnings.push(format!("Could not read training log {normalized}: {e}"));
            MetricRollup::default()
        }
    }
}

fn summarize_training(rows: &[TrainingPoint]) -> MetricRollup {
    let Some(last) = rows.last() else {
        return MetricRollup::default();
    };
    let total_time: f64 = rows.iter().map(|r| r.train_time).sum();
    MetricRollup {
        epochs_completed: Some(last.epoch),
        final_train_loss: Some(last.train_loss),
        final_val_loss: Some(last.val_loss),
        final_val_accuracy: Some(last.val_accuracy),
        best_val_accuracy: rows.iter().map(|r| r.val_accuracy).reduce(f64::max),
        best_val_loss: rows.iter().map(|r| r.val_loss).reduce(f64::min),
        total_training_time_seconds: Some(total_time),
        average_epoch_time_seconds: average_epoch_time(Some(total_time), Some(rows.len())),
    }
}

fn parse_training_series_contents(contents: &str) -> ParsedTrainingSeries {
    let mut out = ParsedTrainingSeries::default();
    let mut skipped = 0usize;
    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || (line_no == 0 && line.starts_with("epoch")) {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 5 {
            skipped += 1;
            continue;
        }
        let parsed = (
            fields[0].parse::<usize>(),
            fields[1].parse::<f64>(),
            fields[2].parse::<f64>(),
            fields[3].parse::<f64>(),
            fields[4].parse::<f64>(),
        );
        if let (Ok(epoch), Ok(train_loss), Ok(train_time), Ok(val_loss), Ok(val_accuracy)) = parsed {
            out.rows.push(TrainingPoint {
                epoch,
                train_loss,
                train_time,
                val_loss,
                val_accuracy,
                learning_rate: fields.get(5).and_then(|v| v.parse::<f64>().ok()),
            });
        } else {
            skipped += 1;
        }
    }
    if skipped > 0 {
        out.warnings.push(format!("Skipped {skipped} malformed training row(s)."));
    }
    out
}

fn parse_gradient_csv_contents(contents: &str) -> Vec<GradientPoint> {
    let mut rows = Vec::new();
    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || (line_no == 0 && line.starts_with("epoch")) {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 4 {
            continue;
        }
        if let (Ok(epoch), Ok(grad_norm_weights), Ok(grad_norm_biases)) = (
            fields[0].parse::<usize>(),
            fields[2].parse::<f64>(),
            fields[3].parse::<f64>(),
        ) {
            rows.push(GradientPoint {
                epoch,
                layer_name: fields[1].to_string(),
                grad_norm_weights,
                grad_norm_biases,
            });
        }
    }
    rows
}

fn build_artifact_preview(relative_path: &str, bytes: &[u8], max_bytes: usize) -> ArtifactPreview {
    let limit = max_bytes.clamp(1, MAX_PREVIEW_BYTES);
    let returned = bytes.len().min(limit);
    ArtifactPreview {
        relative_path: relative_path.to_string(),
        kind: artifact_kind(relative_path).to_string(),
        content: String::from_utf8_lossy(&bytes[..returned]).to_string(),
        truncated: returned < bytes.len(),
        bytes_total: bytes.len(),
        bytes_returned: returned,
    }
}

fn artifact_kind(relative_path: &str) -> &'static str {
    match Path::new(relative_path).extension().and_then(|e| e.to_str()) {
        Some("json") => "json",
        Some("csv") => "csv",
        Some("html") => "html",
        _ => "text",
    }
}

fn normalize_relative_path(path: &str) -> Option<String> {
    let mut normalized = path.trim().replace('\\', "/");
    if normalized.is_empty() {
        return None;
    }

    let root = paths::project_root().to_string_lossy().replace('\\', "/");
    if let Some(stripped) = normalized.strip_prefix(&(root + "/")) {
        normalized = stripped.to_string();
    }
    while let Some(stripped) = normalized.strip_prefix("./") {
        normalized = stripped.to_string();
    }

    (!normalized.is_empty()).then_some(normalized)
}

fn resolve_project_relative(relative_path: &str) -> Result<PathBuf, String> {
    if relative_path.starts_with('/') || relative_path.split('/').any(|part| part == "..") {
        return Err(format!("path must stay inside the project: {relative_path}"));
    }
    paths::resolve_relative(relative_path)
}

fn safe_preview_relative_path(path: &str) -> Result<String, String> {
    let normalized = normalize_relative_path(path)
        .ok_or_else(|| "artifact path must not be empty".to_string())?;
    if normalized.starts_with('/') || normalized.split('/').any(|part| part == "..") {
        return Err(format!("artifact path must stay inside the project: {normalized}"));
    }
    let allowed = normalized.starts_with("logs/")
        || normalized.starts_with("runs/")
        || matches!(
            normalized.as_str(),
            "demo/dashboard_data.json"
                | "demo/architecture_dashboard.html"
                | "demo/gradient_viz.html"
                | "demo/index.html"
        );
    if !allowed {
        return Err(format!("artifact preview is not allowed for {normalized}"));
    }
    Ok(normalized)
}

fn gradient_path_from_extra(extra: &Value) -> Option<String> {
    extra.get("gradient_log_csv")
        .and_then(Value::as_str)
        .and_then(normalize_relative_path)
}

fn infer_existing_gradient_path(model_type: &str) -> Option<String> {
    let candidate = match model_type {
        "mnist_mlp" => "logs/gradients_mlp.csv",
        "mnist_cnn" => "logs/gradients_cnn.csv",
        "mnist_attention" | "mnist_attention_pool" | "transformer_mnist" => {
            "logs/gradients_attention.csv"
        }
        "cifar10_cnn" => "logs/gradients_cifar10.csv",
        _ => return None,
    };
    resolve_project_relative(candidate)
        .ok()
        .filter(|p| p.is_file())
        .map(|_| candidate.to_string())
}

fn infer_model_from_log(path: &str) -> (Option<String>, String) {
    let lower = path.to_ascii_lowercase();
    let model_type = if lower.contains("resnet") {
        "resnet_cifar10"
    } else if lower.contains("vit") {
        "cifar10_vit"
    } else if lower.contains("cifar10") {
        "cifar10_cnn"
    } else if lower.contains("gan") {
        "mnist_gan"
    } else if lower.contains("vae") {
        "mnist_vae"
    } else if lower.contains("ae") || lower.contains("autoencoder") {
        "mnist_autoencoder"
    } else if lower.contains("attention") {
        "mnist_attention"
    } else if lower.contains("cnn") {
        "mnist_cnn"
    } else {
        "mnist_mlp"
    };
    (known_model_id(model_type), model_type.to_string())
}

fn known_model_id(model_type: &str) -> Option<String> {
    models::MODEL_REGISTRY
        .iter()
        .find(|m| m.id == model_type || m.bin_name == model_type)
        .map(|m| m.id.to_string())
}

fn dataset_for_model(model_type: &str) -> Option<String> {
    if model_type.contains("cifar10") || model_type.contains("resnet") {
        Some("cifar10".to_string())
    } else if model_type.contains("mnist") {
        Some("mnist".to_string())
    } else {
        None
    }
}

fn experiment_status(status: &RunStatus) -> ExperimentStatus {
    match status {
        RunStatus::Completed => ExperimentStatus::Completed,
        RunStatus::Failed => ExperimentStatus::Failed,
    }
}

fn average_epoch_time(total_time: Option<f64>, epochs: Option<usize>) -> Option<f64> {
    let epochs = epochs?;
    let total_time = total_time?;
    (epochs > 0).then_some(total_time / epochs as f64)
}

fn warn_if_missing(label: &str, path: Option<&str>, warnings: &mut Vec<String>) {
    let Some(path) = path else {
        return;
    };
    if resolve_project_relative(path).ok().is_none_or(|p| !p.is_file()) {
        warnings.push(format!("Missing {label}: {path}"));
    }
}

fn experiment_sort_key(exp: &ExperimentSummary) -> String {
    exp.timestamp_start
        .clone()
        .unwrap_or_else(|| format!("{:020}", exp.modified_unix_secs.unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_neural_networks::experiment_registry::{
        Artifacts, ConfigSnapshot, Metrics, RunRecord, RunStatus,
    };

    #[test]
    fn parses_training_series_with_optional_learning_rate() {
        let parsed = parse_training_series_contents(
            "epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate\n\
             1,0.9,1.5,0.8,70.0,0.01\n\
             2,0.7,1.25,0.6,75.5,0.005\n",
        );
        assert_eq!(parsed.rows.len(), 2);
        assert_eq!(parsed.rows[0].epoch, 1);
        assert_eq!(parsed.rows[1].learning_rate, Some(0.005));
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn parses_gradient_rows() {
        let rows = parse_gradient_csv_contents(
            "epoch,layer_name,grad_norm_weights,grad_norm_biases\n\
             1,hidden_layer,0.85,0.11\n\
             1,output_layer,0.83,0.06\n",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].layer_name, "hidden_layer");
        assert_eq!(rows[1].grad_norm_biases, 0.06);
    }

    #[test]
    fn normalizes_leading_dot_artifact_paths() {
        assert_eq!(
            normalize_relative_path("./logs/training_loss_cnn.csv"),
            Some("logs/training_loss_cnn.csv".to_string())
        );
    }

    #[test]
    fn registry_logs_claim_matching_raw_logs() {
        let run = RunRecord {
            schema_version: "0".to_string(),
            run_id: "run-1".to_string(),
            run_name: None,
            timestamp_start: "2026-07-04T00:00:00Z".to_string(),
            timestamp_end: None,
            model_type: "mnist_cnn".to_string(),
            command: None,
            status: RunStatus::Completed,
            seed: 1,
            config: ConfigSnapshot {
                config_path: None,
                config_format: None,
                raw: None,
                parsed: None,
            },
            dataset: None,
            metrics: Some(Metrics {
                epochs_completed: 1,
                final_train_loss: Some(0.5),
                final_val_loss: Some(0.4),
                final_val_accuracy: Some(90.0),
                total_training_time_seconds: Some(2.0),
            }),
            artifacts: Some(Artifacts {
                training_log_csv: Some("./logs/a.csv".to_string()),
                checkpoints: Vec::new(),
                plots: Vec::new(),
                extra: None,
            }),
            environment: None,
        };
        let log = LogSummary {
            file_name: "a.csv".to_string(),
            relative_path: "logs/a.csv".to_string(),
            row_count: 1,
            last_epoch: Some(1),
            last_train_loss: Some(0.5),
            last_val_loss: Some(0.4),
            last_val_accuracy: Some(90.0),
            modified_unix_secs: 1,
        };
        let experiments = collect_experiments(vec![run], vec![log], false);
        assert_eq!(experiments.len(), 1);
        assert_eq!(experiments[0].source, ExperimentSource::Registry);
    }

    #[test]
    fn artifact_preview_truncates_content() {
        let preview = build_artifact_preview("logs/a.csv", b"1234567890", 4);
        assert_eq!(preview.kind, "csv");
        assert_eq!(preview.content, "1234");
        assert!(preview.truncated);
        assert_eq!(preview.bytes_total, 10);
        assert_eq!(preview.bytes_returned, 4);
    }
}
