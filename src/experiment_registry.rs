use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io;
use std::path::Path;

use crate::experiment_registry_run_id;

type Result<T> = std::result::Result<T, io::Error>;

/// Minimal schema version for run records.
pub const RUN_RECORD_SCHEMA_VERSION: &str = "0";

pub const DEFAULT_REGISTRY_DIR: &str = "runs";
pub const RUN_RECORD_FILENAME: &str = "run.json";

/// Minimal placeholder dataset metadata when the training code doesn't expose
/// sizes or preprocessing details.
///
/// This keeps the registry schema consistent while allowing binaries to fill in
/// richer values over time.
pub fn dataset_placeholder(name: impl Into<String>) -> DatasetMetadata {
    DatasetMetadata {
        name: name.into(),
        train_size: None,
        val_size: None,
        test_size: None,
        preprocessing: None,
        source: None,
    }
}

pub fn collect_environment() -> Environment {
    Environment {
        git: git_info_from_repo(),
        crate_version: option_env!("CARGO_PKG_VERSION").map(|s| s.to_string()),
        rustc_version: rustc_version_best_effort(),
        os: Some(format!(
            "{} {} ({})",
            std::env::consts::OS,
            std::env::consts::ARCH,
            std::env::consts::FAMILY
        )),
    }
}

fn rustc_version_best_effort() -> Option<String> {
    use std::process::Command;

    let output = Command::new("rustc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    Some(s.trim().to_string())
}

fn git_info_from_repo() -> Option<GitInfo> {
    use std::process::Command;

    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string());

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| !s.trim().is_empty());

    if commit.is_none() && dirty.is_none() {
        return None;
    }

    Some(GitInfo { commit, dirty })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub schema_version: String,

    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_name: Option<String>,

    /// RFC3339 / ISO8601 timestamp.
    pub timestamp_start: String,
    /// RFC3339 / ISO8601 timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_end: Option<String>,

    pub model_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    pub status: RunStatus,

    pub seed: u64,

    pub config: ConfigSnapshot,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetMetadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Artifacts>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<Environment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parsed: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_size: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocessing: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<DatasetSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Metrics {
    pub epochs_completed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_train_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_val_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_val_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_training_time_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifacts {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_log_csv: Option<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plots: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Environment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<GitInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crate_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GitInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
}

/// Writes a run record to `<registry_dir>/<run_id>/run.json`.
///
/// The file is written atomically by writing to a temp file in the same directory
/// and then renaming.
pub fn write_run_record(registry_dir: impl AsRef<Path>, record: &RunRecord) -> Result<()> {
    let dir = experiment_registry_run_id::ensure_run_dir(registry_dir, &record.run_id)?;

    let final_path = dir.join(RUN_RECORD_FILENAME);
    let tmp_path = dir.join(format!("{}.tmp", RUN_RECORD_FILENAME));

    let json = serde_json::to_vec_pretty(record)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    fs::write(&tmp_path, json)?;

    // Best-effort atomic replace.
    if final_path.exists() {
        let _ = fs::remove_file(&final_path);
    }
    fs::rename(&tmp_path, &final_path)?;

    Ok(())
}

pub fn load_run_record(path: impl AsRef<Path>) -> Result<RunRecord> {
    let contents = fs::read_to_string(path)?;
    let record = serde_json::from_str(&contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(record)
}

pub fn run_record_from_sweep_result(
    sweep_result: &crate::sweep::SweepResult,
    run_id: impl Into<String>,
) -> RunRecord {
    let now = chrono::Utc::now().to_rfc3339();

    RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION.to_string(),
        run_id: run_id.into(),
        run_name: None,
        timestamp_start: now.clone(),
        timestamp_end: Some(now),
        model_type: "unknown".to_string(),
        command: None,
        status: RunStatus::Completed,
        seed: 0,
        config: ConfigSnapshot {
            config_path: None,
            config_format: Some("sweep_result".to_string()),
            raw: None,
            parsed: Some(serde_json::to_value(sweep_result).unwrap_or(Value::Null)),
        },
        dataset: None,
        metrics: Some(Metrics {
            epochs_completed: sweep_result.epochs_completed,
            final_train_loss: Some(sweep_result.final_train_loss as f64),
            final_val_loss: Some(sweep_result.final_val_loss as f64),
            final_val_accuracy: Some(sweep_result.final_val_accuracy as f64),
            total_training_time_seconds: Some(sweep_result.total_training_time as f64),
        }),
        artifacts: Some(Artifacts {
            training_log_csv: Some(sweep_result.log_file.clone()),
            checkpoints: Vec::new(),
            plots: Vec::new(),
            extra: None,
        }),
        environment: Some(collect_environment()),
    }
}
