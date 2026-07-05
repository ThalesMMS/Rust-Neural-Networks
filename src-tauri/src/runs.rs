use crate::paths;
use rust_neural_networks::experiment_registry::{load_run_record, RunRecord};
use serde::Serialize;
use std::fs;
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
    let mut lines = contents.lines();
    lines.next(); // header
    let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    let row_count = rows.len();
    let Some(last) = rows.last() else {
        return (0, None, None, None, None);
    };
    let fields: Vec<&str> = last.split(',').collect();
    let get_usize = |i: usize| fields.get(i).and_then(|s| s.trim().parse::<usize>().ok());
    let get_f64 = |i: usize| fields.get(i).and_then(|s| s.trim().parse::<f64>().ok());
    (row_count, get_usize(0), get_f64(1), get_f64(3), get_f64(4))
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
    let path = paths::resolve_relative(&relative_path)?;
    let contents = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for line in contents.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 5 {
            continue;
        }
        if let (Ok(epoch), Ok(train_loss), Ok(train_time), Ok(val_loss), Ok(val_accuracy)) =
            (f[0].parse(), f[1].parse(), f[2].parse(), f[3].parse(), f[4].parse())
        {
            rows.push(CsvRow { epoch, train_loss, train_time, val_loss, val_accuracy });
        }
    }
    Ok(rows)
}
