use crate::paths;
use std::fs;
use std::time::UNIX_EPOCH;

/// Auxiliary log files that are never the "main" per-epoch metrics log for a
/// run (gradient norms, latent-space dumps, generated samples, attention
/// maps) — excluded when guessing which CSV belongs to a run.
const AUX_LOG_EXCLUDE: [&str; 4] = ["gradient", "latent", "sample", "attention_maps"];

/// `mnist_mlp` writes its per-epoch metrics CSV with a `.txt` extension
/// (`training_loss_<optimizer>.txt`) even though the contents are the same
/// 5-column CSV format every other binary uses — so both extensions are
/// treated as candidate metrics logs here.
fn is_metrics_log_extension(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|e| e == "csv" || e == "txt")
}

/// Finds the `logs/` file most likely written by the run that started at
/// `since_unix_secs`: the newest-modified metrics log (excluding auxiliary
/// logs) whose modification time is at or after the run start (with a
/// couple of seconds of slack for filesystem timestamp granularity).
///
/// This avoids hardcoding per-binary log filenames, several of which depend
/// on compile-time constants rather than the run's config.
#[tauri::command]
pub fn find_active_log(since_unix_secs: u64) -> Option<String> {
    let dir = paths::logs_dir();
    let entries = fs::read_dir(&dir).ok()?;

    let mut best: Option<(u64, String)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_metrics_log_extension(&path) {
            continue;
        }
        let Some(file_name) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };
        if AUX_LOG_EXCLUDE.iter().any(|kw| file_name.contains(kw)) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else { continue };
        let Ok(modified) = metadata.modified() else { continue };
        let Ok(modified_secs) = modified.duration_since(UNIX_EPOCH) else { continue };
        let modified_secs = modified_secs.as_secs();

        if modified_secs + 2 < since_unix_secs {
            continue;
        }
        if best.as_ref().is_none_or(|(t, _)| modified_secs > *t) {
            best = Some((modified_secs, format!("logs/{file_name}")));
        }
    }
    best.map(|(_, p)| p)
}
