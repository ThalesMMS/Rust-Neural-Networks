use crate::paths;
use rust_neural_networks::config::{validate_config, TrainingConfig};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ConfigEntry {
    /// Project-root-relative path, forward-slash separated, e.g.
    /// `config/training/mnist_mlp_default.json`.
    pub relative_path: String,
    /// Immediate parent folder name under `config/` (e.g. `training`,
    /// `architectures`, `sweeps`, `benchmarks`), or `(root)`.
    pub group: String,
    pub file_name: String,
}

#[tauri::command]
pub fn list_configs() -> Result<Vec<ConfigEntry>, String> {
    let root = paths::config_dir();
    let mut out = Vec::new();
    collect(&root, &root, &mut out).map_err(|e| e.to_string())?;
    out.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(out)
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<ConfigEntry>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, out)?;
        } else if path.extension().is_some_and(|e| e == "json") {
            let rel = path.strip_prefix(root).unwrap();
            let group = rel
                .parent()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "(root)".to_string());
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            out.push(ConfigEntry {
                relative_path: format!("config/{}", rel.to_string_lossy().replace('\\', "/")),
                group,
                file_name,
            });
        }
    }
    Ok(())
}

#[tauri::command]
pub fn read_config(relative_path: String) -> Result<String, String> {
    let path = paths::resolve_relative(&relative_path)?;
    fs::read_to_string(&path).map_err(|e| format!("Failed to read {relative_path}: {e}"))
}

/// Validates then writes JSON config contents to `relative_path`.
///
/// If the JSON looks like a `TrainingConfig` (has a `scheduler_type` field),
/// it is round-tripped through the real `validate_config` used by every
/// training binary, so the UI can never save something the CLI would reject.
/// Other config shapes (architectures, sweeps, benchmarks) only get a JSON
/// syntax check.
#[tauri::command]
pub fn write_config(relative_path: String, contents: String) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_str(&contents).map_err(|e| format!("Invalid JSON: {e}"))?;

    if value.get("scheduler_type").is_some() {
        let cfg: TrainingConfig = serde_json::from_value(value)
            .map_err(|e| format!("Does not match the training config schema: {e}"))?;
        validate_config(&cfg).map_err(|e| e.to_string())?;
    }

    let path = paths::resolve_relative(&relative_path)?;
    fs::write(&path, contents).map_err(|e| format!("Failed to write {relative_path}: {e}"))
}

#[tauri::command]
pub fn save_config_as(relative_path: String, contents: String) -> Result<(), String> {
    let path: PathBuf = paths::resolve_relative(&relative_path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_config(relative_path, contents)
}
