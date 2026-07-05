use std::path::{Path, PathBuf};

/// Absolute path to the repository root, resolved at compile time from this
/// crate's manifest location (`<repo>/src-tauri/Cargo.toml`). Using a
/// compile-time-baked path means every filesystem lookup is correct
/// regardless of the process's current working directory when the app is
/// launched (double-clicked, launched from Finder, `tauri dev`, etc).
pub fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri always has a parent directory")
        .to_path_buf()
}

pub fn data_dir() -> PathBuf {
    project_root().join("data")
}

pub fn config_dir() -> PathBuf {
    project_root().join("config")
}

pub fn logs_dir() -> PathBuf {
    project_root().join("logs")
}

pub fn runs_dir() -> PathBuf {
    project_root().join("runs")
}

pub fn release_dir() -> PathBuf {
    project_root().join("target").join("release")
}

/// Resolves a path the frontend sent (relative to the project root, using
/// forward slashes) into an absolute path, rejecting attempts to escape the
/// project root via `..` segments.
pub fn resolve_relative(rel: &str) -> Result<PathBuf, String> {
    if rel.contains("..") {
        return Err(format!("path must not contain '..': {rel}"));
    }
    Ok(project_root().join(rel))
}
