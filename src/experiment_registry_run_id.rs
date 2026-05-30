use std::fs;
use std::io;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, io::Error>;

pub fn generate_run_id() -> String {
    // Spec example: `2026-05-28T20-00-00Z_ab12cd34`
    // Use an RFC3339-like UTC timestamp with colon replaced for filesystem-friendliness,
    // plus a short (8 hex) suffix for uniqueness.
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();

    // Random-ish suffix without introducing new dependencies.
    // (Not cryptographically secure; good enough for unique run IDs.)
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let pid = std::process::id();
    let suffix = ((nanos as u64) ^ ((pid as u64) << 16)) as u32;

    format!("{ts}_{suffix:08x}")
}

pub fn run_dir(registry_dir: impl AsRef<Path>, run_id: &str) -> PathBuf {
    registry_dir.as_ref().join(run_id)
}

pub fn ensure_run_dir(registry_dir: impl AsRef<Path>, run_id: &str) -> Result<PathBuf> {
    let dir = run_dir(registry_dir, run_id);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}
