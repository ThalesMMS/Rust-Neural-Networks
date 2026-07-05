use crate::paths;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DataStatus {
    pub mnist_present: bool,
    pub cifar10_present: bool,
}

#[tauri::command]
pub fn data_status() -> DataStatus {
    let data = paths::data_dir();
    let mnist_present = [
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
    ]
    .iter()
    .all(|f| data.join(f).is_file());

    let cifar_dir = data.join("cifar-10-batches-bin");
    let cifar10_present = (1..=5).all(|i| cifar_dir.join(format!("data_batch_{i}.bin")).is_file())
        && cifar_dir.join("test_batch.bin").is_file();

    DataStatus { mnist_present, cifar10_present }
}

/// Returns whether each given project-root-relative path currently exists,
/// in the same order as the input (used to show "trained / not trained yet"
/// badges without one round-trip per checkpoint).
#[tauri::command]
pub fn checkpoint_status(relative_paths: Vec<String>) -> Vec<bool> {
    let root = paths::project_root();
    relative_paths.iter().map(|p| root.join(p).is_file()).collect()
}
