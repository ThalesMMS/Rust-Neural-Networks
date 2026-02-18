//! Generates `demo/dashboard_data.json` for the Architecture Comparison Dashboard.
//!
//! Instantiates the layers used by each MNIST model (MLP, CNN, Attention) to
//! compute parameter counts, FLOPS, and memory footprint. Reads per-epoch CSV
//! training logs from `logs/` and combines everything into a single JSON file
//! consumed by `demo/architecture_dashboard.html`.
//!
//! # Usage
//!
//! ```bash
//! # First, train the models to produce CSV logs:
//! cargo run --release --bin mnist_mlp
//! cargo run --release --bin mnist_cnn
//! cargo run --release --bin mnist_attention_pool
//!
//! # Then generate the dashboard data:
//! cargo run --release --bin generate_dashboard_data
//! ```

use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use serde::Serialize;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ============================================================================
// JSON Output Structures
// ============================================================================

#[derive(Debug, Serialize)]
struct DashboardData {
    generated_at: String,
    models: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
struct ModelData {
    id: String,
    name: String,
    description: String,
    architecture: ArchitectureData,
    training: Option<TrainingData>,
}

#[derive(Debug, Serialize)]
struct ArchitectureData {
    layers: Vec<LayerInfo>,
    total_params: usize,
    total_flops: u64,
    memory_bytes: usize,
}

#[derive(Debug, Serialize)]
struct LayerInfo {
    name: String,
    #[serde(rename = "type")]
    layer_type: String,
    params: usize,
    flops: u64,
}

#[derive(Debug, Serialize)]
struct TrainingData {
    config: TrainingConfig,
    epochs: Vec<EpochData>,
    best_accuracy: f32,
    total_time: f32,
}

#[derive(Debug, Serialize)]
struct TrainingConfig {
    optimizer: String,
    lr: f32,
    batch_size: usize,
}

#[derive(Debug, Serialize)]
struct EpochData {
    epoch: usize,
    train_loss: f32,
    val_loss: f32,
    val_accuracy: f32,
    train_time: f32,
    lr: f32,
}

// ============================================================================
// CSV Parsing
// ============================================================================

/// Reads a CSV training log with format:
/// `epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate`
fn read_training_csv(path: &str) -> Option<Vec<EpochData>> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut epochs = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.ok()?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        // Skip header line
        if i == 0 && line.starts_with("epoch") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let epoch: usize = parts[0].trim().parse().ok()?;
        let train_loss: f32 = parts[1].trim().parse().ok()?;
        let train_time: f32 = parts[2].trim().parse().ok()?;
        let val_loss: f32 = parts[3].trim().parse().ok()?;
        let val_accuracy: f32 = parts[4].trim().parse().ok()?;
        let lr: f32 = parts[5].trim().parse().ok()?;

        epochs.push(EpochData {
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            train_time,
            lr,
        });
    }

    if epochs.is_empty() {
        None
    } else {
        Some(epochs)
    }
}

// ============================================================================
// Model Architecture Definitions
// ============================================================================

fn build_mlp_data(rng: &mut SimpleRng) -> ModelData {
    // MLP: 784 -> 512 -> 10
    let hidden = DenseLayer::new(784, 512, rng);
    let output = DenseLayer::new(512, 10, rng);

    let layers = vec![
        LayerInfo {
            name: "Dense 784 -> 512".to_string(),
            layer_type: "Dense".to_string(),
            params: hidden.parameter_count(),
            flops: hidden.flops_forward(1),
        },
        LayerInfo {
            name: "Dense 512 -> 10".to_string(),
            layer_type: "Dense".to_string(),
            params: output.parameter_count(),
            flops: output.flops_forward(1),
        },
    ];

    let total_params: usize = layers.iter().map(|l| l.params).sum();
    let total_flops: u64 = layers.iter().map(|l| l.flops).sum();
    let memory_bytes = hidden.parameter_memory_bytes() + output.parameter_memory_bytes();

    // Try to read CSV log — MLP uses "./logs/training_loss_adam.txt"
    let training = read_training_csv("./logs/training_loss_adam.txt").map(|epochs| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer: "Adam".to_string(),
                lr: epochs.first().map_or(0.001, |e| e.lr),
                batch_size: 64,
            },
            epochs,
            best_accuracy,
            total_time,
        }
    });

    ModelData {
        id: "mnist_mlp".to_string(),
        name: "MNIST MLP".to_string(),
        description: "2-layer MLP (784 -> 512 -> 10) with BLAS-accelerated GEMM".to_string(),
        architecture: ArchitectureData {
            layers,
            total_params,
            total_flops,
            memory_bytes,
        },
        training,
    }
}

fn build_cnn_data(rng: &mut SimpleRng) -> ModelData {
    // CNN: Conv2D(1,8,3,pad=1,stride=1,28,28) -> MaxPool(2x2) -> Dense(1568,10)
    let conv = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, rng);
    // After conv: 8 channels x 28x28, after MaxPool 2x2 stride 2: 8 x 14 x 14 = 1568
    let fc = DenseLayer::new(1568, 10, rng);

    let layers = vec![
        LayerInfo {
            name: "Conv2D 1 -> 8, 3x3".to_string(),
            layer_type: "Conv2D".to_string(),
            params: conv.parameter_count(),
            flops: conv.flops_forward(1),
        },
        LayerInfo {
            name: "MaxPool 2x2".to_string(),
            layer_type: "MaxPool".to_string(),
            params: 0,
            flops: 0,
        },
        LayerInfo {
            name: "Dense 1568 -> 10".to_string(),
            layer_type: "Dense".to_string(),
            params: fc.parameter_count(),
            flops: fc.flops_forward(1),
        },
    ];

    let total_params: usize = layers.iter().map(|l| l.params).sum();
    let total_flops: u64 = layers.iter().map(|l| l.flops).sum();
    let memory_bytes = conv.parameter_memory_bytes() + fc.parameter_memory_bytes();

    let training = read_training_csv("./logs/training_loss_cnn.csv").map(|epochs| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer: "SGD".to_string(),
                lr: epochs.first().map_or(0.01, |e| e.lr),
                batch_size: 64,
            },
            epochs,
            best_accuracy,
            total_time,
        }
    });

    ModelData {
        id: "mnist_cnn".to_string(),
        name: "MNIST CNN".to_string(),
        description: "Conv(8,3x3) + MaxPool(2x2) + Dense(1568 -> 10), grayscale 28x28".to_string(),
        architecture: ArchitectureData {
            layers,
            total_params,
            total_flops,
            memory_bytes,
        },
        training,
    }
}

fn build_attention_data() -> ModelData {
    // Attention model parameters (from mnist_attention_pool.rs constants):
    // PATCH=4, GRID=7, SEQ_LEN=49, PATCH_DIM=16, D_MODEL=64, FF_DIM=128, NUM_CLASSES=10
    const PATCH_DIM: usize = 16;
    const D_MODEL: usize = 64;
    const FF_DIM: usize = 128;
    const NUM_CLASSES: usize = 10;

    // Patch projection: w_patch[PATCH_DIM * D_MODEL] + b_patch[D_MODEL]
    let patch_params = PATCH_DIM * D_MODEL + D_MODEL;
    // 2 * PATCH_DIM * D_MODEL per sample (matmul)
    let patch_flops = 2 * PATCH_DIM as u64 * D_MODEL as u64;

    // Q, K, V projections: each w[D_MODEL * D_MODEL] + b[D_MODEL]
    let qkv_params = 3 * (D_MODEL * D_MODEL + D_MODEL);
    // 3 projections, each 2 * D_MODEL * D_MODEL per token, 49 tokens
    let qkv_flops = 3 * 2 * 49 * D_MODEL as u64 * D_MODEL as u64;

    // Attention scores: Q @ K^T → 49x49, then softmax, then @ V → 49xD_MODEL
    let attn_flops = 2 * 49 * 49 * D_MODEL as u64 + 2 * 49 * D_MODEL as u64 * 49;

    // Feed-forward: w_ff1[D_MODEL * FF_DIM] + b_ff1[FF_DIM] + w_ff2[FF_DIM * D_MODEL] + b_ff2[D_MODEL]
    let ff_params = D_MODEL * FF_DIM + FF_DIM + FF_DIM * D_MODEL + D_MODEL;
    // 2 matmuls over 49 tokens each
    let ff_flops = 49 * (2 * D_MODEL as u64 * FF_DIM as u64 + 2 * FF_DIM as u64 * D_MODEL as u64);

    // Classifier: w_cls[D_MODEL * NUM_CLASSES] + b_cls[NUM_CLASSES]
    let cls_params = D_MODEL * NUM_CLASSES + NUM_CLASSES;
    let cls_flops = 2 * D_MODEL as u64 * NUM_CLASSES as u64;

    let layers = vec![
        LayerInfo {
            name: "Patch Embed 16 -> 64".to_string(),
            layer_type: "Linear".to_string(),
            params: patch_params,
            flops: patch_flops,
        },
        LayerInfo {
            name: "Self-Attention Q/K/V (d=64)".to_string(),
            layer_type: "Attention".to_string(),
            params: qkv_params,
            flops: qkv_flops + attn_flops,
        },
        LayerInfo {
            name: "Feed-Forward 64 -> 128 -> 64".to_string(),
            layer_type: "FFN".to_string(),
            params: ff_params,
            flops: ff_flops,
        },
        LayerInfo {
            name: "Classifier 64 -> 10".to_string(),
            layer_type: "Dense".to_string(),
            params: cls_params,
            flops: cls_flops,
        },
    ];

    let total_params: usize = layers.iter().map(|l| l.params).sum();
    let total_flops: u64 = layers.iter().map(|l| l.flops).sum();
    let memory_bytes = total_params * 4;

    let training = read_training_csv("./logs/training_loss_attention.csv").map(|epochs| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer: "SGD".to_string(),
                lr: epochs.first().map_or(0.01, |e| e.lr),
                batch_size: 32,
            },
            epochs,
            best_accuracy,
            total_time,
        }
    });

    ModelData {
        id: "mnist_attention".to_string(),
        name: "MNIST Attention".to_string(),
        description:
            "Transformer-style: Patch(4x4) -> Embed(64) -> Self-Attention -> FFN(128) -> Classify"
                .to_string(),
        architecture: ArchitectureData {
            layers,
            total_params,
            total_flops,
            memory_bytes,
        },
        training,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut rng = SimpleRng::new(42);

    println!("Generating dashboard data...");

    let models = vec![
        build_mlp_data(&mut rng),
        build_cnn_data(&mut rng),
        build_attention_data(),
    ];

    // Print summary
    for model in &models {
        let has_training = if model.training.is_some() {
            "YES"
        } else {
            "NO"
        };
        println!(
            "  {} — {} params, {} FLOPS, training data: {}",
            model.name,
            model.architecture.total_params,
            model.architecture.total_flops,
            has_training
        );
    }

    // Generate timestamp
    let generated_at = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Simple ISO 8601 approximation
        let days_since_epoch = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;
        // Simple year/month/day calculation
        let mut remaining_days = days_since_epoch;
        let mut year = 1970u64;
        loop {
            let days_in_year = if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
                366
            } else {
                365
            };
            if remaining_days < days_in_year {
                break;
            }
            remaining_days -= days_in_year;
            year += 1;
        }
        let days_in_months: [u64; 12] = if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 0usize;
        for (i, &d) in days_in_months.iter().enumerate() {
            if remaining_days < d {
                month = i + 1;
                break;
            }
            remaining_days -= d;
        }
        let day = remaining_days + 1;
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            year, month, day, hours, minutes, seconds
        )
    };

    let dashboard = DashboardData {
        generated_at,
        models,
    };

    // Ensure output directory exists
    let output_path = Path::new("demo/dashboard_data.json");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create demo/ directory");
    }

    let json =
        serde_json::to_string_pretty(&dashboard).expect("Failed to serialize dashboard data");
    fs::write(output_path, &json).expect("Failed to write demo/dashboard_data.json");

    println!("\nDashboard data written to demo/dashboard_data.json");
    println!("Open demo/architecture_dashboard.html in a browser to view.");
}
