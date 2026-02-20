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

/// Parse a training log CSV file into a vector of EpochData.
///
/// Expects lines with the columns:
/// `epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate`.
/// An optional header row that begins with "epoch" is skipped. If at least one
/// valid epoch row is parsed, returns `Some(Vec<EpochData>)`. Returns `None` if
/// the file does not exist or no valid rows were found. Parsing and I/O issues
/// are reported to stderr as warning/error messages.
///
/// # Examples
///
/// ```
/// use std::env;
/// use std::fs;
/// let mut path = env::temp_dir();
/// path.push("example_training_log.csv");
/// let contents = "\
/// epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate
/// 1,0.9,12.3,0.8,0.75,0.001
/// 2,0.6,11.8,0.5,0.82,0.001
/// ";
/// fs::write(&path, contents).unwrap();
/// let epochs = read_training_csv(path.to_str().unwrap());
/// assert!(epochs.is_some());
/// let v = epochs.unwrap();
/// assert_eq!(v.len(), 2);
/// ```
fn read_training_csv(path: &str) -> Option<Vec<EpochData>> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return None, // File doesn't exist - this is expected for unrun models
    };

    let reader = BufReader::new(file);
    let mut epochs = Vec::new();
    let mut parse_errors = 0;
    let mut short_lines = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("⚠️  Warning: Failed to read line {} from {}: {}", i + 1, path, e);
                return None;
            }
        };

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
            short_lines += 1;
            continue;
        }

        // Try to parse all fields with helpful error messages
        let epoch: usize = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let train_loss: f32 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let train_time: f32 = match parts[2].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let val_loss: f32 = match parts[3].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let val_accuracy: f32 = match parts[4].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let lr: f32 = match parts[5].trim().parse() {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };

        epochs.push(EpochData {
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            train_time,
            lr,
        });
    }

    // Report parsing issues
    if parse_errors > 0 || short_lines > 0 {
        eprintln!("⚠️  Warning: Issues parsing {}:", path);
        if short_lines > 0 {
            eprintln!("    {} line(s) with fewer than 6 columns (expected: epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate)", short_lines);
        }
        if parse_errors > 0 {
            eprintln!("    {} line(s) with invalid numeric values", parse_errors);
        }
    }

    if epochs.is_empty() {
        if parse_errors > 0 || short_lines > 0 {
            eprintln!("⚠️  Error: {} contains no valid epoch data after parsing errors", path);
            eprintln!("    Expected format: epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate");
        }
        None
    } else {
        Some(epochs)
    }
}

/// Detects and reads the training log file for a given model prefix.
/// Tries multiple optimizer variants (adam, adamw, sgd, rmsprop) and returns
/// the first valid log file found along with the detected optimizer name.
///
/// # Arguments
/// * `model_prefix` - Base name of the model (e.g., "mnist_mlp", "mnist_cnn")
///
/// # Returns
/// `Some((epochs, optimizer_name))` if a valid log file is found, `None` otherwise
fn find_and_read_training_log(model_prefix: &str) -> Option<(Vec<EpochData>, String)> {
    // Define supported optimizers in priority order
    let optimizers = ["adam", "adamw", "sgd", "rmsprop"];

    // Try each optimizer variant
    for opt in &optimizers {
        // Try both .txt and .csv extensions
        for ext in &["txt", "csv"] {
            let path = format!("./logs/training_loss_{}_{}.{}", model_prefix, opt, ext);
            if let Some(epochs) = read_training_csv(&path) {
                let optimizer_name = match *opt {
                    "adam" => "Adam",
                    "adamw" => "AdamW",
                    "sgd" => "SGD",
                    "rmsprop" => "RMSprop",
                    _ => "Unknown",
                };
                return Some((epochs, optimizer_name.to_string()));
            }
        }
    }

    // Fallback: try legacy naming patterns without optimizer suffix
    let legacy_paths = [
        format!("./logs/training_loss_{}.csv", model_prefix),
        format!("./logs/training_loss_{}.txt", model_prefix),
    ];

    for path in &legacy_paths {
        if let Some(epochs) = read_training_csv(path) {
            // Default to SGD for legacy files (historical convention)
            return Some((epochs, "SGD".to_string()));
        }
    }

    None
}

// ============================================================================
// Model Architecture Definitions
// ============================================================================

/// Builds ModelData for a two-layer MNIST MLP (784 → 512 → 10), computing per-layer and aggregate
/// architecture metrics and attaching detected training logs if available.
///
/// The returned ModelData contains LayerInfo entries for the hidden and output dense layers, the
/// total parameter count, total forward FLOPs, and estimated parameter memory in bytes. This
/// function also attempts to auto-detect a training log for the "mlp" prefix; when a valid log is
/// found it populates `training` with the parsed epochs, the detected optimizer name, the initial
/// learning rate (first epoch's lr or 0.001 if missing), a batch size of 64, the best validation
/// accuracy, and the total training time.
///
/// # Returns
///
/// A ModelData describing the MLP architecture and optionally populated training data when a
/// compatible training log is discovered.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::seed_from_u64(42);
/// let model = build_mlp_data(&mut rng);
/// assert_eq!(model.id, "mnist_mlp");
/// assert!(model.architecture.total_params > 0);
/// // `training` may be None if no log file exists in `logs/`
/// ```
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

    // Auto-detect training log for any optimizer type
    let training = find_and_read_training_log("mlp").map(|(epochs, optimizer)| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer,
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

/// Builds metadata for a small CNN model and attempts to attach detected training logs.
///
/// Constructs a model architecture consisting of:
/// - Conv2D with 1 input channel, 8 output channels, 3x3 kernel, padding=1 (preserves 28x28),
/// - 2x2 max pooling (reduces to 8 x 14 x 14 = 1568),
/// - Dense layer from 1568 to 10 outputs.
///
/// The returned ModelData includes per-layer LayerInfo entries, aggregated totals for parameters
/// and FLOPs, memory usage in bytes, and an optional TrainingData populated if a training log
/// for the "cnn" prefix is found.
///
/// # Parameters
///
/// - `rng`: used to initialize layer parameters when constructing layer objects.
///
/// # Returns
///
/// ModelData containing the CNN's id, name, description, architecture metrics (layers, total_params,
/// total_flops, memory_bytes), and `training` set to Some(...) if a matching training log was found.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::seeded(42);
/// let model = build_cnn_data(&mut rng);
/// assert_eq!(model.id, "mnist_cnn");
/// assert!(model.architecture.total_params > 0);
/// ```
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

    // Auto-detect training log for any optimizer type
    let training = find_and_read_training_log("cnn").map(|(epochs, optimizer)| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer,
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

/// Builds a ModelData entry describing a small Transformer-style attention model.
///
/// The returned ModelData contains per-layer parameter counts and FLOP estimates, aggregated
/// totals (parameters, FLOPs, and parameter memory in bytes), and an optional TrainingData
/// when a matching training log for the "attention" model is detected under `logs/`.
///
/// # Examples
///
/// ```
/// let md = build_attention_data();
/// assert_eq!(md.id, "mnist_attention");
/// assert_eq!(md.architecture.layers.len(), 4);
/// ```
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

    // Auto-detect training log for any optimizer type
    let training = find_and_read_training_log("attention").map(|(epochs, optimizer)| {
        let best_accuracy = epochs
            .iter()
            .map(|e| e.val_accuracy)
            .fold(0.0_f32, f32::max);
        let total_time: f32 = epochs.iter().map(|e| e.train_time).sum();
        TrainingData {
            config: TrainingConfig {
                optimizer,
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

/// Generates the demo/dashboard_data.json file consumed by the Architecture Comparison Dashboard.
///
/// This function builds model architecture summaries (MLP, CNN, Attention), attempts to locate and
/// parse per-model training logs under logs/, prints a status summary (which models have training
/// data found or missing), and writes a pretty-printed JSON file to demo/dashboard_data.json.
/// It also prints guidance for generating missing logs when applicable.
fn main() {
    let mut rng = SimpleRng::new(42);

    println!("Generating dashboard data...");
    println!();

    let models = vec![
        build_mlp_data(&mut rng),
        build_cnn_data(&mut rng),
        build_attention_data(),
    ];

    // Check log availability and provide helpful feedback
    let mut missing_logs = Vec::new();
    let mut found_logs = Vec::new();

    for model in &models {
        if model.training.is_some() {
            found_logs.push(model.name.as_str());
        } else {
            missing_logs.push((model.id.as_str(), model.name.as_str()));
        }
    }

    // Print summary with color-coded status
    println!("Training Log Status:");
    println!("-------------------");
    for model in &models {
        if model.training.is_some() {
            println!(
                "  ✓ {} — {} params, {} FLOPS [Training data FOUND]",
                model.name,
                model.architecture.total_params,
                model.architecture.total_flops,
            );
        } else {
            println!(
                "  ✗ {} — {} params, {} FLOPS [Training data MISSING]",
                model.name,
                model.architecture.total_params,
                model.architecture.total_flops,
            );
        }
    }
    println!();

    // Provide helpful guidance if logs are missing
    if !missing_logs.is_empty() {
        println!("⚠️  Warning: Missing training logs for {} model(s)", missing_logs.len());
        println!();
        println!("The dashboard will show architecture data only (no training curves).");
        println!("To generate complete training data, run:");
        println!();
        for (id, _name) in &missing_logs {
            let bin_name = if id.starts_with("mnist_") {
                id.to_string()
            } else {
                id.to_string()
            };
            println!("  cargo run --release --bin {}", bin_name);
        }
        println!();
        println!("Expected log locations:");
        for (id, _) in &missing_logs {
            let model_prefix = if *id == "mnist_mlp" {
                "mlp"
            } else if *id == "mnist_cnn" {
                "cnn"
            } else if *id == "mnist_attention" {
                "attention"
            } else {
                id
            };
            println!("  ./logs/training_loss_{}_<optimizer>.csv", model_prefix);
            println!("    (where <optimizer> is adam, adamw, sgd, or rmsprop)");
        }
        println!();
    } else {
        println!("✓ All training logs found successfully!");
        println!();
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
            let days_in_year = if year.is_multiple_of(4)
                && (!year.is_multiple_of(100) || year.is_multiple_of(400))
            {
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
        let days_in_months: [u64; 12] =
            if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
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