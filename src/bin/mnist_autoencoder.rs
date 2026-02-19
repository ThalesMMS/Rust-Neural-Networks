use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::autoencoder::vanilla::VanillaAutoencoder;
use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::optimizers::{Adam, Optimizer, SGD};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::{
    gather_batch, parse_config_path, parse_step_flag, print_training_config, CsvTrainingLogger,
    EarlyStopping, EarlyStoppingAction, TrainingMetrics,
};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
use rust_neural_networks::utils::rng::SimpleRng;

// Vanilla autoencoder for MNIST unsupervised representation learning.
// Uses shared library layers and utilities.
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_CHANNELS: usize = 1; // Grayscale
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784

// Architecture: 784 -> 256 -> 64 -> 256 -> 784
const ENCODER_HIDDEN: usize = 256;
const LATENT_DIM: usize = 64;
const DECODER_HIDDEN: usize = 256;

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_autoencoder_default.json";

// Optimizer selection
const OPTIMIZER_TYPE: &str = "adam";

// ============================================================================
// Main Logic
// ============================================================================

/// Groups the input dataset slices and sample count for a training or validation split.
///
/// For an autoencoder, images serve as both input and target (reconstruction target).
/// Labels are retained for optional latent space visualization output.
///
/// # Fields
///
/// - `images` – Flattened row-major pixel data (one image per row of `NUM_INPUTS` values).
/// - `labels` – Class labels for each image (values 0–9 for MNIST).
/// - `num_samples` – Number of samples (rows) in this dataset split.
pub struct DataSet<'a> {
    /// Flattened row-major pixel data (one image per row of NUM_INPUTS values).
    pub images: &'a [f32],
    /// Class labels for each image (values 0–9 for MNIST, used for visualization).
    pub labels: &'a [u8],
    /// Number of samples (rows) in this dataset split.
    pub num_samples: usize,
}

/// Groups scalar training hyperparameters to avoid long parameter lists.
///
/// # Fields
///
/// - `learning_rate` – Initial learning rate for the optimiser.
/// - `epochs` – Total number of training epochs to run.
/// - `batch_size` – Number of samples per mini-batch.
/// - `early_stopping_patience` – Epochs without improvement before stopping.
/// - `early_stopping_min_delta` – Minimum validation-loss improvement threshold.
pub struct TrainHyperparams {
    /// Initial learning rate for parameter updates.
    pub learning_rate: f32,
    /// Total number of training epochs.
    pub epochs: usize,
    /// Mini-batch size used during training.
    pub batch_size: usize,
    /// Number of epochs without improvement before early stopping triggers.
    pub early_stopping_patience: usize,
    /// Minimum improvement in validation loss to count as progress.
    pub early_stopping_min_delta: f32,
}

/// Creates and initialises a new `VanillaAutoencoder` with random weights.
///
/// Architecture: `NUM_INPUTS` → `ENCODER_HIDDEN` → `LATENT_DIM` → `DECODER_HIDDEN` → `NUM_INPUTS`.
/// Encoder hidden layers use ReLU; the latent layer is linear.
/// Decoder hidden layers use ReLU; the output layer uses Sigmoid (values in (0, 1)).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let ae = initialize_autoencoder(&mut rng);
/// assert_eq!(ae.input_size(), NUM_INPUTS);
/// assert_eq!(ae.latent_dim(), LATENT_DIM);
/// ```
fn initialize_autoencoder(rng: &mut SimpleRng) -> VanillaAutoencoder {
    VanillaAutoencoder::new(
        NUM_INPUTS,
        &[ENCODER_HIDDEN],
        LATENT_DIM,
        &[DECODER_HIDDEN],
        rng,
    )
}

/// Trains the autoencoder using mini-batch gradient descent with MSE reconstruction loss.
///
/// Evaluates on the provided validation set each epoch, uses the scheduler's current
/// learning rate for parameter updates, saves the best model to `"mnist_ae_model_best.bin"`
/// when validation loss improves, and supports early stopping based on validation loss.
/// Progress is printed to stdout and recorded in CSV format.
fn train(
    ae: &mut VanillaAutoencoder,
    train_data: &DataSet,
    val_data: &DataSet,
    rng: &mut SimpleRng,
    scheduler: &mut dyn LRScheduler,
    params: &TrainHyperparams,
    _debugger: &mut StepDebugger,
) {
    // Attempt to create logs dir if not exists
    std::fs::create_dir_all("./logs").ok();

    // Create optimizer based on OPTIMIZER_TYPE
    let mut optimizer: Box<dyn Optimizer> = match OPTIMIZER_TYPE {
        "sgd" => Box::new(SGD::new(params.learning_rate)),
        "adam" => Box::new(Adam::new(params.learning_rate, 0.9, 0.999, 1e-8)),
        _ => {
            eprintln!("Unknown optimizer type: {}", OPTIMIZER_TYPE);
            process::exit(1);
        }
    };

    let log_filename = format!("./logs/mnist_ae_training_{}.csv", OPTIMIZER_TYPE);
    let mut csv_logger = CsvTrainingLogger::new(&log_filename).unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    csv_logger.write_header().unwrap_or_else(|_| {
        eprintln!("Failed writing CSV header.");
        process::exit(1);
    });

    println!(
        "Using {} optimizer with learning rate {}",
        OPTIMIZER_TYPE.to_uppercase(),
        params.learning_rate
    );

    let mut batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; params.batch_size];

    let mut indices: Vec<usize> = (0..train_data.num_samples).collect();

    // Early stopping using shared EarlyStopping struct
    let mut early_stopping = EarlyStopping::new(
        params.early_stopping_patience,
        params.early_stopping_min_delta,
    );

    for epoch in 0..params.epochs {
        let mut total_loss = 0.0f32;
        let start_time = Instant::now();
        let current_lr = scheduler.get_lr();
        optimizer.set_learning_rate(current_lr);

        // Fisher-Yates shuffle
        if train_data.num_samples > 1 {
            for i in (1..train_data.num_samples).rev() {
                let j = rng.gen_usize(i + 1);
                indices.swap(i, j);
            }
        }

        for batch_start in (0..train_data.num_samples).step_by(params.batch_size) {
            let batch_count = (train_data.num_samples - batch_start).min(params.batch_size);
            let input_len = batch_count * NUM_INPUTS;

            // Gather a random mini-batch into contiguous buffers.
            // No augmentation for autoencoders (unsupervised, standard input = target).
            gather_batch(
                train_data.images,
                train_data.labels,
                &indices,
                batch_start,
                batch_count,
                &mut batch_inputs,
                &mut batch_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                None, // no horizontal flip
                None, // no random crop
                None, // no brightness jitter
                None, // no contrast jitter
                None, // no saturation jitter
                None, // no rng (no augmentation)
            );

            // Forward pass: encode then decode
            let output = ae.forward(&batch_inputs[..input_len], batch_count);

            // MSE reconstruction loss (input = target)
            let batch_loss = ae.compute_loss(&output, &batch_inputs[..input_len]);
            total_loss += batch_loss * batch_count as f32;

            // Backward pass and parameter update
            ae.backward(&batch_inputs[..input_len], batch_count);
            ae.update_with_optimizer(optimizer.as_mut());
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_loss = total_loss / train_data.num_samples as f32;

        // Evaluate on validation set (MSE loss only – no classification accuracy)
        let mut val_total_loss = 0.0f32;
        let mut val_batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];

        for batch_start in (0..val_data.num_samples).step_by(params.batch_size) {
            let batch_count = (val_data.num_samples - batch_start).min(params.batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_data.images[input_start..input_start + input_len]);

            // Forward pass (no grad caching needed for validation)
            let output = ae.forward(&val_batch_inputs[..input_len], batch_count);
            let batch_loss = ae.compute_loss(&output, &val_batch_inputs[..input_len]);
            val_total_loss += batch_loss * batch_count as f32;
        }

        let val_average_loss = val_total_loss / val_data.num_samples as f32;

        println!(
            "Epoch {}/{}, Train MSE: {:.6}, Val MSE: {:.6}, LR: {:.6}, Time: {:.2}s",
            epoch + 1,
            params.epochs,
            average_loss,
            val_average_loss,
            current_lr,
            duration
        );

        // Write epoch metrics using shared CsvTrainingLogger
        // val_accuracy is set to 0.0 (not applicable for autoencoders)
        let metrics = TrainingMetrics {
            train_loss: average_loss,
            val_loss: val_average_loss,
            val_accuracy: 0.0,
            train_time: duration,
            learning_rate: current_lr,
        };
        csv_logger
            .write_epoch(epoch + 1, &metrics)
            .unwrap_or_else(|_| {
                eprintln!("Failed writing training loss data.");
                process::exit(1);
            });

        // Early stopping check using shared EarlyStopping struct
        match early_stopping.check(val_average_loss) {
            EarlyStoppingAction::Improved => save_model(ae, "mnist_ae_model_best.bin"),
            EarlyStoppingAction::Stop => {
                println!(
                    "\nEarly stopping triggered! No improvement for {} epochs. Best validation MSE: {:.6}",
                    params.early_stopping_patience, early_stopping.best_val_loss
                );
                break;
            }
            EarlyStoppingAction::Continue => {}
        }

        // Update learning rate for next epoch
        scheduler.step();
    }
}

/// Evaluates the autoencoder on the test set and prints the mean reconstruction MSE.
fn evaluate(ae: &mut VanillaAutoencoder, images: &[f32], num_samples: usize, batch_size: usize) {
    let mut total_loss = 0.0f32;
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];

    for batch_start in (0..num_samples).step_by(batch_size) {
        let batch_count = (num_samples - batch_start).min(batch_size);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len].copy_from_slice(&images[input_start..input_start + input_len]);

        let output = ae.forward(&batch_inputs[..input_len], batch_count);
        let batch_loss = ae.compute_loss(&output, &batch_inputs[..input_len]);
        total_loss += batch_loss * batch_count as f32;
    }

    let average_loss = total_loss / num_samples as f32;
    println!("Test Reconstruction MSE: {:.6}", average_loss);
}

/// Exports latent codes for the test set to a CSV file for latent space visualisation.
///
/// Each row contains the latent vector followed by the class label, enabling downstream
/// dimensionality reduction (e.g., t-SNE, UMAP) and cluster analysis.
///
/// Output file: `./logs/mnist_ae_latent.csv`
/// Columns: `z0,z1,...,z{LATENT_DIM-1},label`
fn export_latent_codes(
    ae: &mut VanillaAutoencoder,
    images: &[f32],
    labels: &[u8],
    num_samples: usize,
    batch_size: usize,
) {
    std::fs::create_dir_all("./logs").ok();

    let file = File::create("./logs/mnist_ae_latent.csv").unwrap_or_else(|_| {
        eprintln!("Could not open ./logs/mnist_ae_latent.csv for writing.");
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    // Write CSV header
    let header: Vec<String> = (0..LATENT_DIM)
        .map(|i| format!("z{}", i))
        .chain(std::iter::once("label".to_string()))
        .collect();
    writeln!(writer, "{}", header.join(",")).unwrap_or_else(|_| {
        eprintln!("Failed writing latent CSV header.");
        process::exit(1);
    });

    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];

    for batch_start in (0..num_samples).step_by(batch_size) {
        let batch_count = (num_samples - batch_start).min(batch_size);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len].copy_from_slice(&images[input_start..input_start + input_len]);

        // Encode to latent space
        let latent = ae.encode(&batch_inputs[..input_len], batch_count);

        // Write each sample's latent code and label
        for i in 0..batch_count {
            let z_start = i * LATENT_DIM;
            let z_end = z_start + LATENT_DIM;
            let z_values: Vec<String> = latent[z_start..z_end]
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect();
            writeln!(writer, "{},{}", z_values.join(","), labels[batch_start + i]).unwrap_or_else(
                |_| {
                    eprintln!("Failed writing latent code row.");
                    process::exit(1);
                },
            );
        }
    }

    println!(
        "Latent codes exported to ./logs/mnist_ae_latent.csv ({} samples)",
        num_samples
    );
}

/// Saves the autoencoder model to a binary file in little-endian format.
///
/// File format:
/// 1. Architecture header: `num_encoder_layers` (i32), `num_decoder_layers` (i32)
/// 2. For each encoder layer: `input_size` (i32), `output_size` (i32), weights (f32s), biases (f32s)
/// 3. For each decoder layer: `input_size` (i32), `output_size` (i32), weights (f32s), biases (f32s)
fn save_model(ae: &VanillaAutoencoder, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {} for writing model", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    let write_i32 = |writer: &mut BufWriter<File>, value: i32| {
        writer.write_all(&value.to_le_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };
    let write_f32 = |writer: &mut BufWriter<File>, value: f32| {
        writer.write_all(&value.to_le_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };

    let encoder_layers = ae.encoder_layers();
    let decoder_layers = ae.decoder_layers();

    // Write layer counts
    write_i32(&mut writer, encoder_layers.len() as i32);
    write_i32(&mut writer, decoder_layers.len() as i32);

    // Write encoder layers
    for layer in encoder_layers {
        write_i32(&mut writer, layer.input_size() as i32);
        write_i32(&mut writer, layer.output_size() as i32);
        for &value in layer.weights() {
            write_f32(&mut writer, value);
        }
        for &value in layer.biases() {
            write_f32(&mut writer, value);
        }
    }

    // Write decoder layers
    for layer in decoder_layers {
        write_i32(&mut writer, layer.input_size() as i32);
        write_i32(&mut writer, layer.output_size() as i32);
        for &value in layer.weights() {
            write_f32(&mut writer, value);
        }
        for &value in layer.biases() {
            write_f32(&mut writer, value);
        }
    }

    println!("Model saved to {}", filename);
}

fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Program entry point: loads MNIST data, trains a vanilla autoencoder, evaluates
/// reconstruction quality, exports latent codes for visualisation, and saves the model.
fn main() {
    let program_start = Instant::now();

    // Parse command-line arguments for config file path and step mode
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);
    let step_mode = parse_step_flag(&args);

    println!("=== MNIST Vanilla Autoencoder Training ===");
    println!("Loading configuration from: {}", config_path);
    let config = match load_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config from '{}': {}", config_path, e);
            eprintln!("Please ensure the config file exists and is valid JSON.");
            process::exit(1);
        }
    };

    // Extract hyperparameters from config with defaults
    let learning_rate = config.learning_rate.unwrap_or(0.001);
    let epochs = config.epochs.unwrap_or(20);
    let batch_size = config.batch_size.unwrap_or(64);
    let validation_split = config.validation_split.unwrap_or(0.1);
    let early_stopping_patience = config.early_stopping_patience.unwrap_or(5);
    let early_stopping_min_delta = config.early_stopping_min_delta.unwrap_or(0.0001);

    // Print loaded configuration
    print_training_config(
        &config,
        learning_rate,
        epochs,
        batch_size,
        validation_split,
        early_stopping_patience,
        early_stopping_min_delta,
    );

    println!(
        "Architecture: {} -> {} -> {} -> {} -> {}",
        NUM_INPUTS, ENCODER_HIDDEN, LATENT_DIM, DECODER_HIDDEN, NUM_INPUTS
    );

    // Create learning rate scheduler
    let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(&config_path));

    println!("Loading training data...");
    let load_start = Instant::now();
    let mut train_images =
        read_mnist_images("./data/train-images.idx3-ubyte").unwrap_or_else(|e| {
            eprintln!("{e}");
            process::exit(1);
        });
    let mut train_labels =
        read_mnist_labels("./data/train-labels.idx1-ubyte").unwrap_or_else(|e| {
            eprintln!("{e}");
            process::exit(1);
        });

    println!("Loading test data...");
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Data loading time: {:.2} seconds", load_time);

    // Split training data into train and validation sets
    let total_train_samples = train_images.len() / NUM_INPUTS;
    let validation_samples = (total_train_samples as f32 * validation_split) as usize;
    let actual_train_samples = total_train_samples - validation_samples;

    let split_point_images = actual_train_samples * NUM_INPUTS;
    let split_point_labels = actual_train_samples;

    let val_images = train_images.split_off(split_point_images);
    let val_labels = train_labels.split_off(split_point_labels);

    let test_samples = test_images.len() / NUM_INPUTS;

    println!(
        "Data split: {} training samples, {} validation samples, {} test samples",
        actual_train_samples, validation_samples, test_samples
    );

    println!("Initializing autoencoder...");
    let mut rng = SimpleRng::new(1);
    let mut ae = initialize_autoencoder(&mut rng);
    println!("Total parameters: {}", ae.parameter_count());

    println!("Training autoencoder...");
    let train_start = Instant::now();
    let train_data = DataSet {
        images: &train_images,
        labels: &train_labels,
        num_samples: actual_train_samples,
    };
    let val_data = DataSet {
        images: &val_images,
        labels: &val_labels,
        num_samples: validation_samples,
    };
    let hyperparams = TrainHyperparams {
        learning_rate,
        epochs,
        batch_size,
        early_stopping_patience,
        early_stopping_min_delta,
    };
    let mut debugger = StepDebugger::new(step_mode);
    train(
        &mut ae,
        &train_data,
        &val_data,
        &mut rng,
        scheduler.as_mut(),
        &hyperparams,
        &mut debugger,
    );
    let train_time = train_start.elapsed().as_secs_f64();
    println!("Total training time: {:.2} seconds", train_time);

    println!("Evaluating on test set...");
    let test_start = Instant::now();
    evaluate(&mut ae, &test_images, test_samples, batch_size);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Evaluation time: {:.2} seconds", test_time);

    println!("Exporting latent codes for visualisation...");
    export_latent_codes(
        &mut ae,
        &test_images,
        &test_labels,
        test_samples,
        batch_size,
    );

    println!("Saving final model...");
    save_model(&ae, "mnist_ae_model_final.bin");

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Data loading time: {:.2} seconds", load_time);
    println!("Total training time: {:.2} seconds", train_time);
    println!("Evaluation time: {:.2} seconds", test_time);
    println!("Total program time: {:.2} seconds", total_time);
    println!("========================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_autoencoder() {
        let mut rng = SimpleRng::new(42);
        let ae = initialize_autoencoder(&mut rng);
        assert_eq!(ae.input_size(), NUM_INPUTS);
        assert_eq!(ae.latent_dim(), LATENT_DIM);
    }

    #[test]
    fn test_save_model_creates_file() {
        use std::fs;
        let mut rng = SimpleRng::new(0);
        let ae = initialize_autoencoder(&mut rng);
        // Use temp_dir() which respects TMPDIR and sandbox settings
        let mut path = std::env::temp_dir();
        path.push("test_ae_model.bin");
        save_model(&ae, path.to_str().unwrap());
        assert!(
            fs::metadata(&path).is_ok(),
            "Model file should exist after save"
        );
        fs::remove_file(&path).ok();
    }
}
