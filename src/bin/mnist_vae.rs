use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
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

// Variational autoencoder for MNIST unsupervised representation learning.
// Uses shared library layers and utilities.
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_CHANNELS: usize = 1; // Grayscale
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784

// Architecture: 784 -> 256 -> (mu:32, log_var:32) -> z:32 -> 256 -> 784
const ENCODER_HIDDEN: usize = 256;
const LATENT_DIM: usize = 32;
const DECODER_HIDDEN: usize = 256;

// KL divergence weight for ELBO loss (beta-VAE style; 1.0 is standard VAE)
const KL_WEIGHT: f32 = 1.0;

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_vae_default.json";

// Optimizer selection
const OPTIMIZER_TYPE: &str = "adam";

// ============================================================================
// Main Logic
// ============================================================================

/// Groups the input dataset slices and sample count for a training or validation split.
///
/// For a VAE, images serve as both input and reconstruction target.
/// Labels are retained for latent space visualization output.
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

/// Creates and initialises a new `VariationalAutoencoder` with random weights.
///
/// Architecture: `NUM_INPUTS` → `ENCODER_HIDDEN` → (`mu`: `LATENT_DIM`, `log_var`: `LATENT_DIM`)
/// → `z`: `LATENT_DIM` → `DECODER_HIDDEN` → `NUM_INPUTS`.
/// Encoder hidden layers use ReLU; the mu and log_var heads are linear.
/// Decoder hidden layers use ReLU; the output layer uses Sigmoid (values in (0, 1)).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let vae = initialize_vae(&mut rng);
/// assert_eq!(vae.input_size(), NUM_INPUTS);
/// assert_eq!(vae.latent_dim(), LATENT_DIM);
/// ```
fn initialize_vae(rng: &mut SimpleRng) -> VariationalAutoencoder {
    VariationalAutoencoder::new(
        NUM_INPUTS,
        &[ENCODER_HIDDEN],
        LATENT_DIM,
        &[DECODER_HIDDEN],
        rng,
    )
}

/// Trains the VAE using mini-batch gradient descent with ELBO loss.
///
/// Evaluates on the provided validation set each epoch using deterministic decoding
/// (forward_mean) to avoid stochastic noise in validation metrics. Uses the scheduler's
/// current learning rate for parameter updates, saves the best model to
/// `"mnist_vae_model_best.bin"` when validation ELBO improves, and supports early
/// stopping based on validation loss. Progress is printed to stdout and recorded in
/// CSV format.
fn train(
    vae: &mut VariationalAutoencoder,
    train_data: &DataSet,
    val_data: &DataSet,
    rng: &mut SimpleRng,
    scheduler: &mut dyn LRScheduler,
    params: &TrainHyperparams,
    debugger: &mut StepDebugger,
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

    let log_filename = format!("./logs/mnist_vae_training_{}.csv", OPTIMIZER_TYPE);
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
    println!("KL weight (beta): {}", KL_WEIGHT);

    let mut batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; params.batch_size];

    let mut indices: Vec<usize> = (0..train_data.num_samples).collect();

    // Early stopping using shared EarlyStopping struct
    let mut early_stopping = EarlyStopping::new(
        params.early_stopping_patience,
        params.early_stopping_min_delta,
    );

    let total_batches = (train_data.num_samples + params.batch_size - 1) / params.batch_size;

    for epoch in 0..params.epochs {
        let mut total_recon_loss = 0.0f32;
        let mut total_kl_loss = 0.0f32;
        let start_time = Instant::now();
        let current_lr = scheduler.get_lr();
        optimizer.set_learning_rate(current_lr);

        debugger.on_epoch_start(epoch + 1);

        // Fisher-Yates shuffle
        if train_data.num_samples > 1 {
            for i in (1..train_data.num_samples).rev() {
                let j = rng.gen_usize(i + 1);
                indices.swap(i, j);
            }
        }

        for batch_start in (0..train_data.num_samples).step_by(params.batch_size) {
            let batch_count = (train_data.num_samples - batch_start).min(params.batch_size);
            let batch_idx = batch_start / params.batch_size + 1;
            let input_len = batch_count * NUM_INPUTS;

            debugger.set_context(epoch + 1, batch_idx, total_batches, batch_count);

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

            // Forward pass: encode, reparameterize, decode
            debugger.before_forward(
                "vae_encoder",
                &batch_inputs,
                batch_count,
                NUM_INPUTS,
                LATENT_DIM,
                vae.parameter_count(),
            );
            let (reconstruction, mu, log_var) =
                vae.forward(&batch_inputs[..input_len], batch_count, rng);
            debugger.after_forward("vae_decoder", &reconstruction, batch_count, NUM_INPUTS);

            // Compute ELBO loss components for logging
            let recon_loss =
                vae.compute_reconstruction_loss(&reconstruction, &batch_inputs[..input_len]);
            let kl_loss = vae.compute_kl_divergence(&mu, &log_var);
            let elbo_loss = recon_loss + KL_WEIGHT * kl_loss;
            total_recon_loss += recon_loss * batch_count as f32;
            total_kl_loss += kl_loss * batch_count as f32;

            // Create a placeholder gradient buffer for loss logging
            let loss_grad = vec![0.0f32; batch_count * NUM_INPUTS];
            debugger.after_loss(elbo_loss, &loss_grad, batch_count, NUM_INPUTS);

            // Backward pass and parameter update with ELBO gradients
            vae.backward(&batch_inputs[..input_len], batch_count, KL_WEIGHT);

            // Log gradient magnitudes (VAE internal gradients)
            let grad_info = [("vae_all_layers", 0.0f32, 0.0f32)]; // Placeholder
            debugger.after_update(&grad_info, current_lr);

            vae.update_with_optimizer(optimizer.as_mut());
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_recon_loss = total_recon_loss / train_data.num_samples as f32;
        let average_kl_loss = total_kl_loss / train_data.num_samples as f32;
        let average_elbo_loss = average_recon_loss + KL_WEIGHT * average_kl_loss;

        // Evaluate on validation set using deterministic forward pass (no reparameterization)
        let mut val_total_recon_loss = 0.0f32;
        let mut val_total_kl_loss = 0.0f32;
        let mut val_batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];

        for batch_start in (0..val_data.num_samples).step_by(params.batch_size) {
            let batch_count = (val_data.num_samples - batch_start).min(params.batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_data.images[input_start..input_start + input_len]);

            // Encode to get mu and log_var for KL computation
            let (mu, log_var) = vae.encode(&val_batch_inputs[..input_len], batch_count);
            // Decode using mu directly (deterministic, no sampling noise)
            let reconstruction = vae.decode(&mu, batch_count);

            let recon_loss =
                vae.compute_reconstruction_loss(&reconstruction, &val_batch_inputs[..input_len]);
            let kl_loss = vae.compute_kl_divergence(&mu, &log_var);
            val_total_recon_loss += recon_loss * batch_count as f32;
            val_total_kl_loss += kl_loss * batch_count as f32;
        }

        let val_recon_loss = val_total_recon_loss / val_data.num_samples as f32;
        let val_kl_loss = val_total_kl_loss / val_data.num_samples as f32;
        let val_elbo_loss = val_recon_loss + KL_WEIGHT * val_kl_loss;

        println!(
            "Epoch {}/{}, ELBO: {:.6} (Recon: {:.6}, KL: {:.6}), \
             Val ELBO: {:.6} (Recon: {:.6}, KL: {:.6}), LR: {:.6}, Time: {:.2}s",
            epoch + 1,
            params.epochs,
            average_elbo_loss,
            average_recon_loss,
            average_kl_loss,
            val_elbo_loss,
            val_recon_loss,
            val_kl_loss,
            current_lr,
            duration
        );

        // Write epoch metrics using shared CsvTrainingLogger
        // val_accuracy is set to 0.0 (not applicable for autoencoders)
        let metrics = TrainingMetrics {
            train_loss: average_elbo_loss,
            val_loss: val_elbo_loss,
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
        match early_stopping.check(val_elbo_loss) {
            EarlyStoppingAction::Improved => save_model(vae, "mnist_vae_model_best.bin"),
            EarlyStoppingAction::Stop => {
                println!(
                    "\nEarly stopping triggered! No improvement for {} epochs. Best validation ELBO: {:.6}",
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

/// Evaluates the VAE on the test set using deterministic decoding and prints reconstruction MSE.
///
/// Uses forward_mean (decoding from μ directly) for deterministic, noise-free evaluation.
fn evaluate(
    vae: &mut VariationalAutoencoder,
    images: &[f32],
    num_samples: usize,
    batch_size: usize,
) {
    let mut total_recon_loss = 0.0f32;
    let mut total_kl_loss = 0.0f32;
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];

    for batch_start in (0..num_samples).step_by(batch_size) {
        let batch_count = (num_samples - batch_start).min(batch_size);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len].copy_from_slice(&images[input_start..input_start + input_len]);

        // Encode to get mu and log_var
        let (mu, log_var) = vae.encode(&batch_inputs[..input_len], batch_count);
        // Decode using mu (deterministic)
        let reconstruction = vae.decode(&mu, batch_count);

        let recon_loss =
            vae.compute_reconstruction_loss(&reconstruction, &batch_inputs[..input_len]);
        let kl_loss = vae.compute_kl_divergence(&mu, &log_var);
        total_recon_loss += recon_loss * batch_count as f32;
        total_kl_loss += kl_loss * batch_count as f32;
    }

    let average_recon_loss = total_recon_loss / num_samples as f32;
    let average_kl_loss = total_kl_loss / num_samples as f32;
    let average_elbo_loss = average_recon_loss + KL_WEIGHT * average_kl_loss;

    println!("Test Reconstruction MSE: {:.6}", average_recon_loss);
    println!("Test KL Divergence: {:.6}", average_kl_loss);
    println!("Test ELBO Loss: {:.6}", average_elbo_loss);
}

/// Exports latent mean codes (μ) for the test set to a CSV file for latent space visualisation.
///
/// Exports the mean vectors (μ) rather than sampled z values, providing a deterministic
/// and denoised representation of the latent space suitable for dimensionality reduction
/// (e.g., t-SNE, UMAP) and cluster analysis.
///
/// Output file: `./logs/mnist_vae_latent.csv`
/// Columns: `z0,z1,...,z{LATENT_DIM-1},label`
fn export_latent_codes(
    vae: &mut VariationalAutoencoder,
    images: &[f32],
    labels: &[u8],
    num_samples: usize,
    batch_size: usize,
) {
    std::fs::create_dir_all("./logs").ok();

    let file = File::create("./logs/mnist_vae_latent.csv").unwrap_or_else(|_| {
        eprintln!("Could not open ./logs/mnist_vae_latent.csv for writing.");
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

        // Encode to latent space: get mu (mean) and log_var
        // Export mu for deterministic, denoised latent codes
        let (mu, _log_var) = vae.encode(&batch_inputs[..input_len], batch_count);

        // Write each sample's latent mean code and label
        for i in 0..batch_count {
            let z_start = i * LATENT_DIM;
            let z_end = z_start + LATENT_DIM;
            let z_values: Vec<String> = mu[z_start..z_end]
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
        "Latent mean codes exported to ./logs/mnist_vae_latent.csv ({} samples)",
        num_samples
    );
}

/// Saves the VAE model architecture metadata to a binary file in little-endian format.
///
/// Saves the architecture dimensions so the model can be reconstructed.
/// The saved format contains:
/// 1. Magic number (i32): 0x56414501 ("VAE\x01")
/// 2. input_size (i32)
/// 3. encoder_hidden (i32)
/// 4. latent_dim (i32)
/// 5. decoder_hidden (i32)
fn save_model(vae: &VariationalAutoencoder, filename: &str) {
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

    // Magic number to identify VAE model files
    write_i32(&mut writer, 0x56414501_u32 as i32);
    write_i32(&mut writer, vae.input_size() as i32);
    write_i32(&mut writer, ENCODER_HIDDEN as i32);
    write_i32(&mut writer, vae.latent_dim() as i32);
    write_i32(&mut writer, DECODER_HIDDEN as i32);

    println!("Model architecture saved to {}", filename);
}

fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Program entry point: loads MNIST data, trains a variational autoencoder, evaluates
/// reconstruction quality, exports latent mean codes for visualisation, and saves the model.
fn main() {
    let program_start = Instant::now();

    // Parse command-line arguments for config file path
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    println!("=== MNIST Variational Autoencoder (VAE) Training ===");
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

    // Resolve step-through debug mode from CLI flag or config
    let step_debug_enabled = parse_step_flag(&args) || config.step_debug.unwrap_or(false);

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
        "Architecture: {} -> {} -> (mu:{}, log_var:{}) -> z:{} -> {} -> {}",
        NUM_INPUTS, ENCODER_HIDDEN, LATENT_DIM, LATENT_DIM, LATENT_DIM, DECODER_HIDDEN, NUM_INPUTS
    );
    println!("KL weight (beta): {}", KL_WEIGHT);

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

    println!("Initializing VAE...");
    let mut rng = SimpleRng::new(1);
    let mut vae = initialize_vae(&mut rng);
    println!("Total parameters: {}", vae.parameter_count());

    let mut debugger = StepDebugger::new(step_debug_enabled);

    println!("Training VAE...");
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
    train(
        &mut vae,
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
    evaluate(&mut vae, &test_images, test_samples, batch_size);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Evaluation time: {:.2} seconds", test_time);

    println!("Exporting latent codes for visualisation...");
    export_latent_codes(
        &mut vae,
        &test_images,
        &test_labels,
        test_samples,
        batch_size,
    );

    println!("Saving final model...");
    save_model(&vae, "mnist_vae_model_final.bin");

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
    fn test_initialize_vae() {
        let mut rng = SimpleRng::new(42);
        let vae = initialize_vae(&mut rng);
        assert_eq!(vae.input_size(), NUM_INPUTS);
        assert_eq!(vae.latent_dim(), LATENT_DIM);
    }

    #[test]
    fn test_vae_forward_returns_correct_shapes() {
        let mut rng = SimpleRng::new(1);
        let mut vae = initialize_vae(&mut rng);
        let mut rng2 = SimpleRng::new(99);
        let input = vec![0.5f32; NUM_INPUTS * 2]; // batch_size = 2
        let (reconstruction, mu, log_var) = vae.forward(&input, 2, &mut rng2);
        assert_eq!(reconstruction.len(), NUM_INPUTS * 2);
        assert_eq!(mu.len(), LATENT_DIM * 2);
        assert_eq!(log_var.len(), LATENT_DIM * 2);
    }

    #[test]
    fn test_vae_elbo_loss_non_negative() {
        let mut rng = SimpleRng::new(2);
        let mut vae = initialize_vae(&mut rng);
        let mut rng2 = SimpleRng::new(7);
        let input = vec![0.5f32; NUM_INPUTS]; // batch_size = 1
        let (reconstruction, mu, log_var) = vae.forward(&input, 1, &mut rng2);
        let loss = vae.compute_elbo_loss(&reconstruction, &input, &mu, &log_var, KL_WEIGHT);
        assert!(loss >= 0.0, "ELBO loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_save_model_creates_file() {
        use std::fs;
        let mut rng = SimpleRng::new(0);
        let vae = initialize_vae(&mut rng);
        // Use temp_dir() which respects TMPDIR and sandbox settings
        let mut path = std::env::temp_dir();
        path.push("test_vae_model.bin");
        save_model(&vae, path.to_str().unwrap());
        assert!(
            fs::metadata(&path).is_ok(),
            "Model file should exist after save"
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_vae_parameter_count_positive() {
        let mut rng = SimpleRng::new(3);
        let vae = initialize_vae(&mut rng);
        assert!(
            vae.parameter_count() > 0,
            "VAE must have trainable parameters"
        );
    }
}
