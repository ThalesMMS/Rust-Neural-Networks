// mnist_cnn.rs
// Minimal CNN for MNIST on CPU using explicit loops (no external crates).
// Expected files:
//   ./data/train-images.idx3-ubyte
//   ./data/train-labels.idx1-ubyte
//   ./data/t10k-images.idx3-ubyte
//   ./data/t10k-labels.idx1-ubyte
//
// Output:
//   - logs/training_loss_cnn.csv (epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate)
//   - prints test accuracy
//
// Note: educational implementation (no BLAS/GEMM), so it is intentionally slow.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::{
    compute_softmax_cross_entropy, evaluate_batch_accuracy, gather_batch, parse_config_path,
    parse_registry_dir, parse_run_name, parse_seed_override, parse_step_flag,
    print_training_config, CsvTrainingLogger, EarlyStopping, EarlyStoppingAction, TrainingMetrics,
};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
use rust_neural_networks::utils::rng::SimpleRng;

// MNIST constants (images are flat 28x28 in row-major order).
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_CHANNELS: usize = 1; // Grayscale
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784
const NUM_CLASSES: usize = 10;

// CNN topology: 1x28x28 -> conv -> ReLU -> 2x2 maxpool -> FC(10).
const CONV_OUT: usize = 8;
const KERNEL: usize = 3;
const PAD: isize = 1;
const POOL: usize = 2;

const POOL_H: usize = IMG_H / POOL; // 14
const POOL_W: usize = IMG_W / POOL; // 14
const FC_IN: usize = CONV_OUT * POOL_H * POOL_W; // 8*14*14 = 1568

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_cnn_default.json";

// Training hyperparameters (defaults, overridden by config).
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 32;
const VALIDATION_SPLIT: f32 = 0.1; // 10% of training data for validation
const EARLY_STOPPING_PATIENCE: usize = 3; // Number of epochs without improvement before stopping
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001; // Minimum change to be considered an improvement

// ============================================================================
// Main Logic
// ============================================================================

mod conv_layer;
mod io;
mod model;
mod training;

use conv_layer::*;
use io::*;
use model::*;
use training::*;

/// Entry point for training and evaluating a minimal CNN on the MNIST dataset.
///
/// This program loads MNIST IDX files from ./data, trains a small convolutional
/// neural network on the training set while logging epoch loss to ./logs/training_loss_cnn.csv,
/// and prints final test accuracy. It expects the following files to exist:
/// - ./data/train-images.idx3-ubyte
/// - ./data/train-labels.idx1-ubyte
/// - ./data/t10k-images.idx3-ubyte
/// - ./data/t10k-labels.idx1-ubyte
///
/// The training uses a simple SGD update loop over configurable epochs and batch size,
/// and runs entirely on the CPU with explicit loops (educational, non-BLAS implementation).
///
/// # Examples
///
/// ```no_run
/// // Place MNIST IDX files in ./data and run the binary.
/// // Calling `main()` will start training and print progress and final test accuracy.
/// main();
/// ```
fn main() {
    // Parse command-line arguments for config file path and step flag
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    // Load config first to check step_debug field
    println!("=== MNIST CNN Training ===");
    println!("Loading configuration from: {}", config_path);
    let config = match load_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config from '{}': {}", config_path, e);
            eprintln!("Please ensure the config file exists and is valid JSON.");
            process::exit(1);
        }
    };

    // Check for step-debug mode from CLI flag or config
    let step_debug = parse_step_flag(&args) || config.step_debug.unwrap_or(false);

    // Extract hyperparameters from config with defaults
    let learning_rate = config.learning_rate.unwrap_or(LEARNING_RATE);
    let epochs = config.epochs.unwrap_or(EPOCHS);
    let batch_size = config.batch_size.unwrap_or(BATCH_SIZE);
    let validation_split = config.validation_split.unwrap_or(VALIDATION_SPLIT);
    let early_stopping_patience = config
        .early_stopping_patience
        .unwrap_or(EARLY_STOPPING_PATIENCE);
    let early_stopping_min_delta = config
        .early_stopping_min_delta
        .unwrap_or(EARLY_STOPPING_MIN_DELTA);

    if batch_size == 0 {
        eprintln!(
            "Invalid batch_size: {}. Expected a positive value.",
            batch_size
        );
        process::exit(1);
    }
    if !validation_split.is_finite() || !(0.0..=1.0).contains(&validation_split) {
        eprintln!(
            "Invalid validation_split: {}. Expected a finite value in [0.0, 1.0].",
            validation_split
        );
        process::exit(1);
    }

    // Extract augmentation parameters from config
    let enable_augmentation = config.enable_augmentation.unwrap_or(false);
    let horizontal_flip_prob = config.horizontal_flip_prob;
    let random_crop_padding = config.random_crop_padding;
    let brightness_jitter = config.brightness_jitter;
    let contrast_jitter = config.contrast_jitter;

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

    // Create learning rate scheduler
    let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(&config_path));

    println!("Loading MNIST...");
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
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });

    // Split training data into train and validation sets
    let total_train_samples = train_images.len() / NUM_INPUTS;
    let validation_samples = (total_train_samples as f32 * validation_split) as usize;
    let actual_train_samples = total_train_samples
        .checked_sub(validation_samples)
        .unwrap_or_else(|| {
            eprintln!(
                "Invalid validation_split: {} produces more validation samples ({}) than total samples ({}).",
                validation_split, validation_samples, total_train_samples
            );
            process::exit(1);
        });

    if actual_train_samples == 0 {
        eprintln!(
            "Invalid validation_split: {} leaves no training samples from {} total samples.",
            validation_split, total_train_samples
        );
        process::exit(1);
    }
    if validation_samples == 0 {
        eprintln!(
            "Invalid validation_split: {} leaves no validation samples from {} total samples.",
            validation_split, total_train_samples
        );
        process::exit(1);
    }

    let split_point_images = actual_train_samples * NUM_INPUTS;
    let split_point_labels = actual_train_samples;

    let val_images = train_images.split_off(split_point_images);
    let val_labels = train_labels.split_off(split_point_labels);

    let train_n = actual_train_samples;
    let test_n = test_labels.len();
    println!(
        "Data split: {} training samples, {} validation samples, {} test samples",
        actual_train_samples, validation_samples, test_n
    );

    // Registry-related CLI args (optional)
    let run_name = parse_run_name(&args);
    let registry_dir = parse_registry_dir(&args).unwrap_or_else(|| "runs".to_string());

    // Log paths that we want to capture in the run record.
    let training_log_path = "./logs/training_loss_cnn.csv".to_string();
    let gradient_log_path = "./logs/gradients_cnn.csv".to_string();

    let seed = parse_seed_override(&args).unwrap_or(1);
    let run_id = rust_neural_networks::experiment_registry_run_id::generate_run_id();
    let timestamp_start = chrono::Utc::now().to_rfc3339();
    let mut rng = SimpleRng::new(seed);

    // Augmentation RNG for use with gather_batch
    let mut aug_rng = SimpleRng::new(2);

    let mut model = init_cnn(&mut rng);

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let mut logger = CsvTrainingLogger::new("./logs/training_loss_cnn.csv").unwrap_or_else(|_| {
        eprintln!("Could not create logs/training_loss_cnn.csv");
        process::exit(1);
    });
    logger.write_header().unwrap_or_else(|_| {
        eprintln!("Could not write CSV header to logs/training_loss_cnn.csv");
        process::exit(1);
    });

    // Create gradient logging file
    let gradient_log_filename = "./logs/gradients_cnn.csv";
    let gradient_file = File::create(gradient_log_filename).unwrap_or_else(|_| {
        eprintln!("Could not open file for writing gradient logs.");
        process::exit(1);
    });
    let mut gradient_file = BufWriter::new(gradient_file);

    // Write gradient CSV header
    writeln!(
        gradient_file,
        "epoch,layer_name,grad_norm_weights,grad_norm_biases"
    )
    .unwrap_or_else(|_| {
        eprintln!("Failed writing gradient CSV header.");
        process::exit(1);
    });

    // Training buffers (reused each batch to avoid allocations).
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; batch_size];

    let mut conv_out = vec![0.0f32; batch_size * CONV_OUT * IMG_H * IMG_W];
    let mut pool_out = vec![0.0f32; batch_size * FC_IN];
    let mut pool_idx = vec![0u8; batch_size * CONV_OUT * POOL_H * POOL_W];
    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];
    let mut delta = vec![0.0f32; batch_size * NUM_CLASSES];

    let mut d_pool = vec![0.0f32; batch_size * FC_IN];
    let mut d_conv = vec![0.0f32; batch_size * CONV_OUT * IMG_H * IMG_W];
    let mut _grad_input = vec![0.0f32; batch_size * NUM_INPUTS]; // unused (first layer)

    let mut indices: Vec<usize> = (0..train_n).collect();

    // Early stopping state
    let mut early_stopping = EarlyStopping::new(early_stopping_patience, early_stopping_min_delta);

    // Create step debugger
    let mut debugger = StepDebugger::new(step_debug);

    println!(
        "Training CNN: epochs={} batch={} lr={}",
        epochs, batch_size, learning_rate
    );

    let training_start = Instant::now();
    for epoch in 0..epochs {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);
        let current_lr = scheduler.get_lr();

        let mut total_loss = 0.0f32;

        // Accumulate gradient norms for this epoch
        let mut conv_weight_grad_sum = 0.0f32;
        let mut conv_bias_grad_sum = 0.0f32;
        let mut fc_weight_grad_sum = 0.0f32;
        let mut fc_bias_grad_sum = 0.0f32;
        let mut batch_count_total = 0usize;

        debugger.on_epoch_start(epoch + 1);

        let total_batches = (train_n + batch_size - 1) / batch_size;

        for batch_start in (0..train_n).step_by(batch_size) {
            let batch = (train_n - batch_start).min(batch_size);
            let scale = 1.0f32;
            let batch_idx = batch_start / batch_size + 1;

            debugger.set_context(epoch + 1, batch_idx, total_batches, batch);

            // Gather a random mini-batch into contiguous buffers.
            // Apply augmentation only during training if enabled.
            gather_batch(
                &train_images,
                &train_labels,
                &indices,
                batch_start,
                batch,
                &mut batch_inputs,
                &mut batch_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                if enable_augmentation {
                    horizontal_flip_prob
                } else {
                    None
                },
                if enable_augmentation {
                    random_crop_padding
                } else {
                    None
                },
                if enable_augmentation {
                    brightness_jitter
                } else {
                    None
                },
                if enable_augmentation {
                    contrast_jitter
                } else {
                    None
                },
                None, // saturation_jitter: not applicable for grayscale MNIST
                if enable_augmentation {
                    Some(&mut aug_rng)
                } else {
                    None
                },
            );

            // Forward: conv -> pool -> FC -> logits.
            let conv_param_count = CONV_OUT * IMG_CHANNELS * KERNEL * KERNEL + CONV_OUT;
            debugger.before_forward(
                "conv_layer",
                &batch_inputs,
                batch,
                IMG_H * IMG_W * IMG_CHANNELS,
                CONV_OUT * IMG_H * IMG_W,
                conv_param_count,
            );
            conv_forward_relu(&mut model, batch, &batch_inputs, &mut conv_out);
            debugger.after_forward(
                "conv_layer",
                &conv_out[..batch * CONV_OUT * IMG_H * IMG_W],
                batch,
                CONV_OUT * IMG_H * IMG_W,
            );
            debugger.after_activation(
                "ReLU",
                &conv_out[..batch * CONV_OUT * IMG_H * IMG_W],
                batch,
                CONV_OUT * IMG_H * IMG_W,
            );

            maxpool_forward(batch, &conv_out, &mut pool_out, &mut pool_idx);

            let fc_param_count = FC_IN * NUM_CLASSES + NUM_CLASSES;
            debugger.before_forward(
                "fc_layer",
                &pool_out,
                batch,
                FC_IN,
                NUM_CLASSES,
                fc_param_count,
            );
            fc_forward(&mut model, batch, &pool_out, &mut logits);
            debugger.after_forward("fc_layer", &logits, batch, NUM_CLASSES);

            // Softmax + loss + gradient at logits.
            softmax_rows(&mut logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);
            debugger.after_activation(
                "Softmax",
                &logits[..batch * NUM_CLASSES],
                batch,
                NUM_CLASSES,
            );

            let batch_loss = compute_softmax_cross_entropy(
                &logits[..batch * NUM_CLASSES],
                &batch_labels,
                batch,
                NUM_CLASSES,
                &mut delta,
                scale,
            );
            total_loss += batch_loss;
            debugger.after_loss(batch_loss / batch as f32, &delta, batch, NUM_CLASSES);

            // Backward: FC -> pool -> conv.
            fc_backward(&mut model, batch, &pool_out, &delta, &mut d_pool);
            debugger.after_backward(
                "fc_layer",
                &delta,
                &d_pool[..batch * FC_IN],
                batch,
                NUM_CLASSES,
                FC_IN,
            );

            maxpool_backward_relu(batch, &conv_out, &d_pool, &pool_idx, &mut d_conv);
            debugger.after_relu_derivative(
                &d_conv[..batch * CONV_OUT * IMG_H * IMG_W],
                batch,
                CONV_OUT * IMG_H * IMG_W,
            );

            conv_backward(&mut model, batch, &batch_inputs, &d_conv, &mut _grad_input);
            debugger.after_backward(
                "conv_layer",
                &d_conv,
                &_grad_input[..batch * NUM_INPUTS],
                batch,
                CONV_OUT * IMG_H * IMG_W,
                IMG_H * IMG_W * IMG_CHANNELS,
            );

            // Log gradient magnitudes before parameter update (accumulate for epoch)
            let (conv_w_norm, conv_b_norm) = model.conv_layer.get_gradient_magnitude();
            let (fc_w_norm, fc_b_norm) = model.fc_layer.get_gradient_magnitude();
            conv_weight_grad_sum += conv_w_norm;
            conv_bias_grad_sum += conv_b_norm;
            fc_weight_grad_sum += fc_w_norm;
            fc_bias_grad_sum += fc_b_norm;
            batch_count_total += 1;

            // SGD update using Layer trait (no momentum, no weight decay).
            model.fc_layer.update_parameters(current_lr);
            model.conv_layer.update_parameters(current_lr);

            debugger.after_update(
                &[
                    ("conv_layer", conv_w_norm, conv_b_norm),
                    ("fc_layer", fc_w_norm, fc_b_norm),
                ],
                current_lr,
            );
        }

        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;

        // Evaluate on validation set
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
        let mut val_conv_out = vec![0.0f32; batch_size * CONV_OUT * IMG_H * IMG_W];
        let mut val_pool_out = vec![0.0f32; batch_size * FC_IN];
        let mut val_pool_idx = vec![0u8; batch_size * CONV_OUT * POOL_H * POOL_W];
        let mut val_logits = vec![0.0f32; batch_size * NUM_CLASSES];

        for batch_start in (0..validation_samples).step_by(batch_size) {
            let batch_count = (validation_samples - batch_start).min(batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_images[input_start..input_start + input_len]);

            // Forward pass
            conv_forward_relu(
                &mut model,
                batch_count,
                &val_batch_inputs,
                &mut val_conv_out,
            );
            maxpool_forward(
                batch_count,
                &val_conv_out,
                &mut val_pool_out,
                &mut val_pool_idx,
            );
            fc_forward(&mut model, batch_count, &val_pool_out, &mut val_logits);

            // Apply softmax
            softmax_rows(
                &mut val_logits[..batch_count * NUM_CLASSES],
                batch_count,
                NUM_CLASSES,
            );

            // Compute loss and accuracy using shared utility
            let batch_val_labels = &val_labels[batch_start..batch_start + batch_count];
            let (batch_loss, batch_correct) = evaluate_batch_accuracy(
                &val_logits[..batch_count * NUM_CLASSES],
                batch_val_labels,
                batch_count,
                NUM_CLASSES,
            );
            val_total_loss += batch_loss;
            val_correct += batch_correct;
        }

        let val_average_loss = val_total_loss / validation_samples as f32;
        let val_accuracy = val_correct as f32 / validation_samples as f32 * 100.0;

        // Write gradient magnitudes (averaged across batches) to gradient log
        let num_batches = batch_count_total as f32;
        let conv_w_avg = conv_weight_grad_sum / num_batches;
        let conv_b_avg = conv_bias_grad_sum / num_batches;
        let fc_w_avg = fc_weight_grad_sum / num_batches;
        let fc_b_avg = fc_bias_grad_sum / num_batches;

        writeln!(
            gradient_file,
            "{},conv_layer,{},{}",
            epoch + 1,
            conv_w_avg,
            conv_b_avg
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing gradient data.");
            process::exit(1);
        });

        writeln!(
            gradient_file,
            "{},fc_layer,{},{}",
            epoch + 1,
            fc_w_avg,
            fc_b_avg
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing gradient data.");
            process::exit(1);
        });

        println!(
            "Epoch {}, Loss: {:.6}, Val Loss: {:.6}, Val Acc: {:.2}%, Time: {:.6}",
            epoch + 1,
            avg_loss,
            val_average_loss,
            val_accuracy,
            secs
        );
        let metrics = TrainingMetrics {
            train_loss: avg_loss,
            val_loss: val_average_loss,
            val_accuracy,
            train_time: secs,
            learning_rate: current_lr,
        };
        logger.write_epoch(epoch + 1, &metrics).ok();

        // Early stopping check
        match early_stopping.check(val_average_loss) {
            EarlyStoppingAction::Improved => {
                save_model(&model, "mnist_cnn_model_best.bin");
            }
            EarlyStoppingAction::Stop => {
                println!(
                    "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
                    early_stopping_patience, early_stopping.best_val_loss
                );
                break;
            }
            EarlyStoppingAction::Continue => {}
        }

        // Update learning rate scheduler
        scheduler.step();
    }

    save_model(&model, "mnist_cnn_model_final.bin");

    // Registry artifacts captured (minimum: training log path; plus any obvious artifacts).
    let artifacts = rust_neural_networks::experiment_registry::Artifacts {
        training_log_csv: Some(training_log_path),
        checkpoints: vec![
            "mnist_cnn_model_best.bin".to_string(),
            "mnist_cnn_model_final.bin".to_string(),
        ],
        plots: vec![],
        extra: Some(serde_json::json!({
            "gradient_log_csv": gradient_log_path,
        })),
    };

    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);

    let record = rust_neural_networks::experiment_registry::RunRecord {
        schema_version: rust_neural_networks::experiment_registry::RUN_RECORD_SCHEMA_VERSION
            .to_string(),
        run_id,
        run_name,
        timestamp_start,
        timestamp_end: Some(chrono::Utc::now().to_rfc3339()),
        model_type: "mnist_cnn".to_string(),
        command: Some(std::env::args().collect::<Vec<_>>().join(" ")),
        status: rust_neural_networks::experiment_registry::RunStatus::Completed,
        seed,
        config: rust_neural_networks::experiment_registry::ConfigSnapshot {
            config_path: Some(config_path.clone()),
            config_format: Some("json".to_string()),
            raw: std::fs::read_to_string(&config_path).ok(),
            parsed: None,
        },
        dataset: Some(rust_neural_networks::experiment_registry::dataset_placeholder("mnist")),
        metrics: Some(rust_neural_networks::experiment_registry::Metrics {
            epochs_completed: epochs,
            final_train_loss: Some(early_stopping.best_val_loss as f64),
            final_val_loss: Some(early_stopping.best_val_loss as f64),
            final_val_accuracy: Some(acc as f64),
            total_training_time_seconds: Some(training_start.elapsed().as_secs_f64()),
        }),
        artifacts: Some(artifacts),
        environment: Some(rust_neural_networks::experiment_registry::collect_environment()),
    };

    if let Err(e) =
        rust_neural_networks::experiment_registry::write_run_record(&registry_dir, &record)
    {
        eprintln!("Warning: failed to write run record: {e}");
    }
}
