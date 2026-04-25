// cifar10_cnn.rs
// CNN for CIFAR-10 on CPU using shared layers/utilities.
// Expected files:
//   ./data/cifar-10-batches-bin/data_batch_1.bin through data_batch_5.bin
//   ./data/cifar-10-batches-bin/test_batch.bin
//
// Output:
//   - logs/training_loss_cifar10_cnn.txt (epoch,train_loss,time,val_loss,val_accuracy,lr)
//   - prints test accuracy
//
// Note: educational implementation. Conv2D uses explicit loops; DenseLayer uses BLAS.

use std::env;
use std::fs;
use std::io::Write;
use std::process;
use std::time::Instant;

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::config::{load_config, TrainingConfig};
use rust_neural_networks::data::cifar10::{read_cifar10_batch, read_cifar10_batches};
pub use rust_neural_networks::layers::{
    batchnorm::BatchNormLayer, dropout::DropoutLayer, Conv2DLayer, DenseLayer, Layer, ResidualBlock,
};
use rust_neural_networks::optimizers::rmsprop::RMSprop;
use rust_neural_networks::optimizers::{Adam, AdamW, Optimizer, SGD};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::{
    compute_softmax_cross_entropy, evaluate_batch_accuracy, gather_batch, parse_config_path,
    parse_step_flag, print_training_config, CsvGradientLogger, CsvTrainingLogger, EarlyStopping,
    EarlyStoppingAction, TrainingMetrics,
};
pub use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
pub use rust_neural_networks::utils::rng::SimpleRng;

// CIFAR-10 constants (images are 32x32 RGB in pixel-interleaved format).
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_CHANNELS: usize = 3; // RGB
const NUM_INPUTS: usize = IMG_H * IMG_W * IMG_CHANNELS; // 3072
const NUM_CLASSES: usize = 10;

// CNN topology is now loaded from architecture config files.
// The network can support arbitrary layer configurations (Conv2D, BatchNorm, Dropout, Dense).

// Training hyperparameters (defaults, can be overridden by config file).
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 10; // CIFAR-10 needs more epochs than MNIST
const BATCH_SIZE: usize = 32;
const VALIDATION_SPLIT: f32 = 0.1; // 10% of training data for validation
const EARLY_STOPPING_PATIENCE: usize = 3; // Number of epochs without improvement before stopping
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001; // Minimum change to be considered an improvement

// Default config paths
const DEFAULT_CONFIG_PATH: &str = "config/training/cifar10_cnn_default.json";
const DEFAULT_ARCHITECTURE_PATH: &str = "config/architectures/cifar10_cnn_baseline.json";

// Main Logic
// ============================================================================

// CNN with shared layer abstractions.

mod config;
mod model;
mod persistence;

use config::*;
use model::*;
use persistence::*;

/// Entry point that trains a small convolutional neural network on the CIFAR-10 dataset and evaluates it on the test set.
///
/// The program loads CIFAR-10 binary batches from ./data/cifar-10-batches-bin/, splits the training data into training and validation sets, constructs a simple Conv->ReLU->2x2MaxPool->FC network, and runs SGD training with per-epoch validation, logging, early stopping, and learning-rate scheduling. The best model (by validation loss) is saved to cifar10_cnn_model_best.bin and final test accuracy is printed.
///
/// # Examples
///
/// ```ignore
/// // Running the full training pipeline (heavy; ignored in doctests).
/// main();
/// ```
fn main() {
    // Parse command-line arguments for config file paths
    let args: Vec<String> = env::args().collect();
    let arch_path = parse_arch_path(&args);
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    // Load config
    println!("=== CIFAR-10 CNN Training ===");
    println!("Loading training configuration from: {}", config_path);
    let config = match load_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config from '{}': {}", config_path, e);
            eprintln!("Please ensure the config file exists and is valid JSON.");
            process::exit(1);
        }
    };

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

    // Extract augmentation parameters from config
    let enable_augmentation = config.enable_augmentation.unwrap_or(false);
    let horizontal_flip_prob = config.horizontal_flip_prob;
    let random_crop_padding = config.random_crop_padding;
    let brightness_jitter = config.brightness_jitter;
    let contrast_jitter = config.contrast_jitter;
    let saturation_jitter = config.saturation_jitter;

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

    println!("Loading CIFAR-10...");

    // Load all 5 training batches
    let train_filenames = vec![
        "./data/cifar-10-batches-bin/data_batch_1.bin",
        "./data/cifar-10-batches-bin/data_batch_2.bin",
        "./data/cifar-10-batches-bin/data_batch_3.bin",
        "./data/cifar-10-batches-bin/data_batch_4.bin",
        "./data/cifar-10-batches-bin/data_batch_5.bin",
    ];

    let (all_train_images, all_train_labels) = read_cifar10_batches(&train_filenames)
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
    let (test_images, test_labels) =
        read_cifar10_batch("./data/cifar-10-batches-bin/test_batch.bin").unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });

    // Randomly split training data into train and validation sets.
    let total_train_samples = all_train_images.len() / NUM_INPUTS;
    let validation_samples = (total_train_samples as f32 * validation_split) as usize;
    let actual_train_samples = total_train_samples - validation_samples;

    let mut rng = SimpleRng::new(1);
    rng.reseed_from_time();

    let mut split_indices: Vec<usize> = (0..total_train_samples).collect();
    rng.shuffle_usize(&mut split_indices);

    let mut train_images = vec![0.0f32; actual_train_samples * NUM_INPUTS];
    let mut train_labels = vec![0u8; actual_train_samples];
    let mut val_images = vec![0.0f32; validation_samples * NUM_INPUTS];
    let mut val_labels = vec![0u8; validation_samples];

    for (i, &src_index) in split_indices.iter().enumerate() {
        let src_start = src_index * NUM_INPUTS;
        if i < validation_samples {
            let dst_start = i * NUM_INPUTS;
            val_images[dst_start..dst_start + NUM_INPUTS]
                .copy_from_slice(&all_train_images[src_start..src_start + NUM_INPUTS]);
            val_labels[i] = all_train_labels[src_index];
        } else {
            let dst_i = i - validation_samples;
            let dst_start = dst_i * NUM_INPUTS;
            train_images[dst_start..dst_start + NUM_INPUTS]
                .copy_from_slice(&all_train_images[src_start..src_start + NUM_INPUTS]);
            train_labels[dst_i] = all_train_labels[src_index];
        }
    }

    let train_n = actual_train_samples;
    let test_n = test_labels.len();
    println!(
        "Data split: {} training samples, {} validation samples, {} test samples",
        actual_train_samples, validation_samples, test_n
    );

    let mut model = init_cnn(&mut rng, arch_path.as_deref()).unwrap_or_else(|e| {
        eprintln!("{}", e);
        process::exit(1);
    });

    // Create step debugger
    let mut debugger = StepDebugger::new(step_debug_enabled);

    // Create per-layer optimizers (one per layer to avoid shared optimizer state issues
    // with different parameter sizes across layers)
    let mut layer_optimizers: Vec<Box<dyn Optimizer>> = (0..model.layers.len())
        .map(|_| create_optimizer(&config, learning_rate))
        .collect();

    // Print optimizer info
    let optimizer_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    println!("Optimizer: {}", optimizer_type.to_uppercase());

    // Create learning rate scheduler
    let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(&config_path));

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let mut logger =
        CsvTrainingLogger::new("./logs/training_loss_cifar10_cnn.csv").unwrap_or_else(|_| {
            eprintln!("Could not create logs/training_loss_cifar10_cnn.csv");
            process::exit(1);
        });
    logger.write_header().unwrap_or_else(|_| {
        eprintln!("Could not write CSV header to logs/training_loss_cifar10_cnn.csv");
        process::exit(1);
    });

    // Create gradient logging file.
    let mut gradient_logger = CsvGradientLogger::new("./logs/gradients_cifar10.csv")
        .unwrap_or_else(|_| {
            eprintln!("Could not create logs/gradients_cifar10.csv");
            process::exit(1);
        });
    gradient_logger.write_header().unwrap_or_else(|_| {
        eprintln!("Could not write CSV header to logs/gradients_cifar10.csv");
        process::exit(1);
    });

    // Training buffers (reused each batch to avoid allocations).
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; batch_size];

    // Buffers for generic forward/backward passes
    let num_layers = model.layers.len();
    let mut activations = LayerActivations::new(num_layers);
    let mut temp_buffer = Vec::new();
    let mut grad_buffer1 = Vec::new();
    let mut grad_buffer2 = Vec::new();

    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];
    let mut delta = vec![0.0f32; batch_size * NUM_CLASSES];

    // Validation buffers (reused each epoch to avoid repeated allocations).
    let mut val_batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut val_activations = LayerActivations::new(num_layers);
    let mut val_temp_buffer = Vec::new();
    let mut val_logits = vec![0.0f32; batch_size * NUM_CLASSES];

    let mut indices: Vec<usize> = (0..train_n).collect();

    // Early stopping state
    let mut early_stopping = EarlyStopping::new(early_stopping_patience, early_stopping_min_delta);

    println!(
        "Training CIFAR-10 CNN: epochs={} batch={} lr={}",
        epochs, batch_size, learning_rate
    );

    for epoch in 0..epochs {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);
        let current_lr = scheduler.get_lr();

        debugger.on_epoch_start(epoch + 1);

        // Update learning rate on all per-layer optimizers
        for opt in layer_optimizers.iter_mut() {
            opt.set_learning_rate(current_lr);
        }

        // Set BatchNorm and Dropout to training mode
        set_training_mode(&mut model, true);

        let mut total_loss = 0.0f32;

        // Accumulate gradient norms for this epoch (one slot per layer).
        let mut layer_weight_grad_sums = vec![0.0f32; num_layers];
        let mut layer_bias_grad_sums = vec![0.0f32; num_layers];
        let mut batch_count_total = 0usize;

        for batch_start in (0..train_n).step_by(batch_size) {
            let batch = (train_n - batch_start).min(batch_size);
            let scale = 1.0f32;
            let batch_idx = batch_start / batch_size + 1;
            let total_batches = train_n.div_ceil(batch_size);

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
                if enable_augmentation {
                    saturation_jitter
                } else {
                    None
                },
                if enable_augmentation {
                    Some(&mut rng)
                } else {
                    None
                },
            );

            // Forward pass through all layers
            let output_idx = forward_pass(
                &mut model,
                batch,
                &batch_inputs,
                &mut activations,
                &mut temp_buffer,
            );

            // Get logits from the last layer output
            let logits_slice = &mut activations.data[output_idx];
            logits[..logits_slice.len()].copy_from_slice(logits_slice);

            // Softmax + loss + gradient at logits.
            softmax_rows(&mut logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);
            let batch_loss = compute_softmax_cross_entropy(
                &logits[..batch * NUM_CLASSES],
                &batch_labels,
                batch,
                NUM_CLASSES,
                &mut delta,
                scale,
            );
            total_loss += batch_loss;

            debugger.after_loss(
                batch_loss / batch as f32,
                &delta[..batch * NUM_CLASSES],
                batch,
                NUM_CLASSES,
            );

            // Backward pass through all layers
            backward_pass(
                &mut model,
                batch,
                &batch_inputs,
                &activations,
                &delta[..batch * NUM_CLASSES],
                &mut grad_buffer1,
                &mut grad_buffer2,
            );

            // Log gradient magnitudes before parameter update (accumulate for epoch).
            // Also collect current batch gradients for debugger.
            let mut layer_names: Vec<String> = Vec::new();
            let mut gradient_tuples: Vec<(f32, f32)> = Vec::new();

            for (layer_idx, layer) in model.layers.iter().enumerate() {
                let any_layer = layer.as_ref().as_any();
                if let Some(conv_layer) = any_layer.downcast_ref::<Conv2DLayer>() {
                    let (w_norm, b_norm) = conv_layer.get_gradient_magnitude();
                    layer_weight_grad_sums[layer_idx] += w_norm;
                    layer_bias_grad_sums[layer_idx] += b_norm;
                    layer_names.push(format!("conv_{}", layer_idx));
                    gradient_tuples.push((w_norm, b_norm));
                } else if let Some(dense_layer) = any_layer.downcast_ref::<DenseLayer>() {
                    let (w_norm, b_norm) = dense_layer.get_gradient_magnitude();
                    layer_weight_grad_sums[layer_idx] += w_norm;
                    layer_bias_grad_sums[layer_idx] += b_norm;
                    layer_names.push(format!("dense_{}", layer_idx));
                    gradient_tuples.push((w_norm, b_norm));
                }
            }
            batch_count_total += 1;

            // Build gradient info for debugger with string references
            let gradient_info: Vec<(&str, f32, f32)> = layer_names
                .iter()
                .zip(gradient_tuples.iter())
                .map(|(name, &(w, b))| (name.as_str(), w, b))
                .collect();

            debugger.after_update(&gradient_info, current_lr);

            // Update parameters for all layers using per-layer optimizers
            for (layer, opt) in model.layers.iter_mut().zip(layer_optimizers.iter_mut()) {
                layer.update_with_optimizer(opt.as_mut());
            }

            // Print progress every 100 batches
            let batch_idx = batch_start / batch_size;
            let total_batches = train_n.div_ceil(batch_size);
            if batch_idx % 100 == 0 || batch_idx == total_batches - 1 {
                let progress_pct = (batch_idx as f32 / total_batches as f32) * 100.0;
                print!(
                    "\r  Epoch {}/{}: Batch {}/{} ({:.1}%), Loss: {:.4}",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    total_batches,
                    progress_pct,
                    total_loss / (batch_idx + 1) as f32
                );
                std::io::stdout().flush().unwrap();
            }
        }

        println!(); // Newline after progress indicator
        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;

        // Write gradient magnitudes (averaged across batches) to gradient log.
        if batch_count_total > 0 {
            let num_batches = batch_count_total as f32;
            for (layer_idx, layer) in model.layers.iter().enumerate() {
                let any_layer = layer.as_ref().as_any();
                let layer_name = if any_layer.downcast_ref::<Conv2DLayer>().is_some() {
                    format!("layer_{}_conv", layer_idx)
                } else if any_layer.downcast_ref::<DenseLayer>().is_some() {
                    format!("layer_{}_dense", layer_idx)
                } else {
                    continue;
                };
                let avg_w = layer_weight_grad_sums[layer_idx] / num_batches;
                let avg_b = layer_bias_grad_sums[layer_idx] / num_batches;
                gradient_logger
                    .write_layer(epoch + 1, &layer_name, avg_w, avg_b)
                    .unwrap_or_else(|_| {
                        eprintln!("Failed writing gradient data.");
                        process::exit(1);
                    });
            }
            gradient_logger.flush().unwrap_or_else(|_| {
                eprintln!("Failed flushing gradient log.");
                process::exit(1);
            });
        }

        // Set BatchNorm and Dropout to inference mode for validation
        set_training_mode(&mut model, false);

        // Evaluate on validation set
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        for batch_start in (0..validation_samples).step_by(batch_size) {
            let batch_count = (validation_samples - batch_start).min(batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_images[input_start..input_start + input_len]);

            // Forward pass through all layers
            let output_idx = forward_pass(
                &mut model,
                batch_count,
                &val_batch_inputs,
                &mut val_activations,
                &mut val_temp_buffer,
            );

            // Get logits from the last layer output
            let logits_slice = &mut val_activations.data[output_idx];
            val_logits[..logits_slice.len()].copy_from_slice(logits_slice);

            // Apply softmax and compute loss + accuracy using shared utility
            softmax_rows(
                &mut val_logits[..batch_count * NUM_CLASSES],
                batch_count,
                NUM_CLASSES,
            );

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
                save_model(&model, "cifar10_cnn_model_best.bin");
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

    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}

#[cfg(test)]
mod tests;
