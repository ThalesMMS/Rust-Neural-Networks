use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::optimizers::{Adam, Optimizer, SGD};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::{
    compute_softmax_cross_entropy, evaluate_batch_accuracy, gather_batch, parse_config_path,
    parse_step_flag, print_training_config, CsvTrainingLogger, EarlyStopping, EarlyStoppingAction,
    TrainingMetrics,
};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
use rust_neural_networks::utils::rng::SimpleRng;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use rust_neural_networks::gpu;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use std::sync::Arc;

// MLP with minibatches for MNIST (Rust port for study).
// Uses shared library layers and utilities.
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const IMG_CHANNELS: usize = 1; // Grayscale
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_mlp_default.json";

// Optimizer selection: "sgd" or "adam"
const OPTIMIZER_TYPE: &str = "adam";

// ============================================================================
// Main Logic
// ============================================================================

/// Groups the input dataset slices and sample count for a training or validation split.
///
/// Holds borrowed slices so the data does not need to be copied. The lifetime
/// parameter `'a` ties the struct to the lifetime of the underlying data buffers.
///
/// # Fields
///
/// - `images` – Flattened row-major pixel data (one image per row of `NUM_INPUTS` values).
/// - `labels` – Class labels for each image (values 0–9 for MNIST).
/// - `num_samples` – Number of samples (rows) in this dataset split.
pub struct DataSet<'a> {
    /// Flattened row-major pixel data (one image per row of NUM_INPUTS values).
    pub images: &'a [f32],
    /// Class labels for each image (values 0–9 for MNIST).
    pub labels: &'a [u8],
    /// Number of samples (rows) in this dataset split.
    pub num_samples: usize,
}

/// Groups scalar training hyperparameters to avoid long parameter lists.
///
/// Collects the five scalar training settings that control optimisation,
/// batching, and early stopping into a single struct.
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
    /// Whether data augmentation is enabled during training.
    pub enable_augmentation: bool,
    /// Probability of applying random horizontal flip (None to disable).
    pub horizontal_flip_prob: Option<f32>,
    /// Padding for random crop augmentation in pixels (None to disable).
    pub random_crop_padding: Option<usize>,
    /// Brightness jitter delta for color augmentation (None to disable).
    pub brightness_jitter: Option<f32>,
    /// Contrast jitter delta for color augmentation (None to disable).
    pub contrast_jitter: Option<f32>,
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: DenseLayer,
    output_layer: DenseLayer,
}

// Network construction 784 -> 512 -> 10.
/// Create a two-layer feedforward neural network with randomized parameters.
///
/// The network contains a hidden DenseLayer sized NUM_INPUTS -> NUM_HIDDEN and an
/// output DenseLayer sized NUM_HIDDEN -> NUM_OUTPUTS. Layer weights and biases
/// are initialized using the provided RNG. When compiled with a GPU feature and
/// a non-`None` GPU backend is supplied, GPU-backed layers are constructed.
///
/// # Examples
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let nn = initialize_network(&mut rng);
/// assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
/// ```
///
/// ```ignore
/// // With GPU feature enabled:
/// use std::sync::Arc;
/// let mut rng = SimpleRng::new(42);
/// let gpu_backend: Option<Arc<dyn gpu::GpuBackend>> = None;
/// let nn = initialize_network(&mut rng, &gpu_backend);
/// ```
fn initialize_network(
    rng: &mut SimpleRng,
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))] gpu_backend: &Option<
        Arc<dyn gpu::GpuBackend>,
    >,
) -> NeuralNetwork {
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    {
        if let Some(ref backend) = gpu_backend {
            let hidden_layer =
                DenseLayer::new_with_gpu(NUM_INPUTS, NUM_HIDDEN, rng, Arc::clone(backend));
            let output_layer =
                DenseLayer::new_with_gpu(NUM_HIDDEN, NUM_OUTPUTS, rng, Arc::clone(backend));
            return NeuralNetwork {
                hidden_layer,
                output_layer,
            };
        }
    }

    let hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

// Training with shuffling and minibatches.
/// Trains the neural network for the configured number of epochs using minibatch updates and appends per-epoch metrics to ./logs/training_loss_{OPTIMIZER_TYPE}.txt.
///
/// Evaluates on the provided validation set each epoch, uses the scheduler's current learning rate for parameter updates, saves the best model to "mnist_model_best.bin" when validation loss improves, and supports early stopping based on validation loss and configured patience. Progress (training loss, validation loss, validation accuracy, learning rate, and epoch time) is printed to stdout and recorded in CSV format.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut nn = initialize_network(&mut rng);
/// let images = vec![0.0f32; NUM_INPUTS * 1];
/// let labels = vec![0u8; 1];
/// let val_images = vec![0.0f32; NUM_INPUTS * 1];
/// let val_labels = vec![0u8; 1];
/// let train_data = DataSet { images: &images, labels: &labels, num_samples: 1 };
/// let val_data = DataSet { images: &val_images, labels: &val_labels, num_samples: 1 };
/// let params = TrainHyperparams { learning_rate: 0.01, epochs: 1, batch_size: 64, early_stopping_patience: 3, early_stopping_min_delta: 0.001 };
/// train(&mut nn, &train_data, &val_data, &mut rng, scheduler, &params);
/// ```
fn train(
    nn: &mut NeuralNetwork,
    train_data: &DataSet,
    val_data: &DataSet,
    rng: &mut SimpleRng,
    scheduler: &mut dyn LRScheduler,
    params: &TrainHyperparams,
    aug_rng: &mut SimpleRng,
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

    let log_filename = format!("./logs/training_loss_{}.txt", OPTIMIZER_TYPE);
    let mut csv_logger = CsvTrainingLogger::new(&log_filename).unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    csv_logger.write_header().unwrap_or_else(|_| {
        eprintln!("Failed writing CSV header.");
        process::exit(1);
    });

    // Create gradient logging file
    let gradient_log_filename = "./logs/gradients_mlp.csv";
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
    println!(
        "Using {} optimizer with learning rate {}",
        OPTIMIZER_TYPE.to_uppercase(),
        params.learning_rate
    );

    let mut batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; params.batch_size];
    let mut a1 = vec![0.0f32; params.batch_size * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; params.batch_size * NUM_OUTPUTS];
    let mut dz2 = vec![0.0f32; params.batch_size * NUM_OUTPUTS];
    let mut dz1 = vec![0.0f32; params.batch_size * NUM_HIDDEN];

    let mut indices: Vec<usize> = (0..train_data.num_samples).collect();

    let mut unused_grad = vec![0.0f32; params.batch_size * NUM_INPUTS]; // Preallocate reusable buffer.

    // Early stopping using shared EarlyStopping struct
    let mut early_stopping = EarlyStopping::new(
        params.early_stopping_patience,
        params.early_stopping_min_delta,
    );

    let total_batches = (train_data.num_samples + params.batch_size - 1) / params.batch_size;

    for epoch in 0..params.epochs {
        let mut total_loss = 0.0f32;
        let start_time = Instant::now();
        let current_lr = scheduler.get_lr();
        optimizer.set_learning_rate(current_lr);

        // Accumulate gradient norms for this epoch
        let mut hidden_weight_grad_sum = 0.0f32;
        let mut hidden_bias_grad_sum = 0.0f32;
        let mut output_weight_grad_sum = 0.0f32;
        let mut output_bias_grad_sum = 0.0f32;
        let mut batch_count_total = 0usize;

        debugger.on_epoch_start(epoch + 1);

        // Fisher-Yates shuffle.
        if train_data.num_samples > 1 {
            for i in (1..train_data.num_samples).rev() {
                let j = rng.gen_usize(i + 1);
                indices.swap(i, j);
            }
        }

        for batch_start in (0..train_data.num_samples).step_by(params.batch_size) {
            let batch_count = (train_data.num_samples - batch_start).min(params.batch_size);
            let batch_idx = batch_start / params.batch_size + 1;

            debugger.set_context(epoch + 1, batch_idx, total_batches, batch_count);

            // Gather a random mini-batch into contiguous buffers.
            // Apply augmentation only during training if enabled.
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
                if params.enable_augmentation {
                    params.horizontal_flip_prob
                } else {
                    None
                },
                if params.enable_augmentation {
                    params.random_crop_padding
                } else {
                    None
                },
                if params.enable_augmentation {
                    params.brightness_jitter
                } else {
                    None
                },
                if params.enable_augmentation {
                    params.contrast_jitter
                } else {
                    None
                },
                None, // saturation_jitter not applicable for grayscale MNIST
                if params.enable_augmentation {
                    Some(aug_rng)
                } else {
                    None
                },
            );

            // Forward: hidden layer.
            let a1_len = batch_count * NUM_HIDDEN;
            debugger.before_forward(
                "hidden_layer",
                &batch_inputs,
                batch_count,
                NUM_INPUTS,
                NUM_HIDDEN,
                nn.hidden_layer.parameter_count(),
            );
            nn.hidden_layer.forward(&batch_inputs, &mut a1, batch_count);
            debugger.after_forward("hidden_layer", &a1, batch_count, NUM_HIDDEN);
            relu_inplace(&mut a1[..a1_len]);
            debugger.after_activation("ReLU", &a1[..a1_len], batch_count, NUM_HIDDEN);

            // Forward: output layer.
            let a2_len = batch_count * NUM_OUTPUTS;
            debugger.before_forward(
                "output_layer",
                &a1,
                batch_count,
                NUM_HIDDEN,
                NUM_OUTPUTS,
                nn.output_layer.parameter_count(),
            );
            nn.output_layer.forward(&a1, &mut a2, batch_count);
            debugger.after_forward("output_layer", &a2, batch_count, NUM_OUTPUTS);
            assert_eq!(
                a2[..a2_len].len(),
                batch_count * NUM_OUTPUTS,
                "Buffer size mismatch before softmax_rows"
            );
            softmax_rows(&mut a2[..a2_len], batch_count, NUM_OUTPUTS);
            debugger.after_activation("Softmax", &a2[..a2_len], batch_count, NUM_OUTPUTS);

            // Output delta and loss using shared compute_softmax_cross_entropy (scale=1.0).
            let batch_loss = compute_softmax_cross_entropy(
                &a2[..a2_len],
                &batch_labels[..batch_count],
                batch_count,
                NUM_OUTPUTS,
                &mut dz2,
                1.0,
            );
            total_loss += batch_loss;
            debugger.after_loss(
                batch_loss / batch_count as f32,
                &dz2,
                batch_count,
                NUM_OUTPUTS,
            );

            // Backward: output layer.
            nn.output_layer.backward(&a1, &dz2, &mut dz1, batch_count);
            debugger.after_backward(
                "output_layer",
                &dz2,
                &dz1,
                batch_count,
                NUM_OUTPUTS,
                NUM_HIDDEN,
            );

            // Apply ReLU derivative to hidden layer gradient.
            let dz1_len = batch_count * NUM_HIDDEN;
            for i in 0..dz1_len {
                if a1[i] <= 0.0 {
                    dz1[i] = 0.0;
                }
            }
            debugger.after_relu_derivative(&dz1[..dz1_len], batch_count, NUM_HIDDEN);

            // Backward: hidden layer.
            let grad_len = batch_count * NUM_INPUTS;
            nn.hidden_layer.backward(
                &batch_inputs,
                &dz1,
                &mut unused_grad[..grad_len],
                batch_count,
            );
            debugger.after_backward(
                "hidden_layer",
                &dz1,
                &unused_grad[..grad_len],
                batch_count,
                NUM_HIDDEN,
                NUM_INPUTS,
            );

            // Log gradient magnitudes before parameter update (accumulate for epoch)
            let (hidden_w_norm, hidden_b_norm) = nn.hidden_layer.get_gradient_magnitude();
            let (output_w_norm, output_b_norm) = nn.output_layer.get_gradient_magnitude();
            hidden_weight_grad_sum += hidden_w_norm;
            hidden_bias_grad_sum += hidden_b_norm;
            output_weight_grad_sum += output_w_norm;
            output_bias_grad_sum += output_b_norm;
            batch_count_total += 1;

            debugger.after_update(
                &[
                    ("hidden_layer", hidden_w_norm, hidden_b_norm),
                    ("output_layer", output_w_norm, output_b_norm),
                ],
                current_lr,
            );

            // Update parameters using optimizer.
            nn.output_layer.update_with_optimizer(optimizer.as_mut());
            nn.hidden_layer.update_with_optimizer(optimizer.as_mut());
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_loss = total_loss / train_data.num_samples as f32;

        // Evaluate on validation set using shared evaluate_batch_accuracy
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_batch_inputs = vec![0.0f32; params.batch_size * NUM_INPUTS];
        let mut val_a1 = vec![0.0f32; params.batch_size * NUM_HIDDEN];
        let mut val_a2 = vec![0.0f32; params.batch_size * NUM_OUTPUTS];

        for batch_start in (0..val_data.num_samples).step_by(params.batch_size) {
            let batch_count = (val_data.num_samples - batch_start).min(params.batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_data.images[input_start..input_start + input_len]);

            // Forward: hidden layer
            let val_a1_len = batch_count * NUM_HIDDEN;
            nn.hidden_layer
                .forward(&val_batch_inputs, &mut val_a1, batch_count);
            relu_inplace(&mut val_a1[..val_a1_len]);

            // Forward: output layer
            let val_a2_len = batch_count * NUM_OUTPUTS;
            nn.output_layer.forward(&val_a1, &mut val_a2, batch_count);
            softmax_rows(&mut val_a2[..val_a2_len], batch_count, NUM_OUTPUTS);

            // Compute loss and accuracy using shared evaluate_batch_accuracy
            let (batch_loss, batch_correct) = evaluate_batch_accuracy(
                &val_a2[..val_a2_len],
                &val_data.labels[batch_start..batch_start + batch_count],
                batch_count,
                NUM_OUTPUTS,
            );
            val_total_loss += batch_loss;
            val_correct += batch_correct;
        }

        let val_average_loss = val_total_loss / val_data.num_samples as f32;
        let val_accuracy = val_correct as f32 / val_data.num_samples as f32 * 100.0;

        // Write gradient magnitudes (averaged across batches) to gradient log
        let num_batches = batch_count_total as f32;
        let hidden_w_avg = hidden_weight_grad_sum / num_batches;
        let hidden_b_avg = hidden_bias_grad_sum / num_batches;
        let output_w_avg = output_weight_grad_sum / num_batches;
        let output_b_avg = output_bias_grad_sum / num_batches;

        writeln!(
            gradient_file,
            "{},hidden_layer,{},{}",
            epoch + 1,
            hidden_w_avg,
            hidden_b_avg
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing gradient data.");
            process::exit(1);
        });

        writeln!(
            gradient_file,
            "{},output_layer,{},{}",
            epoch + 1,
            output_w_avg,
            output_b_avg
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing gradient data.");
            process::exit(1);
        });

        println!(
            "Epoch {}, Loss: {:.6}, Val Loss: {:.6}, Val Acc: {:.2}%, LR: {:.6}, Time: {:.6}",
            epoch + 1,
            average_loss,
            val_average_loss,
            val_accuracy,
            current_lr,
            duration
        );

        // Write epoch metrics using shared CsvTrainingLogger
        let metrics = TrainingMetrics {
            train_loss: average_loss,
            val_loss: val_average_loss,
            val_accuracy,
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
            EarlyStoppingAction::Improved => save_model(nn, "mnist_model_best.bin"),
            EarlyStoppingAction::Stop => {
                println!(
                    "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
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

// Evaluate accuracy on the test set using batches.
/// Evaluates a trained network on the provided dataset and prints the test accuracy.
///
/// The function processes the dataset in minibatches, performs forward passes (hidden ReLU, output softmax),
/// selects the highest-probability class per sample, counts correct predictions, and prints accuracy as a percentage.
///
/// # Examples
///
/// ```
/// // Assume `nn`, `images`, `labels`, and `num_samples` are prepared:
/// // let mut rng = SimpleRng::new();
/// // let nn = initialize_network(&mut rng);
/// // let images: Vec<f32> = ...; // flattened images
/// // let labels: Vec<u8> = ...; // one label per image
/// test(&nn, &images, &labels, num_samples);
/// ```
fn test(nn: &NeuralNetwork, images: &[f32], labels: &[u8], num_samples: usize, batch_size: usize) {
    let mut correct_predictions = 0usize;
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut a1 = vec![0.0f32; batch_size * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; batch_size * NUM_OUTPUTS];

    for batch_start in (0..num_samples).step_by(batch_size) {
        let batch_count = (num_samples - batch_start).min(batch_size);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len].copy_from_slice(&images[input_start..input_start + input_len]);

        // Forward: hidden layer.
        let a1_len = batch_count * NUM_HIDDEN;
        nn.hidden_layer.forward(&batch_inputs, &mut a1, batch_count);
        relu_inplace(&mut a1[..a1_len]);

        // Forward: output layer.
        let a2_len = batch_count * NUM_OUTPUTS;
        nn.output_layer.forward(&a1, &mut a2, batch_count);
        assert_eq!(
            a2[..a2_len].len(),
            batch_count * NUM_OUTPUTS,
            "Buffer size mismatch before softmax_rows in test"
        );
        softmax_rows(&mut a2[..a2_len], batch_count, NUM_OUTPUTS);

        for row_idx in 0..batch_count {
            let row_start = row_idx * NUM_OUTPUTS;
            let row = &a2[row_start..row_start + NUM_OUTPUTS];
            let mut predicted = 0usize;
            let mut max_prob = row[0];
            for (i, &value) in row.iter().enumerate().skip(1) {
                if value > max_prob {
                    max_prob = value;
                    predicted = i;
                }
            }
            if predicted == labels[batch_start + row_idx] as usize {
                correct_predictions += 1;
            }
        }
    }

    let accuracy = correct_predictions as f32 / num_samples as f32 * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
}

// Save the model in binary (little-endian i32 + f32).
/// Serializes the neural network to a binary file using little-endian encoding.
///
/// The file contains, in order:
/// 1. Three 32-bit integers: hidden layer input size, hidden layer output size, and output layer output size.
/// 2. All hidden layer weights as 32-bit floats.
/// 3. All hidden layer biases as 32-bit floats.
/// 4. All output layer weights as 32-bit floats.
/// 5. All output layer biases as 32-bit floats.
///
/// The function terminates the process with an error message if the file cannot be created or any write fails.
///
/// # Examples
///
/// ```
/// // Serializes `nn` to "mnist_model.bin".
/// save_model(&nn, "mnist_model.bin");
/// ```
fn save_model(nn: &NeuralNetwork, filename: &str) {
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

    write_i32(&mut writer, nn.hidden_layer.input_size() as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size() as i32);
    write_i32(&mut writer, nn.output_layer.output_size() as i32);

    for &value in nn.hidden_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in nn.hidden_layer.biases() {
        write_f32(&mut writer, value);
    }
    for &value in nn.output_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in nn.output_layer.biases() {
        write_f32(&mut writer, value);
    }

    println!("Model saved to {}", filename);
}

/// Create a learning-rate scheduler from the provided learning rate, epoch count, and optional config path.
///
/// The scheduler is configured using the explicit `learning_rate` and `epochs` values and may be further
/// influenced by settings found at `config_path` when provided.
///
/// # Examples
///
/// ```
/// let scheduler = scheduler_from_args(0.001, 50, None);
/// ```
fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Run the full MNIST training, evaluation, and model-save pipeline.
///
/// This binary entrypoint parses an optional CLI config path (uses the bundled default if omitted),
/// loads configuration and MNIST data, constructs the scheduler and network (optionally enabling GPU
/// when compiled with GPU features), executes training with early stopping and augmentation options,
/// evaluates on the test set, and writes the trained parameters to `mnist_model.bin`. Runtime errors
/// such as configuration or data-load failures cause the process to exit with an error message.
///
/// # Examples
///
/// ```
/// // Typical invocation in a binary; runs the full training and evaluation workflow.
/// main();
/// ```
fn main() {
    let program_start = Instant::now();

    // Parse command-line arguments for config file path using shared parse_config_path
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    // Load config
    println!("=== MNIST MLP Training ===");
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
    let learning_rate = config.learning_rate.unwrap_or(0.01);
    let epochs = config.epochs.unwrap_or(10);
    let batch_size = config.batch_size.unwrap_or(64);
    let validation_split = config.validation_split.unwrap_or(0.1);
    let early_stopping_patience = config.early_stopping_patience.unwrap_or(3);
    let early_stopping_min_delta = config.early_stopping_min_delta.unwrap_or(0.001);

    // Extract augmentation parameters from config
    let enable_augmentation = config.enable_augmentation.unwrap_or(false);
    let horizontal_flip_prob = config.horizontal_flip_prob;
    let random_crop_padding = config.random_crop_padding;
    let brightness_jitter = config.brightness_jitter;
    let contrast_jitter = config.contrast_jitter;

    // GPU initialization based on config
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    let gpu_backend = {
        let requested = config.gpu_backend.as_deref().unwrap_or("auto");
        if requested == "cpu" {
            println!("GPU disabled by config (gpu_backend: \"cpu\")");
            None
        } else {
            match gpu::create_gpu_backend() {
                Some(backend) => {
                    gpu::print_backend_info(&*backend);
                    println!("GPU acceleration enabled for training");
                    Some(backend)
                }
                None => {
                    println!("No GPU available, falling back to CPU");
                    None
                }
            }
        }
    };

    // Resolve step-through debug mode from CLI flag or config
    let step_debug_enabled = parse_step_flag(&args) || config.step_debug.unwrap_or(false);

    // Print loaded configuration using shared print_training_config
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

    println!("Initializing neural network...");
    let mut rng = SimpleRng::new(1);
    let mut nn = initialize_network(
        &mut rng,
        #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
        &gpu_backend,
    );

    // Augmentation RNG (reseeded from time for randomness)
    let mut aug_rng = SimpleRng::new(2);
    aug_rng.reseed_from_time();

    let mut debugger = StepDebugger::new(step_debug_enabled);

    println!("Training neural network...");
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
        enable_augmentation,
        horizontal_flip_prob,
        random_crop_padding,
        brightness_jitter,
        contrast_jitter,
    };
    train(
        &mut nn,
        &train_data,
        &val_data,
        &mut rng,
        scheduler.as_mut(),
        &hyperparams,
        &mut aug_rng,
        &mut debugger,
    );
    let train_time = train_start.elapsed().as_secs_f64();
    println!("Total training time: {:.2} seconds", train_time);

    println!("Testing neural network...");
    let test_start = Instant::now();
    test(&nn, &test_images, &test_labels, test_samples, batch_size);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Testing time: {:.2} seconds", test_time);

    println!("Saving model...");
    save_model(&nn, "mnist_model.bin");

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Data loading time: {:.2} seconds", load_time);
    println!("Total training time: {:.2} seconds", train_time);
    println!("Testing time: {:.2} seconds", test_time);
    println!("Total program time: {:.2} seconds", total_time);
    println!("========================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_network() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(
            &mut rng,
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            &None,
        );

        assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
        assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
    }
}