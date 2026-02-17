use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::optimizers::{Adam, Optimizer, SGD};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
use rust_neural_networks::utils::rng::SimpleRng;

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
/// Create a feedforward NeuralNetwork with randomized DenseLayer parameters.
///
/// The returned network contains a hidden DenseLayer sized NUM_INPUTS -> NUM_HIDDEN
/// and an output DenseLayer sized NUM_HIDDEN -> NUM_OUTPUTS. Layer parameters
/// are initialized using the provided RNG (the RNG is reseeded inside the function).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let nn = initialize_network(&mut rng);
/// assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
/// assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
/// ```
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    let hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

/// Computes the total cross-entropy loss for a batch of softmax outputs and writes the corresponding gradient (softmax-backed deltas) into `delta`.
///
/// For each of the first `rows` samples, the function:
/// - Accumulates negative log probability of the true class (using a small epsilon to avoid log(0)) into the returned loss.
/// - Writes a gradient row into `delta` equal to the predicted probabilities with `1.0` subtracted at the true label index (i.e., `p_i` for i != y, and `p_y - 1.0` for the true class).
///
/// # Parameters
///
/// - `outputs`: Flattened row-major softmax output probabilities with length at least `rows * cols`.
/// - `labels`: True class labels for the batch; only the first `rows` entries are used.
/// - `rows`: Number of samples (rows) to process from `outputs` and `labels`.
/// - `cols`: Number of classes (columns) per sample.
/// - `delta`: Mutable flattened buffer where computed gradient rows are written; must have length at least `rows * cols`.
///
/// # Returns
///
/// Total cross-entropy loss summed over the processed `rows` samples.
///
/// # Examples
///
/// ```
/// let outputs = [0.9f32, 0.1f32]; // one sample, two-class softmax
/// let labels = [0u8];
/// let mut delta = [0.0f32; 2];
/// let loss = compute_delta_and_loss(&outputs, &labels, 1, 2, &mut delta);
/// let expected_loss = -(0.9f32).ln();
/// assert!((loss - expected_loss).abs() < 1e-6);
/// // gradient: true class probability minus 1, other classes remain as probabilities
/// assert!((delta[0] - (-0.1f32)).abs() < 1e-6);
/// assert!((delta[1] - 0.1f32).abs() < 1e-6);
/// ```
fn compute_delta_and_loss(
    outputs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
) -> f32 {
    let mut total_loss = 0.0f32;
    let epsilon = 1e-9f32;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;
        let prob = outputs[row_start + label].max(epsilon);
        total_loss -= prob.ln();

        let row = &outputs[row_start..row_start + cols];
        let delta_row = &mut delta[row_start..row_start + cols];
        for (j, value) in row.iter().enumerate() {
            let mut v = *value;
            if j == label {
                v -= 1.0;
            }
            delta_row[j] = v;
        }
    }

    total_loss
}

// Copy a subset of images/labels into contiguous batch buffers with optional augmentation.
/// Copies a contiguous mini-batch of examples and their labels into preallocated output buffers.
///
/// The function reads `count` examples by mapping `indices[start..start+count]` into `images` and
/// `labels`, copying each example's NUM_INPUTS floats into `out_inputs` and the corresponding label
/// into `out_labels` in batch order.
///
/// If augmentation parameters are provided, applies data augmentation to each image after copying.
/// Augmentations are applied in the following order:
/// 1. Random crop (if `crop_padding` is Some)
/// 2. Random horizontal flip (if `flip_prob` is Some)
/// 3. Brightness jitter (if `brightness_jitter` is Some)
/// 4. Contrast jitter (if `contrast_jitter` is Some)
///
/// Note: saturation_jitter is not supported for MNIST (grayscale, 1 channel).
///
/// # Parameters
///
/// - `images`: flat slice of all images laid out as consecutive blocks of `NUM_INPUTS` floats.
/// - `labels`: slice of labels corresponding to `images`.
/// - `indices`: permutation or index list used to select examples from the dataset.
/// - `start`: starting offset in `indices` for this batch.
/// - `count`: number of examples to copy into the outputs.
/// - `out_inputs`: destination buffer for the batch inputs; must have length at least `count * NUM_INPUTS`.
/// - `out_labels`: destination buffer for the batch labels; must have length at least `count`.
/// - `flip_prob`: optional probability (0.0-1.0) for random horizontal flip.
/// - `crop_padding`: optional padding amount for random crop (crops back to IMG_W x IMG_H).
/// - `brightness_jitter`: optional brightness jitter delta (applied uniformly).
/// - `contrast_jitter`: optional contrast jitter delta.
/// - `rng`: optional random number generator for augmentation operations.
///
/// # Examples
///
/// ```
/// // prepare a tiny dataset with NUM_INPUTS per example
/// let mut images = vec![0.0f32; NUM_INPUTS * 3];
/// // fill example 1 and 2 with distinguishable values
/// for i in 0..NUM_INPUTS { images[i] = 1.0; images[NUM_INPUTS * 2 + i] = 3.0; }
/// let labels = vec![0u8, 1u8, 2u8];
/// let indices = vec![2usize, 0, 1];
///
/// let mut batch_inputs = vec![0.0f32; NUM_INPUTS * 2];
/// let mut batch_labels = vec![0u8; 2];
///
/// // gather two examples starting from indices[0] => picks examples 2 and 0
/// gather_batch(&images, &labels, &indices, 0, 2, &mut batch_inputs, &mut batch_labels,
///              None, None, None, None, None);
///
/// // verify the labels and a couple of input values
/// assert_eq!(batch_labels, vec![2u8, 0u8]);
/// assert_eq!(batch_inputs[0], 3.0);
/// assert_eq!(batch_inputs[NUM_INPUTS], 1.0);
/// ```
fn gather_batch(
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    start: usize,
    count: usize,
    out_inputs: &mut [f32],
    out_labels: &mut [u8],
    flip_prob: Option<f32>,
    crop_padding: Option<usize>,
    brightness_jitter: Option<f32>,
    contrast_jitter: Option<f32>,
    mut rng: Option<&mut SimpleRng>,
) {
    use rust_neural_networks::data::augmentation::{
        random_brightness, random_contrast, random_crop, random_horizontal_flip,
    };

    for i in 0..count {
        let src_index = indices[start + i];
        let src_start = src_index * NUM_INPUTS;
        let dst_start = i * NUM_INPUTS;

        // Copy base image to output buffer
        out_inputs[dst_start..dst_start + NUM_INPUTS]
            .copy_from_slice(&images[src_start..src_start + NUM_INPUTS]);
        out_labels[i] = labels[src_index];

        // Apply augmentations if parameters are provided and RNG is available
        if let Some(ref mut rng_ref) = rng {
            let image_slice = &mut out_inputs[dst_start..dst_start + NUM_INPUTS];

            // Apply random crop if padding is specified
            if let Some(padding) = crop_padding {
                // random_crop returns a new Vec, so we need to copy it back
                let cropped = random_crop(
                    image_slice,
                    IMG_W,
                    IMG_H,
                    IMG_CHANNELS,
                    padding,
                    IMG_W, // crop back to original width
                    IMG_H, // crop back to original height
                    rng_ref,
                );
                image_slice.copy_from_slice(&cropped);
            }

            // Apply random horizontal flip if probability is specified
            if let Some(prob) = flip_prob {
                random_horizontal_flip(image_slice, IMG_W, IMG_H, IMG_CHANNELS, prob, rng_ref);
            }

            // Apply color jitter if specified
            if let Some(brightness_delta) = brightness_jitter {
                random_brightness(
                    image_slice,
                    IMG_W,
                    IMG_H,
                    IMG_CHANNELS,
                    brightness_delta,
                    rng_ref,
                );
            }
            if let Some(contrast_delta) = contrast_jitter {
                random_contrast(
                    image_slice,
                    IMG_W,
                    IMG_H,
                    IMG_CHANNELS,
                    contrast_delta,
                    rng_ref,
                );
            }
            // Note: saturation_jitter is skipped for MNIST (grayscale, 1 channel)
        }
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
    let file = File::create(&log_filename).unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    let mut loss_file = BufWriter::new(file);

    // Write CSV header
    writeln!(
        loss_file,
        "epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate"
    )
    .unwrap_or_else(|_| {
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

    // Early stopping state
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

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

        // Fisher-Yates shuffle.
        if train_data.num_samples > 1 {
            for i in (1..train_data.num_samples).rev() {
                let j = rng.gen_usize(i + 1);
                indices.swap(i, j);
            }
        }

        for batch_start in (0..train_data.num_samples).step_by(params.batch_size) {
            let batch_count = (train_data.num_samples - batch_start).min(params.batch_size);

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
                if params.enable_augmentation {
                    Some(aug_rng)
                } else {
                    None
                },
            );

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
                "Buffer size mismatch before softmax_rows"
            );
            softmax_rows(&mut a2[..a2_len], batch_count, NUM_OUTPUTS);

            // Output delta and loss.
            let batch_loss = compute_delta_and_loss(
                &a2[..a2_len],
                &batch_labels[..batch_count],
                batch_count,
                NUM_OUTPUTS,
                &mut dz2,
            );
            total_loss += batch_loss;

            // Backward: output layer.
            nn.output_layer.backward(&a1, &dz2, &mut dz1, batch_count);

            // Apply ReLU derivative to hidden layer gradient.
            let dz1_len = batch_count * NUM_HIDDEN;
            for i in 0..dz1_len {
                if a1[i] <= 0.0 {
                    dz1[i] = 0.0;
                }
            }

            // Backward: hidden layer.
            let grad_len = batch_count * NUM_INPUTS;
            nn.hidden_layer.backward(
                &batch_inputs,
                &dz1,
                &mut unused_grad[..grad_len],
                batch_count,
            );

            // Log gradient magnitudes before parameter update (accumulate for epoch)
            let (hidden_w_norm, hidden_b_norm) = nn.hidden_layer.get_gradient_magnitude();
            let (output_w_norm, output_b_norm) = nn.output_layer.get_gradient_magnitude();
            hidden_weight_grad_sum += hidden_w_norm;
            hidden_bias_grad_sum += hidden_b_norm;
            output_weight_grad_sum += output_w_norm;
            output_bias_grad_sum += output_b_norm;
            batch_count_total += 1;

            // Update parameters using optimizer.
            nn.output_layer.update_with_optimizer(optimizer.as_mut());
            nn.hidden_layer.update_with_optimizer(optimizer.as_mut());
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_loss = total_loss / train_data.num_samples as f32;

        // Evaluate on validation set
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

            // Compute loss
            let epsilon = 1e-9f32;
            for row_idx in 0..batch_count {
                let row_start = row_idx * NUM_OUTPUTS;
                let label = val_data.labels[batch_start + row_idx] as usize;
                let prob = val_a2[row_start + label].max(epsilon);
                val_total_loss -= prob.ln();

                // Compute accuracy
                let row = &val_a2[row_start..row_start + NUM_OUTPUTS];
                let mut predicted = 0usize;
                let mut max_prob = row[0];
                for (i, &value) in row.iter().enumerate().skip(1) {
                    if value > max_prob {
                        max_prob = value;
                        predicted = i;
                    }
                }
                if predicted == label {
                    val_correct += 1;
                }
            }
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
        writeln!(
            loss_file,
            "{},{},{},{},{},{}",
            epoch + 1,
            average_loss,
            duration,
            val_average_loss,
            val_accuracy,
            current_lr
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing training loss data.");
            process::exit(1);
        });

        // Early stopping check
        if val_average_loss < best_val_loss - params.early_stopping_min_delta {
            best_val_loss = val_average_loss;
            epochs_without_improvement = 0;
            // Save best model
            save_model(nn, "mnist_model_best.bin");
        } else {
            epochs_without_improvement += 1;
        }

        if epochs_without_improvement >= params.early_stopping_patience {
            println!(
                "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
                params.early_stopping_patience, best_val_loss
            );
            break;
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

fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Parse command-line arguments to get config file path.
/// Returns the path specified with --config flag, or default path if not provided.
fn parse_config_path(args: &[String]) -> String {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    DEFAULT_CONFIG_PATH.to_string()
}

/// Program entry point that loads MNIST data, constructs a learning-rate scheduler and neural network, trains and evaluates the model, and saves the trained parameters.
///
/// The function parses an optional CLI config path to select a learning-rate scheduler, measures and reports timings for data loading, training, and testing, and writes the final model to `mnist_model.bin`.
///
/// # Examples
///
/// ```
/// // Run the program entry point (typically executed by the runtime).
/// main();
/// ```
fn main() {
    let program_start = Instant::now();

    // Parse command-line arguments for config file path
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args);

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

    // Print loaded configuration
    println!("\nConfiguration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Validation split: {:.1}%", validation_split * 100.0);
    println!("  Early stopping patience: {}", early_stopping_patience);
    println!("  Early stopping min delta: {}", early_stopping_min_delta);
    println!("  Scheduler type: {}", config.scheduler_type);
    if let Some(ref activation) = config.activation_function {
        println!("  Activation function: {}", activation);
    }
    println!(
        "  Data augmentation: {}",
        if enable_augmentation {
            "enabled"
        } else {
            "disabled"
        }
    );
    if enable_augmentation {
        if let Some(prob) = horizontal_flip_prob {
            println!("    Horizontal flip probability: {}", prob);
        }
        if let Some(padding) = random_crop_padding {
            println!("    Random crop padding: {}", padding);
        }
        if let Some(delta) = brightness_jitter {
            println!("    Brightness jitter: {}", delta);
        }
        if let Some(delta) = contrast_jitter {
            println!("    Contrast jitter: {}", delta);
        }
    }
    println!();

    // Create learning rate scheduler
    let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(&config_path));

    println!("Loading training data...");
    let load_start = Instant::now();
    let mut train_images =
        read_mnist_images("./data/train-images.idx3-ubyte").unwrap_or_else(|e| {
            eprintln!("Could not read training images: {}", e);
            process::exit(1);
        });
    let mut train_labels =
        read_mnist_labels("./data/train-labels.idx1-ubyte").unwrap_or_else(|e| {
            eprintln!("Could not read training labels: {}", e);
            process::exit(1);
        });

    println!("Loading test data...");
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("Could not read test images: {}", e);
        process::exit(1);
    });
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("Could not read test labels: {}", e);
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
    let mut nn = initialize_network(&mut rng);

    // Augmentation RNG (reseeded from time for randomness)
    let mut aug_rng = SimpleRng::new(2);
    aug_rng.reseed_from_time();

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
    fn test_compute_delta_and_loss() {
        let outputs = vec![0.1, 0.2, 0.7, 0.3, 0.4, 0.3];
        let labels = vec![2, 1];
        let mut delta = vec![0.0; 6];

        let loss = compute_delta_and_loss(&outputs, &labels, 2, 3, &mut delta);

        assert!(loss > 0.0);
        assert!((delta[2] - (0.7 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gather_batch() {
        let images = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels = [0, 1];
        let indices = [1, 0];
        let mut out_inputs = [0.0; 6];
        let mut out_labels = [0; 2];

        const TEST_NUM_INPUTS: usize = 3;
        let input_stride = TEST_NUM_INPUTS;

        for i in 0..2 {
            let src_index = indices[i];
            let src_start = src_index * input_stride;
            let dst_start = i * input_stride;
            let src_slice = &images[src_start..src_start + input_stride];
            let dst_slice = &mut out_inputs[dst_start..dst_start + input_stride];
            dst_slice.copy_from_slice(src_slice);
            out_labels[i] = labels[src_index];
        }

        assert_eq!(out_labels[0], 1);
        assert_eq!(out_labels[1], 0);
        assert!((out_inputs[0] - 4.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_initialize_network() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(&mut rng);

        assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
        assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
    }
}
