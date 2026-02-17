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
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::config::load_config;
use rust_neural_networks::data::cifar10::{read_cifar10_batch, read_cifar10_batches};
pub use rust_neural_networks::layers::{
    batchnorm::BatchNormLayer, dropout::DropoutLayer, Conv2DLayer, DenseLayer, Layer,
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

// Copy a subset of images/labels into contiguous batch buffers with optional augmentation.
/// Copies a contiguous mini-batch of samples (inputs and labels) from the full dataset
/// into the provided output buffers according to the ordering in `indices`.
///
/// Copies `count` samples starting from `indices[start]` into `out_inputs` (flattened, row-major,
/// length = `count * NUM_INPUTS`) and `out_labels` (length = `count`).
///
/// If augmentation parameters are provided, applies data augmentation to each image after copying.
/// Augmentations are applied in the following order:
/// 1. Random crop (if `crop_padding` is Some)
/// 2. Random horizontal flip (if `flip_prob` is Some)
///
/// # Arguments
///
/// - `images`: flat image buffer where each image occupies `NUM_INPUTS` floats.
/// - `labels`: label buffer aligned with `images`.
/// - `indices`: permutation/index array selecting which samples to gather.
/// - `start`: index within `indices` of the first sample to copy.
/// - `count`: number of samples to copy.
/// - `out_inputs`: destination buffer for `count` images (flattened).
/// - `out_labels`: destination buffer for `count` labels.
/// - `flip_prob`: optional probability (0.0-1.0) for random horizontal flip.
/// - `crop_padding`: optional padding amount for random crop (crops back to IMG_W x IMG_H).
/// - `brightness_jitter`: optional brightness jitter delta (applied uniformly to RGB).
/// - `contrast_jitter`: optional contrast jitter delta.
/// - `saturation_jitter`: optional saturation jitter delta.
/// - `rng`: optional random number generator for augmentation operations.
///
/// # Examples
///
/// ```
/// // gather a batch of size 2 without augmentation
/// let mut out_inputs = vec![0f32; 2 * NUM_INPUTS];
/// let mut out_labels = vec![0u8; 2];
/// gather_batch(&images, &labels, &indices, 10, 2, &mut out_inputs, &mut out_labels,
///              None, None, None, None, None, None);
/// assert_eq!(out_labels[0], labels[indices[10]]);
///
/// // gather a batch with augmentation
/// let mut rng = SimpleRng::new(42);
/// gather_batch(&images, &labels, &indices, 10, 2, &mut out_inputs, &mut out_labels,
///              Some(0.5), Some(4), Some(0.2), Some(0.2), Some(0.2), Some(&mut rng));
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
    saturation_jitter: Option<f32>,
    mut rng: Option<&mut SimpleRng>,
) {
    use rust_neural_networks::data::augmentation::{
        random_brightness, random_contrast, random_crop, random_horizontal_flip, random_saturation,
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
            if let Some(saturation_delta) = saturation_jitter {
                random_saturation(
                    image_slice,
                    IMG_W,
                    IMG_H,
                    IMG_CHANNELS,
                    saturation_delta,
                    rng_ref,
                );
            }
        }
    }
}

// CNN with shared layer abstractions.
struct Cnn {
    layers: Vec<Box<dyn Layer>>,
}

/// Creates a CIFAR-10 CNN from architecture configuration.
///
/// Loads the architecture from the specified path (or default), builds the model layers,
/// and returns all layers in a vector.
///
/// # Arguments
///
/// * `rng` - Random number generator for weight initialization
/// * `arch_path` - Optional path to architecture config file (uses default if None)
///
/// # Returns
///
/// A `Cnn` configured according to the architecture specification.
///
/// # Panics
///
/// Panics if the architecture file cannot be loaded or is invalid.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let model = init_cnn(&mut rng, None); // Uses default architecture
/// // `model` is ready to use for forward/backward passes on CIFAR-10-shaped inputs.
/// ```
fn init_cnn(rng: &mut SimpleRng, arch_path: Option<&str>) -> Cnn {
    let architecture_path = arch_path.unwrap_or(DEFAULT_ARCHITECTURE_PATH);

    println!("Loading architecture from: {}", architecture_path);

    let arch_config = load_architecture(architecture_path).unwrap_or_else(|e| {
        eprintln!(
            "Error loading architecture from '{}': {}",
            architecture_path, e
        );
        eprintln!("Please ensure the architecture file exists and is valid JSON.");
        process::exit(1);
    });

    // Build model from architecture config
    let layers = build_model(&arch_config, rng).unwrap_or_else(|e| {
        eprintln!("Error building model from architecture: {}", e);
        process::exit(1);
    });

    // Print architecture info
    println!("\nArchitecture loaded successfully:");
    println!("  Total layers: {}", layers.len());
    for (i, layer) in layers.iter().enumerate() {
        println!(
            "  Layer {}: input_size={}, output_size={}, params={}",
            i + 1,
            layer.input_size(),
            layer.output_size(),
            layer.parameter_count()
        );
    }
    println!();

    Cnn { layers }
}

/// Set training mode for all layers that support it (BatchNorm, Dropout).
///
/// This function iterates through all layers in the model and uses downcasting
/// to identify BatchNorm and Dropout layers, setting their training mode accordingly.
///
/// # Arguments
///
/// * `model` - Mutable reference to the CNN model
/// * `training` - True for training mode, false for inference mode
///
/// # Examples
///
/// ```ignore
/// // Switch to inference mode for evaluation
/// set_training_mode(&mut model, false);
/// let test_acc = test_accuracy(&mut model, &test_images, &test_labels);
///
/// // Switch back to training mode
/// set_training_mode(&mut model, true);
/// ```
fn set_training_mode(model: &mut Cnn, training: bool) {
    for layer in model.layers.iter_mut() {
        let layer_ref: &mut dyn Layer = &mut **layer;
        let any_layer = layer_ref.as_any_mut();

        // Try to downcast to BatchNormLayer
        if let Some(bn_layer) = any_layer.downcast_mut::<BatchNormLayer>() {
            bn_layer.set_training(training);
        }
        // Try to downcast to DropoutLayer
        else if let Some(dropout_layer) = any_layer.downcast_mut::<DropoutLayer>() {
            dropout_layer.set_training(training);
        }
        // Other layer types (Conv2D, Dense) don't have training-dependent behavior
    }
}

/// Helper struct to store layer activations and metadata for forward/backward passes.
struct LayerActivations {
    data: Vec<Vec<f32>>, // Stores output of each layer
    is_conv: Vec<bool>,  // Tracks which layers are Conv2D (need ReLU)
}

impl LayerActivations {
    fn new(num_layers: usize) -> Self {
        Self {
            data: vec![Vec::new(); num_layers],
            is_conv: vec![false; num_layers],
        }
    }
}

/// Generic forward pass through all layers in the model.
///
/// Applies forward propagation through each layer in sequence, applying ReLU activation
/// after Conv2D layers.
///
/// # Arguments
///
/// * `model` - CNN containing all layers
/// * `batch_size` - Number of samples in the batch
/// * `input` - Input data (batch_size * input_features)
/// * `activations` - Storage for layer outputs (will be populated)
/// * `_temp_buffer` - Temporary buffer for intermediate computations (unused)
///
/// # Returns
///
/// Index of buffer containing final output (activations.data)
fn forward_pass(
    model: &mut Cnn,
    batch_size: usize,
    input: &[f32],
    activations: &mut LayerActivations,
    _temp_buffer: &mut Vec<f32>,
) -> usize {
    if model.layers.is_empty() {
        return 0;
    }

    // First layer: use input directly
    {
        let layer = &model.layers[0];
        let output_size = layer.output_size() * batch_size;
        activations.data[0].resize(output_size, 0.0);
        layer.forward(input, &mut activations.data[0], batch_size);

        // Detect if this is a Conv2D layer and apply ReLU
        activations.is_conv[0] = layer.as_any().downcast_ref::<Conv2DLayer>().is_some();
        if activations.is_conv[0] {
            relu_inplace(&mut activations.data[0]);
        }
    }

    // Subsequent layers: use previous layer's output
    for i in 1..model.layers.len() {
        let output_size = model.layers[i].output_size() * batch_size;
        activations.data[i].resize(output_size, 0.0);

        // Split activations to avoid borrow checker issues
        let (prev_data, curr_data) = activations.data.split_at_mut(i);
        let prev_output = &prev_data[i - 1];
        let curr_output = &mut curr_data[0];

        model.layers[i].forward(prev_output, curr_output, batch_size);

        // Detect if this is a Conv2D layer and apply ReLU
        activations.is_conv[i] = model.layers[i].as_any().downcast_ref::<Conv2DLayer>().is_some();
        if activations.is_conv[i] {
            relu_inplace(curr_output);
        }
    }

    model.layers.len() - 1
}

/// Generic backward pass through all layers in the model.
///
/// Applies backward propagation through each layer in reverse sequence, applying ReLU gradient
/// where needed.
///
/// # Arguments
///
/// * `model` - CNN containing all layers
/// * `batch_size` - Number of samples in the batch
/// * `input` - Original input to the model
/// * `activations` - Layer outputs from forward pass
/// * `initial_grad` - Gradient from loss function
/// * `grad_buffer1` - First working buffer for gradients
/// * `grad_buffer2` - Second working buffer for gradients
fn backward_pass(
    model: &mut Cnn,
    batch_size: usize,
    input: &[f32],
    activations: &LayerActivations,
    initial_grad: &[f32],
    grad_buffer1: &mut Vec<f32>,
    grad_buffer2: &mut Vec<f32>,
) {
    if model.layers.is_empty() {
        return;
    }

    let num_layers = model.layers.len();

    // Copy initial gradient to buffer1
    grad_buffer1.clear();
    grad_buffer1.extend_from_slice(initial_grad);

    // Process layers in reverse, ping-ponging between buffers
    for i in (0..num_layers).rev() {
        let layer = &model.layers[i];
        let input_size = layer.input_size() * batch_size;
        grad_buffer2.resize(input_size, 0.0);

        let layer_input = if i == 0 {
            input
        } else {
            &activations.data[i - 1]
        };

        layer.backward(layer_input, grad_buffer1, grad_buffer2, batch_size);

        // Apply ReLU gradient for Conv2D layers
        if activations.is_conv[i] {
            for j in 0..grad_buffer2.len().min(activations.data[i].len()) {
                if activations.data[i][j] <= 0.0 {
                    grad_buffer2[j] = 0.0;
                }
            }
        }

        // Swap buffers: copy buffer2 to buffer1 for next iteration
        std::mem::swap(grad_buffer1, grad_buffer2);
    }
}

// Softmax + cross-entropy: returns summed loss and writes delta = (probs - onehot) * scale.
/// Converts logits to probabilities, computes cross-entropy loss for each label, and writes
/// the softmax gradient (probabilities minus one-hot labels) scaled by `scale` into `delta`.
///
/// The `probs_inplace` buffer is overwritten with row-wise softmax probabilities for the
/// first `batch` rows (each row length is `NUM_CLASSES`). `delta` is populated with the
/// per-class gradients for each row. Returned value is the sum of cross-entropy losses
/// over the processed batch.
///
/// # Parameters
///
/// - `probs_inplace`: input logits which will be replaced with softmax probabilities for
///   the first `batch * NUM_CLASSES` elements.
/// - `labels`: slice of length at least `batch` containing class indices (0..NUM_CLASSES-1).
/// - `batch`: number of rows (examples) to process.
/// - `delta`: output buffer (length at least `batch * NUM_CLASSES`) which will receive the
///   gradient dL/dlogits = (probs - one_hot) * scale.
/// - `scale`: scalar multiplier applied to the computed gradients written into `delta`.
///
/// # Returns
///
/// Sum of cross-entropy losses across the processed `batch` examples.
///
/// # Examples
///
/// ```rust
/// let mut logits = [2.0f32, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let labels = [0u8];
/// let mut delta = [0.0f32; NUM_CLASSES];
/// let loss = softmax_xent_backward(&mut logits, &labels, 1, &mut delta, 1.0);
/// assert!(loss > 0.0);
/// // gradients for the single row should sum approximately to zero
/// let sum: f32 = delta.iter().sum();
/// assert!(sum.abs() < 1e-6);
/// ```
fn softmax_xent_backward(
    probs_inplace: &mut [f32], // logits overwritten with probs
    labels: &[u8],
    batch: usize,
    delta: &mut [f32],
    scale: f32,
) -> f32 {
    let eps = 1e-9f32;
    let len = batch * NUM_CLASSES;
    softmax_rows(&mut probs_inplace[..len], batch, NUM_CLASSES);

    let mut loss = 0.0f32;
    for (b, &label) in labels.iter().enumerate().take(batch) {
        let base = b * NUM_CLASSES;
        let y = label as usize;

        let p = probs_inplace[base + y].max(eps);
        loss += -p.ln();

        for j in 0..NUM_CLASSES {
            let mut d = probs_inplace[base + j];
            if j == y {
                d -= 1.0;
            }
            delta[base + j] = d * scale;
        }
    }
    loss
}

/// Computes the classification accuracy (percentage) of the CNN on a dataset.
///
/// Runs the model forward in batches, performs convolution+ReLU, 2x2 max-pooling,
/// and the final fully-connected forward pass, then compares the predicted class
/// (argmax of logits) to the provided labels to compute accuracy.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng);
/// // One batch of all-zero images and zero labels.
/// let images = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
/// let labels = vec![0u8; BATCH_SIZE];
/// let acc = test_accuracy(&mut model, &images, &labels);
/// assert!(acc >= 0.0 && acc <= 100.0);
/// ```
fn test_accuracy(model: &mut Cnn, images: &[f32], labels: &[u8]) -> f32 {
    let num_samples = labels.len();
    let mut correct = 0usize;

    // Set BatchNorm and Dropout layers to inference mode
    set_training_mode(model, false);

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];

    // Allocate activations storage for all layers
    let num_layers = model.layers.len();
    let mut activations = LayerActivations::new(num_layers);
    let mut temp_buffer = Vec::new();

    // Run forward passes in batches and compute argmax accuracy.
    for start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch = (num_samples - start).min(BATCH_SIZE);
        let len = batch * NUM_INPUTS;
        batch_inputs[..len].copy_from_slice(&images[start * NUM_INPUTS..start * NUM_INPUTS + len]);

        // Forward pass through all layers
        let output_idx = forward_pass(
            model,
            batch,
            &batch_inputs,
            &mut activations,
            &mut temp_buffer,
        );

        // Get logits from the last layer output
        let logits = &activations.data[output_idx];

        // Compute accuracy
        for b in 0..batch {
            let base = b * NUM_CLASSES;
            let mut best = logits[base];
            let mut arg = 0usize;
            for j in 1..NUM_CLASSES {
                let v = logits[base + j];
                if v > best {
                    best = v;
                    arg = j;
                }
            }
            if arg as u8 == labels[start + b] {
                correct += 1;
            }
        }
    }

    100.0 * (correct as f32) / (num_samples as f32)
}

// Save the CNN model in binary (little-endian i32 + f32).
/// Writes the model's parameters to a binary file.
///
/// Note: This is a placeholder implementation for multi-layer architectures.
/// Full serialization would require iterating through all layers and saving their parameters.
///
/// # Examples
///
/// ```
/// # use crate::{SimpleRng, init_cnn, save_model};
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng, None);
/// save_model(&model, "cifar10_cnn_model.bin");
/// ```
fn save_model(model: &Cnn, filename: &str) {
    use std::io::Write;

    let mut f = BufWriter::new(File::create(filename).expect("Failed to create model file"));

    // Write number of layers
    let num_layers = model.layers.len() as u32;
    f.write_all(&num_layers.to_le_bytes())
        .expect("Failed to write number of layers");

    // Iterate through each layer and save based on type
    for layer in &model.layers {
        let layer_ref: &dyn Layer = &**layer;
        let any_layer = layer_ref.as_any();

        // Try to downcast and save each layer type
        if let Some(dense_layer) = any_layer.downcast_ref::<DenseLayer>() {
            // Layer type ID: 0 = Dense
            f.write_all(&[0u8]).expect("Failed to write layer type");

            // Save dimensions
            let in_size = dense_layer.input_size() as u32;
            let out_size = dense_layer.output_size() as u32;
            f.write_all(&in_size.to_le_bytes())
                .expect("Failed to write input size");
            f.write_all(&out_size.to_le_bytes())
                .expect("Failed to write output size");

            // Save weights and biases
            for &w in dense_layer.weights() {
                f.write_all(&w.to_le_bytes())
                    .expect("Failed to write weight");
            }
            for &b in dense_layer.biases() {
                f.write_all(&b.to_le_bytes()).expect("Failed to write bias");
            }
        } else if let Some(conv_layer) = any_layer.downcast_ref::<Conv2DLayer>() {
            // Layer type ID: 1 = Conv2D
            f.write_all(&[1u8]).expect("Failed to write layer type");

            // Save configuration
            let in_channels = conv_layer.in_channels() as u32;
            let out_channels = conv_layer.out_channels() as u32;
            let kernel_size = conv_layer.kernel_size() as u32;
            let padding = conv_layer.padding() as i32;
            let stride = conv_layer.stride() as u32;
            let input_height = conv_layer.input_height() as u32;
            let input_width = conv_layer.input_width() as u32;

            f.write_all(&in_channels.to_le_bytes()).unwrap();
            f.write_all(&out_channels.to_le_bytes()).unwrap();
            f.write_all(&kernel_size.to_le_bytes()).unwrap();
            f.write_all(&padding.to_le_bytes()).unwrap();
            f.write_all(&stride.to_le_bytes()).unwrap();
            f.write_all(&input_height.to_le_bytes()).unwrap();
            f.write_all(&input_width.to_le_bytes()).unwrap();

            // Save weights and biases
            for &w in conv_layer.weights() {
                f.write_all(&w.to_le_bytes()).unwrap();
            }
            for &b in conv_layer.biases() {
                f.write_all(&b.to_le_bytes()).unwrap();
            }
        } else if let Some(bn_layer) = any_layer.downcast_ref::<BatchNormLayer>() {
            // Layer type ID: 2 = BatchNorm
            f.write_all(&[2u8]).expect("Failed to write layer type");

            // Save size
            let size = bn_layer.output_size() as u32;
            f.write_all(&size.to_le_bytes()).unwrap();

            // Save learnable parameters
            for &g in bn_layer.gamma() {
                f.write_all(&g.to_le_bytes()).unwrap();
            }
            for &b in bn_layer.beta() {
                f.write_all(&b.to_le_bytes()).unwrap();
            }

            // Save running statistics
            for &m in &bn_layer.running_mean() {
                f.write_all(&m.to_le_bytes()).unwrap();
            }
            for &v in &bn_layer.running_var() {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        } else if let Some(dropout_layer) = any_layer.downcast_ref::<DropoutLayer>() {
            // Layer type ID: 3 = Dropout
            f.write_all(&[3u8]).expect("Failed to write layer type");

            // Dropout has no trainable parameters, just save size and drop_rate for reconstruction
            let size = dropout_layer.output_size() as u32;
            f.write_all(&size.to_le_bytes()).unwrap();

            // Note: drop_rate would need a getter method to save, but it's a hyperparameter
            // not a learned parameter, so we can skip it for now
            // For full model persistence, we'd need to save drop_rate too
        } else {
            panic!("Unknown layer type encountered during serialization");
        }
    }

    println!("Model saved to: {}", filename);
}

fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Parse command-line arguments to get config file paths.
/// Returns a tuple of (training_config_path, architecture_config_path).
/// Supports --config for training config and --arch for architecture config.
fn parse_config_paths(args: &[String]) -> (String, Option<String>) {
    let mut training_config = DEFAULT_CONFIG_PATH.to_string();
    let mut arch_config: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: {} [OPTIONS]", args[0]);
            println!("\nOptions:");
            println!("  --config <path>   Path to training configuration file");
            println!("                    (default: {})", DEFAULT_CONFIG_PATH);
            println!("  --arch <path>     Path to architecture configuration file");
            println!(
                "                    (default: {})",
                DEFAULT_ARCHITECTURE_PATH
            );
            println!("  --help, -h        Show this help message");
            process::exit(0);
        } else if args[i] == "--config" && i + 1 < args.len() {
            training_config = args[i + 1].clone();
            i += 2;
        } else if args[i] == "--arch" && i + 1 < args.len() {
            arch_config = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }

    (training_config, arch_config)
}

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
    let (config_path, arch_path) = parse_config_paths(&args);

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
        if let Some(delta) = saturation_jitter {
            println!("    Saturation jitter: {}", delta);
        }
    }
    println!();

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
            eprintln!("Failed to load CIFAR-10 training data: {}", err);
            process::exit(1);
        });
    let (test_images, test_labels) =
        read_cifar10_batch("./data/cifar-10-batches-bin/test_batch.bin").unwrap_or_else(|err| {
            eprintln!("Failed to load CIFAR-10 test batch: {}", err);
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

    let mut model = init_cnn(&mut rng, arch_path.as_deref());

    // Create learning rate scheduler
    let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(&config_path));

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let log_file = File::create("./logs/training_loss_cifar10_cnn.txt").unwrap_or_else(|_| {
        eprintln!("Could not create logs/training_loss_cifar10_cnn.txt");
        process::exit(1);
    });
    let mut log = BufWriter::new(log_file);

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
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

    println!(
        "Training CIFAR-10 CNN: epochs={} batch={} lr={}",
        epochs, batch_size, learning_rate
    );

    for epoch in 0..epochs {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);
        let current_lr = scheduler.get_lr();

        // Set BatchNorm and Dropout to training mode
        set_training_mode(&mut model, true);

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(batch_size) {
            let batch = (train_n - batch_start).min(batch_size);
            let scale = 1.0f32;

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
            let batch_loss =
                softmax_xent_backward(&mut logits, &batch_labels, batch, &mut delta, scale);
            total_loss += batch_loss;

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

            // Update parameters for all layers
            for layer in &mut model.layers {
                layer.update_parameters(current_lr);
            }

            // Print progress every 100 batches
            let batch_idx = batch_start / batch_size;
            let total_batches = (train_n + batch_size - 1) / batch_size;
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

            // Apply softmax and compute loss
            softmax_rows(
                &mut val_logits[..batch_count * NUM_CLASSES],
                batch_count,
                NUM_CLASSES,
            );

            // Compute loss and accuracy
            let epsilon = 1e-9f32;
            for row_idx in 0..batch_count {
                let row_start = row_idx * NUM_CLASSES;
                let label = val_labels[batch_start + row_idx] as usize;
                let prob = val_logits[row_start + label].max(epsilon);
                val_total_loss -= prob.ln();

                // Compute accuracy
                let row = &val_logits[row_start..row_start + NUM_CLASSES];
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
        writeln!(
            log,
            "{},{},{},{},{},{}",
            epoch + 1,
            avg_loss,
            secs,
            val_average_loss,
            val_accuracy,
            current_lr
        )
        .ok();

        // Early stopping check
        if val_average_loss < best_val_loss - early_stopping_min_delta {
            best_val_loss = val_average_loss;
            epochs_without_improvement = 0;
            // Save best model
            save_model(&model, "cifar10_cnn_model_best.bin");
        } else {
            epochs_without_improvement += 1;
        }

        if epochs_without_improvement >= early_stopping_patience {
            println!(
                "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
                early_stopping_patience, best_val_loss
            );
            break;
        }

        // Update learning rate scheduler
        scheduler.step();
    }

    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_batch() {
        let images = vec![1.0; 3072 * 3]; // 3 images
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![0, 1, 2];
        let mut out_inputs = vec![0.0; 3072 * 2]; // batch of 2
        let mut out_labels = vec![0u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            None, // flip_prob
            None, // crop_padding
            None, // brightness_jitter
            None, // contrast_jitter
            None, // saturation_jitter
            None, // rng
        );

        assert_eq!(out_labels[0], 0);
        assert_eq!(out_labels[1], 1);
    }

    #[test]
    fn test_forward_pass_relu_only_after_conv2d() {
        // Build a small deep CNN: Conv2D -> Conv2D -> Conv2D -> Dense
        // Input: 1 channel, 4x4 images
        let mut rng = SimpleRng::new(42);

        // Layer 0: Conv2D (in_ch=1, out_ch=2, kernel=3, padding=1, stride=1, H=4, W=4) -> output 2*4*4=32
        let conv1 = Conv2DLayer::new(1, 2, 3, 1isize, 1, 4, 4, &mut rng);
        // Layer 1: Conv2D (in_ch=2, out_ch=4, kernel=3, padding=1, stride=1, H=4, W=4) -> output 4*4*4=64
        let conv2 = Conv2DLayer::new(2, 4, 3, 1isize, 1, 4, 4, &mut rng);
        // Layer 2: Conv2D (in_ch=4, out_ch=2, kernel=3, padding=1, stride=1, H=4, W=4) -> output 2*4*4=32
        let conv3 = Conv2DLayer::new(4, 2, 3, 1isize, 1, 4, 4, &mut rng);
        // Layer 3: Dense (32 -> 10) - no ReLU should be applied here
        let dense = DenseLayer::new(32, 10, &mut rng);

        let mut model = Cnn {
            layers: vec![
                Box::new(conv1),
                Box::new(conv2),
                Box::new(conv3),
                Box::new(dense),
            ],
        };

        let batch_size = 2;
        let input_size = 1 * 4 * 4; // 1 channel, 4x4 image
        let input = vec![1.0f32; batch_size * input_size];

        let num_layers = model.layers.len();
        let mut activations = LayerActivations::new(num_layers);
        let mut temp_buffer = Vec::new();

        let output_idx = forward_pass(
            &mut model,
            batch_size,
            &input,
            &mut activations,
            &mut temp_buffer,
        );

        // Output index should be the last layer
        assert_eq!(output_idx, 3, "Output should be from the last layer (index 3)");

        // Verify ReLU only applied after Conv2D layers (is_conv flag)
        assert!(
            activations.is_conv[0],
            "Layer 0 (Conv2D) should be marked as conv"
        );
        assert!(
            activations.is_conv[1],
            "Layer 1 (Conv2D) should be marked as conv"
        );
        assert!(
            activations.is_conv[2],
            "Layer 2 (Conv2D) should be marked as conv"
        );
        // Dense layer should NOT be marked as conv (no ReLU)
        assert!(
            !activations.is_conv[3],
            "Layer 3 (Dense) should NOT be marked as conv"
        );

        // All Conv2D layer activations must be non-negative (ReLU was applied)
        for &val in &activations.data[0] {
            assert!(
                val >= 0.0,
                "Conv2D layer 0 output must be >= 0 after ReLU, got {}",
                val
            );
        }
        for &val in &activations.data[1] {
            assert!(
                val >= 0.0,
                "Conv2D layer 1 output must be >= 0 after ReLU, got {}",
                val
            );
        }
        for &val in &activations.data[2] {
            assert!(
                val >= 0.0,
                "Conv2D layer 2 output must be >= 0 after ReLU, got {}",
                val
            );
        }

        // Dense layer output has the expected size (ReLU not applied, values unconstrained)
        assert_eq!(
            activations.data[3].len(),
            batch_size * 10,
            "Dense layer output should have batch_size * num_classes elements"
        );
    }
}
