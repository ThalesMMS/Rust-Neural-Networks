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
use rust_neural_networks::config::{load_config, TrainingConfig};
use rust_neural_networks::data::cifar10::{read_cifar10_batch, read_cifar10_batches};
pub use rust_neural_networks::layers::{
    batchnorm::BatchNormLayer, dropout::DropoutLayer, Conv2DLayer, DenseLayer, Layer,
};
use rust_neural_networks::optimizers::rmsprop::RMSprop;
use rust_neural_networks::optimizers::{Adam, AdamW, Optimizer, SGD};
use rust_neural_networks::training::{
    compute_softmax_cross_entropy, evaluate_batch_accuracy, gather_batch, parse_config_path,
    print_training_config, CsvGradientLogger, CsvTrainingLogger, EarlyStopping,
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
        activations.is_conv[i] = model.layers[i]
            .as_any()
            .downcast_ref::<Conv2DLayer>()
            .is_some();
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

            // Save drop_rate so load_model can reconstruct the layer correctly
            f.write_all(&dropout_layer.drop_rate().to_le_bytes())
                .unwrap();
        } else {
            panic!("Unknown layer type encountered during serialization");
        }
    }

    println!("Model saved to: {}", filename);
}

/// Loads a CNN model from a binary file previously saved by `save_model`.
///
/// Reads the layer count and per-layer parameters from the file, reconstructing
/// each layer with its saved weights and biases. The binary format mirrors
/// exactly what `save_model` writes (little-endian u32/i32/f32 values).
///
/// Layer type IDs:
/// - 0 = Dense
/// - 1 = Conv2D
/// - 2 = BatchNorm
/// - 3 = Dropout
///
/// # Arguments
///
/// * `filename` - Path to the binary model file
///
/// # Returns
///
/// A `Cnn` with all layers restored from the file.
///
/// # Panics
///
/// Panics if the file cannot be opened, read, or contains an unknown layer type.
///
/// # Examples
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let model = init_cnn(&mut rng, None);
/// save_model(&model, "cifar10_cnn_model_best.bin");
/// let loaded = load_model("cifar10_cnn_model_best.bin");
/// ```
fn load_model(filename: &str) -> Cnn {
    use std::io::Read;

    let mut f = File::open(filename).expect("Failed to open model file");

    // Read number of layers
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)
        .expect("Failed to read number of layers");
    let num_layers = u32::from_le_bytes(buf4) as usize;

    let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(num_layers);

    // RNG needed only to construct DropoutLayer (no trainable params, just mask).
    let mut rng = SimpleRng::new(42);

    for _ in 0..num_layers {
        // Read layer type ID
        let mut type_buf = [0u8; 1];
        f.read_exact(&mut type_buf)
            .expect("Failed to read layer type");
        let layer_type = type_buf[0];

        match layer_type {
            0 => {
                // Dense layer
                f.read_exact(&mut buf4).expect("Failed to read input size");
                let in_size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read output size");
                let out_size = u32::from_le_bytes(buf4) as usize;

                let weight_count = in_size * out_size;
                let mut weights = vec![0.0f32; weight_count];
                for w in weights.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read weight");
                    *w = f32::from_le_bytes(buf4);
                }

                let mut biases = vec![0.0f32; out_size];
                for b in biases.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read bias");
                    *b = f32::from_le_bytes(buf4);
                }

                layers.push(Box::new(DenseLayer::new_with_weights(
                    in_size, out_size, weights, biases,
                )));
            }
            1 => {
                // Conv2D layer
                f.read_exact(&mut buf4).expect("Failed to read in_channels");
                let in_channels = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4)
                    .expect("Failed to read out_channels");
                let out_channels = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read kernel_size");
                let kernel_size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read padding");
                let padding = i32::from_le_bytes(buf4) as isize;
                f.read_exact(&mut buf4).expect("Failed to read stride");
                let stride = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4)
                    .expect("Failed to read input_height");
                let input_height = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read input_width");
                let input_width = u32::from_le_bytes(buf4) as usize;

                let weight_count = out_channels * in_channels * kernel_size * kernel_size;
                let mut weights = vec![0.0f32; weight_count];
                for w in weights.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read weight");
                    *w = f32::from_le_bytes(buf4);
                }

                let mut biases = vec![0.0f32; out_channels];
                for b in biases.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read bias");
                    *b = f32::from_le_bytes(buf4);
                }

                layers.push(Box::new(Conv2DLayer::new_with_weights(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    stride,
                    input_height,
                    input_width,
                    weights,
                    biases,
                )));
            }
            2 => {
                // BatchNorm layer
                f.read_exact(&mut buf4).expect("Failed to read size");
                let size = u32::from_le_bytes(buf4) as usize;

                let mut gamma = vec![0.0f32; size];
                for g in gamma.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read gamma");
                    *g = f32::from_le_bytes(buf4);
                }

                let mut beta = vec![0.0f32; size];
                for b in beta.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read beta");
                    *b = f32::from_le_bytes(buf4);
                }

                let mut running_mean = vec![0.0f32; size];
                for m in running_mean.iter_mut() {
                    f.read_exact(&mut buf4)
                        .expect("Failed to read running_mean");
                    *m = f32::from_le_bytes(buf4);
                }

                let mut running_var = vec![0.0f32; size];
                for v in running_var.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read running_var");
                    *v = f32::from_le_bytes(buf4);
                }

                // Use standard epsilon and momentum defaults matching BatchNormLayer::new
                layers.push(Box::new(BatchNormLayer::new_with_params(
                    size,
                    1e-5,
                    0.1,
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                )));
            }
            3 => {
                // Dropout layer (no trainable parameters)
                f.read_exact(&mut buf4).expect("Failed to read size");
                let size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read drop_rate");
                let drop_rate = f32::from_le_bytes(buf4);

                layers.push(Box::new(DropoutLayer::new(size, drop_rate, &mut rng)));
            }
            _ => {
                panic!(
                    "Unknown layer type {} encountered during deserialization",
                    layer_type
                );
            }
        }
    }

    println!("Model loaded from: {}", filename);
    Cnn { layers }
}

fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Creates an optimizer based on the training configuration.
///
/// Reads `optimizer_type` from the config (defaulting to "adamw") and constructs
/// the appropriate optimizer with hyperparameters from the config or sensible defaults.
///
/// # Arguments
///
/// * `config` - Training configuration containing optimizer settings
/// * `lr` - Initial learning rate for the optimizer
///
/// # Returns
///
/// A boxed optimizer implementing the `Optimizer` trait.
///
/// # Supported optimizer types
///
/// - `"sgd"`: Stochastic Gradient Descent
/// - `"adam"`: Adam optimizer
/// - `"adamw"`: AdamW (Adam with decoupled weight decay)
/// - `"rmsprop"`: RMSprop optimizer
/// - Unknown types default to AdamW with a warning.
fn create_optimizer(config: &TrainingConfig, lr: f32) -> Box<dyn Optimizer> {
    let optimizer_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    let beta1 = config.adam_beta1.unwrap_or(0.9);
    let beta2 = config.adam_beta2.unwrap_or(0.999);
    let epsilon = config.adam_epsilon.unwrap_or(1e-8);
    let weight_decay = config.adamw_weight_decay.unwrap_or(0.01);

    match optimizer_type {
        "sgd" => Box::new(SGD::new(lr)),
        "adam" => Box::new(Adam::new(lr, beta1, beta2, epsilon)),
        "adamw" => Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay)),
        "rmsprop" => Box::new(RMSprop::new(lr, 0.9, epsilon)),
        _ => {
            eprintln!(
                "Unknown optimizer type: '{}', defaulting to AdamW",
                optimizer_type
            );
            Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay))
        }
    }
}

/// Parse command-line arguments to get the architecture config file path.
///
/// Scans `args` for `--arch <path>` and returns the path as `Some(path)`.
/// If the flag is absent, `None` is returned (caller should use the default).
/// Also handles `--help` / `-h` to print usage and exit.
fn parse_arch_path(args: &[String]) -> Option<String> {
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
        } else if args[i] == "--arch" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

/// Parses command-line arguments for both the config path and the optional arch path.
///
/// Returns `(config_path, arch_path)` where `config_path` defaults to
/// [`DEFAULT_CONFIG_PATH`] when `--config` is absent, and `arch_path` is
/// `Some(path)` when `--arch` is present, or `None` otherwise.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::parse_config_path;
///
/// let args = vec!["prog".to_string(), "--config".to_string(), "c.json".to_string()];
/// // parse_config_paths(&args) would return ("c.json".to_string(), None)
/// ```
fn parse_config_paths(args: &[String]) -> (String, Option<String>) {
    (
        parse_config_path(args, DEFAULT_CONFIG_PATH),
        parse_arch_path(args),
    )
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

    let mut model = init_cnn(&mut rng, arch_path.as_deref());

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
            for (layer_idx, layer) in model.layers.iter().enumerate() {
                let any_layer = layer.as_ref().as_any();
                if let Some(conv_layer) = any_layer.downcast_ref::<Conv2DLayer>() {
                    let (w_norm, b_norm) = conv_layer.get_gradient_magnitude();
                    layer_weight_grad_sums[layer_idx] += w_norm;
                    layer_bias_grad_sums[layer_idx] += b_norm;
                } else if let Some(dense_layer) = any_layer.downcast_ref::<DenseLayer>() {
                    let (w_norm, b_norm) = dense_layer.get_gradient_magnitude();
                    layer_weight_grad_sums[layer_idx] += w_norm;
                    layer_bias_grad_sums[layer_idx] += b_norm;
                }
            }
            batch_count_total += 1;

            // Update parameters for all layers using per-layer optimizers
            for (layer, opt) in model.layers.iter_mut().zip(layer_optimizers.iter_mut()) {
                layer.update_with_optimizer(opt.as_mut());
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
            IMG_W,        // img_width
            IMG_H,        // img_height
            IMG_CHANNELS, // img_channels
            None,         // flip_prob
            None,         // crop_padding
            None,         // brightness_jitter
            None,         // contrast_jitter
            None,         // saturation_jitter
            None,         // rng
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
        assert_eq!(
            output_idx, 3,
            "Output should be from the last layer (index 3)"
        );

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

    #[test]
    fn test_create_optimizer_from_config() {
        use rust_neural_networks::config::TrainingConfig;

        // Helper to build a minimal TrainingConfig with a given optimizer_type
        fn make_config(optimizer_type: Option<&str>) -> TrainingConfig {
            TrainingConfig {
                scheduler_type: "none".to_string(),
                step_size: None,
                gamma: None,
                decay_rate: None,
                min_lr: None,
                T_max: None,
                activation_function: None,
                leaky_relu_alpha: None,
                elu_alpha: None,
                optimizer_type: optimizer_type.map(|s| s.to_string()),
                adam_beta1: None,
                adam_beta2: None,
                adam_epsilon: None,
                adamw_weight_decay: None,
                rmsprop_decay: None,
                rmsprop_epsilon: None,
                learning_rate: None,
                epochs: None,
                batch_size: None,
                validation_split: None,
                early_stopping_patience: None,
                early_stopping_min_delta: None,
                enable_profiling: None,
                enable_augmentation: None,
                horizontal_flip_prob: None,
                random_crop_padding: None,
                brightness_jitter: None,
                contrast_jitter: None,
                saturation_jitter: None,
                noise_dim: None,
                g_lr: None,
                d_lr: None,
                label_smoothing: None,
            }
        }

        let lr = 0.01f32;

        // Test SGD optimizer
        let config = make_config(Some("sgd"));
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "SGD optimizer should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test Adam optimizer
        let config = make_config(Some("adam"));
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "Adam optimizer should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test AdamW optimizer
        let config = make_config(Some("adamw"));
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "AdamW optimizer should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test RMSprop optimizer
        let config = make_config(Some("rmsprop"));
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "RMSprop optimizer should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test None optimizer_type defaults to AdamW (learning_rate should be set correctly)
        let config = make_config(None);
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "Default (None) optimizer should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test unknown optimizer_type falls back to AdamW
        let config = make_config(Some("unknown_type"));
        let opt = create_optimizer(&config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "Unknown optimizer type should fall back to AdamW with learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test with non-default hyperparameters (Adam with custom beta1/beta2/epsilon)
        let mut adam_config = make_config(Some("adam"));
        adam_config.adam_beta1 = Some(0.95);
        adam_config.adam_beta2 = Some(0.998);
        adam_config.adam_epsilon = Some(1e-7);
        let opt = create_optimizer(&adam_config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "Adam optimizer with custom hyperparams should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );

        // Test with AdamW weight decay set
        let mut adamw_config = make_config(Some("adamw"));
        adamw_config.adamw_weight_decay = Some(0.001);
        let opt = create_optimizer(&adamw_config, lr);
        assert!(
            (opt.learning_rate() - lr).abs() < 1e-6,
            "AdamW optimizer with custom weight_decay should have learning_rate {}, got {}",
            lr,
            opt.learning_rate()
        );
    }
}
