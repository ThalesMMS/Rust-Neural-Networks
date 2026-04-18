use super::*;

pub(crate) struct Cnn {
    pub(crate) layers: Vec<Box<dyn Layer>>,
}

/// Create a `Cnn` from an architecture configuration file.
///
/// Loads the architecture from `arch_path` or the default path, builds the model
/// layers using `rng` for initialization, prints per-layer metadata, and returns
/// the assembled `Cnn`.
///
/// # Arguments
///
/// * `rng` - Random number generator used for weight initialization.
/// * `arch_path` - Optional path to an architecture JSON file; the default path is used when `None`.
///
/// # Returns
///
/// A `Cnn` configured according to the loaded architecture.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let model = init_cnn(&mut rng, None).unwrap(); // uses default architecture file
/// assert!(!model.layers.is_empty());
/// ```
pub(crate) fn init_cnn(
    rng: &mut SimpleRng,
    arch_path: Option<&str>,
) -> Result<Cnn, Box<dyn std::error::Error>> {
    let architecture_path = arch_path.unwrap_or(DEFAULT_ARCHITECTURE_PATH);

    println!("Loading architecture from: {}", architecture_path);

    let arch_config = load_architecture(architecture_path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Error loading architecture from '{}': {}. Please ensure the architecture file exists and is valid JSON.",
                architecture_path, e
            ),
        )
    })?;

    // Build model from architecture config
    let layers = build_model(&arch_config, rng).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Error building model from architecture '{}': {}",
                architecture_path, e
            ),
        )
    })?;

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

    Ok(Cnn { layers })
}

/// Toggle training behavior for layers that support it.
///
/// Sets training (true) or inference (false) mode on model layers that have training-dependent behavior (for example, batch normalization and dropout).
///
/// # Examples
///
/// ```rust
/// // Disable training-dependent behavior for evaluation
/// set_training_mode(&mut model, false);
///
/// // Re-enable training behavior for further training
/// set_training_mode(&mut model, true);
/// ```
pub(crate) fn set_training_mode(model: &mut Cnn, training: bool) {
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
        // Try to downcast to ResidualBlock
        else if let Some(residual_block) = any_layer.downcast_mut::<ResidualBlock>() {
            residual_block.set_training(training);
        }
        // Other layer types (Conv2D, Dense) don't have training-dependent behavior
    }
}

fn capture_training_modes(model: &Cnn) -> Vec<Option<bool>> {
    model
        .layers
        .iter()
        .map(|layer| {
            let any_layer = layer.as_any();
            if let Some(bn_layer) = any_layer.downcast_ref::<BatchNormLayer>() {
                Some(bn_layer.is_training())
            } else if let Some(dropout_layer) = any_layer.downcast_ref::<DropoutLayer>() {
                Some(dropout_layer.is_training())
            } else {
                any_layer
                    .downcast_ref::<ResidualBlock>()
                    .map(|residual_block| residual_block.is_training())
            }
        })
        .collect()
}

fn restore_training_modes(model: &mut Cnn, modes: &[Option<bool>]) {
    for (layer, mode) in model.layers.iter_mut().zip(modes.iter()) {
        let Some(training) = mode else {
            continue;
        };
        let any_layer = layer.as_any_mut();
        if let Some(bn_layer) = any_layer.downcast_mut::<BatchNormLayer>() {
            bn_layer.set_training(*training);
        } else if let Some(dropout_layer) = any_layer.downcast_mut::<DropoutLayer>() {
            dropout_layer.set_training(*training);
        } else if let Some(residual_block) = any_layer.downcast_mut::<ResidualBlock>() {
            residual_block.set_training(*training);
        }
    }
}

/// Helper struct to store layer activations and metadata for forward/backward passes.
pub(crate) struct LayerActivations {
    pub(crate) data: Vec<Vec<f32>>, // Stores output of each layer
    pub(crate) is_conv: Vec<bool>,  // Tracks which layers are Conv2D (need ReLU)
}

impl LayerActivations {
    /// Creates a LayerActivations container for a model with the given number of layers.
    ///
    /// Each entry in `data` is an empty vector reserved for that layer's forward outputs,
    /// and each entry in `is_conv` is initialized to `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// let act = LayerActivations::new(3);
    /// assert_eq!(act.data.len(), 3);
    /// assert_eq!(act.is_conv.len(), 3);
    /// assert!(act.data.iter().all(|v| v.is_empty()));
    /// assert!(act.is_conv.iter().all(|&b| b == false));
    /// ```
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            data: vec![Vec::new(); num_layers],
            is_conv: vec![false; num_layers],
        }
    }
}

/// Run a forward pass through the model, filling per-layer activation buffers.
///
/// Populates `activations.data` with each layer's output for the given `batch_size`,
/// and marks `activations.is_conv` for layers that are convolutional. ReLU is applied
/// in-place to outputs of convolutional layers.
///
/// Returns the index of the activation buffer that contains the final output.
///
/// # Examples
///
/// ```
/// let mut model = Cnn { layers: vec![] };
/// let mut activations = LayerActivations::new(0);
/// let mut temp = Vec::new();
/// let idx = forward_pass(&mut model, 1, &[], &mut activations, &mut temp);
/// assert_eq!(idx, 0);
/// ```
pub(crate) fn forward_pass(
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

/// Performs backpropagation through the model layers in reverse order.
///
/// Uses the provided forward `activations` and `initial_grad` (loss gradient at the model
/// output) to propagate gradients back to the model input. When a layer was identified
/// as a convolutional layer during the forward pass, its ReLU gradient is applied by
/// zeroing gradients where the corresponding forward activation was <= 0.0.
/// `grad_buffer1` and `grad_buffer2` are used as working buffers and will be swapped
/// between iterations; both are mutated by this function.
///
/// # Parameters
///
/// * `activations` - Forward outputs and per-layer `is_conv` flags produced by `forward_pass`.
/// * `initial_grad` - Gradient with respect to the model's final output (loss gradient).
/// * `grad_buffer1` - Working gradient buffer that initially receives `initial_grad` and is
///   used/updated across layers.
/// * `grad_buffer2` - Secondary working buffer used for computing the next-layer gradients.
///
/// # Examples
///
/// ```
/// # use crate::cifar10_cnn::model::{Cnn, LayerActivations, backward_pass};
/// // No-op example: empty model performs no work and leaves buffers empty.
/// let mut model = Cnn { layers: vec![] };
/// let mut buf1: Vec<f32> = Vec::new();
/// let mut buf2: Vec<f32> = Vec::new();
/// backward_pass(&mut model, 1, &[], &LayerActivations::new(0), &[], &mut buf1, &mut buf2);
/// assert!(buf1.is_empty());
/// ```
pub(crate) fn backward_pass(
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
        grad_buffer2.fill(0.0);

        let layer_input = if i == 0 {
            input
        } else {
            &activations.data[i - 1]
        };

        if activations.is_conv[i] {
            for j in 0..grad_buffer1.len().min(activations.data[i].len()) {
                if activations.data[i][j] <= 0.0 {
                    grad_buffer1[j] = 0.0;
                }
            }
        }

        layer.backward(layer_input, grad_buffer1, grad_buffer2, batch_size);

        // Swap buffers: copy buffer2 to buffer1 for next iteration
        std::mem::swap(grad_buffer1, grad_buffer2);
    }
}

/// Compute classification accuracy of the model over the provided images and labels.
///
/// The model is placed into inference mode (disabling training behavior for BatchNorm and Dropout)
/// and evaluated in batches. Predictions are obtained by taking the argmax of the final-layer logits.
///
/// # Returns
///
/// Accuracy as a percentage between 0.0 and 100.0 representing (correct / total) * 100.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng, None).unwrap();
/// // One batch of all-zero images and zero labels.
/// let images = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
/// let labels = vec![0u8; BATCH_SIZE];
/// let acc = test_accuracy(&mut model, &images, &labels);
/// assert!(acc >= 0.0 && acc <= 100.0);
/// ```
pub(crate) fn test_accuracy(model: &mut Cnn, images: &[f32], labels: &[u8]) -> f32 {
    let num_samples = labels.len();
    // Invalid or empty inputs are treated as zero accuracy.
    if num_samples == 0 || images.len() != num_samples * NUM_INPUTS {
        return 0.0;
    }

    let mut correct = 0usize;

    // Set BatchNorm and Dropout layers to inference mode
    let previous_training_modes = capture_training_modes(model);
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

    let accuracy = 100.0 * (correct as f32) / (num_samples as f32);
    restore_training_modes(model, &previous_training_modes);
    accuracy
}

// Save the CNN model in binary (little-endian i32 + f32).
