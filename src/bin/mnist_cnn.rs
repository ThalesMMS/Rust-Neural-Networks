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

use std::any::Any;
use std::cell::RefCell;
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
    parse_step_flag, print_training_config, CsvTrainingLogger, EarlyStopping, EarlyStoppingAction,
    TrainingMetrics,
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

/// 2D Convolutional layer (Manual implementation, unique to this binary).
pub struct Conv2DLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: isize,
    stride: usize,
    input_height: usize,
    input_width: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}

impl Conv2DLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: isize,
        stride: usize,
        input_height: usize,
        input_width: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        assert!(stride > 0, "Stride must be greater than 0");

        let h_num = input_height as isize + 2 * padding - kernel_size as isize;
        let w_num = input_width as isize + 2 * padding - kernel_size as isize;
        if h_num < 0 || w_num < 0 {
            panic!("Invalid Conv2D configuration: output dimensions would be negative");
        }

        let fan_in = (in_channels * kernel_size * kernel_size) as f32;
        let fan_out = (out_channels * kernel_size * kernel_size) as f32;
        let limit = (6.0f32 / (fan_in + fan_out)).sqrt();

        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let mut weights = vec![0.0f32; weight_count];

        for value in &mut weights {
            *value = rng.gen_range_f32(-limit, limit);
        }

        Self {
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            input_height,
            input_width,
            weights,
            biases: vec![0.0f32; out_channels],
            grad_weights: RefCell::new(vec![0.0f32; weight_count]),
            grad_biases: RefCell::new(vec![0.0f32; out_channels]),
        }
    }

    pub fn output_height(&self) -> usize {
        ((self.input_height as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize
            + 1) as usize
    }

    pub fn output_width(&self) -> usize {
        ((self.input_width as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize
            + 1) as usize
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    /// Returns the L2 norm (magnitude) of weight and bias gradients.
    ///
    /// Useful for monitoring gradient flow during training and detecting vanishing/exploding gradients.
    pub fn get_gradient_magnitude(&self) -> (f32, f32) {
        // Compute L2 norm of weight gradients: sqrt(sum(g_i^2))
        let grad_weights = self.grad_weights.borrow();
        let weight_norm: f32 = grad_weights.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Compute L2 norm of bias gradients: sqrt(sum(g_i^2))
        let grad_biases = self.grad_biases.borrow();
        let bias_norm: f32 = grad_biases.iter().map(|g| g * g).sum::<f32>().sqrt();

        (weight_norm, bias_norm)
    }
}

impl Layer for Conv2DLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let out_h = self.output_height();
        let out_w = self.output_width();
        let out_spatial = out_h * out_w;
        let in_spatial = self.input_height * self.input_width;

        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let out_base_b = b * (self.out_channels * out_spatial);

            for oc in 0..self.out_channels {
                let bias = self.biases[oc];
                let out_base = out_base_b + oc * out_spatial;

                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = bias;

                        for ic in 0..self.in_channels {
                            let w_base =
                                (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;
                            let in_base_c = in_base + ic * in_spatial;

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize
                                        - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize
                                        - self.padding;

                                    if iy >= 0
                                        && iy < self.input_height as isize
                                        && ix >= 0
                                        && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base_c + iyy * self.input_width + ixx;
                                        let w_idx = w_base + ky * self.kernel_size + kx;
                                        sum += input[in_idx] * self.weights[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = out_base + oy * out_w + ox;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let scale = 1.0f32 / batch_size as f32;
        let out_h = self.output_height();
        let out_w = self.output_width();
        let out_spatial = out_h * out_w;
        let in_spatial = self.input_height * self.input_width;

        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        // Zero grad_input
        for v in grad_input.iter_mut() {
            *v = 0.0;
        }

        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let g_base_b = b * (self.out_channels * out_spatial);

            for oc in 0..self.out_channels {
                let g_base = g_base_b + oc * out_spatial;

                // Accumulate bias gradients
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let g = grad_output[g_base + oy * out_w + ox];
                        grad_b[oc] += g * scale;
                    }
                }

                // Accumulate weight gradients
                for ic in 0..self.in_channels {
                    let w_base = (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;
                    let in_base_c = in_base + ic * in_spatial;

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let g = grad_output[g_base + oy * out_w + ox];

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize
                                        - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize
                                        - self.padding;

                                    if iy >= 0
                                        && iy < self.input_height as isize
                                        && ix >= 0
                                        && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base_c + iyy * self.input_width + ixx;
                                        let w_idx = w_base + ky * self.kernel_size + kx;

                                        // Accumulate weight grad
                                        grad_w[w_idx] += g * input[in_idx] * scale;
                                        // Accumulate input grad
                                        grad_input[in_idx] += g * self.weights[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
            *w -= learning_rate * g;
        }
        for (b, g) in self.biases.iter_mut().zip(grad_b.iter()) {
            *b -= learning_rate * g;
        }

        for g in grad_w.iter_mut() {
            *g = 0.0;
        }
        for g in grad_b.iter_mut() {
            *g = 0.0;
        }
    }

    fn input_size(&self) -> usize {
        self.in_channels * self.input_height * self.input_width
    }

    fn output_size(&self) -> usize {
        self.out_channels * self.output_height() * self.output_width()
    }

    fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// CNN with shared layer abstractions.
struct Cnn {
    conv_layer: Conv2DLayer,
    fc_layer: DenseLayer,
}

/// Constructs a Cnn with initialized convolutional and fully connected layers.
///
/// The provided RNG is used to randomly initialize layer weights and biases.
///
/// # Parameters
///
/// - `rng`: mutable random number generator used to initialize layer parameters.
///
/// # Returns
///
/// A `Cnn` whose `conv_layer` and `fc_layer` have been allocated and randomized.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::SimpleRng;
///
/// // create an RNG and initialize the model
/// let mut rng = SimpleRng::new(1234);
/// let model = init_cnn(&mut rng);
/// // model is ready for use (forward/backward passes)
/// ```
fn init_cnn(rng: &mut SimpleRng) -> Cnn {
    // Conv: 1 input channel -> CONV_OUT output channels, 3x3 kernel, pad=1, stride=1
    let conv_layer = Conv2DLayer::new(1, CONV_OUT, KERNEL, PAD, 1, IMG_H, IMG_W, rng);

    // FC layer: FC_IN -> NUM_CLASSES
    let fc_layer = DenseLayer::new(FC_IN, NUM_CLASSES, rng);

    Cnn {
        conv_layer,
        fc_layer,
    }
}

// Forward conv + ReLU.
// input: [batch * 784], conv_out: [batch * CONV_OUT * 28 * 28]
/// Runs the convolutional layer forward for a batch and applies ReLU to the outputs in place.
///
/// conv_out is written with the layer's activated outputs for the specified batch index.
///
/// # Parameters
///
/// - `model`: CNN containing the convolutional layer to run.
/// - `batch`: index of the batch within the input buffer to process.
/// - `input`: flattened input batch slice (contains all batch elements).
/// - `conv_out`: writable slice where the convolutional activations for the batch will be stored; values are clamped to be >= 0 by ReLU.
///
/// # Examples
///
/// ```
/// use crate::{init_cnn, conv_forward_relu, SimpleRng, Cnn};
///
/// let mut rng = SimpleRng::new(123);
/// let mut model = init_cnn(&mut rng);
/// let batch_size = 32;
/// let batch_index = 0usize;
/// // Dimensions: BATCH_SIZE=32, NUM_INPUTS=784, CONV_OUT=8, IMG_H=28, IMG_W=28
/// let num_inputs = 784;
/// let conv_out = 8;
/// let img_h = 28;
/// let img_w = 28;
///
/// let mut inputs = vec![0f32; batch_size * num_inputs];
/// let mut conv_out_buf = vec![0f32; batch_size * conv_out * img_h * img_w];
/// // populate inputs[batch_index * NUM_INPUTS .. (batch_index+1) * NUM_INPUTS] as needed
/// conv_forward_relu(&mut model, batch_size, &inputs, &mut conv_out_buf);
/// // after call, conv_out values for the batch are non-negative due to ReLU
/// let start = batch_index * conv_out * img_h * img_w;
/// assert!(conv_out_buf[start..start + conv_out * img_h * img_w].iter().all(|&v| v >= 0.0));
/// ```
fn conv_forward_relu(model: &mut Cnn, batch_size: usize, input: &[f32], conv_out: &mut [f32]) {
    // Use Conv2DLayer for forward pass
    model.conv_layer.forward(input, conv_out, batch_size);

    // Apply ReLU activation
    relu_inplace(conv_out);
}

// MaxPool 2x2 stride 2.
// conv_act: [batch * C * 28 * 28] (post-ReLU)
// pool_out: [batch * C * 14 * 14]
// pool_idx: [batch * C * 14 * 14], stores argmax 0..3 (dy*2+dx)
fn maxpool_forward(batch: usize, conv_act: &[f32], pool_out: &mut [f32], pool_idx: &mut [u8]) {
    let conv_spatial = IMG_H * IMG_W;
    let pool_spatial = POOL_H * POOL_W;

    for b in 0..batch {
        let conv_base_b = b * (CONV_OUT * conv_spatial);
        let pool_base_b = b * (CONV_OUT * pool_spatial);

        for c in 0..CONV_OUT {
            let conv_base = conv_base_b + c * conv_spatial;
            let pool_base = pool_base_b + c * pool_spatial;

            for py in 0..POOL_H {
                for px in 0..POOL_W {
                    let iy0 = py * POOL;
                    let ix0 = px * POOL;

                    // Track argmax to route gradients during backprop.
                    let mut best = -f32::INFINITY;
                    let mut best_idx = 0u8;

                    for dy in 0..POOL {
                        for dx in 0..POOL {
                            let iy = iy0 + dy;
                            let ix = ix0 + dx;
                            let v = conv_act[conv_base + iy * IMG_W + ix];
                            let idx = (dy * POOL + dx) as u8; // 0..3
                            if v > best {
                                best = v;
                                best_idx = idx;
                            }
                        }
                    }

                    let out_i = pool_base + py * POOL_W + px;
                    pool_out[out_i] = best;
                    pool_idx[out_i] = best_idx;
                }
            }
        }
    }
}

// FC forward: logits = X*W + b.
// X: [batch * FC_IN], logits: [batch * 10]
/// Runs the fully connected (dense) layer forward for a batch, writing per-sample class logits.
///
/// `x` must contain `batch * FC_IN` contiguous input features (row-major per sample).
/// `logits` must be sized to hold `batch * NUM_CLASSES` output values and will be overwritten.
///
/// # Examples
///
/// ```
/// use crate::{init_cnn, fc_forward, SimpleRng, FC_IN, NUM_CLASSES};
///
/// let mut rng = SimpleRng::new(123);
/// let mut model = init_cnn(&mut rng);
/// let batch = 1usize;
/// let x = vec![0.0f32; batch * FC_IN];
/// let mut logits = vec![0.0f32; batch * NUM_CLASSES];
/// fc_forward(&mut model, batch, &x, &mut logits);
/// assert_eq!(logits.len(), batch * NUM_CLASSES);
/// ```
fn fc_forward(model: &mut Cnn, batch: usize, x: &[f32], logits: &mut [f32]) {
    // Use DenseLayer for forward pass
    model.fc_layer.forward(x, logits, batch);
}

// FC backward: compute gradW, gradB and dX.
/// Performs the backward pass for the fully connected (dense) layer, accumulating parameter gradients in the model
/// and writing the input-space gradients for the batch.
///
/// - `batch` is the number of examples in the current minibatch.
/// - `x` is the input feature buffer to the dense layer with length `batch * FC_IN`.
/// - `delta` is the gradient w.r.t. the dense layer outputs with length `batch * NUM_CLASSES`.
/// - `d_x` is the output buffer that will receive the gradient w.r.t. `x` (length `batch * FC_IN`).
///
/// # Examples
///
/// ```no_run
/// // Prepare model, batch size and buffers (sizes are illustrative)
/// let mut model = init_cnn(&mut SimpleRng::new(123));
/// let batch = 2;
/// let mut x = vec![0f32; batch * FC_IN];
/// let delta = vec![0f32; batch * NUM_CLASSES];
/// let mut d_x = vec![0f32; batch * FC_IN];
///
/// // Compute backward pass for the dense layer
/// fc_backward(&mut model, batch, &x, &delta, &mut d_x);
/// ```
fn fc_backward(
    model: &mut Cnn,
    batch: usize,
    x: &[f32],
    delta: &[f32],   // [batch*10]
    d_x: &mut [f32], // [batch*FC_IN]
) {
    // Use DenseLayer for backward pass (gradients are accumulated internally)
    model.fc_layer.backward(x, delta, d_x, batch);
}

// MaxPool backward: scatter grads to argmax positions, then apply ReLU mask.
fn maxpool_backward_relu(
    batch: usize,
    conv_act: &[f32],  // post-ReLU
    pool_grad: &[f32], // [batch*C*14*14]
    pool_idx: &[u8],
    conv_grad: &mut [f32], // [batch*C*28*28]
) {
    let conv_spatial = IMG_H * IMG_W;
    let pool_spatial = POOL_H * POOL_W;

    // Zero conv_grad so we can scatter-add into it.
    let used = batch * CONV_OUT * conv_spatial;
    for value in conv_grad.iter_mut().take(used) {
        *value = 0.0;
    }

    for b in 0..batch {
        let conv_base_b = b * (CONV_OUT * conv_spatial);
        let pool_base_b = b * (CONV_OUT * pool_spatial);

        for c in 0..CONV_OUT {
            let conv_base = conv_base_b + c * conv_spatial;
            let pool_base = pool_base_b + c * pool_spatial;

            for py in 0..POOL_H {
                for px in 0..POOL_W {
                    let p_i = pool_base + py * POOL_W + px;
                    let g = pool_grad[p_i];
                    let a = pool_idx[p_i] as usize; // 0..3
                    let dy = a / POOL;
                    let dx = a % POOL;

                    let iy = py * POOL + dy;
                    let ix = px * POOL + dx;

                    let c_i = conv_base + iy * IMG_W + ix;
                    conv_grad[c_i] += g;
                }
            }
        }
    }

    // ReLU backward: zero gradients where activation was <= 0.
    for i in 0..used {
        if conv_act[i] <= 0.0 {
            conv_grad[i] = 0.0;
        }
    }
}

// Conv backward: gradW and gradB (no dInput since this is the first layer).
/// Backpropagates gradients through the convolutional layer and accumulates its parameter gradients.
///
/// The function invokes the convolution layer's backward pass using the provided per-example
/// gradients with respect to the convolution pre-activations. Gradients for layer parameters
/// (kernels and biases) are accumulated inside the layer instance. The `grad_input` buffer is
/// accepted for API compatibility but is unused when this layer is the network's first layer.
///
/// # Parameters
///
/// - `model`: mutable reference to the CNN containing the convolution layer.
/// - `batch`: number of examples in the current mini-batch.
/// - `input`: flattened input batch with length `batch * NUM_INPUTS`.
/// - `conv_grad`: gradients w.r.t. convolution pre-activations, layout `batch * CONV_OUT * IMG_H * IMG_W`.
/// - `_grad_input`: destination buffer for gradients w.r.t. this layer's inputs; unused for the first layer.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let mut model = init_cnn(&mut rng);
/// let batch = 1;
/// let input = vec![0.0f32; batch * NUM_INPUTS];
/// let conv_grad = vec![0.0f32; batch * CONV_OUT * IMG_H * IMG_W];
/// let mut grad_input = vec![0.0f32; batch * NUM_INPUTS];
/// conv_backward(&mut model, batch, &input, &conv_grad, &mut grad_input);
/// ```
fn conv_backward(
    model: &mut Cnn,
    batch: usize,
    input: &[f32],           // [batch*784]
    conv_grad: &[f32],       // [batch*C*28*28]
    _grad_input: &mut [f32], // unused (first layer)
) {
    // Use Conv2DLayer for backward pass (gradients are accumulated internally)
    // Note: grad_input is unused since this is the first layer
    model
        .conv_layer
        .backward(input, conv_grad, _grad_input, batch);
}

/// Compute the model's classification accuracy as a percentage on the given dataset.
///
/// Processes the dataset in batches and runs the model's forward pass to produce predictions.
///
/// # Returns
///
/// The accuracy as a percentage between `0.0` and `100.0`.
///
/// # Examples
///
/// ```
/// // `model`, `images`, and `labels` are prepared elsewhere:
/// // let mut model = init_cnn(&mut rng);
/// // let images: Vec<f32> = ...; // length = num_samples * NUM_INPUTS
/// // let labels: Vec<u8> = ...;  // length = num_samples
/// let acc = test_accuracy(&mut model, &images, &labels);
/// assert!(acc >= 0.0 && acc <= 100.0);
/// ```
fn test_accuracy(model: &mut Cnn, images: &[f32], labels: &[u8]) -> f32 {
    let num_samples = labels.len();
    let mut correct = 0usize;

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut conv_out = vec![0.0f32; BATCH_SIZE * CONV_OUT * IMG_H * IMG_W];
    let mut pool_out = vec![0.0f32; BATCH_SIZE * FC_IN];
    let mut pool_idx = vec![0u8; BATCH_SIZE * CONV_OUT * POOL_H * POOL_W];
    let mut logits = vec![0.0f32; BATCH_SIZE * NUM_CLASSES];

    // Run forward passes in batches and compute argmax accuracy.
    for start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch = (num_samples - start).min(BATCH_SIZE);
        let len = batch * NUM_INPUTS;
        batch_inputs[..len].copy_from_slice(&images[start * NUM_INPUTS..start * NUM_INPUTS + len]);

        conv_forward_relu(model, batch, &batch_inputs, &mut conv_out);
        maxpool_forward(batch, &conv_out, &mut pool_out, &mut pool_idx);
        fc_forward(model, batch, &pool_out, &mut logits);

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
/// Serializes the CNN model to a binary file using little-endian encoding.
///
/// The file contains, in order:
/// 1. Conv layer metadata: out_channels, in_channels, kernel_size, input_height, input_width (as i32)
/// 2. All conv layer weights as 32-bit floats.
/// 3. All conv layer biases as 32-bit floats.
/// 4. FC layer metadata: input_size, output_size (as i32)
/// 5. All FC layer weights as 32-bit floats.
/// 6. All FC layer biases as 32-bit floats.
///
/// The function terminates the process with an error message if the file cannot be created or any write fails.
///
/// # Examples
///
/// ```
/// // Serializes `model` to "mnist_cnn_model_best.bin".
/// save_model(&model, "mnist_cnn_model_best.bin");
/// ```
fn save_model(model: &Cnn, filename: &str) {
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

    // Write conv layer metadata
    write_i32(&mut writer, model.conv_layer.out_channels as i32);
    write_i32(&mut writer, model.conv_layer.in_channels as i32);
    write_i32(&mut writer, model.conv_layer.kernel_size as i32);
    write_i32(&mut writer, model.conv_layer.input_height as i32);
    write_i32(&mut writer, model.conv_layer.input_width as i32);

    // Write conv layer weights and biases
    for &value in model.conv_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in model.conv_layer.biases() {
        write_f32(&mut writer, value);
    }

    // Write FC layer metadata
    write_i32(&mut writer, model.fc_layer.input_size() as i32);
    write_i32(&mut writer, model.fc_layer.output_size() as i32);

    // Write FC layer weights and biases
    for &value in model.fc_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in model.fc_layer.biases() {
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

/// Entry point for training and evaluating a minimal CNN on the MNIST dataset.
///
/// This program loads MNIST IDX files from ./data, trains a small convolutional
/// neural network on the training set while logging epoch loss to ./logs/training_loss_cnn.txt,
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
    let actual_train_samples = total_train_samples - validation_samples;

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

    let mut rng = SimpleRng::new(1);
    rng.reseed_from_time();

    // Augmentation RNG for use with gather_batch
    let mut aug_rng = SimpleRng::new(2);
    aug_rng.reseed_from_time();

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
            debugger.before_forward("fc_layer", &pool_out, batch, FC_IN, NUM_CLASSES, fc_param_count);
            fc_forward(&mut model, batch, &pool_out, &mut logits);
            debugger.after_forward("fc_layer", &logits, batch, NUM_CLASSES);

            // Softmax + loss + gradient at logits.
            softmax_rows(&mut logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);
            debugger.after_activation("Softmax", &logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);

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

    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}
