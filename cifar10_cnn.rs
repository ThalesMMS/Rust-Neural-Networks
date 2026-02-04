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

use rust_neural_networks::data::cifar10::{read_cifar10_batch, read_cifar10_batches};
pub use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
pub use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::{create_scheduler_from_config, LRScheduler};
pub use rust_neural_networks::utils::rng::SimpleRng;

// CIFAR-10 constants (images are 32x32 RGB in pixel-interleaved format).
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_CHANNELS: usize = 3; // RGB
const NUM_INPUTS: usize = IMG_H * IMG_W * IMG_CHANNELS; // 3072
const NUM_CLASSES: usize = 10;

// CNN topology: 3x32x32 -> conv -> ReLU -> 2x2 maxpool -> FC(10).
const CONV_OUT: usize = 16; // More filters for RGB
const KERNEL: usize = 3;
const PAD: isize = 1;
const POOL: usize = 2;

const POOL_H: usize = IMG_H / POOL; // 16
const POOL_W: usize = IMG_W / POOL; // 16
const FC_IN: usize = CONV_OUT * POOL_H * POOL_W; // 16*16*16 = 4096

// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 10; // CIFAR-10 needs more epochs than MNIST
const BATCH_SIZE: usize = 32;
const VALIDATION_SPLIT: f32 = 0.1; // 10% of training data for validation
const EARLY_STOPPING_PATIENCE: usize = 3; // Number of epochs without improvement before stopping
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001; // Minimum change to be considered an improvement

// Main Logic
// ============================================================================

// Copy a subset of images/labels into contiguous batch buffers.
/// Copies a contiguous mini-batch of samples (inputs and labels) from the full dataset
/// into the provided output buffers according to the ordering in `indices`.
///
/// Copies `count` samples starting from `indices[start]` into `out_inputs` (flattened, row-major,
/// length = `count * NUM_INPUTS`) and `out_labels` (length = `count`).
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
///
/// # Examples
///
/// ```
/// // gather a batch of size 2
/// let mut out_inputs = vec![0f32; 2 * NUM_INPUTS];
/// let mut out_labels = vec![0u8; 2];
/// gather_batch(&images, &labels, &indices, 10, 2, &mut out_inputs, &mut out_labels);
/// assert_eq!(out_labels[0], labels[indices[10]]);
/// ```
fn gather_batch(
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    start: usize,
    count: usize,
    out_inputs: &mut [f32],
    out_labels: &mut [u8],
) {
    for i in 0..count {
        let src_index = indices[start + i];
        let src_start = src_index * NUM_INPUTS;
        let dst_start = i * NUM_INPUTS;
        out_inputs[dst_start..dst_start + NUM_INPUTS]
            .copy_from_slice(&images[src_start..src_start + NUM_INPUTS]);
        out_labels[i] = labels[src_index];
    }
}

// CNN with shared layer abstractions.
struct Cnn {
    conv_layer: Conv2DLayer,
    fc_layer: DenseLayer,
}

/// Creates a small CIFAR-10 CNN: one Conv2D layer followed by a fully connected layer.
///
/// The provided RNG is used to initialize all layer weights and biases deterministically.
///
/// # Returns
///
/// A `Cnn` configured with a Conv2D layer (3 input channels -> CONV_OUT filters, 3×3 kernel, padding=1)
/// and a Dense layer (FC_IN -> NUM_CLASSES) ready for training or evaluation.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let model = init_cnn(&mut rng);
/// // `model` is ready to use for forward/backward passes on CIFAR-10-shaped inputs.
/// ```
fn init_cnn(rng: &mut SimpleRng) -> Cnn {
    // Conv: 3 input channels (RGB) -> CONV_OUT output channels, 3x3 kernel, pad=1, stride=1
    let conv_layer = Conv2DLayer::new(IMG_CHANNELS, CONV_OUT, KERNEL, PAD, 1, IMG_H, IMG_W, rng);

    // FC layer: FC_IN -> NUM_CLASSES
    let fc_layer = DenseLayer::new(FC_IN, NUM_CLASSES, rng);

    Cnn {
        conv_layer,
        fc_layer,
    }
}

// Forward conv + ReLU.
// input: [batch * 3072], conv_out: [batch * CONV_OUT * 32 * 32]
/// Runs the convolutional layer on a batch and applies ReLU activation to the convolution outputs.
///
/// # Parameters
///
/// - `model`: CNN containing the convolutional layer to run.
/// - `batch_size`: number of samples in the current batch.
/// - `input`: flattened input buffer for the batch (layout: batch-major, channels-last (NHWC)/pixel-interleaved per sample),
///   matching `Conv2DLayer` expectations.
/// - `conv_out`: preallocated batch-major output buffer that `Conv2DLayer` overwrites with convolution results.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng);
/// let batch_size = 8;
/// let mut input = vec![0.0f32; batch_size * NUM_INPUTS];
/// let mut conv_out = vec![0.0f32; batch_size * model.conv_layer.output_height() * model.conv_layer.output_width() * model.conv_layer.out_channels()];
/// conv_forward_relu(&mut model, batch_size, &input, &mut conv_out);
/// ```
fn conv_forward_relu(model: &mut Cnn, batch_size: usize, input: &[f32], conv_out: &mut [f32]) {
    // Use Conv2DLayer for forward pass
    model.conv_layer.forward(input, conv_out, batch_size);

    // Apply ReLU activation
    relu_inplace(conv_out);
}

// MaxPool 2x2 stride 2.
// conv_act: [batch * C * 32 * 32] (post-ReLU)
// pool_out: [batch * C * 16 * 16]
// pool_idx: [batch * C * 16 * 16], stores argmax 0..3 (dy*2+dx)
/// Performs 2x2 max pooling with stride 2 across spatial dimensions for a batch of convolution activations.
///
/// For each item in the batch and for each output channel, this function scans each non-overlapping 2x2
/// window of the channel's 32x32 activation map, writes the maximum value into `pool_out`, and records
/// the position of that maximum within the 2x2 window (as `0..3`, computed row-major: `dy*POOL + dx`)
/// into `pool_idx`. `pool_out` and `pool_idx` must be sized to hold `batch * CONV_OUT * POOL_H * POOL_W`
/// elements.
///
/// # Examples
///
/// ```
/// let batch = 1usize;
/// // full-size buffers using module constants
/// let mut conv_act = vec![0.0f32; batch * IMG_H * IMG_W * CONV_OUT];
/// let mut pool_out = vec![0.0f32; batch * POOL_H * POOL_W * CONV_OUT];
/// let mut pool_idx = vec![0u8; batch * POOL_H * POOL_W * CONV_OUT];
///
/// // Set a single 2x2 window's top-left element to be the max for channel 0, block (0,0)
/// let conv_spatial = IMG_H * IMG_W;
/// conv_act[0 * (CONV_OUT * conv_spatial) + (0 * IMG_W + 0) * CONV_OUT + 0] = 1.0;
///
/// maxpool_forward(batch, &conv_act, &mut pool_out, &mut pool_idx);
///
/// // The pooled value for channel 0 at pool position (0,0) should be 1.0 and argmax 0 (top-left).
/// assert_eq!(pool_out[0], 1.0);
/// assert_eq!(pool_idx[0], 0u8);
/// ```
fn maxpool_forward(batch: usize, conv_act: &[f32], pool_out: &mut [f32], pool_idx: &mut [u8]) {
    let conv_spatial = IMG_H * IMG_W;
    let pool_spatial = POOL_H * POOL_W;

    for b in 0..batch {
        let conv_base_b = b * (CONV_OUT * conv_spatial);
        let pool_base_b = b * (CONV_OUT * pool_spatial);

        for py in 0..POOL_H {
            for px in 0..POOL_W {
                let iy0 = py * POOL;
                let ix0 = px * POOL;
                let pool_base = pool_base_b + (py * POOL_W + px) * CONV_OUT;

                for c in 0..CONV_OUT {
                    // Track argmax to route gradients during backprop.
                    let mut best = -f32::INFINITY;
                    let mut best_idx = 0u8;

                    for dy in 0..POOL {
                        for dx in 0..POOL {
                            let iy = iy0 + dy;
                            let ix = ix0 + dx;
                            let v = conv_act[conv_base_b + (iy * IMG_W + ix) * CONV_OUT + c];
                            let idx = (dy * POOL + dx) as u8; // 0..3
                            if v > best {
                                best = v;
                                best_idx = idx;
                            }
                        }
                    }

                    let out_i = pool_base + c;
                    pool_out[out_i] = best;
                    pool_idx[out_i] = best_idx;
                }
            }
        }
    }
}

// FC forward: logits = X*W + b.
// X: [batch * FC_IN], logits: [batch * 10]
/// Runs the model's fully connected layer to produce class logits for a batch.
///
/// - `batch` is the number of examples in the input batch.
/// - `x` is the input buffer containing `batch` rows of size `FC_IN` (flattened row-major).
/// - `logits` is the output buffer for `batch` rows of size `NUM_CLASSES` (flattened row-major).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let mut model = init_cnn(&mut rng);
/// let batch = 2;
/// let mut inputs = vec![0.0f32; batch * FC_IN];
/// // fill inputs...
/// let mut logits = vec![0.0f32; batch * NUM_CLASSES];
/// fc_forward(&mut model, batch, &inputs, &mut logits);
/// assert_eq!(logits.len(), batch * NUM_CLASSES);
/// ```
fn fc_forward(model: &mut Cnn, batch: usize, x: &[f32], logits: &mut [f32]) {
    // Use DenseLayer for forward pass
    model.fc_layer.forward(x, logits, batch);
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

// FC backward: compute gradW, gradB and dX.
/// Backpropagates through the model's final fully connected layer, computing gradients
/// with respect to the layer's inputs and accumulating parameter gradients inside the layer.
///
/// # Parameters
///
/// - `model`: CNN containing the fully connected layer to backpropagate through.
/// - `batch`: number of examples in the current minibatch.
/// - `x`: input activations to the fully connected layer with length `batch * FC_IN`.
/// - `delta`: gradient of the loss w.r.t. the FC layer logits with length `batch * NUM_CLASSES`.
/// - `d_x`: output buffer written with gradients w.r.t. `x`, length `batch * FC_IN`.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng);
/// let batch = 1usize;
/// let x = vec![0f32; batch * FC_IN];
/// let delta = vec![0f32; batch * NUM_CLASSES];
/// let mut d_x = vec![0f32; batch * FC_IN];
/// fc_backward(&mut model, batch, &x, &delta, &mut d_x);
/// assert_eq!(d_x.len(), batch * FC_IN);
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
/// Backpropagates through 2×2 max-pooling and the following ReLU, writing gradients into `conv_grad`.
///
/// Distributes each pooled gradient back to the corresponding position in the pre-pooled
/// convolution activation map using `pool_idx`, then applies the ReLU mask by zeroing any
/// gradient where the post-ReLU activation (`conv_act`) is <= 0.
///
/// # Parameters
/// - `batch`: number of examples in the batch.
/// - `conv_act`: convolution activations after ReLU (shape: batch × IMG_H × IMG_W × CONV_OUT).
/// - `pool_grad`: gradients with respect to pooled outputs (shape: batch × POOL_H × POOL_W × CONV_OUT).
/// - `pool_idx`: argmax indices recorded during pooling (values 0..3) indicating which element
///   inside each 2×2 window was selected in the forward pass.
/// - `conv_grad`: output buffer for gradients with respect to convolution inputs (shape:
///   batch × IMG_H × IMG_W × CONV_OUT). This buffer is zeroed and then filled in-place.
///
/// # Examples
///
/// ```
/// let batch = 1;
/// let mut conv_act = vec![1.0f32; CONV_OUT * IMG_H * IMG_W];
/// let pool_spatial = POOL_H * POOL_W;
/// let mut pool_grad = vec![0.0f32; CONV_OUT * pool_spatial];
/// let mut pool_idx = vec![0u8; CONV_OUT * pool_spatial];
///
/// // set a gradient for the first pooled location and route it to the top-left of the 2x2 window
/// pool_grad[0] = 2.0;
/// pool_idx[0] = 0; // top-left within the 2x2 window
///
/// let mut conv_grad = vec![0.0f32; IMG_H * IMG_W * CONV_OUT];
/// maxpool_backward_relu(batch, &conv_act, &pool_grad, &pool_idx, &mut conv_grad);
///
/// // the pooled gradient should be scattered to the corresponding position in conv_grad
/// assert_eq!(conv_grad[0], 2.0);
/// ```
fn maxpool_backward_relu(
    batch: usize,
    conv_act: &[f32],  // post-ReLU
    pool_grad: &[f32], // [batch*POOL_H*POOL_W*CONV_OUT]
    pool_idx: &[u8],
    conv_grad: &mut [f32], // [batch*IMG_H*IMG_W*CONV_OUT]
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

        for py in 0..POOL_H {
            for px in 0..POOL_W {
                let pool_base = pool_base_b + (py * POOL_W + px) * CONV_OUT;

                for c in 0..CONV_OUT {
                    let p_i = pool_base + c;
                    let g = pool_grad[p_i];
                    let a = pool_idx[p_i] as usize; // 0..3
                    let dy = a / POOL;
                    let dx = a % POOL;

                    let iy = py * POOL + dy;
                    let ix = px * POOL + dx;

                    let c_i = conv_base_b + (iy * IMG_W + ix) * CONV_OUT + c;
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
/// Accumulates gradients for the model's convolutional layer from a batch's inputs and convolutional
/// output gradients.
///
/// The function runs the convolutional layer's backward pass, updating the layer's internal gradient
/// buffers (grad_weights and grad_biases) using `input` and `conv_grad`. The provided `_grad_input`
/// buffer is accepted for API compatibility but is unused for the first layer.
///
/// # Parameters
/// - `model`: The CNN containing the convolutional layer to update.
/// - `batch`: Number of examples in the batch.
/// - `input`: Flattened input buffer with shape `[batch * NUM_INPUTS]` (e.g., batch * 3072 for CIFAR-10).
/// - `conv_grad`: Flattened convolutional output gradients with shape `[batch * CONV_OUT * H * W]`
///   (e.g., batch * 16 * 32 * 32).
/// - `_grad_input`: Mutable buffer for gradients w.r.t. the input; unused for the first convolutional layer.
///
/// # Examples
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng);
/// let batch = 2usize;
/// let mut input = vec![0f32; batch * NUM_INPUTS];
/// let mut conv_grad = vec![0f32; batch * CONV_OUT * IMG_H * IMG_W];
/// let mut grad_input = vec![0f32; batch * NUM_INPUTS]; // unused here
///
/// // Populate input and conv_grad as needed...
/// conv_backward(&mut model, batch, &input, &conv_grad, &mut grad_input);
/// ```
fn conv_backward(
    model: &mut Cnn,
    batch: usize,
    input: &[f32],           // [batch*3072]
    conv_grad: &[f32],       // [batch*C*32*32]
    _grad_input: &mut [f32], // unused (first layer)
) {
    // Use Conv2DLayer for backward pass (gradients are accumulated internally)
    // Note: grad_input is unused since this is the first layer
    model
        .conv_layer
        .backward(input, conv_grad, _grad_input, batch);
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
/// Writes the model's parameters to a binary file in a simple little-endian format.
///
/// The file layout is:
/// 1) Conv layer metadata as i32: out_channels, in_channels, kernel_size, input_height, input_width
/// 2) Conv layer weights (f32) in the order returned by `Conv2DLayer::weights()`
/// 3) Conv layer biases (f32) in the order returned by `Conv2DLayer::biases()`
/// 4) FC layer metadata as i32: input_size, output_size
/// 5) FC layer weights (f32) in the order returned by `DenseLayer::weights()`
/// 6) FC layer biases (f32) in the order returned by `DenseLayer::biases()`
///
/// On any file or write error the process exits with a nonzero status.
///
/// # Examples
///
/// ```
/// # use crate::{SimpleRng, init_cnn, save_model};
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng);
/// save_model(&model, "cifar10_cnn_model.bin");
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
    write_i32(&mut writer, model.conv_layer.out_channels() as i32);
    write_i32(&mut writer, model.conv_layer.in_channels() as i32);
    write_i32(&mut writer, model.conv_layer.kernel_size() as i32);
    write_i32(&mut writer, model.conv_layer.input_height() as i32);
    write_i32(&mut writer, model.conv_layer.input_width() as i32);

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

/// Builds a learning-rate scheduler using an optional config file path supplied via command-line arguments.
///
/// If `args` contains a second element, it is treated as the path to a scheduler configuration file; otherwise the default scheduler is created using the module's `LEARNING_RATE` and `EPOCHS` constants.
///
/// # Parameters
///
/// - `args`: command-line arguments slice where `args[1]`, if present, is the optional scheduler config file path.
///
/// # Returns
///
/// A boxed implementation of `LRScheduler` constructed from the provided config path or from the default hyperparameters.
///
/// # Examples
///
/// ```
/// // Simulate invocation with a config path
/// let args = vec!["program".to_string(), "lr_config.toml".to_string()];
/// let scheduler = scheduler_from_args(&args);
/// // `scheduler` implements `LRScheduler` and can be queried for learning rates per epoch.
/// assert!(scheduler.get_lr(0) > 0.0);
/// ```
fn scheduler_from_args(args: &[String]) -> Box<dyn LRScheduler> {
    let config_path = if args.len() > 1 {
        Some(args[1].as_str())
    } else {
        None
    };
    create_scheduler_from_config(LEARNING_RATE, EPOCHS, config_path)
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
    let validation_samples = (total_train_samples as f32 * VALIDATION_SPLIT) as usize;
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

    let mut model = init_cnn(&mut rng);

    // Parse command-line arguments for optional config file
    let args: Vec<String> = env::args().collect();
    let mut scheduler = scheduler_from_args(&args);

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let log_file = File::create("./logs/training_loss_cifar10_cnn.txt").unwrap_or_else(|_| {
        eprintln!("Could not create logs/training_loss_cifar10_cnn.txt");
        process::exit(1);
    });
    let mut log = BufWriter::new(log_file);

    // Training buffers (reused each batch to avoid allocations).
    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut batch_labels = vec![0u8; BATCH_SIZE];

    let mut conv_out = vec![0.0f32; BATCH_SIZE * CONV_OUT * IMG_H * IMG_W];
    let mut pool_out = vec![0.0f32; BATCH_SIZE * FC_IN];
    let mut pool_idx = vec![0u8; BATCH_SIZE * CONV_OUT * POOL_H * POOL_W];
    let mut logits = vec![0.0f32; BATCH_SIZE * NUM_CLASSES];
    let mut delta = vec![0.0f32; BATCH_SIZE * NUM_CLASSES];

    let mut d_pool = vec![0.0f32; BATCH_SIZE * FC_IN];
    let mut d_conv = vec![0.0f32; BATCH_SIZE * CONV_OUT * IMG_H * IMG_W];
    let mut _grad_input = vec![0.0f32; BATCH_SIZE * NUM_INPUTS]; // unused (first layer)

    // Validation buffers (reused each epoch to avoid repeated allocations).
    let mut val_batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut val_conv_out = vec![0.0f32; BATCH_SIZE * CONV_OUT * IMG_H * IMG_W];
    let mut val_pool_out = vec![0.0f32; BATCH_SIZE * FC_IN];
    let mut val_pool_idx = vec![0u8; BATCH_SIZE * CONV_OUT * POOL_H * POOL_W];
    let mut val_logits = vec![0.0f32; BATCH_SIZE * NUM_CLASSES];

    let mut indices: Vec<usize> = (0..train_n).collect();

    // Early stopping state
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

    println!(
        "Training CIFAR-10 CNN: epochs={} batch={} lr={}",
        EPOCHS, BATCH_SIZE, LEARNING_RATE
    );

    for epoch in 0..EPOCHS {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);
        let current_lr = scheduler.get_lr();

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(BATCH_SIZE) {
            let batch = (train_n - batch_start).min(BATCH_SIZE);
            let scale = 1.0f32;

            // Gather a random mini-batch into contiguous buffers.
            gather_batch(
                &train_images,
                &train_labels,
                &indices,
                batch_start,
                batch,
                &mut batch_inputs,
                &mut batch_labels,
            );

            // Forward: conv -> pool -> FC -> logits.
            conv_forward_relu(&mut model, batch, &batch_inputs, &mut conv_out);
            maxpool_forward(batch, &conv_out, &mut pool_out, &mut pool_idx);
            fc_forward(&mut model, batch, &pool_out, &mut logits);

            // Softmax + loss + gradient at logits.
            let batch_loss =
                softmax_xent_backward(&mut logits, &batch_labels, batch, &mut delta, scale);
            total_loss += batch_loss;

            // Backward: FC -> pool -> conv.
            fc_backward(&mut model, batch, &pool_out, &delta, &mut d_pool);
            maxpool_backward_relu(batch, &conv_out, &d_pool, &pool_idx, &mut d_conv);
            conv_backward(&mut model, batch, &batch_inputs, &d_conv, &mut _grad_input);

            // SGD update using Layer trait (no momentum, no weight decay).
            model.fc_layer.update_parameters(current_lr);
            model.conv_layer.update_parameters(current_lr);
        }

        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;

        // Evaluate on validation set
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        for batch_start in (0..validation_samples).step_by(BATCH_SIZE) {
            let batch_count = (validation_samples - batch_start).min(BATCH_SIZE);
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
        if val_average_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA {
            best_val_loss = val_average_loss;
            epochs_without_improvement = 0;
            // Save best model
            save_model(&model, "cifar10_cnn_model_best.bin");
        } else {
            epochs_without_improvement += 1;
        }

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE {
            println!(
                "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
                EARLY_STOPPING_PATIENCE, best_val_loss
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
        );

        assert_eq!(out_labels[0], 0);
        assert_eq!(out_labels[1], 1);
    }
}
