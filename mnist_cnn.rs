// mnist_cnn.rs
// Minimal CNN for MNIST on CPU using explicit loops (no external crates).
// Expected files:
//   ./data/train-images.idx3-ubyte
//   ./data/train-labels.idx1-ubyte
//   ./data/t10k-images.idx3-ubyte
//   ./data/t10k-labels.idx1-ubyte
//
// Output:
//   - logs/training_loss_cnn.txt (epoch,loss,time)
//   - prints test accuracy
//
// Note: educational implementation (no BLAS/GEMM), so it is intentionally slow.

use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::{relu_inplace, softmax_rows, SimpleRng};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

// MNIST constants (images are flat 28x28 in row-major order).
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784
const NUM_CLASSES: usize = 10;
const TRAIN_SAMPLES: usize = 60_000;
const TEST_SAMPLES: usize = 10_000;

// CNN topology: 1x28x28 -> conv -> ReLU -> 2x2 maxpool -> FC(10).
const CONV_OUT: usize = 8;
const KERNEL: usize = 3;
const PAD: isize = 1;
const POOL: usize = 2;

const POOL_H: usize = IMG_H / POOL; // 14
const POOL_W: usize = IMG_W / POOL; // 14
const FC_IN: usize = CONV_OUT * POOL_H * POOL_W; // 8*14*14 = 1568

// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 32;

// Read a big-endian u32 and advance the byte offset (IDX format uses BE).
/// Reads a 32-bit unsigned integer encoded in big-endian order from `data` at `offset` and advances `offset` by 4.
///
/// Returns the 32-bit unsigned integer assembled from the four bytes starting at `offset` in big-endian order.
///
/// # Examples
///
/// ```
/// let buf = [0x00, 0x00, 0x01, 0x02, 0xff];
/// let mut off = 0usize;
/// let v = read_be_u32(&buf, &mut off);
/// assert_eq!(v, 0x00000102);
/// assert_eq!(off, 4);
/// ```
fn read_be_u32(data: &[u8], offset: &mut usize) -> u32 {
    let b0 = (data[*offset] as u32) << 24;
    let b1 = (data[*offset + 1] as u32) << 16;
    let b2 = (data[*offset + 2] as u32) << 8;
    let b3 = data[*offset + 3] as u32;
    *offset += 4;
    b0 | b1 | b2 | b3
}

// Read IDX images and normalize to [0,1] floats.
fn read_mnist_images(filename: &str, num_images: usize) -> Vec<f32> {
    let data = fs::read(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let mut offset = 0usize;
    // IDX header: magic, count, rows, cols.
    let _magic = read_be_u32(&data, &mut offset);
    let total_images = read_be_u32(&data, &mut offset) as usize;
    let rows = read_be_u32(&data, &mut offset) as usize;
    let cols = read_be_u32(&data, &mut offset) as usize;

    if rows != IMG_H || cols != IMG_W {
        eprintln!("Unexpected MNIST image shape: {}x{}", rows, cols);
        process::exit(1);
    }

    let image_size = rows * cols;
    let actual_count = num_images.min(total_images);
    let total_bytes = actual_count * image_size;

    if data.len() < offset + total_bytes {
        eprintln!("MNIST image file is truncated");
        process::exit(1);
    }

    // Flatten images as [N * 784] in row-major order.
    let mut images = vec![0.0f32; total_bytes];
    for i in 0..total_bytes {
        images[i] = data[offset + i] as f32 / 255.0;
    }
    images
}

// Read IDX labels (0-9).
fn read_mnist_labels(filename: &str, num_labels: usize) -> Vec<u8> {
    let data = fs::read(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let mut offset = 0usize;
    let _magic = read_be_u32(&data, &mut offset);
    let total_labels = read_be_u32(&data, &mut offset) as usize;
    let actual_count = num_labels.min(total_labels);

    if data.len() < offset + actual_count {
        eprintln!("MNIST label file is truncated");
        process::exit(1);
    }

    data[offset..offset + actual_count].to_vec()
}

// Copy a subset of images/labels into contiguous batch buffers.
/// Copies a contiguous mini-batch of examples and their labels into preallocated output buffers.
///
/// The function reads `count` examples by mapping `indices[start..start+count]` into `images` and
/// `labels`, copying each example's NUM_INPUTS floats into `out_inputs` and the corresponding label
/// into `out_labels` in batch order.
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
/// gather_batch(&images, &labels, &indices, 0, 2, &mut batch_inputs, &mut batch_labels);
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
/// let mut rng = SimpleRng::seed_from_u64(123);
/// let mut model = init_cnn(&mut rng);
/// let batch = 0usize;
/// let mut inputs = vec![0f32; BATCH_SIZE * NUM_INPUTS];
/// let mut conv_out = vec![0f32; BATCH_SIZE * CONV_OUT * CONV_H * CONV_W];
/// // populate inputs[batch * NUM_INPUTS .. (batch+1) * NUM_INPUTS] as needed
/// conv_forward_relu(&mut model, batch, &inputs, &mut conv_out);
/// // after call, conv_out values for the batch are non-negative due to ReLU
/// let start = batch * CONV_OUT * CONV_H * CONV_W;
/// assert!(conv_out[start..start + CONV_OUT * CONV_H * CONV_W].iter().all(|&v| v >= 0.0));
/// ```
fn conv_forward_relu(model: &mut Cnn, batch: usize, input: &[f32], conv_out: &mut [f32]) {
    // Use Conv2DLayer for forward pass
    model.conv_layer.forward(input, conv_out, batch);

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

// Softmax + cross-entropy: returns summed loss and writes delta = (probs - onehot) * scale.
fn softmax_xent_backward(
    probs_inplace: &mut [f32], // logits overwritten with probs
    labels: &[u8],
    batch: usize,
    delta: &mut [f32],
    scale: f32,
) -> f32 {
    let eps = 1e-9f32;
    softmax_rows(probs_inplace, batch, NUM_CLASSES);

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

/// - `conv_grad`: gradients w.r.t. convolution pre-activations, layout `batch * CONV_OUT * IMAGE_H * IMAGE_W`.

/// - `_grad_input`: destination buffer for gradients w.r.t. this layer's inputs; unused for the first layer.

///

/// # Examples

///

/// ```

/// let mut rng = SimpleRng::new(123);

/// let mut model = init_cnn(&mut rng);

/// let batch = 1;

/// let input = vec![0.0f32; batch * NUM_INPUTS];

/// let conv_grad = vec![0.0f32; batch * CONV_OUT * IMAGE_H * IMAGE_W];

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
    println!("Loading MNIST...");
    let train_images = read_mnist_images("./data/train-images.idx3-ubyte", TRAIN_SAMPLES);
    let train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte", TRAIN_SAMPLES);
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte", TEST_SAMPLES);
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte", TEST_SAMPLES);

    let train_n = train_labels.len();
    let test_n = test_labels.len();
    println!("Train: {} | Test: {}", train_n, test_n);

    let mut rng = SimpleRng::new(1);
    rng.reseed_from_time();

    let mut model = init_cnn(&mut rng);

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let log_file = File::create("./logs/training_loss_cnn.txt").unwrap_or_else(|_| {
        eprintln!("Could not create logs/training_loss_cnn.txt");
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

    let mut indices: Vec<usize> = (0..train_n).collect();

    println!(
        "Training CNN: epochs={} batch={} lr={}",
        EPOCHS, BATCH_SIZE, LEARNING_RATE
    );

    for epoch in 0..EPOCHS {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(BATCH_SIZE) {
            let batch = (train_n - batch_start).min(BATCH_SIZE);
            let scale = 1.0f32 / batch as f32;

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
            model.fc_layer.update_parameters(LEARNING_RATE);
            model.conv_layer.update_parameters(LEARNING_RATE);
        }

        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;
        println!(
            "Epoch {} | loss={:.6} | time={:.3}s",
            epoch + 1,
            avg_loss,
            secs
        );
        writeln!(log, "{},{},{}", epoch + 1, avg_loss, secs).ok();
    }

    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_be_u32() {
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x00, 0x00];
        let mut offset = 0;

        let value = read_be_u32(&data, &mut offset);

        assert_eq!(value, 0x01020304);
        assert_eq!(offset, 4);
    }

    #[test]
    fn test_gather_batch() {
        let images = vec![1.0; 784 * 3]; // 3 images
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![0, 1, 2];
        let mut out_inputs = vec![0.0; 784 * 2]; // batch of 2
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