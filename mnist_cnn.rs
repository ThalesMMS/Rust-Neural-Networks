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

use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

// ============================================================================
// Internal Abstractions (Inlined for self-contained binary)
// ============================================================================

/// Core trait for neural network layers.
pub trait Layer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    );
    fn update_parameters(&mut self, learning_rate: f32);
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn parameter_count(&self) -> usize;
}

/// Simple random number generator.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    pub fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 {
            0x9e3779b97f4a7c15
        } else {
            nanos
        };
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    pub fn shuffle_usize(&mut self, data: &mut [usize]) {
        if data.len() <= 1 {
            return;
        }
        for i in (1..data.len()).rev() {
            let j = (self.next_u32() as usize) % (i + 1);
            data.swap(i, j);
        }
    }
}

/// ReLU activation function applied in-place.
pub fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// Softmax activation function applied row-wise.
pub fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
    for row in outputs.chunks_exact_mut(cols).take(rows) {
        let mut max_value = row[0];
        for &value in row.iter().skip(1) {
            if value > max_value {
                max_value = value;
            }
        }

        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max_value).exp();
            sum += *value;
        }

        let inv_sum = 1.0f32 / sum;
        for value in row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

/// Fully connected layer (manual implementation, no BLAS).
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
        let mut weights = vec![0.0; input_size * output_size];
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        for w in &mut weights {
            *w = rng.gen_range_f32(-limit, limit);
        }

        Self {
            input_size,
            output_size,
            weights,
            biases: vec![0.0; output_size],
            grad_weights: RefCell::new(vec![0.0; input_size * output_size]),
            grad_biases: RefCell::new(vec![0.0; output_size]),
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;
            
            for j in 0..self.output_size {
                let mut sum = self.biases[j];
                for i in 0..self.input_size {
                    sum += input[in_offset + i] * self.weights[i * self.output_size + j];
                }
                output[out_offset + j] = sum;
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
        let scale = 1.0 / batch_size as f32;
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        // Zero out grad_input first as we accumulate into it
        for v in grad_input.iter_mut() { *v = 0.0; }

        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;

            for j in 0..self.output_size {
                let g = grad_output[out_offset + j];
                grad_b[j] += g * scale;

                for i in 0..self.input_size {
                    grad_w[i * self.output_size + j] += input[in_offset + i] * g * scale;
                    grad_input[in_offset + i] += g * self.weights[i * self.output_size + j];
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

        for g in grad_w.iter_mut() { *g = 0.0; }
        for g in grad_b.iter_mut() { *g = 0.0; }
    }

    fn input_size(&self) -> usize { self.input_size }
    fn output_size(&self) -> usize { self.output_size }
    fn parameter_count(&self) -> usize { self.weights.len() + self.biases.len() }
}

/// 2D Convolutional layer (Manual implementation).
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
        for v in grad_input.iter_mut() { *v = 0.0; }

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

        for g in grad_w.iter_mut() { *g = 0.0; }
        for g in grad_b.iter_mut() { *g = 0.0; }
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
}

// ============================================================================
// Main Logic
// ============================================================================

// Read a big-endian u32 and advance the byte offset (IDX format uses BE).
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
