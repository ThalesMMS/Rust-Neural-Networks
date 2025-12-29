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

// Tiny xorshift RNG for reproducible init without external crates.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 { 0x9e3779b97f4a7c15 } else { nanos };
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }

    fn shuffle_usize(&mut self, data: &mut [usize]) {
        if data.len() <= 1 {
            return;
        }
        for i in (1..data.len()).rev() {
            let j = self.gen_usize(i + 1);
            data.swap(i, j);
        }
    }
}

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

// In-place ReLU for a flat buffer.
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

// Softmax row-wise (in-place). Buffer layout: rows * cols.
fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let row = &mut data[base..base + cols];

        let mut maxv = row[0];
        for &v in row.iter().skip(1) {
            if v > maxv {
                maxv = v;
            }
        }

        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }

        let inv = 1.0f32 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

// CNN parameters stored in flat arrays for cache-friendly loops.
struct Cnn {
    // Conv: 1 -> CONV_OUT, kernel KERNELxKERNEL, pad=1, stride=1
    conv_w: Vec<f32>, // [CONV_OUT * KERNEL * KERNEL]
    conv_b: Vec<f32>, // [CONV_OUT]
    // FC: FC_IN -> 10
    fc_w: Vec<f32>, // [FC_IN * 10]
    fc_b: Vec<f32>, // [10]
}

// Xavier/Glorot uniform init for stable signal magnitudes.
fn xavier_init(limit: f32, rng: &mut SimpleRng, w: &mut [f32]) {
    for v in w.iter_mut() {
        *v = rng.gen_range_f32(-limit, limit);
    }
}

fn init_cnn(rng: &mut SimpleRng) -> Cnn {
    // Xavier limits based on approximate fan-in/out.
    let fan_in = (KERNEL * KERNEL) as f32;
    let fan_out = (KERNEL * KERNEL * CONV_OUT) as f32;
    let conv_limit = (6.0f32 / (fan_in + fan_out)).sqrt();

    let mut conv_w = vec![0.0f32; CONV_OUT * KERNEL * KERNEL];
    let conv_b = vec![0.0f32; CONV_OUT];
    xavier_init(conv_limit, rng, &mut conv_w);

    // FC layer init.
    let fc_limit = (6.0f32 / (FC_IN as f32 + NUM_CLASSES as f32)).sqrt();
    let mut fc_w = vec![0.0f32; FC_IN * NUM_CLASSES];
    let fc_b = vec![0.0f32; NUM_CLASSES];
    xavier_init(fc_limit, rng, &mut fc_w);

    Cnn {
        conv_w,
        conv_b,
        fc_w,
        fc_b,
    }
}

// Forward conv + ReLU.
// input: [batch * 784], conv_out: [batch * CONV_OUT * 28 * 28]
fn conv_forward_relu(
    model: &Cnn,
    batch: usize,
    input: &[f32],
    conv_out: &mut [f32],
) {
    let spatial = IMG_H * IMG_W;
    let out_spatial = spatial;

    for b in 0..batch {
        let in_base = b * NUM_INPUTS;
        let out_base_b = b * (CONV_OUT * out_spatial);

        for oc in 0..CONV_OUT {
            let w_base = oc * (KERNEL * KERNEL);
            let bias = model.conv_b[oc];
            let out_base = out_base_b + oc * out_spatial;

            // For each output pixel, accumulate a 3x3 window with zero-padding.
            for oy in 0..IMG_H {
                for ox in 0..IMG_W {
                    let mut sum = bias;
                    for ky in 0..KERNEL {
                        for kx in 0..KERNEL {
                            let iy = oy as isize + ky as isize - PAD;
                            let ix = ox as isize + kx as isize - PAD;
                            if iy >= 0 && iy < IMG_H as isize && ix >= 0 && ix < IMG_W as isize {
                                let iyy = iy as usize;
                                let ixx = ix as usize;
                                let in_idx = in_base + iyy * IMG_W + ixx;
                                let w_idx = w_base + ky * KERNEL + kx;
                                sum += input[in_idx] * model.conv_w[w_idx];
                            }
                        }
                    }
                    let out_idx = out_base + oy * IMG_W + ox;
                    conv_out[out_idx] = if sum > 0.0 { sum } else { 0.0 }; // ReLU activation
                }
            }
        }
    }
}

// MaxPool 2x2 stride 2.
// conv_act: [batch * C * 28 * 28] (post-ReLU)
// pool_out: [batch * C * 14 * 14]
// pool_idx: [batch * C * 14 * 14], stores argmax 0..3 (dy*2+dx)
fn maxpool_forward(
    batch: usize,
    conv_act: &[f32],
    pool_out: &mut [f32],
    pool_idx: &mut [u8],
) {
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
fn fc_forward(model: &Cnn, batch: usize, x: &[f32], logits: &mut [f32]) {
    for b in 0..batch {
        let x_base = b * FC_IN;
        let o_base = b * NUM_CLASSES;
        for j in 0..NUM_CLASSES {
            let mut sum = model.fc_b[j];
            for i in 0..FC_IN {
                sum += x[x_base + i] * model.fc_w[i * NUM_CLASSES + j];
            }
            logits[o_base + j] = sum;
        }
    }
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
    for b in 0..batch {
        let base = b * NUM_CLASSES;
        let y = labels[b] as usize;

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
    model: &Cnn,
    batch: usize,
    x: &[f32],
    delta: &[f32], // [batch*10]
    grad_w: &mut [f32], // [FC_IN*10]
    grad_b: &mut [f32], // [10]
    d_x: &mut [f32],    // [batch*FC_IN]
) {
    // Zero gradients (accumulated over batch).
    for v in grad_w.iter_mut() {
        *v = 0.0;
    }
    for v in grad_b.iter_mut() {
        *v = 0.0;
    }

    // gradW and gradB.
    for b in 0..batch {
        let x_base = b * FC_IN;
        let d_base = b * NUM_CLASSES;

        for j in 0..NUM_CLASSES {
            grad_b[j] += delta[d_base + j];
        }

        for i in 0..FC_IN {
            let xi = x[x_base + i];
            let w_row = i * NUM_CLASSES;
            for j in 0..NUM_CLASSES {
                grad_w[w_row + j] += xi * delta[d_base + j];
            }
        }
    }

    // dX = delta * W^T.
    for b in 0..batch {
        let d_base = b * NUM_CLASSES;
        let out_base = b * FC_IN;

        for i in 0..FC_IN {
            let w_row = i * NUM_CLASSES;
            let mut sum = 0.0f32;
            for j in 0..NUM_CLASSES {
                sum += delta[d_base + j] * model.fc_w[w_row + j];
            }
            d_x[out_base + i] = sum;
        }
    }
}

// MaxPool backward: scatter grads to argmax positions, then apply ReLU mask.
fn maxpool_backward_relu(
    batch: usize,
    conv_act: &[f32], // post-ReLU
    pool_grad: &[f32], // [batch*C*14*14]
    pool_idx: &[u8],
    conv_grad: &mut [f32], // [batch*C*28*28]
) {
    let conv_spatial = IMG_H * IMG_W;
    let pool_spatial = POOL_H * POOL_W;

    // Zero conv_grad so we can scatter-add into it.
    let used = batch * CONV_OUT * conv_spatial;
    for i in 0..used {
        conv_grad[i] = 0.0;
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
    model: &Cnn,
    batch: usize,
    input: &[f32],     // [batch*784]
    conv_grad: &[f32], // [batch*C*28*28]
    grad_w: &mut [f32], // [C*K*K]
    grad_b: &mut [f32], // [C]
) {
    for v in grad_w.iter_mut() {
        *v = 0.0;
    }
    for v in grad_b.iter_mut() {
        *v = 0.0;
    }

    let spatial = IMG_H * IMG_W;
    for b in 0..batch {
        let in_base = b * NUM_INPUTS;
        let g_base_b = b * (CONV_OUT * spatial);

        for oc in 0..CONV_OUT {
            let w_base = oc * (KERNEL * KERNEL);
            let g_base = g_base_b + oc * spatial;

            for oy in 0..IMG_H {
                for ox in 0..IMG_W {
                    let g = conv_grad[g_base + oy * IMG_W + ox];
                    grad_b[oc] += g;

                    for ky in 0..KERNEL {
                        for kx in 0..KERNEL {
                            let iy = oy as isize + ky as isize - PAD;
                            let ix = ox as isize + kx as isize - PAD;
                            if iy >= 0 && iy < IMG_H as isize && ix >= 0 && ix < IMG_W as isize {
                                let iyy = iy as usize;
                                let ixx = ix as usize;
                                let in_idx = in_base + iyy * IMG_W + ixx;
                                let w_idx = w_base + ky * KERNEL + kx;
                                grad_w[w_idx] += g * input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Note: conv_grad already includes the 1/batch scale from delta.
    // Therefore grad_w/grad_b are averages over the batch.
    let _ = model;
}

fn test_accuracy(model: &Cnn, images: &[f32], labels: &[u8]) -> f32 {
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

    let mut grad_fc_w = vec![0.0f32; FC_IN * NUM_CLASSES];
    let mut grad_fc_b = vec![0.0f32; NUM_CLASSES];
    let mut grad_conv_w = vec![0.0f32; CONV_OUT * KERNEL * KERNEL];
    let mut grad_conv_b = vec![0.0f32; CONV_OUT];

    let mut indices: Vec<usize> = (0..train_n).collect();

    println!("Training CNN: epochs={} batch={} lr={}", EPOCHS, BATCH_SIZE, LEARNING_RATE);

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
            conv_forward_relu(&model, batch, &batch_inputs, &mut conv_out);
            maxpool_forward(batch, &conv_out, &mut pool_out, &mut pool_idx);
            fc_forward(&model, batch, &pool_out, &mut logits);

            // Softmax + loss + gradient at logits.
            let batch_loss = softmax_xent_backward(&mut logits, &batch_labels, batch, &mut delta, scale);
            total_loss += batch_loss;

            // Backward: FC -> pool -> conv.
            fc_backward(&model, batch, &pool_out, &delta, &mut grad_fc_w, &mut grad_fc_b, &mut d_pool);
            maxpool_backward_relu(batch, &conv_out, &d_pool, &pool_idx, &mut d_conv);
            conv_backward(&model, batch, &batch_inputs, &d_conv, &mut grad_conv_w, &mut grad_conv_b);

            // SGD update (no momentum, no weight decay).
            for i in 0..model.fc_w.len() {
                model.fc_w[i] -= LEARNING_RATE * grad_fc_w[i];
            }
            for i in 0..model.fc_b.len() {
                model.fc_b[i] -= LEARNING_RATE * grad_fc_b[i];
            }
            for i in 0..model.conv_w.len() {
                model.conv_w[i] -= LEARNING_RATE * grad_conv_w[i];
            }
            for i in 0..model.conv_b.len() {
                model.conv_b[i] -= LEARNING_RATE * grad_conv_b[i];
            }
        }

        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;
        println!("Epoch {} | loss={:.6} | time={:.3}s", epoch + 1, avg_loss, secs);
        writeln!(log, "{},{},{}", epoch + 1, avg_loss, secs).ok();
    }

    println!("Testing...");
    let acc = test_accuracy(&model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}
