// mnist_attention_pool.rs
// Self-attention over patch tokens for MNIST (single-head Transformer-style).
//
// Model:
//   - Split the 28x28 image into 4x4 patches => 7x7 = 49 tokens.
//   - Project each patch to a D-dimensional embedding (linear + bias) and add position embeddings.
//   - Apply ReLU.
//   - Self-attention (1 head): Q/K/V with a 49x49 score matrix per sample.
//   - Feed-forward MLP per token (D -> FF -> D).
//   - Mean-pool tokens and classify to 10 classes.
//
// Focus: educational (CPU loops). No external crates.
// Requires the MNIST IDX files in ./data:
//   train-images.idx3-ubyte
//   train-labels.idx1-ubyte
//   t10k-images.idx3-ubyte
//   t10k-labels.idx1-ubyte

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

// Patch grid and tokenization.
const PATCH: usize = 4;
const GRID: usize = IMG_H / PATCH; // 7
const SEQ_LEN: usize = GRID * GRID; // 49
const PATCH_DIM: usize = PATCH * PATCH; // 16

// Model width. Keep small for fast CPU training.
const D_MODEL: usize = 16;
// Feed-forward hidden size.
const FF_DIM: usize = 32;

// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 5;
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
    let src = &data[offset..offset + total_bytes];
    for (dst, &px) in images.iter_mut().zip(src.iter()) {
        *dst = px as f32 / 255.0;
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

// Softmax in-place for a single vector.
fn softmax_inplace(v: &mut [f32]) {
    let mut maxv = v[0];
    for &x in v.iter().skip(1) {
        if x > maxv {
            maxv = x;
        }
    }

    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - maxv).exp();
        sum += *x;
    }

    let inv = 1.0f32 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

// Softmax row-wise for a flat [rows * cols] buffer.
fn softmax_rows_inplace(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        softmax_inplace(&mut data[base..base + cols]);
    }
}

struct AttnModel {
    // Patch projection: token = patch * W + b.
    w_patch: Vec<f32>, // [PATCH_DIM * D_MODEL]
    b_patch: Vec<f32>, // [D_MODEL]
    // Positional embedding per token.
    pos: Vec<f32>, // [SEQ_LEN * D_MODEL]
    // Self-attention projections.
    w_q: Vec<f32>, // [D_MODEL * D_MODEL]
    b_q: Vec<f32>, // [D_MODEL]
    w_k: Vec<f32>, // [D_MODEL * D_MODEL]
    b_k: Vec<f32>, // [D_MODEL]
    w_v: Vec<f32>, // [D_MODEL * D_MODEL]
    b_v: Vec<f32>, // [D_MODEL]
    // Feed-forward MLP (per token).
    w_ff1: Vec<f32>, // [D_MODEL * FF_DIM]
    b_ff1: Vec<f32>, // [FF_DIM]
    w_ff2: Vec<f32>, // [FF_DIM * D_MODEL]
    b_ff2: Vec<f32>, // [D_MODEL]
    // Classifier head.
    w_cls: Vec<f32>, // [D_MODEL * NUM_CLASSES]
    b_cls: Vec<f32>, // [NUM_CLASSES]
}

struct Grads {
    w_patch: Vec<f32>,
    b_patch: Vec<f32>,
    pos: Vec<f32>,
    w_q: Vec<f32>,
    b_q: Vec<f32>,
    w_k: Vec<f32>,
    b_k: Vec<f32>,
    w_v: Vec<f32>,
    b_v: Vec<f32>,
    w_ff1: Vec<f32>,
    b_ff1: Vec<f32>,
    w_ff2: Vec<f32>,
    b_ff2: Vec<f32>,
    w_cls: Vec<f32>,
    b_cls: Vec<f32>,
}

impl Grads {
    fn new() -> Self {
        Self {
            w_patch: vec![0.0; PATCH_DIM * D_MODEL],
            b_patch: vec![0.0; D_MODEL],
            pos: vec![0.0; SEQ_LEN * D_MODEL],
            w_q: vec![0.0; D_MODEL * D_MODEL],
            b_q: vec![0.0; D_MODEL],
            w_k: vec![0.0; D_MODEL * D_MODEL],
            b_k: vec![0.0; D_MODEL],
            w_v: vec![0.0; D_MODEL * D_MODEL],
            b_v: vec![0.0; D_MODEL],
            w_ff1: vec![0.0; D_MODEL * FF_DIM],
            b_ff1: vec![0.0; FF_DIM],
            w_ff2: vec![0.0; FF_DIM * D_MODEL],
            b_ff2: vec![0.0; D_MODEL],
            w_cls: vec![0.0; D_MODEL * NUM_CLASSES],
            b_cls: vec![0.0; NUM_CLASSES],
        }
    }

    fn zero(&mut self) {
        // Reset all gradients to zero before accumulation.
        for v in self.w_patch.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_patch.iter_mut() {
            *v = 0.0;
        }
        for v in self.pos.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_q.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_q.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_k.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_k.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_v.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_v.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_ff1.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_ff1.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_ff2.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_ff2.iter_mut() {
            *v = 0.0;
        }
        for v in self.w_cls.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_cls.iter_mut() {
            *v = 0.0;
        }
    }
}

struct BatchBuffers {
    // Forward buffers.
    patches: Vec<f32>,   // [BATCH * SEQ * PATCH_DIM]
    tok: Vec<f32>,       // [BATCH * SEQ * D_MODEL] (post-ReLU)
    q: Vec<f32>,         // [BATCH * SEQ * D_MODEL]
    k: Vec<f32>,         // [BATCH * SEQ * D_MODEL]
    v: Vec<f32>,         // [BATCH * SEQ * D_MODEL]
    attn: Vec<f32>,      // [BATCH * SEQ * SEQ]
    attn_out: Vec<f32>,  // [BATCH * SEQ * D_MODEL]
    ffn1: Vec<f32>,      // [BATCH * SEQ * FF_DIM] (post-ReLU)
    ffn2: Vec<f32>,      // [BATCH * SEQ * D_MODEL]
    pooled: Vec<f32>,    // [BATCH * D_MODEL]
    logits: Vec<f32>,    // [BATCH * NUM_CLASSES]
    probs: Vec<f32>,     // [BATCH * NUM_CLASSES]

    // Backward buffers.
    dlogits: Vec<f32>,   // [BATCH * NUM_CLASSES]
    dpooled: Vec<f32>,   // [BATCH * D_MODEL]
    dffn2: Vec<f32>,     // [BATCH * SEQ * D_MODEL]
    dffn1: Vec<f32>,     // [BATCH * SEQ * FF_DIM]
    dattn: Vec<f32>,     // [BATCH * SEQ * D_MODEL]
    dalpha: Vec<f32>,    // [BATCH * SEQ * SEQ]
    dscores: Vec<f32>,   // [BATCH * SEQ * SEQ]
    dq: Vec<f32>,        // [BATCH * SEQ * D_MODEL]
    dk: Vec<f32>,        // [BATCH * SEQ * D_MODEL]
    dv: Vec<f32>,        // [BATCH * SEQ * D_MODEL]
    dtok: Vec<f32>,      // [BATCH * SEQ * D_MODEL]
}

impl BatchBuffers {
    fn new() -> Self {
        Self {
            patches: vec![0.0; BATCH_SIZE * SEQ_LEN * PATCH_DIM],
            tok: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            q: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            k: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            v: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            attn: vec![0.0; BATCH_SIZE * SEQ_LEN * SEQ_LEN],
            attn_out: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            ffn1: vec![0.0; BATCH_SIZE * SEQ_LEN * FF_DIM],
            ffn2: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            pooled: vec![0.0; BATCH_SIZE * D_MODEL],
            logits: vec![0.0; BATCH_SIZE * NUM_CLASSES],
            probs: vec![0.0; BATCH_SIZE * NUM_CLASSES],
            dlogits: vec![0.0; BATCH_SIZE * NUM_CLASSES],
            dpooled: vec![0.0; BATCH_SIZE * D_MODEL],
            dffn2: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            dffn1: vec![0.0; BATCH_SIZE * SEQ_LEN * FF_DIM],
            dattn: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            dalpha: vec![0.0; BATCH_SIZE * SEQ_LEN * SEQ_LEN],
            dscores: vec![0.0; BATCH_SIZE * SEQ_LEN * SEQ_LEN],
            dq: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            dk: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            dv: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
            dtok: vec![0.0; BATCH_SIZE * SEQ_LEN * D_MODEL],
        }
    }
}

fn init_model(rng: &mut SimpleRng) -> AttnModel {
    // Xavier init for patch projection.
    let limit_patch = (6.0f32 / (PATCH_DIM as f32 + D_MODEL as f32)).sqrt();
    let mut w_patch = vec![0.0f32; PATCH_DIM * D_MODEL];
    for v in w_patch.iter_mut() {
        *v = rng.gen_range_f32(-limit_patch, limit_patch);
    }
    let b_patch = vec![0.0f32; D_MODEL];

    // Position embeddings init (small uniform range).
    let mut pos = vec![0.0f32; SEQ_LEN * D_MODEL];
    for v in pos.iter_mut() {
        *v = rng.gen_range_f32(-0.1, 0.1);
    }

    // Xavier init for attention projections.
    let limit_attn = (6.0f32 / (D_MODEL as f32 + D_MODEL as f32)).sqrt();
    let mut w_q = vec![0.0f32; D_MODEL * D_MODEL];
    let mut w_k = vec![0.0f32; D_MODEL * D_MODEL];
    let mut w_v = vec![0.0f32; D_MODEL * D_MODEL];
    for v in w_q.iter_mut() {
        *v = rng.gen_range_f32(-limit_attn, limit_attn);
    }
    for v in w_k.iter_mut() {
        *v = rng.gen_range_f32(-limit_attn, limit_attn);
    }
    for v in w_v.iter_mut() {
        *v = rng.gen_range_f32(-limit_attn, limit_attn);
    }
    let b_q = vec![0.0f32; D_MODEL];
    let b_k = vec![0.0f32; D_MODEL];
    let b_v = vec![0.0f32; D_MODEL];

    // Xavier init for feed-forward MLP.
    let limit_ff1 = (6.0f32 / (D_MODEL as f32 + FF_DIM as f32)).sqrt();
    let mut w_ff1 = vec![0.0f32; D_MODEL * FF_DIM];
    for v in w_ff1.iter_mut() {
        *v = rng.gen_range_f32(-limit_ff1, limit_ff1);
    }
    let b_ff1 = vec![0.0f32; FF_DIM];

    let limit_ff2 = (6.0f32 / (FF_DIM as f32 + D_MODEL as f32)).sqrt();
    let mut w_ff2 = vec![0.0f32; FF_DIM * D_MODEL];
    for v in w_ff2.iter_mut() {
        *v = rng.gen_range_f32(-limit_ff2, limit_ff2);
    }
    let b_ff2 = vec![0.0f32; D_MODEL];

    // Xavier init for classifier head.
    let limit_cls = (6.0f32 / (D_MODEL as f32 + NUM_CLASSES as f32)).sqrt();
    let mut w_cls = vec![0.0f32; D_MODEL * NUM_CLASSES];
    for v in w_cls.iter_mut() {
        *v = rng.gen_range_f32(-limit_cls, limit_cls);
    }
    let b_cls = vec![0.0f32; NUM_CLASSES];

    AttnModel {
        w_patch,
        b_patch,
        pos,
        w_q,
        b_q,
        w_k,
        b_k,
        w_v,
        b_v,
        w_ff1,
        b_ff1,
        w_ff2,
        b_ff2,
        w_cls,
        b_cls,
    }
}

// Extract 4x4 patches from a contiguous batch of images.
// patches shape: [batch_count * SEQ_LEN * PATCH_DIM]
fn extract_patches(batch_inputs: &[f32], batch_count: usize, patches: &mut [f32]) {
    for b in 0..batch_count {
        let img_base = b * NUM_INPUTS;
        for py in 0..GRID {
            for px in 0..GRID {
                let t = py * GRID + px;
                let patch_base = (b * SEQ_LEN + t) * PATCH_DIM;

                for dy in 0..PATCH {
                    for dx in 0..PATCH {
                        let iy = py * PATCH + dy;
                        let ix = px * PATCH + dx;
                        let in_idx = img_base + iy * IMG_W + ix;
                        let j = dy * PATCH + dx;
                        patches[patch_base + j] = batch_inputs[in_idx];
                    }
                }
            }
        }
    }
}

// Forward pass: patch -> token -> self-attention -> FFN -> classifier + loss.
fn forward_batch(
    model: &AttnModel,
    batch_inputs: &[f32],
    batch_labels: &[u8],
    batch_count: usize,
    buf: &mut BatchBuffers,
) -> f32 {
    let used_patches = batch_count * SEQ_LEN * PATCH_DIM;
    let used_tok = batch_count * SEQ_LEN * D_MODEL;
    let used_attn = batch_count * SEQ_LEN * SEQ_LEN;
    let used_ffn1 = batch_count * SEQ_LEN * FF_DIM;
    let used_pooled = batch_count * D_MODEL;
    let used_logits = batch_count * NUM_CLASSES;

    extract_patches(batch_inputs, batch_count, &mut buf.patches[..used_patches]);

    // token = ReLU(patch * W + b + pos)
    for i in 0..used_tok {
        buf.tok[i] = 0.0;
    }

    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let patch_base = (b * SEQ_LEN + t) * PATCH_DIM;
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            let pos_base = t * D_MODEL;

            for d in 0..D_MODEL {
                let mut sum = model.b_patch[d] + model.pos[pos_base + d];
                for j in 0..PATCH_DIM {
                    sum += buf.patches[patch_base + j] * model.w_patch[j * D_MODEL + d];
                }
                // ReLU
                if sum < 0.0 {
                    sum = 0.0;
                }
                buf.tok[tok_base + d] = sum;
            }
        }
    }

    // Q/K/V projections.
    for i in 0..used_tok {
        buf.q[i] = 0.0;
        buf.k[i] = 0.0;
        buf.v[i] = 0.0;
    }
    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            for d_out in 0..D_MODEL {
                let mut sum_q = model.b_q[d_out];
                let mut sum_k = model.b_k[d_out];
                let mut sum_v = model.b_v[d_out];
                for d_in in 0..D_MODEL {
                    let x = buf.tok[tok_base + d_in];
                    sum_q += x * model.w_q[d_in * D_MODEL + d_out];
                    sum_k += x * model.w_k[d_in * D_MODEL + d_out];
                    sum_v += x * model.w_v[d_in * D_MODEL + d_out];
                }
                buf.q[tok_base + d_out] = sum_q;
                buf.k[tok_base + d_out] = sum_k;
                buf.v[tok_base + d_out] = sum_v;
            }
        }
    }

    // Self-attention: scores -> softmax -> weighted sum.
    let inv_sqrt_d = 1.0f32 / (D_MODEL as f32).sqrt();
    for i in 0..used_attn {
        buf.attn[i] = 0.0;
    }
    for i in 0..used_tok {
        buf.attn_out[i] = 0.0;
    }

    for b in 0..batch_count {
        for i in 0..SEQ_LEN {
            let row_base = (b * SEQ_LEN + i) * SEQ_LEN;
            let q_base = (b * SEQ_LEN + i) * D_MODEL;

            for j in 0..SEQ_LEN {
                let k_base = (b * SEQ_LEN + j) * D_MODEL;
                let mut score = 0.0f32;
                for d in 0..D_MODEL {
                    score += buf.q[q_base + d] * buf.k[k_base + d];
                }
                buf.attn[row_base + j] = score * inv_sqrt_d;
            }

            softmax_inplace(&mut buf.attn[row_base..row_base + SEQ_LEN]);

            let out_base = (b * SEQ_LEN + i) * D_MODEL;
            for j in 0..SEQ_LEN {
                let a = buf.attn[row_base + j];
                let v_base = (b * SEQ_LEN + j) * D_MODEL;
                for d in 0..D_MODEL {
                    buf.attn_out[out_base + d] += a * buf.v[v_base + d];
                }
            }
        }
    }

    // Feed-forward MLP per token.
    for i in 0..used_ffn1 {
        buf.ffn1[i] = 0.0;
    }
    for i in 0..used_tok {
        buf.ffn2[i] = 0.0;
    }

    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let attn_base = (b * SEQ_LEN + t) * D_MODEL;
            let ffn1_base = (b * SEQ_LEN + t) * FF_DIM;
            let ffn2_base = (b * SEQ_LEN + t) * D_MODEL;

            for h in 0..FF_DIM {
                let mut sum = model.b_ff1[h];
                for d in 0..D_MODEL {
                    sum += buf.attn_out[attn_base + d] * model.w_ff1[d * FF_DIM + h];
                }
                buf.ffn1[ffn1_base + h] = if sum > 0.0 { sum } else { 0.0 };
            }

            for d in 0..D_MODEL {
                let mut sum = model.b_ff2[d];
                for h in 0..FF_DIM {
                    sum += buf.ffn1[ffn1_base + h] * model.w_ff2[h * D_MODEL + d];
                }
                buf.ffn2[ffn2_base + d] = sum;
            }
        }
    }

    // Mean pool tokens.
    for i in 0..used_pooled {
        buf.pooled[i] = 0.0;
    }
    let inv_seq = 1.0f32 / SEQ_LEN as f32;
    for b in 0..batch_count {
        let pooled_base = b * D_MODEL;
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            for d in 0..D_MODEL {
                buf.pooled[pooled_base + d] += buf.ffn2[tok_base + d] * inv_seq;
            }
        }
    }

    // Classifier logits and softmax.
    for i in 0..used_logits {
        buf.logits[i] = 0.0;
        buf.probs[i] = 0.0;
        buf.dlogits[i] = 0.0;
    }

    for b in 0..batch_count {
        let pooled_base = b * D_MODEL;
        let log_base = b * NUM_CLASSES;

        for c in 0..NUM_CLASSES {
            let mut sum = model.b_cls[c];
            for d in 0..D_MODEL {
                sum += buf.pooled[pooled_base + d] * model.w_cls[d * NUM_CLASSES + c];
            }
            buf.logits[log_base + c] = sum;
            buf.probs[log_base + c] = sum;
        }
    }

    softmax_rows_inplace(&mut buf.probs[..used_logits], batch_count, NUM_CLASSES);

    // Loss + dlogits (softmax cross-entropy).
    let mut total_loss = 0.0f32;
    let eps = 1e-9f32;
    let scale = 1.0f32 / batch_count as f32;

    for b in 0..batch_count {
        let y = batch_labels[b] as usize;
        let base = b * NUM_CLASSES;
        let p = buf.probs[base + y].max(eps);
        total_loss += -p.ln();

        for c in 0..NUM_CLASSES {
            let mut d = buf.probs[base + c];
            if c == y {
                d -= 1.0;
            }
            buf.dlogits[base + c] = d * scale;
        }
    }

    total_loss
}

// Backward pass: classifier -> FFN -> self-attention -> token projection.
fn backward_batch(
    model: &AttnModel,
    batch_count: usize,
    buf: &mut BatchBuffers,
    grads: &mut Grads,
) {
    grads.zero();

    let used_tok = batch_count * SEQ_LEN * D_MODEL;
    let used_attn = batch_count * SEQ_LEN * SEQ_LEN;
    let used_ffn1 = batch_count * SEQ_LEN * FF_DIM;
    let used_logits = batch_count * NUM_CLASSES;
    let used_pooled = batch_count * D_MODEL;

    // Zero backward buffers.
    for i in 0..used_pooled {
        buf.dpooled[i] = 0.0;
    }
    for i in 0..used_tok {
        buf.dffn2[i] = 0.0;
        buf.dattn[i] = 0.0;
        buf.dq[i] = 0.0;
        buf.dk[i] = 0.0;
        buf.dv[i] = 0.0;
        buf.dtok[i] = 0.0;
    }
    for i in 0..used_ffn1 {
        buf.dffn1[i] = 0.0;
    }
    for i in 0..used_attn {
        buf.dalpha[i] = 0.0;
        buf.dscores[i] = 0.0;
    }

    // dpooled, grad_w_cls, grad_b_cls.
    for b in 0..batch_count {
        let base_logits = b * NUM_CLASSES;
        let base_pooled = b * D_MODEL;

        for c in 0..NUM_CLASSES {
            grads.b_cls[c] += buf.dlogits[base_logits + c];
        }

        for d in 0..D_MODEL {
            let pd = buf.pooled[base_pooled + d];
            let w_row = d * NUM_CLASSES;
            let mut acc = 0.0f32;
            for c in 0..NUM_CLASSES {
                let dl = buf.dlogits[base_logits + c];
                grads.w_cls[w_row + c] += pd * dl;
                acc += dl * model.w_cls[w_row + c];
            }
            buf.dpooled[base_pooled + d] = acc;
        }
    }

    // Distribute pooled gradients to tokens (mean pooling).
    let inv_seq = 1.0f32 / SEQ_LEN as f32;
    for b in 0..batch_count {
        let base_pooled = b * D_MODEL;
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            for d in 0..D_MODEL {
                buf.dffn2[tok_base + d] = buf.dpooled[base_pooled + d] * inv_seq;
            }
        }
    }

    // FFN2 grads and dffn1.
    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            let ffn1_base = (b * SEQ_LEN + t) * FF_DIM;

            for d in 0..D_MODEL {
                grads.b_ff2[d] += buf.dffn2[tok_base + d];
            }

            for h in 0..FF_DIM {
                let hval = buf.ffn1[ffn1_base + h];
                let w_row = h * D_MODEL;
                for d in 0..D_MODEL {
                    grads.w_ff2[w_row + d] += hval * buf.dffn2[tok_base + d];
                }
            }

            for h in 0..FF_DIM {
                let w_row = h * D_MODEL;
                let mut sum = 0.0f32;
                for d in 0..D_MODEL {
                    sum += buf.dffn2[tok_base + d] * model.w_ff2[w_row + d];
                }
                buf.dffn1[ffn1_base + h] = sum;
            }
        }
    }

    // ReLU backward for FFN1.
    for i in 0..used_ffn1 {
        if buf.ffn1[i] <= 0.0 {
            buf.dffn1[i] = 0.0;
        }
    }

    // FFN1 grads and dattention.
    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let attn_base = (b * SEQ_LEN + t) * D_MODEL;
            let ffn1_base = (b * SEQ_LEN + t) * FF_DIM;

            for h in 0..FF_DIM {
                grads.b_ff1[h] += buf.dffn1[ffn1_base + h];
            }

            for d in 0..D_MODEL {
                let w_row = d * FF_DIM;
                let mut acc = 0.0f32;
                for h in 0..FF_DIM {
                    let dh = buf.dffn1[ffn1_base + h];
                    grads.w_ff1[w_row + h] += buf.attn_out[attn_base + d] * dh;
                    acc += dh * model.w_ff1[w_row + h];
                }
                buf.dattn[attn_base + d] = acc;
            }
        }
    }

    // Attention backward: dV and dalpha.
    for b in 0..batch_count {
        for i in 0..SEQ_LEN {
            let row_base = (b * SEQ_LEN + i) * SEQ_LEN;
            let d_base = (b * SEQ_LEN + i) * D_MODEL;

            for j in 0..SEQ_LEN {
                let v_base = (b * SEQ_LEN + j) * D_MODEL;
                let mut dot = 0.0f32;
                for d in 0..D_MODEL {
                    dot += buf.dattn[d_base + d] * buf.v[v_base + d];
                }
                buf.dalpha[row_base + j] = dot;
            }

            for j in 0..SEQ_LEN {
                let a = buf.attn[row_base + j];
                let v_base = (b * SEQ_LEN + j) * D_MODEL;
                for d in 0..D_MODEL {
                    buf.dv[v_base + d] += a * buf.dattn[d_base + d];
                }
            }

            // Softmax grad per row.
            let mut sum = 0.0f32;
            for j in 0..SEQ_LEN {
                sum += buf.dalpha[row_base + j] * buf.attn[row_base + j];
            }
            for j in 0..SEQ_LEN {
                let a = buf.attn[row_base + j];
                buf.dscores[row_base + j] = a * (buf.dalpha[row_base + j] - sum);
            }
        }
    }

    // Scores -> dQ and dK.
    let inv_sqrt_d = 1.0f32 / (D_MODEL as f32).sqrt();
    for b in 0..batch_count {
        for i in 0..SEQ_LEN {
            let row_base = (b * SEQ_LEN + i) * SEQ_LEN;
            let q_base = (b * SEQ_LEN + i) * D_MODEL;
            for j in 0..SEQ_LEN {
                let k_base = (b * SEQ_LEN + j) * D_MODEL;
                let ds = buf.dscores[row_base + j] * inv_sqrt_d;
                for d in 0..D_MODEL {
                    buf.dq[q_base + d] += ds * buf.k[k_base + d];
                    buf.dk[k_base + d] += ds * buf.q[q_base + d];
                }
            }
        }
    }

    // Backprop through Q/K/V projections to tokens.
    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;

            for d_out in 0..D_MODEL {
                grads.b_q[d_out] += buf.dq[tok_base + d_out];
                grads.b_k[d_out] += buf.dk[tok_base + d_out];
                grads.b_v[d_out] += buf.dv[tok_base + d_out];
            }

            for d_in in 0..D_MODEL {
                let x = buf.tok[tok_base + d_in];
                let w_row = d_in * D_MODEL;
                let mut acc = 0.0f32;
                for d_out in 0..D_MODEL {
                    let dq = buf.dq[tok_base + d_out];
                    let dk = buf.dk[tok_base + d_out];
                    let dv = buf.dv[tok_base + d_out];
                    grads.w_q[w_row + d_out] += x * dq;
                    grads.w_k[w_row + d_out] += x * dk;
                    grads.w_v[w_row + d_out] += x * dv;
                    acc += dq * model.w_q[w_row + d_out];
                    acc += dk * model.w_k[w_row + d_out];
                    acc += dv * model.w_v[w_row + d_out];
                }
                buf.dtok[tok_base + d_in] = acc;
            }
        }
    }

    // ReLU backward (tok is post-ReLU).
    for i in 0..used_tok {
        if buf.tok[i] <= 0.0 {
            buf.dtok[i] = 0.0;
        }
    }

    // grad pos, grad b_patch, grad w_patch.
    for b in 0..batch_count {
        for t in 0..SEQ_LEN {
            let tok_base = (b * SEQ_LEN + t) * D_MODEL;
            let pos_base = t * D_MODEL;
            let patch_base = (b * SEQ_LEN + t) * PATCH_DIM;

            for d in 0..D_MODEL {
                let g = buf.dtok[tok_base + d];
                grads.pos[pos_base + d] += g;
                grads.b_patch[d] += g;
            }

            for j in 0..PATCH_DIM {
                let x = buf.patches[patch_base + j];
                let w_row = j * D_MODEL;
                for d in 0..D_MODEL {
                    grads.w_patch[w_row + d] += x * buf.dtok[tok_base + d];
                }
            }
        }
    }

    let _ = used_logits; // keep if code is adjusted later
}

fn apply_sgd(model: &mut AttnModel, grads: &Grads, lr: f32) {
    // Plain SGD update (no momentum, no weight decay).
    for i in 0..model.w_patch.len() {
        model.w_patch[i] -= lr * grads.w_patch[i];
    }
    for i in 0..model.b_patch.len() {
        model.b_patch[i] -= lr * grads.b_patch[i];
    }
    for i in 0..model.pos.len() {
        model.pos[i] -= lr * grads.pos[i];
    }
    for i in 0..model.w_q.len() {
        model.w_q[i] -= lr * grads.w_q[i];
    }
    for i in 0..model.b_q.len() {
        model.b_q[i] -= lr * grads.b_q[i];
    }
    for i in 0..model.w_k.len() {
        model.w_k[i] -= lr * grads.w_k[i];
    }
    for i in 0..model.b_k.len() {
        model.b_k[i] -= lr * grads.b_k[i];
    }
    for i in 0..model.w_v.len() {
        model.w_v[i] -= lr * grads.w_v[i];
    }
    for i in 0..model.b_v.len() {
        model.b_v[i] -= lr * grads.b_v[i];
    }
    for i in 0..model.w_ff1.len() {
        model.w_ff1[i] -= lr * grads.w_ff1[i];
    }
    for i in 0..model.b_ff1.len() {
        model.b_ff1[i] -= lr * grads.b_ff1[i];
    }
    for i in 0..model.w_ff2.len() {
        model.w_ff2[i] -= lr * grads.w_ff2[i];
    }
    for i in 0..model.b_ff2.len() {
        model.b_ff2[i] -= lr * grads.b_ff2[i];
    }
    for i in 0..model.w_cls.len() {
        model.w_cls[i] -= lr * grads.w_cls[i];
    }
    for i in 0..model.b_cls.len() {
        model.b_cls[i] -= lr * grads.b_cls[i];
    }
}

fn test_accuracy(model: &AttnModel, images: &[f32], labels: &[u8]) -> f32 {
    let n = labels.len();
    let mut correct = 0usize;

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut buf = BatchBuffers::new();

    for start in (0..n).step_by(BATCH_SIZE) {
        let batch_count = (n - start).min(BATCH_SIZE);
        let len = batch_count * NUM_INPUTS;
        let src_start = start * NUM_INPUTS;
        batch_inputs[..len].copy_from_slice(&images[src_start..src_start + len]);

        // Forward without loss/dlogits.
        let used_patches = batch_count * SEQ_LEN * PATCH_DIM;
        let used_tok = batch_count * SEQ_LEN * D_MODEL;
        let used_attn = batch_count * SEQ_LEN * SEQ_LEN;
        let used_ffn1 = batch_count * SEQ_LEN * FF_DIM;
        let used_pooled = batch_count * D_MODEL;
        let used_logits = batch_count * NUM_CLASSES;

        extract_patches(&batch_inputs, batch_count, &mut buf.patches[..used_patches]);

        // Token projection + ReLU.
        for i in 0..used_tok {
            buf.tok[i] = 0.0;
        }
        for b in 0..batch_count {
            for t in 0..SEQ_LEN {
                let patch_base = (b * SEQ_LEN + t) * PATCH_DIM;
                let tok_base = (b * SEQ_LEN + t) * D_MODEL;
                let pos_base = t * D_MODEL;

                for d in 0..D_MODEL {
                    let mut sum = model.b_patch[d] + model.pos[pos_base + d];
                    for j in 0..PATCH_DIM {
                        sum += buf.patches[patch_base + j] * model.w_patch[j * D_MODEL + d];
                    }
                    if sum < 0.0 {
                        sum = 0.0;
                    }
                    buf.tok[tok_base + d] = sum;
                }
            }
        }

        // Q/K/V projections.
        for i in 0..used_tok {
            buf.q[i] = 0.0;
            buf.k[i] = 0.0;
            buf.v[i] = 0.0;
        }
        for b in 0..batch_count {
            for t in 0..SEQ_LEN {
                let tok_base = (b * SEQ_LEN + t) * D_MODEL;
                for d_out in 0..D_MODEL {
                    let mut sum_q = model.b_q[d_out];
                    let mut sum_k = model.b_k[d_out];
                    let mut sum_v = model.b_v[d_out];
                    for d_in in 0..D_MODEL {
                        let x = buf.tok[tok_base + d_in];
                        sum_q += x * model.w_q[d_in * D_MODEL + d_out];
                        sum_k += x * model.w_k[d_in * D_MODEL + d_out];
                        sum_v += x * model.w_v[d_in * D_MODEL + d_out];
                    }
                    buf.q[tok_base + d_out] = sum_q;
                    buf.k[tok_base + d_out] = sum_k;
                    buf.v[tok_base + d_out] = sum_v;
                }
            }
        }

        // Self-attention.
        let inv_sqrt_d = 1.0f32 / (D_MODEL as f32).sqrt();
        for i in 0..used_attn {
            buf.attn[i] = 0.0;
        }
        for i in 0..used_tok {
            buf.attn_out[i] = 0.0;
        }

        for b in 0..batch_count {
            for i in 0..SEQ_LEN {
                let row_base = (b * SEQ_LEN + i) * SEQ_LEN;
                let q_base = (b * SEQ_LEN + i) * D_MODEL;

                for j in 0..SEQ_LEN {
                    let k_base = (b * SEQ_LEN + j) * D_MODEL;
                    let mut score = 0.0f32;
                    for d in 0..D_MODEL {
                        score += buf.q[q_base + d] * buf.k[k_base + d];
                    }
                    buf.attn[row_base + j] = score * inv_sqrt_d;
                }
                softmax_inplace(&mut buf.attn[row_base..row_base + SEQ_LEN]);

                let out_base = (b * SEQ_LEN + i) * D_MODEL;
                for j in 0..SEQ_LEN {
                    let a = buf.attn[row_base + j];
                    let v_base = (b * SEQ_LEN + j) * D_MODEL;
                    for d in 0..D_MODEL {
                        buf.attn_out[out_base + d] += a * buf.v[v_base + d];
                    }
                }
            }
        }

        // Feed-forward MLP.
        for i in 0..used_ffn1 {
            buf.ffn1[i] = 0.0;
        }
        for i in 0..used_tok {
            buf.ffn2[i] = 0.0;
        }
        for b in 0..batch_count {
            for t in 0..SEQ_LEN {
                let attn_base = (b * SEQ_LEN + t) * D_MODEL;
                let ffn1_base = (b * SEQ_LEN + t) * FF_DIM;
                let ffn2_base = (b * SEQ_LEN + t) * D_MODEL;

                for h in 0..FF_DIM {
                    let mut sum = model.b_ff1[h];
                    for d in 0..D_MODEL {
                        sum += buf.attn_out[attn_base + d] * model.w_ff1[d * FF_DIM + h];
                    }
                    buf.ffn1[ffn1_base + h] = if sum > 0.0 { sum } else { 0.0 };
                }

                for d in 0..D_MODEL {
                    let mut sum = model.b_ff2[d];
                    for h in 0..FF_DIM {
                        sum += buf.ffn1[ffn1_base + h] * model.w_ff2[h * D_MODEL + d];
                    }
                    buf.ffn2[ffn2_base + d] = sum;
                }
            }
        }

        // Mean pool tokens.
        for i in 0..used_pooled {
            buf.pooled[i] = 0.0;
        }
        let inv_seq = 1.0f32 / SEQ_LEN as f32;
        for b in 0..batch_count {
            let pooled_base = b * D_MODEL;
            for t in 0..SEQ_LEN {
                let tok_base = (b * SEQ_LEN + t) * D_MODEL;
                for d in 0..D_MODEL {
                    buf.pooled[pooled_base + d] += buf.ffn2[tok_base + d] * inv_seq;
                }
            }
        }

        // Logits.
        for i in 0..used_logits {
            buf.logits[i] = 0.0;
        }
        for b in 0..batch_count {
            let pooled_base = b * D_MODEL;
            let log_base = b * NUM_CLASSES;
            for c in 0..NUM_CLASSES {
                let mut sum = model.b_cls[c];
                for d in 0..D_MODEL {
                    sum += buf.pooled[pooled_base + d] * model.w_cls[d * NUM_CLASSES + c];
                }
                buf.logits[log_base + c] = sum;
            }
        }

        // Argmax prediction.
        for b in 0..batch_count {
            let base = b * NUM_CLASSES;
            let mut best = buf.logits[base];
            let mut arg = 0usize;
            for c in 1..NUM_CLASSES {
                let v = buf.logits[base + c];
                if v > best {
                    best = v;
                    arg = c;
                }
            }
            if arg as u8 == labels[start + b] {
                correct += 1;
            }
        }
    }

    100.0 * (correct as f32) / (n as f32)
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

    // Training log file.
    fs::create_dir_all("./logs").ok();
    let log_file = File::create("./logs/training_loss_attention_mnist.txt").unwrap_or_else(|_| {
        eprintln!("Could not create logs/training_loss_attention_mnist.txt");
        process::exit(1);
    });
    let mut log = BufWriter::new(log_file);

    let mut rng = SimpleRng::new(1);
    rng.reseed_from_time();

    let mut model = init_model(&mut rng);

    // Shuffled indices for mini-batch sampling.
    let mut indices: Vec<usize> = (0..train_n).collect();

    // Training buffers (reused each batch to avoid allocations).
    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut batch_labels = vec![0u8; BATCH_SIZE];
    let mut buf = BatchBuffers::new();
    let mut grads = Grads::new();

    println!(
        "Training self-attention: D={} ff={} patch={} seq={} epochs={} batch={} lr={}",
        D_MODEL, FF_DIM, PATCH, SEQ_LEN, EPOCHS, BATCH_SIZE, LEARNING_RATE
    );

    for epoch in 0..EPOCHS {
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(BATCH_SIZE) {
            let batch_count = (train_n - batch_start).min(BATCH_SIZE);

            gather_batch(
                &train_images,
                &train_labels,
                &indices,
                batch_start,
                batch_count,
                &mut batch_inputs,
                &mut batch_labels,
            );

            // Forward pass + loss.
            let batch_loss = forward_batch(
                &model,
                &batch_inputs,
                &batch_labels,
                batch_count,
                &mut buf,
            );
            total_loss += batch_loss;

            // Backward pass + SGD update.
            backward_batch(&model, batch_count, &mut buf, &mut grads);
            apply_sgd(&mut model, &grads, LEARNING_RATE);
        }

        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;
        let acc = test_accuracy(&model, &test_images, &test_labels);

        println!(
            "Epoch {} | loss={:.6} | time={:.3}s | test_acc={:.2}%",
            epoch + 1,
            avg_loss,
            secs,
            acc
        );
        writeln!(log, "{},{},{}", epoch + 1, avg_loss, secs).ok();
    }

    println!("Final test accuracy: {:.2}%", test_accuracy(&model, &test_images, &test_labels));
}
