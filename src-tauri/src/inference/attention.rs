//! Inference-only port of `mnist_attention_pool`'s checkpoint format (see
//! `src/bin/mnist_attention_pool/io.rs`): a raw ordered dump of f32 arrays
//! (no dimension header, sizes are fixed by the architecture constants
//! below). Forward pass mirrors `forward_inference` in the same file:
//! patch-embed (+pos, ReLU) -> single-head scaled dot-product self-attention
//! -> per-token feed-forward -> mean pool over tokens -> classifier -> softmax.
//! There are no residual connections or layer norm in the original model —
//! this is a faithful port, not a simplification.

#![allow(clippy::needless_range_loop)] // multi-array indexed math, mirrors the original algorithm shape

use super::{read_f32_vec, to_prediction, Prediction};
use rust_neural_networks::utils::activations::softmax_rows;
use std::fs;

const IMG_W: usize = 28;
const NUM_INPUTS: usize = 784;
const NUM_CLASSES: usize = 10;
const PATCH: usize = 4;
const GRID: usize = 7; // IMG_H / PATCH
const SEQ_LEN: usize = GRID * GRID; // 49
const PATCH_DIM: usize = PATCH * PATCH; // 16
const D_MODEL: usize = 64;
const FF_DIM: usize = 128;

struct AttnWeights {
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

fn load_weights(checkpoint_path: &str) -> Result<AttnWeights, String> {
    let path = crate::paths::resolve_relative(checkpoint_path)?;
    let bytes = fs::read(&path).map_err(|e| format!("failed to read {checkpoint_path}: {e}"))?;
    let mut offset = 0usize;
    let w_patch = read_f32_vec(&bytes, &mut offset, PATCH_DIM * D_MODEL)?;
    let b_patch = read_f32_vec(&bytes, &mut offset, D_MODEL)?;
    let pos = read_f32_vec(&bytes, &mut offset, SEQ_LEN * D_MODEL)?;
    let w_q = read_f32_vec(&bytes, &mut offset, D_MODEL * D_MODEL)?;
    let b_q = read_f32_vec(&bytes, &mut offset, D_MODEL)?;
    let w_k = read_f32_vec(&bytes, &mut offset, D_MODEL * D_MODEL)?;
    let b_k = read_f32_vec(&bytes, &mut offset, D_MODEL)?;
    let w_v = read_f32_vec(&bytes, &mut offset, D_MODEL * D_MODEL)?;
    let b_v = read_f32_vec(&bytes, &mut offset, D_MODEL)?;
    let w_ff1 = read_f32_vec(&bytes, &mut offset, D_MODEL * FF_DIM)?;
    let b_ff1 = read_f32_vec(&bytes, &mut offset, FF_DIM)?;
    let w_ff2 = read_f32_vec(&bytes, &mut offset, FF_DIM * D_MODEL)?;
    let b_ff2 = read_f32_vec(&bytes, &mut offset, D_MODEL)?;
    let w_cls = read_f32_vec(&bytes, &mut offset, D_MODEL * NUM_CLASSES)?;
    let b_cls = read_f32_vec(&bytes, &mut offset, NUM_CLASSES)?;
    Ok(AttnWeights {
        w_patch, b_patch, pos, w_q, b_q, w_k, b_k, w_v, b_v, w_ff1, b_ff1, w_ff2, b_ff2, w_cls, b_cls,
    })
}

fn extract_patches(pixels: &[f32]) -> Vec<f32> {
    let mut patches = vec![0.0f32; SEQ_LEN * PATCH_DIM];
    for py in 0..GRID {
        for px in 0..GRID {
            let t = py * GRID + px;
            let patch_base = t * PATCH_DIM;
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let iy = py * PATCH + dy;
                    let ix = px * PATCH + dx;
                    let j = dy * PATCH + dx;
                    patches[patch_base + j] = pixels[iy * IMG_W + ix];
                }
            }
        }
    }
    patches
}

#[tauri::command]
pub fn predict_attention(checkpoint_path: String, pixels: Vec<f32>) -> Result<Prediction, String> {
    if pixels.len() != NUM_INPUTS {
        return Err(format!("expected {NUM_INPUTS} pixels, got {}", pixels.len()));
    }
    let m = load_weights(&checkpoint_path)?;

    let patches = extract_patches(&pixels);

    // token = ReLU(patch . W_patch + b_patch + pos[t])
    let mut tok = vec![0.0f32; SEQ_LEN * D_MODEL];
    for t in 0..SEQ_LEN {
        let patch_base = t * PATCH_DIM;
        let tok_base = t * D_MODEL;
        let pos_base = t * D_MODEL;
        for d in 0..D_MODEL {
            let mut sum = m.b_patch[d] + m.pos[pos_base + d];
            for j in 0..PATCH_DIM {
                sum += patches[patch_base + j] * m.w_patch[j * D_MODEL + d];
            }
            tok[tok_base + d] = sum.max(0.0);
        }
    }

    // Q/K/V projections.
    let mut q = vec![0.0f32; SEQ_LEN * D_MODEL];
    let mut k = vec![0.0f32; SEQ_LEN * D_MODEL];
    let mut v = vec![0.0f32; SEQ_LEN * D_MODEL];
    for t in 0..SEQ_LEN {
        let tok_base = t * D_MODEL;
        for d_out in 0..D_MODEL {
            let mut sq = m.b_q[d_out];
            let mut sk = m.b_k[d_out];
            let mut sv = m.b_v[d_out];
            for d_in in 0..D_MODEL {
                let x = tok[tok_base + d_in];
                sq += x * m.w_q[d_in * D_MODEL + d_out];
                sk += x * m.w_k[d_in * D_MODEL + d_out];
                sv += x * m.w_v[d_in * D_MODEL + d_out];
            }
            q[tok_base + d_out] = sq;
            k[tok_base + d_out] = sk;
            v[tok_base + d_out] = sv;
        }
    }

    // Scaled dot-product self-attention (single head).
    let inv_sqrt_d = 1.0f32 / (D_MODEL as f32).sqrt();
    let mut attn_out = vec![0.0f32; SEQ_LEN * D_MODEL];
    for i in 0..SEQ_LEN {
        let q_base = i * D_MODEL;
        let mut scores = vec![0.0f32; SEQ_LEN];
        for j in 0..SEQ_LEN {
            let k_base = j * D_MODEL;
            let mut s = 0.0f32;
            for d in 0..D_MODEL {
                s += q[q_base + d] * k[k_base + d];
            }
            scores[j] = s * inv_sqrt_d;
        }
        softmax_rows(&mut scores, 1, SEQ_LEN);

        let out_base = i * D_MODEL;
        for j in 0..SEQ_LEN {
            let a = scores[j];
            let v_base = j * D_MODEL;
            for d in 0..D_MODEL {
                attn_out[out_base + d] += a * v[v_base + d];
            }
        }
    }

    // Per-token feed-forward, then mean pool over tokens.
    let mut pooled = vec![0.0f32; D_MODEL];
    let inv_seq = 1.0f32 / SEQ_LEN as f32;
    for t in 0..SEQ_LEN {
        let attn_base = t * D_MODEL;
        let mut ffn1 = vec![0.0f32; FF_DIM];
        for h in 0..FF_DIM {
            let mut sum = m.b_ff1[h];
            for d in 0..D_MODEL {
                sum += attn_out[attn_base + d] * m.w_ff1[d * FF_DIM + h];
            }
            ffn1[h] = sum.max(0.0);
        }
        for d in 0..D_MODEL {
            let mut sum = m.b_ff2[d];
            for h in 0..FF_DIM {
                sum += ffn1[h] * m.w_ff2[h * D_MODEL + d];
            }
            pooled[d] += sum * inv_seq;
        }
    }

    // Classifier + softmax.
    let mut logits = vec![0.0f32; NUM_CLASSES];
    for c in 0..NUM_CLASSES {
        let mut sum = m.b_cls[c];
        for d in 0..D_MODEL {
            sum += pooled[d] * m.w_cls[d * NUM_CLASSES + c];
        }
        logits[c] = sum;
    }
    softmax_rows(&mut logits, 1, NUM_CLASSES);

    Ok(to_prediction(logits))
}
