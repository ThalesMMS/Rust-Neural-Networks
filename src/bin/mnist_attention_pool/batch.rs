use super::*;

/// Extracts PATCH x PATCH patches from a batch of flattened images and writes them into a token-major buffer.
///
/// The input `batch_inputs` is expected to contain `batch_count` images laid out row-major and flattened
/// (each image length = NUM_INPUTS, width = IMG_W). The output `patches` buffer is filled so that for each
/// image `b` and each token index `t` (row-major patch grid: py then px) the corresponding patch occupies
/// `(b * SEQ_LEN + t) * PATCH_DIM .. (b * SEQ_LEN + t + 1) * PATCH_DIM`. Within each patch the pixels are
/// stored row-major (dy then dx).
///
/// - Does not modify image order or perform any normalization; it only copies pixel values.
/// - Requires `batch_inputs.len() >= batch_count * NUM_INPUTS` and
///   `patches.len() >= batch_count * SEQ_LEN * PATCH_DIM`.
///
/// # Examples
///
/// ```ignore
/// // Prepare a single image where the top-left pixel is 1.0 and the rest are 0.0.
/// let batch_count = 1;
/// let mut batch_inputs = vec![0.0_f32; batch_count * NUM_INPUTS];
/// batch_inputs[0] = 1.0; // pixel at (0,0)
///
/// let mut patches = vec![0.0_f32; batch_count * SEQ_LEN * PATCH_DIM];
/// extract_patches(&batch_inputs, batch_count, &mut patches);
///
/// // The first patch (top-left patch) first element corresponds to image (0,0).
/// assert_eq!(patches[0], 1.0_f32);
/// ```
pub(crate) fn extract_patches(batch_inputs: &[f32], batch_count: usize, patches: &mut [f32]) {
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
/// Runs a forward pass of the attention model for the given batch and populates
/// the provided buffers with forward activations, predicted probabilities, and
/// the gradient of the loss with respect to the logits.
///
/// The `buf` argument is written with all forward activations, `probs`, and
/// `dlogits` (where `dlogits` contains the softmax cross-entropy gradients
/// scaled by 1 / batch_count).
///
/// # Returns
///
/// The total cross-entropy loss summed over the batch (`Σ -ln(p_true_class)`).
///
/// # Examples
///
/// ```
/// // Assume `model`, `inputs`, `labels`, `batch_size`, and `mut buf` are prepared.
/// let loss = forward_batch(&model, &inputs, &labels, batch_size, &mut buf);
/// assert!(loss >= 0.0);
/// ```
pub(crate) fn forward_batch(
    model: &AttnModel,
    batch_inputs: &[f32],
    batch_labels: &[u8],
    batch_count: usize,
    buf: &mut BatchBuffers,
) -> f32 {
    forward_inference(model, batch_inputs, batch_count, buf);

    // Loss + dlogits (softmax cross-entropy).
    let mut total_loss = 0.0f32;
    let eps = 1e-9f32;
    let scale = 1.0f32 / batch_count as f32;

    for (b, &label) in batch_labels.iter().enumerate().take(batch_count) {
        let y = label as usize;
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
/// Backpropagates a batch through the model and accumulates parameter gradients.
///
/// Zeros gradient buffers, backpropagates from classification logits through mean
/// pooling into the two-layer feed-forward network and the self-attention block,
/// computes gradients for Q/K/V projections and token/positional/patch parameters,
/// applies ReLU backward masks for token and FFN activations, and accumulates all
/// parameter gradients into `grads`.
///
/// # Examples
///
/// ```
/// // Assume `model`, `buf`, and `grads` are previously-initialized compatible values
/// // and `batch_count` is the current batch size:
/// // backward_batch(&model, batch_count, &mut buf, &mut grads);
/// ```
pub(crate) fn backward_batch(
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

/// Applies plain stochastic gradient descent updates to all model parameters using the provided gradients and learning rate.
///
/// Updates each parameter in `model` by subtracting `lr * grad` from the corresponding entry in `grads`. This uses simple SGD with no momentum, weight decay, or other optimizer features.
///
/// # Examples
///
/// ```no_run
/// # // Assume `model` and `grads` are previously created and compatible `AttnModel` / `Grads`
/// # let mut model = /* ... */; let grads = /* ... */; let lr = 1e-3_f32;
/// apply_sgd(&mut model, &grads, lr);
/// ```
pub(crate) fn apply_sgd(model: &mut AttnModel, grads: &Grads, lr: f32) {
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

// Compute the L2 (Euclidean) norm of a slice of f32 values.
// Used to measure gradient magnitudes for monitoring training health.
/// Computes the Euclidean (L2) norm of the elements in `v`.
///
/// # Returns
///
/// The square root of the sum of squares of all elements in `v`.
///
/// # Examples
///
/// ```
/// let v = [3.0_f32, 4.0];
/// assert_eq!(l2_norm(&v), 5.0);
/// ```
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// Save the attention model in binary (little-endian f32).
