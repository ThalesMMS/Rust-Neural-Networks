use super::*;

/// Serialize the model's parameters to a file as raw little-endian `f32` values in a fixed order.
///
/// The parameters are written in the following sequence:
/// `w_patch`, `b_patch`, `pos`, `w_q`, `b_q`, `w_k`, `b_k`, `w_v`, `b_v`,
/// `w_ff1`, `b_ff1`, `w_ff2`, `b_ff2`, `w_cls`, `b_cls`.
///
/// On failure to create or write the file the process exits with status code `1`.
///
/// # Examples
///
/// ```no_run
/// // Writes model parameters to "model.bin"
/// let model = load_or_build_attn_model(); // placeholder for obtaining an AttnModel
/// save_model(&model, "model.bin");
/// ```
pub(crate) fn save_model(model: &AttnModel, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {} for writing model", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    let write_f32 = |writer: &mut BufWriter<File>, value: f32| {
        writer.write_all(&value.to_le_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };

    // Write all model parameters in order
    for &value in &model.w_patch {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_patch {
        write_f32(&mut writer, value);
    }
    for &value in &model.pos {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_q {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_q {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_k {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_k {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_v {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_v {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_ff1 {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_ff1 {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_ff2 {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_ff2 {
        write_f32(&mut writer, value);
    }
    for &value in &model.w_cls {
        write_f32(&mut writer, value);
    }
    for &value in &model.b_cls {
        write_f32(&mut writer, value);
    }

    writer.flush().unwrap_or_else(|_| {
        eprintln!("Failed flushing model data");
        process::exit(1);
    });
    writer.get_ref().sync_all().unwrap_or_else(|_| {
        eprintln!("Failed syncing model data");
        process::exit(1);
    });

    println!("Model saved to {}", filename);
}

// Shared forward inference logic (up to logits/probs) without loss computation.
// Populates: patches, tok, q/k/v, attn, ffn, pooled, logits, probs.
/// Runs a full forward pass of the attention classifier for a batch and writes intermediate and final outputs into `buf`.
///
/// This computes patch extraction, patch-to-token projection with positional bias and ReLU, Q/K/V projections, scaled dot-product self-attention with softmax, a two-layer position-wise feed-forward network (ReLU in the hidden layer), mean pooling across tokens, and final classifier logits and probabilities (softmax). Results are stored in the provided `BatchBuffers` (including `logits` and `probs`).
///
/// - `model`: model parameters used for all linear projections, biases, positional embeddings, and classifier weights.
/// - `batch_inputs`: flattened input images for the batch (length = batch_count * NUM_INPUTS).
/// - `batch_count`: number of elements in the batch (may be less than BATCH_SIZE for the last batch).
/// - `buf`: preallocated buffers where intermediate tensors, logits, and probabilities will be written.
///
/// # Examples
///
/// ```
/// // Prepare model, inputs and buffers (constructors omitted for brevity).
/// let model: AttnModel = load_some_model();
/// let batch_count = 4usize;
/// let inputs: Vec<f32> = vec![0.0; batch_count * NUM_INPUTS];
/// let mut buf = BatchBuffers::new();
///
/// forward_inference(&model, &inputs, batch_count, &mut buf);
/// // probabilities are available in buf.probs
/// assert_eq!(buf.probs.len(), batch_count * NUM_CLASSES);
/// ```
pub(crate) fn forward_inference(
    model: &AttnModel,
    batch_inputs: &[f32],
    batch_count: usize,
    buf: &mut BatchBuffers,
) {
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

    // Self-attention: Scaled dot-product attention (Transformer-style).
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

            softmax_rows(&mut buf.attn[row_base..row_base + SEQ_LEN], 1, SEQ_LEN);

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

    // Feed-forward network per token (position-wise MLP).
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

    // Mean pooling over tokens to get image-level representation.
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

    softmax_rows(&mut buf.probs[..used_logits], batch_count, NUM_CLASSES);
}

/// Computes classification accuracy of `model` on the provided images and labels as a percentage.
///
/// Processes the dataset in batches and performs inference (no backprop). For each example, the
/// predicted class is the argmax over model logits and is compared against `labels`.
///
/// # Examples
///
/// ```
/// let acc = test_accuracy(&model, &images, &labels);
/// println!("Test accuracy: {:.2}%", acc);
/// ```
///
/// # Returns
///
/// Accuracy as a percentage in the range 0.0 to 100.0.
pub(crate) fn test_accuracy(model: &AttnModel, images: &[f32], labels: &[u8]) -> f32 {
    let n = labels.len();
    let mut correct = 0usize;

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut buf = BatchBuffers::new();

    for start in (0..n).step_by(BATCH_SIZE) {
        let batch_count = (n - start).min(BATCH_SIZE);
        let len = batch_count * NUM_INPUTS;
        let src_start = start * NUM_INPUTS;
        batch_inputs[..len].copy_from_slice(&images[src_start..src_start + len]);

        forward_inference(model, &batch_inputs, batch_count, &mut buf);

        // Argmax output.
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

// Train model with specified configuration and return final accuracy and loss progression.
