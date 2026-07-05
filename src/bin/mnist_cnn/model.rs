use super::*;

pub(crate) struct Cnn {
    pub(crate) conv_layer: Conv2DLayer,
    pub(crate) fc_layer: DenseLayer,
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
pub(crate) fn init_cnn(rng: &mut SimpleRng) -> Cnn {
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
/// Run the convolutional layer over the provided input batch and apply ReLU to the outputs in place.
///
/// The `conv_out` buffer is written with the layer's activated outputs for the entire batch; values are clamped to be >= 0 by ReLU.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(123);
/// let mut model = init_cnn(&mut rng);
/// let batch_size = 4;
/// let num_inputs = 28 * 28; // example for MNIST-like images
/// let conv_out_per_sample = 8 * 28 * 28; // CONV_OUT * IMG_H * IMG_W
/// let mut inputs = vec![0f32; batch_size * num_inputs];
/// let mut conv_out_buf = vec![0f32; batch_size * conv_out_per_sample];
/// conv_forward_relu(&mut model, batch_size, &inputs, &mut conv_out_buf);
/// assert!(conv_out_buf.iter().all(|&v| v >= 0.0));
/// ```
pub(crate) fn conv_forward_relu(
    model: &mut Cnn,
    batch_size: usize,
    input: &[f32],
    conv_out: &mut [f32],
) {
    // Use Conv2DLayer for forward pass
    model.conv_layer.forward(input, conv_out, batch_size);

    // Apply ReLU activation
    relu_inplace(conv_out);
}

// MaxPool 2x2 stride 2.
// conv_act: [batch * C * 28 * 28] (post-ReLU)
// pool_out: [batch * C * 14 * 14]
// pool_idx: [batch * C * 14 * 14], stores argmax 0..3 (dy*2+dx)
/// Performs spatial max pooling over each channel of convolution activations and records argmax indices for backprop.
///
/// For each example `b` and channel `c`, the activation map is partitioned into POOL_H×POOL_W pooling cells. For each pooling cell at pooled coordinates `(py, px)` this function:
/// - scans the POOL×POOL window in `conv_act`,
/// - writes the maximum value into `pool_out` at the corresponding pooled position,
/// - writes the argmax within the window into `pool_idx` as `dy*POOL + dx` (value in `0..=(POOL*POOL - 1)`).
///
/// # Examples
///
/// ```
/// // Prepare a single example with one output channel and small image for illustration.
/// let batch = 1;
/// let mut conv_act = vec![0.0f32; CONV_OUT * IMG_H * IMG_W * batch];
/// // fill conv_act with some values...
/// let mut pool_out = vec![0.0f32; CONV_OUT * POOL_H * POOL_W * batch];
/// let mut pool_idx = vec![0u8; CONV_OUT * POOL_H * POOL_W * batch];
///
/// maxpool_forward(batch, &conv_act, &mut pool_out, &mut pool_idx);
/// ```
pub(crate) fn maxpool_forward(
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
pub(crate) fn fc_forward(model: &mut Cnn, batch: usize, x: &[f32], logits: &mut [f32]) {
    // Use DenseLayer for forward pass
    model.fc_layer.forward(x, logits, batch);
}

// FC backward: compute gradW, gradB and dX.
/// Performs the backward pass through the model's fully connected layer, accumulating parameter gradients in the layer and writing gradients with respect to the layer inputs into `d_x`.
///
/// `batch` is the number of examples in the minibatch; `x` is the input buffer (length `batch * FC_IN`); `delta` is the gradient w.r.t. the layer outputs (length `batch * NUM_CLASSES`); `d_x` receives gradients w.r.t. `x` (length `batch * FC_IN`).
///
/// # Examples
///
/// ```no_run
/// let mut model = init_cnn(&mut SimpleRng::new(123));
/// let batch = 2;
/// let mut x = vec![0f32; batch * FC_IN];
/// let delta = vec![0f32; batch * NUM_CLASSES];
/// let mut d_x = vec![0f32; batch * FC_IN];
/// fc_backward(&mut model, batch, &x, &delta, &mut d_x);
/// ```
pub(crate) fn fc_backward(
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
/// Scatters max-pooled gradients back into the convolution activations and applies ReLU backprop.
///
/// This performs two steps:
/// 1. Scatters each pooled gradient value from `pool_grad` into `conv_grad` at the argmax positions recorded in `pool_idx` (per-channel, per-example).
/// 2. Zeros entries in `conv_grad` where the corresponding post-ReLU activation in `conv_act` is less than or equal to 0.0.
///
/// Parameters:
/// - `batch`: number of examples in the batch.
/// - `conv_act`: flattened post-ReLU convolution activations for the usable region (length >= batch * CONV_OUT * IMG_H * IMG_W).
/// - `pool_grad`: flattened pooled gradients (length >= batch * CONV_OUT * POOL_H * POOL_W).
/// - `pool_idx`: argmax indices (0..POOL*POOL) recorded during pooling (same length and layout as `pool_grad`).
/// - `conv_grad`: mutable flattened convolution-gradient buffer (length >= batch * CONV_OUT * IMG_H * IMG_W); will be zeroed for the usable region and then receive scattered pooled gradients, followed by ReLU masking.
///
/// # Examples
///
/// ```rust
/// // Minimal usage sketch (constants and layer sizes come from the surrounding crate)
/// let batch = 1;
/// let mut conv_act = vec![1.0f32; CONV_OUT * IMG_H * IMG_W]; // all positive so ReLU passes
/// let pool_grad = vec![2.0f32; CONV_OUT * POOL_H * POOL_W];
/// let pool_idx = vec![0u8; CONV_OUT * POOL_H * POOL_W]; // always pick top-left
/// let mut conv_grad = vec![0.0f32; CONV_OUT * IMG_H * IMG_W];
///
/// maxpool_backward_relu(batch, &conv_act, &pool_grad, &pool_idx, &mut conv_grad);
///
/// // With pool_idx == 0 the scattered gradients should be present at the corresponding top-left positions.
/// assert!(conv_grad.iter().any(|&v| v != 0.0));
/// ```
pub(crate) fn maxpool_backward_relu(
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
pub(crate) fn conv_backward(
    model: &mut Cnn,
    batch: usize,
    input: &[f32],           // [batch*784]
    conv_grad: &[f32],       // [batch*C*28*28]
    _grad_input: &mut [f32], // unused (first layer)
) {
    let mut grad_input_scratch = vec![0.0f32; batch * NUM_INPUTS];
    // Preserve this binary's averaged-gradient behavior when using the shared Conv2D layer.
    let scaled_conv_grad = if batch > 1 {
        let scale = 1.0f32 / batch as f32;
        Some(conv_grad.iter().map(|&g| g * scale).collect::<Vec<_>>())
    } else {
        None
    };
    let conv_grad_for_backward = scaled_conv_grad.as_deref().unwrap_or(conv_grad);

    model.conv_layer.backward(
        input,
        conv_grad_for_backward,
        &mut grad_input_scratch,
        batch,
    );
}

/// Compute model classification accuracy on the dataset as a percentage.
///
/// Processes the inputs in batches through the model's forward pipeline and compares
/// each sample's predicted class (logits argmax) to the provided label.
///
/// # Returns
///
/// Accuracy as a percentage between `0.0` and `100.0`.
///
/// # Examples
///
/// ```
/// // assume `rng`, `model`, `images`, and `labels` are prepared appropriately
/// // let mut rng = SimpleRng::new(...);
/// // let mut model = init_cnn(&mut rng);
/// // let images: Vec<f32> = ...; // length = num_samples * NUM_INPUTS
/// // let labels: Vec<u8> = ...;  // length = num_samples
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

        conv_forward_relu(
            model,
            batch,
            &batch_inputs[..len],
            &mut conv_out[..batch * CONV_OUT * IMG_H * IMG_W],
        );
        maxpool_forward(batch, &conv_out, &mut pool_out, &mut pool_idx);
        fc_forward(
            model,
            batch,
            &pool_out[..batch * FC_IN],
            &mut logits[..batch * NUM_CLASSES],
        );

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
