use crate::autograd::tape::Op;
use crate::autograd::tensor::Tensor;

// ---------------------------------------------------------------------------
// backward_op — reverse-mode gradient dispatcher
// ---------------------------------------------------------------------------

/// Dispatches the backward gradient computation for a single operation node.
///
/// Called by [`Tensor::backward`] once per node in the reverse topological
/// traversal. For each operation, this function:
/// 1. Reads the upstream gradient (`upstream`) flowing from the output of the op.
/// 2. Applies the chain rule to compute the gradient w.r.t. each input tensor.
/// 3. Calls [`Tensor::accumulate_grad`] on inputs that have `requires_grad = true`.
///
/// # Arguments
///
/// * `op`       - The operation that produced the output tensor.
/// * `inputs`   - Input tensors consumed by `op` (same order as the forward pass).
/// * `upstream` - Gradient of the loss w.r.t. the output of this operation.
///
/// # Gradient Rules
///
/// | Op           | grad for input `a`                                   | grad for input `b`                     |
/// |--------------|------------------------------------------------------|----------------------------------------|
/// | Add (same)   | `upstream`                                           | `upstream`                             |
/// | Add (bias)   | `upstream`                                           | `sum(upstream, axis=batch)`            |
/// | Sub          | `upstream`                                           | `-upstream`                            |
/// | Mul          | `upstream * b`                                       | `upstream * a`                         |
/// | MatMul(m,k,n)| `upstream @ b.T` shape `(m, k)`                      | `a.T @ upstream` shape `(k, n)`        |
/// | ReLU         | `upstream * (a > 0)`                                 | —                                      |
/// | Sigmoid      | `upstream * s * (1 - s)` where `s = inputs[1]`       | —                                      |
/// | Tanh         | `upstream * (1 - t²)` where `t = inputs[1]`          | —                                      |
/// | SoftmaxCE    | `upstream * (softmax - one_hot(label)) / batch`      | —                                      |
/// | MSE          | `upstream * 2 * (pred - target) / n`                 | `upstream * -2 * (pred - target) / n`  |
/// | Sum          | broadcast `upstream[0]` to all elements              | —                                      |
/// | Mean         | broadcast `upstream[0] / n` to all elements          | —                                      |
pub fn backward_op(op: &Op, inputs: &[Tensor], upstream: &[f32]) {
    match op {
        Op::Add => {
            let a = &inputs[0];
            let b = &inputs[1];

            if a.requires_grad() {
                a.accumulate_grad(upstream);
            }

            if b.requires_grad() {
                let b_n = b.numel();
                let out_n = upstream.len();
                if b_n == out_n {
                    // Regular element-wise add: same shape, pass gradient directly.
                    b.accumulate_grad(upstream);
                } else {
                    // Bias broadcast: bias shape is (1, features), upstream is (batch, features).
                    // Sum gradient along the batch dimension.
                    let features = b.shape().1;
                    let batch = out_n / features;
                    let mut grad_b = vec![0.0f32; features];
                    for r in 0..batch {
                        for c in 0..features {
                            grad_b[c] += upstream[r * features + c];
                        }
                    }
                    b.accumulate_grad(&grad_b);
                }
            }
        }

        Op::Sub => {
            let a = &inputs[0];
            let b = &inputs[1];

            if a.requires_grad() {
                a.accumulate_grad(upstream);
            }

            if b.requires_grad() {
                let grad_b: Vec<f32> = upstream.iter().map(|&g| -g).collect();
                b.accumulate_grad(&grad_b);
            }
        }

        Op::Mul => {
            let a = &inputs[0];
            let b = &inputs[1];
            let data_b = b.data();
            let data_a = a.data();

            if a.requires_grad() {
                // grad_a[i] = upstream[i] * b[i]
                let grad_a: Vec<f32> = upstream
                    .iter()
                    .zip(data_b.iter())
                    .map(|(&g, &bv)| g * bv)
                    .collect();
                a.accumulate_grad(&grad_a);
            }

            if b.requires_grad() {
                // grad_b[i] = upstream[i] * a[i]
                let grad_b: Vec<f32> = upstream
                    .iter()
                    .zip(data_a.iter())
                    .map(|(&g, &av)| g * av)
                    .collect();
                b.accumulate_grad(&grad_b);
            }
        }

        Op::MatMul { m, k, n } => {
            let a = &inputs[0]; // shape (m, k)
            let b = &inputs[1]; // shape (k, n)
                                // upstream has shape (m, n)

            if a.requires_grad() {
                // grad_a = upstream @ b.T  →  shape (m, k)
                let data_b = b.data();
                let mut grad_a = vec![0.0f32; m * k];
                for i in 0..*m {
                    for p in 0..*k {
                        let mut sum = 0.0f32;
                        for j in 0..*n {
                            sum += upstream[i * n + j] * data_b[p * n + j];
                        }
                        grad_a[i * k + p] = sum;
                    }
                }
                a.accumulate_grad(&grad_a);
            }

            if b.requires_grad() {
                // grad_b = a.T @ upstream  →  shape (k, n)
                let data_a = a.data();
                let mut grad_b = vec![0.0f32; k * n];
                for p in 0..*k {
                    for j in 0..*n {
                        let mut sum = 0.0f32;
                        for i in 0..*m {
                            sum += data_a[i * k + p] * upstream[i * n + j];
                        }
                        grad_b[p * n + j] = sum;
                    }
                }
                b.accumulate_grad(&grad_b);
            }
        }

        Op::ReLU => {
            let a = &inputs[0]; // original input before ReLU
            if a.requires_grad() {
                let data_a = a.data();
                // grad_a[i] = upstream[i] if a[i] > 0 else 0
                let grad_a: Vec<f32> = upstream
                    .iter()
                    .zip(data_a.iter())
                    .map(|(&g, &av)| if av > 0.0 { g } else { 0.0 })
                    .collect();
                a.accumulate_grad(&grad_a);
            }
        }

        Op::Sigmoid => {
            // inputs[0] = original input `a` (requires_grad)
            // inputs[1] = cached sigmoid output `s = sigmoid(a)` (no grad)
            let a = &inputs[0];
            if a.requires_grad() {
                let s = inputs[1].data();
                // grad_a[i] = upstream[i] * s[i] * (1 - s[i])
                let grad_a: Vec<f32> = upstream
                    .iter()
                    .zip(s.iter())
                    .map(|(&g, &sv)| g * sv * (1.0 - sv))
                    .collect();
                a.accumulate_grad(&grad_a);
            }
        }

        Op::Tanh => {
            // inputs[0] = original input `a` (requires_grad)
            // inputs[1] = cached tanh output `t = tanh(a)` (no grad)
            let a = &inputs[0];
            if a.requires_grad() {
                let t = inputs[1].data();
                // grad_a[i] = upstream[i] * (1 - t[i]^2)
                let grad_a: Vec<f32> = upstream
                    .iter()
                    .zip(t.iter())
                    .map(|(&g, &tv)| g * (1.0 - tv * tv))
                    .collect();
                a.accumulate_grad(&grad_a);
            }
        }

        Op::SoftmaxCE => {
            // inputs[0] = logits (requires_grad)
            // inputs[1] = cached softmax probabilities (no grad)
            // inputs[2] = labels as f32 integer class indices (no grad)
            // upstream is a scalar [loss_grad]
            let logits = &inputs[0];
            if logits.requires_grad() {
                let softmax = inputs[1].data();
                let labels = inputs[2].data();
                let (batch, num_classes) = logits.shape();
                let loss_grad = upstream[0];

                // grad_logits[i, j] = upstream * (softmax[i, j] - one_hot[i, label_i]) / batch
                let mut grad_logits = softmax.clone();
                for i in 0..batch {
                    let label_idx = labels[i] as usize;
                    grad_logits[i * num_classes + label_idx] -= 1.0;
                }
                for g in grad_logits.iter_mut() {
                    *g = *g * loss_grad / batch as f32;
                }
                logits.accumulate_grad(&grad_logits);
            }
        }

        Op::MSE => {
            // inputs[0] = predictions (may require_grad)
            // inputs[1] = targets (may require_grad)
            // upstream is a scalar [loss_grad]
            let predictions = &inputs[0];
            let targets = &inputs[1];
            let n = predictions.numel() as f32;
            let loss_grad = upstream[0];

            if predictions.requires_grad() || targets.requires_grad() {
                let data_pred = predictions.data();
                let data_tgt = targets.data();
                // diffs[i] = pred[i] - target[i], shared between both gradient computations
                let diffs: Vec<f32> = data_pred
                    .iter()
                    .zip(data_tgt.iter())
                    .map(|(&p, &t)| p - t)
                    .collect();

                if predictions.requires_grad() {
                    // grad_pred[i] = upstream * 2 * (pred[i] - target[i]) / n
                    let grad_pred: Vec<f32> =
                        diffs.iter().map(|&d| loss_grad * 2.0 * d / n).collect();
                    predictions.accumulate_grad(&grad_pred);
                }

                if targets.requires_grad() {
                    // grad_tgt[i] = upstream * -2 * (pred[i] - target[i]) / n
                    let grad_tgt: Vec<f32> =
                        diffs.iter().map(|&d| loss_grad * -2.0 * d / n).collect();
                    targets.accumulate_grad(&grad_tgt);
                }
            }
        }

        Op::Sum => {
            // inputs[0] = original tensor of any shape
            // upstream is a scalar [grad_out]
            let a = &inputs[0];
            if a.requires_grad() {
                let n = a.numel();
                let grad_val = upstream[0];
                // Broadcast scalar gradient to all input elements.
                let grad_a = vec![grad_val; n];
                a.accumulate_grad(&grad_a);
            }
        }

        Op::Mean => {
            // inputs[0] = original tensor of any shape
            // upstream is a scalar [grad_out]
            let a = &inputs[0];
            if a.requires_grad() {
                let n = a.numel();
                // Each input element receives grad_out / n.
                let grad_val = upstream[0] / n as f32;
                let grad_a = vec![grad_val; n];
                a.accumulate_grad(&grad_a);
            }
        }
    }
}
