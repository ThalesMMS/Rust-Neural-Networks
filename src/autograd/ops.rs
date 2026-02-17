//! Differentiable operations on tensors.
//!
//! This module provides the forward-pass implementations of differentiable
//! operations. Each function:
//!
//! 1. Computes the forward result.
//! 2. Records a [`GradNode`] on the output tensor so the backward pass can
//!    propagate gradients through the computational graph via the chain rule.
//!
//! # Operations
//!
//! | Function              | Description                                      |
//! |-----------------------|--------------------------------------------------|
//! | [`tensor_add`]        | Element-wise addition of two same-shape tensors  |
//! | [`tensor_sub`]        | Element-wise subtraction of two same-shape tensors |
//! | [`tensor_mul_scalar`] | Multiply all elements by a scalar constant       |
//! | [`tensor_add_bias`]   | Add a `(1, features)` bias to a `(batch, features)` matrix |
//! | [`tensor_relu`]       | Element-wise ReLU activation                     |
//! | [`tensor_sigmoid`]    | Element-wise logistic sigmoid activation         |
//! | [`tensor_tanh`]       | Element-wise hyperbolic tangent activation       |
//! | [`tensor_softmax_cross_entropy`] | Fused softmax + cross-entropy loss (numerically stable) |
//! | [`tensor_mse_loss`]   | Mean squared error loss                          |
//!
//! # Gradient recording
//!
//! A [`GradNode`] is only recorded on the output tensor when at least one
//! input has `requires_grad = true`.  Tensors that do not participate in
//! gradient computation have `grad_node = None` and are therefore ignored
//! during backward traversal.

use crate::autograd::tape::{GradNode, Op};
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

            if predictions.requires_grad() {
                let data_pred = predictions.data();
                let data_tgt = targets.data();
                // grad_pred[i] = upstream * 2 * (pred[i] - target[i]) / n
                let grad_pred: Vec<f32> = data_pred
                    .iter()
                    .zip(data_tgt.iter())
                    .map(|(&p, &t)| loss_grad * 2.0 * (p - t) / n)
                    .collect();
                predictions.accumulate_grad(&grad_pred);
            }

            if targets.requires_grad() {
                let data_pred = predictions.data();
                let data_tgt = targets.data();
                // grad_tgt[i] = upstream * -2 * (pred[i] - target[i]) / n
                let grad_tgt: Vec<f32> = data_pred
                    .iter()
                    .zip(data_tgt.iter())
                    .map(|(&p, &t)| loss_grad * -2.0 * (p - t) / n)
                    .collect();
                targets.accumulate_grad(&grad_tgt);
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

// ---------------------------------------------------------------------------
// tensor_add
// ---------------------------------------------------------------------------

/// Element-wise addition of two tensors: `out[i] = a[i] + b[i]`.
///
/// Both tensors must have the same shape. If either input `requires_grad`,
/// the output records an [`Op::Add`] node so gradients can be propagated
/// to both inputs during backward.
///
/// # Arguments
///
/// * `a` - First input tensor.
/// * `b` - Second input tensor (same shape as `a`).
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_add;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
/// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (1, 3), true);
/// let out = tensor_add(&a, &b);
/// assert_eq!(out.data(), vec![5.0f32, 7.0, 9.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(
        shape_a, shape_b,
        "tensor_add: shape mismatch {:?} vs {:?}",
        shape_a, shape_b
    );

    let data_a = a.data();
    let data_b = b.data();
    let out_data: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x + y)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, shape_a, requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Add, vec![a.clone(), b.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_sub
// ---------------------------------------------------------------------------

/// Element-wise subtraction of two tensors: `out[i] = a[i] - b[i]`.
///
/// Both tensors must have the same shape. If either input `requires_grad`,
/// the output records an [`Op::Sub`] node.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sub;
///
/// let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], (1, 3), true);
/// let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
/// let out = tensor_sub(&a, &b);
/// assert_eq!(out.data(), vec![4.0f32, 5.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_sub(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(
        shape_a, shape_b,
        "tensor_sub: shape mismatch {:?} vs {:?}",
        shape_a, shape_b
    );

    let data_a = a.data();
    let data_b = b.data();
    let out_data: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x - y)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, shape_a, requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Sub, vec![a.clone(), b.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_mul_scalar
// ---------------------------------------------------------------------------

/// Multiplies every element of `a` by a scalar constant: `out[i] = a[i] * scalar`.
///
/// Internally the scalar is broadcast into a constant tensor of the same shape
/// as `a` and stored as the second input of an [`Op::Mul`] node.  Because the
/// scalar tensor does not `require_grad`, the backward pass will only propagate
/// gradients to `a`.
///
/// # Arguments
///
/// * `a`      - Input tensor.
/// * `scalar` - The scalar multiplier.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_mul_scalar;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
/// let out = tensor_mul_scalar(&a, 2.0);
/// assert_eq!(out.data(), vec![2.0f32, 4.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_mul_scalar(a: &Tensor, scalar: f32) -> Tensor {
    let shape = a.shape();
    let data_a = a.data();
    let out_data: Vec<f32> = data_a.iter().map(|x| x * scalar).collect();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        // Create a constant tensor holding the broadcast scalar value.
        // requires_grad = false so backward will not accumulate into it.
        let n = shape.0 * shape.1;
        let scalar_tensor = Tensor::new(vec![scalar; n], shape);
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Mul,
            vec![a.clone(), scalar_tensor],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_add_bias
// ---------------------------------------------------------------------------

/// Adds a bias vector to each row of a matrix.
///
/// `a` has shape `(batch, features)` and `bias` has shape `(1, features)`.
/// The bias is broadcast across the batch dimension:
///
/// ```text
/// out[r][c] = a[r][c] + bias[c]   for r in 0..batch, c in 0..features
/// ```
///
/// The output records an [`Op::Add`] node with `inputs = [a, bias]`.  During
/// backward the gradient of `bias` is accumulated by summing `grad_out` along
/// the batch axis (standard broadcast-sum rule for the backward of `+`).
///
/// # Arguments
///
/// * `a`    - Input matrix of shape `(batch, features)`.
/// * `bias` - Bias row-vector of shape `(1, features)`.
///
/// # Panics
///
/// Panics if `bias.shape().0 != 1` or if `bias.shape().1 != a.shape().1`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_add_bias;
///
/// let a    = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
/// let out  = tensor_add_bias(&a, &bias);
/// assert_eq!(out.data(), vec![11.0f32, 22.0, 13.0, 24.0]);
/// ```
pub fn tensor_add_bias(a: &Tensor, bias: &Tensor) -> Tensor {
    let (batch, features) = a.shape();
    let (bias_rows, bias_cols) = bias.shape();
    assert_eq!(
        bias_rows, 1,
        "tensor_add_bias: bias must have exactly 1 row, got {}",
        bias_rows
    );
    assert_eq!(
        bias_cols, features,
        "tensor_add_bias: bias columns {} must match a columns {}",
        bias_cols, features
    );

    let data_a = a.data();
    let data_bias = bias.data();
    let mut out_data = vec![0.0f32; batch * features];
    for r in 0..batch {
        for c in 0..features {
            out_data[r * features + c] = data_a[r * features + c] + data_bias[c];
        }
    }

    let requires_grad = a.requires_grad() || bias.requires_grad();
    let out = Tensor::from_vec(out_data, (batch, features), requires_grad);

    if requires_grad {
        // Record Op::Add with inputs [a, bias].  The backward pass detects the
        // shape difference (batch × features vs 1 × features) and sums grad_out
        // along the batch axis when propagating to bias.
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Add,
            vec![a.clone(), bias.clone()],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_matmul
// ---------------------------------------------------------------------------

/// Matrix multiplication of two 2-D tensors: `out = a @ b`.
///
/// `a` must have shape `(m, k)` and `b` must have shape `(k, n)`.  The output
/// has shape `(m, n)`.  The forward pass uses explicit nested loops for
/// educational clarity (O(m·k·n)).
///
/// If either input `requires_grad`, the output records an
/// [`Op::MatMul { m, k, n }`] node.  The backward pass uses this node to
/// compute:
///
/// ```text
/// grad_a = grad_out  @  b.T    shape (m, k)
/// grad_b = a.T       @  grad_out    shape (k, n)
/// ```
///
/// # Arguments
///
/// * `a` - Left-hand matrix of shape `(m, k)`.
/// * `b` - Right-hand matrix of shape `(k, n)`.
///
/// # Panics
///
/// Panics if the inner dimensions do not match (i.e. `a.shape().1 != b.shape().0`).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_matmul;
///
/// // 2×2 identity times any matrix leaves it unchanged.
/// let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
/// let b   = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), true);
/// let out = tensor_matmul(&eye, &b);
/// assert_eq!(out.shape(), (2, 2));
/// assert_eq!(out.data(), vec![3.0f32, 4.0, 5.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    assert_eq!(
        k_a, k_b,
        "tensor_matmul: inner dimension mismatch — a is ({m}, {k_a}), b is ({k_b}, {n})"
    );
    let k = k_a;

    let data_a = a.data();
    let data_b = b.data();

    // Forward: O(m·k·n) nested loops for educational clarity.
    let mut out_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += data_a[i * k + p] * data_b[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, (m, n), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::MatMul { m, k, n },
            vec![a.clone(), b.clone()],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_sum
// ---------------------------------------------------------------------------

/// Reduces all elements of a tensor to a scalar by summation.
///
/// The output is a `(1, 1)` tensor holding the sum of all input elements.
/// This is primarily used as a loss reduction step (e.g. after computing
/// per-element losses, sum to get a single scalar before calling backward).
///
/// If the input `requires_grad`, the output records an [`Op::Sum`] node.
/// During backward, the scalar gradient is broadcast back to every element
/// of the input (`grad_a[i] = grad_out[0]`).
///
/// # Arguments
///
/// * `a` - Input tensor of any shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sum;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let s = tensor_sum(&a);
/// assert_eq!(s.shape(), (1, 1));
/// assert!((s.data()[0] - 10.0).abs() < 1e-6);
/// assert!(s.requires_grad());
/// ```
pub fn tensor_sum(a: &Tensor) -> Tensor {
    let data_a = a.data();
    let total: f32 = data_a.iter().sum();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(vec![total], (1, 1), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::Sum, vec![a.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_mean
// ---------------------------------------------------------------------------

/// Reduces all elements of a tensor to a scalar by averaging.
///
/// The output is a `(1, 1)` tensor holding the mean of all input elements.
/// Equivalent to `tensor_sum(a) / n`, but recorded as a single [`Op::Mean`]
/// node so the backward pass can apply the correct `1/n` scaling in one step.
///
/// If the input `requires_grad`, the output records an [`Op::Mean`] node.
/// During backward, each input element receives `grad_out[0] / n` where `n`
/// is the total number of elements in the input tensor.
///
/// # Arguments
///
/// * `a` - Input tensor of any shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_mean;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let m = tensor_mean(&a);
/// assert_eq!(m.shape(), (1, 1));
/// assert!((m.data()[0] - 2.5).abs() < 1e-6);
/// assert!(m.requires_grad());
/// ```
pub fn tensor_mean(a: &Tensor) -> Tensor {
    let data_a = a.data();
    let n = data_a.len();
    let total: f32 = data_a.iter().sum();
    let mean = total / n as f32;

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(vec![mean], (1, 1), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::Mean, vec![a.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_relu
// ---------------------------------------------------------------------------

/// Applies the Rectified Linear Unit (ReLU) activation element-wise.
///
/// `out[i] = max(0, a[i])`
///
/// If the input `requires_grad`, the output records an [`Op::ReLU`] node with
/// the original input tensor `a` stored as `inputs[0]`.  The backward pass
/// uses the sign of `inputs[0].data[i]` to reconstruct the activation mask:
///
/// ```text
/// grad_a[i] = grad_out[i]  if a[i] > 0
///           = 0             otherwise
/// ```
///
/// # Arguments
///
/// * `a` - Input tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_relu;
///
/// let a = Tensor::from_vec(vec![-1.0, 0.0, 2.0], (1, 3), true);
/// let out = tensor_relu(&a);
/// assert_eq!(out.data(), vec![0.0f32, 0.0, 2.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_relu(a: &Tensor) -> Tensor {
    let shape = a.shape();
    let data_a = a.data();
    let out_data: Vec<f32> = data_a
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        // Store the original input so backward can check a[i] > 0 for the mask.
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::ReLU, vec![a.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_sigmoid
// ---------------------------------------------------------------------------

/// Applies the logistic sigmoid activation element-wise.
///
/// `out[i] = 1 / (1 + exp(-a[i]))`
///
/// If the input `requires_grad`, the output records an [`Op::Sigmoid`] node.
/// `inputs[0]` is the original input `a` (gradient flows to it) and
/// `inputs[1]` is a constant tensor holding the cached sigmoid output values
/// (no grad) so the backward pass avoids recomputing them:
///
/// ```text
/// grad_a[i] = grad_out[i] * s[i] * (1 - s[i])   where s[i] = sigmoid(a[i])
/// ```
///
/// # Arguments
///
/// * `a` - Input tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sigmoid;
///
/// let a = Tensor::from_vec(vec![0.0], (1, 1), true);
/// let out = tensor_sigmoid(&a);
/// assert!((out.data()[0] - 0.5).abs() < 1e-6);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_sigmoid(a: &Tensor) -> Tensor {
    let shape = a.shape();
    let data_a = a.data();
    let out_data: Vec<f32> = data_a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        // Cache the sigmoid output values (no requires_grad) so the backward pass
        // can compute s * (1 - s) without re-evaluating the exponential.
        let cached_sigmoid = Tensor::new(out.data(), shape);
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Sigmoid,
            vec![a.clone(), cached_sigmoid],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_tanh
// ---------------------------------------------------------------------------

/// Applies the hyperbolic tangent activation element-wise.
///
/// `out[i] = tanh(a[i])`
///
/// If the input `requires_grad`, the output records an [`Op::Tanh`] node.
/// `inputs[0]` is the original input `a` (gradient flows to it) and
/// `inputs[1]` is a constant tensor holding the cached tanh output values
/// (no grad) so the backward pass avoids recomputing them:
///
/// ```text
/// grad_a[i] = grad_out[i] * (1 - t[i]^2)   where t[i] = tanh(a[i])
/// ```
///
/// # Arguments
///
/// * `a` - Input tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_tanh;
///
/// let a = Tensor::from_vec(vec![0.0], (1, 1), true);
/// let out = tensor_tanh(&a);
/// assert!((out.data()[0] - 0.0).abs() < 1e-6);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_tanh(a: &Tensor) -> Tensor {
    let shape = a.shape();
    let data_a = a.data();
    let out_data: Vec<f32> = data_a.iter().map(|&x| x.tanh()).collect();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        // Cache the tanh output values (no requires_grad) so the backward pass
        // can compute 1 - t^2 without re-evaluating tanh.
        let cached_tanh = Tensor::new(out.data(), shape);
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Tanh,
            vec![a.clone(), cached_tanh],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_softmax_cross_entropy
// ---------------------------------------------------------------------------

/// Fused softmax + cross-entropy loss with the log-sum-exp trick for numerical stability.
///
/// Computes the mean cross-entropy loss over a batch:
///
/// ```text
/// loss = mean_i( -logits[i, label_i] + log(sum_j exp(logits[i, j])) )
/// ```
///
/// The log-sum-exp is computed with `max(logits[i])` subtracted for stability.
///
/// # Arguments
///
/// * `logits` - Shape `(batch, num_classes)`.  If `requires_grad`, gradients will
///   be propagated to this tensor during backward.
/// * `labels` - Shape `(batch, 1)` with integer class indices stored as `f32`.
///   This tensor does not participate in gradient computation.
///
/// # Output
///
/// A scalar `(1, 1)` tensor holding the mean cross-entropy loss.
///
/// # Gradient (backward)
///
/// ```text
/// grad_logits[i, j] = upstream * (softmax[i, j] - one_hot[i, label_i]) / batch
/// ```
///
/// The backward pass accesses `inputs[1]` (cached softmax) and `inputs[2]` (labels).
///
/// # Panics
///
/// Panics if `labels.shape() != (batch, 1)` where `batch = logits.shape().0`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
///
/// // batch=1, 3 classes, true label = class 0
/// let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], (1, 3), true);
/// let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
/// let loss = tensor_softmax_cross_entropy(&logits, &labels);
/// assert_eq!(loss.shape(), (1, 1));
/// assert!(loss.data()[0] >= 0.0);
/// assert!(loss.requires_grad());
/// ```
pub fn tensor_softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    let (batch, num_classes) = logits.shape();
    let (label_rows, label_cols) = labels.shape();
    assert_eq!(
        label_rows, batch,
        "tensor_softmax_cross_entropy: labels row count {} must match logits batch size {}",
        label_rows, batch
    );
    assert_eq!(
        label_cols, 1,
        "tensor_softmax_cross_entropy: labels must have shape (batch, 1), got ({}, {})",
        label_rows, label_cols
    );

    let data_logits = logits.data();
    let data_labels = labels.data();

    let mut softmax_cache = vec![0.0f32; batch * num_classes];
    let mut total_loss = 0.0f32;

    for (i, &label_raw) in data_labels.iter().enumerate() {
        let row_start = i * num_classes;
        let row = &data_logits[row_start..row_start + num_classes];

        // Log-sum-exp trick: shift by the row maximum for numerical stability.
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &x in row {
            sum_exp += (x - max_val).exp();
        }
        let log_sum_exp = sum_exp.ln() + max_val;

        // Softmax probabilities cached for the backward pass.
        let inv_sum_exp = 1.0 / sum_exp;
        for j in 0..num_classes {
            softmax_cache[row_start + j] = (row[j] - max_val).exp() * inv_sum_exp;
        }

        // Cross-entropy: -logit[label] + log_sum_exp.
        let label_idx = label_raw as usize;
        total_loss += -data_logits[row_start + label_idx] + log_sum_exp;
    }

    let loss = total_loss / batch as f32;

    let requires_grad = logits.requires_grad();
    let out = Tensor::from_vec(vec![loss], (1, 1), requires_grad);

    if requires_grad {
        // inputs[0] = logits (requires_grad)
        // inputs[1] = cached softmax probabilities (no grad) — used in backward
        // inputs[2] = labels (no grad) — used in backward for one_hot
        let cached_softmax = Tensor::new(softmax_cache, (batch, num_classes));
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::SoftmaxCE,
            vec![logits.clone(), cached_softmax, labels.clone()],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_mse_loss
// ---------------------------------------------------------------------------

/// Mean squared error loss: `out = mean_i((predictions[i] - targets[i])^2)`.
///
/// Both tensors must have the same shape.  The output is a scalar `(1, 1)` tensor.
///
/// # Arguments
///
/// * `predictions` - Predicted values tensor of any shape.  If `requires_grad`,
///   gradients will be propagated to this tensor during backward.
/// * `targets` - Ground-truth target values.  Same shape as `predictions`.
///
/// # Output
///
/// A scalar `(1, 1)` tensor holding the mean squared error.
///
/// # Gradient (backward)
///
/// ```text
/// grad_predictions[i] = upstream * 2 * (predictions[i] - targets[i]) / n
/// grad_targets[i]     = upstream * -2 * (predictions[i] - targets[i]) / n
/// ```
///
/// where `n` is the total number of elements.
///
/// # Panics
///
/// Panics if `predictions` and `targets` have different shapes.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_mse_loss;
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
/// let tgt  = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
/// let loss = tensor_mse_loss(&pred, &tgt);
/// assert_eq!(loss.shape(), (1, 1));
/// assert!((loss.data()[0] - 0.0).abs() < 1e-6); // perfect predictions → loss = 0
/// assert!(loss.requires_grad());
/// ```
pub fn tensor_mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let shape_pred = predictions.shape();
    let shape_tgt = targets.shape();
    assert_eq!(
        shape_pred, shape_tgt,
        "tensor_mse_loss: shape mismatch {:?} vs {:?}",
        shape_pred, shape_tgt
    );

    let data_pred = predictions.data();
    let data_tgt = targets.data();
    let n = data_pred.len();

    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let diff = data_pred[i] - data_tgt[i];
        sum_sq += diff * diff;
    }
    let loss = sum_sq / n as f32;

    let requires_grad = predictions.requires_grad() || targets.requires_grad();
    let out = Tensor::from_vec(vec![loss], (1, 1), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::MSE,
            vec![predictions.clone(), targets.clone()],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (1, 3), true);
        let out = tensor_add(&a, &b);
        assert_eq!(out.data(), vec![5.0f32, 7.0, 9.0]);
        assert_eq!(out.shape(), (1, 3));
    }

    #[test]
    fn test_tensor_add_records_grad_node() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
        let out = tensor_add(&a, &b);
        assert!(out.requires_grad());
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().expect("grad_node should be set");
        assert!(matches!(node.op, Op::Add));
        assert_eq!(node.inputs.len(), 2);
    }

    #[test]
    fn test_tensor_add_no_grad_when_no_inputs_require_grad() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
        let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), false);
        let out = tensor_add(&a, &b);
        assert!(!out.requires_grad());
        assert!(out.0.borrow().grad_node.is_none());
    }

    #[test]
    fn test_tensor_sub_forward() {
        let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], (1, 3), true);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
        let out = tensor_sub(&a, &b);
        assert_eq!(out.data(), vec![4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_sub_records_grad_node() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), true);
        let b = Tensor::from_vec(vec![0.5], (1, 1), false);
        let out = tensor_sub(&a, &b);
        assert!(out.requires_grad());
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().unwrap();
        assert!(matches!(node.op, Op::Sub));
    }

    #[test]
    fn test_tensor_mul_scalar_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
        let out = tensor_mul_scalar(&a, 2.0);
        assert_eq!(out.data(), vec![2.0f32, 4.0, 6.0]);
        assert_eq!(out.shape(), (1, 3));
    }

    #[test]
    fn test_tensor_mul_scalar_records_mul_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        let out = tensor_mul_scalar(&a, 3.0);
        assert!(out.requires_grad());
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().unwrap();
        assert!(matches!(node.op, Op::Mul));
        assert_eq!(node.inputs.len(), 2);
        // Second input is the broadcast scalar tensor (no grad)
        assert!(!node.inputs[1].requires_grad());
    }

    #[test]
    fn test_tensor_mul_scalar_no_grad_when_input_has_no_grad() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
        let out = tensor_mul_scalar(&a, 5.0);
        assert!(!out.requires_grad());
        assert!(out.0.borrow().grad_node.is_none());
    }

    #[test]
    fn test_tensor_add_bias_forward() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
        let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
        let out = tensor_add_bias(&a, &bias);
        assert_eq!(out.data(), vec![11.0f32, 22.0, 13.0, 24.0]);
        assert_eq!(out.shape(), (2, 2));
    }

    #[test]
    fn test_tensor_add_bias_records_grad_node() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
        let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
        let out = tensor_add_bias(&a, &bias);
        assert!(out.requires_grad());
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().expect("grad_node should be set");
        assert!(matches!(node.op, Op::Add));
        assert_eq!(node.inputs.len(), 2);
    }

    #[test]
    fn test_tensor_add_bias_single_batch_row() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
        let bias = Tensor::from_vec(vec![0.1, 0.2, 0.3], (1, 3), true);
        let out = tensor_add_bias(&a, &bias);
        let data = out.data();
        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 2.2).abs() < 1e-6);
        assert!((data[2] - 3.3).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // tensor_relu tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_relu_forward_clamps_negatives() {
        let a = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.5, 3.0], (1, 5), true);
        let out = tensor_relu(&a);
        let data = out.data();
        assert_eq!(data, vec![0.0f32, 0.0, 0.0, 1.5, 3.0]);
    }

    #[test]
    fn test_tensor_relu_shape_preserved() {
        let a = Tensor::from_vec(vec![1.0, -1.0, 2.0, -2.0], (2, 2), true);
        let out = tensor_relu(&a);
        assert_eq!(out.shape(), (2, 2));
    }

    #[test]
    fn test_tensor_relu_requires_grad_propagated() {
        let a = Tensor::from_vec(vec![1.0, -1.0], (1, 2), true);
        let out = tensor_relu(&a);
        assert!(out.requires_grad());
    }

    #[test]
    fn test_tensor_relu_no_grad_when_input_has_no_grad() {
        let a = Tensor::from_vec(vec![1.0, -1.0], (1, 2), false);
        let out = tensor_relu(&a);
        assert!(!out.requires_grad());
        assert!(out.0.borrow().grad_node.is_none());
    }

    #[test]
    fn test_tensor_relu_records_relu_op_with_input() {
        let a = Tensor::from_vec(vec![2.0, -1.0], (1, 2), true);
        let out = tensor_relu(&a);
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().expect("grad_node should be set");
        assert!(matches!(node.op, Op::ReLU));
        // inputs[0] is the original input `a`
        assert_eq!(node.inputs.len(), 1);
        assert!(node.inputs[0].requires_grad());
        // The cached input data matches the original
        assert_eq!(node.inputs[0].data(), vec![2.0f32, -1.0]);
    }

    // -----------------------------------------------------------------------
    // tensor_sigmoid tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_sigmoid_forward_at_zero() {
        let a = Tensor::from_vec(vec![0.0], (1, 1), true);
        let out = tensor_sigmoid(&a);
        assert!((out.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_sigmoid_range_is_open_zero_one() {
        let a = Tensor::from_vec(vec![-10.0, 0.0, 10.0], (1, 3), true);
        let out = tensor_sigmoid(&a);
        let data = out.data();
        assert!(data[0] > 0.0 && data[0] < 0.5);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!(data[2] > 0.5 && data[2] < 1.0);
    }

    #[test]
    fn test_tensor_sigmoid_requires_grad_propagated() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), true);
        let out = tensor_sigmoid(&a);
        assert!(out.requires_grad());
    }

    #[test]
    fn test_tensor_sigmoid_no_grad_when_input_has_no_grad() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), false);
        let out = tensor_sigmoid(&a);
        assert!(!out.requires_grad());
        assert!(out.0.borrow().grad_node.is_none());
    }

    #[test]
    fn test_tensor_sigmoid_records_sigmoid_op_and_caches_output() {
        let a = Tensor::from_vec(vec![0.0], (1, 1), true);
        let out = tensor_sigmoid(&a);
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().expect("grad_node should be set");
        assert!(matches!(node.op, Op::Sigmoid));
        // inputs[0] = original input (requires_grad=true)
        // inputs[1] = cached sigmoid output (no grad)
        assert_eq!(node.inputs.len(), 2);
        assert!(node.inputs[0].requires_grad());
        assert!(!node.inputs[1].requires_grad());
        // Cached sigmoid at 0 should be 0.5
        assert!((node.inputs[1].data()[0] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // tensor_tanh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_tanh_forward_at_zero() {
        let a = Tensor::from_vec(vec![0.0], (1, 1), true);
        let out = tensor_tanh(&a);
        assert!((out.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_tanh_range_is_minus_one_to_one() {
        // Use moderate inputs to stay away from saturation (tanh(±10) rounds to ±1.0 in f32).
        let a = Tensor::from_vec(vec![-2.0, 0.0, 2.0], (1, 3), true);
        let out = tensor_tanh(&a);
        let data = out.data();
        // tanh(-2) ≈ -0.964 — strictly in (-1, 0)
        assert!(data[0] > -1.0 && data[0] < 0.0);
        assert!((data[1] - 0.0).abs() < 1e-6);
        // tanh(2) ≈ 0.964 — strictly in (0, 1)
        assert!(data[2] > 0.0 && data[2] < 1.0);
    }

    #[test]
    fn test_tensor_tanh_requires_grad_propagated() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), true);
        let out = tensor_tanh(&a);
        assert!(out.requires_grad());
    }

    #[test]
    fn test_tensor_tanh_no_grad_when_input_has_no_grad() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), false);
        let out = tensor_tanh(&a);
        assert!(!out.requires_grad());
        assert!(out.0.borrow().grad_node.is_none());
    }

    #[test]
    fn test_tensor_tanh_records_tanh_op_and_caches_output() {
        let a = Tensor::from_vec(vec![0.0], (1, 1), true);
        let out = tensor_tanh(&a);
        let inner = out.0.borrow();
        let node = inner.grad_node.as_ref().expect("grad_node should be set");
        assert!(matches!(node.op, Op::Tanh));
        // inputs[0] = original input (requires_grad=true)
        // inputs[1] = cached tanh output (no grad)
        assert_eq!(node.inputs.len(), 2);
        assert!(node.inputs[0].requires_grad());
        assert!(!node.inputs[1].requires_grad());
        // Cached tanh at 0 should be 0
        assert!((node.inputs[1].data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_tanh_shape_preserved() {
        let a = Tensor::from_vec(vec![1.0, -1.0, 0.0, 2.0], (2, 2), true);
        let out = tensor_tanh(&a);
        assert_eq!(out.shape(), (2, 2));
    }
}
