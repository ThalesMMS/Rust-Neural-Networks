use crate::autograd::tape::{GradNode, Op};
use crate::autograd::tensor::Tensor;

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
/// Panics if `labels.shape() != (batch, 1)` where `batch = logits.shape().0`,
/// if the batch or class count is zero, or if a label is not a finite integer
/// class index in range.
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
    assert!(batch > 0, "tensor_softmax_cross_entropy: batch must be > 0");
    assert!(
        num_classes > 0,
        "tensor_softmax_cross_entropy: num_classes must be > 0"
    );

    let data_logits = logits.data();
    let data_labels = labels.data();

    let mut softmax_cache = vec![0.0f32; batch * num_classes];
    let mut total_loss = 0.0f32;

    for (i, &label_raw) in data_labels.iter().enumerate() {
        assert!(
            label_raw.is_finite() && label_raw >= 0.0 && label_raw.fract() == 0.0,
            "tensor_softmax_cross_entropy: label at row {} must be a finite non-negative integer, got {}",
            i,
            label_raw
        );
        let label_idx = label_raw as usize;
        assert!(
            label_idx < num_classes,
            "tensor_softmax_cross_entropy: label index {} out of range for {} classes",
            label_idx,
            num_classes
        );

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
/// Panics if `predictions` and `targets` have different shapes or are empty.
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
    assert!(n > 0, "tensor_mse_loss: input tensors must not be empty");

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
