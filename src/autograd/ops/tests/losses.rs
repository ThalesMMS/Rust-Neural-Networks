use super::*;

// -----------------------------------------------------------------------
// tensor_softmax_cross_entropy tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_softmax_ce_output_is_scalar() {
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert_eq!(loss.shape(), (1, 1));
}

#[test]
fn test_tensor_softmax_ce_loss_is_non_negative() {
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(loss.data()[0] >= 0.0);
}

#[test]
fn test_tensor_softmax_ce_perfect_prediction_low_loss() {
    // Very high logit for the correct class should give near-zero loss.
    let logits = Tensor::from_vec(vec![100.0, -100.0, -100.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(loss.data()[0] < 1e-3, "loss = {}", loss.data()[0]);
}

#[test]
fn test_tensor_softmax_ce_uniform_logits_gives_log_n_loss() {
    // With uniform logits [0,0,0] for 3 classes, CE = log(3).
    let logits = Tensor::from_vec(vec![0.0, 0.0, 0.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    let expected = (3.0f32).ln();
    assert!(
        (loss.data()[0] - expected).abs() < 1e-5,
        "loss = {}, expected = {}",
        loss.data()[0],
        expected
    );
}

#[test]
fn test_tensor_softmax_ce_requires_grad_propagated() {
    let logits = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let labels = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(loss.requires_grad());
}

#[test]
fn test_tensor_softmax_ce_no_grad_when_logits_have_no_grad() {
    let logits = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let labels = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(!loss.requires_grad());
    assert!(loss.0.borrow().grad_node.is_none());
}

#[test]
fn test_tensor_softmax_ce_records_softmax_ce_op() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 0.5], (1, 3), true);
    let labels = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    let inner = loss.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::SoftmaxCE));
    // inputs[0]=logits, inputs[1]=cached softmax, inputs[2]=labels
    assert_eq!(node.inputs.len(), 3);
    assert!(node.inputs[0].requires_grad());
    assert!(!node.inputs[1].requires_grad()); // cached softmax has no grad
    assert!(!node.inputs[2].requires_grad()); // labels have no grad
}

#[test]
fn test_tensor_softmax_ce_batch_of_two() {
    // Two examples, 2 classes each.
    let logits = Tensor::from_vec(vec![10.0, 0.0, 0.0, 10.0], (2, 2), true);
    let labels = Tensor::from_vec(vec![0.0, 1.0], (2, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    // Both predictions are nearly perfect → low loss
    assert!(loss.data()[0] < 1e-3, "loss = {}", loss.data()[0]);
}

#[test]
fn test_tensor_softmax_ce_numerical_stability_large_logits() {
    // Large logit differences should not produce NaN or Inf.
    let logits = Tensor::from_vec(vec![1000.0, -1000.0], (1, 2), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    let val = loss.data()[0];
    assert!(val.is_finite(), "loss should be finite, got {}", val);
}

#[test]
#[should_panic(expected = "tensor_softmax_cross_entropy: batch must be > 0")]
fn test_tensor_softmax_ce_rejects_empty_batch() {
    let logits = Tensor::from_vec(vec![], (0, 3), true);
    let labels = Tensor::from_vec(vec![], (0, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(
    expected = "tensor_softmax_cross_entropy: labels must have shape (batch, 1), got (1, 2)"
)]
fn test_tensor_softmax_ce_rejects_label_shape_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0, 1.0], (1, 2), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(
    expected = "tensor_softmax_cross_entropy: labels row count 1 must match logits batch size 2"
)]
fn test_tensor_softmax_ce_rejects_label_row_count_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(expected = "tensor_softmax_cross_entropy: num_classes must be > 0")]
fn test_tensor_softmax_ce_rejects_zero_num_classes() {
    let logits = Tensor::from_vec(vec![], (1, 0), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(
    expected = "tensor_softmax_cross_entropy: label at row 0 must be a finite non-negative integer"
)]
fn test_tensor_softmax_ce_rejects_non_integer_label() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![1.5], (1, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(
    expected = "tensor_softmax_cross_entropy: label at row 0 must be a finite non-negative integer"
)]
fn test_tensor_softmax_ce_rejects_non_finite_label() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![f32::NAN], (1, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

#[test]
#[should_panic(expected = "tensor_softmax_cross_entropy: label index 3 out of range for 3 classes")]
fn test_tensor_softmax_ce_rejects_out_of_range_label() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![3.0], (1, 1), false);
    let _ = tensor_softmax_cross_entropy(&logits, &labels);
}

// -----------------------------------------------------------------------
// tensor_mse_loss tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_mse_loss_perfect_predictions_gives_zero() {
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let tgt = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert_eq!(loss.shape(), (1, 1));
    assert!((loss.data()[0]).abs() < 1e-6);
}

#[test]
fn test_tensor_mse_loss_known_value() {
    // MSE([2,4], [0,0]) = ((2-0)^2 + (4-0)^2) / 2 = (4 + 16) / 2 = 10
    let pred = Tensor::from_vec(vec![2.0, 4.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!((loss.data()[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_tensor_mse_loss_is_non_negative() {
    let pred = Tensor::from_vec(vec![-1.0, 5.0, 3.0], (1, 3), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0, 0.0], (1, 3), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(loss.data()[0] >= 0.0);
}

#[test]
fn test_tensor_mse_loss_requires_grad_propagated() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(loss.requires_grad());
}

#[test]
fn test_tensor_mse_loss_requires_grad_propagated_target_true() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), true);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(loss.requires_grad());
    assert!(loss.0.borrow().grad_node.is_some());
}

#[test]
fn test_tensor_mse_loss_no_grad_when_neither_requires_grad() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(!loss.requires_grad());
    assert!(loss.0.borrow().grad_node.is_none());
}

#[test]
fn test_tensor_mse_loss_records_mse_op() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    let inner = loss.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::MSE));
    assert_eq!(node.inputs.len(), 2);
}

#[test]
fn test_tensor_mse_loss_scalar_case() {
    // MSE([2], [1]) = (2-1)^2 / 1 = 1.0
    let pred = Tensor::from_vec(vec![2.0], (1, 1), true);
    let tgt = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!((loss.data()[0] - 1.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "tensor_mse_loss: shape mismatch (1, 2) vs (1, 3)")]
fn test_tensor_mse_loss_rejects_shape_mismatch() {
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
    let _ = tensor_mse_loss(&pred, &tgt);
}

#[test]
#[should_panic(expected = "tensor_mse_loss: input tensors must not be empty")]
fn test_tensor_mse_loss_rejects_empty_tensors() {
    let pred = Tensor::from_vec(vec![], (0, 1), true);
    let tgt = Tensor::from_vec(vec![], (0, 1), false);
    let _ = tensor_mse_loss(&pred, &tgt);
}
