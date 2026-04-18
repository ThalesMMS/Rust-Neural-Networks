use super::*;
use crate::autograd::tape::Op;
use crate::autograd::tensor::Tensor;

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
    let a = Tensor::from_vec(vec![1.0], (1, 1), true);
    let out = tensor_sigmoid(&a);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::Sigmoid));
    // inputs[0] = original input (requires_grad=true)
    // inputs[1] = cached sigmoid output (no grad)
    assert_eq!(node.inputs.len(), 2);
    assert!(node.inputs[0].requires_grad());
    assert!(!node.inputs[1].requires_grad());
    let cached = node.inputs[1].data();
    assert!(cached[0] > 0.5 && cached[0] < 1.0);
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

// -----------------------------------------------------------------------
// tensor_matmul tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_matmul_forward_2x2() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), false);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], (2, 2), false);
    let out = tensor_matmul(&a, &b);
    assert_eq!(out.data(), vec![19.0f32, 22.0, 43.0, 50.0]);
    assert_eq!(out.shape(), (2, 2));
}

#[test]
fn test_tensor_matmul_identity_leaves_matrix_unchanged() {
    let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), true);
    let out = tensor_matmul(&eye, &b);
    assert_eq!(out.data(), vec![3.0f32, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_matmul_non_square() {
    // (2,3) @ (3,1) = (2,1)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), true);
    let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], (3, 1), false);
    let out = tensor_matmul(&a, &b);
    assert_eq!(out.shape(), (2, 1));
    // Row 0: 1+2+3=6, Row 1: 4+5+6=15
    assert!((out.data()[0] - 6.0).abs() < 1e-6);
    assert!((out.data()[1] - 15.0).abs() < 1e-6);
}

#[test]
fn test_tensor_matmul_requires_grad_propagated() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), true);
    let out = tensor_matmul(&a, &b);
    assert!(out.requires_grad());
}

#[test]
fn test_tensor_matmul_no_grad_when_neither_requires_grad() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), false);
    let out = tensor_matmul(&a, &b);
    assert!(!out.requires_grad());
    assert!(out.0.borrow().grad_node.is_none());
}

#[test]
fn test_tensor_matmul_records_matmul_op() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], (2, 2), false);
    let out = tensor_matmul(&a, &b);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::MatMul { m: 2, k: 2, n: 2 }));
    assert_eq!(node.inputs.len(), 2);
}

// -----------------------------------------------------------------------
// tensor_sum tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_sum_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let s = tensor_sum(&a);
    assert_eq!(s.shape(), (1, 1));
    assert!((s.data()[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_tensor_sum_single_element() {
    let a = Tensor::from_vec(vec![7.0], (1, 1), true);
    let s = tensor_sum(&a);
    assert!((s.data()[0] - 7.0).abs() < 1e-6);
}

#[test]
fn test_tensor_sum_requires_grad_propagated() {
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let s = tensor_sum(&a);
    assert!(s.requires_grad());
}

#[test]
fn test_tensor_sum_no_grad_when_input_has_no_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let s = tensor_sum(&a);
    assert!(!s.requires_grad());
    assert!(s.0.borrow().grad_node.is_none());
}

#[test]
fn test_tensor_sum_records_sum_op() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let s = tensor_sum(&a);
    let inner = s.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::Sum));
    assert_eq!(node.inputs.len(), 1);
}

// -----------------------------------------------------------------------
// tensor_mean tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_mean_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let m = tensor_mean(&a);
    assert_eq!(m.shape(), (1, 1));
    assert!((m.data()[0] - 2.5).abs() < 1e-6);
}

#[test]
fn test_tensor_mean_single_element() {
    let a = Tensor::from_vec(vec![5.0], (1, 1), true);
    let m = tensor_mean(&a);
    assert!((m.data()[0] - 5.0).abs() < 1e-6);
}

#[test]
fn test_tensor_mean_requires_grad_propagated() {
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let m = tensor_mean(&a);
    assert!(m.requires_grad());
}

#[test]
fn test_tensor_mean_no_grad_when_input_has_no_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let m = tensor_mean(&a);
    assert!(!m.requires_grad());
    assert!(m.0.borrow().grad_node.is_none());
}

#[test]
fn test_tensor_mean_records_mean_op() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let m = tensor_mean(&a);
    let inner = m.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(matches!(node.op, Op::Mean));
    assert_eq!(node.inputs.len(), 1);
}

#[test]
fn test_tensor_mean_non_uniform_values() {
    // mean([2.0, 4.0, 6.0]) = 4.0
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0], (1, 3), false);
    let m = tensor_mean(&a);
    assert!((m.data()[0] - 4.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "tensor_mean: empty tensor")]
fn test_tensor_mean_rejects_empty_tensor() {
    let a = Tensor::from_vec(vec![], (0, 1), true);
    let _ = tensor_mean(&a);
}

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
#[should_panic(expected = "tensor_mse_loss: input tensors must not be empty")]
fn test_tensor_mse_loss_rejects_empty_tensors() {
    let pred = Tensor::from_vec(vec![], (0, 1), true);
    let tgt = Tensor::from_vec(vec![], (0, 1), false);
    let _ = tensor_mse_loss(&pred, &tgt);
}

// -----------------------------------------------------------------------
// backward_op direct tests
// -----------------------------------------------------------------------

#[test]
fn test_backward_op_add_both_require_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let upstream = vec![1.0f32, 1.0];
    backward_op(&Op::Add, &[a.clone(), b.clone()], &upstream);
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga, vec![1.0f32, 1.0]);
    assert_eq!(gb, vec![1.0f32, 1.0]);
}

#[test]
fn test_backward_op_add_bias_sums_grad_along_batch() {
    // a: (2, 2), bias: (1, 2)
    // upstream: [[g00, g01], [g10, g11]] -> grad_bias = [g00+g10, g01+g11]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), false);
    let bias = Tensor::from_vec(vec![0.0, 0.0], (1, 2), true);
    let upstream = vec![1.0f32, 2.0, 3.0, 4.0]; // shape (2,2)
    backward_op(&Op::Add, &[a.clone(), bias.clone()], &upstream);
    let gb = bias.grad().unwrap();
    assert!((gb[0] - 4.0).abs() < 1e-6); // 1.0 + 3.0
    assert!((gb[1] - 6.0).abs() < 1e-6); // 2.0 + 4.0
}

#[test]
fn test_backward_op_sub_negates_b_gradient() {
    let a = Tensor::from_vec(vec![5.0, 6.0], (1, 2), true);
    let b = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let upstream = vec![1.0f32, 1.0];
    backward_op(&Op::Sub, &[a.clone(), b.clone()], &upstream);
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga, vec![1.0f32, 1.0]);
    assert_eq!(gb, vec![-1.0f32, -1.0]);
}

#[test]
fn test_backward_op_mul_cross_multiplies() {
    let a = Tensor::from_vec(vec![2.0, 3.0], (1, 2), true);
    let b = Tensor::from_vec(vec![4.0, 5.0], (1, 2), true);
    let upstream = vec![1.0f32, 1.0];
    backward_op(&Op::Mul, &[a.clone(), b.clone()], &upstream);
    // grad_a = upstream * b, grad_b = upstream * a
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert!((ga[0] - 4.0).abs() < 1e-6);
    assert!((ga[1] - 5.0).abs() < 1e-6);
    assert!((gb[0] - 2.0).abs() < 1e-6);
    assert!((gb[1] - 3.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_matmul_grad_a() {
    // a: (1,2), b: (2,1), out: (1,1)
    // grad_a = upstream @ b.T  = [1.0] @ [[3.0, 4.0]] = [3.0, 4.0]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), false);
    let upstream = vec![1.0f32]; // (1,1)
    backward_op(
        &Op::MatMul { m: 1, k: 2, n: 1 },
        &[a.clone(), b.clone()],
        &upstream,
    );
    let ga = a.grad().unwrap();
    assert!((ga[0] - 3.0).abs() < 1e-6);
    assert!((ga[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_matmul_grad_b() {
    // a: (1,2), b: (2,1), out: (1,1)
    // grad_b = a.T @ upstream  = [[1.0],[2.0]] @ [[1.0]] = [[1.0],[2.0]]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), true);
    let upstream = vec![1.0f32];
    backward_op(
        &Op::MatMul { m: 1, k: 2, n: 1 },
        &[a.clone(), b.clone()],
        &upstream,
    );
    let gb = b.grad().unwrap();
    assert!((gb[0] - 1.0).abs() < 1e-6);
    assert!((gb[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_relu_passes_grad_where_positive() {
    // input: [-1.0, 0.0, 2.0]
    // upstream: [1.0, 1.0, 1.0]
    // expected grad: [0.0, 0.0, 1.0]
    let a = Tensor::from_vec(vec![-1.0, 0.0, 2.0], (1, 3), true);
    let upstream = vec![1.0f32, 1.0, 1.0];
    backward_op(&Op::ReLU, &[a.clone()], &upstream);
    let ga = a.grad().unwrap();
    assert!((ga[0] - 0.0).abs() < 1e-6);
    assert!((ga[1] - 0.0).abs() < 1e-6);
    assert!((ga[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_sigmoid_derivative() {
    // sigmoid(0.0) = 0.5; derivative = s*(1-s) = 0.5 * 0.5 = 0.25
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let cached_sigmoid = Tensor::from_vec(vec![0.5], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::Sigmoid, &[a.clone(), cached_sigmoid], &upstream);
    let ga = a.grad().unwrap();
    assert!((ga[0] - 0.25).abs() < 1e-6);
}

#[test]
fn test_backward_op_tanh_derivative() {
    // tanh(0.0) = 0.0; derivative = 1 - 0^2 = 1.0
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let cached_tanh = Tensor::from_vec(vec![0.0], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::Tanh, &[a.clone(), cached_tanh], &upstream);
    let ga = a.grad().unwrap();
    assert!((ga[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_softmax_ce_grad_shape_and_sum() {
    // 1 example, 3 classes, label=0
    // softmax approx [0.7, 0.2, 0.1]; one_hot=[1,0,0]
    // grad = (softmax - one_hot) / batch * upstream
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.0], (1, 3), true);
    // Compute actual softmax for [2,1,0]
    let max_v = 2.0f32;
    let exp_vals: Vec<f32> = vec![2.0f32, 1.0, 0.0]
        .iter()
        .map(|&x| (x - max_v).exp())
        .collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let softmax: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();
    let cached_softmax = Tensor::from_vec(softmax.clone(), (1, 3), false);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(
        &Op::SoftmaxCE,
        &[logits.clone(), cached_softmax, labels],
        &upstream,
    );
    let grad = logits.grad().unwrap();
    // Sum of gradients should be 0 (softmax - one_hot sums to 0)
    let grad_sum: f32 = grad.iter().sum();
    assert!(
        grad_sum.abs() < 1e-5,
        "grad sum should be ~0, got {}",
        grad_sum
    );
}

#[test]
fn test_backward_op_mse_grad_predictions() {
    // pred=[2.0], tgt=[0.0], n=1
    // grad = 2*(pred-tgt)/n * upstream = 2*(2-0)/1 * 1 = 4.0
    let predictions = Tensor::from_vec(vec![2.0], (1, 1), true);
    let targets = Tensor::from_vec(vec![0.0], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::MSE, &[predictions.clone(), targets.clone()], &upstream);
    let gp = predictions.grad().unwrap();
    assert!((gp[0] - 4.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_sum_broadcasts_scalar_grad() {
    // Sum of 4 elements; upstream = [3.0]
    // Each element receives 3.0
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let upstream = vec![3.0f32];
    backward_op(&Op::Sum, &[a.clone()], &upstream);
    let ga = a.grad().unwrap();
    assert_eq!(ga, vec![3.0f32, 3.0, 3.0, 3.0]);
}

#[test]
fn test_backward_op_mean_divides_by_n() {
    // Mean of 4 elements; upstream = [4.0]
    // Each element receives 4.0 / 4 = 1.0
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let upstream = vec![4.0f32];
    backward_op(&Op::Mean, &[a.clone()], &upstream);
    let ga = a.grad().unwrap();
    assert_eq!(ga, vec![1.0f32, 1.0, 1.0, 1.0]);
}

#[test]
fn test_backward_op_skips_no_grad_inputs() {
    // Only `b` requires grad; `a` should not have any gradient accumulated.
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let upstream = vec![1.0f32, 1.0];
    backward_op(&Op::Add, &[a.clone(), b.clone()], &upstream);
    assert!(a.grad().is_none()); // a has no grad, should be untouched
    let gb = b.grad().unwrap();
    assert_eq!(gb, vec![1.0f32, 1.0]);
}

// -----------------------------------------------------------------------
// End-to-end backward() tests
// -----------------------------------------------------------------------

#[test]
fn test_end_to_end_mse_backward() {
    // pred=[2.0], tgt=[1.0]
    // loss = (2-1)^2 / 1 = 1.0
    // grad_pred = 2*(2-1)/1 = 2.0
    let pred = Tensor::from_vec(vec![2.0], (1, 1), true);
    let tgt = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!((loss.data()[0] - 1.0).abs() < 1e-6);
    loss.backward();
    let g = pred.grad().unwrap();
    assert!((g[0] - 2.0).abs() < 1e-5, "grad = {}", g[0]);
}

#[test]
fn test_end_to_end_relu_backward() {
    // ReLU([1.0, -1.0]) -> [1.0, 0.0]
    // sum -> 1.0
    // backward: grad_relu_out = [1.0, 1.0] (from sum)
    //           grad_input = [1.0, 0.0]  (mask: a[0]>0, a[1]<=0)
    let a = Tensor::from_vec(vec![1.0, -1.0], (1, 2), true);
    let relu_out = tensor_relu(&a);
    let loss = tensor_sum(&relu_out);
    loss.backward();
    let g = a.grad().unwrap();
    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((g[1] - 0.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_add_backward_both_inputs() {
    // a + b = out; sum(out); backward
    // Both a and b should get gradient = 1.0 per element
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let out = tensor_add(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga, vec![1.0f32, 1.0]);
    assert_eq!(gb, vec![1.0f32, 1.0]);
}

#[test]
fn test_end_to_end_mul_scalar_backward() {
    // a * 3.0 = out; sum(out); backward
    // grad_a = 3.0 per element
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let out = tensor_mul_scalar(&a, 3.0);
    let loss = tensor_sum(&out);
    loss.backward();
    let g = a.grad().unwrap();
    assert!((g[0] - 3.0).abs() < 1e-6);
    assert!((g[1] - 3.0).abs() < 1e-6);
    assert!((g[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_sigmoid_backward() {
    // sigmoid(0.0) = 0.5; loss = sum = 0.5
    // grad: s*(1-s) = 0.5 * 0.5 = 0.25
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let sig_out = tensor_sigmoid(&a);
    let loss = tensor_sum(&sig_out);
    loss.backward();
    let g = a.grad().unwrap();
    assert!((g[0] - 0.25).abs() < 1e-6, "grad = {}", g[0]);
}

#[test]
fn test_end_to_end_tanh_backward() {
    // tanh(0.0) = 0.0; loss = sum = 0.0
    // grad: 1 - 0^2 = 1.0
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let tanh_out = tensor_tanh(&a);
    let loss = tensor_sum(&tanh_out);
    loss.backward();
    let g = a.grad().unwrap();
    assert!((g[0] - 1.0).abs() < 1e-6, "grad = {}", g[0]);
}

#[test]
fn test_end_to_end_matmul_backward_grad_a() {
    // a: (1,2) = [1.0, 2.0]
    // b: (2,1) = [3.0, 4.0]
    // out: (1,1) = [1*3 + 2*4] = [11.0]
    // loss = sum(out) = 11.0
    // grad_a = upstream @ b.T = [1.0] @ [[3.0,4.0]] = [3.0, 4.0]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), false);
    let out = tensor_matmul(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let ga = a.grad().unwrap();
    assert!((ga[0] - 3.0).abs() < 1e-6);
    assert!((ga[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_softmax_ce_backward_gradient_sum_is_zero() {
    // For softmax CE with correct class 0, the sum of gradients for a row
    // should be 0 (since sum(softmax) - sum(one_hot) = 1 - 1 = 0).
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    loss.backward();
    let g = logits.grad().unwrap();
    let sum: f32 = g.iter().sum();
    assert!(sum.abs() < 1e-5, "grad sum should be ~0, got {}", sum);
}

#[test]
fn test_end_to_end_add_bias_backward_sums_over_batch() {
    // a: (3, 2) all zeros; bias: (1, 2) = [1.0, 2.0]
    // out = a + bias (broadcast); sum(out); backward
    // grad_bias should be sum over batch dimension = 3 * upstream_per_element
    let a = Tensor::from_vec(vec![0.0; 6], (3, 2), false);
    let bias = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let out = tensor_add_bias(&a, &bias);
    let loss = tensor_sum(&out);
    loss.backward();
    let gb = bias.grad().unwrap();
    // Each bias element appears in 3 rows, so grad = 3 * 1.0 = 3.0
    assert!((gb[0] - 3.0).abs() < 1e-6, "gb[0] = {}", gb[0]);
    assert!((gb[1] - 3.0).abs() < 1e-6, "gb[1] = {}", gb[1]);
}

#[test]
fn test_end_to_end_mean_backward() {
    // mean([2.0, 4.0, 6.0, 8.0]) = 5.0
    // grad per element = 1.0 / 4 = 0.25
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], (2, 2), true);
    let m = tensor_mean(&a);
    m.backward();
    let g = a.grad().unwrap();
    for &gv in &g {
        assert!((gv - 0.25).abs() < 1e-6, "grad element = {}", gv);
    }
}

#[test]
fn test_end_to_end_sub_backward() {
    // a - b = out; sum(out); backward
    // grad_a = 1.0 per element; grad_b = -1.0 per element
    let a = Tensor::from_vec(vec![5.0, 3.0], (1, 2), true);
    let b = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let out = tensor_sub(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga, vec![1.0f32, 1.0]);
    assert_eq!(gb, vec![-1.0f32, -1.0]);
}
