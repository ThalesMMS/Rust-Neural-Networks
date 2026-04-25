use super::*;

fn assert_activation_grad_node<F>(
    output: &Tensor,
    expected_inputs: usize,
    op_matches: impl FnOnce(&Op) -> bool,
    inspect: F,
) where
    F: FnOnce(&crate::autograd::tape::GradNode),
{
    assert!(output.requires_grad());
    let inner = output.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be set");
    assert!(op_matches(&node.op));
    assert_eq!(node.inputs.len(), expected_inputs);
    assert!(node.inputs[0].requires_grad());
    inspect(node);
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
    assert_activation_grad_node(
        &out,
        1,
        |op| matches!(op, Op::ReLU),
        |node| {
            // The cached input data matches the original
            assert_eq!(node.inputs[0].data(), vec![2.0f32, -1.0]);
        },
    );
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
    assert_activation_grad_node(
        &out,
        2,
        |op| matches!(op, Op::Sigmoid),
        |node| {
            // inputs[1] = cached sigmoid output (no grad)
            assert!(!node.inputs[1].requires_grad());
            let cached = node.inputs[1].data();
            assert!(cached[0] > 0.5 && cached[0] < 1.0);
        },
    );
}

// -----------------------------------------------------------------------
// tensor_tanh tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_tanh_forward_at_zero() {
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let out = tensor_tanh(&a);
    assert!(out.data()[0].abs() < 1e-6);
}

#[test]
fn test_tensor_tanh_range_is_minus_one_to_one() {
    // Use moderate inputs to stay away from saturation (tanh(±10) rounds to ±1.0 in f32).
    let a = Tensor::from_vec(vec![-2.0, 0.0, 2.0], (1, 3), true);
    let out = tensor_tanh(&a);
    let data = out.data();
    // tanh(-2) ≈ -0.964 — strictly in (-1, 0)
    assert!(data[0] > -1.0 && data[0] < 0.0);
    assert!(data[1].abs() < 1e-6);
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
    assert_activation_grad_node(
        &out,
        2,
        |op| matches!(op, Op::Tanh),
        |node| {
            // inputs[1] = cached tanh output (no grad)
            assert!(!node.inputs[1].requires_grad());
            // Cached tanh at 0 should be 0
            assert!(node.inputs[1].data()[0].abs() < 1e-6);
        },
    );
}

#[test]
fn test_tensor_tanh_shape_preserved() {
    let a = Tensor::from_vec(vec![1.0, -1.0, 0.0, 2.0], (2, 2), true);
    let out = tensor_tanh(&a);
    assert_eq!(out.shape(), (2, 2));
}
