use super::*;

fn assert_grad_node<F>(
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
    inspect(node);
}

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
    assert_grad_node(&out, 2, |op| matches!(op, Op::Add), |_| {});
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
    assert_grad_node(&out, 2, |op| matches!(op, Op::Sub), |_| {});
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
    assert_grad_node(
        &out,
        2,
        |op| matches!(op, Op::Mul),
        |node| {
            // Second input is the broadcast scalar tensor (no grad)
            let scalar = &node.inputs[1];
            assert!(!scalar.requires_grad());
            assert_eq!(scalar.shape(), (1, 2));
            assert_eq!(scalar.data(), vec![3.0f32, 3.0]);
        },
    );
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
    assert_grad_node(&out, 2, |op| matches!(op, Op::Add), |_| {});
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
