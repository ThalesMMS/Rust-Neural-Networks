use super::*;

// ============================================================================
// Basic ops tests  (subtask-2-1)
// ============================================================================

#[test]
fn ops_add_forward_values() {
    use rust_neural_networks::autograd::ops::tensor_add;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (1, 3), true);
    let out = tensor_add(&a, &b);
    let data = out.data();
    assert_eq!(data.len(), 3);
    assert!((data[0] - 5.0).abs() < 1e-6);
    assert!((data[1] - 7.0).abs() < 1e-6);
    assert!((data[2] - 9.0).abs() < 1e-6);
}

#[test]
fn ops_add_shape_preserved() {
    use rust_neural_networks::autograd::ops::tensor_add;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let b = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], (2, 2), false);
    let out = tensor_add(&a, &b);
    assert_eq!(out.shape(), (2, 2));
}

#[test]
fn ops_add_requires_grad_propagated() {
    use rust_neural_networks::autograd::ops::tensor_add;
    // Either input requiring grad makes output require grad.
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), false);
    let out = tensor_add(&a, &b);
    assert!(out.requires_grad());
}

#[test]
fn ops_add_records_grad_node() {
    use rust_neural_networks::autograd::ops::tensor_add;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let out = tensor_add(&a, &b);
    assert!(out.requires_grad());
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node should be Some");
    assert!(matches!(node.op, Op::Add));
    assert_eq!(node.inputs.len(), 2);
}

#[test]
fn ops_add_no_grad_node_when_no_grad() {
    use rust_neural_networks::autograd::ops::tensor_add;
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), false);
    let out = tensor_add(&a, &b);
    assert!(!out.requires_grad());
    assert!(out.0.borrow().grad_node.is_none());
}

#[test]
fn ops_add_inputs_are_in_topo_graph() {
    use rust_neural_networks::autograd::ops::tensor_add;
    use rust_neural_networks::autograd::tape::GradNode;
    let a = Tensor::from_vec(vec![1.0], (1, 1), true);
    let b = Tensor::from_vec(vec![2.0], (1, 1), true);
    let out = tensor_add(&a, &b);
    let topo = GradNode::build_topo(&out);
    // Should have 3 nodes: out, a, b.
    assert_eq!(topo.len(), 3);
}

#[test]
fn ops_sub_forward_values() {
    use rust_neural_networks::autograd::ops::tensor_sub;
    let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], (1, 3), true);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
    let out = tensor_sub(&a, &b);
    let data = out.data();
    assert!((data[0] - 4.0).abs() < 1e-6);
    assert!((data[1] - 5.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
}

#[test]
fn ops_sub_records_sub_op() {
    use rust_neural_networks::autograd::ops::tensor_sub;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![3.0], (1, 1), true);
    let b = Tensor::from_vec(vec![1.0], (1, 1), true);
    let out = tensor_sub(&a, &b);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    assert!(matches!(node.op, Op::Sub));
}

#[test]
fn ops_mul_scalar_forward_values() {
    use rust_neural_networks::autograd::ops::tensor_mul_scalar;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let out = tensor_mul_scalar(&a, 2.0);
    let data = out.data();
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
}

#[test]
fn ops_mul_scalar_records_mul_op() {
    use rust_neural_networks::autograd::ops::tensor_mul_scalar;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let out = tensor_mul_scalar(&a, 3.0);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    assert!(matches!(node.op, Op::Mul));
    // Second input is the constant scalar tensor (no grad)
    assert!(!node.inputs[1].requires_grad());
    // Scalar tensor holds the broadcast scalar value
    assert_eq!(node.inputs[1].data(), vec![3.0f32, 3.0]);
}

#[test]
fn ops_add_bias_forward_values() {
    use rust_neural_networks::autograd::ops::tensor_add_bias;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
    let out = tensor_add_bias(&a, &bias);
    assert_eq!(out.shape(), (2, 2));
    let data = out.data();
    assert!((data[0] - 11.0).abs() < 1e-6);
    assert!((data[1] - 22.0).abs() < 1e-6);
    assert!((data[2] - 13.0).abs() < 1e-6);
    assert!((data[3] - 24.0).abs() < 1e-6);
}

#[test]
fn ops_add_bias_records_add_op() {
    use rust_neural_networks::autograd::ops::tensor_add_bias;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let bias = Tensor::from_vec(vec![0.5, 0.5], (1, 2), true);
    let out = tensor_add_bias(&a, &bias);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    assert!(matches!(node.op, Op::Add));
    assert_eq!(node.inputs.len(), 2);
}
