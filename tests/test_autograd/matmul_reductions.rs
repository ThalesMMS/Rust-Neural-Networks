use super::*;

// ============================================================================
// MatMul, Sum, Mean ops tests  (subtask-2-2)
// ============================================================================

#[test]
fn ops_matmul_forward_values() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    // [1 2; 3 4] @ [5 6; 7 8] = [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8]
    //                           = [19, 22; 43, 50]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], (2, 2), true);
    let out = tensor_matmul(&a, &b);
    assert_eq!(out.shape(), (2, 2));
    let data = out.data();
    assert!(
        (data[0] - 19.0).abs() < 1e-5,
        "expected 19.0, got {}",
        data[0]
    );
    assert!(
        (data[1] - 22.0).abs() < 1e-5,
        "expected 22.0, got {}",
        data[1]
    );
    assert!(
        (data[2] - 43.0).abs() < 1e-5,
        "expected 43.0, got {}",
        data[2]
    );
    assert!(
        (data[3] - 50.0).abs() < 1e-5,
        "expected 50.0, got {}",
        data[3]
    );
}

#[test]
fn ops_matmul_shape_mxk_times_kxn() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    // (2×3) @ (3×4) => (2×4)
    let a = Tensor::from_vec(vec![1.0; 6], (2, 3), true);
    let b = Tensor::from_vec(vec![1.0; 12], (3, 4), false);
    let out = tensor_matmul(&a, &b);
    assert_eq!(out.shape(), (2, 4));
    // Each element = 1+1+1 = 3 (sum of 3 ones)
    for v in out.data() {
        assert!((v - 3.0).abs() < 1e-6);
    }
}

#[test]
fn ops_matmul_identity_preserves_matrix() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    // identity (2×2) @ b should return b unchanged
    let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), true);
    let out = tensor_matmul(&eye, &b);
    let data = out.data();
    assert!((data[0] - 3.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
    assert!((data[2] - 5.0).abs() < 1e-6);
    assert!((data[3] - 6.0).abs() < 1e-6);
}

#[test]
fn ops_matmul_requires_grad_propagated() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    // If either input requires_grad, output should too.
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), false);
    let out = tensor_matmul(&a, &b);
    assert!(out.requires_grad());
}

#[test]
fn ops_matmul_no_grad_when_no_inputs_require_grad() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), false);
    let out = tensor_matmul(&a, &b);
    assert!(!out.requires_grad());
    assert!(out.0.borrow().grad_node.is_none());
}

#[test]
fn ops_matmul_records_matmul_op() {
    use rust_neural_networks::autograd::ops::tensor_matmul;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), true);
    let b = Tensor::from_vec(vec![1.0; 12], (3, 4), true);
    let out = tensor_matmul(&a, &b);
    let inner = out.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node must be set");
    assert!(
        matches!(node.op, Op::MatMul { m: 2, k: 3, n: 4 }),
        "expected Op::MatMul{{m:2, k:3, n:4}}, got {:?}",
        node.op
    );
    assert_eq!(node.inputs.len(), 2);
}

#[test]
fn ops_sum_forward_value() {
    use rust_neural_networks::autograd::ops::tensor_sum;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let s = tensor_sum(&a);
    assert_eq!(s.shape(), (1, 1));
    assert!((s.data()[0] - 10.0).abs() < 1e-6);
}

#[test]
fn ops_sum_records_sum_op() {
    use rust_neural_networks::autograd::ops::tensor_sum;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let s = tensor_sum(&a);
    assert!(s.requires_grad());
    let inner = s.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    assert!(matches!(node.op, Op::Sum));
    assert_eq!(node.inputs.len(), 1);
}

#[test]
fn ops_mean_forward_value() {
    use rust_neural_networks::autograd::ops::tensor_mean;
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let m = tensor_mean(&a);
    assert_eq!(m.shape(), (1, 1));
    assert!((m.data()[0] - 2.5).abs() < 1e-6);
}

#[test]
fn ops_mean_records_mean_op() {
    use rust_neural_networks::autograd::ops::tensor_mean;
    use rust_neural_networks::autograd::tape::Op;
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0], (1, 3), true);
    let m = tensor_mean(&a);
    assert!(m.requires_grad());
    let inner = m.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    assert!(matches!(node.op, Op::Mean));
    assert_eq!(node.inputs.len(), 1);
}
