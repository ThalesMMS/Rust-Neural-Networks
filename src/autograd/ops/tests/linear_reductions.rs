use super::*;

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
// Additional edge case tests
// -----------------------------------------------------------------------

#[test]
fn test_tensor_sum_all_zeros() {
    // sum of all-zero tensor is 0
    let a = Tensor::from_vec(vec![0.0; 6], (2, 3), true);
    let s = tensor_sum(&a);
    assert!(s.data()[0].abs() < 1e-6);
}

#[test]
fn test_tensor_sum_large_tensor() {
    // sum of 1.0 * 100 elements = 100.0
    let a = Tensor::from_vec(vec![1.0f32; 100], (10, 10), true);
    let s = tensor_sum(&a);
    assert!((s.data()[0] - 100.0).abs() < 1e-4);
}

#[test]
fn test_tensor_matmul_batch_dim_propagation() {
    // (4,3) @ (3,2) = (4,2): verify shape propagates in multi-row case
    let a = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        (4, 3),
        true,
    );
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), false);
    let out = tensor_matmul(&a, &b);
    assert_eq!(out.shape(), (4, 2));
    // Row 3: [1,1,1] @ [[1,0],[0,1],[1,1]] = [1+0+1, 0+1+1] = [2, 2]
    assert!((out.data()[6] - 2.0).abs() < 1e-6);
    assert!((out.data()[7] - 2.0).abs() < 1e-6);
}

#[test]
fn test_tensor_mean_large_batch() {
    // mean of 1..=10 = 5.5
    let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let a = Tensor::from_vec(data, (2, 5), true);
    let m = tensor_mean(&a);
    assert!((m.data()[0] - 5.5).abs() < 1e-5);
}
