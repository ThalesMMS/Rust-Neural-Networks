// Tests for the automatic differentiation engine.
//
// This file grows incrementally as each subtask is implemented.
// Phase 1 (subtask-1-1): tensor construction and basic accessor tests.
// Phase 2 (subtask-2-1): basic element-wise and scalar operations.
// Phase 6 (subtask-6-1): numerical gradient checking via finite differences.
// Phase 6 (subtask-6-2): XOR MLP integration test - loss decrease, numgrad, hand-coded backward.

use approx::assert_relative_eq;
use rust_neural_networks::autograd::tensor::Tensor;

// ============================================================================
// Tensor construction tests  (subtask-1-1)
// ============================================================================

#[test]
fn tensor_construction_new() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    assert_eq!(t.shape(), (2, 2));
    assert_eq!(t.data(), vec![1.0f32, 2.0, 3.0, 4.0]);
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
    assert_eq!(t.numel(), 4);
}

#[test]
fn tensor_construction_zeros() {
    let t = Tensor::zeros((3, 4));
    assert_eq!(t.shape(), (3, 4));
    assert_eq!(t.numel(), 12);
    assert_eq!(t.data(), vec![0.0f32; 12]);
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_from_vec_no_grad() {
    let t = Tensor::from_vec(vec![0.5f32, -0.3, 0.1], (1, 3), false);
    assert_eq!(t.shape(), (1, 3));
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_from_vec_with_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), true);
    assert_eq!(t.shape(), (1, 3));
    assert!(t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_accumulate_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), true);
    assert!(t.grad().is_none());

    t.accumulate_grad(&[0.1, 0.2, 0.3]);
    let g = t
        .grad()
        .expect("gradient should be Some after first accumulate_grad");
    assert!((g[0] - 0.1).abs() < 1e-6);
    assert!((g[1] - 0.2).abs() < 1e-6);
    assert!((g[2] - 0.3).abs() < 1e-6);

    // Second accumulate adds to the existing gradient
    t.accumulate_grad(&[0.1, 0.2, 0.3]);
    let g2 = t.grad().unwrap();
    assert!((g2[0] - 0.2).abs() < 1e-6);
    assert!((g2[1] - 0.4).abs() < 1e-6);
    assert!((g2[2] - 0.6).abs() < 1e-6);
}

#[test]
fn tensor_construction_zero_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), true);
    t.accumulate_grad(&[5.0, 6.0]);
    t.zero_grad();
    let g = t.grad().expect("grad should still be Some after zero_grad");
    assert_eq!(g, vec![0.0f32, 0.0]);
}

#[test]
fn tensor_construction_clone_shares_inner() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), true);
    let b = a.clone();
    // Accumulate via `a`; `b` shares the same inner, so must also see it.
    a.accumulate_grad(&[1.0, 1.0]);
    let g = b.grad().expect("b should share gradient with a");
    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((g[1] - 1.0).abs() < 1e-6);
}

#[test]
fn tensor_construction_row_vector() {
    let t = Tensor::new(vec![10.0, 20.0, 30.0], (1, 3));
    assert_eq!(t.shape(), (1, 3));
    assert_eq!(t.numel(), 3);
}

#[test]
fn tensor_construction_column_vector() {
    let t = Tensor::new(vec![10.0, 20.0, 30.0], (3, 1));
    assert_eq!(t.shape(), (3, 1));
    assert_eq!(t.numel(), 3);
}

#[test]
fn tensor_construction_scalar() {
    let t = Tensor::new(vec![42.0], (1, 1));
    assert_eq!(t.shape(), (1, 1));
    assert_eq!(t.numel(), 1);
    assert_eq!(t.data(), vec![42.0f32]);
}

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

// ============================================================================
// Loss function tests  (subtask-3-2)
// ============================================================================

#[test]
fn loss_softmax_ce_scalar_output() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // batch=2, 3 classes; true labels = [0, 2]
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.5, 0.3, 0.8, 2.5], (2, 3), true);
    let labels = Tensor::from_vec(vec![0.0, 2.0], (2, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert_eq!(loss.shape(), (1, 1));
    assert!(
        loss.data()[0] >= 0.0,
        "cross-entropy loss must be non-negative"
    );
}

#[test]
fn loss_softmax_ce_requires_grad_from_logits() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    let logits = Tensor::from_vec(vec![1.0, 0.5, 0.2], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(loss.requires_grad());
}

#[test]
fn loss_softmax_ce_no_grad_when_logits_no_grad() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    let logits = Tensor::from_vec(vec![1.0, 0.5, 0.2], (1, 3), false);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(!loss.requires_grad());
    assert!(loss.0.borrow().grad_node.is_none());
}

#[test]
fn loss_softmax_ce_records_softmaxce_op_with_three_inputs() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    use rust_neural_networks::autograd::tape::Op;
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![2.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    let inner = loss.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node must be set");
    assert!(matches!(node.op, Op::SoftmaxCE));
    // inputs[0]=logits, inputs[1]=cached softmax, inputs[2]=labels
    assert_eq!(node.inputs.len(), 3);
    assert!(node.inputs[0].requires_grad());
    assert!(!node.inputs[1].requires_grad()); // cached softmax has no grad
    assert!(!node.inputs[2].requires_grad()); // labels have no grad
}

#[test]
fn loss_softmax_ce_cached_softmax_sums_to_one() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // batch=2, 4 classes
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], (2, 4), true);
    let labels = Tensor::from_vec(vec![3.0, 0.0], (2, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    let inner = loss.0.borrow();
    let node = inner.grad_node.as_ref().unwrap();
    // inputs[1] is cached softmax shape (2, 4)
    let softmax_data = node.inputs[1].data();
    // Each row of the softmax must sum to 1.
    let row0_sum: f32 = softmax_data[..4].iter().sum();
    let row1_sum: f32 = softmax_data[4..].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < 1e-5,
        "row0 softmax sum = {}",
        row0_sum
    );
    assert!(
        (row1_sum - 1.0).abs() < 1e-5,
        "row1 softmax sum = {}",
        row1_sum
    );
}

#[test]
fn loss_softmax_ce_numerical_stability_large_logits() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // Very large logits — without the log-sum-exp trick these would overflow.
    let logits = Tensor::from_vec(vec![1000.0, 999.0, 998.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(
        loss.data()[0].is_finite(),
        "loss must be finite even with large logits"
    );
}

#[test]
fn loss_softmax_ce_perfect_prediction_is_small() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // When the correct class has a very high logit, the loss should be near zero.
    let logits = Tensor::from_vec(vec![100.0, -100.0, -100.0], (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    assert!(loss.data()[0] < 1e-4, "loss = {}", loss.data()[0]);
}

#[test]
fn loss_mse_forward_zero_when_perfect() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let tgt = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert_eq!(loss.shape(), (1, 1));
    assert!((loss.data()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn loss_mse_forward_value() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    // pred=[1,3], tgt=[0,0] → ((1-0)^2 + (3-0)^2) / 2 = (1+9)/2 = 5
    let pred = Tensor::from_vec(vec![1.0, 3.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(
        (loss.data()[0] - 5.0).abs() < 1e-5,
        "expected 5.0, got {}",
        loss.data()[0]
    );
}

#[test]
fn loss_mse_requires_grad_from_predictions() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(loss.requires_grad());
}

#[test]
fn loss_mse_no_grad_when_no_inputs_require_grad() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let tgt = Tensor::from_vec(vec![0.5, 1.5], (1, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert!(!loss.requires_grad());
    assert!(loss.0.borrow().grad_node.is_none());
}

#[test]
fn loss_mse_records_mse_op() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    use rust_neural_networks::autograd::tape::Op;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0, 0.0], (1, 3), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    let inner = loss.0.borrow();
    let node = inner.grad_node.as_ref().expect("grad_node must be set");
    assert!(matches!(node.op, Op::MSE));
    assert_eq!(node.inputs.len(), 2);
    assert!(node.inputs[0].requires_grad());
    assert!(!node.inputs[1].requires_grad());
}

#[test]
fn loss_mse_shape_preserved_scalar_output() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], (2, 2), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    assert_eq!(loss.shape(), (1, 1));
}

// ============================================================================
// Backward pass tests  (subtask-4-1)
// ============================================================================

#[test]
fn backward_initializes_root_grad_to_ones() {
    // Calling backward() on a scalar leaf sets its gradient to [1.0].
    let t = Tensor::from_vec(vec![42.0], (1, 1), true);
    t.backward();
    let g = t.grad().expect("grad should be Some after backward");
    assert_eq!(g.len(), 1);
    assert!((g[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", g[0]);
}

#[test]
fn backward_add_propagates_to_both_inputs() {
    use rust_neural_networks::autograd::ops::tensor_add;
    // loss = a + b (scalar)
    let a = Tensor::from_vec(vec![3.0], (1, 1), true);
    let b = Tensor::from_vec(vec![2.0], (1, 1), true);
    let loss = tensor_add(&a, &b);
    loss.backward();

    // d(a+b)/da = 1, d(a+b)/db = 1
    let ga = a.grad().expect("a must have gradient");
    let gb = b.grad().expect("b must have gradient");
    assert!(
        (ga[0] - 1.0).abs() < 1e-6,
        "grad_a expected 1.0, got {}",
        ga[0]
    );
    assert!(
        (gb[0] - 1.0).abs() < 1e-6,
        "grad_b expected 1.0, got {}",
        gb[0]
    );
}

#[test]
fn backward_sub_propagates_with_negation() {
    use rust_neural_networks::autograd::ops::tensor_sub;
    // loss = a - b (scalar)
    let a = Tensor::from_vec(vec![5.0], (1, 1), true);
    let b = Tensor::from_vec(vec![2.0], (1, 1), true);
    let loss = tensor_sub(&a, &b);
    loss.backward();

    // d(a-b)/da = 1, d(a-b)/db = -1
    let ga = a.grad().expect("a must have gradient");
    let gb = b.grad().expect("b must have gradient");
    assert!(
        (ga[0] - 1.0).abs() < 1e-6,
        "grad_a expected 1.0, got {}",
        ga[0]
    );
    assert!(
        (gb[0] - (-1.0)).abs() < 1e-6,
        "grad_b expected -1.0, got {}",
        gb[0]
    );
}

#[test]
fn backward_mul_scalar_propagates() {
    use rust_neural_networks::autograd::ops::tensor_mul_scalar;
    // loss = a * 3.0 (scalar)
    let a = Tensor::from_vec(vec![4.0], (1, 1), true);
    let loss = tensor_mul_scalar(&a, 3.0);
    loss.backward();

    // d(3a)/da = 3
    let ga = a.grad().expect("a must have gradient");
    assert!(
        (ga[0] - 3.0).abs() < 1e-6,
        "grad_a expected 3.0, got {}",
        ga[0]
    );
}

#[test]
fn backward_relu_masks_negative_inputs() {
    use rust_neural_networks::autograd::ops::{tensor_relu, tensor_sum};
    // out = relu([−1, 2, −0.5, 3]), then sum to scalar
    let a = Tensor::from_vec(vec![-1.0, 2.0, -0.5, 3.0], (1, 4), true);
    let relu_out = tensor_relu(&a);
    let loss = tensor_sum(&relu_out);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    // Positive inputs pass gradient through; negative are zeroed.
    assert!(
        (ga[0] - 0.0).abs() < 1e-6,
        "grad[0] expected 0.0, got {}",
        ga[0]
    );
    assert!(
        (ga[1] - 1.0).abs() < 1e-6,
        "grad[1] expected 1.0, got {}",
        ga[1]
    );
    assert!(
        (ga[2] - 0.0).abs() < 1e-6,
        "grad[2] expected 0.0, got {}",
        ga[2]
    );
    assert!(
        (ga[3] - 1.0).abs() < 1e-6,
        "grad[3] expected 1.0, got {}",
        ga[3]
    );
}

#[test]
fn backward_sigmoid_gradient() {
    use rust_neural_networks::autograd::ops::tensor_sigmoid;
    // loss = sigmoid(0.0) = 0.5; upstream = 1.0
    // grad = 1.0 * 0.5 * 0.5 = 0.25
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let loss = tensor_sigmoid(&a);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    assert!(
        (ga[0] - 0.25).abs() < 1e-5,
        "grad expected 0.25, got {}",
        ga[0]
    );
}

#[test]
fn backward_tanh_gradient() {
    use rust_neural_networks::autograd::ops::tensor_tanh;
    // loss = tanh(0.0) = 0.0; upstream = 1.0
    // grad = 1.0 * (1 - 0^2) = 1.0
    let a = Tensor::from_vec(vec![0.0], (1, 1), true);
    let loss = tensor_tanh(&a);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    assert!(
        (ga[0] - 1.0).abs() < 1e-5,
        "grad expected 1.0, got {}",
        ga[0]
    );
}

#[test]
fn backward_sum_broadcasts_scalar_grad() {
    use rust_neural_networks::autograd::ops::tensor_sum;
    // loss = sum([1, 2, 3]) = 6; every input element gets grad = 1.0
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let loss = tensor_sum(&a);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    for (i, &g) in ga.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "grad[{}] expected 1.0, got {}",
            i,
            g
        );
    }
}

#[test]
fn backward_mean_scales_grad_by_one_over_n() {
    use rust_neural_networks::autograd::ops::tensor_mean;
    // loss = mean([2, 4, 6]) = 4; every input element gets grad = 1/3
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0], (1, 3), true);
    let loss = tensor_mean(&a);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    let expected = 1.0f32 / 3.0;
    for (i, &g) in ga.iter().enumerate() {
        assert!(
            (g - expected).abs() < 1e-6,
            "grad[{}] expected {}, got {}",
            i,
            expected,
            g
        );
    }
}

#[test]
fn backward_matmul_grad_a() {
    use rust_neural_networks::autograd::ops::{tensor_matmul, tensor_sum};
    // A (1×2) @ B (2×1) = C (1×1); dL/dA = dL/dC * B.T
    // A = [1, 2], B = [3; 4], C = [1*3 + 2*4] = [11]
    // dL/dC = 1 (after sum backward), dL/dA = [3, 4]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), false);
    let c = tensor_matmul(&a, &b);
    let loss = tensor_sum(&c);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    assert!(
        (ga[0] - 3.0).abs() < 1e-5,
        "grad_a[0] expected 3.0, got {}",
        ga[0]
    );
    assert!(
        (ga[1] - 4.0).abs() < 1e-5,
        "grad_a[1] expected 4.0, got {}",
        ga[1]
    );
}

#[test]
fn backward_matmul_grad_b() {
    use rust_neural_networks::autograd::ops::{tensor_matmul, tensor_sum};
    // A (1×2) @ B (2×1) = C (1×1); dL/dB = A.T * dL/dC
    // A = [1, 2], B = [3; 4], C = [11]
    // dL/dC = 1, dL/dB = [[1]; [2]]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), false);
    let b = Tensor::from_vec(vec![3.0, 4.0], (2, 1), true);
    let c = tensor_matmul(&a, &b);
    let loss = tensor_sum(&c);
    loss.backward();

    let gb = b.grad().expect("b must have gradient");
    assert!(
        (gb[0] - 1.0).abs() < 1e-5,
        "grad_b[0] expected 1.0, got {}",
        gb[0]
    );
    assert!(
        (gb[1] - 2.0).abs() < 1e-5,
        "grad_b[1] expected 2.0, got {}",
        gb[1]
    );
}

#[test]
fn backward_mse_loss_gradient() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    // pred = [2.0], tgt = [1.0], loss = (2-1)^2 / 1 = 1.0
    // grad_pred = 2*(pred-tgt)/n = 2*(2-1)/1 = 2.0
    let pred = Tensor::from_vec(vec![2.0], (1, 1), true);
    let tgt = Tensor::from_vec(vec![1.0], (1, 1), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    loss.backward();

    let gp = pred.grad().expect("pred must have gradient");
    assert!(
        (gp[0] - 2.0).abs() < 1e-5,
        "grad_pred expected 2.0, got {}",
        gp[0]
    );
}

#[test]
fn backward_softmax_ce_gradient() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // batch=1, 2 classes; logits = [2.0, 0.0], label = 0
    // softmax = [e^2/(e^2+1), 1/(e^2+1)]
    // grad_logits[0] = (softmax[0] - 1) / 1
    // grad_logits[1] = (softmax[1] - 0) / 1
    let logits = Tensor::from_vec(vec![2.0, 0.0], (1, 2), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    loss.backward();

    let g = logits.grad().expect("logits must have gradient");
    // grad[0] should be negative (correct class, push probability up → gradient is negative of residual)
    // grad[1] should be positive (wrong class, push probability down)
    assert!(
        g[0] < 0.0,
        "grad for correct class should be negative, got {}",
        g[0]
    );
    assert!(
        g[1] > 0.0,
        "grad for wrong class should be positive, got {}",
        g[1]
    );
    // Sum of gradients should be zero (softmax is normalised).
    let sum: f32 = g.iter().sum();
    assert!(sum.abs() < 1e-5, "gradient sum should be ~0, got {}", sum);
}

#[test]
fn backward_chained_ops_linear() {
    use rust_neural_networks::autograd::ops::{tensor_add, tensor_mul_scalar, tensor_sum};
    // loss = sum(a * 2 + b)  where a = [1, 2], b = [3, 4]
    // grad_a = 2 (from mul_scalar) * 1 (from sum) = 2
    // grad_b = 1 (from add) * 1 (from sum) = 1
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let a2 = tensor_mul_scalar(&a, 2.0);
    let added = tensor_add(&a2, &b);
    let loss = tensor_sum(&added);
    loss.backward();

    let ga = a.grad().expect("a must have gradient");
    let gb = b.grad().expect("b must have gradient");
    for (i, &g) in ga.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-5,
            "grad_a[{}] expected 2.0, got {}",
            i,
            g
        );
    }
    for (i, &g) in gb.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-5,
            "grad_b[{}] expected 1.0, got {}",
            i,
            g
        );
    }
}

#[test]
fn backward_add_bias_sums_grad_over_batch() {
    use rust_neural_networks::autograd::ops::{tensor_add_bias, tensor_sum};
    // a = [[1,2],[3,4]] (2×2), bias = [10, 20] (1×2)
    // loss = sum(a + bias_broadcast)
    // grad_bias = sum over batch rows: [1+1, 1+1] = [2, 2]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), false);
    let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
    let out = tensor_add_bias(&a, &bias);
    let loss = tensor_sum(&out);
    loss.backward();

    let gb = bias.grad().expect("bias must have gradient");
    // Each bias element receives gradient from both rows → 1+1 = 2
    assert!(
        (gb[0] - 2.0).abs() < 1e-5,
        "grad_bias[0] expected 2.0, got {}",
        gb[0]
    );
    assert!(
        (gb[1] - 2.0).abs() < 1e-5,
        "grad_bias[1] expected 2.0, got {}",
        gb[1]
    );
}

// ============================================================================
// Numerical gradient checking  (subtask-6-1)
//
// For each differentiable operation we compare the analytical gradient
// produced by autograd `.backward()` against the finite-difference
// approximation:
//
//   numerical_grad[i] = (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)
//
// with ε = 1e-4 and tolerance 1e-3.
// ============================================================================

/// Compute the finite-difference numerical gradient of a scalar loss
/// with respect to each element of `x`.
///
/// `f` is a closure that rebuilds the computation graph from the
/// perturbed data slice and returns the scalar loss value.
fn numerical_gradient<F>(x: &[f32], eps: f32, mut f: F) -> Vec<f32>
where
    F: FnMut(&[f32]) -> f32,
{
    let mut grad = vec![0.0f32; x.len()];
    let mut x_mut = x.to_vec();
    for i in 0..x.len() {
        let orig = x_mut[i];
        x_mut[i] = orig + eps;
        let loss_plus = f(&x_mut);
        x_mut[i] = orig - eps;
        let loss_minus = f(&x_mut);
        x_mut[i] = orig;
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }
    grad
}

#[test]
fn numgrad_add_wrt_a() {
    use rust_neural_networks::autograd::ops::{tensor_add, tensor_sum};
    // loss = sum(a + b) for fixed b; analytical grad_a = [1, 1, 1].
    // Use small values so that f32 finite-difference error stays < 1e-3
    // (loss value must be < ~1.7 to maintain that accuracy with eps = 1e-4).
    let a_data = vec![0.1f32, 0.2, 0.3];
    let b_data = vec![0.1f32, 0.1, 0.1]; // total sum = 0.9

    // Analytical gradient via autograd.
    let a = Tensor::from_vec(a_data.clone(), (1, 3), true);
    let b = Tensor::from_vec(b_data.clone(), (1, 3), false);
    let out = tensor_add(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    // Numerical gradient via finite differences.
    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let b2 = Tensor::from_vec(b_data.clone(), (1, 3), false);
        let o = tensor_add(&a2, &b2);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_add_wrt_b() {
    use rust_neural_networks::autograd::ops::{tensor_add, tensor_sum};
    // Small values to keep loss < 1.7 for f32 precision.
    let a_data = vec![0.1f32, 0.2, 0.3];
    let b_data = vec![0.1f32, 0.1, 0.1];

    let a = Tensor::from_vec(a_data.clone(), (1, 3), false);
    let b = Tensor::from_vec(b_data.clone(), (1, 3), true);
    let out = tensor_add(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = b.grad().expect("b must have gradient");

    let numerical = numerical_gradient(&b_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(a_data.clone(), (1, 3), false);
        let b2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let o = tensor_add(&a2, &b2);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_mul_scalar() {
    use rust_neural_networks::autograd::ops::{tensor_mul_scalar, tensor_sum};
    // loss = sum(a * 3.0); analytical grad_a = [3, 3, 3].
    // Use small values so that scalar * sum stays < 1.7 for f32 precision.
    let a_data = vec![0.1f32, -0.1, 0.1]; // 3 * (0.1) = 0.3 total

    let a = Tensor::from_vec(a_data.clone(), (1, 3), true);
    let out = tensor_mul_scalar(&a, 3.0);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let o = tensor_mul_scalar(&a2, 3.0);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_relu() {
    use rust_neural_networks::autograd::ops::{tensor_relu, tensor_sum};
    // Test a mix of positive and negative values.
    // For positive x, grad = 1; for negative x, grad = 0.
    // Avoid x=0 (non-differentiable point). Use small positive values to keep
    // the loss sum < 1.7 for f32 precision with eps = 1e-4.
    let a_data = vec![0.3f32, -0.3, 0.4, -0.1]; // sum(relu) = 0.7

    let a = Tensor::from_vec(a_data.clone(), (1, 4), true);
    let out = tensor_relu(&a);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 4), false);
        let o = tensor_relu(&a2);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_sigmoid() {
    use rust_neural_networks::autograd::ops::{tensor_sigmoid, tensor_sum};
    // loss = sum(sigmoid(a)); grad_a[i] = sigmoid(a[i]) * (1 - sigmoid(a[i])).
    // Use negative inputs so sigmoid values are small (sum < 1.7).
    let a_data = vec![-2.0f32, -1.0, -0.5, 0.0]; // sum(sigmoid) ≈ 1.27

    let a = Tensor::from_vec(a_data.clone(), (1, 4), true);
    let out = tensor_sigmoid(&a);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 4), false);
        let o = tensor_sigmoid(&a2);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_tanh() {
    use rust_neural_networks::autograd::ops::{tensor_sum, tensor_tanh};
    // loss = sum(tanh(a)); grad_a[i] = 1 - tanh(a[i])^2
    let a_data = vec![0.0f32, 0.5, -0.5, 1.0];

    let a = Tensor::from_vec(a_data.clone(), (1, 4), true);
    let out = tensor_tanh(&a);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 4), false);
        let o = tensor_tanh(&a2);
        let s = tensor_sum(&o);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_matmul_wrt_a() {
    use rust_neural_networks::autograd::ops::{tensor_matmul, tensor_sum};
    // A (2×3) @ B (3×2) = C (2×2); loss = sum(C), dL/dA = ones @ B.T.
    // Use small values to keep sum(C) < 1.7 for f32 precision.
    let a_data = vec![0.1f32, 0.2, 0.3, 0.1, 0.2, 0.3]; // (2×3)
    let b_data = vec![0.1f32, 0.1, 0.2, 0.2, 0.3, 0.3]; // (3×2) sum(C) ≈ 0.56

    let a = Tensor::from_vec(a_data.clone(), (2, 3), true);
    let b = Tensor::from_vec(b_data.clone(), (3, 2), false);
    let c = tensor_matmul(&a, &b);
    let loss = tensor_sum(&c);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (2, 3), false);
        let b2 = Tensor::from_vec(b_data.clone(), (3, 2), false);
        let c2 = tensor_matmul(&a2, &b2);
        let s = tensor_sum(&c2);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_matmul_wrt_b() {
    use rust_neural_networks::autograd::ops::{tensor_matmul, tensor_sum};
    // A (2×3) @ B (3×2) = C (2×2); loss = sum(C), dL/dB = A.T @ ones.
    // Use small values to keep sum(C) < 1.7 for f32 precision.
    let a_data = vec![0.1f32, 0.2, 0.3, 0.1, 0.2, 0.3]; // (2×3)
    let b_data = vec![0.1f32, 0.1, 0.2, 0.2, 0.3, 0.3]; // (3×2) sum(C) ≈ 0.56

    let a = Tensor::from_vec(a_data.clone(), (2, 3), false);
    let b = Tensor::from_vec(b_data.clone(), (3, 2), true);
    let c = tensor_matmul(&a, &b);
    let loss = tensor_sum(&c);
    loss.backward();
    let analytical = b.grad().expect("b must have gradient");

    let numerical = numerical_gradient(&b_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(a_data.clone(), (2, 3), false);
        let b2 = Tensor::from_vec(x.to_vec(), (3, 2), false);
        let c2 = tensor_matmul(&a2, &b2);
        let s = tensor_sum(&c2);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_mse_loss() {
    use rust_neural_networks::autograd::ops::tensor_mse_loss;
    // loss = MSE(pred, tgt); grad_pred[i] = 2*(pred[i]-tgt[i]) / n
    let pred_data = vec![1.5f32, -0.5, 2.0];
    let tgt_data = vec![1.0f32, 0.0, 1.5];

    let pred = Tensor::from_vec(pred_data.clone(), (1, 3), true);
    let tgt = Tensor::from_vec(tgt_data.clone(), (1, 3), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    loss.backward();
    let analytical = pred.grad().expect("pred must have gradient");

    let numerical = numerical_gradient(&pred_data, 1e-4, |x| {
        let p2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let t2 = Tensor::from_vec(tgt_data.clone(), (1, 3), false);
        let l = tensor_mse_loss(&p2, &t2);
        l.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_softmax_ce() {
    use rust_neural_networks::autograd::ops::tensor_softmax_cross_entropy;
    // batch=1, 3 classes; label = 1
    let logits_data = vec![1.0f32, 2.0, 0.5];
    let label_data = vec![1.0f32];

    let logits = Tensor::from_vec(logits_data.clone(), (1, 3), true);
    let labels = Tensor::from_vec(label_data.clone(), (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    loss.backward();
    let analytical = logits.grad().expect("logits must have gradient");

    let numerical = numerical_gradient(&logits_data, 1e-4, |x| {
        let l2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let lbl2 = Tensor::from_vec(label_data.clone(), (1, 1), false);
        let loss2 = tensor_softmax_cross_entropy(&l2, &lbl2);
        loss2.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_chained_sigmoid_sum() {
    use rust_neural_networks::autograd::ops::{tensor_add, tensor_sigmoid, tensor_sum};
    // loss = sum(sigmoid(a + b)); gradients through chain rule
    let a_data = vec![0.3f32, -0.5, 1.0];
    let b_data = vec![0.1f32, 0.2, -0.3];

    let a = Tensor::from_vec(a_data.clone(), (1, 3), true);
    let b = Tensor::from_vec(b_data.clone(), (1, 3), false);
    let added = tensor_add(&a, &b);
    let sig = tensor_sigmoid(&added);
    let loss = tensor_sum(&sig);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    let numerical = numerical_gradient(&a_data, 1e-4, |x| {
        let a2 = Tensor::from_vec(x.to_vec(), (1, 3), false);
        let b2 = Tensor::from_vec(b_data.clone(), (1, 3), false);
        let added2 = tensor_add(&a2, &b2);
        let sig2 = tensor_sigmoid(&added2);
        let s = tensor_sum(&sig2);
        s.data()[0]
    });

    for (&a_g, &n_g) in analytical.iter().zip(numerical.iter()) {
        assert_relative_eq!(a_g, n_g, epsilon = 1e-3_f32);
    }
}

#[test]
fn numgrad_shape_check_matmul_2x2() {
    use rust_neural_networks::autograd::ops::{tensor_matmul, tensor_sum};
    // Verify shape is preserved after gradient computation for a 2×2 matmul.
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, 1.0, 1.5, 2.0];

    let a = Tensor::from_vec(a_data.clone(), (2, 2), true);
    let b = Tensor::from_vec(b_data.clone(), (2, 2), true);
    let c = tensor_matmul(&a, &b);
    assert_eq!(c.shape(), (2, 2));
    let loss = tensor_sum(&c);
    loss.backward();

    // Gradient of A should have the same shape as A.
    let ga = a.grad().expect("a must have gradient");
    assert_eq!(ga.len(), 4, "grad_a must have 4 elements (2×2)");
    // Gradient of B should have the same shape as B.
    let gb = b.grad().expect("b must have gradient");
    assert_eq!(gb.len(), 4, "grad_b must have 4 elements (2×2)");
}

#[test]
fn numgrad_scalar_mul_value_check() {
    use rust_neural_networks::autograd::ops::{tensor_mul_scalar, tensor_sum};
    // Explicit value: loss = sum(a * 5); grad_a = 5 everywhere.
    let a_data = vec![2.0f32, -1.0, 0.5];

    let a = Tensor::from_vec(a_data.clone(), (1, 3), true);
    let out = tensor_mul_scalar(&a, 5.0);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    for &g in &analytical {
        assert_relative_eq!(g, 5.0_f32, epsilon = 1e-6_f32);
    }
}

#[test]
fn numgrad_add_gradient_is_ones() {
    use rust_neural_networks::autograd::ops::{tensor_add, tensor_sum};
    // loss = sum(a + b); analytical grad_a = 1 everywhere.
    let a_data = vec![0.3f32, -0.7, 1.5, 2.2];
    let b_data = vec![0.1f32, 0.9, -0.4, 0.8];

    let a = Tensor::from_vec(a_data.clone(), (1, 4), true);
    let b = Tensor::from_vec(b_data.clone(), (1, 4), false);
    let out = tensor_add(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    let analytical = a.grad().expect("a must have gradient");

    for &g in &analytical {
        assert_relative_eq!(g, 1.0_f32, epsilon = 1e-6_f32);
    }
}

// ============================================================================
// XOR MLP integration tests (subtask-6-2)
//
// Builds a 2-4-1 MLP matching mlp_simple.rs architecture using autograd ops,
// trains it on XOR data, and verifies:
//   (1) loss decreases over multiple training steps,
//   (2) parameter gradients match numerical gradient checking,
//   (3) gradient values are close to those from the hand-coded backward pass.
//
// Architecture: input (4,2) -> W1 (2,4) + b1 (1,4) -> sigmoid
//                           -> W2 (4,1) + b2 (1,1) -> sigmoid -> MSE loss
// ============================================================================

// Helper: run the 2-4-1 XOR MLP forward (no grad) and return the scalar MSE loss.
// Used by the numerical gradient check.
fn xor_mlp_compute_loss(w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> f32 {
    use rust_neural_networks::autograd::ops::{
        tensor_add_bias, tensor_matmul, tensor_mse_loss, tensor_sigmoid,
    };
    let xor_x: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let xor_tgt: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    let x_t = Tensor::from_vec(xor_x, (4, 2), false);
    let tgt_t = Tensor::from_vec(xor_tgt, (4, 1), false);
    let w1_t = Tensor::from_vec(w1.to_vec(), (2, 4), false);
    let b1_t = Tensor::from_vec(b1.to_vec(), (1, 4), false);
    let w2_t = Tensor::from_vec(w2.to_vec(), (4, 1), false);
    let b2_t = Tensor::from_vec(b2.to_vec(), (1, 1), false);

    let z1 = tensor_add_bias(&tensor_matmul(&x_t, &w1_t), &b1_t);
    let h1 = tensor_sigmoid(&z1);
    let z2 = tensor_add_bias(&tensor_matmul(&h1, &w2_t), &b2_t);
    let out = tensor_sigmoid(&z2);
    let loss = tensor_mse_loss(&out, &tgt_t);
    loss.data()[0]
}

// Helper: hand-coded forward + backward pass for the 2-4-1 XOR MLP (f32, sigmoid, MSE).
// Returns (grad_w1, grad_b1, grad_w2, grad_b2) for the full 4-sample XOR batch.
// Used to validate that autograd produces the same gradients as manual backpropagation.
fn xor_mlp_hand_coded_grads(
    w1: &[f32], // (2, 4) row-major: w1[k * 4 + j]
    b1: &[f32], // (4,)
    w2: &[f32], // (4, 1) = (4,): w2[j]
    b2: &[f32], // (1,)
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let tgt: [f32; 4] = [0.0, 1.0, 1.0, 0.0];

    let n_in = 2usize;
    let n_hid = 4usize;
    let n_batch = 4usize;

    let sig = |v: f32| 1.0f32 / (1.0 + (-v).exp());

    // ---- Forward pass ----
    // z1[i*n_hid + j] = sum_k x[i][k] * w1[k*n_hid + j] + b1[j]
    let mut h1 = vec![0.0f32; n_batch * n_hid];
    let mut z1_cache = vec![0.0f32; n_batch * n_hid];
    for i in 0..n_batch {
        for j in 0..n_hid {
            let mut s = b1[j];
            for k in 0..n_in {
                s += x[i][k] * w1[k * n_hid + j];
            }
            z1_cache[i * n_hid + j] = s;
            h1[i * n_hid + j] = sig(s);
        }
    }

    // z2[i] = sum_j h1[i][j] * w2[j] + b2[0]
    let mut out = vec![0.0f32; n_batch];
    for i in 0..n_batch {
        let mut s = b2[0];
        for j in 0..n_hid {
            s += h1[i * n_hid + j] * w2[j];
        }
        out[i] = sig(s);
    }

    // ---- Backward pass ----
    // MSE loss = mean((out - tgt)^2); upstream seed = 1.0
    // grad_out[i] = 1.0 * 2 * (out[i] - tgt[i]) / n_batch
    let mut grad_out = vec![0.0f32; n_batch];
    for i in 0..n_batch {
        grad_out[i] = 2.0 * (out[i] - tgt[i]) / n_batch as f32;
    }

    // Sigmoid backward at output: grad_z2[i] = grad_out[i] * out[i] * (1 - out[i])
    let mut grad_z2 = vec![0.0f32; n_batch];
    for i in 0..n_batch {
        grad_z2[i] = grad_out[i] * out[i] * (1.0 - out[i]);
    }

    // grad_w2[j] = sum_i h1[i][j] * grad_z2[i]   (MatMul backward w.r.t. W2)
    let mut grad_w2 = vec![0.0f32; n_hid];
    for j in 0..n_hid {
        for i in 0..n_batch {
            grad_w2[j] += h1[i * n_hid + j] * grad_z2[i];
        }
    }

    // grad_b2[0] = sum_i grad_z2[i]   (bias broadcast backward)
    let grad_b2 = vec![grad_z2.iter().sum::<f32>()];

    // grad_h1[i][j] = grad_z2[i] * w2[j]   (MatMul backward w.r.t. h1)
    let mut grad_h1 = vec![0.0f32; n_batch * n_hid];
    for i in 0..n_batch {
        for j in 0..n_hid {
            grad_h1[i * n_hid + j] = grad_z2[i] * w2[j];
        }
    }

    // Sigmoid backward at h1: grad_z1[i][j] = grad_h1[i][j] * h1[i][j] * (1 - h1[i][j])
    let mut grad_z1 = vec![0.0f32; n_batch * n_hid];
    for i in 0..n_batch {
        for j in 0..n_hid {
            let h = h1[i * n_hid + j];
            grad_z1[i * n_hid + j] = grad_h1[i * n_hid + j] * h * (1.0 - h);
        }
    }

    // grad_w1[k*n_hid + j] = sum_i x[i][k] * grad_z1[i][j]   (MatMul backward w.r.t. W1)
    let mut grad_w1 = vec![0.0f32; n_in * n_hid];
    for k in 0..n_in {
        for j in 0..n_hid {
            for i in 0..n_batch {
                grad_w1[k * n_hid + j] += x[i][k] * grad_z1[i * n_hid + j];
            }
        }
    }

    // grad_b1[j] = sum_i grad_z1[i][j]   (bias broadcast backward)
    let mut grad_b1 = vec![0.0f32; n_hid];
    for j in 0..n_hid {
        for i in 0..n_batch {
            grad_b1[j] += grad_z1[i * n_hid + j];
        }
    }

    (grad_w1, grad_b1, grad_w2, grad_b2)
}

// (1) Train the 2-4-1 MLP on XOR for 30 SGD steps and verify loss decreases.
#[test]
fn xor_mlp_loss_decreases() {
    use rust_neural_networks::autograd::ops::{
        tensor_add_bias, tensor_matmul, tensor_mse_loss, tensor_sigmoid,
    };

    let xor_x: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let xor_tgt: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Fixed small initial weights for reproducibility.
    let mut w1 = vec![0.5f32, -0.3, 0.2, -0.4, -0.2, 0.4, 0.3, -0.5];
    let mut b1 = vec![0.1f32, -0.1, 0.1, -0.1];
    let mut w2 = vec![0.3f32, -0.3, 0.4, -0.4];
    let mut b2 = vec![0.1f32];

    let lr = 0.5f32;
    let num_steps = 30;
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..num_steps {
        let x_t = Tensor::from_vec(xor_x.clone(), (4, 2), false);
        let tgt_t = Tensor::from_vec(xor_tgt.clone(), (4, 1), false);
        let w1_t = Tensor::from_vec(w1.clone(), (2, 4), true);
        let b1_t = Tensor::from_vec(b1.clone(), (1, 4), true);
        let w2_t = Tensor::from_vec(w2.clone(), (4, 1), true);
        let b2_t = Tensor::from_vec(b2.clone(), (1, 1), true);

        // Forward: 2 → 4 → 1, sigmoid activations, MSE loss.
        let z1 = tensor_add_bias(&tensor_matmul(&x_t, &w1_t), &b1_t);
        let h1 = tensor_sigmoid(&z1);
        let z2 = tensor_add_bias(&tensor_matmul(&h1, &w2_t), &b2_t);
        let out = tensor_sigmoid(&z2);
        let loss = tensor_mse_loss(&out, &tgt_t);

        let loss_val = loss.data()[0];
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward();

        // SGD parameter update.
        for (p, g) in w1.iter_mut().zip(w1_t.grad().expect("w1 grad").iter()) {
            *p -= lr * g;
        }
        for (p, g) in b1.iter_mut().zip(b1_t.grad().expect("b1 grad").iter()) {
            *p -= lr * g;
        }
        for (p, g) in w2.iter_mut().zip(w2_t.grad().expect("w2 grad").iter()) {
            *p -= lr * g;
        }
        for (p, g) in b2.iter_mut().zip(b2_t.grad().expect("b2 grad").iter()) {
            *p -= lr * g;
        }
    }

    assert!(
        last_loss < first_loss,
        "XOR MLP loss should decrease after {} SGD steps: first={:.4}, last={:.4}",
        num_steps,
        first_loss,
        last_loss
    );
}

// (2) Verify that autograd parameter gradients match finite-difference numerical gradients.
#[test]
fn xor_mlp_gradients_match_numerical_check() {
    use rust_neural_networks::autograd::ops::{
        tensor_add_bias, tensor_matmul, tensor_mse_loss, tensor_sigmoid,
    };

    let w1_data = vec![0.5f32, -0.3, 0.2, -0.4, -0.2, 0.4, 0.3, -0.5];
    let b1_data = vec![0.1f32, -0.1, 0.1, -0.1];
    let w2_data = vec![0.3f32, -0.3, 0.4, -0.4];
    let b2_data = vec![0.1f32];

    let xor_x: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let xor_tgt: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Autograd backward to get analytical gradients.
    let x_t = Tensor::from_vec(xor_x, (4, 2), false);
    let tgt_t = Tensor::from_vec(xor_tgt, (4, 1), false);
    let w1_t = Tensor::from_vec(w1_data.clone(), (2, 4), true);
    let b1_t = Tensor::from_vec(b1_data.clone(), (1, 4), true);
    let w2_t = Tensor::from_vec(w2_data.clone(), (4, 1), true);
    let b2_t = Tensor::from_vec(b2_data.clone(), (1, 1), true);

    let z1 = tensor_add_bias(&tensor_matmul(&x_t, &w1_t), &b1_t);
    let h1 = tensor_sigmoid(&z1);
    let z2 = tensor_add_bias(&tensor_matmul(&h1, &w2_t), &b2_t);
    let out = tensor_sigmoid(&z2);
    let loss = tensor_mse_loss(&out, &tgt_t);
    loss.backward();

    let ag_gw1 = w1_t.grad().expect("w1 must have grad");
    let ag_gb1 = b1_t.grad().expect("b1 must have grad");
    let ag_gw2 = w2_t.grad().expect("w2 must have grad");
    let ag_gb2 = b2_t.grad().expect("b2 must have grad");

    // Numerical gradients via finite differences (ε = 1e-4).
    let eps = 1e-4f32;

    let num_gw1 = numerical_gradient(&w1_data, eps, |w1| {
        xor_mlp_compute_loss(w1, &b1_data, &w2_data, &b2_data)
    });
    for (i, (&ag, &ng)) in ag_gw1.iter().zip(num_gw1.iter()).enumerate() {
        assert!(
            (ag - ng).abs() < 3e-3_f32,
            "w1[{}]: autograd={:.6}, numerical={:.6}, |diff|={:.2e}",
            i,
            ag,
            ng,
            (ag - ng).abs()
        );
    }

    let num_gb1 = numerical_gradient(&b1_data, eps, |b1| {
        xor_mlp_compute_loss(&w1_data, b1, &w2_data, &b2_data)
    });
    for (i, (&ag, &ng)) in ag_gb1.iter().zip(num_gb1.iter()).enumerate() {
        assert!(
            (ag - ng).abs() < 3e-3_f32,
            "b1[{}]: autograd={:.6}, numerical={:.6}, |diff|={:.2e}",
            i,
            ag,
            ng,
            (ag - ng).abs()
        );
    }

    let num_gw2 = numerical_gradient(&w2_data, eps, |w2| {
        xor_mlp_compute_loss(&w1_data, &b1_data, w2, &b2_data)
    });
    for (i, (&ag, &ng)) in ag_gw2.iter().zip(num_gw2.iter()).enumerate() {
        assert!(
            (ag - ng).abs() < 3e-3_f32,
            "w2[{}]: autograd={:.6}, numerical={:.6}, |diff|={:.2e}",
            i,
            ag,
            ng,
            (ag - ng).abs()
        );
    }

    let num_gb2 = numerical_gradient(&b2_data, eps, |b2| {
        xor_mlp_compute_loss(&w1_data, &b1_data, &w2_data, b2)
    });
    for (i, (&ag, &ng)) in ag_gb2.iter().zip(num_gb2.iter()).enumerate() {
        assert!(
            (ag - ng).abs() < 3e-3_f32,
            "b2[{}]: autograd={:.6}, numerical={:.6}, |diff|={:.2e}",
            i,
            ag,
            ng,
            (ag - ng).abs()
        );
    }
}

// (3) Verify that autograd gradients exactly match the hand-coded backward pass.
#[test]
fn xor_mlp_gradients_match_hand_coded_backward() {
    use rust_neural_networks::autograd::ops::{
        tensor_add_bias, tensor_matmul, tensor_mse_loss, tensor_sigmoid,
    };

    let w1_data = vec![0.5f32, -0.3, 0.2, -0.4, -0.2, 0.4, 0.3, -0.5];
    let b1_data = vec![0.1f32, -0.1, 0.1, -0.1];
    let w2_data = vec![0.3f32, -0.3, 0.4, -0.4];
    let b2_data = vec![0.1f32];

    let xor_x: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let xor_tgt: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Autograd backward.
    let x_t = Tensor::from_vec(xor_x, (4, 2), false);
    let tgt_t = Tensor::from_vec(xor_tgt, (4, 1), false);
    let w1_t = Tensor::from_vec(w1_data.clone(), (2, 4), true);
    let b1_t = Tensor::from_vec(b1_data.clone(), (1, 4), true);
    let w2_t = Tensor::from_vec(w2_data.clone(), (4, 1), true);
    let b2_t = Tensor::from_vec(b2_data.clone(), (1, 1), true);

    let z1 = tensor_add_bias(&tensor_matmul(&x_t, &w1_t), &b1_t);
    let h1 = tensor_sigmoid(&z1);
    let z2 = tensor_add_bias(&tensor_matmul(&h1, &w2_t), &b2_t);
    let out = tensor_sigmoid(&z2);
    let loss = tensor_mse_loss(&out, &tgt_t);
    loss.backward();

    let ag_gw1 = w1_t.grad().expect("w1 must have grad");
    let ag_gb1 = b1_t.grad().expect("b1 must have grad");
    let ag_gw2 = w2_t.grad().expect("w2 must have grad");
    let ag_gb2 = b2_t.grad().expect("b2 must have grad");

    // Hand-coded backward pass for comparison.
    let (hc_gw1, hc_gb1, hc_gw2, hc_gb2) =
        xor_mlp_hand_coded_grads(&w1_data, &b1_data, &w2_data, &b2_data);

    // Both computations are f32 and use the same arithmetic order,
    // so differences should be at f32 machine precision (~1e-7).
    let tol = 1e-5_f32;
    for (i, (&ag, &hc)) in ag_gw1.iter().zip(hc_gw1.iter()).enumerate() {
        assert!(
            (ag - hc).abs() < tol,
            "w1[{}]: autograd={:.8}, hand-coded={:.8}, |diff|={:.2e}",
            i,
            ag,
            hc,
            (ag - hc).abs()
        );
    }
    for (i, (&ag, &hc)) in ag_gb1.iter().zip(hc_gb1.iter()).enumerate() {
        assert!(
            (ag - hc).abs() < tol,
            "b1[{}]: autograd={:.8}, hand-coded={:.8}, |diff|={:.2e}",
            i,
            ag,
            hc,
            (ag - hc).abs()
        );
    }
    for (i, (&ag, &hc)) in ag_gw2.iter().zip(hc_gw2.iter()).enumerate() {
        assert!(
            (ag - hc).abs() < tol,
            "w2[{}]: autograd={:.8}, hand-coded={:.8}, |diff|={:.2e}",
            i,
            ag,
            hc,
            (ag - hc).abs()
        );
    }
    for (i, (&ag, &hc)) in ag_gb2.iter().zip(hc_gb2.iter()).enumerate() {
        assert!(
            (ag - hc).abs() < tol,
            "b2[{}]: autograd={:.8}, hand-coded={:.8}, |diff|={:.2e}",
            i,
            ag,
            hc,
            (ag - hc).abs()
        );
    }
}
