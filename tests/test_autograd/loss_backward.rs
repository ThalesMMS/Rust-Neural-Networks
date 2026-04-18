use super::*;

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
