use super::*;

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
pub(super) fn numerical_gradient<F>(x: &[f32], eps: f32, mut f: F) -> Vec<f32>
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
