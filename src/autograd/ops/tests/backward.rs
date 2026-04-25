use super::*;

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
    assert!(ga[0].abs() < 1e-6);
    assert!(ga[1].abs() < 1e-6);
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
    let expected = vec![softmax[0] - 1.0, softmax[1], softmax[2]];
    for (i, (&actual, &expected)) in grad.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "grad[{}] = {}, expected {}",
            i,
            actual,
            expected
        );
    }
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
    let targets = Tensor::from_vec(vec![0.0], (1, 1), true);
    let upstream = vec![1.0f32];
    backward_op(&Op::MSE, &[predictions.clone(), targets.clone()], &upstream);
    let gp = predictions.grad().unwrap();
    let gt = targets.grad().unwrap();
    assert!((gp[0] - 4.0).abs() < 1e-6);
    assert!((gt[0] + 4.0).abs() < 1e-6);
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

#[test]
fn test_backward_op_relu_all_negative_gives_zero_grad() {
    // All negative inputs → all zero gradients
    let a = Tensor::from_vec(vec![-3.0, -2.0, -1.0], (1, 3), true);
    let upstream = vec![5.0f32, 5.0, 5.0];
    backward_op(&Op::ReLU, &[a.clone()], &upstream);
    let ga = a.grad().unwrap();
    assert!(ga.iter().all(|&g| g.abs() < 1e-6));
}

#[test]
fn test_backward_op_sigmoid_saturated_near_one() {
    // sigmoid(10) ≈ 1.0; derivative ≈ 0
    let s = 1.0f32 / (1.0 + (-10.0f32).exp()); // ≈ 0.9999546
    let a = Tensor::from_vec(vec![10.0], (1, 1), true);
    let cached_sigmoid = Tensor::from_vec(vec![s], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::Sigmoid, &[a.clone(), cached_sigmoid], &upstream);
    let ga = a.grad().unwrap();
    // grad = s*(1-s) ≈ 0.9999546 * 0.0000454 ≈ 4.54e-5 (very small)
    assert!(ga[0] >= 0.0);
    assert!(ga[0] < 1e-3);
}

#[test]
fn test_backward_op_tanh_saturated_near_one() {
    // tanh(5) ≈ 0.9999; derivative ≈ 1 - 0.9999^2 ≈ 0.0002
    let t = 5.0f32.tanh(); // close to 1.0
    let a = Tensor::from_vec(vec![5.0], (1, 1), true);
    let cached_tanh = Tensor::from_vec(vec![t], (1, 1), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::Tanh, &[a.clone(), cached_tanh], &upstream);
    let ga = a.grad().unwrap();
    // derivative = 1 - t^2
    let expected = 1.0 - t * t;
    assert!((ga[0] - expected).abs() < 1e-5);
}

#[test]
fn test_backward_op_mul_with_non_unit_upstream() {
    // upstream = [3.0, 2.0]; a=[1,2], b=[3,4]
    // grad_a = upstream * b = [9, 8]; grad_b = upstream * a = [3, 4]
    let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    let upstream = vec![3.0f32, 2.0];
    backward_op(&Op::Mul, &[a.clone(), b.clone()], &upstream);
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert!((ga[0] - 9.0).abs() < 1e-6);
    assert!((ga[1] - 8.0).abs() < 1e-6);
    assert!((gb[0] - 3.0).abs() < 1e-6);
    assert!((gb[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_backward_op_mse_multi_element() {
    // pred=[1.0, 2.0, 3.0], tgt=[0.0, 0.0, 0.0], n=3
    // grad_pred[i] = 2*(pred[i]-tgt[i])/n * upstream
    // = [2/3, 4/3, 6/3] * 1.0
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0, 0.0], (1, 3), false);
    let upstream = vec![1.0f32];
    backward_op(&Op::MSE, &[pred.clone(), tgt.clone()], &upstream);
    let gp = pred.grad().unwrap();
    assert!((gp[0] - 2.0 / 3.0).abs() < 1e-5, "gp[0]={}", gp[0]);
    assert!((gp[1] - 4.0 / 3.0).abs() < 1e-5, "gp[1]={}", gp[1]);
    assert!((gp[2] - 6.0 / 3.0).abs() < 1e-5, "gp[2]={}", gp[2]);
}
