use super::*;

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
    assert!(g[1].abs() < 1e-6);
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
    let logits_data = vec![2.0f32, 1.0, 0.0];
    let logits = Tensor::from_vec(logits_data.clone(), (1, 3), true);
    let labels = Tensor::from_vec(vec![0.0], (1, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    loss.backward();
    let g = logits.grad().unwrap();

    let max_logit = logits_data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits_data
        .iter()
        .map(|&value| (value - max_logit).exp())
        .collect();
    let exp_sum: f32 = exp_vals.iter().sum();
    let softmax: Vec<f32> = exp_vals.iter().map(|&value| value / exp_sum).collect();
    let expected = vec![softmax[0] - 1.0, softmax[1], softmax[2]];

    for (i, (&actual, &expected)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "grad[{}] = {}, expected {}",
            i,
            actual,
            expected
        );
    }
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

#[test]
fn test_end_to_end_matmul_backward_grad_b() {
    // a: (2,1), b: (1,2)
    // out: (2,2) = outer product
    // loss = sum(out) = (a[0]+a[1])*(b[0]+b[1])
    // grad_b = a.T @ upstream = [[1,1],[1,1]] summed column-wise
    let a = Tensor::from_vec(vec![2.0, 3.0], (2, 1), false);
    let b = Tensor::from_vec(vec![4.0, 5.0], (1, 2), true);
    let out = tensor_matmul(&a, &b);
    let loss = tensor_sum(&out);
    loss.backward();
    // grad_b = a.T @ ones(2,2) = [2+3, 2+3] = [5.0, 5.0]
    let gb = b.grad().unwrap();
    assert!((gb[0] - 5.0).abs() < 1e-5, "gb[0] = {}", gb[0]);
    assert!((gb[1] - 5.0).abs() < 1e-5, "gb[1] = {}", gb[1]);
}

#[test]
fn test_end_to_end_chained_ops_backward() {
    // Chain: (a + b) * 2.0 -> sum -> backward
    // out = (a + b) * 2, loss = sum(out)
    // grad_a = 2.0 per element, grad_b = 2.0 per element
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (1, 3), true);
    let added = tensor_add(&a, &b);
    let scaled = tensor_mul_scalar(&added, 2.0);
    let loss = tensor_sum(&scaled);
    loss.backward();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    for &g in &ga {
        assert!((g - 2.0).abs() < 1e-5, "ga element = {g}");
    }
    for &g in &gb {
        assert!((g - 2.0).abs() < 1e-5, "gb element = {g}");
    }
}

#[test]
fn test_end_to_end_softmax_ce_multi_batch_backward() {
    // 4 examples, 3 classes, all predictions correct
    let logits = Tensor::from_vec(
        vec![
            10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, 10.0, 10.0, -10.0, -10.0,
        ],
        (4, 3),
        true,
    );
    let labels = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], (4, 1), false);
    let loss = tensor_softmax_cross_entropy(&logits, &labels);
    // All predictions are near-perfect, loss should be very low
    assert!(loss.data()[0] < 1e-3, "loss = {}", loss.data()[0]);
    loss.backward();
    let g = logits.grad().unwrap();
    // Gradient sum per row should be ~0
    for row in 0..4 {
        let row_sum: f32 = g[row * 3..(row + 1) * 3].iter().sum();
        assert!(row_sum.abs() < 1e-4, "row {row} grad sum = {row_sum}");
    }
}

#[test]
fn test_end_to_end_mse_batch_backward() {
    // pred: (4,1), tgt: (4,1) - batch of predictions
    // MSE = mean over all 4 elements
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (4, 1), true);
    let tgt = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], (4, 1), false);
    let loss = tensor_mse_loss(&pred, &tgt);
    // MSE = (1+4+9+16)/4 = 7.5
    assert!(
        (loss.data()[0] - 7.5).abs() < 1e-4,
        "loss = {}",
        loss.data()[0]
    );
    loss.backward();
    let g = pred.grad().unwrap();
    // grad[i] = 2 * pred[i] / n
    let expected = vec![2.0 / 4.0, 4.0 / 4.0, 6.0 / 4.0, 8.0 / 4.0];
    for (i, (&actual, &exp)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "grad[{i}] = {actual}, expected {exp}"
        );
    }
}
