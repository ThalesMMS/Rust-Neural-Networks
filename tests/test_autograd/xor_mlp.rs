use super::*;

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
