// Tests for GAN components: activation functions, loss functions, layer dimensions,
// and gradient computations used by the mnist_gan binary.
//
// Since Generator and Discriminator are private to the binary, these tests
// validate the core building blocks through the shared library and locally
// replicated utility functions (following the same pattern as test_matrix_ops.rs).

use approx::assert_relative_eq;
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Architecture Constants (mirrors mnist_gan.rs)
// ============================================================================

const NOISE_DIM: usize = 100;
const G_HIDDEN1: usize = 256;
const G_HIDDEN2: usize = 512;
const IMG_SIZE: usize = 784;
const D_HIDDEN1: usize = 512;
const D_HIDDEN2: usize = 256;
const LEAKY_RELU_ALPHA: f32 = 0.2;

// ============================================================================
// Utility Functions (replicated from mnist_gan.rs for testing)
// ============================================================================

/// Sigmoid activation: maps any real to (0, 1).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid in-place.
fn sigmoid_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = sigmoid(*x);
    }
}

/// Apply tanh in-place.
fn tanh_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

/// Apply Leaky ReLU in-place: f(x) = x if x ≥ 0, else alpha * x.
fn leaky_relu_inplace(v: &mut [f32], alpha: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= alpha;
        }
    }
}

/// Binary cross-entropy loss: L = −[t·log(p+ε) + (1−t)·log(1−p+ε)]
fn bce_loss(pred: f32, target: f32) -> f32 {
    -(target * (pred + 1e-7).ln() + (1.0 - target) * (1.0 - pred + 1e-7).ln())
}

/// Compute diversity metric: std-dev of per-sample mean pixel values.
/// Used for mode-collapse detection.
fn compute_diversity(samples: &[f32], n_samples: usize, img_size: usize) -> f32 {
    let means: Vec<f32> = (0..n_samples)
        .map(|i| {
            let start = i * img_size;
            let end = start + img_size;
            samples[start..end].iter().sum::<f32>() / img_size as f32
        })
        .collect();

    let mean_of_means = means.iter().sum::<f32>() / n_samples as f32;
    let variance = means
        .iter()
        .map(|&m| (m - mean_of_means) * (m - mean_of_means))
        .sum::<f32>()
        / n_samples as f32;
    variance.sqrt()
}

/// Generate n_samples × noise_dim uniform noise in [−1, 1].
fn generate_noise(rng: &mut SimpleRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Sigmoid activation tests
    // ========================================================================

    #[test]
    fn test_sigmoid_zero_returns_half() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive_approaches_one() {
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_large_negative_approaches_zero() {
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_sigmoid_output_in_open_unit_interval() {
        // f32 saturates to exactly 0.0 or 1.0 for |x| > ~15, so restrict range.
        for i in -14..=14 {
            let x = i as f32;
            let y = sigmoid(x);
            assert!(y > 0.0 && y < 1.0, "sigmoid({}) = {} not in (0,1)", x, y);
        }
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // σ(x) + σ(−x) = 1
        let x = 3.5f32;
        assert_relative_eq!(sigmoid(x) + sigmoid(-x), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_monotone_increasing() {
        let a = sigmoid(-1.0);
        let b = sigmoid(0.0);
        let c = sigmoid(1.0);
        assert!(a < b && b < c);
    }

    #[test]
    fn test_sigmoid_inplace_batch() {
        let mut v = vec![-2.0f32, 0.0, 2.0];
        sigmoid_inplace(&mut v);
        assert_relative_eq!(v[0], sigmoid(-2.0), epsilon = 1e-6);
        assert_relative_eq!(v[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(v[2], sigmoid(2.0), epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_inplace_no_nan() {
        let mut v: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        sigmoid_inplace(&mut v);
        for &x in &v {
            assert!(!x.is_nan() && !x.is_infinite());
        }
    }

    // ========================================================================
    // Tanh activation tests
    // ========================================================================

    #[test]
    fn test_tanh_inplace_zero_is_zero() {
        let mut v = vec![0.0f32];
        tanh_inplace(&mut v);
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_tanh_inplace_positive() {
        let mut v = vec![1.0f32];
        tanh_inplace(&mut v);
        assert_relative_eq!(v[0], 1.0f32.tanh(), epsilon = 1e-6);
        assert!(v[0] > 0.0 && v[0] < 1.0);
    }

    #[test]
    fn test_tanh_inplace_negative() {
        let mut v = vec![-1.0f32];
        tanh_inplace(&mut v);
        assert_relative_eq!(v[0], (-1.0f32).tanh(), epsilon = 1e-6);
        assert!(v[0] < 0.0 && v[0] > -1.0);
    }

    #[test]
    fn test_tanh_inplace_output_in_open_interval() {
        // f32 saturates to exactly ±1.0 for |x| >= ~9, so restrict range.
        let mut v: Vec<f32> = (-8..=8).map(|i| i as f32).collect();
        tanh_inplace(&mut v);
        for &x in &v {
            assert!(x > -1.0 && x < 1.0, "tanh output {} not in (-1,1)", x);
        }
    }

    #[test]
    fn test_tanh_inplace_symmetry() {
        let mut pos = vec![2.0f32];
        let mut neg = vec![-2.0f32];
        tanh_inplace(&mut pos);
        tanh_inplace(&mut neg);
        assert_relative_eq!(pos[0], -neg[0], epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_inplace_batch_no_nan() {
        let mut v: Vec<f32> = (-100..=100).map(|i| i as f32).collect();
        tanh_inplace(&mut v);
        for &x in &v {
            assert!(!x.is_nan() && !x.is_infinite());
        }
    }

    // ========================================================================
    // Leaky ReLU tests
    // ========================================================================

    #[test]
    fn test_leaky_relu_positive_unchanged() {
        let mut v = vec![1.0f32, 2.0, 5.0];
        let expected = v.clone();
        leaky_relu_inplace(&mut v, LEAKY_RELU_ALPHA);
        for (&got, &exp) in v.iter().zip(expected.iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_leaky_relu_zero_unchanged() {
        let mut v = vec![0.0f32];
        leaky_relu_inplace(&mut v, LEAKY_RELU_ALPHA);
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_leaky_relu_negative_scaled() {
        let mut v = vec![-1.0f32];
        leaky_relu_inplace(&mut v, LEAKY_RELU_ALPHA);
        assert_relative_eq!(v[0], -LEAKY_RELU_ALPHA, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_alpha_02_scaling() {
        // α=0.2 (GAN convention)
        let mut v = vec![-5.0f32];
        leaky_relu_inplace(&mut v, 0.2);
        assert_relative_eq!(v[0], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_mixed_batch() {
        let mut v = vec![-2.0f32, 0.0, 3.0, -0.5];
        leaky_relu_inplace(&mut v, LEAKY_RELU_ALPHA);
        assert_relative_eq!(v[0], -2.0 * LEAKY_RELU_ALPHA, epsilon = 1e-6);
        assert_relative_eq!(v[1], 0.0, epsilon = 1e-7);
        assert_relative_eq!(v[2], 3.0, epsilon = 1e-7);
        assert_relative_eq!(v[3], -0.5 * LEAKY_RELU_ALPHA, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_preserves_sign_direction() {
        // Negative outputs remain negative (not zero, unlike ReLU)
        let mut v = vec![-10.0f32, -1.0, -0.001];
        leaky_relu_inplace(&mut v, LEAKY_RELU_ALPHA);
        for &x in &v {
            assert!(x < 0.0, "Leaky ReLU negative should stay negative: {}", x);
        }
    }

    // ========================================================================
    // Binary Cross-Entropy loss tests
    // ========================================================================

    #[test]
    fn test_bce_loss_perfect_real_prediction() {
        // pred ≈ 1, target = 1 → loss near 0
        let loss = bce_loss(0.999, 1.0);
        assert!(
            loss < 0.01,
            "BCE should be near 0 for correct real: {}",
            loss
        );
    }

    #[test]
    fn test_bce_loss_perfect_fake_prediction() {
        // pred ≈ 0, target = 0 → loss near 0
        let loss = bce_loss(0.001, 0.0);
        assert!(
            loss < 0.01,
            "BCE should be near 0 for correct fake: {}",
            loss
        );
    }

    #[test]
    fn test_bce_loss_wrong_real_prediction() {
        // pred ≈ 0, target = 1 → high loss
        let loss = bce_loss(0.01, 1.0);
        assert!(loss > 4.0, "BCE should be high for wrong real: {}", loss);
    }

    #[test]
    fn test_bce_loss_wrong_fake_prediction() {
        // pred ≈ 1, target = 0 → high loss
        let loss = bce_loss(0.99, 0.0);
        assert!(loss > 4.0, "BCE should be high for wrong fake: {}", loss);
    }

    #[test]
    fn test_bce_loss_midpoint() {
        // pred = 0.5, target = 0.5 → moderate loss
        let loss = bce_loss(0.5, 0.5);
        assert!(!loss.is_nan() && !loss.is_infinite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_bce_loss_no_nan_across_range() {
        // ε guard should prevent NaN at boundaries
        for i in 0..=100 {
            let pred = i as f32 / 100.0;
            let l1 = bce_loss(pred, 1.0);
            let l2 = bce_loss(pred, 0.0);
            assert!(!l1.is_nan(), "NaN at pred={} target=1", pred);
            assert!(!l2.is_nan(), "NaN at pred={} target=0", pred);
        }
    }

    #[test]
    fn test_bce_loss_non_negative() {
        // The ε guard (1e-7) can produce a very small negative value near
        // boundaries in f32 (e.g. bce_loss(0, 0) ≈ -1e-6 due to ln(1 + ε)).
        // Allow a tiny tolerance.
        let tol = 1e-5f32;
        for i in 0..=100 {
            let pred = i as f32 / 100.0;
            assert!(bce_loss(pred, 1.0) >= -tol);
            assert!(bce_loss(pred, 0.0) >= -tol);
        }
    }

    #[test]
    fn test_bce_loss_label_smoothing_penalises_overconfidence() {
        // With label_smoothing=0.1, real target = 0.9.
        // The BCE minimum for target=0.9 is at pred=0.9, NOT at pred=1.0.
        // Verify that bce_loss(0.9, 0.9) < bce_loss(1.0, 0.9):
        // pushing pred all the way to 1.0 incurs a penalty via the fake term.
        let at_target = bce_loss(0.9, 0.9);
        let overconfident = bce_loss(1.0, 0.9);
        assert!(
            at_target < overconfident,
            "Loss at smoothed target ({}) should be < overconfident ({})",
            at_target,
            overconfident
        );
    }

    // ========================================================================
    // BCE gradient tests (pred − target)
    // ========================================================================

    #[test]
    fn test_bce_gradient_real_images() {
        // pred=0.8, target=1.0 → grad = −0.2 (negative: push D up)
        let pred = 0.8f32;
        let target = 1.0f32;
        let grad = pred - target;
        assert_relative_eq!(grad, -0.2, epsilon = 1e-6);
    }

    #[test]
    fn test_bce_gradient_fake_images() {
        // pred=0.3, target=0.0 → grad = 0.3 (positive: push D down)
        let pred = 0.3f32;
        let target = 0.0f32;
        let grad = pred - target;
        assert_relative_eq!(grad, 0.3, epsilon = 1e-6);
    }

    #[test]
    fn test_bce_gradient_generator_step() {
        // Generator wants pred=1.0 but gets pred=0.4 → grad = −0.6
        let pred = 0.4f32;
        let gen_target = 1.0f32;
        let grad = pred - gen_target;
        assert_relative_eq!(grad, -0.6, epsilon = 1e-6);
    }

    // ========================================================================
    // Generator layer dimension tests (using DenseLayer from shared library)
    // ========================================================================

    #[test]
    fn test_generator_layer1_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(NOISE_DIM, G_HIDDEN1, &mut rng);
        assert_eq!(layer.input_size(), NOISE_DIM);
        assert_eq!(layer.output_size(), G_HIDDEN1);
    }

    #[test]
    fn test_generator_layer2_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, &mut rng);
        assert_eq!(layer.input_size(), G_HIDDEN1);
        assert_eq!(layer.output_size(), G_HIDDEN2);
    }

    #[test]
    fn test_generator_layer3_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(G_HIDDEN2, IMG_SIZE, &mut rng);
        assert_eq!(layer.input_size(), G_HIDDEN2);
        assert_eq!(layer.output_size(), IMG_SIZE);
    }

    #[test]
    fn test_generator_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(NOISE_DIM, G_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(G_HIDDEN2, IMG_SIZE, &mut rng);

        // (in * out) + out bias per layer
        let expected_l1 = NOISE_DIM * G_HIDDEN1 + G_HIDDEN1;
        let expected_l2 = G_HIDDEN1 * G_HIDDEN2 + G_HIDDEN2;
        let expected_l3 = G_HIDDEN2 * IMG_SIZE + IMG_SIZE;

        assert_eq!(l1.parameter_count(), expected_l1);
        assert_eq!(l2.parameter_count(), expected_l2);
        assert_eq!(l3.parameter_count(), expected_l3);
    }

    // ========================================================================
    // Discriminator layer dimension tests (using DenseLayer from shared library)
    // ========================================================================

    #[test]
    fn test_discriminator_layer1_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(IMG_SIZE, D_HIDDEN1, &mut rng);
        assert_eq!(layer.input_size(), IMG_SIZE);
        assert_eq!(layer.output_size(), D_HIDDEN1);
    }

    #[test]
    fn test_discriminator_layer2_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, &mut rng);
        assert_eq!(layer.input_size(), D_HIDDEN1);
        assert_eq!(layer.output_size(), D_HIDDEN2);
    }

    #[test]
    fn test_discriminator_layer3_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(D_HIDDEN2, 1, &mut rng);
        assert_eq!(layer.input_size(), D_HIDDEN2);
        assert_eq!(layer.output_size(), 1);
    }

    #[test]
    fn test_discriminator_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(IMG_SIZE, D_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(D_HIDDEN2, 1, &mut rng);

        let expected_l1 = IMG_SIZE * D_HIDDEN1 + D_HIDDEN1;
        let expected_l2 = D_HIDDEN1 * D_HIDDEN2 + D_HIDDEN2;
        let expected_l3 = D_HIDDEN2 * 1 + 1;

        assert_eq!(l1.parameter_count(), expected_l1);
        assert_eq!(l2.parameter_count(), expected_l2);
        assert_eq!(l3.parameter_count(), expected_l3);
    }

    // ========================================================================
    // Generator forward pass shape tests
    // ========================================================================

    #[test]
    fn test_generator_forward_output_shape_batch1() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(NOISE_DIM, G_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(G_HIDDEN2, IMG_SIZE, &mut rng);

        let batch_size = 1;
        let noise = vec![0.5f32; batch_size * NOISE_DIM];

        let mut a1 = vec![0.0f32; batch_size * G_HIDDEN1];
        l1.forward(&noise, &mut a1, batch_size);
        assert_eq!(a1.len(), batch_size * G_HIDDEN1);

        let mut a2 = vec![0.0f32; batch_size * G_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);
        assert_eq!(a2.len(), batch_size * G_HIDDEN2);

        let mut output = vec![0.0f32; batch_size * IMG_SIZE];
        l3.forward(&a2, &mut output, batch_size);
        assert_eq!(output.len(), batch_size * IMG_SIZE);
    }

    #[test]
    fn test_generator_forward_output_shape_batch4() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(NOISE_DIM, G_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(G_HIDDEN2, IMG_SIZE, &mut rng);

        let batch_size = 4;
        let noise = vec![0.0f32; batch_size * NOISE_DIM];

        let mut a1 = vec![0.0f32; batch_size * G_HIDDEN1];
        l1.forward(&noise, &mut a1, batch_size);

        let mut a2 = vec![0.0f32; batch_size * G_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);

        let mut output = vec![0.0f32; batch_size * IMG_SIZE];
        l3.forward(&a2, &mut output, batch_size);

        assert_eq!(output.len(), batch_size * IMG_SIZE);
    }

    #[test]
    fn test_generator_output_tanh_range_after_activation() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(NOISE_DIM, G_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(G_HIDDEN2, IMG_SIZE, &mut rng);

        let batch_size = 2;
        let noise = generate_noise(&mut SimpleRng::new(7919), batch_size * NOISE_DIM);

        let mut a1 = vec![0.0f32; batch_size * G_HIDDEN1];
        l1.forward(&noise, &mut a1, batch_size);
        leaky_relu_inplace(&mut a1, LEAKY_RELU_ALPHA);

        let mut a2 = vec![0.0f32; batch_size * G_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);
        leaky_relu_inplace(&mut a2, LEAKY_RELU_ALPHA);

        let mut output = vec![0.0f32; batch_size * IMG_SIZE];
        l3.forward(&a2, &mut output, batch_size);
        tanh_inplace(&mut output);

        for &v in &output {
            assert!(v > -1.0 && v < 1.0, "Generator output {} not in (-1, 1)", v);
        }
    }

    // ========================================================================
    // Discriminator forward pass shape tests
    // ========================================================================

    #[test]
    fn test_discriminator_forward_output_shape_batch1() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(IMG_SIZE, D_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(D_HIDDEN2, 1, &mut rng);

        let batch_size = 1;
        let images = vec![0.0f32; batch_size * IMG_SIZE];

        let mut a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        l1.forward(&images, &mut a1, batch_size);
        assert_eq!(a1.len(), batch_size * D_HIDDEN1);

        let mut a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);
        assert_eq!(a2.len(), batch_size * D_HIDDEN2);

        let mut output = vec![0.0f32; batch_size];
        l3.forward(&a2, &mut output, batch_size);
        assert_eq!(output.len(), batch_size);
    }

    #[test]
    fn test_discriminator_output_sigmoid_range() {
        let mut rng = SimpleRng::new(42);
        let l1 = DenseLayer::new(IMG_SIZE, D_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(D_HIDDEN2, 1, &mut rng);

        let batch_size = 4;
        let images = vec![0.0f32; batch_size * IMG_SIZE];

        let mut a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        l1.forward(&images, &mut a1, batch_size);
        leaky_relu_inplace(&mut a1, LEAKY_RELU_ALPHA);

        let mut a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);
        leaky_relu_inplace(&mut a2, LEAKY_RELU_ALPHA);

        let mut output = vec![0.0f32; batch_size];
        l3.forward(&a2, &mut output, batch_size);
        sigmoid_inplace(&mut output);

        for &v in &output {
            assert!(
                v > 0.0 && v < 1.0,
                "Discriminator output {} not in (0, 1)",
                v
            );
        }
    }

    #[test]
    fn test_discriminator_forward_no_nan() {
        let mut rng = SimpleRng::new(1337);
        let l1 = DenseLayer::new(IMG_SIZE, D_HIDDEN1, &mut rng);
        let l2 = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, &mut rng);
        let l3 = DenseLayer::new(D_HIDDEN2, 1, &mut rng);

        let batch_size = 2;
        // Random images in [−1, 1]
        let mut image_rng = SimpleRng::new(999);
        let images = generate_noise(&mut image_rng, batch_size * IMG_SIZE);

        let mut a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        l1.forward(&images, &mut a1, batch_size);
        leaky_relu_inplace(&mut a1, LEAKY_RELU_ALPHA);

        let mut a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        l2.forward(&a1, &mut a2, batch_size);
        leaky_relu_inplace(&mut a2, LEAKY_RELU_ALPHA);

        let mut output = vec![0.0f32; batch_size];
        l3.forward(&a2, &mut output, batch_size);
        sigmoid_inplace(&mut output);

        for &v in &output {
            assert!(!v.is_nan(), "Discriminator output is NaN");
            assert!(!v.is_infinite(), "Discriminator output is infinite");
        }
    }

    // ========================================================================
    // Noise generation tests
    // ========================================================================

    #[test]
    fn test_noise_length() {
        let mut rng = SimpleRng::new(42);
        let batch_size = 8;
        let noise = generate_noise(&mut rng, batch_size * NOISE_DIM);
        assert_eq!(noise.len(), batch_size * NOISE_DIM);
    }

    #[test]
    fn test_noise_range_within_minus1_plus1() {
        let mut rng = SimpleRng::new(42);
        let noise = generate_noise(&mut rng, 1000);
        for &v in &noise {
            assert!(v >= -1.0 && v <= 1.0, "Noise {} outside [-1, 1]", v);
        }
    }

    #[test]
    fn test_noise_is_not_constant() {
        let mut rng = SimpleRng::new(42);
        let noise = generate_noise(&mut rng, NOISE_DIM);
        let first = noise[0];
        let all_same = noise.iter().all(|&x| x == first);
        assert!(!all_same, "All noise values are identical");
    }

    #[test]
    fn test_noise_different_seeds_differ() {
        let noise1 = generate_noise(&mut SimpleRng::new(1), NOISE_DIM);
        let noise2 = generate_noise(&mut SimpleRng::new(2), NOISE_DIM);
        assert_ne!(noise1, noise2, "Different seeds produced identical noise");
    }

    #[test]
    fn test_noise_reproducible_with_same_seed() {
        let noise1 = generate_noise(&mut SimpleRng::new(7919), NOISE_DIM);
        let noise2 = generate_noise(&mut SimpleRng::new(7919), NOISE_DIM);
        assert_eq!(noise1, noise2, "Same seed should produce identical noise");
    }

    // ========================================================================
    // Tanh derivative (gradient) tests – used in Generator backward pass
    // ========================================================================

    #[test]
    fn test_tanh_derivative_at_zero_is_one() {
        // d/dx tanh(x)|_{x=0} = 1 − tanh(0)² = 1 − 0 = 1
        let y = 0.0f32.tanh(); // = 0
        let deriv = 1.0 - y * y;
        assert_relative_eq!(deriv, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_derivative_range_is_zero_to_one() {
        // For any y = tanh(x) ∈ (−1, 1), derivative = 1 − y² ∈ (0, 1]
        for i in -99..=99 {
            let x = i as f32 * 0.1;
            let y = x.tanh();
            let deriv = 1.0 - y * y;
            assert!(
                (0.0..=1.0).contains(&deriv),
                "tanh deriv {} out of [0,1] for x={}",
                deriv,
                x
            );
        }
    }

    #[test]
    fn test_tanh_derivative_maximised_at_zero() {
        let deriv_zero = 1.0 - 0.0f32.tanh().powi(2);
        let deriv_one = 1.0 - 1.0f32.tanh().powi(2);
        assert!(deriv_zero > deriv_one);
    }

    #[test]
    fn test_tanh_backward_gradient_scaling() {
        // Simulate one element's contribution to Generator backward:
        // d_logit3 = grad_output * (1 − tanh_out²)
        let gen_output = 0.8f32; // post-tanh value
        let grad_output = -0.5f32; // incoming gradient
        let d_logit = grad_output * (1.0 - gen_output * gen_output);
        // 1 − 0.64 = 0.36; −0.5 * 0.36 = −0.18
        assert_relative_eq!(d_logit, -0.18, epsilon = 1e-6);
    }

    // ========================================================================
    // Leaky ReLU backward (gradient masking) tests
    // ========================================================================

    #[test]
    fn test_leaky_relu_backward_positive_passes_through() {
        // Positive activation → gradient unchanged
        let a = 2.0f32;
        let grad = 0.5f32;
        let d = if a <= 0.0 {
            grad * LEAKY_RELU_ALPHA
        } else {
            grad
        };
        assert_relative_eq!(d, 0.5, epsilon = 1e-7);
    }

    #[test]
    fn test_leaky_relu_backward_zero_uses_alpha() {
        // Zero activation treated as non-positive (≤ 0) → scaled by alpha
        let a = 0.0f32;
        let grad = 1.0f32;
        let d = if a <= 0.0 {
            grad * LEAKY_RELU_ALPHA
        } else {
            grad
        };
        assert_relative_eq!(d, LEAKY_RELU_ALPHA, epsilon = 1e-7);
    }

    #[test]
    fn test_leaky_relu_backward_negative_uses_alpha() {
        let a = -1.0f32;
        let grad = 1.0f32;
        let d = if a <= 0.0 {
            grad * LEAKY_RELU_ALPHA
        } else {
            grad
        };
        assert_relative_eq!(d, LEAKY_RELU_ALPHA, epsilon = 1e-7);
    }

    #[test]
    fn test_leaky_relu_backward_batch() {
        // Mirrors the backward-pass loop in Generator and Discriminator
        let a = vec![-2.0f32, 0.0, 1.0, -0.5, 3.0];
        let mut grad = vec![1.0f32; a.len()];
        for i in 0..a.len() {
            if a[i] <= 0.0 {
                grad[i] *= LEAKY_RELU_ALPHA;
            }
        }
        assert_relative_eq!(grad[0], LEAKY_RELU_ALPHA, epsilon = 1e-7); // negative → alpha
        assert_relative_eq!(grad[1], LEAKY_RELU_ALPHA, epsilon = 1e-7); // zero → alpha
        assert_relative_eq!(grad[2], 1.0, epsilon = 1e-7); // positive → 1
        assert_relative_eq!(grad[3], LEAKY_RELU_ALPHA, epsilon = 1e-7); // negative → alpha
        assert_relative_eq!(grad[4], 1.0, epsilon = 1e-7); // positive → 1
    }

    // ========================================================================
    // Diversity / mode-collapse detection tests
    // ========================================================================

    #[test]
    fn test_diversity_constant_samples_is_zero() {
        // All identical images → std-dev of means = 0
        let n_samples = 10;
        let samples = vec![0.5f32; n_samples * IMG_SIZE];
        let div = compute_diversity(&samples, n_samples, IMG_SIZE);
        assert_relative_eq!(div, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_diversity_different_samples_is_positive() {
        let mut rng = SimpleRng::new(42);
        let n_samples = 20;
        let samples = generate_noise(&mut rng, n_samples * IMG_SIZE);
        let div = compute_diversity(&samples, n_samples, IMG_SIZE);
        assert!(div > 0.0, "Diverse samples should have positive diversity");
    }

    #[test]
    fn test_diversity_mode_collapse_threshold() {
        // Mode-collapse threshold is < 0.02 in mnist_gan.rs
        let collapse_threshold = 0.02f32;

        // Simulate collapsed: almost identical images with tiny noise
        let n_samples = 10;
        let mut samples = vec![0.5f32; n_samples * IMG_SIZE];
        // Add tiny per-sample perturbation
        for i in 0..n_samples {
            samples[i * IMG_SIZE] += (i as f32) * 0.0001;
        }
        let div = compute_diversity(&samples, n_samples, IMG_SIZE);
        assert!(
            div < collapse_threshold,
            "Near-constant samples should signal collapse: div={}",
            div
        );
    }

    #[test]
    fn test_diversity_no_nan_for_random_samples() {
        let mut rng = SimpleRng::new(2024);
        let n_samples = 50;
        let samples = generate_noise(&mut rng, n_samples * IMG_SIZE);
        let div = compute_diversity(&samples, n_samples, IMG_SIZE);
        assert!(!div.is_nan(), "Diversity should not be NaN");
        assert!(!div.is_infinite(), "Diversity should not be infinite");
    }

    // ========================================================================
    // Label smoothing tests
    // ========================================================================

    #[test]
    fn test_label_smoothing_real_target() {
        let label_smoothing = 0.1f32;
        let real_target = 1.0 - label_smoothing;
        assert_relative_eq!(real_target, 0.9, epsilon = 1e-6);
    }

    #[test]
    fn test_label_smoothing_fake_target_unchanged() {
        // Fake target for discriminator remains 0 (no smoothing on fake side)
        let fake_target_d = 0.0f32;
        assert_relative_eq!(fake_target_d, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_label_smoothing_minimum_at_smoothed_target() {
        // With smoothed target = 0.9, the BCE minimum is at pred = 0.9 (not 1.0).
        // pred=0.9 (matches target) should have lower loss than pred=0.5 or pred=0.0.
        let loss_at_target = bce_loss(0.9, 0.9);
        let loss_at_mid = bce_loss(0.5, 0.9);
        let loss_at_zero = bce_loss(0.0, 0.9);
        assert!(
            loss_at_target < loss_at_mid,
            "Loss at target ({}) should be < midpoint ({})",
            loss_at_target,
            loss_at_mid
        );
        assert!(
            loss_at_mid < loss_at_zero,
            "Midpoint loss ({}) should be < zero pred ({})",
            loss_at_mid,
            loss_at_zero
        );
    }

    // ========================================================================
    // Image rescaling tests [0,1] → [−1,1]
    // ========================================================================

    #[test]
    fn test_image_rescale_zero_maps_to_minus_one() {
        let pixel = 0.0f32;
        let rescaled = pixel * 2.0 - 1.0;
        assert_relative_eq!(rescaled, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_image_rescale_one_maps_to_plus_one() {
        let pixel = 1.0f32;
        let rescaled = pixel * 2.0 - 1.0;
        assert_relative_eq!(rescaled, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_image_rescale_half_maps_to_zero() {
        let pixel = 0.5f32;
        let rescaled = pixel * 2.0 - 1.0;
        assert_relative_eq!(rescaled, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_image_rescale_full_batch() {
        let original = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let rescaled: Vec<f32> = original.iter().map(|&p| p * 2.0 - 1.0).collect();
        assert_relative_eq!(rescaled[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(rescaled[1], -0.5, epsilon = 1e-6);
        assert_relative_eq!(rescaled[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(rescaled[3], 0.5, epsilon = 1e-6);
        assert_relative_eq!(rescaled[4], 1.0, epsilon = 1e-6);
    }

    // ========================================================================
    // Architecture symmetry / correctness tests
    // ========================================================================

    #[test]
    fn test_noise_dim_constant() {
        assert_eq!(NOISE_DIM, 100);
    }

    #[test]
    fn test_img_size_constant() {
        // 28 × 28 MNIST images
        assert_eq!(IMG_SIZE, 784);
    }

    #[test]
    fn test_leaky_relu_alpha_constant() {
        // GAN convention uses α = 0.2
        assert_relative_eq!(LEAKY_RELU_ALPHA, 0.2, epsilon = 1e-7);
    }

    #[test]
    fn test_generator_hidden_sizes_increase() {
        // G_HIDDEN1 < G_HIDDEN2: upsampling path
        assert!(
            G_HIDDEN1 < G_HIDDEN2,
            "Generator hidden sizes should increase: {} < {}",
            G_HIDDEN1,
            G_HIDDEN2
        );
    }

    #[test]
    fn test_discriminator_hidden_sizes_decrease() {
        // D_HIDDEN1 > D_HIDDEN2: downsampling path
        assert!(
            D_HIDDEN1 > D_HIDDEN2,
            "Discriminator hidden sizes should decrease: {} > {}",
            D_HIDDEN1,
            D_HIDDEN2
        );
    }
}
