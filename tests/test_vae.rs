// Integration tests for the VariationalAutoencoder.
// Tests construction, forward/backward pass, loss computation, and training.
// Mirrors the patterns established in test_autoencoder.rs.

use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Construction and Initialization Tests
// ============================================================================

#[test]
fn test_vae_construction_basic() {
    // Test basic VAE construction
    let mut rng = SimpleRng::new(42);
    let vae = VariationalAutoencoder::new(784, &[256], 32, &[256], &mut rng);

    assert_eq!(vae.input_size(), 784);
    assert_eq!(vae.latent_dim(), 32);
}

#[test]
fn test_vae_construction_no_hidden_layers() {
    // Minimal VAE: 10 -> (mu:2, log_var:2) -> z:2 -> 10
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(10, &[], 2, &[], &mut rng);

    assert_eq!(vae.input_size(), 10);
    assert_eq!(vae.latent_dim(), 2);
}

#[test]
fn test_vae_construction_deep_architecture() {
    // Test deeper architecture with multiple hidden layers
    let mut rng = SimpleRng::new(7);
    // Architecture: 32 -> 16 -> 8 -> (mu:4, log_var:4) -> z:4 -> 8 -> 16 -> 32
    let vae = VariationalAutoencoder::new(32, &[16, 8], 4, &[8, 16], &mut rng);

    assert_eq!(vae.input_size(), 32);
    assert_eq!(vae.latent_dim(), 4);
}

#[test]
fn test_vae_construction_asymmetric_encoder_decoder() {
    // Test asymmetric architecture (different encoder/decoder hidden sizes)
    let mut rng = SimpleRng::new(5);
    let vae = VariationalAutoencoder::new(20, &[10, 6], 3, &[5], &mut rng);

    assert_eq!(vae.input_size(), 20);
    assert_eq!(vae.latent_dim(), 3);
}

#[test]
fn test_vae_construction_layer_sizes() {
    // Test that encoder_layer_sizes and decoder_layer_sizes are correct
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);

    let enc_sizes = vae.encoder_layer_sizes();
    // [8, 4, 2]
    assert_eq!(enc_sizes[0], 8);
    assert_eq!(enc_sizes[1], 4);
    assert_eq!(enc_sizes[2], 2);

    let dec_sizes = vae.decoder_layer_sizes();
    // [2, 4, 8]
    assert_eq!(dec_sizes[0], 2);
    assert_eq!(dec_sizes[1], 4);
    assert_eq!(dec_sizes[2], 8);
}

#[test]
fn test_vae_construction_layer_sizes_no_hidden() {
    // Test layer size vectors with no hidden layers
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(6, &[], 2, &[], &mut rng);

    let enc_sizes = vae.encoder_layer_sizes();
    // [6, 2]
    assert_eq!(enc_sizes.len(), 2);
    assert_eq!(enc_sizes[0], 6);
    assert_eq!(enc_sizes[1], 2);

    let dec_sizes = vae.decoder_layer_sizes();
    // [2, 6]
    assert_eq!(dec_sizes.len(), 2);
    assert_eq!(dec_sizes[0], 2);
    assert_eq!(dec_sizes[1], 6);
}

// ============================================================================
// Parameter Count Tests
// ============================================================================

#[test]
fn test_vae_parameter_count_no_hidden() {
    // Verify parameter count with no hidden layers
    // mu_layer: input*latent + latent
    // log_var_layer: input*latent + latent
    // decoder layer: latent*input + input
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    // mu: 4*2+2=10, log_var: 4*2+2=10, decoder: 2*4+4=12; Total=32
    let expected = (4 * 2 + 2) + (4 * 2 + 2) + (2 * 4 + 4);
    assert_eq!(vae.parameter_count(), expected);
}

#[test]
fn test_vae_parameter_count_with_hidden() {
    // Verify parameter count with hidden layers
    let mut rng = SimpleRng::new(0);
    // 4 -> 3 (trunk), (mu: 3->2, log_var: 3->2), 2->3->4
    let vae = VariationalAutoencoder::new(4, &[3], 2, &[3], &mut rng);
    // Encoder trunk: 4*3+3=15
    // mu_layer: 3*2+2=8
    // log_var_layer: 3*2+2=8
    // Decoder[0]: 2*3+3=9
    // Decoder[1]: 3*4+4=16
    let expected = (4 * 3 + 3) + (3 * 2 + 2) + (3 * 2 + 2) + (2 * 3 + 3) + (3 * 4 + 4);
    assert_eq!(vae.parameter_count(), expected);
}

#[test]
fn test_vae_parameter_count_scales_with_size() {
    // Verify that larger architecture has more parameters
    let mut rng = SimpleRng::new(1);
    let vae_small = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let vae_large = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    assert!(vae_large.parameter_count() > vae_small.parameter_count());
}

#[test]
fn test_vae_parameter_count_mu_and_log_var_extra() {
    // VAE should have more params than a vanilla AE of same structure (due to extra head)
    // VAE: enc_trunk + mu_layer + log_var_layer + decoder
    // Vanilla AE: enc_layers + dec_layers (no extra head)
    // For VAE with no hidden: mu + log_var + decoder = 3 sets vs vanilla's 2
    let mut rng = SimpleRng::new(42);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    // mu: 10, log_var: 10, decoder: 12 = 32 params
    // A vanilla 4->2->4 AE would have: enc(4*2+2=10) + dec(2*4+4=12) = 22 params
    // VAE must have more
    assert!(vae.parameter_count() > 22);
}

// ============================================================================
// Forward Pass Shape Tests
// ============================================================================

#[test]
fn test_vae_forward_output_shapes_batch1() {
    // Test output shapes for batch size 1
    let mut rng = SimpleRng::new(1);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(99);
    let input = vec![0.5f32; 8];
    let (reconstruction, mu, log_var) = vae.forward(&input, 1, &mut rng2);
    assert_eq!(reconstruction.len(), 8);
    assert_eq!(mu.len(), 2);
    assert_eq!(log_var.len(), 2);
}

#[test]
fn test_vae_forward_output_shapes_batch4() {
    // Test output shapes for batch size 4
    let mut rng = SimpleRng::new(1);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(99);
    let input = vec![0.5f32; 8 * 4];
    let (reconstruction, mu, log_var) = vae.forward(&input, 4, &mut rng2);
    assert_eq!(reconstruction.len(), 8 * 4);
    assert_eq!(mu.len(), 2 * 4);
    assert_eq!(log_var.len(), 2 * 4);
}

#[test]
fn test_vae_forward_output_shapes_batch_large() {
    // Test output shapes for a larger batch
    let mut rng = SimpleRng::new(2);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let mut rng2 = SimpleRng::new(7);
    let batch_size = 32;
    let input = vec![0.3f32; 16 * batch_size];
    let (reconstruction, mu, log_var) = vae.forward(&input, batch_size, &mut rng2);
    assert_eq!(reconstruction.len(), 16 * batch_size);
    assert_eq!(mu.len(), 4 * batch_size);
    assert_eq!(log_var.len(), 4 * batch_size);
}

#[test]
fn test_vae_forward_output_shapes_no_hidden() {
    // Test output shapes with no hidden layers
    let mut rng = SimpleRng::new(3);
    let mut vae = VariationalAutoencoder::new(6, &[], 2, &[], &mut rng);
    let mut rng2 = SimpleRng::new(11);
    let input = vec![0.4f32; 6 * 3];
    let (reconstruction, mu, log_var) = vae.forward(&input, 3, &mut rng2);
    assert_eq!(reconstruction.len(), 6 * 3);
    assert_eq!(mu.len(), 2 * 3);
    assert_eq!(log_var.len(), 2 * 3);
}

// ============================================================================
// Output Range Tests (Sigmoid Activation on Reconstruction)
// ============================================================================

#[test]
fn test_vae_reconstruction_in_sigmoid_range_batch1() {
    // All reconstruction values should be strictly in (0, 1) due to Sigmoid
    let mut rng = SimpleRng::new(2);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(7);
    let input = vec![0.3f32; 8];
    let (reconstruction, _, _) = vae.forward(&input, 1, &mut rng2);
    for &v in &reconstruction {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vae_reconstruction_in_sigmoid_range_batch4() {
    // Verify sigmoid range with multiple samples in batch
    let mut rng = SimpleRng::new(3);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(13);
    let input = vec![0.7f32; 8 * 4];
    let (reconstruction, _, _) = vae.forward(&input, 4, &mut rng2);
    for &v in &reconstruction {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vae_reconstruction_in_sigmoid_range_zero_input() {
    // Test with zero input
    let mut rng = SimpleRng::new(4);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(21);
    let input = vec![0.0f32; 8];
    let (reconstruction, _, _) = vae.forward(&input, 1, &mut rng2);
    for &v in &reconstruction {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vae_reconstruction_in_sigmoid_range_ones_input() {
    // Test with all-ones input
    let mut rng = SimpleRng::new(5);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(33);
    let input = vec![1.0f32; 8];
    let (reconstruction, _, _) = vae.forward(&input, 1, &mut rng2);
    for &v in &reconstruction {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

// ============================================================================
// Encode Tests
// ============================================================================

#[test]
fn test_vae_encode_output_shapes_batch1() {
    // Test that encode returns (mu, log_var) each of shape batch_size × latent_dim
    let mut rng = SimpleRng::new(10);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16];
    let (mu, log_var) = vae.encode(&input, 1);
    assert_eq!(mu.len(), 4);
    assert_eq!(log_var.len(), 4);
}

#[test]
fn test_vae_encode_output_shapes_batch2() {
    // Test that encode returns correct shapes for batch of 2
    let mut rng = SimpleRng::new(10);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16 * 2];
    let (mu, log_var) = vae.encode(&input, 2);
    assert_eq!(mu.len(), 4 * 2);
    assert_eq!(log_var.len(), 4 * 2);
}

#[test]
fn test_vae_encode_mu_is_finite() {
    // Encoded mu should always be finite
    let mut rng = SimpleRng::new(11);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let (mu, _) = vae.encode(&input, 1);
    for &v in &mu {
        assert!(v.is_finite(), "mu value {} should be finite", v);
    }
}

#[test]
fn test_vae_encode_log_var_is_finite() {
    // Encoded log_var should always be finite
    let mut rng = SimpleRng::new(12);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let (_, log_var) = vae.encode(&input, 1);
    for &v in &log_var {
        assert!(v.is_finite(), "log_var value {} should be finite", v);
    }
}

// ============================================================================
// Decode Tests
// ============================================================================

#[test]
fn test_vae_decode_output_shape_batch1() {
    // Test that decode returns batch_size × input_size
    let mut rng = SimpleRng::new(11);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let z = vec![0.1f32; 4];
    let output = vae.decode(&z, 1);
    assert_eq!(output.len(), 16);
}

#[test]
fn test_vae_decode_output_shape_batch2() {
    // Test that decode returns batch_size × input_size for batch of 2
    let mut rng = SimpleRng::new(11);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let z = vec![0.1f32; 4 * 2];
    let output = vae.decode(&z, 2);
    assert_eq!(output.len(), 16 * 2);
}

#[test]
fn test_vae_decode_output_in_sigmoid_range() {
    // Decode output should always be in (0, 1)
    let mut rng = SimpleRng::new(12);
    let mut vae = VariationalAutoencoder::new(10, &[6], 3, &[6], &mut rng);
    let z = vec![0.5f32; 3 * 4];
    let output = vae.decode(&z, 4);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Decode output {} not in (0,1)", v);
    }
}

// ============================================================================
// Reparameterization Tests
// ============================================================================

#[test]
fn test_vae_reparameterize_output_shape() {
    // reparameterize should return z of shape batch_size × latent_dim
    let mut rng = SimpleRng::new(5);
    let mut vae = VariationalAutoencoder::new(8, &[], 4, &[], &mut rng);
    let mut rng2 = SimpleRng::new(99);
    let mu = vec![0.0f32; 4 * 2];
    let log_var = vec![0.0f32; 4 * 2];
    let z = vae.reparameterize(&mu, &log_var, &mut rng2);
    assert_eq!(z.len(), 4 * 2);
}

#[test]
fn test_vae_reparameterize_z_is_finite() {
    // Sampled z should be finite
    let mut rng = SimpleRng::new(6);
    let mut vae = VariationalAutoencoder::new(8, &[], 4, &[], &mut rng);
    let mut rng2 = SimpleRng::new(55);
    let mu = vec![0.5f32, -0.3, 0.1, 0.0, 0.2, -0.1, 0.4, 0.3];
    let log_var = vec![-0.5f32, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, -0.3];
    let z = vae.reparameterize(&mu, &log_var, &mut rng2);
    for &v in &z {
        assert!(v.is_finite(), "Reparameterized z={} should be finite", v);
    }
}

#[test]
fn test_vae_reparameterize_at_prior_close_to_mu() {
    // With log_var=0 (std=1), z should be close to mu on average over many samples
    // For a single sample this just tests that the computation runs without panics
    let mut rng = SimpleRng::new(7);
    let mut vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let mut rng2 = SimpleRng::new(77);
    // mu=0.0, log_var=0.0 means var=1, z = 0 + eps*1 = eps ~ N(0,1)
    let mu = vec![0.0f32; 2];
    let log_var = vec![0.0f32; 2];
    let z = vae.reparameterize(&mu, &log_var, &mut rng2);
    assert_eq!(z.len(), 2);
    for &v in &z {
        assert!(v.is_finite(), "z={} should be finite", v);
    }
}

#[test]
fn test_vae_reparameterize_different_rngs_differ() {
    // Different RNG seeds should produce different z samples
    let mut rng = SimpleRng::new(8);
    let mut vae1 = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let mut vae2 = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);

    let mu = vec![0.0f32; 2];
    let log_var = vec![0.0f32; 2];

    let mut rng_a = SimpleRng::new(1);
    let mut rng_b = SimpleRng::new(999);
    let z1 = vae1.reparameterize(&mu, &log_var, &mut rng_a);
    let z2 = vae2.reparameterize(&mu, &log_var, &mut rng_b);

    let mut differs = false;
    for (a, b) in z1.iter().zip(z2.iter()) {
        if (a - b).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "Different RNG seeds should produce different z samples"
    );
}

// ============================================================================
// Forward Mean Tests (Deterministic Inference)
// ============================================================================

#[test]
fn test_vae_forward_mean_output_shape_batch1() {
    // forward_mean should return batch_size × input_size
    let mut rng = SimpleRng::new(11);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16];
    let output = vae.forward_mean(&input, 1);
    assert_eq!(output.len(), 16);
}

#[test]
fn test_vae_forward_mean_output_shape_batch2() {
    // forward_mean with batch of 2
    let mut rng = SimpleRng::new(11);
    let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16 * 2];
    let output = vae.forward_mean(&input, 2);
    assert_eq!(output.len(), 16 * 2);
}

#[test]
fn test_vae_forward_mean_in_sigmoid_range() {
    // forward_mean output should be in (0, 1)
    let mut rng = SimpleRng::new(12);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8 * 3];
    let output = vae.forward_mean(&input, 3);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "forward_mean output {} not in (0,1)", v);
    }
}

#[test]
fn test_vae_forward_mean_is_deterministic() {
    // forward_mean should produce the same output on repeated calls (no randomness)
    let mut rng = SimpleRng::new(13);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.3f32; 8];

    let output1 = vae.forward_mean(&input, 1);
    let output2 = vae.forward_mean(&input, 1);

    for (a, b) in output1.iter().zip(output2.iter()) {
        assert_eq!(
            a, b,
            "forward_mean should produce identical outputs without randomness"
        );
    }
}

// ============================================================================
// Reconstruction Loss Computation Tests
// ============================================================================

#[test]
fn test_vae_reconstruction_loss_zero_on_perfect_reconstruction() {
    // MSE should be 0 when reconstruction == target
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.5f32; 4];
    let target = vec![0.5f32; 4];
    let loss = vae.compute_reconstruction_loss(&reconstruction, &target);
    assert!(
        loss < 1e-6,
        "Loss should be ~0 for perfect reconstruction, got {}",
        loss
    );
}

#[test]
fn test_vae_reconstruction_loss_nonzero_on_imperfect() {
    // MSE should be nonzero when reconstruction != target
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.0f32; 4];
    let target = vec![1.0f32; 4];
    let loss = vae.compute_reconstruction_loss(&reconstruction, &target);
    // MSE = mean((0-1)^2) = 1.0
    assert!((loss - 1.0).abs() < 1e-5, "Expected MSE=1.0, got {}", loss);
}

#[test]
fn test_vae_reconstruction_loss_non_negative() {
    // MSE is always non-negative
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(8, &[4], 2, &[], &mut rng);
    let reconstruction = vec![0.2f32, 0.8, 0.1, 0.9, 0.5, 0.3, 0.7, 0.4];
    let target = vec![0.9f32, 0.2, 0.8, 0.1, 0.6, 0.4, 0.3, 0.7];
    let loss = vae.compute_reconstruction_loss(&reconstruction, &target);
    assert!(loss >= 0.0, "MSE should be non-negative, got {}", loss);
}

#[test]
fn test_vae_reconstruction_loss_symmetric() {
    // MSE(a, b) == MSE(b, a)
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let a = vec![0.2f32, 0.8, 0.3, 0.7];
    let b = vec![0.5f32, 0.4, 0.6, 0.1];
    let loss_ab = vae.compute_reconstruction_loss(&a, &b);
    let loss_ba = vae.compute_reconstruction_loss(&b, &a);
    assert!(
        (loss_ab - loss_ba).abs() < 1e-6,
        "MSE should be symmetric: {} vs {}",
        loss_ab,
        loss_ba
    );
}

#[test]
fn test_vae_reconstruction_loss_computed_from_forward() {
    // Test loss when computed on actual forward output
    let mut rng = SimpleRng::new(13);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(55);
    let input = vec![0.5f32; 8];
    let (reconstruction, _, _) = vae.forward(&input, 1, &mut rng2);
    let loss = vae.compute_reconstruction_loss(&reconstruction, &input);
    // Loss should be finite and non-negative
    assert!(loss.is_finite(), "Loss should be finite");
    assert!(loss >= 0.0, "Loss should be non-negative");
}

// ============================================================================
// KL Divergence Tests
// ============================================================================

#[test]
fn test_vae_kl_divergence_non_negative() {
    // KL divergence must be non-negative
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let mu = vec![0.5f32, -0.3, 0.1, 0.0];
    let log_var = vec![-0.5f32, 0.2, -0.1, 0.0];
    let kl = vae.compute_kl_divergence(&mu, &log_var);
    assert!(kl >= 0.0, "KL divergence must be non-negative, got {}", kl);
}

#[test]
fn test_vae_kl_divergence_zero_at_prior() {
    // KL(N(0,1) || N(0,1)) = 0
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let mu = vec![0.0f32; 4];
    let log_var = vec![0.0f32; 4]; // log_var=0 means var=1, std=1
    let kl = vae.compute_kl_divergence(&mu, &log_var);
    assert!(
        kl.abs() < 1e-5,
        "KL at prior N(0,1) should be ~0, got {}",
        kl
    );
}

#[test]
fn test_vae_kl_divergence_increases_with_deviation_from_prior() {
    // KL should increase as distribution deviates further from N(0,1)
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(2, &[], 2, &[], &mut rng);

    // Small deviation
    let mu_small = vec![0.1f32, 0.1];
    let log_var_small = vec![0.1f32, 0.1];
    let kl_small = vae.compute_kl_divergence(&mu_small, &log_var_small);

    // Larger deviation
    let mu_large = vec![2.0f32, 2.0];
    let log_var_large = vec![1.0f32, 1.0];
    let kl_large = vae.compute_kl_divergence(&mu_large, &log_var_large);

    assert!(
        kl_large > kl_small,
        "KL should be larger for greater deviation from prior: {} vs {}",
        kl_large,
        kl_small
    );
}

#[test]
fn test_vae_kl_divergence_with_forward_outputs() {
    // KL computed from actual forward outputs should be finite and non-negative
    let mut rng = SimpleRng::new(14);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(66);
    let input = vec![0.5f32; 8];
    let (_, mu, log_var) = vae.forward(&input, 1, &mut rng2);
    let kl = vae.compute_kl_divergence(&mu, &log_var);
    assert!(kl.is_finite(), "KL should be finite");
    assert!(kl >= 0.0, "KL should be non-negative");
}

// ============================================================================
// ELBO Loss Computation Tests
// ============================================================================

#[test]
fn test_vae_elbo_loss_at_ideal_point() {
    // ELBO loss should be ~0 when reconstruction is perfect and distribution is prior
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.5f32; 4];
    let target = vec![0.5f32; 4];
    let mu = vec![0.0f32; 2];
    let log_var = vec![0.0f32; 2];
    let elbo = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, 1.0);
    // recon_loss=0, kl=0 => ELBO=0
    assert!(
        elbo.abs() < 1e-5,
        "ELBO at ideal point should be ~0, got {}",
        elbo
    );
}

#[test]
fn test_vae_elbo_loss_non_negative() {
    // ELBO loss should generally be non-negative in practice
    let mut rng = SimpleRng::new(1);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(77);
    let input = vec![0.5f32; 8];
    let (reconstruction, mu, log_var) = vae.forward(&input, 1, &mut rng2);
    let elbo = vae.compute_elbo_loss(&reconstruction, &input, &mu, &log_var, 1.0);
    assert!(
        elbo >= 0.0,
        "ELBO loss should be non-negative in practice, got {}",
        elbo
    );
}

#[test]
fn test_vae_elbo_loss_kl_weight_zero() {
    // With kl_weight=0, ELBO should equal reconstruction loss
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.2f32, 0.8, 0.4, 0.6];
    let target = vec![0.5f32, 0.5, 0.5, 0.5];
    let mu = vec![1.0f32, -1.0];
    let log_var = vec![0.5f32, -0.5];

    let recon_loss = vae.compute_reconstruction_loss(&reconstruction, &target);
    let elbo = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, 0.0);

    assert!(
        (elbo - recon_loss).abs() < 1e-6,
        "ELBO with kl_weight=0 should equal recon_loss: {} vs {}",
        elbo,
        recon_loss
    );
}

#[test]
fn test_vae_elbo_loss_kl_weight_scales_kl_contribution() {
    // Higher kl_weight should produce higher ELBO when KL > 0
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.5f32; 4];
    let target = vec![0.5f32; 4]; // Perfect reconstruction
    let mu = vec![1.0f32, -1.0]; // Deviation from prior → KL > 0
    let log_var = vec![0.5f32, 0.5];

    let elbo_low = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, 0.5);
    let elbo_high = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, 2.0);

    assert!(
        elbo_high > elbo_low,
        "Higher kl_weight should give larger ELBO when KL>0: high={}, low={}",
        elbo_high,
        elbo_low
    );
}

#[test]
fn test_vae_elbo_loss_is_sum_of_components() {
    // ELBO = recon_loss + kl_weight * KL
    let mut rng = SimpleRng::new(0);
    let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let reconstruction = vec![0.3f32, 0.7, 0.2, 0.8];
    let target = vec![0.5f32, 0.5, 0.5, 0.5];
    let mu = vec![0.5f32, -0.5];
    let log_var = vec![0.2f32, -0.2];
    let kl_weight = 0.5f32;

    let recon_loss = vae.compute_reconstruction_loss(&reconstruction, &target);
    let kl = vae.compute_kl_divergence(&mu, &log_var);
    let expected = recon_loss + kl_weight * kl;
    let elbo = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, kl_weight);

    assert!(
        (elbo - expected).abs() < 1e-5,
        "ELBO={} should equal recon_loss+kl_weight*KL={}",
        elbo,
        expected
    );
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

#[test]
fn test_vae_backward_does_not_panic_batch1() {
    // Backward pass should not panic with batch_size=1
    let mut rng = SimpleRng::new(42);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(7);
    let input = vec![0.5f32; 8];
    vae.forward(&input, 1, &mut rng2);
    vae.backward(&input, 1, 1.0);
}

#[test]
fn test_vae_backward_does_not_panic_batch4() {
    // Backward pass should not panic with batch_size=4
    let mut rng = SimpleRng::new(43);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(17);
    let input = vec![0.5f32; 8 * 4];
    vae.forward(&input, 4, &mut rng2);
    vae.backward(&input, 4, 1.0);
}

#[test]
fn test_vae_backward_does_not_panic_no_hidden() {
    // Backward should work with minimal architecture
    let mut rng = SimpleRng::new(44);
    let mut vae = VariationalAutoencoder::new(6, &[], 2, &[], &mut rng);
    let mut rng2 = SimpleRng::new(27);
    let input = vec![0.3f32; 6];
    vae.forward(&input, 1, &mut rng2);
    vae.backward(&input, 1, 1.0);
}

#[test]
fn test_vae_backward_does_not_panic_deep() {
    // Backward should work with deeper architecture
    let mut rng = SimpleRng::new(45);
    let mut vae = VariationalAutoencoder::new(16, &[8, 4], 2, &[4, 8], &mut rng);
    let mut rng2 = SimpleRng::new(37);
    let input = vec![0.5f32; 16 * 3];
    vae.forward(&input, 3, &mut rng2);
    vae.backward(&input, 3, 1.0);
}

#[test]
fn test_vae_backward_does_not_panic_kl_weight_zero() {
    // Backward should work with kl_weight=0 (no KL penalty)
    let mut rng = SimpleRng::new(46);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(47);
    let input = vec![0.5f32; 8];
    vae.forward(&input, 1, &mut rng2);
    vae.backward(&input, 1, 0.0);
}

#[test]
fn test_vae_backward_does_not_panic_kl_weight_large() {
    // Backward should work with large kl_weight
    let mut rng = SimpleRng::new(48);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(49);
    let input = vec![0.5f32; 8];
    vae.forward(&input, 1, &mut rng2);
    vae.backward(&input, 1, 10.0);
}

// ============================================================================
// Parameter Update Tests
// ============================================================================

#[test]
fn test_vae_update_parameters_changes_behavior() {
    // Verify that update_parameters changes the model behavior
    let mut rng = SimpleRng::new(50);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng2 = SimpleRng::new(51);
    let mut rng3 = SimpleRng::new(51); // Same seed for fair comparison
    let input = vec![0.5f32; 8];

    // Forward pass 1 (before update)
    let (output_before, _, _) = vae.forward(&input, 1, &mut rng2);

    // Backward and update
    vae.backward(&input, 1, 1.0);
    vae.update_parameters(0.1);

    // Forward pass 2 - output should differ because weights changed
    let (output_after, _, _) = vae.forward(&input, 1, &mut rng3);

    let mut changed = false;
    for (a, b) in output_before.iter().zip(output_after.iter()) {
        if (a - b).abs() > 1e-6 {
            changed = true;
            break;
        }
    }
    assert!(changed, "Parameters should change after update");
}

#[test]
fn test_vae_update_parameters_high_lr_larger_change() {
    // Higher learning rate should produce larger parameter changes
    let input = vec![0.5f32; 8];

    // Small learning rate
    let mut rng_s = SimpleRng::new(51);
    let mut vae_small = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng_s);
    let mut rng_s2 = SimpleRng::new(99);
    let mut rng_s3 = SimpleRng::new(99);
    let (output_before_small, _, _) = vae_small.forward(&input, 1, &mut rng_s2);
    vae_small.backward(&input, 1, 1.0);
    vae_small.update_parameters(0.001);
    let (output_after_small, _, _) = vae_small.forward(&input, 1, &mut rng_s3);

    // Large learning rate (same seed)
    let mut rng_l = SimpleRng::new(51);
    let mut vae_large = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng_l);
    let mut rng_l2 = SimpleRng::new(99);
    let mut rng_l3 = SimpleRng::new(99);
    let (output_before_large, _, _) = vae_large.forward(&input, 1, &mut rng_l2);
    vae_large.backward(&input, 1, 1.0);
    vae_large.update_parameters(0.5);
    let (output_after_large, _, _) = vae_large.forward(&input, 1, &mut rng_l3);

    let diff_small: f32 = output_before_small
        .iter()
        .zip(output_after_small.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_large: f32 = output_before_large
        .iter()
        .zip(output_after_large.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff_large > diff_small,
        "Higher LR should cause larger changes: large_diff={}, small_diff={}",
        diff_large,
        diff_small
    );
}

// ============================================================================
// Training Convergence Tests
// ============================================================================

#[test]
fn test_vae_elbo_loss_decreases_with_training() {
    // ELBO loss should decrease over multiple training steps
    let mut rng = SimpleRng::new(100);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![
        0.9f32, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, // sample 1
        0.1f32, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, // sample 2
    ];
    let batch_size = 2;
    let lr = 0.01;
    let kl_weight = 1.0;

    let mut rng_init = SimpleRng::new(200);
    let (recon_init, mu_init, log_var_init) = vae.forward(&input, batch_size, &mut rng_init);
    let initial_loss =
        vae.compute_elbo_loss(&recon_init, &input, &mu_init, &log_var_init, kl_weight);

    // Train for several steps
    let mut rng_train = SimpleRng::new(201);
    for _ in 0..50 {
        let (recon, mu, log_var) = vae.forward(&input, batch_size, &mut rng_train);
        let _ = vae.compute_elbo_loss(&recon, &input, &mu, &log_var, kl_weight);
        vae.backward(&input, batch_size, kl_weight);
        vae.update_parameters(lr);
    }

    let mut rng_final = SimpleRng::new(202);
    let (recon_final, mu_final, log_var_final) = vae.forward(&input, batch_size, &mut rng_final);
    let final_loss =
        vae.compute_elbo_loss(&recon_final, &input, &mu_final, &log_var_final, kl_weight);

    assert!(
        final_loss < initial_loss,
        "ELBO loss should decrease during training: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_vae_reconstruction_loss_decreases_with_training() {
    // Reconstruction loss should decrease with training when evaluated deterministically
    // (using forward_mean avoids sampling noise that would obscure training progress)
    let mut rng = SimpleRng::new(110);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.8f32, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.3];
    let lr = 0.01;

    // Evaluate using forward_mean (deterministic, no sampling noise)
    let recon_init = vae.forward_mean(&input, 1);
    let initial_recon_loss = vae.compute_reconstruction_loss(&recon_init, &input);

    let mut rng_train = SimpleRng::new(211);
    for _ in 0..50 {
        let (_, _, _) = vae.forward(&input, 1, &mut rng_train);
        vae.backward(&input, 1, 1.0);
        vae.update_parameters(lr);
    }

    // Deterministic evaluation after training
    let recon_final = vae.forward_mean(&input, 1);
    let final_recon_loss = vae.compute_reconstruction_loss(&recon_final, &input);

    assert!(
        final_recon_loss < initial_recon_loss,
        "Reconstruction loss should decrease: initial={}, final={}",
        initial_recon_loss,
        final_recon_loss
    );
}

#[test]
fn test_vae_elbo_loss_finite_during_training() {
    // ELBO loss should remain finite throughout training
    let mut rng = SimpleRng::new(300);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let mut rng_train = SimpleRng::new(301);

    for i in 0..20 {
        let (recon, mu, log_var) = vae.forward(&input, 1, &mut rng_train);
        let loss = vae.compute_elbo_loss(&recon, &input, &mu, &log_var, 1.0);
        assert!(
            loss.is_finite(),
            "ELBO loss should be finite at step {}, got {}",
            i,
            loss
        );
        vae.backward(&input, 1, 1.0);
        vae.update_parameters(0.001);
    }
}

#[test]
fn test_vae_training_no_hidden_converges() {
    // Minimal VAE without hidden layers should also be trainable
    let mut rng = SimpleRng::new(200);
    let mut vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
    let input = vec![0.8f32, 0.2, 0.6, 0.4];
    let lr = 0.01;

    let mut rng_init = SimpleRng::new(300);
    let (recon_init, mu_init, log_var_init) = vae.forward(&input, 1, &mut rng_init);
    let initial_loss = vae.compute_elbo_loss(&recon_init, &input, &mu_init, &log_var_init, 1.0);

    let mut rng_train = SimpleRng::new(301);
    for _ in 0..100 {
        let (_, _, _) = vae.forward(&input, 1, &mut rng_train);
        vae.backward(&input, 1, 1.0);
        vae.update_parameters(lr);
    }

    let mut rng_final = SimpleRng::new(302);
    let (recon_final, mu_final, log_var_final) = vae.forward(&input, 1, &mut rng_final);
    let final_loss = vae.compute_elbo_loss(&recon_final, &input, &mu_final, &log_var_final, 1.0);

    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

// ============================================================================
// Reproducibility Tests
// ============================================================================

#[test]
fn test_vae_deterministic_with_same_seed() {
    // Same model seed AND same rng2 seed should produce identical outputs
    let input = vec![0.3f32, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4];

    let mut rng1 = SimpleRng::new(99);
    let mut vae1 = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng1);
    let mut rng_fwd1 = SimpleRng::new(77);
    let (recon1, _, _) = vae1.forward(&input, 1, &mut rng_fwd1);

    let mut rng2 = SimpleRng::new(99);
    let mut vae2 = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let mut rng_fwd2 = SimpleRng::new(77);
    let (recon2, _, _) = vae2.forward(&input, 1, &mut rng_fwd2);

    for (a, b) in recon1.iter().zip(recon2.iter()) {
        assert_eq!(a, b, "Same seed should produce identical outputs");
    }
}

#[test]
fn test_vae_different_model_seeds_produce_different_outputs() {
    // Different model seeds should produce different outputs
    let input = vec![0.5f32; 8];
    let mut rng_fwd = SimpleRng::new(77);

    let mut rng1 = SimpleRng::new(1);
    let mut vae1 = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng1);
    let (recon1, _, _) = vae1.forward(&input, 1, &mut rng_fwd);

    let mut rng2 = SimpleRng::new(999);
    let mut vae2 = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let mut rng_fwd2 = SimpleRng::new(77);
    let (recon2, _, _) = vae2.forward(&input, 1, &mut rng_fwd2);

    let mut differs = false;
    for (a, b) in recon1.iter().zip(recon2.iter()) {
        if (a - b).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "Different model seeds should produce different outputs"
    );
}

// ============================================================================
// Batch Size Variation Tests
// ============================================================================

#[test]
fn test_vae_forward_various_batch_sizes() {
    // Test forward pass with multiple batch sizes
    let mut rng = SimpleRng::new(60);
    let input_size = 8;
    let batch_sizes = [1, 2, 4, 8, 16];

    for &batch_size in &batch_sizes {
        let mut vae = VariationalAutoencoder::new(input_size, &[4], 2, &[4], &mut rng);
        let mut rng_fwd = SimpleRng::new(batch_size as u64 + 100);
        let input = vec![0.5f32; input_size * batch_size];
        let (recon, mu, log_var) = vae.forward(&input, batch_size, &mut rng_fwd);
        assert_eq!(
            recon.len(),
            input_size * batch_size,
            "Reconstruction shape mismatch for batch_size={}",
            batch_size
        );
        assert_eq!(
            mu.len(),
            2 * batch_size,
            "mu shape mismatch for batch_size={}",
            batch_size
        );
        assert_eq!(
            log_var.len(),
            2 * batch_size,
            "log_var shape mismatch for batch_size={}",
            batch_size
        );
        for &v in &recon {
            assert!(
                v > 0.0 && v < 1.0,
                "Sigmoid output out of range for batch_size={}",
                batch_size
            );
        }
    }
}

#[test]
fn test_vae_backward_various_batch_sizes() {
    // Test backward pass with multiple batch sizes
    let input_size = 8;
    let batch_sizes = [1, 2, 4];

    for &batch_size in &batch_sizes {
        let mut rng = SimpleRng::new(61);
        let mut vae = VariationalAutoencoder::new(input_size, &[4], 2, &[4], &mut rng);
        let mut rng_fwd = SimpleRng::new(batch_size as u64 + 200);
        let input = vec![0.5f32; input_size * batch_size];
        vae.forward(&input, batch_size, &mut rng_fwd);
        // Should not panic
        vae.backward(&input, batch_size, 1.0);
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_vae_forward_large_input_values() {
    // Outputs should remain finite for large input values
    let mut rng = SimpleRng::new(70);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng_fwd = SimpleRng::new(71);
    let input = vec![100.0f32; 8];
    let (recon, mu, log_var) = vae.forward(&input, 1, &mut rng_fwd);
    for &v in &recon {
        assert!(
            v.is_finite(),
            "Reconstruction should be finite for large inputs"
        );
        assert!(
            (0.0..=1.0).contains(&v),
            "Sigmoid output should be in [0,1]"
        );
    }
    for &v in &mu {
        assert!(v.is_finite(), "mu should be finite for large inputs");
    }
    for &v in &log_var {
        assert!(v.is_finite(), "log_var should be finite for large inputs");
    }
}

#[test]
fn test_vae_forward_negative_input_values() {
    // Outputs should remain finite for negative input values
    let mut rng = SimpleRng::new(72);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng_fwd = SimpleRng::new(73);
    let input = vec![-10.0f32; 8];
    let (recon, _, _) = vae.forward(&input, 1, &mut rng_fwd);
    for &v in &recon {
        assert!(
            v.is_finite(),
            "Reconstruction should be finite for negative inputs"
        );
        // Sigmoid can saturate to exactly 0.0 or 1.0 for extreme pre-activations
        assert!(
            (0.0..=1.0).contains(&v),
            "Sigmoid output should be in [0,1]"
        );
    }
}

#[test]
fn test_vae_forward_mixed_input_values() {
    // Outputs should remain finite for mixed input values
    let mut rng = SimpleRng::new(74);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut rng_fwd = SimpleRng::new(75);
    let input = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0];
    let (recon, _, _) = vae.forward(&input, 1, &mut rng_fwd);
    for &v in &recon {
        assert!(
            v.is_finite(),
            "Reconstruction should be finite for mixed inputs"
        );
    }
}

// ============================================================================
// Multiple Forward/Backward Iteration Tests
// ============================================================================

#[test]
fn test_vae_multiple_forward_backward_iterations() {
    // Multiple training iterations should work without error
    let mut rng = SimpleRng::new(80);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8 * 3];
    let batch_size = 3;
    let lr = 0.005;
    let mut rng_train = SimpleRng::new(81);

    let mut last_loss = f32::MAX;
    for i in 0..10 {
        let (recon, mu, log_var) = vae.forward(&input, batch_size, &mut rng_train);
        let loss = vae.compute_elbo_loss(&recon, &input, &mu, &log_var, 1.0);
        assert!(loss.is_finite(), "ELBO must be finite at iteration {}", i);
        vae.backward(&input, batch_size, 1.0);
        vae.update_parameters(lr);
        last_loss = loss;
    }
    assert!(last_loss >= 0.0, "Final loss should be non-negative");
}

#[test]
fn test_vae_caches_work_across_multiple_calls() {
    // Each forward call should produce valid gradients even after multiple calls
    let mut rng = SimpleRng::new(90);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let mut rng_train = SimpleRng::new(91);

    // Multiple forward-backward cycles should not panic
    for _ in 0..5 {
        vae.forward(&input, 1, &mut rng_train);
        vae.backward(&input, 1, 1.0);
        vae.update_parameters(0.001);
    }
}
