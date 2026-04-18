use super::*;

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
