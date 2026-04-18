use super::*;

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
