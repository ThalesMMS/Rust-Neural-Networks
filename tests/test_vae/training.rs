use super::*;

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

#[test]
fn test_vae_gradient_magnitudes_report_layer_norms_after_backward() {
    let mut rng = SimpleRng::new(52);
    let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let mut forward_rng = SimpleRng::new(53);
    let input = vec![0.9f32, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4];

    vae.forward(&input, 1, &mut forward_rng);
    vae.backward(&input, 1, 1.0);

    let gradients = vae.gradient_magnitudes();
    let names: Vec<&str> = gradients.iter().map(|(name, _, _)| name.as_str()).collect();

    assert_eq!(
        names,
        vec!["encoder_0", "mu", "log_var", "decoder_0", "decoder_1"]
    );
    assert!(!names.contains(&"vae_all_layers"));
    assert!(gradients.iter().all(|(_, weight_norm, bias_norm)| {
        weight_norm.is_finite() && *weight_norm >= 0.0 && bias_norm.is_finite() && *bias_norm >= 0.0
    }));
    assert!(
        gradients
            .iter()
            .any(|(_, weight_norm, bias_norm)| *weight_norm > 0.0 || *bias_norm > 0.0),
        "at least one VAE layer should have a non-zero accumulated gradient norm"
    );
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
