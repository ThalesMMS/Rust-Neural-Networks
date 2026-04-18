use super::*;

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
