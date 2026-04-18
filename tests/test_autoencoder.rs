// Integration tests for the VanillaAutoencoder.
// Tests construction, forward/backward pass, loss computation, and training.

use rust_neural_networks::autoencoder::vanilla::VanillaAutoencoder;
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Construction and Initialization Tests
// ============================================================================

#[test]
fn test_vanilla_ae_construction_basic() {
    // Test basic autoencoder construction
    let mut rng = SimpleRng::new(42);
    let ae = VanillaAutoencoder::new(784, &[256], 64, &[256], &mut rng);

    assert_eq!(ae.input_size(), 784);
    assert_eq!(ae.latent_dim(), 64);
}

#[test]
fn test_vanilla_ae_construction_encoder_decoder_layers() {
    // Test that encoder and decoder layers are constructed correctly
    let mut rng = SimpleRng::new(42);
    // Architecture: 16 -> 8 -> 4 (encoder), 4 -> 8 -> 16 (decoder)
    let ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);

    // Encoder: 16->8 + 8->4 = 2 layers
    assert_eq!(ae.encoder_layers().len(), 2);
    // Decoder: 4->8 + 8->16 = 2 layers
    assert_eq!(ae.decoder_layers().len(), 2);
}

#[test]
fn test_vanilla_ae_construction_no_hidden_layers() {
    // Minimal AE: 10 -> 2 -> 10 (no hidden layers)
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(10, &[], 2, &[], &mut rng);

    assert_eq!(ae.encoder_layers().len(), 1);
    assert_eq!(ae.decoder_layers().len(), 1);
    assert_eq!(ae.input_size(), 10);
    assert_eq!(ae.latent_dim(), 2);
}

#[test]
fn test_vanilla_ae_construction_deep_architecture() {
    // Test deeper architecture with multiple hidden layers
    let mut rng = SimpleRng::new(7);
    // Architecture: 32 -> 16 -> 8 -> 4 (encoder), 4 -> 8 -> 16 -> 32 (decoder)
    let ae = VanillaAutoencoder::new(32, &[16, 8], 4, &[8, 16], &mut rng);

    assert_eq!(ae.encoder_layers().len(), 3); // 32->16, 16->8, 8->4
    assert_eq!(ae.decoder_layers().len(), 3); // 4->8, 8->16, 16->32
}

#[test]
fn test_vanilla_ae_construction_layer_sizes() {
    // Test that encoder_layer_sizes and decoder_layer_sizes are correct
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);

    let enc_sizes = ae.encoder_layer_sizes();
    assert_eq!(enc_sizes[0], 8);
    assert_eq!(enc_sizes[1], 4);
    assert_eq!(enc_sizes[2], 2);

    let dec_sizes = ae.decoder_layer_sizes();
    assert_eq!(dec_sizes[0], 2);
    assert_eq!(dec_sizes[1], 4);
    assert_eq!(dec_sizes[2], 8);
}

#[test]
fn test_vanilla_ae_construction_asymmetric_encoder_decoder() {
    // Test asymmetric architecture (different encoder/decoder hidden sizes)
    let mut rng = SimpleRng::new(5);
    let ae = VanillaAutoencoder::new(20, &[10, 6], 3, &[5], &mut rng);

    assert_eq!(ae.encoder_layers().len(), 3); // 20->10, 10->6, 6->3
    assert_eq!(ae.decoder_layers().len(), 2); // 3->5, 5->20
    assert_eq!(ae.input_size(), 20);
    assert_eq!(ae.latent_dim(), 3);
}

// ============================================================================
// Parameter Count Tests
// ============================================================================

#[test]
fn test_vanilla_ae_parameter_count_no_hidden() {
    // Verify parameter count with no hidden layers
    let mut rng = SimpleRng::new(0);
    // Encoder: 4->2 = 4*2+2 = 10; Decoder: 2->4 = 2*4+4 = 12; Total = 22
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let expected = (4 * 2 + 2) + (2 * 4 + 4);
    assert_eq!(ae.parameter_count(), expected);
}

#[test]
fn test_vanilla_ae_parameter_count_with_hidden() {
    // Verify parameter count with hidden layers
    let mut rng = SimpleRng::new(0);
    // Encoder: layer0 = 4*3+3 = 15, layer1 = 3*2+2 = 8 → total 23
    // Decoder: layer0 = 2*3+3 = 9, layer1 = 3*4+4 = 16 → total 25
    let ae = VanillaAutoencoder::new(4, &[3], 2, &[3], &mut rng);
    let expected = (4 * 3 + 3) + (3 * 2 + 2) + (2 * 3 + 3) + (3 * 4 + 4);
    assert_eq!(ae.parameter_count(), expected);
}

#[test]
fn test_vanilla_ae_parameter_count_scales_with_size() {
    // Verify that larger architecture has more parameters
    let mut rng = SimpleRng::new(1);
    let ae_small = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let ae_large = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    assert!(ae_large.parameter_count() > ae_small.parameter_count());
}

// ============================================================================
// Forward Pass Shape Tests
// ============================================================================

#[test]
fn test_vanilla_ae_forward_output_shape_batch1() {
    // Test output shape for batch size 1
    let mut rng = SimpleRng::new(1);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let output = ae.forward(&input, 1);
    assert_eq!(output.len(), 8);
}

#[test]
fn test_vanilla_ae_forward_output_shape_batch4() {
    // Test output shape for batch size 4
    let mut rng = SimpleRng::new(1);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8 * 4];
    let output = ae.forward(&input, 4);
    assert_eq!(output.len(), 8 * 4);
}

#[test]
fn test_vanilla_ae_forward_output_shape_batch_large() {
    // Test output shape for larger batch
    let mut rng = SimpleRng::new(2);
    let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let batch_size = 32;
    let input = vec![0.3f32; 16 * batch_size];
    let output = ae.forward(&input, batch_size);
    assert_eq!(output.len(), 16 * batch_size);
}

// ============================================================================
// Output Range Tests (Sigmoid Activation)
// ============================================================================

#[test]
fn test_vanilla_ae_output_in_sigmoid_range_batch1() {
    // All outputs should be strictly in (0, 1) due to Sigmoid
    let mut rng = SimpleRng::new(2);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.3f32; 8];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vanilla_ae_output_in_sigmoid_range_batch4() {
    // Verify sigmoid range with multiple samples in batch
    let mut rng = SimpleRng::new(3);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.7f32; 8 * 4];
    let output = ae.forward(&input, 4);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vanilla_ae_output_in_sigmoid_range_zero_input() {
    // Test with zero input
    let mut rng = SimpleRng::new(4);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.0f32; 8];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

#[test]
fn test_vanilla_ae_output_in_sigmoid_range_ones_input() {
    // Test with all-ones input
    let mut rng = SimpleRng::new(5);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![1.0f32; 8];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
    }
}

// ============================================================================
// Encode / Decode Separately Tests
// ============================================================================

#[test]
fn test_vanilla_ae_encode_output_shape_batch1() {
    // Test that encode returns batch_size × latent_dim
    let mut rng = SimpleRng::new(10);
    let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16];
    let latent = ae.encode(&input, 1);
    assert_eq!(latent.len(), 4);
}

#[test]
fn test_vanilla_ae_encode_output_shape_batch2() {
    // Test that encode returns batch_size × latent_dim
    let mut rng = SimpleRng::new(10);
    let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let input = vec![0.5f32; 16 * 2];
    let latent = ae.encode(&input, 2);
    assert_eq!(latent.len(), 4 * 2);
}

#[test]
fn test_vanilla_ae_decode_output_shape_batch1() {
    // Test that decode returns batch_size × input_size
    let mut rng = SimpleRng::new(11);
    let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let latent = vec![0.1f32; 4];
    let output = ae.decode(&latent, 1);
    assert_eq!(output.len(), 16);
}

#[test]
fn test_vanilla_ae_decode_output_shape_batch2() {
    // Test that decode returns batch_size × input_size for batch of 2
    let mut rng = SimpleRng::new(11);
    let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
    let latent = vec![0.1f32; 4 * 2];
    let output = ae.decode(&latent, 2);
    assert_eq!(output.len(), 16 * 2);
}

#[test]
fn test_vanilla_ae_decode_output_in_sigmoid_range() {
    // Decode output should always be in (0, 1)
    let mut rng = SimpleRng::new(12);
    let mut ae = VanillaAutoencoder::new(10, &[6], 3, &[6], &mut rng);
    let latent = vec![0.5f32; 3 * 4];
    let output = ae.decode(&latent, 4);
    for &v in &output {
        assert!(v > 0.0 && v < 1.0, "Decode output {} not in (0,1)", v);
    }
}

#[test]
fn test_vanilla_ae_encode_then_decode_matches_forward() {
    // encode + decode should produce the same result as forward
    let mut rng = SimpleRng::new(42);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.3f32; 8 * 2];

    // Encode then decode
    let latent = ae.encode(&input, 2);
    let output_decode = ae.decode(&latent, 2);

    // Full forward (re-initialize to reset caches)
    let mut rng2 = SimpleRng::new(42);
    let mut ae2 = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let output_forward = ae2.forward(&input, 2);

    assert_eq!(output_decode.len(), output_forward.len());
    for (a, b) in output_decode.iter().zip(output_forward.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "encode+decode should match forward, got {} vs {}",
            a,
            b
        );
    }
}

// ============================================================================
// Loss Computation Tests
// ============================================================================

#[test]
fn test_vanilla_ae_loss_zero_on_perfect_reconstruction() {
    // MSE should be 0 when output == target
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let output = vec![0.5f32; 4];
    let target = vec![0.5f32; 4];
    let loss = ae.compute_loss(&output, &target);
    assert!(
        loss < 1e-6,
        "Loss should be ~0 for perfect reconstruction, got {}",
        loss
    );
}

#[test]
fn test_vanilla_ae_loss_nonzero_on_imperfect_reconstruction() {
    // MSE should be nonzero when output != target
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let output = vec![0.0f32; 4];
    let target = vec![1.0f32; 4];
    let loss = ae.compute_loss(&output, &target);
    // MSE = mean((0-1)^2) = 1.0
    assert!((loss - 1.0).abs() < 1e-5, "Expected MSE=1.0, got {}", loss);
}

#[test]
fn test_vanilla_ae_loss_non_negative() {
    // MSE is always non-negative
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(8, &[4], 2, &[], &mut rng);
    let output = vec![0.2f32, 0.8, 0.1, 0.9, 0.5, 0.3, 0.7, 0.4];
    let target = vec![0.9f32, 0.2, 0.8, 0.1, 0.6, 0.4, 0.3, 0.7];
    let loss = ae.compute_loss(&output, &target);
    assert!(loss >= 0.0, "MSE should be non-negative, got {}", loss);
}

#[test]
fn test_vanilla_ae_loss_symmetric() {
    // MSE(a, b) == MSE(b, a)
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let a = vec![0.2f32, 0.8, 0.3, 0.7];
    let b = vec![0.5f32, 0.4, 0.6, 0.1];
    let loss_ab = ae.compute_loss(&a, &b);
    let loss_ba = ae.compute_loss(&b, &a);
    assert!(
        (loss_ab - loss_ba).abs() < 1e-6,
        "MSE should be symmetric: {} vs {}",
        loss_ab,
        loss_ba
    );
}

#[test]
fn test_vanilla_ae_loss_scales_with_batch_size() {
    // MSE averages over all elements, so consistent regardless of how we split
    let mut rng = SimpleRng::new(0);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);

    // Single sample
    let output_single = vec![0.0f32, 0.0, 0.0, 0.0];
    let target_single = vec![1.0f32, 1.0, 1.0, 1.0];
    let loss_single = ae.compute_loss(&output_single, &target_single);

    // Two identical samples
    let output_batch = vec![0.0f32; 8];
    let target_batch = vec![1.0f32; 8];
    let loss_batch = ae.compute_loss(&output_batch, &target_batch);

    // Both should give MSE = 1.0
    assert!(
        (loss_single - loss_batch).abs() < 1e-5,
        "Loss should be consistent: single={}, batch={}",
        loss_single,
        loss_batch
    );
}

#[test]
fn test_vanilla_ae_loss_computed_from_forward_output() {
    // Test loss when computed on actual forward output
    let mut rng = SimpleRng::new(13);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    let output = ae.forward(&input, 1);
    let loss = ae.compute_loss(&output, &input);
    // Loss should be finite and non-negative
    assert!(loss.is_finite(), "Loss should be finite");
    assert!(loss >= 0.0, "Loss should be non-negative");
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

#[test]
fn test_vanilla_ae_backward_does_not_panic_batch1() {
    // Backward pass should not panic with batch_size=1
    let mut rng = SimpleRng::new(42);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];
    ae.forward(&input, 1);
    ae.backward(&input, 1);
}

#[test]
fn test_vanilla_ae_backward_does_not_panic_batch4() {
    // Backward pass should not panic with batch_size=4
    let mut rng = SimpleRng::new(43);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8 * 4];
    ae.forward(&input, 4);
    ae.backward(&input, 4);
}

#[test]
fn test_vanilla_ae_backward_does_not_panic_no_hidden() {
    // Backward should work with minimal architecture
    let mut rng = SimpleRng::new(44);
    let mut ae = VanillaAutoencoder::new(6, &[], 2, &[], &mut rng);
    let input = vec![0.3f32; 6];
    ae.forward(&input, 1);
    ae.backward(&input, 1);
}

#[test]
fn test_vanilla_ae_backward_does_not_panic_deep() {
    // Backward should work with deeper architecture
    let mut rng = SimpleRng::new(45);
    let mut ae = VanillaAutoencoder::new(16, &[8, 4], 2, &[4, 8], &mut rng);
    let input = vec![0.5f32; 16 * 3];
    ae.forward(&input, 3);
    ae.backward(&input, 3);
}

// ============================================================================
// Parameter Update Tests
// ============================================================================

#[test]
fn test_vanilla_ae_update_parameters_changes_weights() {
    // Verify that update_parameters changes the model behavior
    let mut rng = SimpleRng::new(50);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];

    // Forward pass 1
    let output_before = ae.forward(&input, 1);

    // Backward and update
    ae.backward(&input, 1);
    ae.update_parameters(0.01);

    // Forward pass 2 - output should differ because weights changed
    let output_after = ae.forward(&input, 1);

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
fn test_vanilla_ae_update_parameters_high_lr_larger_change() {
    // Higher learning rate should produce larger parameter changes
    let mut rng = SimpleRng::new(51);
    let input = vec![0.5f32; 8];

    // Small learning rate
    let mut ae_small = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let output_before_small = ae_small.forward(&input, 1);
    ae_small.backward(&input, 1);
    ae_small.update_parameters(0.001);
    let output_after_small = ae_small.forward(&input, 1);

    // Large learning rate (same seed)
    let mut rng2 = SimpleRng::new(51);
    let mut ae_large = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let output_before_large = ae_large.forward(&input, 1);
    ae_large.backward(&input, 1);
    ae_large.update_parameters(0.1);
    let output_after_large = ae_large.forward(&input, 1);

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
fn test_vanilla_ae_loss_decreases_with_training() {
    // Loss should decrease over multiple training steps
    let mut rng = SimpleRng::new(100);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![
        0.9f32, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, // sample 1
        0.1f32, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, // sample 2
    ];
    let batch_size = 2;
    let lr = 0.01;

    let initial_output = ae.forward(&input, batch_size);
    let initial_loss = ae.compute_loss(&initial_output, &input);

    // Train for several steps
    for _ in 0..50 {
        let output = ae.forward(&input, batch_size);
        let _ = ae.compute_loss(&output, &input);
        ae.backward(&input, batch_size);
        ae.update_parameters(lr);
    }

    let final_output = ae.forward(&input, batch_size);
    let final_loss = ae.compute_loss(&final_output, &input);

    assert!(
        final_loss < initial_loss,
        "Loss should decrease during training: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_vanilla_ae_loss_decreases_no_hidden() {
    // Loss should also decrease for minimal architecture
    let mut rng = SimpleRng::new(200);
    let mut ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let input = vec![0.8f32, 0.2, 0.6, 0.4];
    let lr = 0.01;

    let initial_output = ae.forward(&input, 1);
    let initial_loss = ae.compute_loss(&initial_output, &input);

    for _ in 0..100 {
        let output = ae.forward(&input, 1);
        let _ = ae.compute_loss(&output, &input);
        ae.backward(&input, 1);
        ae.update_parameters(lr);
    }

    let final_output = ae.forward(&input, 1);
    let final_loss = ae.compute_loss(&final_output, &input);

    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={}, final={}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_vanilla_ae_loss_finite_during_training() {
    // Loss should remain finite throughout training
    let mut rng = SimpleRng::new(300);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];

    for i in 0..20 {
        let output = ae.forward(&input, 1);
        let loss = ae.compute_loss(&output, &input);
        assert!(
            loss.is_finite(),
            "Loss should be finite at step {}, got {}",
            i,
            loss
        );
        ae.backward(&input, 1);
        ae.update_parameters(0.001);
    }
}

// ============================================================================
// Reproducibility Tests
// ============================================================================

#[test]
fn test_vanilla_ae_deterministic_with_same_seed() {
    // Same seed should produce identical outputs
    let input = vec![0.3f32, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4];

    let mut rng1 = SimpleRng::new(99);
    let mut ae1 = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng1);
    let output1 = ae1.forward(&input, 1);

    let mut rng2 = SimpleRng::new(99);
    let mut ae2 = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let output2 = ae2.forward(&input, 1);

    for (a, b) in output1.iter().zip(output2.iter()) {
        assert_eq!(a, b, "Same seed should produce identical outputs");
    }
}

#[test]
fn test_vanilla_ae_different_seeds_produce_different_outputs() {
    // Different seeds should produce different outputs (with high probability)
    let input = vec![0.5f32; 8];

    let mut rng1 = SimpleRng::new(1);
    let mut ae1 = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng1);
    let output1 = ae1.forward(&input, 1);

    let mut rng2 = SimpleRng::new(999);
    let mut ae2 = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng2);
    let output2 = ae2.forward(&input, 1);

    let mut differs = false;
    for (a, b) in output1.iter().zip(output2.iter()) {
        if (a - b).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(differs, "Different seeds should produce different outputs");
}

// ============================================================================
// Batch Size Variation Tests
// ============================================================================

#[test]
fn test_vanilla_ae_forward_various_batch_sizes() {
    // Test forward pass with multiple batch sizes
    let mut rng = SimpleRng::new(60);
    let input_size = 8;
    let batch_sizes = [1, 2, 4, 8, 16];

    for &batch_size in &batch_sizes {
        let mut ae = VanillaAutoencoder::new(input_size, &[4], 2, &[4], &mut rng);
        let input = vec![0.5f32; input_size * batch_size];
        let output = ae.forward(&input, batch_size);
        assert_eq!(
            output.len(),
            input_size * batch_size,
            "Output shape mismatch for batch_size={}",
            batch_size
        );
        for &v in &output {
            assert!(
                v > 0.0 && v < 1.0,
                "Sigmoid output out of range for batch_size={}",
                batch_size
            );
        }
    }
}

#[test]
fn test_vanilla_ae_backward_various_batch_sizes() {
    // Test backward pass with multiple batch sizes
    let input_size = 8;
    let batch_sizes = [1, 2, 4];

    for &batch_size in &batch_sizes {
        let mut rng = SimpleRng::new(61);
        let mut ae = VanillaAutoencoder::new(input_size, &[4], 2, &[4], &mut rng);
        let input = vec![0.5f32; input_size * batch_size];
        ae.forward(&input, batch_size);
        // Should not panic
        ae.backward(&input, batch_size);
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_vanilla_ae_forward_large_input_values() {
    // Outputs should remain finite for large input values.
    // Note: Sigmoid can saturate to exactly 0.0 or 1.0 with large f32 inputs.
    let mut rng = SimpleRng::new(70);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![100.0f32; 8];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v.is_finite(), "Output should be finite for large inputs");
        assert!(
            (0.0..=1.0).contains(&v),
            "Sigmoid output should be in [0,1]"
        );
    }
}

#[test]
fn test_vanilla_ae_forward_negative_input_values() {
    // Outputs should remain finite for negative input values
    let mut rng = SimpleRng::new(71);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![-10.0f32; 8];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v.is_finite(), "Output should be finite for negative inputs");
        assert!(v > 0.0 && v < 1.0, "Sigmoid output should be in (0,1)");
    }
}

#[test]
fn test_vanilla_ae_forward_mixed_input_values() {
    // Outputs should remain finite for mixed input values
    let mut rng = SimpleRng::new(72);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0];
    let output = ae.forward(&input, 1);
    for &v in &output {
        assert!(v.is_finite(), "Output should be finite for mixed inputs");
    }
}

// ============================================================================
// Multiple Forward/Backward Iteration Tests
// ============================================================================

#[test]
fn test_vanilla_ae_multiple_forward_backward_iterations() {
    // Multiple training iterations should work without error
    let mut rng = SimpleRng::new(80);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8 * 3];
    let batch_size = 3;
    let lr = 0.005;

    let mut last_loss = f32::MAX;
    for i in 0..10 {
        let output = ae.forward(&input, batch_size);
        let loss = ae.compute_loss(&output, &input);
        assert!(loss.is_finite(), "Loss must be finite at iteration {}", i);
        ae.backward(&input, batch_size);
        ae.update_parameters(lr);
        last_loss = loss;
    }
    assert!(last_loss >= 0.0, "Final loss should be non-negative");
}

#[test]
fn test_vanilla_ae_caches_reset_between_forward_calls() {
    // Each forward call should produce valid gradients even after multiple calls
    let mut rng = SimpleRng::new(90);
    let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
    let input = vec![0.5f32; 8];

    // Multiple forward-backward cycles should not panic
    for _ in 0..5 {
        ae.forward(&input, 1);
        ae.backward(&input, 1);
        ae.update_parameters(0.001);
    }
}

// ============================================================================
// Reconstruction Quality Tests
// ============================================================================

#[test]
fn test_vanilla_ae_reconstruction_improves_towards_constant_target() {
    // With a constant target, reconstruction should improve
    let mut rng = SimpleRng::new(400);
    let mut ae = VanillaAutoencoder::new(4, &[3], 2, &[3], &mut rng);
    // Target: pixel values all 0.8
    let input = vec![0.8f32, 0.8, 0.8, 0.8];

    let initial_output = ae.forward(&input, 1);
    let initial_loss = ae.compute_loss(&initial_output, &input);

    // Train for many steps
    for _ in 0..200 {
        let output = ae.forward(&input, 1);
        let _ = ae.compute_loss(&output, &input);
        ae.backward(&input, 1);
        ae.update_parameters(0.01);
    }

    let final_output = ae.forward(&input, 1);
    let final_loss = ae.compute_loss(&final_output, &input);

    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={:.6}, final={:.6}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_vanilla_ae_reconstruction_loss_all_zeros_target() {
    // Reconstruction loss when target is all zeros
    let mut rng = SimpleRng::new(500);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    // Since output is sigmoid, it's always > 0, so loss from zero target > 0
    let output = vec![0.5f32; 4]; // Sigmoid midpoint
    let target = vec![0.0f32; 4];
    let loss = ae.compute_loss(&output, &target);
    // MSE = 0.5^2 = 0.25
    assert!(
        (loss - 0.25).abs() < 1e-5,
        "Expected MSE=0.25, got {}",
        loss
    );
}

#[test]
fn test_vanilla_ae_reconstruction_loss_all_ones_target() {
    // Reconstruction loss when target is all ones
    let mut rng = SimpleRng::new(501);
    let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
    let output = vec![0.5f32; 4]; // Sigmoid midpoint
    let target = vec![1.0f32; 4];
    let loss = ae.compute_loss(&output, &target);
    // MSE = (0.5-1.0)^2 = 0.25
    assert!(
        (loss - 0.25).abs() < 1e-5,
        "Expected MSE=0.25, got {}",
        loss
    );
}

// ============================================================================
// Layer Accessor Tests
// ============================================================================

#[test]
fn test_vanilla_ae_encoder_layers_accessor() {
    // encoder_layers() should return the correct number of layers
    let mut rng = SimpleRng::new(600);
    let ae = VanillaAutoencoder::new(16, &[8, 6], 4, &[6, 8], &mut rng);
    // Encoder: 16->8, 8->6, 6->4 = 3 layers
    assert_eq!(ae.encoder_layers().len(), 3);
}

#[test]
fn test_vanilla_ae_decoder_layers_accessor() {
    // decoder_layers() should return the correct number of layers
    let mut rng = SimpleRng::new(601);
    let ae = VanillaAutoencoder::new(16, &[8], 4, &[6, 8], &mut rng);
    // Decoder: 4->6, 6->8, 8->16 = 3 layers
    assert_eq!(ae.decoder_layers().len(), 3);
}

#[test]
fn test_vanilla_ae_encoder_layer_sizes_accessor() {
    // encoder_layer_sizes() should return complete sequence
    let mut rng = SimpleRng::new(602);
    let ae = VanillaAutoencoder::new(12, &[6], 3, &[], &mut rng);
    let enc = ae.encoder_layer_sizes();
    // Should be [12, 6, 3]
    assert_eq!(enc.len(), 3);
    assert_eq!(enc[0], 12);
    assert_eq!(enc[1], 6);
    assert_eq!(enc[2], 3);
}

#[test]
fn test_vanilla_ae_decoder_layer_sizes_accessor() {
    // decoder_layer_sizes() should return complete sequence
    let mut rng = SimpleRng::new(603);
    let ae = VanillaAutoencoder::new(12, &[], 3, &[6], &mut rng);
    let dec = ae.decoder_layer_sizes();
    // Should be [3, 6, 12]
    assert_eq!(dec.len(), 3);
    assert_eq!(dec[0], 3);
    assert_eq!(dec[1], 6);
    assert_eq!(dec[2], 12);
}
