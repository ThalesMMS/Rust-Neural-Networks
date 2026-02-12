// Integration tests for layer normalization layer.
// Tests layer normalization behavior in isolation and integration with other layers.

use rust_neural_networks::layers::{layernorm::LayerNormLayer, DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Basic Forward Pass Tests
// ============================================================================

#[test]
fn test_layernorm_initialization() {
    // Test that layer initializes with correct parameters
    let layer = LayerNormLayer::new(128, 1e-5);

    assert_eq!(layer.input_size(), 128);
    assert_eq!(layer.output_size(), 128);
    assert_eq!(layer.parameter_count(), 256); // 128 gamma + 128 beta
    assert_eq!(layer.epsilon(), 1e-5);

    // Check gamma initialized to 1.0
    for &g in layer.gamma() {
        assert_eq!(g, 1.0);
    }

    // Check beta initialized to 0.0
    for &b in layer.beta() {
        assert_eq!(b, 0.0);
    }
}

#[test]
fn test_layernorm_output_dimensions() {
    // Test that output dimensions match input dimensions
    let layer = LayerNormLayer::new(64, 1e-5);
    let batch_size = 8;

    let input = vec![1.0f32; 64 * batch_size];
    let mut output = vec![0.0f32; 64 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), 64 * batch_size);
}

#[test]
fn test_layernorm_normalization_across_features() {
    // Test that layer norm normalizes across features for each sample
    let layer = LayerNormLayer::new(100, 1e-5);

    let batch_size = 5;
    // Create input with varying feature values per sample
    let mut input = vec![0.0f32; 100 * batch_size];
    for i in 0..batch_size {
        for j in 0..100 {
            input[i * 100 + j] = (i as f32) * 10.0 + (j as f32);
        }
    }

    let mut output = vec![0.0f32; 100 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    // With gamma=1 and beta=0 (default initialization), output should have
    // zero mean and unit variance per sample across features
    for i in 0..batch_size {
        // Compute mean for sample i across all features
        let mut mean = 0.0f32;
        for j in 0..100 {
            mean += output[i * 100 + j];
        }
        mean /= 100.0;

        // Compute variance for sample i across all features
        let mut variance = 0.0f32;
        for j in 0..100 {
            let diff = output[i * 100 + j] - mean;
            variance += diff * diff;
        }
        variance /= 100.0;

        // Check mean is close to 0
        assert!(
            mean.abs() < 1e-5,
            "Sample {}: mean should be ~0, got {}",
            i,
            mean
        );

        // Check variance is close to 1
        assert!(
            (variance - 1.0).abs() < 1e-4,
            "Sample {}: variance should be ~1, got {}",
            i,
            variance
        );
    }
}

#[test]
fn test_layernorm_deterministic_behavior() {
    // Test that same input produces same output (deterministic)
    let layer = LayerNormLayer::new(50, 1e-5);
    let batch_size = 10;

    let input = vec![1.0f32; 50 * batch_size];
    let mut output1 = vec![0.0f32; 50 * batch_size];
    let mut output2 = vec![0.0f32; 50 * batch_size];

    layer.forward(&input, &mut output1, batch_size);
    layer.forward(&input, &mut output2, batch_size);

    // Outputs should be identical (deterministic)
    for i in 0..output1.len() {
        assert_eq!(
            output1[i], output2[i],
            "Forward pass should be deterministic"
        );
    }
}

#[test]
fn test_layernorm_per_sample_independence() {
    // Test that normalization of each sample is independent
    let layer = LayerNormLayer::new(10, 1e-5);

    // Single sample
    let input_single = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut output_single = vec![0.0f32; 10];
    layer.forward(&input_single, &mut output_single, 1);

    // Two samples (first one same as above)
    let mut input_batch = input_single.clone();
    input_batch.extend_from_slice(&[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]);
    let mut output_batch = vec![0.0f32; 20];
    layer.forward(&input_batch, &mut output_batch, 2);

    // First sample in batch should match single sample output
    for i in 0..10 {
        assert!(
            (output_single[i] - output_batch[i]).abs() < 1e-6,
            "Layer norm should be independent per sample"
        );
    }
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

#[test]
fn test_layernorm_gradient_flow() {
    // Test that gradients flow correctly through layer norm
    let layer = LayerNormLayer::new(8, 1e-5);

    let batch_size = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].repeat(batch_size);
    let mut output = vec![0.0f32; 8 * batch_size];

    // Forward pass
    layer.forward(&input, &mut output, batch_size);

    // Backward pass with unit gradients
    let grad_output = vec![1.0f32; 8 * batch_size];
    let mut grad_input = vec![0.0f32; 8 * batch_size];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should be non-zero and finite
    for &grad in &grad_input {
        assert!(grad.is_finite(), "Gradient should be finite");
    }

    // For layer norm, sum of gradients across features for each sample should be ~0
    for i in 0..batch_size {
        let mut sum = 0.0f32;
        for j in 0..8 {
            sum += grad_input[i * 8 + j];
        }
        assert!(
            sum.abs() < 1e-4,
            "Sum of gradients across features should be ~0, got {}",
            sum
        );
    }
}

#[test]
fn test_layernorm_parameter_gradients() {
    // Test that gamma and beta gradients are accumulated correctly
    let mut layer = LayerNormLayer::new(4, 1e-5);

    let batch_size = 3;
    let input = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0];
    let mut output = vec![0.0f32; 4 * batch_size];

    // Forward pass
    layer.forward(&input, &mut output, batch_size);

    // Backward pass with varying gradients
    let grad_output = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let mut grad_input = vec![0.0f32; 4 * batch_size];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Parameter gradients are internal and will be used in update
    // We can't directly check them, but we can verify update works
    let gamma_before = layer.gamma().to_vec();
    let beta_before = layer.beta().to_vec();

    layer.update_parameters(0.01);

    let gamma_after = layer.gamma();
    let beta_after = layer.beta();

    // Parameters should have changed after update
    let mut gamma_changed = false;
    let mut beta_changed = false;

    for i in 0..4 {
        if (gamma_before[i] - gamma_after[i]).abs() > 1e-6 {
            gamma_changed = true;
        }
        if (beta_before[i] - beta_after[i]).abs() > 1e-6 {
            beta_changed = true;
        }
    }

    assert!(gamma_changed, "Gamma should be updated");
    assert!(beta_changed, "Beta should be updated");
}

// ============================================================================
// Affine Transformation Tests
// ============================================================================

#[test]
fn test_layernorm_identity_with_default_parameters() {
    // Test that with gamma=1, beta=0, layer norm normalizes to zero mean, unit variance
    let layer = LayerNormLayer::new(50, 1e-5);

    let batch_size = 10;
    // Create input with varying values across features for each sample
    let mut input = vec![0.0f32; 50 * batch_size];
    for i in 0..batch_size {
        for j in 0..50 {
            // Ensure variation across features: avoid zero variance
            input[i * 50 + j] = (i as f32 + 1.0) * (j as f32) * 0.1 + (j as f32);
        }
    }

    let mut output = vec![0.0f32; 50 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    // With gamma=1, beta=0, output should have zero mean and unit variance per sample
    for i in 0..batch_size {
        let mut mean = 0.0f32;
        let mut variance = 0.0f32;

        for j in 0..50 {
            mean += output[i * 50 + j];
        }
        mean /= 50.0;

        for j in 0..50 {
            let diff = output[i * 50 + j] - mean;
            variance += diff * diff;
        }
        variance /= 50.0;

        assert!(mean.abs() < 1e-5, "Mean should be ~0");
        assert!((variance - 1.0).abs() < 1e-4, "Variance should be ~1");
    }
}

// ============================================================================
// Batch Size Tests
// ============================================================================

#[test]
fn test_layernorm_single_sample() {
    // Test layer norm with batch_size=1
    let layer = LayerNormLayer::new(10, 1e-5);

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut output = vec![0.0f32; 10];

    layer.forward(&input, &mut output, 1);

    // Check that output is normalized across features
    let mut mean = 0.0f32;
    let mut variance = 0.0f32;

    for &val in &output {
        mean += val;
    }
    mean /= 10.0;

    for &val in &output {
        let diff = val - mean;
        variance += diff * diff;
    }
    variance /= 10.0;

    assert!(mean.abs() < 1e-5, "Mean should be ~0");
    assert!((variance - 1.0).abs() < 1e-4, "Variance should be ~1");
}

#[test]
fn test_layernorm_varying_batch_sizes() {
    // Test that layer norm works with different batch sizes
    let layer = LayerNormLayer::new(16, 1e-5);

    let batch_sizes = [1, 2, 4, 8, 16, 32];

    for &batch_size in &batch_sizes {
        let input = vec![1.0f32; 16 * batch_size];
        let mut output = vec![0.0f32; 16 * batch_size];

        layer.forward(&input, &mut output, batch_size);

        assert_eq!(output.len(), 16 * batch_size);
        for &val in &output {
            assert!(
                val.is_finite(),
                "Output should be finite for batch_size={}",
                batch_size
            );
        }
    }
}

// ============================================================================
// Epsilon Tests
// ============================================================================

#[test]
fn test_layernorm_different_epsilon_values() {
    // Test layer norm with different epsilon values
    let epsilons = [1e-3, 1e-5, 1e-7, 1e-10];

    for &eps in &epsilons {
        let layer = LayerNormLayer::new(8, eps);
        assert_eq!(layer.epsilon(), eps);

        let batch_size = 5;
        let input = vec![1.0f32; 8 * batch_size];
        let mut output = vec![0.0f32; 8 * batch_size];

        layer.forward(&input, &mut output, batch_size);

        // All outputs should be finite
        for &val in &output {
            assert!(
                val.is_finite(),
                "Output should be finite with epsilon={}",
                eps
            );
        }
    }
}

#[test]
fn test_layernorm_epsilon_prevents_division_by_zero() {
    // Test that epsilon prevents division by zero with constant features
    let layer = LayerNormLayer::new(10, 1e-5);

    let batch_size = 5;
    let input = vec![5.0f32; 10 * batch_size]; // Constant values (zero variance)
    let mut output = vec![0.0f32; 10 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    // With zero variance, normalization would divide by zero without epsilon
    // Output should still be finite
    for &val in &output {
        assert!(val.is_finite(), "Epsilon should prevent division by zero");
    }
}

// ============================================================================
// Integration with Other Layers
// ============================================================================

#[test]
fn test_layernorm_with_dense_layer() {
    // Test layer norm integrated with dense layer
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(10, 8, &mut rng);
    let layernorm = LayerNormLayer::new(8, 1e-5);

    let batch_size = 4;
    let input = vec![0.5f32; 10 * batch_size];
    let mut dense_output = vec![0.0f32; 8 * batch_size];
    let mut ln_output = vec![0.0f32; 8 * batch_size];

    // Forward through dense then layer norm
    dense.forward(&input, &mut dense_output, batch_size);
    layernorm.forward(&dense_output, &mut ln_output, batch_size);

    // Verify dimensions
    assert_eq!(ln_output.len(), 8 * batch_size);

    // Verify layer norm normalized the dense output per sample
    for i in 0..batch_size {
        let mut mean = 0.0f32;
        for j in 0..8 {
            mean += ln_output[i * 8 + j];
        }
        mean /= 8.0;

        assert!(
            mean.abs() < 1e-5,
            "Layer norm should normalize dense layer output"
        );
    }
}

#[test]
fn test_layernorm_backward_with_dense_layer() {
    // Test gradient flow through layer norm + dense layer
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(6, 4, &mut rng);
    let layernorm = LayerNormLayer::new(4, 1e-5);

    let batch_size = 8;
    let input = vec![0.5f32; 6 * batch_size];
    let mut dense_output = vec![0.0f32; 4 * batch_size];
    let mut ln_output = vec![0.0f32; 4 * batch_size];

    // Forward pass
    dense.forward(&input, &mut dense_output, batch_size);
    layernorm.forward(&dense_output, &mut ln_output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; 4 * batch_size];
    let mut grad_ln_input = vec![0.0f32; 4 * batch_size];
    let mut grad_dense_input = vec![0.0f32; 6 * batch_size];

    layernorm.backward(&dense_output, &grad_output, &mut grad_ln_input, batch_size);
    dense.backward(&input, &grad_ln_input, &mut grad_dense_input, batch_size);

    // All gradients should be finite
    for &grad in &grad_dense_input {
        assert!(grad.is_finite(), "Gradient should flow through both layers");
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_layernorm_large_values() {
    // Test layer norm with large input values
    let layer = LayerNormLayer::new(10, 1e-5);

    let batch_size = 5;
    let input: Vec<f32> = (0..10 * batch_size).map(|i| (i as f32) * 1000.0).collect();
    let mut output = vec![0.0f32; 10 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite despite large inputs
    for &val in &output {
        assert!(val.is_finite(), "Should handle large values");
    }
}

#[test]
fn test_layernorm_small_values() {
    // Test layer norm with small input values
    let layer = LayerNormLayer::new(10, 1e-5);

    let batch_size = 5;
    let input: Vec<f32> = (0..10 * batch_size).map(|i| (i as f32) * 1e-6).collect();
    let mut output = vec![0.0f32; 10 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite despite small inputs
    for &val in &output {
        assert!(val.is_finite(), "Should handle small values");
    }
}

#[test]
fn test_layernorm_mixed_positive_negative() {
    // Test layer norm with mixed positive and negative values
    let layer = LayerNormLayer::new(20, 1e-5);

    let batch_size = 10;
    let mut input = vec![0.0f32; 20 * batch_size];
    for i in 0..batch_size {
        for j in 0..20 {
            input[i * 20 + j] = if (i + j) % 2 == 0 {
                (i * j) as f32
            } else {
                -((i * j) as f32)
            };
        }
    }

    let mut output = vec![0.0f32; 20 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite
    for &val in &output {
        assert!(
            val.is_finite(),
            "Should handle mixed positive/negative values"
        );
    }
}

// ============================================================================
// Parameter Update Tests
// ============================================================================

#[test]
fn test_layernorm_parameter_update_clears_gradients() {
    // Test that update_parameters clears accumulated gradients
    let mut layer = LayerNormLayer::new(4, 1e-5);

    let batch_size = 8;
    // Create input with variation across features
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0, 5.0,
        10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0,
    ];
    let mut output = vec![0.0f32; 32]; // 4 features * 8 batch_size

    // First iteration: forward + backward + update
    layer.forward(&input, &mut output, batch_size);
    // Use non-uniform gradients to create non-zero gamma gradients
    let grad_output: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let mut grad_input = vec![0.0f32; 32];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Store parameters before first update
    let gamma_before_update1 = layer.gamma().to_vec();

    // First update (should apply gradients and clear them)
    layer.update_parameters(0.01);

    let gamma_after_update1 = layer.gamma().to_vec();

    // Parameters should have changed after first update
    for i in 0..4 {
        assert_ne!(
            gamma_before_update1[i], gamma_after_update1[i],
            "First update should change parameters"
        );
    }

    // Second iteration: forward + backward (gradients should start from zero)
    layer.forward(&input, &mut output, batch_size);
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Parameters should not change from just backward pass
    let gamma_after_backward2 = layer.gamma().to_vec();
    for i in 0..4 {
        assert_eq!(
            gamma_after_update1[i], gamma_after_backward2[i],
            "Backward pass alone should not change parameters"
        );
    }

    // Second update
    layer.update_parameters(0.01);

    let gamma_after_update2 = layer.gamma().to_vec();

    // If gradients were cleared properly after first update, the second update
    // should produce similar parameter changes (since input/gradients are same)
    for i in 0..4 {
        let change1 = gamma_after_update1[i] - gamma_before_update1[i];
        let change2 = gamma_after_update2[i] - gamma_after_update1[i];

        // Changes should be similar (within tolerance for numerical precision)
        assert!(
            (change1 - change2).abs() < 1e-5,
            "Parameter changes should be consistent if gradients are properly cleared"
        );
    }
}

// ============================================================================
// Single Feature Dimension Edge Case
// ============================================================================

#[test]
fn test_layernorm_single_feature_dimension() {
    // Test layer norm with size=1 (edge case)
    let layer = LayerNormLayer::new(1, 1e-5);

    let batch_size = 10;
    let input = vec![5.0f32; batch_size]; // Single feature per sample
    let mut output = vec![0.0f32; batch_size];

    // With single feature, variance is 0, so output depends on epsilon
    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite
    for &val in &output {
        assert!(
            val.is_finite(),
            "Output should be finite even with single feature"
        );
    }
}
