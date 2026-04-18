// Integration tests for batch normalization layer.
// Tests batch normalization behavior in isolation and integration with other layers.

use rust_neural_networks::layers::{batchnorm::BatchNormLayer, DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Basic Forward Pass Tests
// ============================================================================

#[test]
fn test_batchnorm_initialization() {
    // Test that layer initializes with correct parameters
    let layer = BatchNormLayer::new(128, 1e-5, 0.9);

    assert_eq!(layer.input_size(), 128);
    assert_eq!(layer.output_size(), 128);
    assert_eq!(layer.parameter_count(), 256); // 128 gamma + 128 beta
    assert!(layer.is_training()); // Default to training mode
    assert_eq!(layer.epsilon(), 1e-5);
    assert_eq!(layer.momentum(), 0.9);

    // Check gamma initialized to 1.0
    for &g in layer.gamma() {
        assert_eq!(g, 1.0);
    }

    // Check beta initialized to 0.0
    for &b in layer.beta() {
        assert_eq!(b, 0.0);
    }

    // Check running statistics initialized to 0.0
    for &m in &layer.running_mean() {
        assert_eq!(m, 0.0);
    }
    for &v in &layer.running_var() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_batchnorm_output_dimensions() {
    // Test that output dimensions match input dimensions
    let layer = BatchNormLayer::new(64, 1e-5, 0.9);
    let batch_size = 8;

    let input = vec![1.0f32; 64 * batch_size];
    let mut output = vec![0.0f32; 64 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), 64 * batch_size);
}

#[test]
fn test_batchnorm_training_mode_normalization() {
    // Test that training mode produces zero mean and unit variance
    let mut layer = BatchNormLayer::new(10, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 100;
    // Create input with varying statistics per feature
    let mut input = vec![0.0f32; 10 * batch_size];
    for i in 0..batch_size {
        for j in 0..10 {
            input[i * 10 + j] = (i as f32) * 0.5 + (j as f32) * 10.0;
        }
    }

    let mut output = vec![0.0f32; 10 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    // With gamma=1 and beta=0 (default initialization), output should have
    // zero mean and unit variance per feature
    for j in 0..10 {
        // Compute mean for feature j
        let mut mean = 0.0f32;
        for i in 0..batch_size {
            mean += output[i * 10 + j];
        }
        mean /= batch_size as f32;

        // Compute variance for feature j
        let mut variance = 0.0f32;
        for i in 0..batch_size {
            let diff = output[i * 10 + j] - mean;
            variance += diff * diff;
        }
        variance /= batch_size as f32;

        // Check mean is close to 0
        assert!(
            mean.abs() < 1e-5,
            "Feature {}: mean should be ~0, got {}",
            j,
            mean
        );

        // Check variance is close to 1
        assert!(
            (variance - 1.0).abs() < 1e-4,
            "Feature {}: variance should be ~1, got {}",
            j,
            variance
        );
    }
}

#[test]
fn test_batchnorm_inference_mode_uses_running_stats() {
    // Test that inference mode uses running statistics
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);

    let batch_size = 10;
    // Create training batch with known statistics
    let input_train = vec![
        1.0, 2.0, 3.0, 4.0, // Sample 1
        2.0, 4.0, 6.0, 8.0, // Sample 2
        3.0, 6.0, 9.0, 12.0, // Sample 3
        4.0, 8.0, 12.0, 16.0, // Sample 4
        5.0, 10.0, 15.0, 20.0, // Sample 5
        6.0, 12.0, 18.0, 24.0, // Sample 6
        7.0, 14.0, 21.0, 28.0, // Sample 7
        8.0, 16.0, 24.0, 32.0, // Sample 8
        9.0, 18.0, 27.0, 36.0, // Sample 9
        10.0, 20.0, 30.0, 40.0, // Sample 10
    ];

    let mut output_train = vec![0.0f32; 4 * batch_size];

    // Forward pass in training mode to populate running statistics
    layer.set_training(true);
    layer.forward(&input_train, &mut output_train, batch_size);

    // Now test inference mode with a single sample
    layer.set_training(false);
    let input_test = vec![5.5, 11.0, 16.5, 22.0]; // Single sample
    let mut output_test = vec![0.0f32; 4];
    layer.forward(&input_test, &mut output_test, 1);

    // The output should be normalized using running statistics
    // Since running stats were updated, output should be reasonable
    for &val in &output_test {
        assert!(val.is_finite(), "Output should be finite");
    }
}

// ============================================================================
// Training vs Inference Mode Tests
// ============================================================================

#[test]
fn test_batchnorm_mode_switching() {
    // Test switching between training and inference modes
    let mut layer = BatchNormLayer::new(8, 1e-5, 0.9);
    let batch_size = 4;

    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].repeat(batch_size);
    let mut output_train = vec![0.0f32; 8 * batch_size];
    let mut output_inference = vec![0.0f32; 8 * batch_size];

    // Training mode
    layer.set_training(true);
    assert!(layer.is_training());
    layer.forward(&input, &mut output_train, batch_size);

    // Inference mode (with no prior training, running stats are zero)
    layer.set_training(false);
    assert!(!layer.is_training());
    layer.forward(&input, &mut output_inference, batch_size);

    // Outputs should differ because modes behave differently
    let mut differs = false;
    for i in 0..output_train.len() {
        if (output_train[i] - output_inference[i]).abs() > 1e-6 {
            differs = true;
            break;
        }
    }
    assert!(differs, "Training and inference outputs should differ");
}

#[test]
fn test_batchnorm_training_mode_deterministic_with_same_input() {
    // Test that same input produces same output in training mode
    let layer = BatchNormLayer::new(6, 1e-5, 0.9);
    let batch_size = 5;

    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].repeat(batch_size);
    let mut output1 = vec![0.0f32; 6 * batch_size];
    let mut output2 = vec![0.0f32; 6 * batch_size];

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

// ============================================================================
// Running Statistics Tests
// ============================================================================

#[test]
fn test_batchnorm_running_statistics_update() {
    // Test that running statistics are updated during training
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 10;
    // Create input with known mean and variance
    // Feature 0: mean=5, Feature 1: mean=10, Feature 2: mean=15
    let mut input = vec![0.0f32; 3 * batch_size];
    for i in 0..batch_size {
        input[i * 3] = 5.0 + (i as f32 - 5.0);
        input[i * 3 + 1] = 10.0 + (i as f32 - 5.0);
        input[i * 3 + 2] = 15.0 + (i as f32 - 5.0);
    }

    let mut output = vec![0.0f32; 3 * batch_size];

    // Initial running statistics should be zero
    let running_mean_before = layer.running_mean();
    let running_var_before = layer.running_var();
    for &m in &running_mean_before {
        assert_eq!(m, 0.0);
    }
    for &v in &running_var_before {
        assert_eq!(v, 0.0);
    }

    // Forward pass should update running statistics
    layer.forward(&input, &mut output, batch_size);

    let running_mean_after = layer.running_mean();
    let running_var_after = layer.running_var();

    // Running statistics should be non-zero now
    // With momentum=0.9: running = 0.9 * 0 + 0.1 * batch_stat
    for i in 0..3 {
        assert!(
            running_mean_after[i] != 0.0,
            "Running mean should be updated"
        );
        assert!(running_var_after[i] != 0.0, "Running var should be updated");
    }
}

#[test]
fn test_batchnorm_running_statistics_momentum() {
    // Test that momentum parameter affects running statistics correctly
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.8); // momentum = 0.8
    layer.set_training(true);

    let batch_size = 10;
    let input = [1.0, 2.0].repeat(batch_size); // Simple constant features
    let mut output = vec![0.0f32; 2 * batch_size];

    // First forward pass
    layer.forward(&input, &mut output, batch_size);
    let running_mean_1 = layer.running_mean();

    // Running mean should be (1 - 0.8) * batch_mean = 0.2 * batch_mean
    // For constant inputs, batch_mean = input values
    // Feature 0: 0.2 * 1.0 = 0.2
    // Feature 1: 0.2 * 2.0 = 0.4
    assert!(
        (running_mean_1[0] - 0.2).abs() < 1e-5,
        "Running mean should follow momentum update"
    );
    assert!(
        (running_mean_1[1] - 0.4).abs() < 1e-5,
        "Running mean should follow momentum update"
    );
}

#[test]
fn test_batchnorm_inference_mode_no_statistics_update() {
    // Test that inference mode doesn't update running statistics
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(false);

    let batch_size = 5;
    let input = [1.0, 2.0, 3.0, 4.0].repeat(batch_size);
    let mut output = vec![0.0f32; 4 * batch_size];

    let running_mean_before = layer.running_mean();
    let running_var_before = layer.running_var();

    layer.forward(&input, &mut output, batch_size);

    let running_mean_after = layer.running_mean();
    let running_var_after = layer.running_var();

    // Running statistics should remain unchanged in inference mode
    for i in 0..4 {
        assert_eq!(
            running_mean_before[i], running_mean_after[i],
            "Inference mode should not update running mean"
        );
        assert_eq!(
            running_var_before[i], running_var_after[i],
            "Inference mode should not update running var"
        );
    }
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

#[test]
fn test_batchnorm_gradient_flow_training_mode() {
    // Test that gradients flow correctly through batch norm in training mode
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 8;
    let input = [1.0, 2.0, 3.0, 4.0].repeat(batch_size);
    let mut output = vec![0.0f32; 4 * batch_size];

    // Forward pass
    layer.forward(&input, &mut output, batch_size);

    // Backward pass with unit gradients
    let grad_output = vec![1.0f32; 4 * batch_size];
    let mut grad_input = vec![0.0f32; 4 * batch_size];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should be non-zero and finite
    for &grad in &grad_input {
        assert!(grad.is_finite(), "Gradient should be finite");
    }

    // Sum of gradients across batch should be zero (property of batch norm)
    for j in 0..4 {
        let mut sum = 0.0f32;
        for i in 0..batch_size {
            sum += grad_input[i * 4 + j];
        }
        assert!(
            sum.abs() < 1e-4,
            "Sum of gradients across batch should be ~0, got {}",
            sum
        );
    }
}

#[test]
fn test_batchnorm_gradient_flow_inference_mode() {
    // Test gradient flow in inference mode (simpler pass-through)
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);

    // First, populate running statistics with training
    layer.set_training(true);
    let batch_size = 10;
    let input_train = [1.0, 2.0, 3.0, 4.0].repeat(batch_size);
    let mut output_train = vec![0.0f32; 4 * batch_size];
    layer.forward(&input_train, &mut output_train, batch_size);

    // Now test inference mode backward pass
    layer.set_training(false);
    let input = [1.5, 2.5, 3.5, 4.5].repeat(batch_size);
    let mut output = vec![0.0f32; 4 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; 4 * batch_size];
    let mut grad_input = vec![0.0f32; 4 * batch_size];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // All gradients should be finite
    for &grad in &grad_input {
        assert!(grad.is_finite(), "Gradient should be finite");
    }
}

#[test]
fn test_batchnorm_parameter_gradients() {
    // Test that gamma and beta gradients are accumulated correctly
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 4;
    let input = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0];
    let mut output = vec![0.0f32; 3 * batch_size];

    // Forward pass
    layer.forward(&input, &mut output, batch_size);

    // Backward pass with varying gradients
    let grad_output = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let mut grad_input = vec![0.0f32; 3 * batch_size];

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

    for i in 0..3 {
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
fn test_batchnorm_identity_with_default_parameters() {
    // Test that with gamma=1, beta=0, batch norm is close to identity
    // (after normalization to zero mean, unit variance)
    let layer = BatchNormLayer::new(5, 1e-5, 0.9);

    let batch_size = 20;
    // Create input with zero mean and unit variance per feature
    let mut input = vec![0.0f32; 5 * batch_size];
    for i in 0..batch_size {
        for j in 0..5 {
            // Sample from approximate standard normal
            input[i * 5 + j] = (i as f32) / (batch_size as f32) * 6.0 - 3.0;
        }
    }

    let mut output = vec![0.0f32; 5 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    // With gamma=1, beta=0, output is just normalized input
    // Check that output has zero mean and unit variance
    for j in 0..5 {
        let mut mean = 0.0f32;
        let mut variance = 0.0f32;

        for i in 0..batch_size {
            mean += output[i * 5 + j];
        }
        mean /= batch_size as f32;

        for i in 0..batch_size {
            let diff = output[i * 5 + j] - mean;
            variance += diff * diff;
        }
        variance /= batch_size as f32;

        assert!(mean.abs() < 1e-5, "Mean should be ~0");
        assert!((variance - 1.0).abs() < 1e-4, "Variance should be ~1");
    }
}

// ============================================================================
// Batch Size Tests
// ============================================================================

#[test]
fn test_batchnorm_single_sample() {
    // Test batch norm with batch_size=1 (edge case)
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(true);

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];

    // With batch_size=1, variance is 0, so output depends on epsilon
    layer.forward(&input, &mut output, 1);

    // All outputs should be finite
    for &val in &output {
        assert!(
            val.is_finite(),
            "Output should be finite even with batch_size=1"
        );
    }
}

#[test]
fn test_batchnorm_varying_batch_sizes() {
    // Test that batch norm works with different batch sizes
    let layer = BatchNormLayer::new(8, 1e-5, 0.9);

    let batch_sizes = [1, 2, 4, 8, 16, 32];

    for &batch_size in &batch_sizes {
        let input = vec![1.0f32; 8 * batch_size];
        let mut output = vec![0.0f32; 8 * batch_size];

        layer.forward(&input, &mut output, batch_size);

        assert_eq!(output.len(), 8 * batch_size);
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
fn test_batchnorm_different_epsilon_values() {
    // Test batch norm with different epsilon values
    let epsilons = [1e-3, 1e-5, 1e-7, 1e-10];

    for &eps in &epsilons {
        let layer = BatchNormLayer::new(6, eps, 0.9);
        assert_eq!(layer.epsilon(), eps);

        let batch_size = 10;
        let input = vec![1.0f32; 6 * batch_size];
        let mut output = vec![0.0f32; 6 * batch_size];

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
fn test_batchnorm_epsilon_prevents_division_by_zero() {
    // Test that epsilon prevents division by zero with constant input
    let layer = BatchNormLayer::new(3, 1e-5, 0.9);

    let batch_size = 10;
    let input = vec![5.0f32; 3 * batch_size]; // Constant values (zero variance)
    let mut output = vec![0.0f32; 3 * batch_size];

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
fn test_batchnorm_with_dense_layer() {
    // Test batch norm integrated with dense layer
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(10, 8, &mut rng);
    let mut batchnorm = BatchNormLayer::new(8, 1e-5, 0.9);
    batchnorm.set_training(true);

    let batch_size = 4;
    let input = vec![0.5f32; 10 * batch_size];
    let mut dense_output = vec![0.0f32; 8 * batch_size];
    let mut bn_output = vec![0.0f32; 8 * batch_size];

    // Forward through dense then batch norm
    dense.forward(&input, &mut dense_output, batch_size);
    batchnorm.forward(&dense_output, &mut bn_output, batch_size);

    // Verify dimensions
    assert_eq!(bn_output.len(), 8 * batch_size);

    // Verify batch norm normalized the dense output
    for j in 0..8 {
        let mut mean = 0.0f32;
        for i in 0..batch_size {
            mean += bn_output[i * 8 + j];
        }
        mean /= batch_size as f32;

        assert!(
            mean.abs() < 1e-5,
            "Batch norm should normalize dense layer output"
        );
    }
}

#[test]
fn test_batchnorm_backward_with_dense_layer() {
    // Test gradient flow through batch norm + dense layer
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(6, 4, &mut rng);
    let mut batchnorm = BatchNormLayer::new(4, 1e-5, 0.9);
    batchnorm.set_training(true);

    let batch_size = 8;
    let input = vec![0.5f32; 6 * batch_size];
    let mut dense_output = vec![0.0f32; 4 * batch_size];
    let mut bn_output = vec![0.0f32; 4 * batch_size];

    // Forward pass
    dense.forward(&input, &mut dense_output, batch_size);
    batchnorm.forward(&dense_output, &mut bn_output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; 4 * batch_size];
    let mut grad_bn_input = vec![0.0f32; 4 * batch_size];
    let mut grad_dense_input = vec![0.0f32; 6 * batch_size];

    batchnorm.backward(&dense_output, &grad_output, &mut grad_bn_input, batch_size);
    dense.backward(&input, &grad_bn_input, &mut grad_dense_input, batch_size);

    // All gradients should be finite
    for &grad in &grad_dense_input {
        assert!(grad.is_finite(), "Gradient should flow through both layers");
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_batchnorm_large_values() {
    // Test batch norm with large input values
    let layer = BatchNormLayer::new(4, 1e-5, 0.9);

    let batch_size = 10;
    let input: Vec<f32> = (0..4 * batch_size).map(|i| (i as f32) * 1000.0).collect();
    let mut output = vec![0.0f32; 4 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite despite large inputs
    for &val in &output {
        assert!(val.is_finite(), "Should handle large values");
    }
}

#[test]
fn test_batchnorm_small_values() {
    // Test batch norm with small input values
    let layer = BatchNormLayer::new(4, 1e-5, 0.9);

    let batch_size = 10;
    let input: Vec<f32> = (0..4 * batch_size).map(|i| (i as f32) * 1e-6).collect();
    let mut output = vec![0.0f32; 4 * batch_size];

    layer.forward(&input, &mut output, batch_size);

    // All outputs should be finite despite small inputs
    for &val in &output {
        assert!(val.is_finite(), "Should handle small values");
    }
}

#[test]
fn test_batchnorm_mixed_positive_negative() {
    // Test batch norm with mixed positive and negative values
    let layer = BatchNormLayer::new(5, 1e-5, 0.9);

    let batch_size = 10;
    let mut input = vec![0.0f32; 5 * batch_size];
    for i in 0..batch_size {
        for j in 0..5 {
            input[i * 5 + j] = if (i + j) % 2 == 0 {
                (i * j) as f32
            } else {
                -((i * j) as f32)
            };
        }
    }

    let mut output = vec![0.0f32; 5 * batch_size];
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
fn test_batchnorm_parameter_update_clears_gradients() {
    // Test that update_parameters clears accumulated gradients
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 8;
    // Create input with variation across batch
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0, 5.0,
        10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0, 7.0, 14.0, 21.0, 28.0, 8.0, 16.0, 24.0, 32.0,
    ];
    let mut output = vec![0.0f32; 32]; // 4 features * 8 batch_size

    // First iteration: forward + backward + update
    layer.forward(&input, &mut output, batch_size);
    // Use non-uniform gradients to create non-zero gamma gradients
    let grad_output: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let mut grad_input = vec![0.0f32; 32]; // 4 features * 8 batch_size
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
// Multiple Training Iterations
// ============================================================================

#[test]
fn test_batchnorm_multiple_forward_passes_update_running_stats() {
    // Test that multiple forward passes continue updating running statistics
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 10;

    // First batch
    let input1 = [1.0, 2.0, 3.0].repeat(batch_size);
    let mut output1 = vec![0.0f32; 3 * batch_size];
    layer.forward(&input1, &mut output1, batch_size);
    let running_mean_1 = layer.running_mean();

    // Second batch (different values)
    let input2 = [2.0, 4.0, 6.0].repeat(batch_size);
    let mut output2 = vec![0.0f32; 3 * batch_size];
    layer.forward(&input2, &mut output2, batch_size);
    let running_mean_2 = layer.running_mean();

    // Running statistics should continue to update
    for i in 0..3 {
        assert_ne!(
            running_mean_1[i], running_mean_2[i],
            "Running mean should update on each forward pass"
        );
    }
}

#[test]
fn test_batchnorm_convergence_of_running_statistics() {
    // Test that running statistics converge with repeated identical batches
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
    layer.set_training(true);

    let batch_size = 10;
    let input = [3.0, 7.0].repeat(batch_size);
    let mut output = vec![0.0f32; 2 * batch_size];

    // Run many forward passes with same input
    for _ in 0..100 {
        layer.forward(&input, &mut output, batch_size);
    }

    let running_mean = layer.running_mean();

    // Running mean should converge close to batch mean (3.0, 7.0)
    assert!(
        (running_mean[0] - 3.0).abs() < 0.1,
        "Running mean should converge to batch mean"
    );
    assert!(
        (running_mean[1] - 7.0).abs() < 0.1,
        "Running mean should converge to batch mean"
    );
}
