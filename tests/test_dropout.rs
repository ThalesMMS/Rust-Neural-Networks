// Integration tests for dropout layer.
// Tests dropout behavior in isolation and integration with other layers.

use rust_neural_networks::layers::{DenseLayer, DropoutLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Statistical Dropout Rate Verification
// ============================================================================

#[test]
fn test_dropout_rate_statistical_verification() {
    // Test that the actual dropout rate matches the configured rate
    // over a large number of samples
    let mut rng = SimpleRng::new(42);
    let drop_rate = 0.5;
    let size = 1000;
    let batch_size = 100; // Test with 100 samples

    let mut layer = DropoutLayer::new(size, drop_rate, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; size * batch_size];
    let mut output = vec![0.0f32; size * batch_size];

    // Run forward pass
    layer.forward(&input, &mut output, batch_size);

    // Count dropped units (zeros)
    let dropped_count = output.iter().filter(|&&x| x == 0.0).count();
    let total_count = output.len();
    let actual_drop_rate = dropped_count as f32 / total_count as f32;

    // Allow 5% tolerance for statistical variation
    let tolerance = 0.05;
    assert!(
        (actual_drop_rate - drop_rate).abs() < tolerance,
        "Expected drop rate ~{}, got {} (dropped {}/{})",
        drop_rate,
        actual_drop_rate,
        dropped_count,
        total_count
    );
}

#[test]
fn test_dropout_different_rates() {
    // Test various dropout rates to ensure they all work correctly
    let rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9];
    let size = 1000;

    for &rate in &rates {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(size, rate, &mut rng);
        layer.set_training(true);

        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];

        layer.forward(&input, &mut output, 1);

        let dropped_count = output.iter().filter(|&&x| x == 0.0).count();
        let actual_rate = dropped_count as f32 / size as f32;

        // Allow 10% tolerance for smaller sample sizes
        let tolerance = 0.1;
        assert!(
            (actual_rate - rate).abs() < tolerance,
            "Rate {}: expected ~{}, got {}",
            rate,
            rate,
            actual_rate
        );
    }
}

#[test]
fn test_dropout_inference_mode_no_dropout() {
    // Test that inference mode passes values through unchanged
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(100, 0.5, &mut rng);
    layer.set_training(false);

    let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 100];

    layer.forward(&input, &mut output, 1);

    // In inference mode, all values should pass through unchanged
    for i in 0..100 {
        assert_eq!(
            output[i], input[i],
            "Inference mode should not modify values"
        );
    }
}

#[test]
fn test_dropout_training_mode_applies_dropout() {
    // Test that training mode actually applies dropout
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(100, 0.5, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 100];

    layer.forward(&input, &mut output, 1);

    // In training mode, some values should be zero (dropped)
    let dropped_count = output.iter().filter(|&&x| x == 0.0).count();
    let kept_count = output.iter().filter(|&&x| x != 0.0).count();

    assert!(dropped_count > 0, "Training mode should drop some units");
    assert!(kept_count > 0, "Training mode should keep some units");
}

#[test]
fn test_dropout_mode_switching() {
    // Test switching between training and inference modes
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(100, 0.5, &mut rng);

    let input = vec![1.0f32; 100];
    let mut output_train = vec![0.0f32; 100];
    let mut output_inference = vec![0.0f32; 100];

    // Test training mode
    layer.set_training(true);
    layer.forward(&input, &mut output_train, 1);
    let dropped_train = output_train.iter().filter(|&&x| x == 0.0).count();

    // Switch to inference mode
    layer.set_training(false);
    layer.forward(&input, &mut output_inference, 1);
    let dropped_inference = output_inference.iter().filter(|&&x| x == 0.0).count();

    assert!(dropped_train > 0, "Training mode should drop units");
    assert_eq!(dropped_inference, 0, "Inference mode should not drop units");
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

#[test]
fn test_dropout_gradient_flow_training_mode() {
    // Test that gradients flow correctly through dropout in training mode
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(10, 0.5, &mut rng);
    layer.set_training(true);

    // Forward pass
    let input = vec![1.0f32; 10];
    let mut output = vec![0.0f32; 10];
    layer.forward(&input, &mut output, 1);

    // Backward pass with unit gradients
    let grad_output = vec![1.0f32; 10];
    let mut grad_input = vec![0.0f32; 10];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Gradient should be zero where output was zero (dropped)
    // and scaled where output was non-zero (kept)
    let scale = 1.0 / (1.0 - 0.5);
    for i in 0..10 {
        if output[i] == 0.0 {
            assert_eq!(
                grad_input[i], 0.0,
                "Gradient should be zero for dropped units"
            );
        } else {
            assert!(
                (grad_input[i] - scale).abs() < 1e-6,
                "Gradient should be scaled for kept units"
            );
        }
    }
}

#[test]
fn test_dropout_gradient_flow_inference_mode() {
    // Test that gradients pass through unchanged in inference mode
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(10, 0.5, &mut rng);
    layer.set_training(false);

    let input = vec![1.0f32; 10];
    let mut output = vec![0.0f32; 10];
    layer.forward(&input, &mut output, 1);

    let grad_output: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let mut grad_input = vec![0.0f32; 10];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // In inference mode, gradient should pass through unchanged
    for i in 0..10 {
        assert_eq!(
            grad_input[i], grad_output[i],
            "Inference mode should not modify gradients"
        );
    }
}

#[test]
fn test_dropout_gradient_mask_consistency() {
    // Test that backward pass uses the same mask as forward pass
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(100, 0.5, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 100];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0f32; 100];
    let mut grad_input = vec![0.0f32; 100];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Every dropped unit in forward should have zero gradient in backward
    for i in 0..100 {
        if output[i] == 0.0 {
            assert_eq!(grad_input[i], 0.0, "Dropped unit should have zero gradient");
        } else {
            assert_ne!(
                grad_input[i], 0.0,
                "Kept unit should have non-zero gradient"
            );
        }
    }
}

#[test]
fn test_dropout_batch_gradient_flow() {
    // Test gradient flow with multiple samples in a batch
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(5, 0.5, &mut rng);
    layer.set_training(true);

    let batch_size = 3;
    let input = vec![1.0f32; 5 * batch_size];
    let mut output = vec![0.0f32; 5 * batch_size];
    layer.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; 5 * batch_size];
    let mut grad_input = vec![0.0f32; 5 * batch_size];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradient consistency for each sample
    for i in 0..(5 * batch_size) {
        if output[i] == 0.0 {
            assert_eq!(grad_input[i], 0.0);
        } else {
            assert_ne!(grad_input[i], 0.0);
        }
    }
}

// ============================================================================
// Integration with DenseLayer
// ============================================================================

#[test]
fn test_dropout_with_dense_layer_forward() {
    // Test dropout layer working with a dense layer in forward pass
    let mut rng = SimpleRng::new(42);

    // Create a small network: Dense(10 -> 5) -> Dropout(0.5)
    let dense = DenseLayer::new(10, 5, &mut rng);
    let mut dropout = DropoutLayer::new(5, 0.5, &mut rng);
    dropout.set_training(true);

    let input = vec![1.0f32; 10];
    let mut dense_output = vec![0.0f32; 5];
    let mut dropout_output = vec![0.0f32; 5];

    // Forward through dense then dropout
    dense.forward(&input, &mut dense_output, 1);
    dropout.forward(&dense_output, &mut dropout_output, 1);

    // Check that dropout output has some zeros
    let dropped_count = dropout_output.iter().filter(|&&x| x == 0.0).count();
    assert!(dropped_count > 0, "Dropout should drop some activations");

    // Check that kept values are scaled
    for i in 0..5 {
        if dropout_output[i] != 0.0 {
            assert!(
                dropout_output[i].abs() > dense_output[i].abs(),
                "Kept values should be scaled up"
            );
        }
    }
}

#[test]
fn test_dropout_with_dense_layer_backward() {
    // Test dropout layer working with a dense layer in backward pass
    let mut rng = SimpleRng::new(42);

    // Create a small network: Dense(10 -> 5) -> Dropout(0.5)
    let dense = DenseLayer::new(10, 5, &mut rng);
    let mut dropout = DropoutLayer::new(5, 0.5, &mut rng);
    dropout.set_training(true);

    let input = vec![1.0f32; 10];
    let mut dense_output = vec![0.0f32; 5];
    let mut dropout_output = vec![0.0f32; 5];

    // Forward pass
    dense.forward(&input, &mut dense_output, 1);
    dropout.forward(&dense_output, &mut dropout_output, 1);

    // Backward pass
    let grad_loss = vec![1.0f32; 5];
    let mut grad_dropout = vec![0.0f32; 5];
    let mut grad_dense = vec![0.0f32; 10];

    dropout.backward(&dense_output, &grad_loss, &mut grad_dropout, 1);
    dense.backward(&input, &grad_dropout, &mut grad_dense, 1);

    // Check that gradients are finite
    assert!(grad_dropout.iter().all(|&x| x.is_finite()));
    assert!(grad_dense.iter().all(|&x| x.is_finite()));

    // Check that gradient masking works
    for i in 0..5 {
        if dropout_output[i] == 0.0 {
            assert_eq!(
                grad_dropout[i], 0.0,
                "Dropped unit should have zero gradient"
            );
        }
    }
}

#[test]
fn test_dropout_with_dense_layer_inference() {
    // Test that inference mode works correctly in a network
    let mut rng = SimpleRng::new(42);

    let dense = DenseLayer::new(10, 5, &mut rng);
    let mut dropout = DropoutLayer::new(5, 0.5, &mut rng);
    dropout.set_training(false);

    let input = vec![1.0f32; 10];
    let mut dense_output = vec![0.0f32; 5];
    let mut dropout_output = vec![0.0f32; 5];

    // Forward pass in inference mode
    dense.forward(&input, &mut dense_output, 1);
    dropout.forward(&dense_output, &mut dropout_output, 1);

    // In inference mode, dropout output should equal dense output
    for i in 0..5 {
        assert_eq!(
            dropout_output[i], dense_output[i],
            "Inference mode should not modify activations"
        );
    }
}

#[test]
fn test_dropout_between_dense_layers() {
    // Test dropout sandwiched between two dense layers
    let mut rng = SimpleRng::new(42);

    // Network: Dense(10 -> 8) -> Dropout(0.5) -> Dense(8 -> 5)
    let dense1 = DenseLayer::new(10, 8, &mut rng);
    let mut dropout = DropoutLayer::new(8, 0.5, &mut rng);
    dropout.set_training(true);
    let dense2 = DenseLayer::new(8, 5, &mut rng);

    let input = vec![1.0f32; 10];
    let mut output1 = vec![0.0f32; 8];
    let mut dropout_output = vec![0.0f32; 8];
    let mut output2 = vec![0.0f32; 5];

    // Forward pass
    dense1.forward(&input, &mut output1, 1);
    dropout.forward(&output1, &mut dropout_output, 1);
    dense2.forward(&dropout_output, &mut output2, 1);

    // Check outputs are finite
    assert!(output1.iter().all(|&x| x.is_finite()));
    assert!(dropout_output.iter().all(|&x| x.is_finite()));
    assert!(output2.iter().all(|&x| x.is_finite()));

    // Check dropout is working
    let dropped_count = dropout_output.iter().filter(|&&x| x == 0.0).count();
    assert!(dropped_count > 0, "Dropout should drop some units");

    // Backward pass
    let grad_output = vec![1.0f32; 5];
    let mut grad_dropout = vec![0.0f32; 8];
    let mut grad_dense1 = vec![0.0f32; 8];
    let mut grad_input = vec![0.0f32; 10];

    dense2.backward(&dropout_output, &grad_output, &mut grad_dropout, 1);
    dropout.backward(&output1, &grad_dropout, &mut grad_dense1, 1);
    dense1.backward(&input, &grad_dense1, &mut grad_input, 1);

    // Check all gradients are finite
    assert!(grad_dropout.iter().all(|&x| x.is_finite()));
    assert!(grad_dense1.iter().all(|&x| x.is_finite()));
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

// ============================================================================
// Scaling and Expected Value Tests
// ============================================================================

#[test]
fn test_dropout_scaling_maintains_expected_value() {
    // Test that scaling maintains expected value over many samples
    let mut rng = SimpleRng::new(42);
    let drop_rate = 0.5;
    let size = 1000;

    let mut layer = DropoutLayer::new(size, drop_rate, &mut rng);
    layer.set_training(true);

    // Use constant input
    let input = vec![2.0f32; size];
    let mut output = vec![0.0f32; size];

    layer.forward(&input, &mut output, 1);

    // Calculate expected values
    let input_sum: f32 = input.iter().sum();
    let output_sum: f32 = output.iter().sum();

    // With scaling, output sum should be close to input sum
    // Allow 15% tolerance due to randomness
    let tolerance = input_sum * 0.15;
    assert!(
        (output_sum - input_sum).abs() < tolerance,
        "Expected sum ~{}, got {} (diff: {})",
        input_sum,
        output_sum,
        (output_sum - input_sum).abs()
    );
}

#[test]
fn test_dropout_scaling_factor() {
    // Test that scaling factor is correctly applied
    let mut rng = SimpleRng::new(42);
    let drop_rate = 0.5;
    let expected_scale = 1.0 / (1.0 - drop_rate);

    let mut layer = DropoutLayer::new(10, drop_rate, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; 10];
    let mut output = vec![0.0f32; 10];

    layer.forward(&input, &mut output, 1);

    // Check that non-zero outputs are scaled
    for &val in output.iter().filter(|&&x| x != 0.0) {
        assert!(
            (val - expected_scale).abs() < 1e-6,
            "Expected scale {}, got {}",
            expected_scale,
            val
        );
    }
}

#[test]
fn test_dropout_high_rate_scaling() {
    // Test scaling with high dropout rate (0.9)
    let mut rng = SimpleRng::new(42);
    let drop_rate = 0.9;
    let expected_scale = 1.0 / (1.0 - drop_rate); // = 10.0

    let mut layer = DropoutLayer::new(100, drop_rate, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; 100];
    let mut output = vec![0.0f32; 100];

    layer.forward(&input, &mut output, 1);

    // Check kept units are scaled by ~10
    for &val in output.iter().filter(|&&x| x != 0.0) {
        assert!(
            (val - expected_scale).abs() < 1e-5,
            "Expected scale {}, got {}",
            expected_scale,
            val
        );
    }

    // Check that approximately 90% are dropped
    let dropped_count = output.iter().filter(|&&x| x == 0.0).count();
    let actual_rate = dropped_count as f32 / 100.0;
    assert!(
        (actual_rate - drop_rate).abs() < 0.15,
        "Expected drop rate ~{}, got {}",
        drop_rate,
        actual_rate
    );
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

#[test]
fn test_dropout_zero_rate_no_dropout() {
    // Test that 0.0 drop rate means no dropout
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(100, 0.0, &mut rng);
    layer.set_training(true);

    let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 100];

    layer.forward(&input, &mut output, 1);

    // With 0.0 drop rate, all values should pass through unchanged
    for i in 0..100 {
        assert_eq!(output[i], input[i], "Zero drop rate should keep all values");
    }
}

#[test]
fn test_dropout_determinism_with_seed() {
    // Test that same seed produces same results
    let size = 50;
    let drop_rate = 0.5;

    let mut rng1 = SimpleRng::new(12345);
    let mut layer1 = DropoutLayer::new(size, drop_rate, &mut rng1);
    layer1.set_training(true);

    let mut rng2 = SimpleRng::new(12345);
    let mut layer2 = DropoutLayer::new(size, drop_rate, &mut rng2);
    layer2.set_training(true);

    let input = vec![1.0f32; size];
    let mut output1 = vec![0.0f32; size];
    let mut output2 = vec![0.0f32; size];

    layer1.forward(&input, &mut output1, 1);
    layer2.forward(&input, &mut output2, 1);

    // Same seed should produce identical outputs
    for i in 0..size {
        assert_eq!(
            output1[i], output2[i],
            "Same seed should produce identical results"
        );
    }
}

#[test]
fn test_dropout_different_seeds() {
    // Test that different seeds produce different results
    let size = 50;
    let drop_rate = 0.5;

    let mut rng1 = SimpleRng::new(12345);
    let mut layer1 = DropoutLayer::new(size, drop_rate, &mut rng1);
    layer1.set_training(true);

    let mut rng2 = SimpleRng::new(54321);
    let mut layer2 = DropoutLayer::new(size, drop_rate, &mut rng2);
    layer2.set_training(true);

    let input = vec![1.0f32; size];
    let mut output1 = vec![0.0f32; size];
    let mut output2 = vec![0.0f32; size];

    layer1.forward(&input, &mut output1, 1);
    layer2.forward(&input, &mut output2, 1);

    // Different seeds should (very likely) produce different outputs
    let differences = output1
        .iter()
        .zip(output2.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    assert!(
        differences > 0,
        "Different seeds should produce different results"
    );
}

#[test]
fn test_dropout_multiple_forward_passes() {
    // Test that multiple forward passes produce different masks
    let mut rng = SimpleRng::new(42);
    let mut layer = DropoutLayer::new(50, 0.5, &mut rng);
    layer.set_training(true);

    let input = vec![1.0f32; 50];
    let mut output1 = vec![0.0f32; 50];
    let mut output2 = vec![0.0f32; 50];

    // Two consecutive forward passes should use different masks
    layer.forward(&input, &mut output1, 1);
    layer.forward(&input, &mut output2, 1);

    // Check that masks are different (at least some positions differ)
    let differences = output1
        .iter()
        .zip(output2.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    assert!(
        differences > 0,
        "Multiple forward passes should generate different masks"
    );
}

#[test]
fn test_dropout_no_trainable_parameters() {
    // Verify that dropout has no trainable parameters
    let mut rng = SimpleRng::new(42);
    let layer = DropoutLayer::new(100, 0.5, &mut rng);

    assert_eq!(
        layer.parameter_count(),
        0,
        "Dropout should have no trainable parameters"
    );
    assert_eq!(
        layer.input_size(),
        layer.output_size(),
        "Dropout should not change dimensions"
    );
}
