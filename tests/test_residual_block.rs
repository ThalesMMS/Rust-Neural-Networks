// Integration tests for ResidualBlock layer.
// Tests identity shortcut, projection shortcut, forward/backward correctness,
// training mode propagation, and parameter updates.

use rust_neural_networks::layers::{residual::ResidualBlock, Layer};
use rust_neural_networks::optimizers::{Adam, SGD};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Identity Shortcut Tests
// ============================================================================

#[test]
fn test_identity_shortcut_dimensions() {
    // Same channels, stride=1 → identity shortcut
    let mut rng = SimpleRng::new(42);
    let block = ResidualBlock::new(8, 8, 1, 4, 4, &mut rng);

    assert_eq!(block.in_channels(), 8);
    assert_eq!(block.out_channels(), 8);
    assert_eq!(block.out_height(), 4);
    assert_eq!(block.out_width(), 4);
    assert_eq!(block.input_size(), 8 * 4 * 4);
    assert_eq!(block.output_size(), 8 * 4 * 4);
    assert!(!block.has_projection_shortcut());
}

#[test]
fn test_identity_shortcut_parameter_count_positive() {
    // Parameter count must be positive and contain at least conv1+bn1+conv2+bn2 params
    let mut rng = SimpleRng::new(1);
    let block = ResidualBlock::new(4, 4, 1, 6, 6, &mut rng);

    assert!(!block.has_projection_shortcut());
    assert!(
        block.parameter_count() > 0,
        "Identity block parameter count should be > 0"
    );
}

#[test]
fn test_identity_shortcut_forward_shape() {
    // Output must have length = batch_size * output_size()
    let mut rng = SimpleRng::new(10);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 3;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), batch_size * block.output_size());
}

#[test]
fn test_identity_shortcut_forward_nonnegative() {
    // Final ReLU guarantees all outputs ≥ 0
    let mut rng = SimpleRng::new(20);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| i as f32 * 0.07 - 0.5)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert!(
        output.iter().all(|&v| v >= 0.0),
        "All outputs should be >= 0 after final ReLU"
    );
}

#[test]
fn test_identity_shortcut_forward_finite() {
    // All output values must be finite
    let mut rng = SimpleRng::new(30);
    let mut block = ResidualBlock::new(4, 4, 1, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert!(
        output.iter().all(|&v| v.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
fn test_identity_shortcut_backward_grad_shape() {
    // grad_input length must equal batch_size * input_size()
    let mut rng = SimpleRng::new(40);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert_eq!(grad_input.len(), batch_size * block.input_size());
}

#[test]
fn test_identity_shortcut_backward_nonzero_gradients() {
    // At least some gradients should be non-zero
    let mut rng = SimpleRng::new(50);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(
        grad_input.iter().any(|&v| v.abs() > 1e-10),
        "Identity shortcut backward: expected non-zero grad_input"
    );
}

#[test]
fn test_identity_shortcut_backward_finite_gradients() {
    // All gradients must be finite
    let mut rng = SimpleRng::new(60);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(
        grad_input.iter().all(|&v| v.is_finite()),
        "All grad_input values should be finite"
    );
}

// ============================================================================
// Projection Shortcut Tests (different channels)
// ============================================================================

#[test]
fn test_projection_shortcut_different_channels_dimensions() {
    // in_channels != out_channels triggers projection shortcut
    let mut rng = SimpleRng::new(100);
    let block = ResidualBlock::new(4, 8, 1, 8, 8, &mut rng);

    assert_eq!(block.in_channels(), 4);
    assert_eq!(block.out_channels(), 8);
    assert_eq!(block.out_height(), 8);
    assert_eq!(block.out_width(), 8);
    assert_eq!(block.input_size(), 4 * 8 * 8);
    assert_eq!(block.output_size(), 8 * 8 * 8);
    assert!(block.has_projection_shortcut());
}

#[test]
fn test_projection_shortcut_stride2_dimensions() {
    // stride=2 halves spatial dimensions and triggers projection shortcut
    let mut rng = SimpleRng::new(110);
    let block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);

    // out_dim = floor((8 + 2 - 3) / 2) + 1 = floor(7 / 2) + 1 = 3 + 1 = 4
    assert_eq!(block.out_height(), 4);
    assert_eq!(block.out_width(), 4);
    assert_eq!(block.input_size(), 4 * 8 * 8);
    assert_eq!(block.output_size(), 8 * 4 * 4);
    assert!(block.has_projection_shortcut());
}

#[test]
fn test_projection_shortcut_stride2_same_channels() {
    // stride=2 alone (with same channels) still requires projection shortcut
    let mut rng = SimpleRng::new(120);
    let block = ResidualBlock::new(8, 8, 2, 8, 8, &mut rng);

    assert!(block.has_projection_shortcut());
    assert_eq!(block.out_height(), 4);
    assert_eq!(block.out_width(), 4);
}

#[test]
fn test_projection_shortcut_parameter_count_larger_than_identity() {
    // Projection shortcut adds extra parameters (shortcut_conv + shortcut_bn)
    let mut rng_id = SimpleRng::new(130);
    let block_id = ResidualBlock::new(4, 4, 1, 8, 8, &mut rng_id);

    let mut rng_proj = SimpleRng::new(130);
    let block_proj = ResidualBlock::new(4, 8, 1, 8, 8, &mut rng_proj);

    assert!(
        block_proj.parameter_count() > block_id.parameter_count(),
        "Projection block should have more params than identity block"
    );
}

#[test]
fn test_projection_shortcut_forward_shape() {
    // Output must have correct length for projected output dimensions
    let mut rng = SimpleRng::new(140);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), batch_size * block.output_size());
}

#[test]
fn test_projection_shortcut_forward_nonnegative() {
    // Final ReLU guarantees all outputs ≥ 0 for projection shortcut too
    let mut rng = SimpleRng::new(150);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| i as f32 * 0.05 - 0.3)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert!(
        output.iter().all(|&v| v >= 0.0),
        "Projection shortcut forward: all outputs should be >= 0"
    );
}

#[test]
fn test_projection_shortcut_forward_finite() {
    // All outputs must be finite
    let mut rng = SimpleRng::new(160);
    let mut block = ResidualBlock::new(4, 8, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert!(
        output.iter().all(|&v| v.is_finite()),
        "Projection shortcut forward: all outputs should be finite"
    );
}

#[test]
fn test_projection_shortcut_backward_grad_shape() {
    // grad_input shape must match batch_size * input_size()
    let mut rng = SimpleRng::new(170);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert_eq!(grad_input.len(), batch_size * block.input_size());
}

#[test]
fn test_projection_shortcut_backward_nonzero_gradients() {
    // Gradients must not be all zeros
    let mut rng = SimpleRng::new(180);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(
        grad_input.iter().any(|&v| v.abs() > 1e-10),
        "Projection shortcut backward: expected non-zero grad_input"
    );
}

#[test]
fn test_projection_shortcut_backward_finite_gradients() {
    // All gradients must be finite
    let mut rng = SimpleRng::new(190);
    let mut block = ResidualBlock::new(4, 8, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.05)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(
        grad_input.iter().all(|&v| v.is_finite()),
        "All grad_input values should be finite"
    );
}

// ============================================================================
// Training Mode Tests
// ============================================================================

#[test]
fn test_training_mode_default_is_training() {
    // BatchNorm layers default to training mode
    let mut rng = SimpleRng::new(200);
    let block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);

    assert!(block.is_training(), "Block should default to training mode");
}

#[test]
fn test_training_mode_toggle() {
    // set_training() must toggle the mode correctly
    let mut rng = SimpleRng::new(210);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);

    block.set_training(true);
    assert!(block.is_training(), "Block should be in training mode");

    block.set_training(false);
    assert!(!block.is_training(), "Block should be in inference mode");

    block.set_training(true);
    assert!(block.is_training(), "Block should be back in training mode");
}

#[test]
fn test_training_mode_propagates_to_projection_shortcut() {
    // set_training() must propagate to shortcut_bn layers too
    let mut rng = SimpleRng::new(220);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);

    // Initially training
    assert!(block.is_training());

    block.set_training(false);
    assert!(!block.is_training());

    block.set_training(true);
    assert!(block.is_training());
}

#[test]
fn test_training_vs_inference_output_differs() {
    // After some training forward passes, training mode and inference mode
    // should produce different outputs (because running stats differ from batch stats)
    let mut rng = SimpleRng::new(230);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);

    let batch_size = 8;
    let in_size = block.input_size();
    let out_size = block.output_size();

    // Run several training forward passes to populate running statistics
    block.set_training(true);
    for _ in 0..5 {
        let input: Vec<f32> = (0..batch_size * in_size)
            .map(|i| i as f32 * 0.07 + 0.1)
            .collect();
        let mut output = vec![0.0f32; batch_size * out_size];
        block.forward(&input, &mut output, batch_size);
    }

    // Compare training vs inference on the same batch
    let input: Vec<f32> = (0..batch_size * in_size)
        .map(|i| i as f32 * 0.07 + 0.1)
        .collect();

    let mut output_train = vec![0.0f32; batch_size * out_size];
    block.set_training(true);
    block.forward(&input, &mut output_train, batch_size);

    let mut output_infer = vec![0.0f32; batch_size * out_size];
    block.set_training(false);
    block.forward(&input, &mut output_infer, batch_size);

    // Both outputs should be finite
    assert!(output_train.iter().all(|&v| v.is_finite()));
    assert!(output_infer.iter().all(|&v| v.is_finite()));

    // Training and inference outputs should differ
    let differs = output_train
        .iter()
        .zip(output_infer.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "Training and inference outputs should differ after running stats are populated"
    );
}

// ============================================================================
// Forward / Backward Correctness Tests
// ============================================================================

#[test]
fn test_forward_batch_size_one() {
    // Single-sample batch must work correctly
    let mut rng = SimpleRng::new(300);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 1;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), block.output_size());
    assert!(output.iter().all(|&v| v >= 0.0));
    assert!(output.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_forward_larger_batch() {
    // Larger batches should work correctly
    let mut rng = SimpleRng::new(310);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 16;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), batch_size * block.output_size());
    assert!(output.iter().all(|&v| v >= 0.0));
    assert!(output.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_forward_deterministic_same_input() {
    // Given the same input, forward pass should produce the same output
    let mut rng = SimpleRng::new(320);
    let block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    let mut output1 = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output1, batch_size);

    let mut output2 = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output2, batch_size);

    for (i, (&a, &b)) in output1.iter().zip(output2.iter()).enumerate() {
        assert_eq!(a, b, "Output mismatch at index {}", i);
    }
}

#[test]
fn test_backward_after_forward() {
    // backward() must be callable after forward() without panic
    let mut rng = SimpleRng::new(330);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.5f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];

    // Must not panic
    block.backward(&input, &grad_output, &mut grad_input, batch_size);
}

#[test]
fn test_backward_zero_grad_output_produces_zero_grad_input() {
    // When upstream gradient is zero, grad_input should be zero
    let mut rng = SimpleRng::new(340);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(
        grad_input.iter().all(|&v| v == 0.0),
        "Zero upstream gradient should produce zero grad_input"
    );
}

#[test]
fn test_backward_projection_larger_grad_output() {
    // Varying the grad_output scale should scale grad_input proportionally (approx)
    let mut rng1 = SimpleRng::new(350);
    let mut block1 = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng1);
    block1.set_training(true);

    let mut rng2 = SimpleRng::new(350);
    let mut block2 = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng2);
    block2.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block1.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    // Forward both blocks with the same input
    let mut output = vec![0.0f32; batch_size * block1.output_size()];
    block1.forward(&input, &mut output, batch_size);
    block2.forward(&input, &mut output, batch_size);

    // Scale factor for grad_output
    let scale = 2.0f32;
    let grad_output1 = vec![1.0f32; batch_size * block1.output_size()];
    let grad_output2: Vec<f32> = grad_output1.iter().map(|&g| g * scale).collect();

    let mut grad_input1 = vec![0.0f32; batch_size * block1.input_size()];
    let mut grad_input2 = vec![0.0f32; batch_size * block2.input_size()];

    block1.backward(&input, &grad_output1, &mut grad_input1, batch_size);
    block2.backward(&input, &grad_output2, &mut grad_input2, batch_size);

    // grad_input2 should be approximately scale * grad_input1
    let max_relative_error = grad_input1
        .iter()
        .zip(grad_input2.iter())
        .map(|(&g1, &g2)| {
            let expected = g1 * scale;
            if expected.abs() > 1e-10 {
                (g2 - expected).abs() / expected.abs()
            } else {
                (g2 - expected).abs()
            }
        })
        .fold(0.0f32, f32::max);

    assert!(
        max_relative_error < 0.05,
        "Backward pass should scale linearly with grad_output (max relative error = {})",
        max_relative_error
    );
}

// ============================================================================
// Parameter Update Tests
// ============================================================================

#[test]
fn test_update_parameters_does_not_panic() {
    // update_parameters() should not panic
    let mut rng = SimpleRng::new(400);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.01f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Must not panic
    block.update_parameters(0.01);
}

#[test]
fn test_update_parameters_projection_does_not_panic() {
    // update_parameters() with projection shortcut should not panic
    let mut rng = SimpleRng::new(410);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.01f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    block.update_parameters(0.01);
}

#[test]
fn test_update_with_sgd_optimizer() {
    // update_with_optimizer() with SGD should not panic
    let mut rng = SimpleRng::new(420);
    let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.01f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    let mut optimizer = SGD::new(0.01);
    block.update_with_optimizer(&mut optimizer);
}

#[test]
fn test_update_with_adam_optimizer() {
    // update_with_optimizer() with Adam should not panic
    let mut rng = SimpleRng::new(430);
    let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block.set_training(true);

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.01f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    block.update_with_optimizer(&mut optimizer);
}

// ============================================================================
// Layer Trait Integration Tests
// ============================================================================

#[test]
fn test_layer_trait_input_output_size_identity() {
    // Layer trait input_size() / output_size() via trait object
    let mut rng = SimpleRng::new(500);
    let block: Box<dyn Layer> = Box::new(ResidualBlock::new(4, 4, 1, 6, 6, &mut rng));

    assert_eq!(block.input_size(), 4 * 6 * 6);
    assert_eq!(block.output_size(), 4 * 6 * 6);
}

#[test]
fn test_layer_trait_input_output_size_projection() {
    // Layer trait sizes are correct for projection shortcut
    let mut rng = SimpleRng::new(510);
    let block: Box<dyn Layer> = Box::new(ResidualBlock::new(4, 8, 2, 8, 8, &mut rng));

    assert_eq!(block.input_size(), 4 * 8 * 8);
    assert_eq!(block.output_size(), 8 * 4 * 4);
}

#[test]
fn test_layer_trait_parameter_count_positive() {
    // parameter_count() should be positive through trait object
    let mut rng = SimpleRng::new(520);
    let block: Box<dyn Layer> = Box::new(ResidualBlock::new(4, 4, 1, 4, 4, &mut rng));

    assert!(
        block.parameter_count() > 0,
        "parameter_count() should be positive"
    );
}

#[test]
fn test_layer_trait_forward_backward_through_trait() {
    // Test that forward and backward work correctly through the Layer trait object
    let mut rng = SimpleRng::new(530);
    let block: Box<dyn Layer> = Box::new(ResidualBlock::new(4, 4, 1, 4, 4, &mut rng));

    let batch_size = 2;
    let input: Vec<f32> = (0..batch_size * block.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut output = vec![0.0f32; batch_size * block.output_size()];
    block.forward(&input, &mut output, batch_size);

    assert_eq!(output.len(), batch_size * block.output_size());
    assert!(output.iter().all(|&v| v >= 0.0));

    let grad_output = vec![1.0f32; batch_size * block.output_size()];
    let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert_eq!(grad_input.len(), batch_size * block.input_size());
    assert!(
        grad_input.iter().any(|&v| v.abs() > 1e-10),
        "Expected non-zero gradients through Layer trait"
    );
}

// ============================================================================
// Combined smoke test (mirrors the verification command expectation)
// ============================================================================

#[test]
fn test_residual_block_smoke() {
    // Comprehensive test: identity block and projection block, forward + backward
    let mut rng = SimpleRng::new(999);

    // --- Identity block ---
    let mut block_id = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
    block_id.set_training(true);

    assert_eq!(block_id.input_size(), 4 * 4 * 4);
    assert_eq!(block_id.output_size(), 4 * 4 * 4);
    assert!(!block_id.has_projection_shortcut());
    assert!(block_id.parameter_count() > 0);

    let batch = 2;
    let input_id: Vec<f32> = (0..batch * block_id.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut out_id = vec![0.0f32; batch * block_id.output_size()];
    block_id.forward(&input_id, &mut out_id, batch);

    assert_eq!(out_id.len(), batch * block_id.output_size());
    assert!(out_id.iter().all(|&v| v >= 0.0));
    assert!(out_id.iter().all(|&v| v.is_finite()));

    let grad_out_id = vec![1.0f32; batch * block_id.output_size()];
    let mut grad_in_id = vec![0.0f32; batch * block_id.input_size()];
    block_id.backward(&input_id, &grad_out_id, &mut grad_in_id, batch);

    assert_eq!(grad_in_id.len(), batch * block_id.input_size());
    assert!(
        grad_in_id.iter().any(|&v| v.abs() > 1e-10),
        "Identity block: expected non-zero grad_input"
    );
    assert!(grad_in_id.iter().all(|&v| v.is_finite()));

    // --- Projection block (stride=2, different channels) ---
    let mut block_proj = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    block_proj.set_training(true);

    assert_eq!(block_proj.input_size(), 4 * 8 * 8);
    assert_eq!(block_proj.output_size(), 8 * 4 * 4);
    assert!(block_proj.has_projection_shortcut());
    assert!(block_proj.parameter_count() > 0);

    let input_proj: Vec<f32> = (0..batch * block_proj.input_size())
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut out_proj = vec![0.0f32; batch * block_proj.output_size()];
    block_proj.forward(&input_proj, &mut out_proj, batch);

    assert_eq!(out_proj.len(), batch * block_proj.output_size());
    assert!(out_proj.iter().all(|&v| v >= 0.0));
    assert!(out_proj.iter().all(|&v| v.is_finite()));

    let grad_out_proj = vec![1.0f32; batch * block_proj.output_size()];
    let mut grad_in_proj = vec![0.0f32; batch * block_proj.input_size()];
    block_proj.backward(&input_proj, &grad_out_proj, &mut grad_in_proj, batch);

    assert_eq!(grad_in_proj.len(), batch * block_proj.input_size());
    assert!(
        grad_in_proj.iter().any(|&v| v.abs() > 1e-10),
        "Projection block: expected non-zero grad_input"
    );
    assert!(grad_in_proj.iter().all(|&v| v.is_finite()));

    // --- Training mode toggle ---
    block_id.set_training(false);
    assert!(!block_id.is_training());
    block_id.set_training(true);
    assert!(block_id.is_training());
}
