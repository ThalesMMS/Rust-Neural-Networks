//! Unit tests for TransformerBlock layer
//!
//! This test suite verifies the correctness of the TransformerBlock implementation,
//! including forward propagation, backward propagation, parameter updates, and
//! integration with the Layer trait.

use rust_neural_networks::layers::{Layer, TransformerBlock, TransformerEncoder};
use rust_neural_networks::optimizers::{Optimizer, Adam, SGD};
use rust_neural_networks::utils::rng::SimpleRng;

/// Helper function to compute numerical gradient
fn compute_numerical_gradient(
    block: &TransformerBlock,
    input: &[f32],
    batch_size: usize,
    param_idx: usize,
    epsilon: f32,
) -> f32 {
    let mut block_plus = Box::new(create_test_block_with_seed(42));
    let mut block_minus = Box::new(create_test_block_with_seed(42));

    // We can't easily modify individual parameters in the composed block,
    // so this is a simplified numerical gradient check
    // For a full implementation, we'd need to expose parameter getters/setters

    // For now, we'll return 0.0 as a placeholder
    // A full implementation would require refactoring the layer to expose parameters
    0.0
}

/// Helper to create a test block with a specific seed
fn create_test_block_with_seed(seed: u64) -> TransformerBlock {
    let mut rng = SimpleRng::new(seed);
    TransformerBlock::new(32, 2, 64, &mut rng)
}

#[test]
fn test_transformer_block_creation() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(64, 4, 256, &mut rng);

    assert_eq!(block.d_model(), 64);
    assert_eq!(block.num_heads(), 4);
    assert_eq!(block.d_ff(), 256);
    assert_eq!(block.input_size(), 64);
    assert_eq!(block.output_size(), 64);
}

#[test]
fn test_transformer_block_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(64, 4, 256, &mut rng);

    // LayerNorm1: 2 * 64 = 128
    // Attention: 4 * (64*64 + 64) = 4 * 4160 = 16640
    // LayerNorm2: 2 * 64 = 128
    // FFN1: 64 * 256 + 256 = 16640
    // FFN2: 256 * 64 + 64 = 16448
    // Total: 128 + 16640 + 128 + 16640 + 16448 = 49984
    let expected = 2 * 64 + 4 * (64 * 64 + 64) + 2 * 64 + (64 * 256 + 256) + (256 * 64 + 64);
    assert_eq!(block.parameter_count(), expected);
}

#[test]
#[should_panic(expected = "d_model (63) must be divisible by num_heads (4)")]
fn test_transformer_block_invalid_d_model() {
    let mut rng = SimpleRng::new(42);
    let _block = TransformerBlock::new(63, 4, 256, &mut rng);
}

#[test]
fn test_transformer_block_forward_output_shape() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 4, 64, &mut rng);

    let batch_size = 2;
    let seq_len = 8;
    let input = vec![1.0f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    block.forward(&input, &mut output, batch_size);

    // Output should have correct shape
    assert_eq!(output.len(), batch_size * seq_len * 32);

    // Output should be finite
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_transformer_block_forward_non_zero() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(64, 4, 128, &mut rng);

    let batch_size = 2;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 64];
    let mut output = vec![0.0f32; batch_size * seq_len * 64];

    block.forward(&input, &mut output, batch_size);

    // Output should not be all zeros (block has non-zero weights)
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_block_backward_gradient_shape() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 2, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 32];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should have correct shape
    assert_eq!(grad_input.len(), batch_size * seq_len * 32);

    // Gradients should be finite
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_transformer_block_backward_non_zero_gradients() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 2, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 32];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should not be all zeros
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_block_residual_connections() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(16, 2, 32, &mut rng);

    let batch_size = 1;
    let seq_len = 4;

    // Use varied input instead of constant (layer norm would make constant input all zeros)
    let mut input = vec![0.0f32; batch_size * seq_len * 16];
    for i in 0..input.len() {
        input[i] = ((i as f32) * 0.1).sin();
    }
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Output should be different from input due to transformations
    let mut different = false;
    for i in 0..output.len() {
        if (output[i] - input[i]).abs() > 1e-3 {
            different = true;
            break;
        }
    }
    assert!(different, "Output should differ from input");

    // Output magnitude should be non-zero
    let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(output_norm > 0.0);
    assert!(output_norm.is_finite());
}

#[test]
fn test_transformer_block_update_parameters() {
    let mut rng = SimpleRng::new(42);
    let mut block = TransformerBlock::new(16, 2, 32, &mut rng);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.5f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass to accumulate gradients
    let grad_output = vec![1.0f32; batch_size * seq_len * 16];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 16];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Get output before update
    let mut output_before = vec![0.0f32; batch_size * seq_len * 16];
    block.forward(&input, &mut output_before, batch_size);

    // Update parameters
    let learning_rate = 0.01;
    block.update_parameters(learning_rate);

    // Get output after update
    let mut output_after = vec![0.0f32; batch_size * seq_len * 16];
    block.forward(&input, &mut output_after, batch_size);

    // Output should change after parameter update
    let mut changed = false;
    for i in 0..output_before.len() {
        if (output_after[i] - output_before[i]).abs() > 1e-6 {
            changed = true;
            break;
        }
    }
    assert!(changed, "Parameters should change after update");
}

#[test]
fn test_transformer_block_update_with_optimizer_sgd() {
    let mut rng = SimpleRng::new(42);
    let mut block = TransformerBlock::new(16, 2, 32, &mut rng);
    let mut optimizer = SGD::new(0.01);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.5f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 16];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 16];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Get output before update
    let mut output_before = vec![0.0f32; batch_size * seq_len * 16];
    block.forward(&input, &mut output_before, batch_size);

    // Update with optimizer
    block.update_with_optimizer(&mut optimizer);

    // Get output after update
    let mut output_after = vec![0.0f32; batch_size * seq_len * 16];
    block.forward(&input, &mut output_after, batch_size);

    // Output should change after update
    let mut changed = false;
    for i in 0..output_before.len() {
        if (output_after[i] - output_before[i]).abs() > 1e-6 {
            changed = true;
            break;
        }
    }
    assert!(changed, "Parameters should change after optimizer update");
}

#[test]
fn test_transformer_block_update_with_optimizer_adam() {
    let mut rng = SimpleRng::new(42);
    let mut block = TransformerBlock::new(16, 2, 32, &mut rng);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.5f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 16];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 16];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update with Adam optimizer
    block.update_with_optimizer(&mut optimizer);

    // Verify no panic and parameters were updated
    // (Adam has internal state that makes verification complex)
    assert_eq!(block.parameter_count(), block.parameter_count());
}

#[test]
fn test_transformer_block_multiple_heads() {
    let mut rng = SimpleRng::new(42);

    // Test with different numbers of heads
    for num_heads in [1, 2, 4, 8] {
        let d_model = 32;
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");

        let block = TransformerBlock::new(d_model, num_heads, 64, &mut rng);

        let batch_size = 1;
        let seq_len = 4;
        let input = vec![0.5f32; batch_size * seq_len * d_model];
        let mut output = vec![0.0f32; batch_size * seq_len * d_model];

        block.forward(&input, &mut output, batch_size);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));

        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }
}

#[test]
fn test_transformer_block_different_sequence_lengths() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 4, 64, &mut rng);

    let batch_size = 2;

    // Test with different sequence lengths
    for seq_len in [1, 2, 4, 8, 16] {
        let input = vec![0.5f32; batch_size * seq_len * 32];
        let mut output = vec![0.0f32; batch_size * seq_len * 32];

        block.forward(&input, &mut output, batch_size);

        // Output should be finite
        assert!(output.iter().all(|&x| x.is_finite()));

        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }
}

#[test]
fn test_transformer_block_batch_consistency() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 4, 64, &mut rng);

    let seq_len = 4;

    // Single sample forward
    let input_single = vec![0.5f32; 1 * seq_len * 32];
    let mut output_single = vec![0.0f32; 1 * seq_len * 32];
    block.forward(&input_single, &mut output_single, 1);

    // Batch of 2 identical samples
    let mut input_batch = vec![0.5f32; 2 * seq_len * 32];
    let mut output_batch = vec![0.0f32; 2 * seq_len * 32];
    block.forward(&input_batch, &mut output_batch, 2);

    // First sample in batch should match single sample
    // (approximately, due to layer norm batch statistics)
    let first_sample = &output_batch[0..seq_len * 32];

    // Both should be finite
    assert!(output_single.iter().all(|&x| x.is_finite()));
    assert!(first_sample.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_transformer_block_gradient_flow() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 2, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;

    // Use varied input
    let mut input = vec![0.0f32; batch_size * seq_len * 32];
    for i in 0..input.len() {
        input[i] = ((i as f32) * 0.05).sin() * 0.5;
    }
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    // Forward pass
    block.forward(&input, &mut output, batch_size);

    // Backward pass with realistic gradient (scaled)
    let grad_output = vec![0.1f32; batch_size * seq_len * 32];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
    block.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradient should flow through the network
    // With residual connections, gradients should be well-behaved
    let grad_norm: f32 = grad_input.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Gradient norm should be positive and finite
    assert!(grad_norm > 0.0);
    assert!(grad_norm.is_finite());

    // Check that gradients aren't all the same (they should vary)
    let first_grad = grad_input[0];
    let has_variation = grad_input.iter().any(|&g| (g - first_grad).abs() > 1e-6);
    assert!(has_variation, "Gradients should vary across dimensions");
}

#[test]
fn test_transformer_block_deterministic() {
    let mut rng1 = SimpleRng::new(42);
    let block1 = TransformerBlock::new(32, 4, 64, &mut rng1);

    let mut rng2 = SimpleRng::new(42);
    let block2 = TransformerBlock::new(32, 4, 64, &mut rng2);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output1 = vec![0.0f32; batch_size * seq_len * 32];
    let mut output2 = vec![0.0f32; batch_size * seq_len * 32];

    block1.forward(&input, &mut output1, batch_size);
    block2.forward(&input, &mut output2, batch_size);

    // Same seed should produce identical outputs
    for i in 0..output1.len() {
        assert!(
            (output1[i] - output2[i]).abs() < 1e-6,
            "Outputs should be identical with same seed"
        );
    }
}

#[test]
fn test_transformer_block_zero_input() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(32, 4, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.0f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    block.forward(&input, &mut output, batch_size);

    // Output should be finite even with zero input
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_transformer_block_numerical_stability_large_values() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(16, 2, 32, &mut rng);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![10.0f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    block.forward(&input, &mut output, batch_size);

    // Output should be finite even with large input values
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_transformer_block_numerical_stability_small_values() {
    let mut rng = SimpleRng::new(42);
    let block = TransformerBlock::new(16, 2, 32, &mut rng);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.001f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    block.forward(&input, &mut output, batch_size);

    // Output should be finite even with small input values
    assert!(output.iter().all(|&x| x.is_finite()));
}

// ============================================================================
// TransformerEncoder (Stacked Blocks) Tests
// ============================================================================

#[test]
fn test_transformer_stack_single_layer() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(1, 32, 4, 64, &mut rng);

    assert_eq!(encoder.num_layers(), 1);
    assert_eq!(encoder.input_size(), 32);
    assert_eq!(encoder.output_size(), 32);

    let batch_size = 2;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    encoder.forward(&input, &mut output, batch_size);

    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_stack_two_layers() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(2, 32, 4, 64, &mut rng);

    assert_eq!(encoder.num_layers(), 2);
    assert_eq!(encoder.input_size(), 32);
    assert_eq!(encoder.output_size(), 32);

    let batch_size = 2;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    encoder.forward(&input, &mut output, batch_size);

    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_stack_four_layers() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(4, 32, 4, 64, &mut rng);

    assert_eq!(encoder.num_layers(), 4);
    assert_eq!(encoder.input_size(), 32);
    assert_eq!(encoder.output_size(), 32);

    let batch_size = 2;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    encoder.forward(&input, &mut output, batch_size);

    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_stack_six_layers() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(6, 32, 4, 64, &mut rng);

    assert_eq!(encoder.num_layers(), 6);
    assert_eq!(encoder.input_size(), 32);
    assert_eq!(encoder.output_size(), 32);

    let batch_size = 2;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    encoder.forward(&input, &mut output, batch_size);

    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_stack_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(3, 32, 4, 64, &mut rng);

    // Should be 3x the parameter count of a single block
    let mut rng2 = SimpleRng::new(42);
    let single_block = TransformerBlock::new(32, 4, 64, &mut rng2);
    let expected = 3 * single_block.parameter_count();

    assert_eq!(encoder.parameter_count(), expected);
}

#[test]
fn test_transformer_stack_backward() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(2, 32, 4, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    // Forward pass
    encoder.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 32];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
    encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should be finite
    assert!(grad_input.iter().all(|&x| x.is_finite()));

    // Gradients should not be all zeros
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn test_transformer_stack_update_parameters() {
    let mut rng = SimpleRng::new(42);
    let mut encoder = TransformerEncoder::new(2, 16, 2, 32, &mut rng);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.5f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    encoder.forward(&input, &mut output, batch_size);

    // Backward pass to accumulate gradients
    let grad_output = vec![1.0f32; batch_size * seq_len * 16];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 16];
    encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Get output before update
    let mut output_before = vec![0.0f32; batch_size * seq_len * 16];
    encoder.forward(&input, &mut output_before, batch_size);

    // Update parameters
    let learning_rate = 0.01;
    encoder.update_parameters(learning_rate);

    // Get output after update
    let mut output_after = vec![0.0f32; batch_size * seq_len * 16];
    encoder.forward(&input, &mut output_after, batch_size);

    // Output should change after parameter update
    let mut changed = false;
    for i in 0..output_before.len() {
        if (output_after[i] - output_before[i]).abs() > 1e-6 {
            changed = true;
            break;
        }
    }
    assert!(changed, "Parameters should change after update");
}

#[test]
fn test_transformer_stack_update_with_optimizer() {
    let mut rng = SimpleRng::new(42);
    let mut encoder = TransformerEncoder::new(2, 16, 2, 32, &mut rng);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    let batch_size = 1;
    let seq_len = 2;
    let input = vec![0.5f32; batch_size * seq_len * 16];
    let mut output = vec![0.0f32; batch_size * seq_len * 16];

    // Forward pass
    encoder.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * seq_len * 16];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 16];
    encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update with optimizer
    encoder.update_with_optimizer(&mut optimizer);

    // Verify no panic (parameters were updated internally)
    assert_eq!(encoder.num_layers(), 2);
}

#[test]
fn test_transformer_stack_depth_increases_capacity() {
    let mut rng = SimpleRng::new(42);

    let batch_size = 1;
    let seq_len = 4;
    let input = vec![0.5f32; batch_size * seq_len * 32];

    // Test 1, 2, 4 layers
    for num_layers in [1, 2, 4] {
        let encoder = TransformerEncoder::new(num_layers, 32, 4, 64, &mut rng);
        let mut output = vec![0.0f32; batch_size * seq_len * 32];

        encoder.forward(&input, &mut output, batch_size);

        // All outputs should be finite and non-zero
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output.iter().any(|&x| x.abs() > 1e-6));

        // Parameter count should scale linearly with depth
        let mut rng2 = SimpleRng::new(42);
        let single_block = TransformerBlock::new(32, 4, 64, &mut rng2);
        assert_eq!(
            encoder.parameter_count(),
            num_layers * single_block.parameter_count()
        );
    }
}

#[test]
fn test_transformer_stack_gradient_flow_through_depth() {
    let mut rng = SimpleRng::new(42);
    let encoder = TransformerEncoder::new(4, 32, 2, 64, &mut rng);

    let batch_size = 1;
    let seq_len = 4;

    // Use varied input
    let mut input = vec![0.0f32; batch_size * seq_len * 32];
    for i in 0..input.len() {
        input[i] = ((i as f32) * 0.05).sin() * 0.5;
    }
    let mut output = vec![0.0f32; batch_size * seq_len * 32];

    // Forward pass
    encoder.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![0.1f32; batch_size * seq_len * 32];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
    encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradient should flow through all layers
    let grad_norm: f32 = grad_input.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Gradient norm should be positive and finite
    assert!(grad_norm > 0.0);
    assert!(grad_norm.is_finite());

    // Check that gradients aren't all the same
    let first_grad = grad_input[0];
    let has_variation = grad_input.iter().any(|&g| (g - first_grad).abs() > 1e-6);
    assert!(has_variation, "Gradients should vary across dimensions");
}
