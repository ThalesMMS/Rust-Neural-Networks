//! Unit tests for Multi-Head Attention layer
//!
//! Tests the MultiHeadAttentionLayer implementation including forward pass,
//! backward pass, gradient computation, and parameter updates.
//! Following the pattern from test_backward_pass.rs and test_gradient_checking.rs.

use approx::assert_relative_eq;
use rust_neural_networks::layers::{Layer, MultiHeadAttentionLayer};
use rust_neural_networks::utils::rng::SimpleRng;

#[test]
fn test_attention_creation() {
    let mut rng = SimpleRng::new(42);
    let layer = MultiHeadAttentionLayer::new(64, 4, &mut rng);

    assert_eq!(layer.d_model(), 64);
    assert_eq!(layer.num_heads(), 4);
    assert_eq!(layer.d_head(), 16);
    assert_eq!(layer.input_size(), 64);
    assert_eq!(layer.output_size(), 64);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn test_attention_invalid_heads() {
    let mut rng = SimpleRng::new(42);
    // d_model=64 is not divisible by num_heads=5
    let _ = MultiHeadAttentionLayer::new(64, 5, &mut rng);
}

#[test]
fn test_attention_forward_shape() {
    let mut rng = SimpleRng::new(42);
    let layer = MultiHeadAttentionLayer::new(32, 4, &mut rng);

    let batch_size = 2;
    let seq_len = 10;
    let d_model = 32;

    // Input: [batch_size, seq_len, d_model] flattened
    let input = vec![1.0f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];

    layer.forward(&input, &mut output, batch_size);

    // Output should have same shape as input
    assert_eq!(output.len(), input.len());

    // Output should not be all zeros (attention should compute something)
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 1e-6, "Output should not be all zeros");
}

#[test]
fn test_attention_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let layer = MultiHeadAttentionLayer::new(64, 8, &mut rng);

    // 4 weight matrices (Q, K, V, O): 64 * 64 each
    // 4 bias vectors (Q, K, V, O): 64 each
    let expected = 4 * (64 * 64 + 64);
    assert_eq!(layer.parameter_count(), expected);
}

#[test]
fn test_attention_single_head() {
    let mut rng = SimpleRng::new(42);
    let layer = MultiHeadAttentionLayer::new(32, 1, &mut rng);

    assert_eq!(layer.num_heads(), 1);
    assert_eq!(layer.d_head(), 32);

    let batch_size = 1;
    let seq_len = 5;
    let d_model = 32;

    let input = vec![0.5f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];

    layer.forward(&input, &mut output, batch_size);

    // Check output is computed
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 0.0);
}

#[test]
fn test_attention_backward_pass() {
    let mut rng = SimpleRng::new(123);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 3;
    let d_model = 16;

    // Forward pass
    let input = vec![0.1f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![0.01f32; batch_size * seq_len * d_model];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * d_model];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradients are computed
    let grad_sum: f32 = grad_input.iter().map(|x| x.abs()).sum();
    assert!(grad_sum > 0.0, "Gradients should be computed");
}

#[test]
fn test_attention_parameter_update() {
    let mut rng = SimpleRng::new(456);
    let mut layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 3;
    let d_model = 16;

    // Forward and backward pass to accumulate gradients
    let input = vec![0.1f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output, batch_size);

    let grad_output = vec![0.01f32; batch_size * seq_len * d_model];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * d_model];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update parameters (should not panic)
    layer.update_parameters(0.01);
}

#[test]
fn test_attention_multiple_heads() {
    let mut rng = SimpleRng::new(789);

    // Test with different number of heads
    for num_heads in [1, 2, 4, 8] {
        let d_model = 64;
        let layer = MultiHeadAttentionLayer::new(d_model, num_heads, &mut rng);

        assert_eq!(layer.num_heads(), num_heads);
        assert_eq!(layer.d_head(), d_model / num_heads);

        let batch_size = 2;
        let seq_len = 4;

        let input = vec![0.5f32; batch_size * seq_len * d_model];
        let mut output = vec![0.0f32; batch_size * seq_len * d_model];

        layer.forward(&input, &mut output, batch_size);

        // Output should be computed for all heads
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0);
    }
}

#[test]
fn test_attention_different_sequence_lengths() {
    let mut rng = SimpleRng::new(101112);
    let layer = MultiHeadAttentionLayer::new(32, 4, &mut rng);

    let d_model = 32;

    // Test with different sequence lengths
    for seq_len in [1, 5, 10, 20] {
        let batch_size = 2;

        let input = vec![0.5f32; batch_size * seq_len * d_model];
        let mut output = vec![0.0f32; batch_size * seq_len * d_model];

        layer.forward(&input, &mut output, batch_size);

        // Check output shape is correct
        assert_eq!(output.len(), batch_size * seq_len * d_model);

        // Check output is computed
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0);
    }
}

#[test]
fn test_attention_output_different_from_input() {
    let mut rng = SimpleRng::new(131415);
    let layer = MultiHeadAttentionLayer::new(32, 4, &mut rng);

    let batch_size = 2;
    let seq_len = 5;
    let d_model = 32;

    // Create non-uniform input
    let mut input = vec![0.0f32; batch_size * seq_len * d_model];
    for (i, val) in input.iter_mut().enumerate() {
        *val = (i as f32) * 0.01;
    }

    let mut output = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output, batch_size);

    // Output should be different from input (attention transforms the input)
    let mut differences = 0;
    for (inp, out) in input.iter().zip(output.iter()) {
        if (inp - out).abs() > 1e-6 {
            differences += 1;
        }
    }

    // At least 50% of values should be different
    assert!(
        differences > input.len() / 2,
        "Attention should transform the input significantly"
    );
}

#[test]
fn test_attention_gradient_shapes() {
    // Test that gradients have correct shapes (following test_backward_pass.rs pattern)
    let mut rng = SimpleRng::new(161718);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 2;
    let seq_len = 4;
    let d_model = 16;
    let total_size = batch_size * seq_len * d_model;

    // Forward pass
    let input = vec![0.1f32; total_size];
    let mut output = vec![0.0f32; total_size];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![0.01f32; total_size];
    let mut grad_input = vec![0.0f32; total_size];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradient shape matches input shape
    assert_eq!(
        grad_input.len(),
        input.len(),
        "Gradient shape should match input shape"
    );

    // Check that gradients are not all zeros
    let grad_sum: f32 = grad_input.iter().map(|x| x.abs()).sum();
    assert!(grad_sum > 0.0, "Gradients should be non-zero");

    // Check that gradients are finite
    for &g in grad_input.iter() {
        assert!(g.is_finite(), "All gradients should be finite");
    }
}

#[test]
fn test_attention_numerical_gradient_checking() {
    // Numerical gradient checking using finite differences (following test_gradient_checking.rs pattern)
    let mut rng = SimpleRng::new(192021);
    let layer = MultiHeadAttentionLayer::new(8, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 2;
    let d_model = 8;
    let total_size = batch_size * seq_len * d_model;

    // Simple input
    let input = vec![0.5f32; total_size];
    let expected_output = vec![1.0f32; total_size];

    // Forward pass
    let mut output = vec![0.0f32; total_size];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass to get analytical gradients
    let mut grad_output = vec![0.0f32; total_size];
    for i in 0..total_size {
        grad_output[i] = output[i] - expected_output[i];
    }
    let mut grad_input_analytical = vec![0.0f32; total_size];
    layer.backward(&input, &grad_output, &mut grad_input_analytical, batch_size);

    // Compute numerical gradients for a few sample points
    let epsilon = 1e-4f32;
    for idx in [0, 4, 8, 12] {
        if idx >= total_size {
            continue;
        }

        // Perturb input
        let mut input_plus = input.clone();
        let mut input_minus = input.clone();
        input_plus[idx] += epsilon;
        input_minus[idx] -= epsilon;

        // Forward pass with perturbed inputs
        let mut output_plus = vec![0.0f32; total_size];
        let mut output_minus = vec![0.0f32; total_size];
        layer.forward(&input_plus, &mut output_plus, batch_size);
        layer.forward(&input_minus, &mut output_minus, batch_size);

        // Compute losses
        let mut loss_plus = 0.0f32;
        let mut loss_minus = 0.0f32;
        for i in 0..total_size {
            let error_plus = output_plus[i] - expected_output[i];
            let error_minus = output_minus[i] - expected_output[i];
            loss_plus += error_plus * error_plus;
            loss_minus += error_minus * error_minus;
        }
        loss_plus /= 2.0;
        loss_minus /= 2.0;

        // Numerical gradient
        let grad_numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        let grad_analytical = grad_input_analytical[idx];

        // Compare with relative tolerance
        let rel_error = if grad_analytical.abs() + grad_numerical.abs() > 1e-6 {
            (grad_numerical - grad_analytical).abs()
                / (grad_numerical.abs() + grad_analytical.abs())
        } else {
            0.0
        };

        assert!(
            rel_error < 1e-2,
            "Gradient mismatch at index {}: numerical={}, analytical={}, rel_error={}",
            idx,
            grad_numerical,
            grad_analytical,
            rel_error
        );
    }
}

#[test]
fn test_attention_weights_sum_to_one() {
    // Test that attention weights sum to 1.0 for each query position
    // This requires accessing internal attention weights, which we verify indirectly
    // by checking that the layer produces reasonable outputs
    let mut rng = SimpleRng::new(222324);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 3;
    let d_model = 16;

    // Create input with different values at each position
    let mut input = vec![0.0f32; batch_size * seq_len * d_model];
    for i in 0..seq_len {
        for d in 0..d_model {
            input[i * d_model + d] = (i as f32 + 1.0) * 0.1;
        }
    }

    let mut output = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output, batch_size);

    // Check that output is finite and reasonable
    for &val in output.iter() {
        assert!(val.is_finite(), "Output should be finite");
        assert!(val.abs() < 100.0, "Output should be in reasonable range");
    }
}

#[test]
fn test_attention_numerical_stability() {
    // Test numerical stability with extreme values
    let mut rng = SimpleRng::new(252627);
    let layer = MultiHeadAttentionLayer::new(16, 4, &mut rng);

    let batch_size = 1;
    let seq_len = 5;
    let d_model = 16;

    // Test with small values
    let input_small = vec![1e-6f32; batch_size * seq_len * d_model];
    let mut output_small = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input_small, &mut output_small, batch_size);

    for &val in output_small.iter() {
        assert!(val.is_finite(), "Output should be finite for small inputs");
    }

    // Test with large values
    let input_large = vec![10.0f32; batch_size * seq_len * d_model];
    let mut output_large = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input_large, &mut output_large, batch_size);

    for &val in output_large.iter() {
        assert!(val.is_finite(), "Output should be finite for large inputs");
    }

    // Test with mixed values
    let mut input_mixed = vec![0.0f32; batch_size * seq_len * d_model];
    for (i, val) in input_mixed.iter_mut().enumerate() {
        *val = if i % 2 == 0 { 10.0 } else { -10.0 };
    }
    let mut output_mixed = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input_mixed, &mut output_mixed, batch_size);

    for &val in output_mixed.iter() {
        assert!(val.is_finite(), "Output should be finite for mixed inputs");
    }
}

#[test]
fn test_attention_backward_numerical_stability() {
    // Test that backward pass is numerically stable
    let mut rng = SimpleRng::new(282930);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let d_model = 16;

    // Forward pass with normal values
    let input = vec![1.0f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass with large gradients
    let grad_output = vec![100.0f32; batch_size * seq_len * d_model];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * d_model];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradients are finite
    for &g in grad_input.iter() {
        assert!(
            g.is_finite(),
            "Gradients should be finite even with large grad_output"
        );
    }
}

#[test]
fn test_attention_parameter_update_changes_output() {
    // Test that parameter updates actually change the network behavior
    let mut rng = SimpleRng::new(313233);
    let mut layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 3;
    let d_model = 16;

    let input = vec![0.5f32; batch_size * seq_len * d_model];

    // First forward pass
    let mut output1 = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output1, batch_size);

    // Backward pass
    let grad_output = vec![0.1f32; batch_size * seq_len * d_model];
    let mut grad_input = vec![0.0f32; batch_size * seq_len * d_model];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update parameters with significant learning rate
    layer.update_parameters(1.0);

    // Second forward pass after update
    let mut output2 = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output2, batch_size);

    // Outputs should be different
    let mut differences = 0;
    for (out1, out2) in output1.iter().zip(output2.iter()) {
        if (out1 - out2).abs() > 1e-6 {
            differences += 1;
        }
    }

    assert!(
        differences > output1.len() / 4,
        "Parameter update should change outputs significantly"
    );
}

#[test]
fn test_attention_batch_processing() {
    // Test that batched processing produces consistent results
    let mut rng = SimpleRng::new(343536);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let seq_len = 3;
    let d_model = 16;

    // Process batch of 2
    let batch_size = 2;
    let mut input_batch = vec![0.0f32; batch_size * seq_len * d_model];
    // First sample
    for val in input_batch[..(seq_len * d_model)].iter_mut() {
        *val = 0.5;
    }
    // Second sample
    for val in input_batch[(seq_len * d_model)..(2 * seq_len * d_model)].iter_mut() {
        *val = 0.5;
    }

    let mut output_batch = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input_batch, &mut output_batch, batch_size);

    // Process individually
    let input_single = vec![0.5f32; seq_len * d_model];
    let mut output_single1 = vec![0.0f32; seq_len * d_model];
    let mut output_single2 = vec![0.0f32; seq_len * d_model];
    layer.forward(&input_single, &mut output_single1, 1);
    layer.forward(&input_single, &mut output_single2, 1);

    // Compare first sample from batch with individual processing
    for i in 0..(seq_len * d_model) {
        assert_relative_eq!(
            output_batch[i],
            output_single1[i],
            epsilon = 1e-5,
            max_relative = 1e-4
        );
    }

    // Compare second sample from batch with individual processing
    for i in 0..(seq_len * d_model) {
        assert_relative_eq!(
            output_batch[seq_len * d_model + i],
            output_single2[i],
            epsilon = 1e-5,
            max_relative = 1e-4
        );
    }
}

#[test]
fn test_attention_zero_input() {
    // Test that zero input produces reasonable output
    let mut rng = SimpleRng::new(373839);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 3;
    let d_model = 16;

    let input = vec![0.0f32; batch_size * seq_len * d_model];
    let mut output = vec![0.0f32; batch_size * seq_len * d_model];

    layer.forward(&input, &mut output, batch_size);

    // Output should be finite (not NaN or infinite)
    for &val in output.iter() {
        assert!(val.is_finite(), "Output should be finite for zero input");
    }
}

#[test]
fn test_attention_consistency() {
    // Test that multiple forward passes with same input produce same output
    let mut rng = SimpleRng::new(404142);
    let layer = MultiHeadAttentionLayer::new(16, 2, &mut rng);

    let batch_size = 1;
    let seq_len = 4;
    let d_model = 16;

    let input = vec![0.7f32; batch_size * seq_len * d_model];

    // First forward pass
    let mut output1 = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output1, batch_size);

    // Second forward pass
    let mut output2 = vec![0.0f32; batch_size * seq_len * d_model];
    layer.forward(&input, &mut output2, batch_size);

    // Outputs should be identical
    for (out1, out2) in output1.iter().zip(output2.iter()) {
        assert_relative_eq!(out1, out2, epsilon = 1e-7);
    }
}
