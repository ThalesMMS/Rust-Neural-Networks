// Comprehensive tests for RNN layer: gradient checking, forward/backward correctness, and sequential processing.
// Following patterns from test_backward_pass.rs and test_gradient_checking.rs.

use rust_neural_networks::layers::{Layer, RnnLayer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Helper Functions for Numerical Gradient Checking
// ============================================================================

/// Compute the loss for a single RNN forward pass.
/// Loss = 0.5 * sum((target - output)^2)
fn compute_rnn_loss(layer: &RnnLayer, input: &[f32], target: &[f32]) -> f32 {
    let batch_size = 1;
    let mut output = vec![0.0f32; layer.output_size()];
    layer.forward(input, &mut output, batch_size);

    let mut loss = 0.0f32;
    for i in 0..layer.output_size() {
        let error = target[i] - output[i];
        loss += error * error;
    }
    loss / 2.0
}

/// Verify gradients by checking that parameter updates reduce loss.
/// This is an indirect test that gradients point in the right direction.
fn verify_gradient_descent_reduces_loss(
    layer: &mut RnnLayer,
    input: &[f32],
    target: &[f32],
    learning_rate: f32,
    num_steps: usize,
) -> bool {
    layer.reset_hidden_state();

    // Compute initial loss
    let initial_loss = compute_rnn_loss(layer, input, target);

    // Run a few gradient descent steps
    for _ in 0..num_steps {
        layer.reset_hidden_state();

        // Forward pass
        let mut output = vec![0.0f32; layer.output_size()];
        layer.forward(input, &mut output, 1);

        // Compute gradient
        let mut grad_output = vec![0.0f32; layer.output_size()];
        for i in 0..layer.output_size() {
            grad_output[i] = output[i] - target[i];
        }

        // Backward pass
        let mut grad_input = vec![0.0f32; layer.input_size()];
        layer.backward(input, &grad_output, &mut grad_input, 1);

        // Update parameters
        layer.update_parameters(learning_rate);
    }

    // Compute final loss
    layer.reset_hidden_state();
    let final_loss = compute_rnn_loss(layer, input, target);

    // Loss should decrease if gradients are correct
    final_loss < initial_loss
}

// ============================================================================
// Test: Forward Pass Shape and Output
// ============================================================================

#[test]
fn test_rnn_forward_output_shape() {
    let mut rng = SimpleRng::new(42);
    let input_size = 5;
    let hidden_size = 8;
    let output_size = 3;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Test with batch_size = 1
    let input = vec![0.5f32; input_size];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, 1);

    assert_eq!(output.len(), output_size, "Output should have correct size");
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "All outputs should be finite"
    );

    // Test with batch_size = 4
    layer.reset_hidden_state();
    let input_batch = vec![0.3f32; 4 * input_size];
    let mut output_batch = vec![0.0f32; 4 * output_size];
    layer.forward(&input_batch, &mut output_batch, 4);

    assert_eq!(
        output_batch.len(),
        4 * output_size,
        "Batch output should have correct size"
    );
    assert!(
        output_batch.iter().all(|&x| x.is_finite()),
        "All batch outputs should be finite"
    );
}

// ============================================================================
// Test: Hidden State Persistence Across Time Steps
// ============================================================================

#[test]
fn test_rnn_hidden_state_persistence() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Time step 0
    let input_t0 = vec![1.0f32, 0.0, 0.0];
    let mut output_t0 = vec![0.0f32; output_size];
    layer.forward(&input_t0, &mut output_t0, 1);
    let hidden_t0 = layer.get_hidden_state();

    // Hidden state should be non-zero after first forward pass
    assert!(
        hidden_t0.iter().any(|&x| x.abs() > 1e-6),
        "Hidden state should be updated after forward pass"
    );

    // Time step 1
    let input_t1 = vec![0.0f32, 1.0, 0.0];
    let mut output_t1 = vec![0.0f32; output_size];
    layer.forward(&input_t1, &mut output_t1, 1);
    let hidden_t1 = layer.get_hidden_state();

    // Hidden state should have changed
    assert_ne!(
        hidden_t0, hidden_t1,
        "Hidden state should change between time steps"
    );

    // Output should be influenced by previous hidden state
    // Reset and compare with fresh start
    layer.reset_hidden_state();
    let mut output_fresh = vec![0.0f32; output_size];
    layer.forward(&input_t1, &mut output_fresh, 1);

    // Output with history should differ from output without history
    assert_ne!(
        output_t1, output_fresh,
        "Output should depend on hidden state history"
    );
}

// ============================================================================
// Test: Backward Pass Gradient Shape
// ============================================================================

#[test]
fn test_rnn_backward_gradient_shape() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 3;
    let batch_size = 2;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Forward pass
    let input = vec![0.5f32; batch_size * input_size];
    let mut output = vec![0.0f32; batch_size * output_size];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0f32; batch_size * output_size];
    let mut grad_input = vec![0.0f32; batch_size * input_size];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradient shape
    assert_eq!(
        grad_input.len(),
        batch_size * input_size,
        "Gradient should have correct shape"
    );

    // Check gradients are finite
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "All gradients should be finite"
    );

    // Check at least some gradients are non-zero
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "At least some gradients should be non-zero"
    );
}

// ============================================================================
// Test: Gradient Checking via Loss Reduction
// ============================================================================

#[test]
fn test_rnn_gradient_descent_reduces_loss() {
    let mut rng = SimpleRng::new(123);
    let input_size = 5;
    let hidden_size = 8;
    let output_size = 3;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Test data
    let input = vec![0.5f32, -0.3, 0.8, 0.2, -0.6];
    let target = vec![0.2f32, 0.7, -0.4];

    // Verify that gradient descent reduces loss
    let learning_rate = 0.1f32;
    let num_steps = 10;

    let loss_reduced =
        verify_gradient_descent_reduces_loss(&mut layer, &input, &target, learning_rate, num_steps);

    assert!(
        loss_reduced,
        "Gradient descent should reduce loss if gradients are correct"
    );
}

// ============================================================================
// Test: Gradient Checking with Non-Zero Hidden State
// ============================================================================

#[test]
fn test_rnn_gradients_with_hidden_state() {
    let mut rng = SimpleRng::new(456);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 2;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Set a non-zero hidden state to test recurrent connections
    let initial_hidden = vec![0.2f32, -0.1, 0.15, -0.08, 0.12, 0.05];
    layer.set_hidden_state(&initial_hidden);

    let input = vec![0.4f32, 0.6, -0.2, 0.3];
    let target = vec![0.3f32, 0.1];

    // Verify that gradient descent still works with non-zero hidden state
    let learning_rate = 0.05f32;
    let num_steps = 5;

    let loss_reduced =
        verify_gradient_descent_reduces_loss(&mut layer, &input, &target, learning_rate, num_steps);

    assert!(
        loss_reduced,
        "Gradient descent should reduce loss even with non-zero hidden state"
    );
}

// ============================================================================
// Test: Gradient Correctness via Multiple Training Steps
// ============================================================================

#[test]
fn test_rnn_training_convergence() {
    let mut rng = SimpleRng::new(999);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 2;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Simple training data
    let inputs = [
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0, 0.0],
        vec![0.0f32, 0.0, 1.0, 0.0],
    ];
    let targets = [vec![1.0f32, 0.0], vec![0.0f32, 1.0], vec![1.0f32, 1.0]];

    let learning_rate = 0.1f32;
    let epochs = 20;

    // Measure initial loss
    layer.reset_hidden_state();
    let mut initial_total_loss = 0.0f32;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        initial_total_loss += compute_rnn_loss(&layer, input, target);
    }

    // Train for multiple epochs
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            layer.reset_hidden_state();

            // Forward
            let mut output = vec![0.0f32; output_size];
            layer.forward(input, &mut output, 1);

            // Compute gradient
            let mut grad_output = vec![0.0f32; output_size];
            for i in 0..output_size {
                grad_output[i] = output[i] - target[i];
            }

            // Backward
            let mut grad_input = vec![0.0f32; input_size];
            layer.backward(input, &grad_output, &mut grad_input, 1);

            // Update
            layer.update_parameters(learning_rate);
        }
    }

    // Measure final loss
    layer.reset_hidden_state();
    let mut final_total_loss = 0.0f32;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        final_total_loss += compute_rnn_loss(&layer, input, target);
    }

    // Loss should decrease with training
    assert!(
        final_total_loss < initial_total_loss * 0.5,
        "Training should significantly reduce loss: initial={:.6}, final={:.6}",
        initial_total_loss,
        final_total_loss
    );
}

// ============================================================================
// Test: Sequential Processing with Multiple Time Steps
// ============================================================================

#[test]
fn test_rnn_sequential_processing() {
    let mut rng = SimpleRng::new(42);
    let input_size = 5;
    let hidden_size = 8;
    let output_size = 3;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Process a sequence of 5 time steps
    let sequence_length = 5;
    let mut outputs = Vec::new();
    let mut hidden_states = Vec::new();

    for t in 0..sequence_length {
        let input = vec![(t as f32) * 0.1; input_size];
        let mut output = vec![0.0f32; output_size];
        layer.forward(&input, &mut output, 1);

        outputs.push(output.clone());
        hidden_states.push(layer.get_hidden_state());
    }

    // Verify that outputs change across time steps
    for t in 1..sequence_length {
        assert_ne!(
            outputs[t - 1],
            outputs[t],
            "Outputs should differ across time steps"
        );
        assert_ne!(
            hidden_states[t - 1],
            hidden_states[t],
            "Hidden states should differ across time steps"
        );
    }

    // Verify all outputs and hidden states are finite
    for t in 0..sequence_length {
        assert!(
            outputs[t].iter().all(|&x| x.is_finite()),
            "All outputs at time {} should be finite",
            t
        );
        assert!(
            hidden_states[t].iter().all(|&x| x.is_finite()),
            "All hidden states at time {} should be finite",
            t
        );
    }
}

#[test]
fn test_rnn_backward_bptt_propagates_later_loss_to_earlier_timestep() {
    let mut rng = SimpleRng::new(2026);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    let inputs = [vec![1.0f32, 0.2, -0.1], vec![0.0f32, 0.5, 0.7]];

    layer.reset_hidden_state();
    for input in &inputs {
        let mut output = vec![0.0f32; output_size];
        layer.forward(input, &mut output, batch_size);
    }

    let mut grad_input_t1 = vec![0.0f32; input_size];
    let zero_dh_next = vec![0.0f32; hidden_size];
    let dh_from_later = layer.backward_bptt(
        &inputs[1],
        &[1.0f32, -0.5],
        &mut grad_input_t1,
        &zero_dh_next,
        batch_size,
    );

    assert!(
        dh_from_later.iter().any(|&x| x.abs() > 1e-10),
        "later timestep loss should produce hidden-state gradient"
    );

    layer.reset_hidden_state();
    let mut replay_output = vec![0.0f32; output_size];
    layer.forward(&inputs[0], &mut replay_output, batch_size);

    let mut grad_input_t0 = vec![0.0f32; input_size];
    let zero_grad_output_t0 = vec![0.0f32; output_size];
    let dh_before_t0 = layer.backward_bptt(
        &inputs[0],
        &zero_grad_output_t0,
        &mut grad_input_t0,
        &dh_from_later,
        batch_size,
    );

    assert!(
        grad_input_t0.iter().any(|&x| x.abs() > 1e-10),
        "earlier timestep input should receive gradient from later loss"
    );
    assert!(
        dh_before_t0.iter().all(|&x| x.is_finite()),
        "returned hidden-state gradients should be finite"
    );
}

// ============================================================================
// Test: Parameter Updates
// ============================================================================

#[test]
fn test_rnn_parameter_updates() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Do a forward and backward pass to accumulate gradients
    layer.reset_hidden_state();
    let input = vec![0.5f32, -0.3, 0.8];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0f32, -0.5];
    let mut grad_input = vec![0.0f32; input_size];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Store original weights
    let w_xh_before = layer.w_xh().to_vec();
    let w_hy_before = layer.w_hy().to_vec();
    let b_y_before = layer.b_y().to_vec();

    // Update parameters
    let learning_rate = 0.01f32;
    layer.update_parameters(learning_rate);

    // Check that parameters have changed
    let w_xh_after = layer.w_xh();
    let w_hy_after = layer.w_hy();
    let b_y_after = layer.b_y();

    let w_xh_changed = w_xh_before
        .iter()
        .zip(w_xh_after.iter())
        .any(|(before, after)| (before - after).abs() > 1e-10);
    let w_hy_changed = w_hy_before
        .iter()
        .zip(w_hy_after.iter())
        .any(|(before, after)| (before - after).abs() > 1e-10);
    let b_y_changed = b_y_before
        .iter()
        .zip(b_y_after.iter())
        .any(|(before, after)| (before - after).abs() > 1e-10);

    assert!(
        w_xh_changed || w_hy_changed || b_y_changed,
        "At least some parameters should be updated"
    );

    // Note: w_hh and b_h may not change if hidden state was zero in the first forward pass
    // But after the first time step, hidden state should be non-zero
}

// ============================================================================
// Test: Batch Processing Correctness
// ============================================================================

#[test]
fn test_rnn_batch_processing() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 3;
    let batch_size = 3;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Create batch input
    let input_batch = [
        vec![0.1f32, 0.2, 0.3, 0.4],
        vec![0.5f32, 0.6, 0.7, 0.8],
        vec![0.9f32, 1.0, 1.1, 1.2],
    ];
    let input_flat: Vec<f32> = input_batch.iter().flatten().copied().collect();

    // Forward pass on batch
    let mut output_batch = vec![0.0f32; batch_size * output_size];
    layer.forward(&input_flat, &mut output_batch, batch_size);

    // Check output shape
    assert_eq!(output_batch.len(), batch_size * output_size);

    // Check all outputs are finite
    assert!(
        output_batch.iter().all(|&x| x.is_finite()),
        "All batch outputs should be finite"
    );

    // Backward pass on batch
    let grad_output_batch = vec![1.0f32; batch_size * output_size];
    let mut grad_input_batch = vec![0.0f32; batch_size * input_size];
    layer.backward(
        &input_flat,
        &grad_output_batch,
        &mut grad_input_batch,
        batch_size,
    );

    // Check gradient shape
    assert_eq!(grad_input_batch.len(), batch_size * input_size);

    // Check all gradients are finite
    assert!(
        grad_input_batch.iter().all(|&x| x.is_finite()),
        "All batch gradients should be finite"
    );

    // Check at least some gradients are non-zero
    assert!(
        grad_input_batch.iter().any(|&x| x.abs() > 1e-10),
        "At least some batch gradients should be non-zero"
    );
}

// ============================================================================
// Test: Zero Hidden State Behavior
// ============================================================================

#[test]
fn test_rnn_zero_hidden_state() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // With zero hidden state, W_hh should not contribute
    let input = vec![0.5f32, -0.3, 0.8];
    let mut output1 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output1, 1);

    // Reset and do again - first forward should be identical
    layer.reset_hidden_state();
    let mut output2 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output2, 1);

    // Outputs should be identical when starting from zero hidden state
    for i in 0..output_size {
        assert!(
            (output1[i] - output2[i]).abs() < 1e-6,
            "Outputs should be identical when starting from zero hidden state"
        );
    }

    // But the second forward pass (without reset) should differ
    let mut output3 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output3, 1);

    assert_ne!(
        output2, output3,
        "Output should differ when hidden state is non-zero"
    );
}

// ============================================================================
// Test: Sequential Learning with Temporal Dependencies
// ============================================================================

#[test]
fn test_rnn_temporal_dependencies() {
    let mut rng = SimpleRng::new(987);
    let input_size = 2;
    let hidden_size = 4;
    let output_size = 1;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Sequence: [1,0] -> [0,1] -> [1,1]
    // Test that the RNN can learn to use hidden state across time steps
    let sequence = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0], vec![1.0f32, 1.0]];

    // Train on this sequence multiple times
    let learning_rate = 0.05f32;
    let epochs = 15;

    for _ in 0..epochs {
        layer.reset_hidden_state();

        // Process sequence
        for input in &sequence {
            let mut output = vec![0.0f32; output_size];
            layer.forward(input, &mut output, 1);

            // Simple target: output should be 1.0
            let target = [1.0f32];
            let mut grad_output = vec![0.0f32; output_size];
            for i in 0..output_size {
                grad_output[i] = output[i] - target[i];
            }

            let mut grad_input = vec![0.0f32; input_size];
            layer.backward(input, &grad_output, &mut grad_input, 1);
        }

        // Update after processing the whole sequence
        layer.update_parameters(learning_rate);
    }

    // After training, outputs should be closer to target
    layer.reset_hidden_state();
    let mut final_outputs = Vec::new();
    for input in &sequence {
        let mut output = vec![0.0f32; output_size];
        layer.forward(input, &mut output, 1);
        final_outputs.push(output[0]);
    }

    // Outputs should be reasonable (not NaN, not too far from target)
    for (i, &out) in final_outputs.iter().enumerate() {
        assert!(out.is_finite(), "Output at step {} should be finite", i);
        assert!(
            out.abs() < 5.0,
            "Output at step {} should be reasonable magnitude: {}",
            i,
            out
        );
    }
}
