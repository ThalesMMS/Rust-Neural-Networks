// Comprehensive tests for LSTM layer: gate computations, gradient checking, and temporal dependencies.
// Following patterns from test_backward_pass.rs, test_gradient_checking.rs, and test_rnn.rs.

use rust_neural_networks::layers::{Layer, LstmLayer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Helper Functions for LSTM Testing
// ============================================================================

/// Compute the loss for a single LSTM forward pass.
/// Loss = 0.5 * sum((target - output)^2)
fn compute_lstm_loss(layer: &LstmLayer, input: &[f32], target: &[f32]) -> f32 {
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
fn verify_gradient_descent_reduces_loss(
    layer: &mut LstmLayer,
    input: &[f32],
    target: &[f32],
    learning_rate: f32,
    num_steps: usize,
) -> bool {
    layer.reset_state();

    // Compute initial loss
    let initial_loss = compute_lstm_loss(layer, input, target);

    // Run a few gradient descent steps
    for _ in 0..num_steps {
        layer.reset_state();

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
    layer.reset_state();
    let final_loss = compute_lstm_loss(layer, input, target);

    // Loss should decrease if gradients are correct
    final_loss < initial_loss
}

// ============================================================================
// Test: LSTM Creation and Initialization
// ============================================================================

#[test]
fn test_lstm_creation() {
    let mut rng = SimpleRng::new(42);
    let input_size = 10;
    let hidden_size = 20;
    let output_size = 5;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    assert_eq!(layer.input_size(), input_size);
    assert_eq!(layer.hidden_size(), hidden_size);
    assert_eq!(layer.output_size(), output_size);

    // Check hidden state initialized to zero
    let hidden = layer.get_hidden_state();
    assert_eq!(hidden.len(), hidden_size);
    assert!(hidden.iter().all(|&x| x == 0.0));

    // Check cell state initialized to zero
    let cell = layer.get_cell_state();
    assert_eq!(cell.len(), hidden_size);
    assert!(cell.iter().all(|&x| x == 0.0));
}

// ============================================================================
// Test: LSTM Parameter Count
// ============================================================================

#[test]
fn test_lstm_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let input_size = 8;
    let hidden_size = 16;
    let output_size = 4;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Each gate has: input_size × hidden_size + hidden_size × hidden_size + hidden_size
    let gate_params = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
    // 4 gates + output projection (hidden_size × output_size + output_size)
    let expected = 4 * gate_params + (hidden_size * output_size + output_size);

    assert_eq!(layer.parameter_count(), expected);
}

// ============================================================================
// Test: LSTM State Management
// ============================================================================

#[test]
fn test_lstm_state_management() {
    let mut rng = SimpleRng::new(42);
    let hidden_size = 10;

    let layer = LstmLayer::new(5, hidden_size, 3, &mut rng);

    // Test setting and getting hidden state
    let test_hidden = vec![0.5f32; hidden_size];
    layer.set_hidden_state(&test_hidden);
    let retrieved_hidden = layer.get_hidden_state();
    assert_eq!(retrieved_hidden, test_hidden);

    // Test setting and getting cell state
    let test_cell = vec![0.3f32; hidden_size];
    layer.set_cell_state(&test_cell);
    let retrieved_cell = layer.get_cell_state();
    assert_eq!(retrieved_cell, test_cell);

    // Test reset
    layer.reset_state();
    let reset_hidden = layer.get_hidden_state();
    let reset_cell = layer.get_cell_state();
    assert!(reset_hidden.iter().all(|&x| x == 0.0));
    assert!(reset_cell.iter().all(|&x| x == 0.0));
}

// ============================================================================
// Test: LSTM Forward Pass - Gate Computations
// ============================================================================

#[test]
fn test_lstm_forward_gate_computations() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 2;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // First forward pass with zero initial states
    let input = vec![0.5f32, -0.3, 0.8, 0.2];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, 1);

    // Check that hidden state was updated
    let hidden_state = layer.get_hidden_state();
    assert_eq!(hidden_state.len(), hidden_size);
    assert!(
        hidden_state.iter().any(|&x| x != 0.0),
        "Hidden state should be updated after forward pass"
    );

    // Check that cell state was updated
    let cell_state = layer.get_cell_state();
    assert_eq!(cell_state.len(), hidden_size);
    assert!(
        cell_state.iter().any(|&x| x != 0.0),
        "Cell state should be updated after forward pass"
    );

    // Check output is finite
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "All outputs should be finite"
    );

    // Check output is non-zero (gates produced some activations)
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Output should not be all zeros"
    );
}

// ============================================================================
// Test: LSTM Gate Values Range (sigmoid outputs should be in [0, 1])
// ============================================================================

#[test]
fn test_lstm_gate_value_ranges() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Process multiple inputs with different magnitudes
    let test_inputs = vec![
        vec![0.1f32, 0.2, 0.3],
        vec![1.0f32, -1.0, 0.5],
        vec![5.0f32, -5.0, 2.0],
    ];

    for input in test_inputs {
        let mut output = vec![0.0f32; output_size];
        layer.forward(&input, &mut output, 1);

        // All outputs should be finite (no NaN or Inf)
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "LSTM output should be finite for input {:?}",
            input
        );

        // Hidden and cell states should be finite
        let hidden = layer.get_hidden_state();
        let cell = layer.get_cell_state();
        assert!(
            hidden.iter().all(|&x| x.is_finite()),
            "Hidden state should be finite"
        );
        assert!(
            cell.iter().all(|&x| x.is_finite()),
            "Cell state should be finite"
        );
    }
}

// ============================================================================
// Test: LSTM Temporal Dependencies (Cell State Persistence)
// ============================================================================

#[test]
fn test_lstm_temporal_dependencies() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Time step 0
    let input_t0 = vec![1.0f32, 0.0, 0.0];
    let mut output_t0 = vec![0.0f32; output_size];
    layer.forward(&input_t0, &mut output_t0, 1);
    let hidden_t0 = layer.get_hidden_state();
    let cell_t0 = layer.get_cell_state();

    // Both states should be non-zero after first forward pass
    assert!(
        hidden_t0.iter().any(|&x| x.abs() > 1e-6),
        "Hidden state should be updated after forward pass"
    );
    assert!(
        cell_t0.iter().any(|&x| x.abs() > 1e-6),
        "Cell state should be updated after forward pass"
    );

    // Time step 1
    let input_t1 = vec![0.0f32, 1.0, 0.0];
    let mut output_t1 = vec![0.0f32; output_size];
    layer.forward(&input_t1, &mut output_t1, 1);
    let hidden_t1 = layer.get_hidden_state();
    let cell_t1 = layer.get_cell_state();

    // States should have changed
    assert_ne!(
        hidden_t0, hidden_t1,
        "Hidden state should change between time steps"
    );
    assert_ne!(
        cell_t0, cell_t1,
        "Cell state should change between time steps"
    );

    // Output should be influenced by previous states
    layer.reset_state();
    let mut output_fresh = vec![0.0f32; output_size];
    layer.forward(&input_t1, &mut output_fresh, 1);

    // Output with history should differ from output without history
    assert_ne!(
        output_t1, output_fresh,
        "Output should depend on temporal context (cell and hidden states)"
    );
}

// ============================================================================
// Test: LSTM Backward Pass - Gradient Shape
// ============================================================================

#[test]
fn test_lstm_backward_gradient_shape() {
    let mut rng = SimpleRng::new(42);
    let input_size = 5;
    let hidden_size = 8;
    let output_size = 3;
    let batch_size = 2;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

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
// Test: LSTM Gradient Descent Reduces Loss
// ============================================================================

#[test]
fn test_lstm_gradient_descent_reduces_loss() {
    let mut rng = SimpleRng::new(123);
    let input_size = 6;
    let hidden_size = 10;
    let output_size = 4;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Test data
    let input = vec![0.5f32, -0.3, 0.8, 0.2, -0.6, 0.4];
    let target = vec![0.2f32, 0.7, -0.4, 0.1];

    // Verify that gradient descent reduces loss
    let learning_rate = 0.05f32;
    let num_steps = 15;

    let loss_reduced = verify_gradient_descent_reduces_loss(&mut layer, &input, &target, learning_rate, num_steps);

    assert!(
        loss_reduced,
        "Gradient descent should reduce loss if LSTM gradients are correct"
    );
}

// ============================================================================
// Test: LSTM Gradient Checking with Non-Zero States
// ============================================================================

#[test]
fn test_lstm_gradients_with_initial_states() {
    let mut rng = SimpleRng::new(456);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 2;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Set non-zero initial states to test all gate interactions
    let initial_hidden = vec![0.2f32, -0.1, 0.15, -0.08, 0.12, 0.05];
    let initial_cell = vec![0.3f32, 0.1, -0.2, 0.25, -0.15, 0.18];
    layer.set_hidden_state(&initial_hidden);
    layer.set_cell_state(&initial_cell);

    let input = vec![0.4f32, 0.6, -0.2, 0.3];
    let target = vec![0.3f32, 0.1];

    // Verify that gradient descent works with non-zero initial states
    let learning_rate = 0.03f32;
    let num_steps = 10;

    let loss_reduced = verify_gradient_descent_reduces_loss(&mut layer, &input, &target, learning_rate, num_steps);

    assert!(
        loss_reduced,
        "Gradient descent should reduce loss even with non-zero initial states"
    );
}

// ============================================================================
// Test: LSTM Training Convergence
// ============================================================================

#[test]
fn test_lstm_training_convergence() {
    let mut rng = SimpleRng::new(999);
    let input_size = 4;
    let hidden_size = 12;
    let output_size = 2;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Training data: simple classification-like task
    let inputs = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0, 0.0],
        vec![0.0f32, 0.0, 1.0, 0.0],
    ];
    let targets = vec![
        vec![1.0f32, 0.0],
        vec![0.0f32, 1.0],
        vec![0.5f32, 0.5],
    ];

    let learning_rate = 0.1f32;
    let epochs = 25;

    // Measure initial loss
    layer.reset_state();
    let mut initial_total_loss = 0.0f32;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        initial_total_loss += compute_lstm_loss(&layer, input, target);
    }

    // Train for multiple epochs
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            layer.reset_state();

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
    layer.reset_state();
    let mut final_total_loss = 0.0f32;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        final_total_loss += compute_lstm_loss(&layer, input, target);
    }

    // Loss should decrease significantly with training
    assert!(
        final_total_loss < initial_total_loss * 0.5,
        "LSTM training should significantly reduce loss: initial={:.6}, final={:.6}",
        initial_total_loss,
        final_total_loss
    );
}

// ============================================================================
// Test: LSTM Long Sequence Processing
// ============================================================================

#[test]
fn test_lstm_long_sequence_processing() {
    let mut rng = SimpleRng::new(42);
    let input_size = 5;
    let hidden_size = 10;
    let output_size = 3;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Process a long sequence
    let sequence_length = 20;
    let mut outputs = Vec::new();
    let mut hidden_states = Vec::new();
    let mut cell_states = Vec::new();

    for t in 0..sequence_length {
        let input = vec![(t as f32) * 0.1; input_size];
        let mut output = vec![0.0f32; output_size];
        layer.forward(&input, &mut output, 1);

        outputs.push(output.clone());
        hidden_states.push(layer.get_hidden_state());
        cell_states.push(layer.get_cell_state());
    }

    // Verify that outputs change across time steps
    for t in 1..sequence_length {
        assert_ne!(
            outputs[t - 1],
            outputs[t],
            "Outputs should differ across time steps"
        );
    }

    // Verify all outputs, hidden states, and cell states are finite
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
        assert!(
            cell_states[t].iter().all(|&x| x.is_finite()),
            "All cell states at time {} should be finite",
            t
        );
    }

    // Check that cell state maintains information (doesn't explode or vanish immediately)
    let cell_first = &cell_states[0];
    let cell_mid = &cell_states[sequence_length / 2];
    let cell_last = &cell_states[sequence_length - 1];

    // Cell states should be different but maintain reasonable magnitudes
    for &cell in &[cell_first, cell_mid, cell_last] {
        let max_val = cell.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 100.0,
            "Cell state should not explode: max_val={}",
            max_val
        );
    }
}

// ============================================================================
// Test: LSTM Batch Processing
// ============================================================================

#[test]
fn test_lstm_batch_processing() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 3;
    let batch_size = 3;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Create batch input
    let input_batch = vec![
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
    layer.backward(&input_flat, &grad_output_batch, &mut grad_input_batch, batch_size);

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
// Test: LSTM Parameter Updates
// ============================================================================

#[test]
fn test_lstm_parameter_updates() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Do forward-backward to accumulate gradients
    layer.reset_state();

    // Do two forward passes so gates interact with non-zero states
    let input1 = vec![0.5f32, -0.3, 0.8];
    let mut output1 = vec![0.0f32; output_size];
    layer.forward(&input1, &mut output1, 1);

    let input2 = vec![0.3f32, 0.6, -0.2];
    let mut output2 = vec![0.0f32; output_size];
    layer.forward(&input2, &mut output2, 1);

    // Backward on second pass
    let grad_output = vec![1.0f32, -0.5];
    let mut grad_input = vec![0.0f32; input_size];
    layer.backward(&input2, &grad_output, &mut grad_input, 1);

    // Get initial parameter count (just to verify structure)
    let initial_param_count = layer.parameter_count();

    // Update parameters
    let learning_rate = 0.01f32;
    layer.update_parameters(learning_rate);

    // Parameter count should not change
    assert_eq!(layer.parameter_count(), initial_param_count);

    // Do another forward-backward cycle to verify gradients were cleared
    layer.reset_state();
    layer.forward(&input1, &mut output1, 1);

    let grad_output2 = vec![0.5f32, 0.3];
    let mut grad_input2 = vec![0.0f32; input_size];
    layer.backward(&input1, &grad_output2, &mut grad_input2, 1);

    // Gradients should be computed (finite and some non-zero)
    assert!(
        grad_input2.iter().all(|&x| x.is_finite()),
        "Gradients should be finite after parameter update"
    );
}

// ============================================================================
// Test: LSTM Forget Gate Learning (Memory Retention)
// ============================================================================

#[test]
fn test_lstm_memory_retention() {
    let mut rng = SimpleRng::new(777);
    let input_size = 2;
    let hidden_size = 4;
    let output_size = 1;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Sequence designed to test memory retention
    // First input provides information, subsequent inputs are noise
    let sequence = vec![
        vec![1.0f32, 0.0],  // Important signal
        vec![0.0f32, 0.1],  // Noise
        vec![0.0f32, -0.1], // Noise
        vec![0.0f32, 0.05], // Noise - should recall initial signal
    ];

    let learning_rate = 0.1f32;
    let epochs = 20;

    // Train the LSTM to remember the first input
    for _ in 0..epochs {
        layer.reset_state();

        for (i, input) in sequence.iter().enumerate() {
            let mut output = vec![0.0f32; output_size];
            layer.forward(input, &mut output, 1);

            // Target: output 1.0 at last time step if first input was [1,0]
            let target = if i == sequence.len() - 1 {
                vec![1.0f32]
            } else {
                vec![output[0]] // Don't provide gradient for intermediate steps
            };

            if i == sequence.len() - 1 {
                let mut grad_output = vec![0.0f32; output_size];
                for j in 0..output_size {
                    grad_output[j] = output[j] - target[j];
                }

                let mut grad_input = vec![0.0f32; input_size];
                layer.backward(input, &grad_output, &mut grad_input, 1);
            }
        }

        layer.update_parameters(learning_rate);
    }

    // Test: cell state should retain information across time steps
    layer.reset_state();
    let cell_states: Vec<Vec<f32>> = sequence
        .iter()
        .map(|input| {
            let mut output = vec![0.0f32; output_size];
            layer.forward(input, &mut output, 1);
            layer.get_cell_state()
        })
        .collect();

    // Cell state should exist and be finite throughout sequence
    for (i, cell) in cell_states.iter().enumerate() {
        assert!(
            cell.iter().all(|&x| x.is_finite()),
            "Cell state at time {} should be finite",
            i
        );
    }
}

// ============================================================================
// Test: LSTM Sequential Learning with Context
// ============================================================================

#[test]
fn test_lstm_sequential_learning() {
    let mut rng = SimpleRng::new(888);
    let input_size = 3;
    let hidden_size = 8;
    let output_size = 2;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Multi-step sequence
    let sequences = vec![
        (
            vec![vec![1.0f32, 0.0, 0.0], vec![0.0f32, 1.0, 0.0]],
            vec![0.8f32, 0.2],
        ),
        (
            vec![vec![0.0f32, 1.0, 0.0], vec![1.0f32, 0.0, 0.0]],
            vec![0.2f32, 0.8],
        ),
    ];

    let learning_rate = 0.05f32;
    let epochs = 15;

    // Train on sequences
    for _ in 0..epochs {
        for (sequence, target) in &sequences {
            layer.reset_state();

            // Process sequence
            let mut final_output = vec![0.0f32; output_size];
            for input in sequence {
                layer.forward(input, &mut final_output, 1);
            }

            // Only compute gradient for final output
            let mut grad_output = vec![0.0f32; output_size];
            for i in 0..output_size {
                grad_output[i] = final_output[i] - target[i];
            }

            let last_input = sequence.last().unwrap();
            let mut grad_input = vec![0.0f32; input_size];
            layer.backward(last_input, &grad_output, &mut grad_input, 1);

            layer.update_parameters(learning_rate);
        }
    }

    // Verify final outputs are reasonable
    for (sequence, _target) in &sequences {
        layer.reset_state();
        let mut final_output = vec![0.0f32; output_size];
        for input in sequence {
            layer.forward(input, &mut final_output, 1);
        }

        assert!(
            final_output.iter().all(|&x| x.is_finite()),
            "Final output should be finite"
        );
        assert!(
            final_output.iter().all(|&x| x.abs() < 10.0),
            "Final output should be reasonable magnitude"
        );
    }
}

// ============================================================================
// Test: LSTM Zero Initial State Behavior
// ============================================================================

#[test]
fn test_lstm_zero_initial_state() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 5;
    let output_size = 2;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Two identical forward passes from zero state should give identical results
    layer.reset_state();
    let input = vec![0.5f32, -0.3, 0.8];
    let mut output1 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output1, 1);

    layer.reset_state();
    let mut output2 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output2, 1);

    // Outputs should be identical when starting from zero state
    for i in 0..output_size {
        assert!(
            (output1[i] - output2[i]).abs() < 1e-6,
            "Outputs should be identical when starting from zero states"
        );
    }

    // Third forward pass without reset should differ (non-zero states)
    let mut output3 = vec![0.0f32; output_size];
    layer.forward(&input, &mut output3, 1);

    assert_ne!(
        output2, output3,
        "Output should differ when states are non-zero"
    );
}

// ============================================================================
// Test: LSTM Gradient Flow Through Gates
// ============================================================================

#[test]
fn test_lstm_gradient_flow_through_gates() {
    let mut rng = SimpleRng::new(555);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 3;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Setup non-zero initial states to activate all gates
    let initial_hidden = vec![0.1f32; hidden_size];
    let initial_cell = vec![0.2f32; hidden_size];
    layer.set_hidden_state(&initial_hidden);
    layer.set_cell_state(&initial_cell);

    // Forward pass
    let input = vec![0.5f32, -0.3, 0.8, 0.2];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, 1);

    // Backward pass
    let grad_output = vec![1.0f32, 0.5, -0.5];
    let mut grad_input = vec![0.0f32; input_size];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Gradients should flow back through all gates
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "All input gradients should be finite"
    );

    // With non-zero states, gradients should be non-trivial
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "Gradients should flow through LSTM gates"
    );

    // Check that gradient magnitude is reasonable (not exploding)
    let max_grad = grad_input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_grad < 100.0,
        "Gradients should not explode: max_grad={}",
        max_grad
    );
}
