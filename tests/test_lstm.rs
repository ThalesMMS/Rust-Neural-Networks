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

#[path = "test_lstm/backward_training.rs"]
mod backward_training;
#[path = "test_lstm/basic.rs"]
mod basic;
#[path = "test_lstm/bptt.rs"]
mod bptt;
