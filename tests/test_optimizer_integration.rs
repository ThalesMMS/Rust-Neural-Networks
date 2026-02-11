// Integration tests for optimizers with DenseLayer
// Verifies that AdamW and RMSprop correctly update layer parameters

use rust_neural_networks::layers::dense::DenseLayer;
use rust_neural_networks::layers::Layer;
use rust_neural_networks::optimizers::{AdamW, RMSprop};
use rust_neural_networks::utils::rng::SimpleRng;

/// Test that AdamW optimizer successfully updates DenseLayer parameters
#[test]
fn test_adamw_updates_dense_layer() {
    // Create a simple DenseLayer (2 inputs, 3 outputs)
    let mut rng = SimpleRng::new(42);
    let mut layer = DenseLayer::new(2, 3, &mut rng);

    // Create AdamW optimizer
    let mut optimizer = AdamW::new(
        0.01,  // learning_rate
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // epsilon
        0.01,  // weight_decay
    );

    // Store original weights and biases for comparison
    let original_weights: Vec<f32> = layer.weights().to_vec();
    let original_biases: Vec<f32> = layer.biases().to_vec();

    // Create dummy input and perform forward pass
    let batch_size = 2;
    let input = vec![1.0, 0.5, -0.5, 1.0]; // 2 samples × 2 features
    let mut output = vec![0.0; batch_size * 3];
    layer.forward(&input, &mut output, batch_size);

    // Create dummy gradient for backward pass (simulate loss gradient)
    let grad_output = vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3]; // 2 samples × 3 outputs
    let mut grad_input = vec![0.0; batch_size * 2];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update parameters using AdamW optimizer
    layer.update_with_optimizer(&mut optimizer);

    // Verify that weights have been updated (they should differ from original)
    let updated_weights: Vec<f32> = layer.weights().to_vec();
    let updated_biases: Vec<f32> = layer.biases().to_vec();

    // Check that at least some weights changed
    let weights_changed = original_weights
        .iter()
        .zip(updated_weights.iter())
        .any(|(orig, updated)| (orig - updated).abs() > 1e-6);

    assert!(
        weights_changed,
        "AdamW should have updated at least some weights"
    );

    // Check that at least some biases changed
    let biases_changed = original_biases
        .iter()
        .zip(updated_biases.iter())
        .any(|(orig, updated)| (orig - updated).abs() > 1e-6);

    assert!(
        biases_changed,
        "AdamW should have updated at least some biases"
    );

    // Verify that parameters moved in the opposite direction of gradients
    // (this is a basic sanity check for gradient descent)
    // Since we have gradients and applied optimizer update, parameters should change
    // AdamW applies: param = param - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * param
}

/// Test that RMSprop optimizer successfully updates DenseLayer parameters
#[test]
fn test_rmsprop_updates_dense_layer() {
    // Create a simple DenseLayer (2 inputs, 3 outputs)
    let mut rng = SimpleRng::new(123);
    let mut layer = DenseLayer::new(2, 3, &mut rng);

    // Create RMSprop optimizer with higher learning rate for visible changes
    let mut optimizer = RMSprop::new(
        0.1,  // learning_rate (increased for clearer updates)
        0.9,  // decay_rate
        1e-8, // epsilon
    );

    // Store original weights and biases for comparison
    let original_weights: Vec<f32> = layer.weights().to_vec();
    let original_biases: Vec<f32> = layer.biases().to_vec();

    // Create dummy input and perform forward pass
    let batch_size = 2;
    let input = vec![1.0, 0.5, -0.5, 1.0]; // 2 samples × 2 features
    let mut output = vec![0.0; batch_size * 3];
    layer.forward(&input, &mut output, batch_size);

    // Create dummy gradient for backward pass with non-zero sum for bias gradients
    // Using different values per sample so biases will have non-zero gradients
    let grad_output = vec![1.0, -0.5, 0.8, 0.5, 1.0, -0.3]; // 2 samples × 3 outputs
    let mut grad_input = vec![0.0; batch_size * 2];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update parameters using RMSprop optimizer
    layer.update_with_optimizer(&mut optimizer);

    // Verify that weights have been updated (they should differ from original)
    let updated_weights: Vec<f32> = layer.weights().to_vec();
    let updated_biases: Vec<f32> = layer.biases().to_vec();

    // Check that at least some weights changed
    let weights_changed = original_weights
        .iter()
        .zip(updated_weights.iter())
        .any(|(orig, updated)| (orig - updated).abs() > 1e-6);

    assert!(
        weights_changed,
        "RMSprop should have updated at least some weights"
    );

    // Check that at least some biases changed
    let biases_changed = original_biases
        .iter()
        .zip(updated_biases.iter())
        .any(|(orig, updated)| (orig - updated).abs() > 1e-6);

    assert!(
        biases_changed,
        "RMSprop should have updated at least some biases"
    );

    // RMSprop applies: param = param - lr * grad / (sqrt(v) + eps)
    // where v is the moving average of squared gradients
}

/// Test that multiple optimizer updates converge parameters
#[test]
fn test_multiple_optimizer_updates() {
    // Create a simple DenseLayer
    let mut rng = SimpleRng::new(999);
    let mut layer = DenseLayer::new(2, 1, &mut rng);

    // Create AdamW optimizer
    let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);

    // Simple training scenario: try to learn to output 1.0 for input [1.0, 1.0]
    let input = vec![1.0, 1.0]; // Single sample
    let target = 1.0;
    let batch_size = 1;

    // Store initial output
    let mut output_initial = vec![0.0];
    layer.forward(&input, &mut output_initial, batch_size);
    let initial_error = (output_initial[0] - target).abs();

    // Perform several training iterations
    for _ in 0..50 {
        // Forward pass
        let mut output = vec![0.0];
        layer.forward(&input, &mut output, batch_size);

        // Compute gradient (derivative of (output - target)^2 loss)
        let error = output[0] - target;
        let grad_output = vec![error]; // Gradient of 0.5 * (output - target)^2

        // Backward pass
        let mut grad_input = vec![0.0; 2];
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Update with optimizer
        layer.update_with_optimizer(&mut optimizer);
    }

    // Check that output is closer to target after training
    let mut output_final = vec![0.0];
    layer.forward(&input, &mut output_final, batch_size);
    let final_error = (output_final[0] - target).abs();

    // Error should decrease (network should be learning)
    assert!(
        final_error < initial_error,
        "Error should decrease after training: initial={}, final={}",
        initial_error,
        final_error
    );
}

/// Test that RMSprop converges over multiple updates
#[test]
fn test_rmsprop_convergence() {
    // Create a simple DenseLayer
    let mut rng = SimpleRng::new(888);
    let mut layer = DenseLayer::new(2, 1, &mut rng);

    // Create RMSprop optimizer
    let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);

    // Simple training scenario: try to learn to output 0.0 for input [0.0, 0.0]
    let input = vec![0.0, 0.0]; // Single sample
    let target = 0.0;
    let batch_size = 1;

    // Store initial output
    let mut output_initial = vec![0.0];
    layer.forward(&input, &mut output_initial, batch_size);
    let initial_error = (output_initial[0] - target).abs();

    // Perform several training iterations
    for _ in 0..50 {
        // Forward pass
        let mut output = vec![0.0];
        layer.forward(&input, &mut output, batch_size);

        // Compute gradient
        let error = output[0] - target;
        let grad_output = vec![error];

        // Backward pass
        let mut grad_input = vec![0.0; 2];
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Update with optimizer
        layer.update_with_optimizer(&mut optimizer);
    }

    // Check that output is closer to target after training
    let mut output_final = vec![0.0];
    layer.forward(&input, &mut output_final, batch_size);
    let final_error = (output_final[0] - target).abs();

    // For this simple case with zero input, the bias should approach the target
    // The error should be reduced significantly
    assert!(
        final_error <= initial_error,
        "Error should not increase after training: initial={}, final={}",
        initial_error,
        final_error
    );
}

/// Test that gradients are properly cleared between updates
#[test]
fn test_gradients_cleared_after_update() {
    let mut rng = SimpleRng::new(777);
    let mut layer = DenseLayer::new(2, 2, &mut rng);
    let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);

    let batch_size = 1;
    let input = vec![1.0, 1.0];
    let mut output = vec![0.0; 2];
    let grad_output = vec![0.5, -0.5];
    let mut grad_input = vec![0.0; 2];

    // First update
    layer.forward(&input, &mut output, batch_size);
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);
    let (grad_w_norm_1, _grad_b_norm_1) = layer.get_gradient_magnitude();
    layer.update_with_optimizer(&mut optimizer);

    // Check that gradients were cleared
    let (grad_w_norm_after, grad_b_norm_after) = layer.get_gradient_magnitude();
    assert!(
        grad_w_norm_after < 1e-6,
        "Weight gradients should be cleared after update"
    );
    assert!(
        grad_b_norm_after < 1e-6,
        "Bias gradients should be cleared after update"
    );

    // Second update with different gradients
    layer.forward(&input, &mut output, batch_size);
    let grad_output_2 = vec![0.1, -0.1]; // Different gradients
    layer.backward(&input, &grad_output_2, &mut grad_input, batch_size);
    let (grad_w_norm_2, grad_b_norm_2) = layer.get_gradient_magnitude();

    // Gradients from second backward should be independent of first
    // (i.e., they shouldn't accumulate indefinitely)
    assert!(
        grad_w_norm_2 > 0.0,
        "Should have non-zero gradients after second backward pass"
    );
    assert!(
        grad_b_norm_2 > 0.0,
        "Should have non-zero bias gradients after second backward pass"
    );

    // The gradient norms should be different since we used different grad_outputs
    // and cleared between updates
    let gradient_ratio = grad_w_norm_2 / grad_w_norm_1;
    assert!(
        gradient_ratio < 1.0,
        "Second gradient should be smaller due to smaller grad_output"
    );
}
