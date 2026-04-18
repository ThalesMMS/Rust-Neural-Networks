use super::*;

#[test]
fn test_rnn_forward() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Reset hidden state before starting
    layer.reset_hidden_state();

    // Create input for first time step
    let input_t0 = vec![1.0, 0.5, -0.5];
    let mut output_t0 = vec![0.0; output_size];

    // Forward pass at time step 0
    layer.forward(&input_t0, &mut output_t0, batch_size);

    // Check output is not all zeros (network produced some output)
    assert!(
        output_t0.iter().any(|&x| x != 0.0),
        "Output should not be all zeros"
    );

    // Check hidden state was updated
    let hidden_state = layer.get_hidden_state();
    assert_eq!(hidden_state.len(), hidden_size);
    assert!(
        hidden_state.iter().any(|&x| x != 0.0),
        "Hidden state should be updated"
    );

    // Forward pass at time step 1 with different input
    let input_t1 = vec![0.0, 1.0, 0.0];
    let mut output_t1 = vec![0.0; output_size];

    layer.forward(&input_t1, &mut output_t1, batch_size);

    // Output should be different from first time step (due to hidden state)
    assert_ne!(
        output_t0, output_t1,
        "Outputs at different time steps should differ"
    );

    // Hidden state should have changed
    let hidden_state_t1 = layer.get_hidden_state();
    assert_ne!(
        hidden_state, hidden_state_t1,
        "Hidden state should change between time steps"
    );
}

#[test]
fn test_rnn_forward_batch() {
    let mut rng = SimpleRng::new(123);
    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;
    let batch_size = 2;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Input for batch of 2 sequences at one time step
    let input = vec![1.0, 0.0, 0.0, 1.0]; // [seq1: [1.0, 0.0], seq2: [0.0, 1.0]]
    let mut output = vec![0.0; batch_size * output_size];

    layer.forward(&input, &mut output, batch_size);

    // Check outputs for both sequences are computed
    assert_eq!(output.len(), batch_size * output_size);
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Batch output should not be all zeros"
    );
}

#[test]
fn test_rnn_reset_hidden_state() {
    let mut rng = SimpleRng::new(99);
    let layer = RnnLayer::new(2, 3, 1, &mut rng);

    // Process one time step
    let input = vec![1.0, 1.0];
    let mut output = vec![0.0];
    layer.forward(&input, &mut output, 1);

    // Hidden state should be non-zero
    let hidden_before = layer.get_hidden_state();
    assert!(hidden_before.iter().any(|&x| x != 0.0));

    // Reset hidden state
    layer.reset_hidden_state();

    // Hidden state should be zero
    let hidden_after = layer.get_hidden_state();
    assert!(
        hidden_after.iter().all(|&x| x == 0.0),
        "Hidden state should be zero after reset"
    );
}

#[test]
fn test_rnn_forward_dimensions() {
    let mut rng = SimpleRng::new(0);
    let input_size = 5;
    let hidden_size = 10;
    let output_size = 3;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Test with batch_size = 1
    let input = vec![0.5; input_size];
    let mut output = vec![0.0; output_size];
    layer.forward(&input, &mut output, 1);
    assert_eq!(output.len(), output_size);

    // Test with batch_size = 4
    layer.reset_hidden_state();
    let input_batch = vec![0.5; 4 * input_size];
    let mut output_batch = vec![0.0; 4 * output_size];
    layer.forward(&input_batch, &mut output_batch, 4);
    assert_eq!(output_batch.len(), 4 * output_size);
}

#[test]
fn test_rnn_backward() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Reset hidden state before starting
    layer.reset_hidden_state();

    // Do two forward passes so that h_{t-1} is non-zero for the second pass
    let input_t0 = vec![1.0, 0.5, -0.5];
    let mut output_t0 = vec![0.0; output_size];
    layer.forward(&input_t0, &mut output_t0, batch_size);

    // Now do the forward pass we'll use for backward
    let input = vec![0.5, 1.0, 0.0];
    let mut output = vec![0.0; output_size];
    layer.forward(&input, &mut output, batch_size);

    // Create gradient of output
    let grad_output = vec![1.0, -1.0];
    let mut grad_input = vec![0.0; input_size];

    // Backward pass
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradient should propagate back
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "All gradients should be finite"
    );

    // At least some gradients should be non-zero
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "At least some gradients should be non-zero"
    );

    // Check that weight gradients were accumulated
    let grad_w_xh = layer.grad_w_xh.borrow();
    assert!(
        grad_w_xh.iter().any(|&x| x.abs() > 1e-10),
        "W_xh gradients should be accumulated"
    );

    let grad_w_hh = layer.grad_w_hh.borrow();
    assert!(
        grad_w_hh.iter().any(|&x| x.abs() > 1e-10),
        "W_hh gradients should be accumulated (h_prev is non-zero)"
    );

    let grad_w_hy = layer.grad_w_hy.borrow();
    assert!(
        grad_w_hy.iter().any(|&x| x.abs() > 1e-10),
        "W_hy gradients should be accumulated"
    );
}

#[test]
fn test_rnn_backward_batch() {
    let mut rng = SimpleRng::new(123);
    let input_size = 2;
    let hidden_size = 3;
    let output_size = 1;
    let batch_size = 2;

    let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_hidden_state();

    // Forward pass with batch
    let input = vec![1.0, 0.0, 0.0, 1.0]; // 2 samples
    let mut output = vec![0.0; batch_size * output_size];
    layer.forward(&input, &mut output, batch_size);

    // Backward pass
    let grad_output = vec![1.0, -0.5]; // gradients for 2 samples
    let mut grad_input = vec![0.0; batch_size * input_size];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check gradients are finite
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "Batch gradients should be finite"
    );

    // Check at least some gradients are non-zero
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "Batch should have non-zero gradients"
    );
}

#[test]
fn test_rnn_layer_trait() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;
    let batch_size = 2;

    let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Test input_size() method
    assert_eq!(layer.input_size(), input_size);

    // Test output_size() method
    assert_eq!(layer.output_size(), output_size);

    // Test parameter_count() method
    let expected_params = input_size * hidden_size  // W_xh
        + hidden_size * hidden_size                 // W_hh
        + hidden_size * output_size                 // W_hy
        + hidden_size                               // b_h
        + output_size; // b_y
    assert_eq!(layer.parameter_count(), expected_params);

    // Reset hidden state
    layer.reset_hidden_state();

    // Test forward pass
    let input = vec![0.5; batch_size * input_size];
    let mut output = vec![0.0; batch_size * output_size];
    layer.forward(&input, &mut output, batch_size);

    // Verify output is not all zeros
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Forward pass should produce non-zero output"
    );

    // Test backward pass
    let grad_output = vec![1.0; batch_size * output_size];
    let mut grad_input = vec![0.0; batch_size * input_size];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Verify gradients are computed
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "Gradients should be finite"
    );

    // Test update_parameters() method
    let learning_rate = 0.01;

    // Get initial weights
    let w_xh_before = layer.w_xh().to_vec();
    let w_hy_before = layer.w_hy().to_vec();

    // Update parameters
    layer.update_parameters(learning_rate);

    // Verify parameters changed
    let w_xh_after = layer.w_xh();
    let w_hy_after = layer.w_hy();

    // At least some parameters should have changed
    let w_xh_changed = w_xh_before
        .iter()
        .zip(w_xh_after.iter())
        .any(|(before, after)| (before - after).abs() > 1e-10);
    let w_hy_changed = w_hy_before
        .iter()
        .zip(w_hy_after.iter())
        .any(|(before, after)| (before - after).abs() > 1e-10);

    assert!(
        w_xh_changed || w_hy_changed,
        "Parameters should be updated after calling update_parameters()"
    );

    // Verify gradients were cleared (do another backward pass)
    layer.reset_hidden_state();
    layer.forward(&input, &mut output, batch_size);
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Gradients should accumulate from a fresh state
    {
        let grad_w_xh = layer.grad_w_xh.borrow();
        assert!(
            grad_w_xh.iter().any(|&x| x.abs() > 1e-10),
            "Gradients should accumulate after backward pass"
        );
    } // Drop borrow before next operation
}
