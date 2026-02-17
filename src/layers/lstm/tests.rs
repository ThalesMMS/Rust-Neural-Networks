use super::LstmLayer;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;

#[test]
fn test_lstm_creation() {
    let mut rng = SimpleRng::new(42);
    let layer = LstmLayer::new(64, 128, 10, &mut rng);

    assert_eq!(layer.input_size(), 64);
    assert_eq!(layer.hidden_size(), 128);
    assert_eq!(layer.output_size(), 10);

    // Verify state vectors are initialized to zero
    let hidden = layer.get_hidden_state();
    assert_eq!(hidden.len(), 128);
    assert!(hidden.iter().all(|&x| x == 0.0));

    let cell = layer.get_cell_state();
    assert_eq!(cell.len(), 128);
    assert!(cell.iter().all(|&x| x == 0.0));
}

#[test]
fn test_lstm_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let layer = LstmLayer::new(64, 128, 10, &mut rng);

    // For each gate: input_size × hidden_size + hidden_size × hidden_size + hidden_size
    let gate_params = 64 * 128 + 128 * 128 + 128;
    // 4 gates + output projection (hidden_size × output_size + output_size)
    let expected = 4 * gate_params + (128 * 10 + 10);

    assert_eq!(layer.parameter_count(), expected);
}

#[test]
fn test_lstm_state_management() {
    let mut rng = SimpleRng::new(42);
    let layer = LstmLayer::new(32, 64, 5, &mut rng);

    // Test setting and getting hidden state
    let test_hidden = vec![0.5f32; 64];
    layer.set_hidden_state(&test_hidden);
    let retrieved_hidden = layer.get_hidden_state();
    assert_eq!(retrieved_hidden, test_hidden);

    // Test setting and getting cell state
    let test_cell = vec![0.3f32; 64];
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

#[test]
#[should_panic(expected = "Hidden state length must match hidden_size")]
fn test_lstm_invalid_hidden_state_length() {
    let mut rng = SimpleRng::new(42);
    let layer = LstmLayer::new(32, 64, 5, &mut rng);
    layer.set_hidden_state(&[0.0f32; 32]); // Wrong size
}

#[test]
#[should_panic(expected = "Cell state length must match hidden_size")]
fn test_lstm_invalid_cell_state_length() {
    let mut rng = SimpleRng::new(42);
    let layer = LstmLayer::new(32, 64, 5, &mut rng);
    layer.set_cell_state(&[0.0f32; 32]); // Wrong size
}

#[test]
fn test_lstm_forward() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Reset states before starting
    layer.reset_state();

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

    // Check cell state was updated
    let cell_state = layer.get_cell_state();
    assert_eq!(cell_state.len(), hidden_size);
    assert!(
        cell_state.iter().any(|&x| x != 0.0),
        "Cell state should be updated"
    );

    // Forward pass at time step 1 with different input
    let input_t1 = vec![0.0, 1.0, 0.0];
    let mut output_t1 = vec![0.0; output_size];

    layer.forward(&input_t1, &mut output_t1, batch_size);

    // Output should be different from first time step (due to hidden/cell state)
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

    // Cell state should have changed
    let cell_state_t1 = layer.get_cell_state();
    assert_ne!(
        cell_state, cell_state_t1,
        "Cell state should change between time steps"
    );

    // Verify all outputs are finite (no NaN or Inf)
    assert!(
        output_t1.iter().all(|&x| x.is_finite()),
        "All outputs should be finite"
    );
    assert!(
        hidden_state_t1.iter().all(|&x| x.is_finite()),
        "All hidden state values should be finite"
    );
    assert!(
        cell_state_t1.iter().all(|&x| x.is_finite()),
        "All cell state values should be finite"
    );
}

#[test]
fn test_lstm_backward() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Reset states before starting
    layer.reset_state();

    // Do two forward passes so that h_{t-1} and c_{t-1} are non-zero for the second pass
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

    // Check that gate weight gradients were accumulated
    let grad_w_xf = layer.grad_w_xf.borrow();
    assert!(
        grad_w_xf.iter().any(|&x| x.abs() > 1e-10),
        "W_xf gradients should be accumulated"
    );

    let grad_w_xi = layer.grad_w_xi.borrow();
    assert!(
        grad_w_xi.iter().any(|&x| x.abs() > 1e-10),
        "W_xi gradients should be accumulated"
    );

    let grad_w_xc = layer.grad_w_xc.borrow();
    assert!(
        grad_w_xc.iter().any(|&x| x.abs() > 1e-10),
        "W_xc gradients should be accumulated"
    );

    let grad_w_xo = layer.grad_w_xo.borrow();
    assert!(
        grad_w_xo.iter().any(|&x| x.abs() > 1e-10),
        "W_xo gradients should be accumulated"
    );

    let grad_w_hy = layer.grad_w_hy.borrow();
    assert!(
        grad_w_hy.iter().any(|&x| x.abs() > 1e-10),
        "W_hy gradients should be accumulated"
    );
}

#[test]
fn test_lstm_backward_bptt_returns_state_gradients() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Warm-up pass so h_{t-1} and c_{t-1} are non-zero
    let input_t0 = vec![1.0f32, 0.5, -0.5];
    let mut output_t0 = vec![0.0f32; output_size];
    layer.forward(&input_t0, &mut output_t0, batch_size);

    // Forward pass whose gradients we want
    let input = vec![0.5f32, 1.0, 0.0];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, batch_size);

    // backward_bptt with zero incoming state gradients (last time step scenario)
    let dh_next = vec![0.0f32; hidden_size];
    let dc_next = vec![0.0f32; hidden_size];
    let grad_output = vec![1.0f32, -1.0];
    let mut grad_input = vec![0.0f32; input_size];

    let (dh_prev, dc_prev) = layer.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input,
        &dh_next,
        &dc_next,
        batch_size,
    );

    // Returned state gradients must have correct size
    assert_eq!(
        dh_prev.len(),
        hidden_size,
        "dh_prev must have length hidden_size"
    );
    assert_eq!(
        dc_prev.len(),
        hidden_size,
        "dc_prev must have length hidden_size"
    );

    // All values must be finite
    assert!(
        dh_prev.iter().all(|&x| x.is_finite()),
        "dh_prev values must be finite"
    );
    assert!(
        dc_prev.iter().all(|&x| x.is_finite()),
        "dc_prev values must be finite"
    );

    // At least some state gradients must be non-zero (gradient flows back through states)
    assert!(
        dh_prev.iter().any(|&x| x.abs() > 1e-10),
        "dh_prev should have non-zero values indicating gradient flow"
    );
    assert!(
        dc_prev.iter().any(|&x| x.abs() > 1e-10),
        "dc_prev should have non-zero values indicating gradient flow"
    );
}

#[test]
fn test_lstm_backward_bptt_non_zero_incoming_changes_gradients() {
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    // ---- First run: zero incoming state gradients ----
    let layer_a = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer_a.reset_state();

    let mut rng_b = SimpleRng::new(42);
    let layer_b = LstmLayer::new(input_size, hidden_size, output_size, &mut rng_b);
    layer_b.reset_state();

    // Same warm-up for both
    let input_t0 = vec![1.0f32, 0.5, -0.5];
    let mut out_a = vec![0.0f32; output_size];
    let mut out_b = vec![0.0f32; output_size];
    layer_a.forward(&input_t0, &mut out_a, batch_size);
    layer_b.forward(&input_t0, &mut out_b, batch_size);

    // Same forward pass
    let input = vec![0.5f32, 1.0, 0.0];
    let mut output_a = vec![0.0f32; output_size];
    let mut output_b = vec![0.0f32; output_size];
    layer_a.forward(&input, &mut output_a, batch_size);
    layer_b.forward(&input, &mut output_b, batch_size);

    let grad_output = vec![1.0f32, -1.0];
    let mut grad_input_a = vec![0.0f32; input_size];
    let mut grad_input_b = vec![0.0f32; input_size];

    // Run A with zero incoming gradients
    let dh_zero = vec![0.0f32; hidden_size];
    let dc_zero = vec![0.0f32; hidden_size];
    let (dh_prev_zero, dc_prev_zero) = layer_a.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input_a,
        &dh_zero,
        &dc_zero,
        batch_size,
    );

    // Run B with non-zero incoming gradients
    let dh_nonzero = vec![0.5f32; hidden_size];
    let dc_nonzero = vec![0.3f32; hidden_size];
    let (dh_prev_nonzero, dc_prev_nonzero) = layer_b.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input_b,
        &dh_nonzero,
        &dc_nonzero,
        batch_size,
    );

    // Non-zero incoming gradients should produce different (larger magnitude) returned gradients
    assert_ne!(
        dh_prev_zero, dh_prev_nonzero,
        "Non-zero dh_next should change the returned dh_prev"
    );
    assert_ne!(
        dc_prev_zero, dc_prev_nonzero,
        "Non-zero dc_next should change the returned dc_prev"
    );
}

#[test]
fn test_lstm_backward_bptt_accumulates_weight_gradients() {
    // Verify that multiple backward_bptt calls accumulate (add to) weight gradients
    // rather than replacing them.
    let mut rng = SimpleRng::new(42);
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;
    let batch_size = 1;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Warm-up pass so h and c states are non-zero
    let input_t0 = vec![1.0f32, 0.5, -0.5];
    let mut output_t0 = vec![0.0f32; output_size];
    layer.forward(&input_t0, &mut output_t0, batch_size);

    // First forward + backward_bptt pass
    let input_t1 = vec![0.5f32, 1.0, 0.0];
    let mut output_t1 = vec![0.0f32; output_size];
    layer.forward(&input_t1, &mut output_t1, batch_size);

    let dh_zero = vec![0.0f32; hidden_size];
    let dc_zero = vec![0.0f32; hidden_size];
    let grad_output_t1 = vec![1.0f32, -1.0];
    let mut grad_input_t1 = vec![0.0f32; input_size];
    layer.backward_bptt(
        &input_t1,
        &grad_output_t1,
        &mut grad_input_t1,
        &dh_zero,
        &dc_zero,
        batch_size,
    );

    // W_hy depends on h_t (always non-zero after forward) – verify it is accumulated
    let grad_w_hy_after_first: Vec<f32> = layer.grad_w_hy.borrow().clone();
    assert!(
        grad_w_hy_after_first.iter().any(|&x| x.abs() > 1e-10),
        "W_hy gradients should be accumulated after first backward_bptt"
    );

    // b_y is also always non-zero when grad_output is non-zero
    let grad_b_y_after_first: Vec<f32> = layer.grad_b_y.borrow().clone();
    assert!(
        grad_b_y_after_first.iter().any(|&x| x.abs() > 1e-10),
        "b_y gradients should be accumulated after first backward_bptt"
    );

    // Second forward + backward_bptt pass continuing from the current state (no reset)
    let input_t2 = vec![0.2f32, 0.8, 0.5];
    let mut output_t2 = vec![0.0f32; output_size];
    layer.forward(&input_t2, &mut output_t2, batch_size);

    let grad_output_t2 = vec![0.5f32, 0.5];
    let mut grad_input_t2 = vec![0.0f32; input_size];
    layer.backward_bptt(
        &input_t2,
        &grad_output_t2,
        &mut grad_input_t2,
        &dh_zero,
        &dc_zero,
        batch_size,
    );

    // Gradients must have changed (second call accumulated on top of first)
    let grad_w_hy_after_second: Vec<f32> = layer.grad_w_hy.borrow().clone();
    assert_ne!(
        grad_w_hy_after_first, grad_w_hy_after_second,
        "W_hy gradients should accumulate across multiple backward_bptt calls"
    );

    let grad_b_y_after_second: Vec<f32> = layer.grad_b_y.borrow().clone();
    assert_ne!(
        grad_b_y_after_first, grad_b_y_after_second,
        "b_y gradients should accumulate across multiple backward_bptt calls"
    );
}

#[test]
fn test_lstm_backward_bptt_sequence_full_pass() {
    // Full forward + BPTT backward through a sequence of time steps.
    // Verifies that gradient propagation chains correctly through the sequence.
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 4;
    let batch_size = 1;
    let seq_len = 3;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Build a small sequence of inputs and run the full forward pass
    let inputs: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| vec![(t as f32 + 1.0) * 0.1; input_size])
        .collect();
    let mut outputs: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![0.0f32; output_size]).collect();

    for t in 0..seq_len {
        layer.forward(&inputs[t], &mut outputs[t], batch_size);
    }

    // Backward pass in reverse order, threading state gradients through time
    let mut dh = vec![0.0f32; hidden_size]; // zero for last time step
    let mut dc = vec![0.0f32; hidden_size];

    for t in (0..seq_len).rev() {
        let grad_out = vec![1.0f32; output_size];
        let mut grad_in = vec![0.0f32; input_size];

        // Re-run the forward up to time step t to restore cached values
        layer.reset_state();
        for k in 0..=t {
            let mut dummy_out = vec![0.0f32; output_size];
            layer.forward(&inputs[k], &mut dummy_out, batch_size);
        }

        (dh, dc) = layer.backward_bptt(&inputs[t], &grad_out, &mut grad_in, &dh, &dc, batch_size);

        // Every returned state gradient must be finite
        assert!(
            dh.iter().all(|&x| x.is_finite()),
            "dh at t={t} must be finite"
        );
        assert!(
            dc.iter().all(|&x| x.is_finite()),
            "dc at t={t} must be finite"
        );

        // Input gradients must also be finite
        assert!(
            grad_in.iter().all(|&x| x.is_finite()),
            "grad_in at t={t} must be finite"
        );
    }

    // After processing all time steps, at least one state gradient must be non-zero
    assert!(
        dh.iter().any(|&x| x.abs() > 1e-10) || dc.iter().any(|&x| x.abs() > 1e-10),
        "Final BPTT state gradients should be non-zero after propagating through the sequence"
    );

    // Weight gradients must have been accumulated over the entire sequence
    {
        let grad_w_xi = layer.grad_w_xi.borrow();
        assert!(
            grad_w_xi.iter().any(|&x| x.abs() > 1e-10),
            "W_xi gradients should be accumulated over the full BPTT sequence"
        );
    }
    {
        let grad_b_y = layer.grad_b_y.borrow();
        assert!(
            grad_b_y.iter().any(|&x| x.abs() > 1e-10),
            "b_y gradients should be accumulated over the full BPTT sequence"
        );
    }
}

#[test]
fn test_lstm_layer_trait() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;
    let batch_size = 2;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    // Test input_size() method
    assert_eq!(layer.input_size(), input_size);

    // Test output_size() method
    assert_eq!(layer.output_size(), output_size);

    // Test parameter_count() method
    // For each of 4 gates: input_size × hidden_size + hidden_size × hidden_size + hidden_size
    let gate_params = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
    // Output projection: hidden_size × output_size + output_size
    let output_params = hidden_size * output_size + output_size;
    let expected_params = 4 * gate_params + output_params;
    assert_eq!(layer.parameter_count(), expected_params);

    // Reset states
    layer.reset_state();

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

    // Verify at least some input gradients are non-zero
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "At least some input gradients should be non-zero"
    );

    // Test update_parameters() method
    let learning_rate = 0.01;

    // Call update parameters (this should clear gradients even if parameters don't change much)
    layer.update_parameters(learning_rate);

    // Verify gradients were cleared after update
    {
        let grad_w_xf = layer.grad_w_xf.borrow();
        assert!(
            grad_w_xf.iter().all(|&x| x == 0.0),
            "W_xf gradients should be cleared after update_parameters()"
        );
    }
    {
        let grad_w_hy = layer.grad_w_hy.borrow();
        assert!(
            grad_w_hy.iter().all(|&x| x == 0.0),
            "W_hy gradients should be cleared after update_parameters()"
        );
    }
    {
        let grad_b_y = layer.grad_b_y.borrow();
        assert!(
            grad_b_y.iter().all(|&x| x == 0.0),
            "b_y gradients should be cleared after update_parameters()"
        );
    }
    {
        let grad_w_xi = layer.grad_w_xi.borrow();
        assert!(
            grad_w_xi.iter().all(|&x| x == 0.0),
            "All gradients should be cleared after update_parameters()"
        );
    }
}
