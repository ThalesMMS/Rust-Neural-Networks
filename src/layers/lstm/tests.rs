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
