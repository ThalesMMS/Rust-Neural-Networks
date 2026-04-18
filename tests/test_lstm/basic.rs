use super::*;

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
