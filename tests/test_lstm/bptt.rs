use super::*;

// ============================================================================
// BPTT Helper: Compute total MSE loss over a sequence
// ============================================================================

/// Compute total MSE loss over a sequence of time steps.
/// Loss = sum_t 0.5 * sum_i (target_t[i] - output_t[i])^2
fn compute_sequence_loss(layer: &LstmLayer, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    layer.reset_state();
    let output_size = layer.output_size();
    let mut total_loss = 0.0f32;

    for (input, target) in inputs.iter().zip(targets.iter()) {
        let mut output = vec![0.0f32; output_size];
        layer.forward(input, &mut output, 1);
        for i in 0..output_size {
            let err = output[i] - target[i];
            total_loss += 0.5 * err * err;
        }
    }
    total_loss
}

// ============================================================================
// Test: BPTT - Return values are correct shape and finite
// ============================================================================

#[test]
fn test_bptt_return_values_are_finite() {
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);
    layer.reset_state();

    // Single forward pass
    let input = vec![0.5f32, -0.3, 0.8, 0.2];
    let mut output = vec![0.0f32; output_size];
    layer.forward(&input, &mut output, 1);

    // backward_bptt with zero incoming state gradients (last time step)
    let dh_next = vec![0.0f32; hidden_size];
    let dc_next = vec![0.0f32; hidden_size];
    let grad_output = vec![1.0f32, -0.5, 0.3];
    let mut grad_input = vec![0.0f32; input_size];

    let (dh_prev, dc_prev) =
        layer.backward_bptt(&input, &grad_output, &mut grad_input, &dh_next, &dc_next, 1);

    // Returned state gradient vectors should have hidden_size elements
    assert_eq!(
        dh_prev.len(),
        hidden_size,
        "dh_prev should have hidden_size elements"
    );
    assert_eq!(
        dc_prev.len(),
        hidden_size,
        "dc_prev should have hidden_size elements"
    );

    // All returned values should be finite
    assert!(
        dh_prev.iter().all(|&x| x.is_finite()),
        "dh_prev should be finite"
    );
    assert!(
        dc_prev.iter().all(|&x| x.is_finite()),
        "dc_prev should be finite"
    );
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "grad_input should be finite after backward_bptt"
    );

    // With a non-trivial loss, state gradients should be non-zero
    assert!(
        dh_prev.iter().any(|&x| x.abs() > 1e-10),
        "dh_prev should be non-zero for non-trivial grad_output"
    );
    assert!(
        dc_prev.iter().any(|&x| x.abs() > 1e-10),
        "dc_prev should be non-zero for non-trivial grad_output"
    );
}

// ============================================================================
// Test: BPTT - Zero incoming state gradients match regular backward
// ============================================================================

#[test]
fn test_bptt_zero_incoming_matches_backward() {
    // When dh_next=0 and dc_next=0, backward_bptt should compute the same
    // grad_input as the regular backward pass.
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    let input = vec![0.5f32, -0.3, 0.8, 0.2];
    let grad_output = vec![0.4f32, -0.2, 0.6];

    // Layer A: use regular backward
    let mut rng_a = SimpleRng::new(42);
    let layer_a = LstmLayer::new(input_size, hidden_size, output_size, &mut rng_a);
    layer_a.reset_state();
    let mut out_a = vec![0.0f32; output_size];
    layer_a.forward(&input, &mut out_a, 1);
    let mut grad_input_a = vec![0.0f32; input_size];
    layer_a.backward(&input, &grad_output, &mut grad_input_a, 1);

    // Layer B: use backward_bptt with zero state gradients
    let mut rng_b = SimpleRng::new(42); // same seed → identical weights
    let layer_b = LstmLayer::new(input_size, hidden_size, output_size, &mut rng_b);
    layer_b.reset_state();
    let mut out_b = vec![0.0f32; output_size];
    layer_b.forward(&input, &mut out_b, 1);
    let dh_zero = vec![0.0f32; hidden_size];
    let dc_zero = vec![0.0f32; hidden_size];
    let mut grad_input_b = vec![0.0f32; input_size];
    let (_, _) = layer_b.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input_b,
        &dh_zero,
        &dc_zero,
        1,
    );

    // grad_input should be equal when incoming state gradients are zero
    for i in 0..input_size {
        assert!(
            (grad_input_a[i] - grad_input_b[i]).abs() < 1e-5,
            "grad_input[{}] should match between backward and backward_bptt with zero \
             state gradients: backward={:.8}, bptt={:.8}",
            i,
            grad_input_a[i],
            grad_input_b[i]
        );
    }
}

// ============================================================================
// Test: BPTT - Non-zero incoming state gradients change the result
// ============================================================================

#[test]
fn test_bptt_nonzero_incoming_changes_result() {
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 3;

    let input = vec![0.5f32, -0.3, 0.8, 0.2];
    let grad_output = vec![0.4f32, -0.2, 0.6];

    // Case A: backward_bptt with zero incoming state gradients
    let mut rng_a = SimpleRng::new(42);
    let layer_a = LstmLayer::new(input_size, hidden_size, output_size, &mut rng_a);
    layer_a.reset_state();
    let mut out_a = vec![0.0f32; output_size];
    layer_a.forward(&input, &mut out_a, 1);
    let mut grad_input_a = vec![0.0f32; input_size];
    let (dh_prev_a, dc_prev_a) = layer_a.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input_a,
        &vec![0.0f32; hidden_size],
        &vec![0.0f32; hidden_size],
        1,
    );

    // Case B: backward_bptt with non-zero incoming state gradients
    let mut rng_b = SimpleRng::new(42); // same seed → same weights
    let layer_b = LstmLayer::new(input_size, hidden_size, output_size, &mut rng_b);
    layer_b.reset_state();
    let mut out_b = vec![0.0f32; output_size];
    layer_b.forward(&input, &mut out_b, 1);
    // Construct non-trivial incoming state gradients
    let dh_next: Vec<f32> = (0..hidden_size).map(|i| 0.1 * (i as f32)).collect();
    let dc_next: Vec<f32> = (0..hidden_size).map(|i| -0.05 * (i as f32)).collect();
    let mut grad_input_b = vec![0.0f32; input_size];
    let (dh_prev_b, dc_prev_b) = layer_b.backward_bptt(
        &input,
        &grad_output,
        &mut grad_input_b,
        &dh_next,
        &dc_next,
        1,
    );

    // All returned values should be finite
    assert!(dh_prev_b.iter().all(|&x| x.is_finite()), "dh_prev_b finite");
    assert!(dc_prev_b.iter().all(|&x| x.is_finite()), "dc_prev_b finite");

    // The returned state gradients should differ because incoming dh_next/dc_next
    // flows through all four gates and accumulates in dh_prev and dc_prev.
    let dh_differ = dh_prev_a
        .iter()
        .zip(dh_prev_b.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-6);
    let dc_differ = dc_prev_a
        .iter()
        .zip(dc_prev_b.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-6);

    assert!(
        dh_differ,
        "dh_prev should change when non-zero dh_next is provided via backward_bptt"
    );
    assert!(
        dc_differ,
        "dc_prev should change when non-zero dc_next is provided via backward_bptt"
    );
}

// ============================================================================
// Test: BPTT - Multi-step gradient chain propagates correctly
// ============================================================================

#[test]
fn test_bptt_multi_step_gradient_chain() {
    // Verify that state gradients (dh_prev, dc_prev) are chained correctly
    // across multiple time steps when backward_bptt is called in reverse order.
    //
    // Strategy: forward through seq_len steps, then in reverse replay each step's
    // forward pass to repopulate the cache, then call backward_bptt.  The state
    // gradients at t=0 should be non-zero even when only the last step has a
    // non-trivial output loss (gradient flows through BPTT).
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 6;
    let output_size = 2;
    let seq_len = 3;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    let inputs: Vec<Vec<f32>> = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0, 0.0],
        vec![0.0f32, 0.0, 1.0, 0.0],
    ];

    // Forward pass: save hidden/cell state before each time step so we can replay
    let mut saved_h: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    let mut saved_c: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

    layer.reset_state();
    for input in &inputs {
        // Save state BEFORE this time step's forward pass
        saved_h.push(layer.get_hidden_state());
        saved_c.push(layer.get_cell_state());

        let mut output = vec![0.0f32; output_size];
        layer.forward(input, &mut output, 1);
    }

    // Backward pass in reverse order using backward_bptt.
    // For each step t we restore the state to saved_h[t]/saved_c[t], replay the
    // forward to repopulate the cache, then call backward_bptt.
    let mut dh = vec![0.0f32; hidden_size];
    let mut dc = vec![0.0f32; hidden_size];

    for t in (0..seq_len).rev() {
        // Restore pre-step state to replay this time step's forward pass
        layer.set_hidden_state(&saved_h[t]);
        layer.set_cell_state(&saved_c[t]);

        let mut output = vec![0.0f32; output_size];
        layer.forward(&inputs[t], &mut output, 1);

        // Only the last time step has a loss; earlier steps pass zero grad_output
        let grad_output = if t == seq_len - 1 {
            vec![1.0f32, -1.0] // non-trivial gradient at final step
        } else {
            vec![0.0f32; output_size]
        };

        let mut grad_input = vec![0.0f32; input_size];
        let (new_dh, new_dc) =
            layer.backward_bptt(&inputs[t], &grad_output, &mut grad_input, &dh, &dc, 1);

        // State gradients should always be finite
        assert!(
            new_dh.iter().all(|&x| x.is_finite()),
            "dh_prev should be finite at t={}",
            t
        );
        assert!(
            new_dc.iter().all(|&x| x.is_finite()),
            "dc_prev should be finite at t={}",
            t
        );

        dh = new_dh;
        dc = new_dc;
    }

    // After full backward pass, the state gradients at t=-1 (before t=0) should
    // be non-zero: the loss at t=seq_len-1 has propagated all the way back.
    assert!(
        dh.iter().any(|&x| x.abs() > 1e-10),
        "dh should be non-zero at t=0 after BPTT from final loss"
    );
    assert!(
        dc.iter().any(|&x| x.abs() > 1e-10),
        "dc should be non-zero at t=0 after BPTT from final loss"
    );

    // Update parameters to verify gradient accumulation is coherent
    layer.update_parameters(0.01);
}

// ============================================================================
// Test: BPTT - Sequence training reduces total loss
// ============================================================================

#[test]
fn test_bptt_sequence_training_reduces_loss() {
    // Verify that using backward_bptt in a training loop reduces the total
    // sequence loss, confirming that the gradients are correctly oriented.
    let mut rng = SimpleRng::new(42);
    let input_size = 4;
    let hidden_size = 8;
    let output_size = 2;
    let seq_len = 4;

    let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

    let inputs: Vec<Vec<f32>> = vec![
        vec![0.5f32, 0.3, -0.2, 0.1],
        vec![-0.3f32, 0.6, 0.4, -0.5],
        vec![0.2f32, -0.4, 0.7, 0.3],
        vec![-0.1f32, 0.5, -0.3, 0.8],
    ];
    let targets: Vec<Vec<f32>> = vec![
        vec![0.8f32, 0.2],
        vec![0.3f32, 0.7],
        vec![0.6f32, 0.4],
        vec![0.1f32, 0.9],
    ];

    // Measure initial loss
    let initial_loss = compute_sequence_loss(&layer, &inputs, &targets);

    // Train using BPTT for several epochs
    let learning_rate = 0.05f32;
    let num_epochs = 20;

    for _ in 0..num_epochs {
        // Forward pass: save hidden/cell state before each time step
        let mut saved_h: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut saved_c: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        layer.reset_state();
        for input in &inputs {
            saved_h.push(layer.get_hidden_state());
            saved_c.push(layer.get_cell_state());
            let mut output = vec![0.0f32; output_size];
            layer.forward(input, &mut output, 1);
        }

        // Backward pass in reverse using backward_bptt
        let mut dh = vec![0.0f32; hidden_size];
        let mut dc = vec![0.0f32; hidden_size];

        for t in (0..seq_len).rev() {
            // Restore pre-step state and replay forward to update cache
            layer.set_hidden_state(&saved_h[t]);
            layer.set_cell_state(&saved_c[t]);
            let mut output = vec![0.0f32; output_size];
            layer.forward(&inputs[t], &mut output, 1);

            // MSE gradient
            let mut grad_output = vec![0.0f32; output_size];
            for i in 0..output_size {
                grad_output[i] = output[i] - targets[t][i];
            }

            let mut grad_input = vec![0.0f32; input_size];
            let (new_dh, new_dc) =
                layer.backward_bptt(&inputs[t], &grad_output, &mut grad_input, &dh, &dc, 1);
            dh = new_dh;
            dc = new_dc;
        }

        layer.update_parameters(learning_rate);
    }

    // Loss should decrease with BPTT training
    let final_loss = compute_sequence_loss(&layer, &inputs, &targets);
    assert!(
        final_loss < initial_loss,
        "BPTT sequence training should reduce loss: initial={:.6}, final={:.6}",
        initial_loss,
        final_loss
    );
}
