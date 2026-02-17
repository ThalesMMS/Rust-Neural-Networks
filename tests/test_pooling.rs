// Integration tests for global average pooling layer.
// Tests GlobalAvgPoolLayer behavior in isolation and integration with other layers.

use rust_neural_networks::layers::{DenseLayer, GlobalAvgPoolLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Initialization Tests
// ============================================================================

#[test]
fn test_global_avg_pool_initialization() {
    // Test that layer initializes with correct dimensions and zero parameters
    let layer = GlobalAvgPoolLayer::new(4, 4, 8);

    assert_eq!(layer.in_height(), 4);
    assert_eq!(layer.in_width(), 4);
    assert_eq!(layer.channels(), 8);
    assert_eq!(layer.input_size(), 4 * 4 * 8); // 128
    assert_eq!(layer.output_size(), 8);
    assert_eq!(layer.parameter_count(), 0); // No trainable parameters
    assert_eq!(layer.parameter_memory_bytes(), 0);
}

#[test]
fn test_global_avg_pool_dimensions_non_square() {
    // Test with non-square spatial dimensions
    let layer = GlobalAvgPoolLayer::new(7, 7, 64);
    assert_eq!(layer.input_size(), 7 * 7 * 64); // 3136
    assert_eq!(layer.output_size(), 64);
    assert_eq!(layer.parameter_count(), 0);
}

#[test]
fn test_global_avg_pool_single_channel() {
    // Test with a single channel
    let layer = GlobalAvgPoolLayer::new(3, 3, 1);
    assert_eq!(layer.input_size(), 9);
    assert_eq!(layer.output_size(), 1);
    assert_eq!(layer.parameter_count(), 0);
}

#[test]
fn test_global_avg_pool_1x1_spatial() {
    // Test with 1×1 spatial map (trivial case: output == input)
    let layer = GlobalAvgPoolLayer::new(1, 1, 4);
    assert_eq!(layer.input_size(), 4);
    assert_eq!(layer.output_size(), 4);
    assert_eq!(layer.parameter_count(), 0);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
fn test_global_avg_pool_forward_uniform_values() {
    // Test forward pass with uniform spatial values per channel
    // All spatial positions have the same value, so mean == that value
    // H=2, W=2, C=3; channel values: [1.0, 2.0, 3.0]
    let layer = GlobalAvgPoolLayer::new(2, 2, 3);

    // Input layout: channels-last, HWC order
    // [h=0,w=0,c=0..2], [h=0,w=1,c=0..2], [h=1,w=0,c=0..2], [h=1,w=1,c=0..2]
    let input = vec![
        1.0_f32, 2.0, 3.0, // h=0, w=0
        1.0, 2.0, 3.0, // h=0, w=1
        1.0, 2.0, 3.0, // h=1, w=0
        1.0, 2.0, 3.0, // h=1, w=1
    ];
    let mut output = vec![0.0_f32; 3];
    layer.forward(&input, &mut output, 1);

    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "ch0 expected 1.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 2.0).abs() < 1e-6,
        "ch1 expected 2.0, got {}",
        output[1]
    );
    assert!(
        (output[2] - 3.0).abs() < 1e-6,
        "ch2 expected 3.0, got {}",
        output[2]
    );
}

#[test]
fn test_global_avg_pool_forward_non_uniform_values() {
    // Test forward pass where spatial positions have different values
    // H=1, W=4, C=2
    // ch0 values: 2, 4, 6, 8  -> mean = 5.0
    // ch1 values: 1, 3, 5, 7  -> mean = 4.0
    let layer = GlobalAvgPoolLayer::new(1, 4, 2);
    let input = vec![
        2.0_f32, 1.0, // w=0
        4.0, 3.0, // w=1
        6.0, 5.0, // w=2
        8.0, 7.0, // w=3
    ];
    let mut output = vec![0.0_f32; 2];
    layer.forward(&input, &mut output, 1);

    assert!(
        (output[0] - 5.0).abs() < 1e-6,
        "ch0 expected 5.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 4.0).abs() < 1e-6,
        "ch1 expected 4.0, got {}",
        output[1]
    );
}

#[test]
fn test_global_avg_pool_forward_known_mean() {
    // Test with a 2x2 spatial map where we can easily verify the mean
    // H=2, W=2, C=1
    // values: 1, 2, 3, 4 -> mean = 2.5
    let layer = GlobalAvgPoolLayer::new(2, 2, 1);
    let input = vec![
        1.0_f32, // h=0, w=0
        2.0,     // h=0, w=1
        3.0,     // h=1, w=0
        4.0,     // h=1, w=1
    ];
    let mut output = vec![0.0_f32; 1];
    layer.forward(&input, &mut output, 1);

    assert!(
        (output[0] - 2.5).abs() < 1e-6,
        "Expected mean 2.5, got {}",
        output[0]
    );
}

#[test]
fn test_global_avg_pool_forward_1x1_spatial() {
    // For 1×1 spatial map, output should equal input exactly
    let layer = GlobalAvgPoolLayer::new(1, 1, 4);
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0_f32; 4];
    layer.forward(&input, &mut output, 1);

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        assert!(
            (inp - out).abs() < 1e-6,
            "channel {}: expected {}, got {}",
            i,
            inp,
            out
        );
    }
}

#[test]
fn test_global_avg_pool_forward_batch() {
    // Test forward pass with batch of 2 samples
    // batch=2, H=2, W=2, C=2
    // sample 0: all positions -> [1.0, 2.0]; expected output [1.0, 2.0]
    // sample 1: all positions -> [3.0, 4.0]; expected output [3.0, 4.0]
    let layer = GlobalAvgPoolLayer::new(2, 2, 2);
    let input = vec![
        // sample 0
        1.0_f32, 2.0, // h=0, w=0
        1.0, 2.0, // h=0, w=1
        1.0, 2.0, // h=1, w=0
        1.0, 2.0, // h=1, w=1
        // sample 1
        3.0, 4.0, // h=0, w=0
        3.0, 4.0, // h=0, w=1
        3.0, 4.0, // h=1, w=0
        3.0, 4.0, // h=1, w=1
    ];
    let mut output = vec![0.0_f32; 4]; // batch=2, C=2

    layer.forward(&input, &mut output, 2);

    // Sample 0
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "sample0 ch0: expected 1.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 2.0).abs() < 1e-6,
        "sample0 ch1: expected 2.0, got {}",
        output[1]
    );
    // Sample 1
    assert!(
        (output[2] - 3.0).abs() < 1e-6,
        "sample1 ch0: expected 3.0, got {}",
        output[2]
    );
    assert!(
        (output[3] - 4.0).abs() < 1e-6,
        "sample1 ch1: expected 4.0, got {}",
        output[3]
    );
}

#[test]
fn test_global_avg_pool_forward_larger_batch() {
    // Test with larger batch size to ensure correct per-sample processing
    let h = 3usize;
    let w = 3usize;
    let c = 4usize;
    let batch_size = 8;
    let layer = GlobalAvgPoolLayer::new(h, w, c);

    let hw = h * w;
    let hwc = hw * c;

    // Each sample i has all spatial positions set to [i*1.0, i*2.0, i*3.0, i*4.0]
    let mut input = vec![0.0_f32; batch_size * hwc];
    for b in 0..batch_size {
        for pos in 0..hw {
            for ch in 0..c {
                input[b * hwc + pos * c + ch] = (b as f32 + 1.0) * (ch as f32 + 1.0);
            }
        }
    }

    let mut output = vec![0.0_f32; batch_size * c];
    layer.forward(&input, &mut output, batch_size);

    // Expected: output[b * c + ch] == (b+1) * (ch+1)
    for b in 0..batch_size {
        for ch in 0..c {
            let expected = (b as f32 + 1.0) * (ch as f32 + 1.0);
            let got = output[b * c + ch];
            assert!(
                (got - expected).abs() < 1e-5,
                "batch {} ch {}: expected {}, got {}",
                b,
                ch,
                expected,
                got
            );
        }
    }
}

#[test]
fn test_global_avg_pool_forward_zero_input() {
    // Test that all-zero input produces all-zero output
    let layer = GlobalAvgPoolLayer::new(4, 4, 8);
    let input = vec![0.0_f32; 4 * 4 * 8];
    let mut output = vec![99.0_f32; 8]; // Pre-fill with non-zero to detect overwrite

    layer.forward(&input, &mut output, 1);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.abs() < 1e-6, "channel {}: expected 0.0, got {}", i, val);
    }
}

#[test]
fn test_global_avg_pool_forward_negative_values() {
    // Test forward pass with negative input values
    // H=1, W=2, C=2
    // ch0: -2.0, -4.0  -> mean = -3.0
    // ch1:  1.0, -1.0  -> mean =  0.0
    let layer = GlobalAvgPoolLayer::new(1, 2, 2);
    let input = vec![
        -2.0_f32, 1.0, // w=0
        -4.0, -1.0, // w=1
    ];
    let mut output = vec![0.0_f32; 2];
    layer.forward(&input, &mut output, 1);

    assert!(
        (output[0] - (-3.0)).abs() < 1e-6,
        "ch0 expected -3.0, got {}",
        output[0]
    );
    assert!(
        (output[1] - 0.0).abs() < 1e-6,
        "ch1 expected 0.0, got {}",
        output[1]
    );
}

#[test]
fn test_global_avg_pool_forward_preserves_other_output() {
    // Test that forward pass zeroes accumulator before summing
    // (verifies output is not corrupted by previous state)
    let layer = GlobalAvgPoolLayer::new(2, 2, 3);
    let input = vec![
        4.0_f32, 8.0, 12.0, // h=0, w=0
        4.0, 8.0, 12.0, // h=0, w=1
        4.0, 8.0, 12.0, // h=1, w=0
        4.0, 8.0, 12.0, // h=1, w=1
    ];

    // Run twice - the second call should still produce correct output
    let mut output = vec![0.0_f32; 3];
    layer.forward(&input, &mut output, 1);
    layer.forward(&input, &mut output, 1); // Second call

    assert!((output[0] - 4.0).abs() < 1e-6, "ch0: got {}", output[0]);
    assert!((output[1] - 8.0).abs() < 1e-6, "ch1: got {}", output[1]);
    assert!((output[2] - 12.0).abs() < 1e-6, "ch2: got {}", output[2]);
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

#[test]
fn test_global_avg_pool_backward_uniform_gradient() {
    // Test that backward distributes gradient uniformly to all spatial positions
    // H=2, W=3, C=2  -> HW=6
    // grad_output = [6.0, 12.0] for one sample
    // Every spatial position should receive [6.0/6, 12.0/6] = [1.0, 2.0]
    let h = 2usize;
    let w = 3usize;
    let c = 2usize;
    let layer = GlobalAvgPoolLayer::new(h, w, c);

    let g0 = 6.0_f32;
    let g1 = 12.0_f32;
    let grad_output = vec![g0, g1];
    let input = vec![0.0_f32; h * w * c]; // unused in backward
    let mut grad_input = vec![0.0_f32; h * w * c];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    let expected_g0 = g0 / (h * w) as f32; // 1.0
    let expected_g1 = g1 / (h * w) as f32; // 2.0

    for pos in 0..(h * w) {
        assert!(
            (grad_input[pos * c] - expected_g0).abs() < 1e-6,
            "pos {} ch0: expected {}, got {}",
            pos,
            expected_g0,
            grad_input[pos * c]
        );
        assert!(
            (grad_input[pos * c + 1] - expected_g1).abs() < 1e-6,
            "pos {} ch1: expected {}, got {}",
            pos,
            expected_g1,
            grad_input[pos * c + 1]
        );
    }
}

#[test]
fn test_global_avg_pool_backward_1x1_spatial() {
    // For 1×1 spatial map, backward gradient == forward gradient (no scaling)
    let layer = GlobalAvgPoolLayer::new(1, 1, 4);
    let grad_output = vec![0.5_f32, 1.0, 1.5, 2.0];
    let input = vec![0.0_f32; 4]; // unused
    let mut grad_input = vec![0.0_f32; 4];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    for (i, (&g_out, &g_in)) in grad_output.iter().zip(grad_input.iter()).enumerate() {
        assert!(
            (g_out - g_in).abs() < 1e-6,
            "channel {}: expected {}, got {}",
            i,
            g_out,
            g_in
        );
    }
}

#[test]
fn test_global_avg_pool_backward_batch() {
    // Test backward with batch of 2 samples
    // batch=2, H=2, W=2, C=1
    // sample 0 grad_output = [4.0],  expected grad_input = [1.0, 1.0, 1.0, 1.0]
    // sample 1 grad_output = [8.0],  expected grad_input = [2.0, 2.0, 2.0, 2.0]
    let layer = GlobalAvgPoolLayer::new(2, 2, 1);
    let grad_output = vec![4.0_f32, 8.0];
    let input = vec![0.0_f32; 2 * 4];
    let mut grad_input = vec![0.0_f32; 2 * 4];

    layer.backward(&input, &grad_output, &mut grad_input, 2);

    // Sample 0: gradient spread over 4 positions -> 4.0 / 4 = 1.0 each
    for i in 0..4 {
        assert!(
            (grad_input[i] - 1.0).abs() < 1e-6,
            "s0 pos{}: expected 1.0, got {}",
            i,
            grad_input[i]
        );
    }
    // Sample 1: gradient spread over 4 positions -> 8.0 / 4 = 2.0 each
    for i in 4..8 {
        assert!(
            (grad_input[i] - 2.0).abs() < 1e-6,
            "s1 pos{}: expected 2.0, got {}",
            i,
            grad_input[i]
        );
    }
}

#[test]
fn test_global_avg_pool_backward_gradient_sum_conservation() {
    // Test that the sum of grad_input equals the sum of grad_output
    // (gradient magnitudes are conserved across the spatial dimension)
    let h = 4usize;
    let w = 4usize;
    let c = 3usize;
    let layer = GlobalAvgPoolLayer::new(h, w, c);

    let grad_output = vec![1.0_f32, 2.0, 3.0];
    let input = vec![0.0_f32; h * w * c];
    let mut grad_input = vec![0.0_f32; h * w * c];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Sum of grad_input for each channel should equal grad_output for that channel
    for ch in 0..c {
        let mut sum = 0.0_f32;
        for pos in 0..(h * w) {
            sum += grad_input[pos * c + ch];
        }
        assert!(
            (sum - grad_output[ch]).abs() < 1e-5,
            "channel {}: sum of grad_input ({}) != grad_output ({})",
            ch,
            sum,
            grad_output[ch]
        );
    }
}

#[test]
fn test_global_avg_pool_backward_negative_gradients() {
    // Test backward with negative gradient values
    // H=1, W=3, C=2
    // grad_output = [-3.0, 6.0]
    // Expected: ch0 at each position -> -3.0/3 = -1.0, ch1 -> 6.0/3 = 2.0
    let layer = GlobalAvgPoolLayer::new(1, 3, 2);
    let grad_output = vec![-3.0_f32, 6.0];
    let input = vec![0.0_f32; 6];
    let mut grad_input = vec![0.0_f32; 6];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    for pos in 0..3 {
        assert!(
            (grad_input[pos * 2] - (-1.0)).abs() < 1e-6,
            "pos {} ch0: expected -1.0, got {}",
            pos,
            grad_input[pos * 2]
        );
        assert!(
            (grad_input[pos * 2 + 1] - 2.0).abs() < 1e-6,
            "pos {} ch1: expected 2.0, got {}",
            pos,
            grad_input[pos * 2 + 1]
        );
    }
}

// ============================================================================
// Parameter Update Tests
// ============================================================================

#[test]
fn test_global_avg_pool_update_parameters_noop() {
    // Test that update_parameters is a no-op (no parameters to update)
    let mut layer = GlobalAvgPoolLayer::new(4, 4, 8);

    // Should not panic or change anything
    layer.update_parameters(0.01);
    layer.update_parameters(1.0);
    layer.update_parameters(0.0);

    // Layer dimensions unchanged
    assert_eq!(layer.in_height(), 4);
    assert_eq!(layer.in_width(), 4);
    assert_eq!(layer.channels(), 8);
    assert_eq!(layer.parameter_count(), 0);
}

// ============================================================================
// Forward/Backward Consistency Tests (Numerical Gradient Check)
// ============================================================================

#[test]
fn test_global_avg_pool_gradient_numerical_check() {
    // Numerical gradient check: verify backward gradients match numerical estimates
    // Use a small layer for tractability: H=2, W=2, C=2
    let h = 2usize;
    let w = 2usize;
    let c = 2usize;
    let layer = GlobalAvgPoolLayer::new(h, w, c);

    let input: Vec<f32> = (0..(h * w * c)).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut output = vec![0.0_f32; c];

    layer.forward(&input, &mut output, 1);

    // Simulate a scalar loss: L = sum(output)
    let grad_output = vec![1.0_f32; c];
    let mut grad_input_analytic = vec![0.0_f32; h * w * c];
    layer.backward(&input, &grad_output, &mut grad_input_analytic, 1);

    // Numerical gradient: dL/d(input[k]) ≈ (L(input + eps*e_k) - L(input - eps*e_k)) / (2*eps)
    let eps = 1e-4_f32;
    for k in 0..(h * w * c) {
        let mut input_plus = input.clone();
        let mut input_minus = input.clone();
        input_plus[k] += eps;
        input_minus[k] -= eps;

        let mut out_plus = vec![0.0_f32; c];
        let mut out_minus = vec![0.0_f32; c];
        layer.forward(&input_plus, &mut out_plus, 1);
        layer.forward(&input_minus, &mut out_minus, 1);

        // L = sum(output) = sum over channels
        let loss_plus: f32 = out_plus.iter().sum();
        let loss_minus: f32 = out_minus.iter().sum();
        let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

        // Use relaxed tolerance for f32 numerical gradient check:
        // f32 has ~7 decimal digits of precision, so with eps=1e-4 and
        // values O(1), numerical errors can reach ~1e-3.
        assert!(
            (grad_input_analytic[k] - numerical_grad).abs() < 5e-3,
            "Gradient mismatch at input[{}]: analytic={}, numerical={}",
            k,
            grad_input_analytic[k],
            numerical_grad
        );
    }
}

// ============================================================================
// Integration Tests with Other Layers
// ============================================================================

#[test]
fn test_global_avg_pool_followed_by_dense() {
    // Test GlobalAvgPoolLayer chained with DenseLayer
    // Simulates the typical classifier head: GAP -> Dense (FC)
    let mut rng = SimpleRng::new(42);
    let h = 4usize;
    let w = 4usize;
    let c = 8usize;
    let num_classes = 10usize;

    let gap = GlobalAvgPoolLayer::new(h, w, c);
    let dense = DenseLayer::new(c, num_classes, &mut rng);

    let batch_size = 2;
    let input = vec![1.0_f32; batch_size * h * w * c];
    let mut gap_output = vec![0.0_f32; batch_size * c];
    let mut dense_output = vec![0.0_f32; batch_size * num_classes];

    // Forward: input -> GAP -> Dense
    gap.forward(&input, &mut gap_output, batch_size);
    dense.forward(&gap_output, &mut dense_output, batch_size);

    // Verify output dimensions are correct
    assert_eq!(gap_output.len(), batch_size * c);
    assert_eq!(dense_output.len(), batch_size * num_classes);

    // All values should be finite
    for &v in &gap_output {
        assert!(v.is_finite(), "GAP output should be finite");
    }
    for &v in &dense_output {
        assert!(v.is_finite(), "Dense output should be finite");
    }
}

#[test]
fn test_global_avg_pool_forward_backward_pipeline() {
    // Full forward + backward pipeline test
    // Verifies that the backward pass can be used for gradient computation
    // that would be used to update a preceding learnable layer (Dense in this case)
    let h = 3usize;
    let w = 3usize;
    let c = 4usize;
    let layer = GlobalAvgPoolLayer::new(h, w, c);

    let batch_size = 2;
    let hwc = h * w * c;

    let input: Vec<f32> = (0..(batch_size * hwc))
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    // Forward pass
    let mut output = vec![0.0_f32; batch_size * c];
    layer.forward(&input, &mut output, batch_size);

    // Check forward output dimensions
    assert_eq!(output.len(), batch_size * c);
    for &v in &output {
        assert!(v.is_finite(), "Forward output should be finite");
    }

    // Backward pass with simulated upstream gradient
    let grad_output: Vec<f32> = (0..(batch_size * c))
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let mut grad_input = vec![0.0_f32; batch_size * hwc];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Check backward gradient dimensions and finiteness
    assert_eq!(grad_input.len(), batch_size * hwc);
    for &g in &grad_input {
        assert!(g.is_finite(), "Backward gradient should be finite");
    }
}

#[test]
fn test_global_avg_pool_output_size_matches_layer_trait() {
    // Test that output_size() and input_size() correctly reflect layer dimensions
    // These are used by other components to wire up layers automatically
    let configurations = [(1, 1, 1), (2, 3, 4), (7, 7, 64), (14, 14, 128), (4, 4, 256)];

    for &(h, w, c) in &configurations {
        let layer = GlobalAvgPoolLayer::new(h, w, c);
        assert_eq!(
            layer.input_size(),
            h * w * c,
            "input_size mismatch for ({}, {}, {})",
            h,
            w,
            c
        );
        assert_eq!(
            layer.output_size(),
            c,
            "output_size mismatch for ({}, {}, {})",
            h,
            w,
            c
        );
        assert_eq!(
            layer.parameter_count(),
            0,
            "parameter_count should be 0 for ({}, {}, {})",
            h,
            w,
            c
        );
    }
}
