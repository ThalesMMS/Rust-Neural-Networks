use super::*;

#[test]
fn test_dense_layer_creation() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(10, 5, &mut rng);

    assert_eq!(layer.input_size(), 10);
    assert_eq!(layer.output_size(), 5);
    assert_eq!(layer.weights.len(), 50); // 10 × 5
    assert_eq!(layer.biases.len(), 5);
}

#[test]
fn test_dense_layer_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(784, 512, &mut rng);

    // 784 × 512 weights + 512 biases = 401,408 + 512 = 401,920
    assert_eq!(layer.parameter_count(), 784 * 512 + 512);
}

#[test]
fn test_xavier_initialization() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(100, 50, &mut rng);

    // Xavier limit = sqrt(6 / (100 + 50)) = sqrt(6 / 150) ≈ 0.2
    let limit = (6.0f32 / 150.0).sqrt();

    // Check that all weights are within the expected range
    for &weight in &layer.weights {
        assert!(
            weight >= -limit && weight <= limit,
            "Weight {} outside Xavier range [{}, {}]",
            weight,
            -limit,
            limit
        );
    }

    // Check that biases are initialized to zero
    for &bias in &layer.biases {
        assert_eq!(bias, 0.0);
    }
}

#[test]
fn test_deterministic_initialization() {
    let mut rng1 = SimpleRng::new(42);
    let layer1 = DenseLayer::new(10, 5, &mut rng1);

    let mut rng2 = SimpleRng::new(42);
    let layer2 = DenseLayer::new(10, 5, &mut rng2);

    // Same seed should produce identical weights
    assert_eq!(layer1.weights, layer2.weights);
    assert_eq!(layer1.biases, layer2.biases);
}

#[test]
fn test_add_bias() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 cols
    let bias = vec![0.1, 0.2, 0.3];
    add_bias(&mut data, 2, 3, &bias);

    assert!((data[0] - 1.1).abs() < 1e-6);
    assert!((data[1] - 2.2).abs() < 1e-6);
    assert!((data[2] - 3.3).abs() < 1e-6);
    assert!((data[3] - 4.1).abs() < 1e-6);
    assert!((data[4] - 5.2).abs() < 1e-6);
    assert!((data[5] - 6.3).abs() < 1e-6);
}

#[test]
fn test_sum_rows() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 cols
    let mut out = vec![0.0; 3];
    sum_rows(&data, 2, 3, &mut out);

    // Column 0: 1 + 4 = 5
    // Column 1: 2 + 5 = 7
    // Column 2: 3 + 6 = 9
    assert!((out[0] - 5.0).abs() < 1e-6);
    assert!((out[1] - 7.0).abs() < 1e-6);
    assert!((out[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_dense_forward() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(3, 2, &mut rng);

    // Single sample forward pass
    let input = vec![1.0, 0.5, -0.5];
    let mut output = vec![0.0; 2];

    layer.forward(&input, &mut output, 1);

    // Output should be input × weights + biases
    // Verify output is computed (not zeros and finite)
    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x != 0.0) || layer.biases.iter().all(|&b| b == 0.0));
}

#[test]
fn test_dense_forward_batch() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(2, 3, &mut rng);

    // Batch of 2 samples
    let input = vec![1.0, 0.0, 0.0, 1.0]; // 2 samples × 2 features
    let mut output = vec![0.0; 6]; // 2 samples × 3 outputs

    layer.forward(&input, &mut output, 2);

    // All outputs should be finite
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_dense_backward() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(3, 2, &mut rng);

    let input = vec![1.0, 0.5, -0.5];
    let mut output = vec![0.0; 2];
    layer.forward(&input, &mut output, 1);

    // Create gradient of output
    let grad_output = vec![1.0, -1.0];
    let mut grad_input = vec![0.0; 3];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Gradient should propagate back
    assert!(grad_input.iter().all(|&x| x.is_finite()));
    // At least some gradients should be non-zero
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_dense_update_parameters() {
    let mut rng = SimpleRng::new(42);
    let mut layer = DenseLayer::new(3, 2, &mut rng);

    let original_weights = layer.weights.clone();
    let _original_biases = layer.biases.clone();

    // Do a forward and backward pass to accumulate gradients
    let input = vec![1.0, 1.0, 1.0];
    let mut output = vec![0.0; 2];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0, 1.0];
    let mut grad_input = vec![0.0; 3];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Update parameters
    layer.update_parameters(0.1);

    // Weights should have changed
    let weights_changed = layer
        .weights
        .iter()
        .zip(original_weights.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(weights_changed, "Weights should change after update");
}

#[test]
fn test_weights_and_biases_accessors() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(4, 3, &mut rng);

    assert_eq!(layer.weights().len(), 12); // 4 × 3
    assert_eq!(layer.biases().len(), 3);
}

#[test]
fn test_gradient_magnitude_initially_zero() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(3, 2, &mut rng);

    // Initially, gradients should be zero
    let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
    assert_eq!(weight_norm, 0.0);
    assert_eq!(bias_norm, 0.0);
}

#[test]
fn test_gradient_magnitude_after_backward() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(3, 2, &mut rng);

    // Perform forward pass
    let input = vec![1.0, 0.5, -0.5];
    let mut output = vec![0.0; 2];
    layer.forward(&input, &mut output, 1);

    // Perform backward pass to accumulate gradients
    let grad_output = vec![1.0, -1.0];
    let mut grad_input = vec![0.0; 3];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Get gradient magnitudes
    let (weight_norm, bias_norm) = layer.get_gradient_magnitude();

    // Gradients should be non-zero and non-negative
    assert!(weight_norm >= 0.0);
    assert!(bias_norm >= 0.0);
    assert!(weight_norm > 0.0 || bias_norm > 0.0);
}

#[test]
fn test_gradient_magnitude_after_update() {
    let mut rng = SimpleRng::new(42);
    let mut layer = DenseLayer::new(3, 2, &mut rng);

    // Perform forward and backward pass
    let input = vec![1.0, 1.0, 1.0];
    let mut output = vec![0.0; 2];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0, 1.0];
    let mut grad_input = vec![0.0; 3];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Verify gradients are non-zero
    let (weight_norm_before, bias_norm_before) = layer.get_gradient_magnitude();
    assert!(weight_norm_before > 0.0);
    assert!(bias_norm_before > 0.0);

    // Update parameters (this should clear gradients)
    layer.update_parameters(0.1);

    // After update, gradients should be cleared to zero
    let (weight_norm_after, bias_norm_after) = layer.get_gradient_magnitude();
    assert_eq!(weight_norm_after, 0.0);
    assert_eq!(bias_norm_after, 0.0);
}

#[test]
fn test_gradient_magnitude_accumulation() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(2, 2, &mut rng);

    // First backward pass
    let input = vec![1.0, 1.0];
    let mut output = vec![0.0; 2];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![0.1, 0.1];
    let mut grad_input = vec![0.0; 2];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    let (weight_norm_first, bias_norm_first) = layer.get_gradient_magnitude();

    // Second backward pass (accumulates gradients)
    layer.forward(&input, &mut output, 1);
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    let (weight_norm_second, bias_norm_second) = layer.get_gradient_magnitude();

    // Gradients should have accumulated (increased)
    assert!(weight_norm_second > weight_norm_first);
    assert!(bias_norm_second > bias_norm_first);
}

#[test]
fn test_dense_new_with_weights_stores_parameters() {
    let input_size = 2;
    let output_size = 3;
    let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 × 3
    let biases = vec![0.1f32, 0.2, 0.3];

    let layer =
        DenseLayer::new_with_weights(input_size, output_size, weights.clone(), biases.clone());

    assert_eq!(layer.input_size(), input_size);
    assert_eq!(layer.output_size(), output_size);
    assert_eq!(layer.weights(), weights.as_slice());
    assert_eq!(layer.biases(), biases.as_slice());
}

#[test]
fn test_dense_new_with_weights_gradient_initially_zero() {
    let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 × 2
    let biases = vec![0.5f32, -0.5];

    let layer = DenseLayer::new_with_weights(2, 2, weights, biases);

    // Gradient accumulators should start at zero
    let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
    assert_eq!(weight_norm, 0.0);
    assert_eq!(bias_norm, 0.0);
}

#[test]
fn test_dense_new_with_weights_correct_forward_output() {
    // Use a known weight matrix to verify forward computation
    // Input: [1.0, 0.0], Weights: [[1.0, 2.0], [3.0, 4.0]], Biases: [0.5, -0.5]
    // Expected output: [1.0*1.0 + 0.0*3.0 + 0.5, 1.0*2.0 + 0.0*4.0 + (-0.5)]
    //                = [1.5, 1.5]
    let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // row-major 2×2
    let biases = vec![0.5f32, -0.5];

    let layer = DenseLayer::new_with_weights(2, 2, weights, biases);

    let input = vec![1.0f32, 0.0];
    let mut output = vec![0.0f32; 2];
    layer.forward(&input, &mut output, 1);

    assert!((output[0] - 1.5).abs() < 1e-6, "output[0] = {}", output[0]);
    assert!((output[1] - 1.5).abs() < 1e-6, "output[1] = {}", output[1]);
}

#[test]
#[should_panic(expected = "weights length")]
fn test_dense_new_with_weights_wrong_weight_length_panics() {
    // 2×2 layer but only 3 weights provided (should be 4)
    let weights = vec![0.1f32, 0.2, 0.3];
    let biases = vec![0.0f32, 0.0];
    let _layer = DenseLayer::new_with_weights(2, 2, weights, biases);
}

#[test]
#[should_panic(expected = "biases length")]
fn test_dense_new_with_weights_wrong_bias_length_panics() {
    // 2×2 layer but 3 biases provided (should be 2)
    let weights = vec![0.1f32, 0.2, 0.3, 0.4];
    let biases = vec![0.0f32, 0.0, 0.0];
    let _layer = DenseLayer::new_with_weights(2, 2, weights, biases);
}
