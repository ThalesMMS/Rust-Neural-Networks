use super::*;
use crate::layers::Layer;

#[test]
fn test_conv2d_initialization() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    assert_eq!(layer.in_channels(), 1);
    assert_eq!(layer.out_channels(), 8);
    assert_eq!(layer.kernel_size(), 3);
    assert_eq!(layer.padding(), 1);
    assert_eq!(layer.stride(), 1);
}

#[test]
fn test_conv2d_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    // weights: 8 * 1 * 3 * 3 = 72
    // biases: 8
    // total: 80
    assert_eq!(layer.parameter_count(), 80);
}

#[test]
fn test_conv2d_output_dimensions() {
    let mut rng = SimpleRng::new(42);
    // With padding=1 and stride=1, 3x3 kernel maintains spatial dimensions
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    assert_eq!(layer.output_height(), 28);
    assert_eq!(layer.output_width(), 28);
}

#[test]
fn test_conv2d_output_dimensions_no_padding() {
    let mut rng = SimpleRng::new(42);
    // Without padding, 3x3 kernel reduces dimensions by 2 on each side
    let layer = Conv2DLayer::new(1, 8, 3, 0, 1, 28, 28, &mut rng);

    assert_eq!(layer.output_height(), 26); // 28 - 3 + 1 = 26
    assert_eq!(layer.output_width(), 26);
}

#[test]
fn test_conv2d_xavier_initialization_bounds() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    // Xavier limit for this configuration
    let fan_in = (3 * 3) as f32;
    let fan_out = (8 * 3 * 3) as f32;
    let limit = (6.0f32 / (fan_in + fan_out)).sqrt();

    // All weights should be within [-limit, limit]
    for &weight in &layer.weights {
        assert!(
            weight >= -limit && weight <= limit,
            "Weight {} outside Xavier bounds [{}, {}]",
            weight,
            -limit,
            limit
        );
    }

    // All biases should be initialized to zero
    for &bias in &layer.biases {
        assert_eq!(bias, 0.0);
    }
}

#[test]
fn test_conv2d_deterministic_initialization() {
    let mut rng1 = SimpleRng::new(12345);
    let layer1 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng1);

    let mut rng2 = SimpleRng::new(12345);
    let layer2 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng2);

    // Same seed should produce identical weights
    assert_eq!(layer1.weights, layer2.weights);
    assert_eq!(layer1.biases, layer2.biases);
}

#[test]
fn test_conv2d_forward() {
    let mut rng = SimpleRng::new(42);
    // 1 input channel, 2 output channels, 3x3 kernel, padding=1, stride=1, 4x4 input
    let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

    // Single sample: 1 channel × 4 × 4 = 16 values
    let input = vec![1.0f32; 16];
    // Output: 2 channels × 4 × 4 = 32 values
    let mut output = vec![0.0f32; 32];

    layer.forward(&input, &mut output, 1);

    // Output should be computed (finite values)
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_forward_batch() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

    // Batch of 2 samples
    let input = vec![1.0f32; 32]; // 2 × 16
    let mut output = vec![0.0f32; 64]; // 2 × 32

    layer.forward(&input, &mut output, 2);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_backward() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

    let input = vec![1.0f32; 16];
    let mut output = vec![0.0f32; 32];
    layer.forward(&input, &mut output, 1);

    // Gradient from loss
    let grad_output = vec![1.0f32; 32];
    let mut grad_input = vec![0.0f32; 16];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Gradients should be finite
    assert!(grad_input.iter().all(|&x| x.is_finite()));
    // At least some gradients should be non-zero
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_conv2d_update_parameters() {
    let mut rng = SimpleRng::new(42);
    let mut layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

    let original_weights = layer.weights.clone();

    let input = vec![1.0f32; 16];
    let mut output = vec![0.0f32; 32];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0f32; 32];
    let mut grad_input = vec![0.0f32; 16];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

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
fn test_conv2d_input_output_size() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(3, 8, 3, 1, 1, 28, 28, &mut rng);

    // Input size: 3 channels × 28 × 28
    assert_eq!(layer.input_size(), 3 * 28 * 28);
    // Output size: 8 channels × 28 × 28 (with padding=1, same dimensions)
    assert_eq!(layer.output_size(), 8 * 28 * 28);
}

#[test]
fn test_conv2d_stride_2() {
    let mut rng = SimpleRng::new(42);
    // Stride 2 should halve the output dimensions
    let layer = Conv2DLayer::new(1, 4, 3, 1, 2, 8, 8, &mut rng);

    // (8 + 2*1 - 3) / 2 + 1 = 4
    assert_eq!(layer.output_height(), 4);
    assert_eq!(layer.output_width(), 4);
}

#[test]
fn test_conv2d_new_with_weights_stores_parameters() {
    let in_channels = 1;
    let out_channels = 2;
    let kernel_size = 3;
    let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    let weights = vec![0.1f32; weight_count];
    let biases = vec![0.0f32, 0.5];

    let layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        1,
        1,
        28,
        28,
        weights.clone(),
        biases.clone(),
    );

    assert_eq!(layer.in_channels(), in_channels);
    assert_eq!(layer.out_channels(), out_channels);
    assert_eq!(layer.kernel_size(), kernel_size);
    assert_eq!(layer.padding(), 1);
    assert_eq!(layer.stride(), 1);
    assert_eq!(layer.weights(), weights.as_slice());
    assert_eq!(layer.biases(), biases.as_slice());
}

#[test]
fn test_conv2d_new_with_weights_gradient_initially_zero() {
    let in_channels = 1;
    let out_channels = 2;
    let kernel_size = 3;
    let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    let weights = vec![0.1f32; weight_count];
    let biases = vec![0.0f32; out_channels];

    let layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        1,
        1,
        28,
        28,
        weights,
        biases,
    );

    // Gradient accumulators should start at zero
    let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
    assert_eq!(weight_norm, 0.0);
    assert_eq!(bias_norm, 0.0);
}

#[test]
fn test_conv2d_new_with_weights_parameter_count() {
    let in_channels = 3;
    let out_channels = 8;
    let kernel_size = 3;
    let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    let weights = vec![0.0f32; weight_count];
    let biases = vec![0.0f32; out_channels];

    let layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        1,
        1,
        32,
        32,
        weights,
        biases,
    );

    // 8 * 3 * 3 * 3 weights + 8 biases = 216 + 8 = 224
    assert_eq!(layer.parameter_count(), weight_count + out_channels);
}

#[test]
#[should_panic(expected = "weights length")]
fn test_conv2d_new_with_weights_wrong_weight_length_panics() {
    // 1 in, 2 out, 3×3 kernel: expects 2*1*3*3 = 18 weights, give 10
    let weights = vec![0.1f32; 10];
    let biases = vec![0.0f32; 2];
    let _layer = Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
}

#[test]
#[should_panic(expected = "biases length")]
fn test_conv2d_new_with_weights_wrong_bias_length_panics() {
    // 1 in, 2 out, 3×3 kernel: expects 2 biases, give 5
    let weights = vec![0.1f32; 2 * 3 * 3];
    let biases = vec![0.0f32; 5];
    let _layer = Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
}
