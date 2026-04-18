use super::*;

#[test]
fn test_conv2d_creation_basic() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    assert_eq!(layer.in_channels(), 1);
    assert_eq!(layer.out_channels(), 8);
    assert_eq!(layer.kernel_size(), 3);
    assert_eq!(layer.padding(), 1);
    assert_eq!(layer.stride(), 1);
}

#[test]
fn test_conv2d_creation_multi_channel() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng);

    assert_eq!(layer.in_channels(), 3);
    assert_eq!(layer.out_channels(), 16);
    assert_eq!(layer.kernel_size(), 5);
}

#[test]
fn test_conv2d_parameter_count() {
    let mut rng = SimpleRng::new(42);

    // 1 input, 8 output, 3×3 kernel
    // weights: 8 * 1 * 3 * 3 = 72, biases: 8, total: 80
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    assert_eq!(layer.parameter_count(), 80);

    // 3 input, 16 output, 5×5 kernel
    // weights: 16 * 3 * 5 * 5 = 1200, biases: 16, total: 1216
    let mut rng2 = SimpleRng::new(42);
    let layer2 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng2);
    assert_eq!(layer2.parameter_count(), 1216);
}

#[test]
fn test_conv2d_output_dimensions_same_padding() {
    let mut rng = SimpleRng::new(42);
    // padding=1, kernel=3, stride=1 should maintain dimensions
    let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

    assert_eq!(layer.output_height(), 28);
    assert_eq!(layer.output_width(), 28);
}

#[test]
fn test_conv2d_output_dimensions_no_padding() {
    let mut rng = SimpleRng::new(42);
    // padding=0, kernel=3, stride=1 reduces by 2 each side
    let layer = Conv2DLayer::new(1, 8, 3, 0, 1, 28, 28, &mut rng);

    assert_eq!(layer.output_height(), 26);
    assert_eq!(layer.output_width(), 26);
}

#[test]
fn test_conv2d_output_dimensions_stride_2() {
    let mut rng = SimpleRng::new(42);
    // padding=1, kernel=3, stride=2 halves dimensions
    let layer = Conv2DLayer::new(1, 8, 3, 1, 2, 28, 28, &mut rng);

    // (28 + 2*1 - 3) / 2 + 1 = 14
    assert_eq!(layer.output_height(), 14);
    assert_eq!(layer.output_width(), 14);
}

#[test]
fn test_conv2d_output_dimensions_5x5_kernel() {
    let mut rng = SimpleRng::new(42);
    // padding=2, kernel=5, stride=1 should maintain dimensions
    let layer = Conv2DLayer::new(1, 8, 5, 2, 1, 28, 28, &mut rng);

    assert_eq!(layer.output_height(), 28);
    assert_eq!(layer.output_width(), 28);
}

#[test]
fn test_conv2d_input_output_size() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(3, 8, 3, 1, 1, 28, 28, &mut rng);

    // Input: 3 channels × 28 × 28 = 2352
    assert_eq!(layer.input_size(), 3 * 28 * 28);
    // Output: 8 channels × 28 × 28 = 6272
    assert_eq!(layer.output_size(), 8 * 28 * 28);
}

#[test]
fn test_conv2d_deterministic() {
    let mut rng1 = SimpleRng::new(12345);
    let layer1 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng1);

    let mut rng2 = SimpleRng::new(12345);
    let layer2 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng2);

    // Would need to expose weights for direct comparison
    // For now, test that outputs match
    let input = vec![1.0f32; 3 * 32 * 32];
    let mut out1 = vec![0.0; 16 * 32 * 32];
    let mut out2 = vec![0.0; 16 * 32 * 32];

    layer1.forward(&input, &mut out1, 1);
    layer2.forward(&input, &mut out2, 1);

    assert_eq!(out1, out2);
}

#[test]
fn test_conv2d_forward_single_sample() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    let input = vec![1.0f32; 64]; // 1 × 8 × 8
    let mut output = vec![0.0; 256]; // 4 × 8 × 8

    layer.forward(&input, &mut output, 1);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_forward_batch() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    let batch_size = 4;
    let input = vec![1.0f32; batch_size * 64];
    let mut output = vec![0.0; batch_size * 256];

    layer.forward(&input, &mut output, batch_size);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_forward_multi_channel() {
    let mut rng = SimpleRng::new(42);
    // 3 input channels (like RGB)
    let layer = Conv2DLayer::new(3, 8, 3, 1, 1, 16, 16, &mut rng);

    let input = vec![1.0f32; 3 * 16 * 16];
    let mut output = vec![0.0; 8 * 16 * 16];

    layer.forward(&input, &mut output, 1);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_forward_consistency() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    let input = vec![1.0f32; 64];
    let mut output1 = vec![0.0; 256];
    let mut output2 = vec![0.0; 256];

    layer.forward(&input, &mut output1, 1);
    layer.forward(&input, &mut output2, 1);

    assert_eq!(output1, output2);
}

#[test]
fn test_conv2d_backward_gradient_shape() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    let input = vec![1.0f32; 64];
    let mut output = vec![0.0; 256];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0f32; 256];
    let mut grad_input = vec![0.0; 64];

    layer.backward(&input, &grad_output, &mut grad_input, 1);

    assert!(grad_input.iter().all(|&x| x.is_finite()));
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_conv2d_backward_batch() {
    let mut rng = SimpleRng::new(42);
    let layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    let batch_size = 4;
    let input = vec![1.0f32; batch_size * 64];
    let mut output = vec![0.0; batch_size * 256];
    layer.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * 256];
    let mut grad_input = vec![0.0; batch_size * 64];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_conv2d_update_parameters() {
    let mut rng = SimpleRng::new(42);
    let mut layer = Conv2DLayer::new(1, 4, 3, 1, 1, 8, 8, &mut rng);

    // Get initial output
    let input = vec![1.0f32; 64];
    let mut output1 = vec![0.0; 256];
    layer.forward(&input, &mut output1, 1);

    // Backward pass
    let grad_output = vec![1.0f32; 256];
    let mut grad_input = vec![0.0; 64];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // Update
    layer.update_parameters(0.1);

    // Output should be different now
    let mut output2 = vec![0.0; 256];
    layer.forward(&input, &mut output2, 1);

    assert_ne!(output1, output2);
}

#[test]
fn test_conv2d_training_loop() {
    let mut rng = SimpleRng::new(42);
    let mut layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

    let input = vec![1.0f32; 16];
    let grad_output = vec![0.1f32; 32]; // 2 × 4 × 4

    for _ in 0..10 {
        let mut output = vec![0.0; 32];
        layer.forward(&input, &mut output, 1);

        let mut grad_input = vec![0.0; 16];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        layer.update_parameters(0.01);
    }

    // Should still be valid
    let mut final_output = vec![0.0; 32];
    layer.forward(&input, &mut final_output, 1);
    assert!(final_output.iter().all(|&x| x.is_finite()));
}
