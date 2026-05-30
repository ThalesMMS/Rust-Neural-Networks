use super::*;
use approx::assert_relative_eq;

fn assert_all_close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_relative_eq!(x, y, epsilon = abs, max_relative = rel);
        if !((x - y).abs() <= abs + rel * y.abs()) {
            panic!("mismatch at index {i}: {x} vs {y}");
        }
    }
}

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

fn naive_conv2d_forward_nhwc(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch_size: usize,
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_c: usize,
    k: usize,
    padding: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * out_c * out_h * out_w];

    for b in 0..batch_size {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias[oc];
                    for ic in 0..in_c {
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;

                                if (0..in_h as isize).contains(&ih)
                                    && (0..in_w as isize).contains(&iw)
                                {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let x_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic; // NHWC
                                    let w_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                                    sum += input[x_idx] * weights[w_idx];
                                }
                            }
                        }
                    }
                    let y_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow; // NCHW
                    out[y_idx] = sum;
                }
            }
        }
    }

    out
}

#[test]
fn test_conv2d_forward_matches_naive_reference() {
    let in_channels = 2;
    let out_channels = 3;
    let kernel_size = 3;
    let padding = 1;
    let stride = 2;
    let in_h = 7;
    let in_w = 6;

    let out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    let batch_size = 2;

    // Deterministic, non-trivial values
    let mut input = vec![0.0f32; batch_size * in_h * in_w * in_channels];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32 * 0.01).sin();
    }

    let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    let mut weights = vec![0.0f32; weight_count];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = ((i as f32) * 0.03).cos() * 0.1;
    }

    let mut bias = vec![0.0f32; out_channels];
    for (i, b) in bias.iter_mut().enumerate() {
        *b = (i as f32) * 0.01;
    }

    let layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        padding as isize,
        stride,
        in_h,
        in_w,
        weights.clone(),
        bias.clone(),
    );

    let expected = naive_conv2d_forward_nhwc(
        &input,
        &weights,
        &bias,
        batch_size,
        in_h,
        in_w,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        out_h,
        out_w,
    );

    let mut actual = vec![0.0f32; batch_size * out_channels * out_h * out_w];
    layer.forward(&input, &mut actual, batch_size);

    assert_all_close(&actual, &expected, 1e-5, 1e-5);
}

#[test]
fn test_conv2d_forward_padding_partial_overlap_matches_reference() {
    // Small input + larger padding so many kernel taps fall outside.
    let in_channels = 1;
    let out_channels = 1;
    let kernel_size = 3;
    let padding = 2;
    let stride = 1;
    let in_h = 2;
    let in_w = 3;

    let out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    let batch_size = 1;

    let input: Vec<f32> = (0..(batch_size * in_h * in_w * in_channels))
        .map(|i| (i as f32) * 0.1 - 0.2)
        .collect();

    // Simple weights so edge effects are easy to catch.
    let weights = vec![
        0.1f32, 0.2, 0.3, //
        0.4, 0.5, 0.6, //
        0.7, 0.8, 0.9,
    ];
    let bias = vec![0.01f32];

    let layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        padding as isize,
        stride,
        in_h,
        in_w,
        weights.clone(),
        bias.clone(),
    );

    let expected = naive_conv2d_forward_nhwc(
        &input,
        &weights,
        &bias,
        batch_size,
        in_h,
        in_w,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        out_h,
        out_w,
    );

    let mut actual = vec![0.0f32; batch_size * out_channels * out_h * out_w];
    layer.forward(&input, &mut actual, batch_size);

    // Explicitly ensure some border positions match the reference.
    // This primarily targets padding semantics.
    assert_all_close(&actual, &expected, 1e-5, 1e-5);
    assert_relative_eq!(actual[0], expected[0], epsilon = 1e-5, max_relative = 1e-5);
    let center_idx = (out_h / 2) * out_w + (out_w / 2);
    assert_relative_eq!(
        actual[center_idx],
        expected[center_idx],
        epsilon = 1e-5,
        max_relative = 1e-5
    );
}
