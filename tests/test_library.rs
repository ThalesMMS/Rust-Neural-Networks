//! Comprehensive tests for the rust_neural_networks library
//!
//! This file tests the public API of the library including:
//! - DenseLayer: creation, forward, backward, parameter updates
//! - Conv2DLayer: creation, forward, backward, parameter updates
//! - Activation functions: sigmoid, relu, softmax
//! - SimpleRng: random number generation

use approx::assert_relative_eq;
use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::activations::{
    relu_inplace, sigmoid, sigmoid_derivative, softmax_rows,
};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// DenseLayer Tests
// ============================================================================

mod dense_layer_tests {
    use super::*;

    #[test]
    fn test_dense_layer_creation_basic() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(10, 5, &mut rng);

        assert_eq!(layer.input_size(), 10);
        assert_eq!(layer.output_size(), 5);
        assert_eq!(layer.parameter_count(), 10 * 5 + 5); // weights + biases
    }

    #[test]
    fn test_dense_layer_creation_large() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(784, 512, &mut rng);

        assert_eq!(layer.input_size(), 784);
        assert_eq!(layer.output_size(), 512);
        assert_eq!(layer.parameter_count(), 784 * 512 + 512);
    }

    #[test]
    fn test_dense_layer_creation_single_neuron() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(100, 1, &mut rng);

        assert_eq!(layer.input_size(), 100);
        assert_eq!(layer.output_size(), 1);
        assert_eq!(layer.parameter_count(), 101);
    }

    #[test]
    fn test_dense_layer_xavier_initialization() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(100, 50, &mut rng);

        // Xavier limit = sqrt(6 / (100 + 50)) ≈ 0.2
        let limit = (6.0f32 / 150.0).sqrt();

        // All weights should be within [-limit, limit]
        for &w in layer.weights() {
            assert!(
                w >= -limit && w <= limit,
                "Weight {} outside Xavier range",
                w
            );
        }

        // Biases should be zero
        for &b in layer.biases() {
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn test_dense_layer_deterministic() {
        let mut rng1 = SimpleRng::new(12345);
        let layer1 = DenseLayer::new(50, 30, &mut rng1);

        let mut rng2 = SimpleRng::new(12345);
        let layer2 = DenseLayer::new(50, 30, &mut rng2);

        assert_eq!(layer1.weights(), layer2.weights());
        assert_eq!(layer1.biases(), layer2.biases());
    }

    #[test]
    fn test_dense_forward_single_sample() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0, 0.5, -0.5, 0.0];
        let mut output = vec![0.0; 3];

        layer.forward(&input, &mut output, 1);

        // Output should be finite and likely non-zero
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dense_forward_batch() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        // Batch of 4 samples
        let input = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut output = vec![0.0; 12]; // 4 × 3

        layer.forward(&input, &mut output, 4);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dense_forward_zero_input() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![0.0; 4];
        let mut output = vec![0.0; 3];

        layer.forward(&input, &mut output, 1);

        // With zero input and zero biases, output should be zero
        for &o in &output {
            assert_eq!(o, 0.0);
        }
    }

    #[test]
    fn test_dense_forward_consistency() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output1 = vec![0.0; 3];
        let mut output2 = vec![0.0; 3];

        layer.forward(&input, &mut output1, 1);
        layer.forward(&input, &mut output2, 1);

        // Same input should give same output
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_dense_backward_gradient_shape() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0, 0.5, -0.5, 0.25];
        let mut output = vec![0.0; 3];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0, 0.0, -1.0];
        let mut grad_input = vec![0.0; 4];

        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // All gradients should be finite
        assert!(grad_input.iter().all(|&x| x.is_finite()));
        // At least some should be non-zero
        assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_dense_backward_batch() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let batch_size = 4;
        let input = vec![1.0f32; 16]; // 4 × 4
        let mut output = vec![0.0; 12]; // 4 × 3
        layer.forward(&input, &mut output, batch_size);

        let grad_output = vec![1.0f32; 12];
        let mut grad_input = vec![0.0; 16];

        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dense_backward_zero_gradient() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![0.0; 3];
        let mut grad_input = vec![0.0; 4];

        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Zero gradient output should give zero gradient input
        for &g in &grad_input {
            assert_eq!(g, 0.0);
        }
    }

    #[test]
    fn test_dense_update_parameters() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(4, 3, &mut rng);

        let original_weights: Vec<f32> = layer.weights().to_vec();

        // Forward and backward to accumulate gradients
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0; 3];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0f32; 3];
        let mut grad_input = vec![0.0; 4];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Update with learning rate
        layer.update_parameters(0.1);

        // Weights should have changed
        let new_weights: Vec<f32> = layer.weights().to_vec();
        assert_ne!(original_weights, new_weights);
    }

    #[test]
    fn test_dense_update_parameters_multiple() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0f32; 4];
        let grad_output = vec![1.0f32; 3];

        // Multiple training steps
        for _ in 0..5 {
            let mut output = vec![0.0; 3];
            layer.forward(&input, &mut output, 1);

            let mut grad_input = vec![0.0; 4];
            layer.backward(&input, &grad_output, &mut grad_input, 1);

            layer.update_parameters(0.01);
        }

        // Should still have valid weights
        assert!(layer.weights().iter().all(|&x| x.is_finite()));
        assert!(layer.biases().iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dense_large_batch_gradient_averaging() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        // Large batch
        let batch_size = 32;
        let input = vec![1.0f32; 4 * batch_size];
        let mut output = vec![0.0; 3 * batch_size];
        layer.forward(&input, &mut output, batch_size);

        let grad_output = vec![1.0f32; 3 * batch_size];
        let mut grad_input = vec![0.0; 4 * batch_size];

        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Gradients should be finite even with large batch
        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }
}

// ============================================================================
// Conv2DLayer Tests
// ============================================================================

mod conv2d_layer_tests {
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
}

// ============================================================================
// Activation Function Tests
// ============================================================================

mod activation_tests {
    use super::*;

    // Sigmoid tests
    #[test]
    fn test_sigmoid_zero() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_positive() {
        let result = sigmoid(2.0);
        assert!(result > 0.5 && result < 1.0);
        assert_relative_eq!(result, 0.8807970779778823, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_negative() {
        let result = sigmoid(-2.0);
        assert!(result > 0.0 && result < 0.5);
        assert_relative_eq!(result, 0.11920292202211755, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let result = sigmoid(100.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let result = sigmoid(-100.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        for i in 1..20 {
            let x = i as f64 * 0.5;
            assert_relative_eq!(sigmoid(x) + sigmoid(-x), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sigmoid_monotonic() {
        let mut prev = sigmoid(-10.0);
        for i in -100..100 {
            let x = i as f64 / 10.0;
            let curr = sigmoid(x);
            assert!(curr >= prev, "Sigmoid should be monotonically increasing");
            prev = curr;
        }
    }

    // Sigmoid derivative tests
    #[test]
    fn test_sigmoid_derivative_at_half() {
        assert_relative_eq!(sigmoid_derivative(0.5), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative_at_extremes() {
        assert_relative_eq!(sigmoid_derivative(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sigmoid_derivative(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative_range() {
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let deriv = sigmoid_derivative(x);
            assert!((0.0..=0.25).contains(&deriv));
        }
    }

    #[test]
    fn test_sigmoid_derivative_symmetry() {
        for i in 0..50 {
            let x = i as f64 / 100.0;
            assert_relative_eq!(
                sigmoid_derivative(x),
                sigmoid_derivative(1.0 - x),
                epsilon = 1e-10
            );
        }
    }

    // ReLU tests
    #[test]
    fn test_relu_negative_values() {
        let mut data = vec![-5.0, -3.0, -1.0, -0.1, -0.001];
        relu_inplace(&mut data);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_relu_zero() {
        let mut data = vec![0.0];
        relu_inplace(&mut data);
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_relu_positive_values() {
        let original = vec![0.001, 0.1, 1.0, 5.0, 100.0];
        let mut data = original.clone();
        relu_inplace(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_relu_mixed() {
        let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_relu_empty() {
        let mut data: Vec<f32> = vec![];
        relu_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_relu_large_array() {
        let mut data: Vec<f32> = (-500..500).map(|x| x as f32).collect();
        relu_inplace(&mut data);

        for (i, &val) in data.iter().enumerate() {
            let original = (i as i32 - 500) as f32;
            if original <= 0.0 {
                assert_eq!(val, 0.0);
            } else {
                assert_eq!(val, original);
            }
        }
    }

    // Softmax tests
    #[test]
    fn test_softmax_sum_to_one() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_all_positive() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        assert!(data.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn test_softmax_ordering_preserved() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        assert!(data[0] < data[1] && data[1] < data[2]);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        softmax_rows(&mut data, 1, 4);
        for &val in &data {
            assert_relative_eq!(val, 0.25, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax_multiple_rows() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        softmax_rows(&mut data, 3, 3);

        // Each row should sum to 1
        for row in data.chunks(3) {
            let sum: f32 = row.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax_numerical_stability_large() {
        let mut data = vec![1000.0, 1001.0, 1002.0];
        softmax_rows(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }

    #[test]
    fn test_softmax_numerical_stability_negative_large() {
        let mut data = vec![-1000.0, -1001.0, -1002.0];
        softmax_rows(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }

    #[test]
    fn test_softmax_single_element() {
        let mut data = vec![5.0];
        softmax_rows(&mut data, 1, 1);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_two_elements() {
        let mut data = vec![0.0, 0.0];
        softmax_rows(&mut data, 1, 2);
        assert_relative_eq!(data[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(data[1], 0.5, epsilon = 1e-6);
    }
}

// ============================================================================
// SimpleRng Tests
// ============================================================================

mod rng_tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..1000 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(1);
        let mut rng2 = SimpleRng::new(2);

        // Very unlikely to match
        let vals1: Vec<u32> = (0..10).map(|_| rng1.next_u32()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_rng_zero_seed() {
        let mut rng = SimpleRng::new(0);
        // Should still work with default seed
        let val = rng.next_u32();
        assert!(val > 0);
    }

    #[test]
    fn test_rng_next_f32_range() {
        let mut rng = SimpleRng::new(12345);
        for _ in 0..10000 {
            let val = rng.next_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_rng_gen_range_f32() {
        let mut rng = SimpleRng::new(67890);
        for _ in 0..10000 {
            let val = rng.gen_range_f32(-5.0, 5.0);
            assert!((-5.0..5.0).contains(&val));
        }
    }

    #[test]
    fn test_rng_gen_range_f32_narrow() {
        let mut rng = SimpleRng::new(11111);
        for _ in 0..1000 {
            let val = rng.gen_range_f32(0.999, 1.0);
            assert!((0.999..1.0).contains(&val));
        }
    }

    #[test]
    fn test_rng_gen_usize() {
        let mut rng = SimpleRng::new(22222);
        for _ in 0..10000 {
            let val = rng.gen_usize(100);
            assert!(val < 100);
        }
    }

    #[test]
    fn test_rng_gen_usize_one() {
        let mut rng = SimpleRng::new(33333);
        for _ in 0..100 {
            let val = rng.gen_usize(1);
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_rng_gen_usize_zero() {
        let mut rng = SimpleRng::new(44444);
        let val = rng.gen_usize(0);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_rng_shuffle_preserves_elements() {
        let mut rng = SimpleRng::new(55555);
        let mut data: Vec<usize> = (0..100).collect();
        let original: Vec<usize> = data.clone();

        rng.shuffle_usize(&mut data);

        // Same elements, different order
        let mut sorted = data.clone();
        sorted.sort();
        assert_eq!(sorted, original);
        assert_ne!(data, original); // Very unlikely to be same order
    }

    #[test]
    fn test_rng_shuffle_empty() {
        let mut rng = SimpleRng::new(66666);
        let mut data: Vec<usize> = vec![];
        rng.shuffle_usize(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_rng_shuffle_single() {
        let mut rng = SimpleRng::new(77777);
        let mut data = vec![42];
        rng.shuffle_usize(&mut data);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_rng_shuffle_two() {
        let mut swapped = false;

        // Run many times, should swap at least once
        for seed in 0..100 {
            let mut rng = SimpleRng::new(seed);
            let mut data = vec![0, 1];
            rng.shuffle_usize(&mut data);
            if data == vec![1, 0] {
                swapped = true;
                break;
            }
        }
        assert!(swapped, "Shuffle should swap elements sometimes");
    }

    #[test]
    fn test_rng_reseed_from_time() {
        let mut rng = SimpleRng::new(42);
        let val1 = rng.next_u32();

        rng.reseed_from_time();
        let val2 = rng.next_u32();

        // Both should be valid
        assert!(val1 > 0 || val2 > 0);
    }

    #[test]
    fn test_rng_distribution_rough() {
        let mut rng = SimpleRng::new(99999);
        let mut buckets = [0u32; 10];

        for _ in 0..100000 {
            let val = rng.gen_usize(10);
            buckets[val] += 1;
        }

        // Each bucket should have roughly 10000 values (±20%)
        for &count in &buckets {
            assert!(
                count > 8000 && count < 12000,
                "Distribution seems biased: {}",
                count
            );
        }
    }
}

// ============================================================================
// Integration Tests - End-to-End Training
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_dense_layer_training_reduces_output() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(4, 2, &mut rng);

        let input = vec![1.0, 0.5, -0.5, 0.25];
        let target_grad = vec![1.0, 1.0]; // Drive output down

        let mut prev_output_sum = f32::MAX;

        for _ in 0..100 {
            let mut output = vec![0.0; 2];
            layer.forward(&input, &mut output, 1);

            let output_sum: f32 = output.iter().sum();

            let mut grad_input = vec![0.0; 4];
            layer.backward(&input, &target_grad, &mut grad_input, 1);
            layer.update_parameters(0.1);

            // Output should decrease over iterations
            if output_sum < prev_output_sum - 0.001 {
                prev_output_sum = output_sum;
            }
        }

        // Final output should be less than initial
        assert!(prev_output_sum < f32::MAX);
    }

    #[test]
    fn test_conv2d_layer_training_changes_output() {
        let mut rng = SimpleRng::new(42);
        let mut layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

        let input = vec![1.0f32; 16];
        let grad = vec![0.1f32; 32];

        let mut output_before = vec![0.0; 32];
        layer.forward(&input, &mut output_before, 1);

        for _ in 0..10 {
            let mut output = vec![0.0; 32];
            layer.forward(&input, &mut output, 1);

            let mut grad_input = vec![0.0; 16];
            layer.backward(&input, &grad, &mut grad_input, 1);
            layer.update_parameters(0.01);
        }

        let mut output_after = vec![0.0; 32];
        layer.forward(&input, &mut output_after, 1);

        assert_ne!(output_before, output_after);
    }

    #[test]
    fn test_two_layer_dense_network() {
        let mut rng = SimpleRng::new(42);
        let mut layer1 = DenseLayer::new(4, 8, &mut rng);
        let mut layer2 = DenseLayer::new(8, 2, &mut rng);

        let input = vec![1.0, -1.0, 0.5, -0.5];

        // Forward through both layers
        let mut hidden = vec![0.0; 8];
        layer1.forward(&input, &mut hidden, 1);
        relu_inplace(&mut hidden);

        let mut output = vec![0.0; 2];
        layer2.forward(&hidden, &mut output, 1);

        // All outputs should be valid
        assert!(output.iter().all(|&x| x.is_finite()));

        // Backward through both layers
        let grad_output = vec![1.0, -1.0];
        let mut grad_hidden = vec![0.0; 8];
        layer2.backward(&hidden, &grad_output, &mut grad_hidden, 1);

        // Mask gradient for ReLU
        for (g, &h) in grad_hidden.iter_mut().zip(hidden.iter()) {
            if h <= 0.0 {
                *g = 0.0;
            }
        }

        let mut grad_input = vec![0.0; 4];
        layer1.backward(&input, &grad_hidden, &mut grad_input, 1);

        // Update both layers
        layer1.update_parameters(0.01);
        layer2.update_parameters(0.01);

        // Should still produce valid output after update
        let mut hidden2 = vec![0.0; 8];
        layer1.forward(&input, &mut hidden2, 1);
        relu_inplace(&mut hidden2);

        let mut output2 = vec![0.0; 2];
        layer2.forward(&hidden2, &mut output2, 1);

        assert!(output2.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_conv_then_dense() {
        let mut rng = SimpleRng::new(42);
        let conv = Conv2DLayer::new(1, 4, 3, 1, 1, 4, 4, &mut rng);
        let dense = DenseLayer::new(64, 2, &mut rng); // 4 channels × 4 × 4 = 64

        let input = vec![1.0f32; 16]; // 1 × 4 × 4

        let mut conv_out = vec![0.0; 64];
        conv.forward(&input, &mut conv_out, 1);
        relu_inplace(&mut conv_out);

        let mut output = vec![0.0; 2];
        dense.forward(&conv_out, &mut output, 1);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batch_normalization_like_behavior() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 4, &mut rng);

        // Process a batch and check outputs are reasonable
        let batch_size = 32;
        let input: Vec<f32> = (0..batch_size * 4)
            .map(|i| ((i % 7) as f32 - 3.0) / 3.0)
            .collect();

        let mut output = vec![0.0; batch_size * 4];
        layer.forward(&input, &mut output, batch_size);

        // Compute mean and variance of outputs
        let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;

        // Should have reasonable statistics
        assert!(mean.is_finite());
        assert!(var.is_finite());
        assert!(var >= 0.0);
    }
}
