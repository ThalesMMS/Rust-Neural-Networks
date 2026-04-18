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
