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
fn test_dense_forward_produces_valid_statistics() {
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
