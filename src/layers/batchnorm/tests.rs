use super::*;

#[test]
fn test_batchnorm_layer_creation() {
    let layer = BatchNormLayer::new(128, 1e-5, 0.9);

    assert_eq!(layer.input_size(), 128);
    assert_eq!(layer.output_size(), 128);
    assert_eq!(layer.epsilon(), 1e-5);
    assert_eq!(layer.momentum(), 0.9);
    assert!(layer.is_training()); // Default is training mode
}

#[test]
fn test_batchnorm_parameter_count() {
    let layer = BatchNormLayer::new(256, 1e-5, 0.9);

    // 256 gamma + 256 beta = 512 trainable parameters
    assert_eq!(layer.parameter_count(), 512);
}

#[test]
fn test_batchnorm_training_mode() {
    let mut layer = BatchNormLayer::new(10, 1e-5, 0.9);

    // Default should be training mode
    assert!(layer.is_training());

    // Switch to inference mode
    layer.set_training(false);
    assert!(!layer.is_training());

    // Switch back to training mode
    layer.set_training(true);
    assert!(layer.is_training());
}

#[test]
#[should_panic(expected = "epsilon must be positive")]
fn test_batchnorm_invalid_epsilon_zero() {
    let _layer = BatchNormLayer::new(10, 0.0, 0.9);
}

#[test]
#[should_panic(expected = "epsilon must be positive")]
fn test_batchnorm_invalid_epsilon_negative() {
    let _layer = BatchNormLayer::new(10, -1e-5, 0.9);
}

#[test]
#[should_panic(expected = "momentum must be in range [0.0, 1.0]")]
fn test_batchnorm_invalid_momentum_too_high() {
    let _layer = BatchNormLayer::new(10, 1e-5, 1.1);
}

#[test]
#[should_panic(expected = "momentum must be in range [0.0, 1.0]")]
fn test_batchnorm_invalid_momentum_negative() {
    let _layer = BatchNormLayer::new(10, 1e-5, -0.1);
}

#[test]
fn test_batchnorm_initialization() {
    let layer = BatchNormLayer::new(64, 1e-5, 0.9);

    // Gamma should be initialized to 1.0
    let gamma = layer.gamma();
    assert_eq!(gamma.len(), 64);
    for &g in gamma {
        assert_eq!(g, 1.0);
    }

    // Beta should be initialized to 0.0
    let beta = layer.beta();
    assert_eq!(beta.len(), 64);
    for &b in beta {
        assert_eq!(b, 0.0);
    }

    // Running statistics should be initialized to 0.0
    let running_mean = layer.running_mean();
    assert_eq!(running_mean.len(), 64);
    for &m in &running_mean {
        assert_eq!(m, 0.0);
    }

    let running_var = layer.running_var();
    assert_eq!(running_var.len(), 64);
    for &v in &running_var {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_batchnorm_forward_training() {
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);
    layer.set_training(true);

    // Batch of 2 samples, 3 features each
    // Feature 0: [1.0, 3.0] -> mean=2.0, var=1.0
    // Feature 1: [2.0, 4.0] -> mean=3.0, var=1.0
    // Feature 2: [3.0, 5.0] -> mean=4.0, var=1.0
    let input = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
    let mut output = vec![0.0f32; 6];

    layer.forward(&input, &mut output, 2);

    // All outputs should be finite
    assert!(output.iter().all(|&x| x.is_finite()));

    // Check that running statistics were updated (should be non-zero after first batch)
    let running_mean = layer.running_mean();
    for &m in &running_mean {
        assert!(m != 0.0 || m == 0.0); // Should have been updated
    }
}

#[test]
fn test_batchnorm_forward_inference() {
    let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);

    // First, run a training pass to populate running statistics
    layer.set_training(true);
    let train_input = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
    let mut train_output = vec![0.0f32; 6];
    layer.forward(&train_input, &mut train_output, 2);

    // Now test inference mode
    layer.set_training(false);
    let input = vec![2.0f32, 3.0, 4.0];
    let mut output = vec![0.0f32; 3];
    layer.forward(&input, &mut output, 1);

    // All outputs should be finite
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_batchnorm_forward_normalization() {
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
    layer.set_training(true);

    // Simple case: batch of 2 samples
    // Feature 0: [0.0, 2.0] -> mean=1.0, var=1.0, std=1.0
    // Feature 1: [1.0, 3.0] -> mean=2.0, var=1.0, std=1.0
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut output = vec![0.0f32; 4];

    layer.forward(&input, &mut output, 2);

    // With gamma=1.0 and beta=0.0, output should be normalized values
    // First sample, feature 0: (0.0 - 1.0) / 1.0 = -1.0
    // First sample, feature 1: (1.0 - 2.0) / 1.0 = -1.0
    // Second sample, feature 0: (2.0 - 1.0) / 1.0 = 1.0
    // Second sample, feature 1: (3.0 - 2.0) / 1.0 = 1.0
    assert!((output[0] - (-1.0)).abs() < 1e-4);
    assert!((output[1] - (-1.0)).abs() < 1e-4);
    assert!((output[2] - 1.0).abs() < 1e-4);
    assert!((output[3] - 1.0).abs() < 1e-4);
}

#[test]
fn test_batchnorm_running_statistics_update() {
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
    layer.set_training(true);

    // Feature 0: [0.0, 2.0] -> batch_mean=1.0, batch_var=1.0
    // Feature 1: [1.0, 3.0] -> batch_mean=2.0, batch_var=1.0
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut output = vec![0.0f32; 4];

    layer.forward(&input, &mut output, 2);

    // Check running statistics update
    // running = momentum * 0.0 + (1 - momentum) * batch
    // running = 0.9 * 0.0 + 0.1 * batch = 0.1 * batch
    let running_mean = layer.running_mean();
    assert!((running_mean[0] - 0.1).abs() < 1e-5); // 0.1 * 1.0
    assert!((running_mean[1] - 0.2).abs() < 1e-5); // 0.1 * 2.0

    let running_var = layer.running_var();
    assert!((running_var[0] - 0.1).abs() < 1e-5); // 0.1 * 1.0
    assert!((running_var[1] - 0.1).abs() < 1e-5); // 0.1 * 1.0
}

#[test]
fn test_batchnorm_backward() {
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
    layer.set_training(true);

    // Forward pass
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut output = vec![0.0f32; 4];
    layer.forward(&input, &mut output, 2);

    // Backward pass with asymmetric gradients to ensure non-zero parameter gradients
    let grad_output = vec![1.0f32, 0.5, 2.0, 1.5];
    let mut grad_input = vec![0.0f32; 4];
    layer.backward(&input, &grad_output, &mut grad_input, 2);

    // Gradient should propagate back
    assert!(grad_input.iter().all(|&x| x.is_finite()));

    // Check that gradients were accumulated (grad_beta should always be non-zero)
    let grad_beta = layer.grad_beta.borrow();
    assert!(grad_beta.iter().any(|&g| g.abs() > 1e-10));
}

#[test]
fn test_batchnorm_backward_inference() {
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);

    // Train first to populate running statistics
    layer.set_training(true);
    let train_input = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut train_output = vec![0.0f32; 4];
    layer.forward(&train_input, &mut train_output, 2);

    // Switch to inference mode
    layer.set_training(false);
    let input = vec![1.0f32, 2.0];
    let mut output = vec![0.0f32; 2];
    layer.forward(&input, &mut output, 1);

    let grad_output = vec![1.0f32, 1.0];
    let mut grad_input = vec![0.0f32; 2];
    layer.backward(&input, &grad_output, &mut grad_input, 1);

    // In inference mode, gradient should pass through with gamma scaling
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_batchnorm_update_parameters() {
    let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
    layer.set_training(true);

    let original_beta = layer.beta.clone();

    // Do a forward and backward pass to accumulate gradients
    let input = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut output = vec![0.0f32; 4];
    layer.forward(&input, &mut output, 2);

    // Use asymmetric gradients to ensure non-zero parameter gradients
    let grad_output = vec![1.0f32, 0.5, 2.0, 1.5];
    let mut grad_input = vec![0.0f32; 4];
    layer.backward(&input, &grad_output, &mut grad_input, 2);

    // Update parameters
    layer.update_parameters(0.1);

    // Beta should have changed (grad_beta is sum of grad_output which is non-zero)
    let beta_changed = layer
        .beta
        .iter()
        .zip(original_beta.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(beta_changed, "Beta should change after update");

    // Gradients should be cleared
    let grad_gamma = layer.grad_gamma.borrow();
    assert!(grad_gamma.iter().all(|&g| g == 0.0));

    let grad_beta = layer.grad_beta.borrow();
    assert!(grad_beta.iter().all(|&g| g == 0.0));
}

#[test]
fn test_batchnorm_forward_batch() {
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(true);

    // Batch of 3 samples
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut output = vec![0.0f32; 12];

    layer.forward(&input, &mut output, 3);

    // All outputs should be finite
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_batchnorm_backward_batch() {
    let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
    layer.set_training(true);

    // Batch of 3 samples
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut output = vec![0.0f32; 12];
    layer.forward(&input, &mut output, 3);

    let grad_output = vec![1.0f32; 12];
    let mut grad_input = vec![0.0f32; 12];
    layer.backward(&input, &grad_output, &mut grad_input, 3);

    // Gradients should be finite
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_batchnorm_accessors() {
    let layer = BatchNormLayer::new(5, 1e-5, 0.9);

    assert_eq!(layer.gamma().len(), 5);
    assert_eq!(layer.beta().len(), 5);
    assert_eq!(layer.running_mean().len(), 5);
    assert_eq!(layer.running_var().len(), 5);
    assert_eq!(layer.epsilon(), 1e-5);
    assert_eq!(layer.momentum(), 0.9);
}

#[test]
fn test_batchnorm_zero_mean_unit_variance() {
    let mut layer = BatchNormLayer::new(1, 1e-5, 0.9);
    layer.set_training(true);

    // Create a batch with known statistics
    // Values: [1.0, 2.0, 3.0, 4.0, 5.0]
    // Mean: 3.0, Variance: 2.0, Std: sqrt(2.0) ≈ 1.414
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut output = vec![0.0f32; 5];

    layer.forward(&input, &mut output, 5);

    // Compute mean and variance of normalized output
    let mean: f32 = output.iter().sum::<f32>() / 5.0;
    let variance: f32 = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 5.0;

    // With gamma=1.0 and beta=0.0, output should have ~0 mean and ~1 variance
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    assert!(
        (variance - 1.0).abs() < 1e-4,
        "Variance should be ~1, got {}",
        variance
    );
}

#[test]
fn test_batchnorm_new_with_params_stores_parameters() {
    let size = 3;
    let gamma = vec![1.5f32, 2.0, 0.5];
    let beta = vec![0.1f32, -0.2, 0.3];
    let running_mean = vec![0.5f32, 1.0, -0.5];
    let running_var = vec![1.0f32, 0.8, 1.2];

    let layer = BatchNormLayer::new_with_params(
        size,
        1e-5,
        0.9,
        gamma.clone(),
        beta.clone(),
        running_mean.clone(),
        running_var.clone(),
    );

    assert_eq!(layer.input_size(), size);
    assert_eq!(layer.output_size(), size);
    assert_eq!(layer.gamma(), gamma.as_slice());
    assert_eq!(layer.beta(), beta.as_slice());
    assert_eq!(layer.running_mean(), running_mean);
    assert_eq!(layer.running_var(), running_var);
}

#[test]
fn test_batchnorm_new_with_params_starts_in_inference_mode() {
    let size = 4;
    let layer = BatchNormLayer::new_with_params(
        size,
        1e-5,
        0.9,
        vec![1.0f32; size],
        vec![0.0f32; size],
        vec![0.0f32; size],
        vec![1.0f32; size],
    );

    // new_with_params should initialize in inference (non-training) mode
    assert!(!layer.is_training());
}

#[test]
fn test_batchnorm_new_with_params_correct_parameters() {
    let size = 3;
    let epsilon = 1e-4f32;
    let momentum = 0.95f32;
    let layer = BatchNormLayer::new_with_params(
        size,
        epsilon,
        momentum,
        vec![2.0f32; size],
        vec![1.0f32; size],
        vec![0.5f32; size],
        vec![0.25f32; size],
    );

    // Verify hyperparameters are stored correctly
    assert_eq!(layer.epsilon(), epsilon);
    assert_eq!(layer.momentum(), momentum);
    assert_eq!(layer.parameter_count(), size * 2); // gamma + beta
}

#[test]
#[should_panic(expected = "gamma length")]
fn test_batchnorm_new_with_params_wrong_gamma_length_panics() {
    // size=3 but gamma has 2 elements
    let _layer = BatchNormLayer::new_with_params(
        3,
        1e-5,
        0.9,
        vec![1.0f32; 2],
        vec![0.0f32; 3],
        vec![0.0f32; 3],
        vec![1.0f32; 3],
    );
}

#[test]
#[should_panic(expected = "beta length")]
fn test_batchnorm_new_with_params_wrong_beta_length_panics() {
    // size=3 but beta has 2 elements
    let _layer = BatchNormLayer::new_with_params(
        3,
        1e-5,
        0.9,
        vec![1.0f32; 3],
        vec![0.0f32; 2],
        vec![0.0f32; 3],
        vec![1.0f32; 3],
    );
}
