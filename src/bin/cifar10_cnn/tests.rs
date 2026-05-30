use super::*;

#[test]
fn test_gather_batch() {
    let images = vec![1.0; 3072 * 3]; // 3 images
    let labels = vec![0u8, 1u8, 2u8];
    let indices = vec![0, 1, 2];
    let mut out_inputs = vec![0.0; 3072 * 2]; // batch of 2
    let mut out_labels = vec![0u8; 2];

    gather_batch(
        &images,
        &labels,
        &indices,
        0,
        2,
        &mut out_inputs,
        &mut out_labels,
        IMG_W,        // img_width
        IMG_H,        // img_height
        IMG_CHANNELS, // img_channels
        None,         // flip_prob
        None,         // crop_padding
        None,         // brightness_jitter
        None,         // contrast_jitter
        None,         // saturation_jitter
        None,         // rng
    );

    assert_eq!(out_labels[0], 0);
    assert_eq!(out_labels[1], 1);
}

#[test]
fn test_forward_pass_relu_only_after_conv2d() {
    // Build a small deep CNN: Conv2D -> Conv2D -> Conv2D -> Dense
    // Input: 1 channel, 4x4 images
    let mut rng = SimpleRng::new(42);

    // Layer 0: Conv2D (in_ch=1, out_ch=2, kernel=3, padding=1, stride=1, H=4, W=4) -> output 2*4*4=32
    let conv1 = Conv2DLayer::new(1, 2, 3, 1isize, 1, 4, 4, &mut rng);
    // Layer 1: Conv2D (in_ch=2, out_ch=4, kernel=3, padding=1, stride=1, H=4, W=4) -> output 4*4*4=64
    let conv2 = Conv2DLayer::new(2, 4, 3, 1isize, 1, 4, 4, &mut rng);
    // Layer 2: Conv2D (in_ch=4, out_ch=2, kernel=3, padding=1, stride=1, H=4, W=4) -> output 2*4*4=32
    let conv3 = Conv2DLayer::new(4, 2, 3, 1isize, 1, 4, 4, &mut rng);
    // Layer 3: Dense (32 -> 10) - no ReLU should be applied here
    let dense = DenseLayer::new(32, 10, &mut rng);

    let mut model = Cnn {
        layers: vec![
            Box::new(conv1),
            Box::new(conv2),
            Box::new(conv3),
            Box::new(dense),
        ],
    };

    let batch_size = 2;
    let input_size = 4 * 4; // 1 channel, 4x4 image
    let input = vec![1.0f32; batch_size * input_size];

    let num_layers = model.layers.len();
    let mut activations = LayerActivations::new(num_layers);
    let mut temp_buffer = Vec::new();

    let output_idx = forward_pass(
        &mut model,
        batch_size,
        &input,
        &mut activations,
        &mut temp_buffer,
    );

    // Output index should be the last layer
    assert_eq!(
        output_idx, 3,
        "Output should be from the last layer (index 3)"
    );

    // Verify ReLU only applied after Conv2D layers (is_conv flag)
    assert!(
        activations.is_conv[0],
        "Layer 0 (Conv2D) should be marked as conv"
    );
    assert!(
        activations.is_conv[1],
        "Layer 1 (Conv2D) should be marked as conv"
    );
    assert!(
        activations.is_conv[2],
        "Layer 2 (Conv2D) should be marked as conv"
    );
    // Dense layer should NOT be marked as conv (no ReLU)
    assert!(
        !activations.is_conv[3],
        "Layer 3 (Dense) should NOT be marked as conv"
    );

    // All Conv2D layer activations must be non-negative (ReLU was applied)
    for &val in &activations.data[0] {
        assert!(
            val >= 0.0,
            "Conv2D layer 0 output must be >= 0 after ReLU, got {}",
            val
        );
    }
    for &val in &activations.data[1] {
        assert!(
            val >= 0.0,
            "Conv2D layer 1 output must be >= 0 after ReLU, got {}",
            val
        );
    }
    for &val in &activations.data[2] {
        assert!(
            val >= 0.0,
            "Conv2D layer 2 output must be >= 0 after ReLU, got {}",
            val
        );
    }

    // Dense layer output has the expected size (ReLU not applied, values unconstrained)
    assert_eq!(
        activations.data[3].len(),
        batch_size * 10,
        "Dense layer output should have batch_size * num_classes elements"
    );
}

#[test]
fn test_accuracy_restores_training_mode() {
    let mut rng = SimpleRng::new(42);
    let dropout = DropoutLayer::new(NUM_INPUTS, 0.5, &mut rng);
    let dense = DenseLayer::new(NUM_INPUTS, NUM_CLASSES, &mut rng);
    let mut model = Cnn {
        layers: vec![Box::new(dropout), Box::new(dense)],
    };

    set_training_mode(&mut model, true);
    let images = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let labels = vec![0u8; BATCH_SIZE];

    let _ = test_accuracy(&mut model, &images, &labels);

    let dropout_layer = model.layers[0]
        .as_any()
        .downcast_ref::<DropoutLayer>()
        .expect("first layer should be dropout");
    assert!(dropout_layer.is_training());
}

#[test]
fn test_accuracy_invalid_inputs_return_zero() {
    let mut model = Cnn { layers: vec![] };

    assert_eq!(test_accuracy(&mut model, &[], &[]), 0.0);
    assert_eq!(test_accuracy(&mut model, &[], &[0]), 0.0);
}

#[test]
fn test_backward_pass_masks_conv_relu_before_backward() {
    let mut rng = SimpleRng::new(42);
    let conv = Conv2DLayer::new(1, 1, 3, 1isize, 1, 4, 4, &mut rng);
    let mut model = Cnn {
        layers: vec![Box::new(conv)],
    };

    let input = vec![0.0f32; 4 * 4];
    let mut activations = LayerActivations::new(1);
    let mut temp_buffer = Vec::new();
    let output_idx = forward_pass(&mut model, 1, &input, &mut activations, &mut temp_buffer);
    let initial_grad = vec![1.0f32; activations.data[output_idx].len()];
    let mut grad_buffer1 = Vec::new();
    let mut grad_buffer2 = vec![123.0f32; input.len()];

    backward_pass(
        &mut model,
        1,
        &input,
        &activations,
        &initial_grad,
        &mut grad_buffer1,
        &mut grad_buffer2,
    );

    let conv_layer = model.layers[0]
        .as_any()
        .downcast_ref::<Conv2DLayer>()
        .expect("layer should be Conv2D");
    let (weight_norm, bias_norm) = conv_layer.get_gradient_magnitude();
    assert_eq!(weight_norm, 0.0);
    assert_eq!(bias_norm, 0.0);
}

#[test]
fn test_create_optimizer_from_config() {
    use rust_neural_networks::config::TrainingConfig;

    // Helper to build a minimal TrainingConfig with a given optimizer_type
    /// Create a TrainingConfig populated with default values (most fields set to `None`) and an optional optimizer type.
    ///
    /// The returned config sets `scheduler_type` to `"none"` and assigns `optimizer_type` from the function argument; all other training, scheduler, optimizer, and augmentation fields remain `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let cfg = make_config(Some("adam"));
    /// assert_eq!(cfg.scheduler_type, "none");
    /// assert_eq!(cfg.optimizer_type.as_deref(), Some("adam"));
    /// ```
    fn make_config(optimizer_type: Option<&str>) -> TrainingConfig {
        TrainingConfig {
            scheduler_type: "none".to_string(),
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
            optimizer_type: optimizer_type.map(|s| s.to_string()),
            adam_beta1: None,
            adam_beta2: None,
            adam_epsilon: None,
            adamw_weight_decay: None,
            rmsprop_decay: None,
            rmsprop_epsilon: None,
            learning_rate: None,
            epochs: None,
            batch_size: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            enable_profiling: None,
            enable_augmentation: None,
            horizontal_flip_prob: None,
            random_crop_padding: None,
            brightness_jitter: None,
            contrast_jitter: None,
            saturation_jitter: None,
            noise_dim: None,
            g_lr: None,
            d_lr: None,
            label_smoothing: None,
            gpu_backend: None,
            gpu_device_id: None,
            step_debug: None,
            warmup: None,
            cyclical_lr: None,
            regularization: None,
            gradient_clipping: None,
        }
    }

    let lr = 0.01f32;

    // Test SGD optimizer
    let config = make_config(Some("sgd"));
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "SGD optimizer should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test Adam optimizer
    let config = make_config(Some("adam"));
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "Adam optimizer should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test AdamW optimizer
    let config = make_config(Some("adamw"));
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "AdamW optimizer should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test RMSprop optimizer
    let config = make_config(Some("rmsprop"));
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "RMSprop optimizer should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test None optimizer_type defaults to AdamW (learning_rate should be set correctly)
    let config = make_config(None);
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "Default (None) optimizer should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test unknown optimizer_type falls back to AdamW
    let config = make_config(Some("unknown_type"));
    let opt = create_optimizer(&config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "Unknown optimizer type should fall back to AdamW with learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test with non-default hyperparameters (Adam with custom beta1/beta2/epsilon)
    let mut adam_config = make_config(Some("adam"));
    adam_config.adam_beta1 = Some(0.95);
    adam_config.adam_beta2 = Some(0.998);
    adam_config.adam_epsilon = Some(1e-7);
    let opt = create_optimizer(&adam_config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "Adam optimizer with custom hyperparams should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );

    // Test with AdamW weight decay set
    let mut adamw_config = make_config(Some("adamw"));
    adamw_config.adamw_weight_decay = Some(0.001);
    let opt = create_optimizer(&adamw_config, lr);
    assert!(
        (opt.learning_rate() - lr).abs() < 1e-6,
        "AdamW optimizer with custom weight_decay should have learning_rate {}, got {}",
        lr,
        opt.learning_rate()
    );
}
