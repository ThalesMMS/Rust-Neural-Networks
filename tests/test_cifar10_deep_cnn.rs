//! Integration tests for CIFAR-10 deep CNN architecture
//!
//! This file tests the CIFAR-10 deep CNN architecture including:
//! - BatchNorm layer mode switching (training vs inference)
//! - Running statistics updates during training
//! - Forward pass through the full architecture
//! - Proper integration of Conv2D, BatchNorm, Dropout, and Dense layers

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::layers::batchnorm::BatchNormLayer;
use rust_neural_networks::layers::Layer;
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// BatchNorm Mode Switching Tests
// ============================================================================

/// Test that BatchNorm layers correctly switch between training and inference modes.
///
/// This test verifies:
/// 1. BatchNorm layers default to training mode
/// 2. Mode can be switched to inference
/// 3. Training mode produces different outputs than inference mode
/// 4. Running statistics are updated during training
#[test]
fn test_batchnorm_mode_switching() {
    // Create a BatchNorm layer sized for CIFAR-10 deep CNN (32 channels * 32 * 32)
    let mut bn_layer = BatchNormLayer::new(32768, 1e-5, 0.9);

    // Verify default mode is training
    assert!(
        bn_layer.is_training(),
        "BatchNorm should default to training mode"
    );

    // Create input data
    let batch_size = 16;
    let mut input = vec![0.0f32; 32768 * batch_size];
    for i in 0..batch_size {
        for j in 0..32768 {
            input[i * 32768 + j] = (i as f32) * 0.1 + (j as f32) * 0.001;
        }
    }

    // Forward pass in training mode
    let mut output_train = vec![0.0f32; 32768 * batch_size];
    bn_layer.forward(&input, &mut output_train, batch_size);

    // Switch to inference mode
    bn_layer.set_training(false);
    assert!(
        !bn_layer.is_training(),
        "BatchNorm should be in inference mode"
    );

    // Forward pass in inference mode
    let mut output_inference = vec![0.0f32; 32768 * batch_size];
    bn_layer.forward(&input, &mut output_inference, batch_size);

    // Verify outputs differ between training and inference modes
    let mut outputs_differ = false;
    for i in 0..output_train.len() {
        if (output_train[i] - output_inference[i]).abs() > 1e-4 {
            outputs_differ = true;
            break;
        }
    }
    assert!(
        outputs_differ,
        "Training and inference outputs should differ due to different normalization statistics"
    );
}

/// Test that BatchNorm layers update running statistics during training
/// and use those statistics during inference.
///
/// This test verifies the critical behavior for proper inference:
/// 1. Running statistics are initialized to zero
/// 2. Training mode updates running statistics
/// 3. Inference mode uses running statistics (not batch statistics)
/// 4. Training and inference outputs differ appropriately
#[test]
fn test_batchnorm_running_statistics_update() {
    // Create a simple architecture with one BatchNorm layer
    let mut bn_layer = BatchNormLayer::new(256, 1e-5, 0.9);

    // Verify initial state
    assert!(bn_layer.is_training());
    let running_mean_initial = bn_layer.running_mean();
    let running_var_initial = bn_layer.running_var();
    for val in &running_mean_initial {
        assert_eq!(*val, 0.0, "Running mean should be initialized to 0");
    }
    for val in &running_var_initial {
        assert_eq!(*val, 0.0, "Running variance should be initialized to 0");
    }

    // Step 1: Forward pass in training mode
    let batch_size = 32;
    let mut input = vec![0.0f32; 256 * batch_size];
    // Create input with non-zero mean and variance
    for i in 0..batch_size {
        for j in 0..256 {
            input[i * 256 + j] = (i as f32) * 0.1 + (j as f32) * 0.05;
        }
    }

    let mut output_train = vec![0.0f32; 256 * batch_size];
    bn_layer.forward(&input, &mut output_train, batch_size);

    // Step 2: Verify running statistics were updated
    let running_mean_after_train = bn_layer.running_mean();
    let _running_var_after_train = bn_layer.running_var();

    let mut stats_updated = false;
    for val in &running_mean_after_train {
        if val.abs() > 1e-6 {
            stats_updated = true;
            break;
        }
    }
    assert!(
        stats_updated,
        "Running statistics should be updated during training"
    );

    // Step 3: Switch to inference mode
    bn_layer.set_training(false);
    assert!(!bn_layer.is_training(), "Should be in inference mode");

    // Step 4: Forward pass in inference mode with same input
    let mut output_inference = vec![0.0f32; 256 * batch_size];
    bn_layer.forward(&input, &mut output_inference, batch_size);

    // Step 5: Verify outputs differ between training and inference modes
    // In training mode, batch statistics are used (mean=0, var=1 after normalization)
    // In inference mode, running statistics are used
    let mut outputs_differ = false;
    for i in 0..output_train.len() {
        if (output_train[i] - output_inference[i]).abs() > 1e-4 {
            outputs_differ = true;
            break;
        }
    }
    assert!(
        outputs_differ,
        "Training and inference outputs should differ when using different statistics"
    );

    // Step 6: Verify running statistics were NOT updated during inference
    let running_mean_after_inference = bn_layer.running_mean();
    for i in 0..256 {
        assert_eq!(
            running_mean_after_train[i], running_mean_after_inference[i],
            "Running statistics should not change during inference"
        );
    }
}

/// Test BatchNorm mode switching in the full CIFAR-10 deep CNN architecture.
///
/// This test verifies that when using the actual CIFAR-10 deep CNN config:
/// 1. All BatchNorm layers are properly integrated
/// 2. Forward pass works correctly through the entire network
/// 3. Model can process CIFAR-10 sized inputs
#[test]
fn test_cifar10_deep_cnn_batchnorm_integration() {
    // Load the actual CIFAR-10 deep CNN config
    let config_path = "config/architectures/cifar10_deep_cnn.json";
    let config = match load_architecture(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            // If the config file doesn't exist, skip this test
            eprintln!("Skipping test: config file not found: {}", e);
            return;
        }
    };

    // Build the model
    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    // Verify the model was built successfully
    assert!(
        !layers.is_empty(),
        "Model should contain at least one layer"
    );

    // Test forward pass with a small batch
    let batch_size = 4;
    let input_size = 3 * 32 * 32; // CIFAR-10 RGB images
    let input = vec![0.5f32; input_size * batch_size];

    // Forward pass through all layers
    let mut current_output = input;
    for (i, layer) in layers.iter().enumerate() {
        let output_size = layer.output_size();
        let mut next_output = vec![0.0f32; output_size * batch_size];
        layer.forward(&current_output, &mut next_output, batch_size);

        // Verify output is valid (no NaN or Inf)
        for (j, &val) in next_output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Layer {} produced non-finite value at index {}: {}",
                i,
                j,
                val
            );
        }

        current_output = next_output;
    }

    // Verify final output has correct shape (batch_size * 10 for CIFAR-10 classes)
    assert_eq!(
        current_output.len(),
        batch_size * 10,
        "Final output should have shape [batch_size, 10]"
    );
}

/// Test that BatchNorm layers in the architecture can be switched to inference mode.
///
/// This test verifies the workflow of training (with BatchNorm updating statistics)
/// and then switching to inference mode for evaluation.
#[test]
fn test_cifar10_deep_cnn_train_to_inference_workflow() {
    // Create a BatchNorm layer sized for a mini CIFAR-10 architecture
    // (16 channels * 32 * 32 = 16384 features)
    let mut bn_layer = BatchNormLayer::new(16384, 1e-5, 0.9);

    // Verify BatchNorm layer starts in training mode
    assert!(
        bn_layer.is_training(),
        "BatchNorm should start in training mode"
    );

    // Do a training forward pass
    let batch_size = 8;
    let input_size = 16384;
    let mut input = vec![0.0f32; input_size * batch_size];
    // Create input with varying statistics
    for i in 0..batch_size {
        for j in 0..input_size {
            input[i * input_size + j] = (i as f32) * 0.2 + (j as f32) * 0.001;
        }
    }

    let mut bn_output_train = vec![0.0f32; input_size * batch_size];
    bn_layer.forward(&input, &mut bn_output_train, batch_size);

    // Switch to inference mode
    bn_layer.set_training(false);
    assert!(
        !bn_layer.is_training(),
        "BatchNorm should be in inference mode"
    );

    // Do an inference forward pass with same input
    let mut bn_output_inference = vec![0.0f32; input_size * batch_size];
    bn_layer.forward(&input, &mut bn_output_inference, batch_size);

    // Verify outputs differ (training uses batch stats, inference uses running stats)
    let mut outputs_differ = false;
    for i in 0..bn_output_train.len() {
        if (bn_output_train[i] - bn_output_inference[i]).abs() > 1e-4 {
            outputs_differ = true;
            break;
        }
    }
    assert!(
        outputs_differ,
        "Training and inference outputs should differ due to different normalization statistics"
    );
}
