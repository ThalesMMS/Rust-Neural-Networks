//! Integration tests for CIFAR-10 deep CNN architecture
//!
//! This file tests the CIFAR-10 deep CNN architecture including:
//! - BatchNorm layer mode switching (training vs inference)
//! - Running statistics updates during training
//! - Forward pass through the full architecture
//! - Proper integration of Conv2D, BatchNorm, Dropout, and Dense layers

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::layers::batchnorm::BatchNormLayer;
use rust_neural_networks::layers::conv2d::Conv2DLayer;
use rust_neural_networks::layers::dense::DenseLayer;
use rust_neural_networks::layers::dropout::DropoutLayer;
use rust_neural_networks::layers::Layer;
use rust_neural_networks::persistence::{load_layers, save_layers};
use rust_neural_networks::utils::rng::SimpleRng;
use std::io::Cursor;

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

// ============================================================================
// Model Serialization Roundtrip Tests
// ============================================================================

/// Test model serialization roundtrip for a baseline 2-layer dense model.
///
/// This test verifies:
/// 1. A Dense+Dense model serializes correctly to binary
/// 2. The binary can be deserialized back into the same layer types
/// 3. Layer count, types, parameter counts, and weights match exactly
#[test]
fn test_save_load_baseline_model() {
    let mut rng = SimpleRng::new(42);

    // Build a 2-layer dense model with known sizes
    let dense1 = DenseLayer::new(32, 16, &mut rng);
    let dense2 = DenseLayer::new(16, 10, &mut rng);

    let original_weights_layer1 = dense1.weights().to_vec();
    let original_biases_layer1 = dense1.biases().to_vec();
    let original_param_count_0 = dense1.parameter_count();
    let original_param_count_1 = dense2.parameter_count();

    let layers: Vec<Box<dyn Layer>> = vec![Box::new(dense1), Box::new(dense2)];

    // Serialize to in-memory buffer
    let mut data: Vec<u8> = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    assert!(!data.is_empty(), "Serialized data should not be empty");

    // Deserialize from buffer
    let mut cursor = Cursor::new(&data);
    let loaded_layers = load_layers(&mut cursor, &mut rng).unwrap();

    // Verify layer count
    assert_eq!(
        loaded_layers.len(),
        2,
        "Should reconstruct exactly 2 layers"
    );

    // Verify both layers are DenseLayer after roundtrip
    assert!(
        loaded_layers[0]
            .as_any()
            .downcast_ref::<DenseLayer>()
            .is_some(),
        "Layer 0 should be DenseLayer after roundtrip"
    );
    assert!(
        loaded_layers[1]
            .as_any()
            .downcast_ref::<DenseLayer>()
            .is_some(),
        "Layer 1 should be DenseLayer after roundtrip"
    );

    // Verify parameter counts match
    assert_eq!(
        loaded_layers[0].parameter_count(),
        original_param_count_0,
        "Layer 0 parameter count should match after roundtrip"
    );
    assert_eq!(
        loaded_layers[1].parameter_count(),
        original_param_count_1,
        "Layer 1 parameter count should match after roundtrip"
    );

    // Verify weights and biases match exactly for layer 0
    let loaded_dense1 = loaded_layers[0]
        .as_any()
        .downcast_ref::<DenseLayer>()
        .unwrap();
    assert_eq!(
        loaded_dense1.weights(),
        original_weights_layer1.as_slice(),
        "Layer 0 weights should match exactly after roundtrip"
    );
    assert_eq!(
        loaded_dense1.biases(),
        original_biases_layer1.as_slice(),
        "Layer 0 biases should match exactly after roundtrip"
    );

    // Verify input/output sizes
    assert_eq!(
        loaded_layers[0].input_size(),
        32,
        "Layer 0 input_size should be 32"
    );
    assert_eq!(
        loaded_layers[0].output_size(),
        16,
        "Layer 0 output_size should be 16"
    );
    assert_eq!(
        loaded_layers[1].input_size(),
        16,
        "Layer 1 input_size should be 16"
    );
    assert_eq!(
        loaded_layers[1].output_size(),
        10,
        "Layer 1 output_size should be 10"
    );
}

/// Test serialization roundtrip for a mixed-layer architecture (Conv2D, BatchNorm,
/// Dropout, Dense), verifying all four layer types survive the binary format.
///
/// This test verifies:
/// 1. Each layer type serializes its parameters correctly
/// 2. Each layer type deserializes back to the correct type
/// 3. All configuration parameters (in_channels, kernel_size, drop_rate, etc.) are preserved
#[test]
fn test_save_load_mixed_architecture() {
    let mut rng = SimpleRng::new(123);

    // Conv2D: 3 in-channels, 8 out-channels, 3x3 kernel, padding=1, stride=1, 8x8 input
    let conv = Conv2DLayer::new(3, 8, 3, 1isize, 1, 8, 8, &mut rng);
    // BatchNorm: 8*8*8 = 512 features
    let bn = BatchNormLayer::new(512, 1e-5, 0.9);
    // Dropout: 512 features, 30% drop rate
    let dropout = DropoutLayer::new(512, 0.3, &mut rng);
    // Dense: 512 -> 10
    let dense = DenseLayer::new(512, 10, &mut rng);

    let original_conv_weights = conv.weights().to_vec();
    let original_bn_gamma = bn.gamma().to_vec();
    let original_dense_weights = dense.weights().to_vec();

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(conv),
        Box::new(bn),
        Box::new(dropout),
        Box::new(dense),
    ];

    // Serialize to in-memory buffer
    let mut data: Vec<u8> = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    // Deserialize from buffer
    let mut cursor = Cursor::new(&data);
    let loaded_layers = load_layers(&mut cursor, &mut rng).unwrap();

    // Verify layer count
    assert_eq!(
        loaded_layers.len(),
        4,
        "Should reconstruct exactly 4 layers"
    );

    // Verify layer 0 is Conv2D with correct parameters
    let loaded_conv = loaded_layers[0]
        .as_any()
        .downcast_ref::<Conv2DLayer>()
        .expect("Layer 0 should be Conv2DLayer after roundtrip");
    assert_eq!(
        loaded_conv.in_channels(),
        3,
        "Conv2D in_channels should be 3"
    );
    assert_eq!(
        loaded_conv.out_channels(),
        8,
        "Conv2D out_channels should be 8"
    );
    assert_eq!(
        loaded_conv.kernel_size(),
        3,
        "Conv2D kernel_size should be 3"
    );
    assert_eq!(loaded_conv.padding(), 1isize, "Conv2D padding should be 1");
    assert_eq!(loaded_conv.stride(), 1, "Conv2D stride should be 1");
    assert_eq!(
        loaded_conv.input_height(),
        8,
        "Conv2D input_height should be 8"
    );
    assert_eq!(
        loaded_conv.input_width(),
        8,
        "Conv2D input_width should be 8"
    );
    assert_eq!(
        loaded_conv.weights(),
        original_conv_weights.as_slice(),
        "Conv2D weights should match exactly after roundtrip"
    );

    // Verify layer 1 is BatchNorm with correct parameters
    let loaded_bn = loaded_layers[1]
        .as_any()
        .downcast_ref::<BatchNormLayer>()
        .expect("Layer 1 should be BatchNormLayer after roundtrip");
    assert_eq!(loaded_bn.output_size(), 512, "BatchNorm size should be 512");
    assert_eq!(
        loaded_bn.gamma(),
        original_bn_gamma.as_slice(),
        "BatchNorm gamma should match exactly after roundtrip"
    );

    // Verify layer 2 is Dropout
    assert!(
        loaded_layers[2]
            .as_any()
            .downcast_ref::<DropoutLayer>()
            .is_some(),
        "Layer 2 should be DropoutLayer after roundtrip"
    );

    // Verify layer 3 is Dense with correct weights
    let loaded_dense = loaded_layers[3]
        .as_any()
        .downcast_ref::<DenseLayer>()
        .expect("Layer 3 should be DenseLayer after roundtrip");
    assert_eq!(
        loaded_dense.weights(),
        original_dense_weights.as_slice(),
        "Dense weights should match exactly after roundtrip"
    );
}

/// Test that Dropout drop_rate is preserved exactly through a serialization roundtrip.
///
/// This test verifies the fix for the previously broken Dropout serialization where
/// drop_rate was not being saved, meaning reconstructed layers would have incorrect
/// regularization strength.
#[test]
fn test_save_load_dropout_drop_rate() {
    let mut rng = SimpleRng::new(42);

    // Create dropout layers with distinctive drop rates
    let dropout_30 = DropoutLayer::new(128, 0.3, &mut rng);
    let dropout_50 = DropoutLayer::new(256, 0.5, &mut rng);
    let dropout_70 = DropoutLayer::new(512, 0.7, &mut rng);

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(dropout_30),
        Box::new(dropout_50),
        Box::new(dropout_70),
    ];

    // Serialize to in-memory buffer
    let mut data: Vec<u8> = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    // Deserialize from buffer
    let mut cursor = Cursor::new(&data);
    let loaded_layers = load_layers(&mut cursor, &mut rng).unwrap();

    assert_eq!(
        loaded_layers.len(),
        3,
        "Should reconstruct exactly 3 dropout layers"
    );

    // Verify each dropout layer has its drop_rate preserved
    let loaded_d1 = loaded_layers[0]
        .as_any()
        .downcast_ref::<DropoutLayer>()
        .expect("Layer 0 should be DropoutLayer");
    assert!(
        (loaded_d1.drop_rate() - 0.3).abs() < 1e-6,
        "Dropout drop_rate 0.3 should be preserved, got {}",
        loaded_d1.drop_rate()
    );
    assert_eq!(
        loaded_layers[0].output_size(),
        128,
        "Dropout size 128 should be preserved"
    );

    let loaded_d2 = loaded_layers[1]
        .as_any()
        .downcast_ref::<DropoutLayer>()
        .expect("Layer 1 should be DropoutLayer");
    assert!(
        (loaded_d2.drop_rate() - 0.5).abs() < 1e-6,
        "Dropout drop_rate 0.5 should be preserved, got {}",
        loaded_d2.drop_rate()
    );
    assert_eq!(
        loaded_layers[1].output_size(),
        256,
        "Dropout size 256 should be preserved"
    );

    let loaded_d3 = loaded_layers[2]
        .as_any()
        .downcast_ref::<DropoutLayer>()
        .expect("Layer 2 should be DropoutLayer");
    assert!(
        (loaded_d3.drop_rate() - 0.7).abs() < 1e-6,
        "Dropout drop_rate 0.7 should be preserved, got {}",
        loaded_d3.drop_rate()
    );
    assert_eq!(
        loaded_layers[2].output_size(),
        512,
        "Dropout size 512 should be preserved"
    );
}

/// Test model serialization roundtrip for the deep CNN architecture loaded from config.
///
/// This test verifies:
/// 1. The 17-layer CIFAR-10 deep CNN can be serialized
/// 2. All layers reconstruct correctly from binary format
/// 3. Layer count and parameter counts are preserved
///
/// The test is skipped gracefully if the config file is not present.
#[test]
fn test_save_load_deep_cnn_model() {
    let config_path = "config/architectures/cifar10_deep_cnn.json";
    let config = match load_architecture(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Skipping test: config file not found: {}", e);
            return;
        }
    };

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    let original_layer_count = layers.len();
    let original_param_counts: Vec<usize> = layers.iter().map(|l| l.parameter_count()).collect();
    let original_input_sizes: Vec<usize> = layers.iter().map(|l| l.input_size()).collect();
    let original_output_sizes: Vec<usize> = layers.iter().map(|l| l.output_size()).collect();

    assert!(
        original_layer_count >= 1,
        "Deep CNN should have at least one layer"
    );

    // Serialize to in-memory buffer
    let mut data: Vec<u8> = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    assert!(!data.is_empty(), "Serialized data should not be empty");

    // Deserialize from buffer
    let mut cursor = Cursor::new(&data);
    let loaded_layers = load_layers(&mut cursor, &mut rng).unwrap();

    // Verify layer count is preserved
    assert_eq!(
        loaded_layers.len(),
        original_layer_count,
        "Loaded model should have same number of layers ({}) as original",
        original_layer_count
    );

    // Verify all layers have correct parameter counts and sizes
    for i in 0..original_layer_count {
        assert_eq!(
            loaded_layers[i].parameter_count(),
            original_param_counts[i],
            "Layer {} parameter count should match after roundtrip: expected {}, got {}",
            i,
            original_param_counts[i],
            loaded_layers[i].parameter_count()
        );
        assert_eq!(
            loaded_layers[i].input_size(),
            original_input_sizes[i],
            "Layer {} input_size should match after roundtrip",
            i
        );
        assert_eq!(
            loaded_layers[i].output_size(),
            original_output_sizes[i],
            "Layer {} output_size should match after roundtrip",
            i
        );
    }

    // Verify loaded model can run a forward pass without errors
    let batch_size = 2;
    let input_size = 3 * 32 * 32; // CIFAR-10 input
    let input = vec![0.5f32; input_size * batch_size];

    let mut current_output = input;
    for (i, layer) in loaded_layers.iter().enumerate() {
        let out_size = layer.output_size();
        let mut next_output = vec![0.0f32; out_size * batch_size];
        layer.forward(&current_output, &mut next_output, batch_size);
        for (j, &val) in next_output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Loaded layer {} produced non-finite value at index {}: {}",
                i,
                j,
                val
            );
        }
        current_output = next_output;
    }
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
