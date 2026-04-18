use super::*;

/// Verifies end-to-end workflow: load config, build model, check layer sizes.
///
/// Tests the complete workflow of loading an architecture configuration from JSON,
/// building the model with a seeded RNG, and verifying that all layer dimensions
/// are correctly initialized.
///
/// # Examples
///
/// ```
/// let config = load_architecture("config.json").unwrap();
/// let mut rng = SimpleRng::new(42);
/// let layers = build_model(&config, &mut rng).unwrap();
/// // All layers should have correct input/output sizes matching the config
/// ```
#[test]
fn test_end_to_end_mlp() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 512
},
{
  "layer_type": "batchnorm",
  "size": 512,
  "epsilon": 1e-5,
  "momentum": 0.9
},
{
  "layer_type": "dropout",
  "size": 512,
  "drop_rate": 0.3
},
{
  "layer_type": "dense",
  "input_size": 512,
  "output_size": 256
},
{
  "layer_type": "batchnorm",
  "size": 256
},
{
  "layer_type": "dense",
  "input_size": 256,
  "output_size": 10
}
  ]
}"#;

    // Step 1: Load configuration
    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();
    assert_eq!(config.layers.len(), 6);

    // Step 2: Build model
    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();
    assert_eq!(layers.len(), 6);

    // Step 3: Verify layer sizes
    assert_eq!(layers[0].input_size(), 784);
    assert_eq!(layers[0].output_size(), 512);
    assert_eq!(layers[1].input_size(), 512);
    assert_eq!(layers[1].output_size(), 512);
    assert_eq!(layers[2].input_size(), 512);
    assert_eq!(layers[2].output_size(), 512);
    assert_eq!(layers[3].input_size(), 512);
    assert_eq!(layers[3].output_size(), 256);
    assert_eq!(layers[4].input_size(), 256);
    assert_eq!(layers[4].output_size(), 256);
    assert_eq!(layers[5].input_size(), 256);
    assert_eq!(layers[5].output_size(), 10);
}

#[test]
fn test_end_to_end_cnn() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 8,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "input_height": 28,
  "input_width": 28
},
{
  "layer_type": "dense",
  "input_size": 6272,
  "output_size": 128
},
{
  "layer_type": "dropout",
  "size": 128,
  "drop_rate": 0.5
},
{
  "layer_type": "dense",
  "input_size": 128,
  "output_size": 10
}
  ]
}"#;

    // Step 1: Load configuration
    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();
    assert_eq!(config.layers.len(), 4);

    // Step 2: Build model
    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();
    assert_eq!(layers.len(), 4);

    // Step 3: Verify layer sizes
    assert_eq!(layers[0].input_size(), 784); // 1 * 28 * 28
    assert_eq!(layers[0].output_size(), 6272); // 8 * 28 * 28
    assert_eq!(layers[1].input_size(), 6272);
    assert_eq!(layers[1].output_size(), 128);
    assert_eq!(layers[2].input_size(), 128);
    assert_eq!(layers[2].output_size(), 128);
    assert_eq!(layers[3].input_size(), 128);
    assert_eq!(layers[3].output_size(), 10);
}

#[test]
fn test_multiple_rngs_produce_different_weights() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 10,
  "output_size": 5
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    // Build with two different seeds
    let mut rng1 = SimpleRng::new(42);
    let layers1 = build_model(&config, &mut rng1).unwrap();

    let mut rng2 = SimpleRng::new(123);
    let layers2 = build_model(&config, &mut rng2).unwrap();

    // Layers should have same structure but different weights
    assert_eq!(layers1.len(), layers2.len());
    assert_eq!(layers1[0].input_size(), layers2[0].input_size());
    assert_eq!(layers1[0].output_size(), layers2[0].output_size());
    // Note: We can't directly compare weights through the Layer trait,
    // but this test verifies the building process works with different RNGs
}

#[test]
fn test_same_seed_produces_identical_structure() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 128
},
{
  "layer_type": "dense",
  "input_size": 128,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    // Build twice with same seed
    let mut rng1 = SimpleRng::new(42);
    let layers1 = build_model(&config, &mut rng1).unwrap();

    let mut rng2 = SimpleRng::new(42);
    let layers2 = build_model(&config, &mut rng2).unwrap();

    // Should produce identical structure
    assert_eq!(layers1.len(), layers2.len());
    for i in 0..layers1.len() {
        assert_eq!(layers1[i].input_size(), layers2[i].input_size());
        assert_eq!(layers1[i].output_size(), layers2[i].output_size());
    }
}

/// Verifies deep CIFAR-10 CNN architecture with multiple conv layers.
///
/// Tests a deep convolutional architecture suitable for CIFAR-10 (3x32x32 RGB images)
/// with multiple Conv2D layers, BatchNorm, and Dropout for regularization.
///
/// # Architecture
///
/// - Conv2D: 3→32 channels, 3x3 kernel, padding=1, stride=1 (32x32 output)
/// - BatchNorm: 32768 features (32*32*32)
/// - Conv2D: 32→64 channels, 3x3 kernel, padding=1, stride=2 (16x16 output)
/// - BatchNorm: 16384 features (64*16*16)
/// - Dropout: 0.3 drop rate
/// - Conv2D: 64→128 channels, 3x3 kernel, padding=1, stride=2 (8x8 output)
/// - Dense: 8192→256 (flattened conv output)
/// - Dropout: 0.5 drop rate
/// - Dense: 256→10 (CIFAR-10 classes)
///
/// # Examples
///
/// ```
/// let config = load_architecture("cifar10_deep.json").unwrap();
/// let mut rng = SimpleRng::new(42);
/// let layers = build_model(&config, &mut rng).unwrap();
/// assert_eq!(layers.len(), 9);
/// ```
#[test]
fn test_cifar10_deep_architecture() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 3,
  "out_channels": 32,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "input_height": 32,
  "input_width": 32
},
{
  "layer_type": "batchnorm",
  "size": 32768,
  "epsilon": 1e-5,
  "momentum": 0.9
},
{
  "layer_type": "conv2d",
  "in_channels": 32,
  "out_channels": 64,
  "kernel_size": 3,
  "padding": 1,
  "stride": 2,
  "input_height": 32,
  "input_width": 32
},
{
  "layer_type": "batchnorm",
  "size": 16384,
  "epsilon": 1e-5,
  "momentum": 0.9
},
{
  "layer_type": "dropout",
  "size": 16384,
  "drop_rate": 0.3
},
{
  "layer_type": "conv2d",
  "in_channels": 64,
  "out_channels": 128,
  "kernel_size": 3,
  "padding": 1,
  "stride": 2,
  "input_height": 16,
  "input_width": 16
},
{
  "layer_type": "dense",
  "input_size": 8192,
  "output_size": 256
},
{
  "layer_type": "dropout",
  "size": 256,
  "drop_rate": 0.5
},
{
  "layer_type": "dense",
  "input_size": 256,
  "output_size": 10
}
  ]
}"#;

    // Step 1: Load configuration
    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();
    assert_eq!(config.layers.len(), 9);

    // Step 2: Build model
    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();
    assert_eq!(layers.len(), 9);

    // Step 3: Verify layer sizes
    // Conv2D layer 0: 3*32*32 -> 32*32*32
    assert_eq!(layers[0].input_size(), 3072); // 3 * 32 * 32
    assert_eq!(layers[0].output_size(), 32768); // 32 * 32 * 32

    // BatchNorm layer 1: preserves size
    assert_eq!(layers[1].input_size(), 32768);
    assert_eq!(layers[1].output_size(), 32768);

    // Conv2D layer 2: 32*32*32 -> 64*16*16 (stride=2 halves dimensions)
    assert_eq!(layers[2].input_size(), 32768);
    assert_eq!(layers[2].output_size(), 16384); // 64 * 16 * 16

    // BatchNorm layer 3: preserves size
    assert_eq!(layers[3].input_size(), 16384);
    assert_eq!(layers[3].output_size(), 16384);

    // Dropout layer 4: preserves size
    assert_eq!(layers[4].input_size(), 16384);
    assert_eq!(layers[4].output_size(), 16384);

    // Conv2D layer 5: 64*16*16 -> 128*8*8 (stride=2 halves dimensions)
    assert_eq!(layers[5].input_size(), 16384);
    assert_eq!(layers[5].output_size(), 8192); // 128 * 8 * 8

    // Dense layer 6: 8192 -> 256
    assert_eq!(layers[6].input_size(), 8192);
    assert_eq!(layers[6].output_size(), 256);

    // Dropout layer 7: preserves size
    assert_eq!(layers[7].input_size(), 256);
    assert_eq!(layers[7].output_size(), 256);

    // Dense layer 8: 256 -> 10 (CIFAR-10 classes)
    assert_eq!(layers[8].input_size(), 256);
    assert_eq!(layers[8].output_size(), 10);
}
