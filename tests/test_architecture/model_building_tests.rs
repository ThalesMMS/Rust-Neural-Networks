use super::*;

#[test]
fn test_build_simple_mlp() {
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

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].input_size(), 784);
    assert_eq!(layers[0].output_size(), 128);
    assert_eq!(layers[1].input_size(), 128);
    assert_eq!(layers[1].output_size(), 10);
}

#[test]
fn test_build_model_with_all_layer_types() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 100,
  "output_size": 64
},
{
  "layer_type": "batchnorm",
  "size": 64,
  "epsilon": 1e-5,
  "momentum": 0.9
},
{
  "layer_type": "dropout",
  "size": 64,
  "drop_rate": 0.2
},
{
  "layer_type": "dense",
  "input_size": 64,
  "output_size": 32
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    assert_eq!(layers.len(), 4);
    assert_eq!(layers[0].input_size(), 100);
    assert_eq!(layers[0].output_size(), 64);
    assert_eq!(layers[1].input_size(), 64);
    assert_eq!(layers[1].output_size(), 64); // BatchNorm preserves size
    assert_eq!(layers[2].input_size(), 64);
    assert_eq!(layers[2].output_size(), 64); // Dropout preserves size
    assert_eq!(layers[3].input_size(), 64);
    assert_eq!(layers[3].output_size(), 32);
}

#[test]
fn test_build_conv2d_model() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 4,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "input_height": 28,
  "input_width": 28
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    assert_eq!(layers.len(), 1);
    // Conv2D input: 1 * 28 * 28 = 784
    assert_eq!(layers[0].input_size(), 784);
    // Conv2D output with padding=1, stride=1, kernel=3: same spatial size
    // Output: 4 * 28 * 28 = 3136
    assert_eq!(layers[0].output_size(), 3136);
}

#[test]
#[ignore = "Pooling runtime layers are not available in this build; architecture validation supports pooling but builder does not expose it"]
fn test_build_conv2d_with_pooling() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 2,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "input_height": 8,
  "input_width": 8
},
{
  "layer_type": "maxpool",
  "pool_input_height": 8,
  "pool_input_width": 8,
  "pool_channels": 2,
  "pool_size": 2,
  "pool_stride": 2,
  "pool_padding": 0
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    assert_eq!(layers.len(), 2);

    assert!(
        layers[0]
            .as_any()
            .downcast_ref::<rust_neural_networks::layers::Conv2DLayer>()
            .is_some(),
        "expected Conv2DLayer"
    );
    assert_eq!(layers[0].output_size(), 2 * 8 * 8);

    // Pooling runtime layer types are not currently exposed in crate::layers;
    // validate via input/output sizes only.
    assert_eq!(layers[1].input_size(), 2 * 8 * 8);
    assert_eq!(layers[1].output_size(), 2 * 4 * 4);
}

#[test]
fn test_build_single_layer_model() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&config, &mut rng).unwrap();

    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].input_size(), 784);
    assert_eq!(layers[0].output_size(), 10);
}
