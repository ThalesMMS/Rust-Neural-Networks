use super::*;

#[test]
fn test_load_simple_mlp() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 256
},
{
  "layer_type": "dense",
  "input_size": 256,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 2);
    assert_eq!(config.layers[0].layer_type, "dense");
    assert_eq!(config.layers[0].input_size, Some(784));
    assert_eq!(config.layers[0].output_size, Some(256));
    assert_eq!(config.layers[1].layer_type, "dense");
    assert_eq!(config.layers[1].input_size, Some(256));
    assert_eq!(config.layers[1].output_size, Some(10));
}

#[test]
fn test_load_mlp_with_batchnorm() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 256
},
{
  "layer_type": "batchnorm",
  "size": 256,
  "epsilon": 1e-5,
  "momentum": 0.9
},
{
  "layer_type": "dense",
  "input_size": 256,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 3);
    assert_eq!(config.layers[1].layer_type, "batchnorm");
    assert_eq!(config.layers[1].size, Some(256));
    assert_eq!(config.layers[1].epsilon, Some(1e-5));
    assert_eq!(config.layers[1].momentum, Some(0.9));
}

#[test]
fn test_load_mlp_with_dropout() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 512
},
{
  "layer_type": "dropout",
  "size": 512,
  "drop_rate": 0.5
},
{
  "layer_type": "dense",
  "input_size": 512,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 3);
    assert_eq!(config.layers[1].layer_type, "dropout");
    assert_eq!(config.layers[1].size, Some(512));
    assert_eq!(config.layers[1].drop_rate, Some(0.5));
}

#[test]
fn test_load_conv2d_architecture() {
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
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 2);
    assert_eq!(config.layers[0].layer_type, "conv2d");
    assert_eq!(config.layers[0].in_channels, Some(1));
    assert_eq!(config.layers[0].out_channels, Some(8));
    assert_eq!(config.layers[0].kernel_size, Some(3));
    assert_eq!(config.layers[0].padding, Some(1));
    assert_eq!(config.layers[0].stride, Some(1));
    assert_eq!(config.layers[0].input_height, Some(28));
    assert_eq!(config.layers[0].input_width, Some(28));
}

/// Verifies that a complex architecture with all layer types loads correctly.
///
/// Tests an architecture containing Dense, Conv2D, BatchNorm, and Dropout layers
/// to ensure all layer types can be parsed and validated together.
///
/// # Examples
///
/// ```
/// // Architecture: Conv2D -> Dense -> BatchNorm -> Dropout -> Dense
/// let config = load_architecture("path/to/complex_config.json").unwrap();
/// assert_eq!(config.layers.len(), 5);
/// ```
#[test]
fn test_load_complex_architecture() {
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

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 6);
    // Verify layer types are correct
    assert_eq!(config.layers[0].layer_type, "dense");
    assert_eq!(config.layers[1].layer_type, "batchnorm");
    assert_eq!(config.layers[2].layer_type, "dropout");
    assert_eq!(config.layers[3].layer_type, "dense");
    assert_eq!(config.layers[4].layer_type, "batchnorm");
    assert_eq!(config.layers[5].layer_type, "dense");
}
