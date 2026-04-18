use super::*;

#[test]
fn test_single_dense_layer() {
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

    assert_eq!(config.layers.len(), 1);
}

#[test]
fn test_single_conv2d_layer() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 3,
  "out_channels": 16,
  "kernel_size": 5,
  "padding": 2,
  "stride": 1,
  "input_height": 32,
  "input_width": 32
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 1);
}

#[test]
fn test_very_deep_network() {
    let mut layers = Vec::new();

    // First layer
    layers.push(
        r#"    {
  "layer_type": "dense",
  "input_size": 100,
  "output_size": 64
}"#,
    );

    // 8 middle layers (64 -> 64)
    let middle_layer = r#"    {
  "layer_type": "dense",
  "input_size": 64,
  "output_size": 64
}"#;
    layers.extend(std::iter::repeat_n(middle_layer, 8));

    // Final layer
    layers.push(
        r#"    {
  "layer_type": "dense",
  "input_size": 64,
  "output_size": 10
}"#,
    );

    let config_json = format!(
        r#"{{
  "layers": [
{}
  ]
}}"#,
        layers.join(",\n")
    );

    let temp_file = write_temp_config(&config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 10);
}

#[test]
fn test_batchnorm_with_defaults() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
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

    // Should use default epsilon and momentum
    assert_eq!(config.layers[1].epsilon, None); // Will use 1e-5 default in build
    assert_eq!(config.layers[1].momentum, None); // Will use 0.9 default in build
}

#[test]
fn test_conv2d_with_defaults() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 8,
  "kernel_size": 3,
  "input_height": 28,
  "input_width": 28
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    // Should use default padding (0) and stride (1)
    assert_eq!(config.layers[0].padding, None); // Will use 0 default in build
    assert_eq!(config.layers[0].stride, None); // Will use 1 default in build
}

#[test]
fn test_case_insensitive_layer_types() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "Dense",
  "input_size": 784,
  "output_size": 256
},
{
  "layer_type": "BATCHNORM",
  "size": 256
},
{
  "layer_type": "Dropout",
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

    let temp_file = write_temp_config(config_json);
    let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.layers.len(), 4);
    // Layer types are stored as provided, but validation handles case-insensitively
}
