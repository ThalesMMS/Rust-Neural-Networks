use super::*;

#[test]
fn test_invalid_json() {
    let invalid_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 256
},  // trailing comma
  ]
}"#;

    let temp_file = write_temp_config(invalid_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
}

#[test]
fn test_missing_file() {
    let result = load_architecture("nonexistent_file.json");
    assert!(result.is_err());
}

#[test]
fn test_empty_layers() {
    let config_json = r#"{
  "layers": []
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("at least one layer"));
}

#[test]
fn test_invalid_layer_type() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "invalid_layer",
  "input_size": 784,
  "output_size": 256
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Invalid layer type"));
}

#[test]
fn test_missing_required_field_dense() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("output_size"));
}

#[test]
fn test_missing_required_field_conv2d() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 8,
  "kernel_size": 3
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("input_height") || error_msg.contains("input_width"));
}

#[test]
fn test_missing_required_field_dropout() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dropout",
  "size": 256
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("drop_rate"));
}

#[test]
fn test_layer_connection_mismatch() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 256
},
{
  "layer_type": "dense",
  "input_size": 128,
  "output_size": 10
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Layer connection mismatch"));
}

#[test]
fn test_zero_input_size() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 0,
  "output_size": 256
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("greater than 0"));
}

#[test]
fn test_invalid_dropout_rate() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dropout",
  "size": 256,
  "drop_rate": 1.5
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("drop_rate"));
}

#[test]
fn test_invalid_batchnorm_epsilon() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "batchnorm",
  "size": 256,
  "epsilon": -0.001
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("epsilon"));
}

#[test]
fn test_invalid_batchnorm_momentum() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "batchnorm",
  "size": 256,
  "momentum": 1.5
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("momentum"));
}

#[test]
fn test_missing_required_field_maxpool() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "maxpool",
  "pool_size": 2,
  "pool_input_height": 28,
  "pool_input_width": 28
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("pool_channels"));
}

#[test]
fn test_invalid_pool_size_zero() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "avgpool",
  "pool_size": 0,
  "pool_stride": 2,
  "pool_padding": 0,
  "pool_input_height": 28,
  "pool_input_width": 28,
  "pool_channels": 8
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("pool_size"));
    assert!(error_msg.contains("greater than 0"));
}

#[test]
fn test_invalid_pool_stride_zero() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "maxpool",
  "pool_size": 2,
  "pool_stride": 0,
  "pool_padding": 0,
  "pool_input_height": 28,
  "pool_input_width": 28,
  "pool_channels": 8
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("pool_stride"));
    assert!(error_msg.contains("greater than 0"));
}

#[test]
fn test_invalid_pooling_shape_too_large_kernel() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "avgpool",
  "pool_size": 5,
  "pool_stride": 1,
  "pool_padding": 0,
  "pool_input_height": 3,
  "pool_input_width": 3,
  "pool_channels": 8
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("invalid pooling configuration"));
}

#[test]
fn test_pool_layer_requires_pool_mode() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "pool",
  "pool_size": 2,
  "pool_stride": 2,
  "pool_padding": 0,
  "pool_input_height": 28,
  "pool_input_width": 28,
  "pool_channels": 8
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("requires 'pool_mode'"));
}
