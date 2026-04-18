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
