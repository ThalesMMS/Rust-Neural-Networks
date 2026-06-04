use super::*;

#[test]
fn test_valid_dense_to_dense() {
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
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_ok());
}

#[test]
fn test_valid_dense_to_batchnorm() {
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
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_ok());
}

#[test]
fn test_valid_dense_to_dropout() {
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
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_ok());
}

#[test]
fn test_valid_conv2d_to_dense() {
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
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_ok());
}

#[test]
fn test_invalid_dense_to_batchnorm_size_mismatch() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 256
},
{
  "layer_type": "batchnorm",
  "size": 128
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
fn test_invalid_conv2d_to_dense_size_mismatch() {
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
  "input_size": 1000,
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
fn test_invalid_residual_block_to_globalavgpool_shape_mismatch() {
    let config_json = r#"{
  "layers": [
{
  "layer_type": "residual_block",
  "in_channels": 16,
  "out_channels": 32,
  "stride": 2,
  "input_height": 32,
  "input_width": 32
},
{
  "layer_type": "globalavgpool",
  "pool_input_height": 32,
  "pool_input_width": 16,
  "pool_channels": 32
}
  ]
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_architecture(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Layer connection mismatch"));
    assert!(error_msg.contains("output height"));
}
