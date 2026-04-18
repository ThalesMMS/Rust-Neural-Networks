use super::*;

#[test]
fn test_valid_relu_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("relu".to_string()));
}

#[test]
fn test_valid_leaky_relu_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("leaky_relu".to_string()));
    assert_eq!(config.leaky_relu_alpha, Some(0.2));
}

#[test]
fn test_valid_elu_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": 1.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("elu".to_string()));
    assert_eq!(config.elu_alpha, Some(1.5));
}

#[test]
fn test_valid_gelu_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "gelu"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("gelu".to_string()));
}

#[test]
fn test_valid_swish_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "swish"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("swish".to_string()));
}

#[test]
fn test_valid_tanh_activation() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "tanh"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, Some("tanh".to_string()));
}

#[test]
fn test_invalid_activation_function() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "invalid_activation"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail on invalid activation function name"
    );
}

#[test]
fn test_negative_leaky_relu_alpha() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative leaky_relu_alpha");
}

#[test]
fn test_zero_elu_alpha() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on zero elu_alpha");
}

#[test]
fn test_negative_elu_alpha() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": -1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative elu_alpha");
}

#[test]
fn test_activation_function_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.activation_function, None);
    assert_eq!(config.leaky_relu_alpha, None);
    assert_eq!(config.elu_alpha, None);
}
