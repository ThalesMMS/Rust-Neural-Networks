use super::*;

#[test]
fn test_valid_rmsprop_decay() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "rmsprop",
  "rmsprop_decay": 0.9
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.rmsprop_decay, Some(0.9));
}

#[test]
fn test_rmsprop_decay_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "rmsprop_decay": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.rmsprop_decay, Some(0.0));
}

#[test]
fn test_rmsprop_decay_at_one_invalid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "rmsprop_decay": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when rmsprop_decay is 1.0");
}

#[test]
fn test_rmsprop_decay_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "rmsprop_decay": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative rmsprop_decay");
}

#[test]
fn test_rmsprop_epsilon_positive() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "rmsprop_epsilon": 1e-8
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.rmsprop_epsilon, Some(1e-8));
}

#[test]
fn test_rmsprop_epsilon_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "rmsprop_epsilon": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when rmsprop_epsilon is zero");
}

#[test]
fn test_rmsprop_hyperparameters_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "rmsprop"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.rmsprop_decay, None);
    assert_eq!(config.rmsprop_epsilon, None);
}
