use super::*;

#[test]
fn test_valid_adam_hyperparameters() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adam",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adam_beta1, Some(0.9));
    assert_eq!(config.adam_beta2, Some(0.999));
    assert_eq!(config.adam_epsilon, Some(1e-8));
}

#[test]
fn test_adam_beta1_boundary_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta1": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adam_beta1, Some(0.0));
}

#[test]
fn test_adam_beta1_at_one_invalid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta1": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when adam_beta1 is exactly 1.0"
    );
}

#[test]
fn test_adam_beta1_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta1": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative adam_beta1");
}

#[test]
fn test_adam_beta1_greater_than_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta1": 1.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when adam_beta1 > 1.0");
}

#[test]
fn test_adam_beta2_boundary_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta2": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adam_beta2, Some(0.0));
}

#[test]
fn test_adam_beta2_at_one_invalid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta2": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when adam_beta2 is exactly 1.0"
    );
}

#[test]
fn test_adam_beta2_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta2": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative adam_beta2");
}

#[test]
fn test_adam_epsilon_positive() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_epsilon": 1e-10
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adam_epsilon, Some(1e-10));
}

#[test]
fn test_adam_epsilon_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_epsilon": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when adam_epsilon is zero");
}

#[test]
fn test_adam_epsilon_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_epsilon": -1e-8
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative adam_epsilon");
}

#[test]
fn test_adam_hyperparameters_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adam"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adam_beta1, None);
    assert_eq!(config.adam_beta2, None);
    assert_eq!(config.adam_epsilon, None);
}
