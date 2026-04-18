use super::*;

#[test]
fn test_valid_adamw_weight_decay() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adamw",
  "adamw_weight_decay": 0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adamw_weight_decay, Some(0.01));
}

#[test]
fn test_adamw_weight_decay_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adamw_weight_decay": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adamw_weight_decay, Some(0.0));
}

#[test]
fn test_adamw_weight_decay_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adamw_weight_decay": -0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail on negative adamw_weight_decay"
    );
}

#[test]
fn test_adamw_weight_decay_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adamw"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.adamw_weight_decay, None);
}

#[test]
fn test_adamw_with_all_adam_params() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adamw",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "adamw_weight_decay": 0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, Some("adamw".to_string()));
    assert_eq!(config.adam_beta1, Some(0.9));
    assert_eq!(config.adam_beta2, Some(0.999));
    assert_eq!(config.adam_epsilon, Some(1e-8));
    assert_eq!(config.adamw_weight_decay, Some(0.01));
}
