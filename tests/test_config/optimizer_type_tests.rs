use super::*;

#[test]
fn test_valid_sgd_optimizer() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "sgd"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, Some("sgd".to_string()));
}

#[test]
fn test_valid_adam_optimizer() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adam"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, Some("adam".to_string()));
}

#[test]
fn test_valid_adamw_optimizer() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "adamw"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, Some("adamw".to_string()));
}

#[test]
fn test_valid_rmsprop_optimizer() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "rmsprop"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, Some("rmsprop".to_string()));
}

#[test]
fn test_invalid_optimizer_type() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "optimizer_type": "invalid_optimizer"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on invalid optimizer type");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("invalid_optimizer") || err_msg.contains("optimizer"),
        "Error should mention invalid optimizer but got: {}",
        err_msg
    );
}

#[test]
fn test_optimizer_type_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, None);
}

#[test]
fn test_load_optimizer_adamw_demo_config() {
    let config = load_config("config/optimizer_adamw_demo.json")
        .expect("Failed to load optimizer_adamw_demo config");

    assert_eq!(config.optimizer_type, Some("adamw".to_string()));
    assert_eq!(config.adam_beta1, Some(0.9));
    assert_eq!(config.adam_beta2, Some(0.999));
    assert_eq!(config.adam_epsilon, Some(1e-8));
    assert_eq!(config.adamw_weight_decay, Some(0.01));
}

#[test]
fn test_load_optimizer_rmsprop_demo_config() {
    let config = load_config("config/optimizer_rmsprop_demo.json")
        .expect("Failed to load optimizer_rmsprop_demo config");

    assert_eq!(config.optimizer_type, Some("rmsprop".to_string()));
}
