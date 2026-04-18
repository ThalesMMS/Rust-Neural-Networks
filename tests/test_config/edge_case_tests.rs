use super::*;

#[test]
fn test_zero_values() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 0,
  "gamma": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.step_size, Some(0));
    assert_eq!(config.gamma, Some(0.0));
}

#[test]
fn test_large_values() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 1000000,
  "gamma": 0.999999
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.step_size, Some(1000000));
    assert!((config.gamma.unwrap() - 0.999999).abs() < 1e-6);
}

#[test]
fn test_negative_float_values() {
    let config_json = r#"{
  "scheduler_type": "test",
  "gamma": -0.5,
  "min_lr": -0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative numeric values");
}

#[test]
fn test_negative_decay_rate() {
    let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": -0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative decay_rate");
}

#[test]
fn test_extra_whitespace() {
    let config_json = r#"

    {
        "scheduler_type"   :   "step_decay"   ,
        "step_size"        :   3               ,
        "gamma"            :   0.5
    }

    "#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "step_decay");
    assert_eq!(config.step_size, Some(3));
    assert_eq!(config.gamma, Some(0.5));
}

#[test]
fn test_unicode_in_strings() {
    let config_json = r#"{
  "scheduler_type": "step_decay_🚀",
  "step_size": 3,
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "step_decay_🚀");
}

#[test]
fn test_scientific_notation() {
    let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": 1e-3,
  "min_lr": 1.5e-4
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert!((config.decay_rate.unwrap() - 0.001).abs() < 1e-6);
    assert!((config.min_lr.unwrap() - 0.00015).abs() < 1e-6);
}
