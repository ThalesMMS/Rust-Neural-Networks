use super::*;

#[test]
fn test_noise_dim_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "noise_dim": 100
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.noise_dim, Some(100));
}

#[test]
fn test_noise_dim_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "noise_dim": 0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when noise_dim is zero");
}

#[test]
fn test_g_lr_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "g_lr": 0.0002
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.g_lr, Some(0.0002));
}

#[test]
fn test_g_lr_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "g_lr": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when g_lr is zero");
}

#[test]
fn test_g_lr_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "g_lr": -0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative g_lr");
}

#[test]
fn test_d_lr_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "d_lr": 0.0002
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.d_lr, Some(0.0002));
}

#[test]
fn test_d_lr_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "d_lr": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when d_lr is zero");
}

#[test]
fn test_d_lr_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "d_lr": -0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative d_lr");
}

#[test]
fn test_label_smoothing_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "label_smoothing": 0.9
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.label_smoothing, Some(0.9));
}

#[test]
fn test_label_smoothing_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "label_smoothing": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.label_smoothing, Some(0.0));
}

#[test]
fn test_label_smoothing_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "label_smoothing": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.label_smoothing, Some(1.0));
}

#[test]
fn test_label_smoothing_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "label_smoothing": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative label_smoothing");
}

#[test]
fn test_label_smoothing_greater_than_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "label_smoothing": 1.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when label_smoothing > 1.0");
}

#[test]
fn test_all_gan_parameters() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "noise_dim": 100,
  "g_lr": 0.0002,
  "d_lr": 0.0002,
  "label_smoothing": 0.9
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.noise_dim, Some(100));
    assert_eq!(config.g_lr, Some(0.0002));
    assert_eq!(config.d_lr, Some(0.0002));
    assert_eq!(config.label_smoothing, Some(0.9));
}
