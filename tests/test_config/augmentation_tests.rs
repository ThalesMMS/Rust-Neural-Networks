use super::*;

#[test]
fn test_enable_augmentation_true() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "enable_augmentation": true
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.enable_augmentation, Some(true));
}

#[test]
fn test_enable_augmentation_false() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "enable_augmentation": false
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.enable_augmentation, Some(false));
}

#[test]
fn test_horizontal_flip_prob_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "horizontal_flip_prob": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.horizontal_flip_prob, Some(0.5));
}

#[test]
fn test_horizontal_flip_prob_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "horizontal_flip_prob": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.horizontal_flip_prob, Some(0.0));
}

#[test]
fn test_horizontal_flip_prob_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "horizontal_flip_prob": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.horizontal_flip_prob, Some(1.0));
}

#[test]
fn test_horizontal_flip_prob_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "horizontal_flip_prob": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail on negative horizontal_flip_prob"
    );
}

#[test]
fn test_horizontal_flip_prob_greater_than_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "horizontal_flip_prob": 1.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when horizontal_flip_prob > 1.0"
    );
}

#[test]
fn test_random_crop_padding_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "random_crop_padding": 4
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.random_crop_padding, Some(4));
}

#[test]
fn test_brightness_jitter_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "brightness_jitter": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.brightness_jitter, Some(0.2));
}

#[test]
fn test_brightness_jitter_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "brightness_jitter": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.brightness_jitter, Some(0.0));
}

#[test]
fn test_brightness_jitter_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "brightness_jitter": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative brightness_jitter");
}

#[test]
fn test_contrast_jitter_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "contrast_jitter": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.contrast_jitter, Some(0.2));
}

#[test]
fn test_contrast_jitter_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "contrast_jitter": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative contrast_jitter");
}

#[test]
fn test_saturation_jitter_valid() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "saturation_jitter": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.saturation_jitter, Some(0.2));
}

#[test]
fn test_saturation_jitter_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "saturation_jitter": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative saturation_jitter");
}

#[test]
fn test_all_augmentation_parameters() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 4,
  "brightness_jitter": 0.2,
  "contrast_jitter": 0.2,
  "saturation_jitter": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.enable_augmentation, Some(true));
    assert_eq!(config.horizontal_flip_prob, Some(0.5));
    assert_eq!(config.random_crop_padding, Some(4));
    assert_eq!(config.brightness_jitter, Some(0.2));
    assert_eq!(config.contrast_jitter, Some(0.2));
    assert_eq!(config.saturation_jitter, Some(0.2));
}
