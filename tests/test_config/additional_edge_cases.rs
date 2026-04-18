use super::*;

#[test]
fn test_very_small_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": 1e-10
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.learning_rate, Some(1e-10));
}

#[test]
fn test_very_large_batch_size() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "batch_size": 1000000
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.batch_size, Some(1000000));
}

#[test]
fn test_all_optional_fields_present() {
    let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001,
  "T_max": 10,
  "activation_function": "gelu",
  "optimizer_type": "adamw",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "adamw_weight_decay": 0.01,
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 128,
  "validation_split": 0.2,
  "early_stopping_patience": 10,
  "early_stopping_min_delta": 0.0001,
  "enable_profiling": true,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 4,
  "brightness_jitter": 0.2,
  "contrast_jitter": 0.2,
  "saturation_jitter": 0.2,
  "noise_dim": 100,
  "g_lr": 0.0002,
  "d_lr": 0.0002,
  "label_smoothing": 0.9,
  "step_debug": false
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.optimizer_type, Some("adamw".to_string()));
    assert_eq!(config.activation_function, Some("gelu".to_string()));
    assert_eq!(config.enable_profiling, Some(true));
    assert_eq!(config.enable_augmentation, Some(true));
    assert_eq!(config.step_debug, Some(false));
}

#[test]
fn test_multiple_validation_failures() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": -0.01,
  "batch_size": 0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when multiple validation errors exist"
    );
}

#[test]
fn test_optimizer_without_type_has_params() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.optimizer_type, None);
    assert_eq!(config.adam_beta1, Some(0.9));
    assert_eq!(config.adam_beta2, Some(0.999));
    assert_eq!(config.adam_epsilon, Some(1e-8));
}

#[test]
fn test_config_with_only_required_field() {
    let config_json = r#"{
  "scheduler_type": "none"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "none");
    assert_eq!(config.step_size, None);
    assert_eq!(config.gamma, None);
    assert_eq!(config.optimizer_type, None);
    assert_eq!(config.activation_function, None);
}

#[test]
fn test_boundary_value_epochs_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "epochs": 1
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.epochs, Some(1));
}

#[test]
fn test_boundary_value_batch_size_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "batch_size": 1
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.batch_size, Some(1));
}
