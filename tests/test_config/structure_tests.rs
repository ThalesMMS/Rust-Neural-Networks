use super::*;

#[test]
fn test_config_clone() {
    let config = load_config("config/mnist_mlp_step.json").unwrap();
    let cloned = config.clone();

    assert_eq!(config.scheduler_type, cloned.scheduler_type);
    assert_eq!(config.step_size, cloned.step_size);
    assert_eq!(config.gamma, cloned.gamma);
}

#[test]
fn test_config_debug() {
    let config = load_config("config/mnist_mlp_step.json").unwrap();
    let debug_str = format!("{:?}", config);

    // Debug string should contain the struct name and fields
    assert!(debug_str.contains("TrainingConfig"));
    assert!(debug_str.contains("step_decay"));
}

/// Verifies that optional scheduler configuration fields are absent when not provided.
///
/// Loads a minimal config containing only `scheduler_type` and asserts `step_size`, `gamma`, `decay_rate`, `min_lr`, and `T_max` are `None`.
///
/// # Examples
///
/// ```no_run
/// use std::fs;
/// use rust_neural_networks::config::load_config;
///
/// fs::write("tmp_minimal.json", r#"{"scheduler_type":"custom"}"#).unwrap();
/// let cfg = load_config("tmp_minimal.json").unwrap();
/// fs::remove_file("tmp_minimal.json").unwrap();
/// assert!(cfg.step_size.is_none());
/// assert!(cfg.gamma.is_none());
/// assert!(cfg.decay_rate.is_none());
/// assert!(cfg.min_lr.is_none());
/// assert!(cfg.T_max.is_none());
/// ```
#[test]
fn test_optional_fields_are_optional() {
    // Create a config with only required field
    let config_json = r#"{
  "scheduler_type": "custom"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    // All optional fields should be None
    assert!(config.step_size.is_none());
    assert!(config.gamma.is_none());
    assert!(config.decay_rate.is_none());
    assert!(config.min_lr.is_none());
    assert!(config.T_max.is_none());
}
