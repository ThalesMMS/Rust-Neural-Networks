use super::*;

#[test]
fn test_parse_minimal_step_decay() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 5,
  "gamma": 0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "step_decay");
    assert_eq!(config.step_size, Some(5));
    assert_eq!(config.gamma, Some(0.1));
}

#[test]
fn test_parse_minimal_exponential() {
    let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": 0.9
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "exponential");
    assert_eq!(config.decay_rate, Some(0.9));
}

#[test]
fn test_parse_minimal_cosine() {
    let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.00001,
  "T_max": 20
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.min_lr, Some(0.00001));
    assert_eq!(config.T_max, Some(20));
}

/// Verifies that a JSON config containing all scheduler fields parses with every field populated.
///
/// Writes a temporary config including `scheduler_type`, `step_size`, `gamma`, `decay_rate`, `min_lr`, and `T_max`, loads it via `load_config`, and asserts the parsed `TrainingConfig` contains the expected values for each field.
///
/// # Examples
///
/// ```
/// // Equivalent to the test: create a temp file with all fields, call `load_config`, then assert each field.
/// ```
#[test]
fn test_parse_all_fields() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.7,
  "decay_rate": 0.99,
  "min_lr": 0.0005,
  "T_max": 15
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    // All fields should be present
    assert_eq!(config.scheduler_type, "step_decay");
    assert_eq!(config.step_size, Some(2));
    assert_eq!(config.gamma, Some(0.7));
    assert_eq!(config.decay_rate, Some(0.99));
    assert_eq!(config.min_lr, Some(0.0005));
    assert_eq!(config.T_max, Some(15));
}

#[test]
fn test_parse_only_scheduler_type() {
    let config_json = r#"{
  "scheduler_type": "constant"
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    // Only scheduler_type required, all optional fields should be None
    assert_eq!(config.scheduler_type, "constant");
    assert_eq!(config.step_size, None);
    assert_eq!(config.gamma, None);
    assert_eq!(config.decay_rate, None);
    assert_eq!(config.min_lr, None);
    assert_eq!(config.T_max, None);
}
