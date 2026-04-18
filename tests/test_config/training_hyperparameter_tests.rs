use super::*;

// Learning Rate Tests
#[test]
fn test_valid_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": 0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.learning_rate, Some(0.01));
}

#[test]
fn test_negative_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": -0.01
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative learning_rate");
}

#[test]
fn test_zero_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on zero learning_rate");
}

// Epochs Tests
#[test]
fn test_valid_epochs() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "epochs": 10
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.epochs, Some(10));
}

#[test]
fn test_zero_epochs() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "epochs": 0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on zero epochs");
}

// Batch Size Tests
#[test]
fn test_valid_batch_size() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "batch_size": 64
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.batch_size, Some(64));
}

#[test]
fn test_zero_batch_size() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "batch_size": 0
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on zero batch_size");
}

// Validation Split Tests
#[test]
fn test_valid_validation_split() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "validation_split": 0.2
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.validation_split, Some(0.2));
}

#[test]
fn test_validation_split_zero() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "validation_split": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.validation_split, Some(0.0));
}

#[test]
fn test_validation_split_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "validation_split": 1.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.validation_split, Some(1.0));
}

#[test]
fn test_validation_split_negative() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "validation_split": -0.1
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on negative validation_split");
}

#[test]
fn test_validation_split_greater_than_one() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "validation_split": 1.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail on validation_split greater than 1.0"
    );
}

// Early Stopping Patience Tests
#[test]
fn test_valid_early_stopping_patience() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "early_stopping_patience": 5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.early_stopping_patience, Some(5));
}

#[test]
fn test_zero_early_stopping_patience() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "early_stopping_patience": 0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.early_stopping_patience, Some(0));
}

// Early Stopping Min Delta Tests
#[test]
fn test_valid_early_stopping_min_delta() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "early_stopping_min_delta": 0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.early_stopping_min_delta, Some(0.001));
}

#[test]
fn test_zero_early_stopping_min_delta() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "early_stopping_min_delta": 0.0
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.early_stopping_min_delta, Some(0.0));
}

#[test]
fn test_negative_early_stopping_min_delta() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "early_stopping_min_delta": -0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail on negative early_stopping_min_delta"
    );
}

// Combined Hyperparameters Tests
#[test]
fn test_all_training_hyperparameters() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.learning_rate, Some(0.01));
    assert_eq!(config.epochs, Some(10));
    assert_eq!(config.batch_size, Some(64));
    assert_eq!(config.validation_split, Some(0.1));
    assert_eq!(config.early_stopping_patience, Some(3));
    assert_eq!(config.early_stopping_min_delta, Some(0.001));
}

#[test]
fn test_training_hyperparameters_optional() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

    assert_eq!(config.learning_rate, None);
    assert_eq!(config.epochs, None);
    assert_eq!(config.batch_size, None);
    assert_eq!(config.validation_split, None);
    assert_eq!(config.early_stopping_patience, None);
    assert_eq!(config.early_stopping_min_delta, None);
}
