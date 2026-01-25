//! Comprehensive tests for configuration parsing
//!
//! This file tests the config module including:
//! - Loading valid JSON config files
//! - Parsing different scheduler types (StepDecay, ExponentialDecay, CosineAnnealing)
//! - Handling invalid JSON
//! - Handling missing files
//! - Handling missing optional fields with defaults

use rust_neural_networks::config::load_config;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_temp_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    file.write_all(contents.as_bytes())
        .expect("failed to write temp config");
    file
}

// ============================================================================
// Valid Config Loading Tests
// ============================================================================

mod valid_config_tests {
    use super::*;

    #[test]
    fn test_load_step_decay_config() {
        let config =
            load_config("config/mnist_mlp_step.json").expect("Failed to load step decay config");

        assert_eq!(config.scheduler_type, "step_decay");
        assert_eq!(config.step_size, Some(3));
        assert_eq!(config.gamma, Some(0.5));
        assert_eq!(config.decay_rate, None);
        assert_eq!(config.min_lr, None);
        assert_eq!(config.T_max, None);
    }

    #[test]
    fn test_load_exponential_config() {
        let config = load_config("config/mnist_mlp_exponential.json")
            .expect("Failed to load exponential config");

        assert_eq!(config.scheduler_type, "exponential");
        assert_eq!(config.decay_rate, Some(0.95));
        assert_eq!(config.step_size, None);
        assert_eq!(config.gamma, None);
        assert_eq!(config.min_lr, None);
        assert_eq!(config.T_max, None);
    }

    #[test]
    fn test_load_cosine_annealing_config() {
        let config = load_config("config/mnist_mlp_cosine.json")
            .expect("Failed to load cosine annealing config");

        assert_eq!(config.scheduler_type, "cosine_annealing");
        assert_eq!(config.min_lr, Some(0.0001));
        assert_eq!(config.T_max, Some(10));
        assert_eq!(config.step_size, None);
        assert_eq!(config.gamma, None);
        assert_eq!(config.decay_rate, None);
    }

    #[test]
    fn test_config_values_step_decay() {
        let config = load_config("config/mnist_mlp_step.json").unwrap();

        // Verify specific values
        assert_eq!(config.step_size.unwrap(), 3);
        assert!((config.gamma.unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_values_exponential() {
        let config = load_config("config/mnist_mlp_exponential.json").unwrap();

        // Verify specific values
        assert!((config.decay_rate.unwrap() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_config_values_cosine() {
        let config = load_config("config/mnist_mlp_cosine.json").unwrap();

        // Verify specific values
        assert!((config.min_lr.unwrap() - 0.0001).abs() < 1e-6);
        assert_eq!(config.T_max.unwrap(), 10);
    }
}

// ============================================================================
// Temporary Config Creation Tests
// ============================================================================

mod temp_config_tests {
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
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_missing_file() {
        let result = load_config("nonexistent_config.json");
        assert!(result.is_err(), "Should fail on missing file");
    }

    #[test]
    fn test_invalid_json_syntax() {
        let invalid_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
  // Missing closing brace
"#;

        let temp_file = write_temp_config(invalid_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on invalid JSON");
    }

    #[test]
    fn test_malformed_json() {
        let malformed_json = "not valid json at all";

        let temp_file = write_temp_config(malformed_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on malformed JSON");
    }

    #[test]
    fn test_missing_scheduler_type() {
        let config_json = r#"{
  "step_size": 3,
  "gamma": 0.5
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when scheduler_type is missing"
        );
    }

    #[test]
    fn test_wrong_type_scheduler_type() {
        let config_json = r#"{
  "scheduler_type": 123,
  "step_size": 3
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when scheduler_type is not a string"
        );
    }

    #[test]
    fn test_wrong_type_step_size() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": "three",
  "gamma": 0.5
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when step_size is not a number"
        );
    }

    #[test]
    fn test_wrong_type_gamma() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": "zero point five"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail when gamma is not a number");
    }

    #[test]
    fn test_empty_file() {
        let temp_file = write_temp_config("");
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on empty file");
    }

    #[test]
    fn test_empty_json_object() {
        let temp_file = write_temp_config("{}");
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on empty JSON object");
    }
}

// ============================================================================
// TrainingConfig Structure Tests
// ============================================================================

mod structure_tests {
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
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
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
}
