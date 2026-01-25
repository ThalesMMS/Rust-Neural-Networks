//! Comprehensive tests for configuration parsing
//!
//! This file tests the config module including:
//! - Loading valid JSON config files
//! - Parsing different scheduler types (StepDecay, ExponentialDecay, CosineAnnealing)
//! - Handling invalid JSON
//! - Handling missing files
//! - Handling missing optional fields with defaults

use rust_neural_networks::config::load_config;
use std::fs;

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
        let temp_file = "test_temp_step.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 5,
  "gamma": 0.1
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.scheduler_type, "step_decay");
        assert_eq!(config.step_size, Some(5));
        assert_eq!(config.gamma, Some(0.1));
    }

    #[test]
    fn test_parse_minimal_exponential() {
        let temp_file = "test_temp_exponential.json";
        let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": 0.9
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.scheduler_type, "exponential");
        assert_eq!(config.decay_rate, Some(0.9));
    }

    #[test]
    fn test_parse_minimal_cosine() {
        let temp_file = "test_temp_cosine.json";
        let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.00001,
  "T_max": 20
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.scheduler_type, "cosine_annealing");
        assert_eq!(config.min_lr, Some(0.00001));
        assert_eq!(config.T_max, Some(20));
    }

    #[test]
    fn test_parse_all_fields() {
        let temp_file = "test_temp_all_fields.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.7,
  "decay_rate": 0.99,
  "min_lr": 0.0005,
  "T_max": 15
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

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
        let temp_file = "test_temp_minimal.json";
        let config_json = r#"{
  "scheduler_type": "constant"
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

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
        let temp_file = "test_temp_invalid.json";
        let invalid_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
  // Missing closing brace
"#;

        fs::write(temp_file, invalid_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(result.is_err(), "Should fail on invalid JSON");
    }

    #[test]
    fn test_malformed_json() {
        let temp_file = "test_temp_malformed.json";
        let malformed_json = "not valid json at all";

        fs::write(temp_file, malformed_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(result.is_err(), "Should fail on malformed JSON");
    }

    #[test]
    fn test_missing_scheduler_type() {
        let temp_file = "test_temp_missing_type.json";
        let config_json = r#"{
  "step_size": 3,
  "gamma": 0.5
}"#;

        fs::write(temp_file, config_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(
            result.is_err(),
            "Should fail when scheduler_type is missing"
        );
    }

    #[test]
    fn test_wrong_type_scheduler_type() {
        let temp_file = "test_temp_wrong_type.json";
        let config_json = r#"{
  "scheduler_type": 123,
  "step_size": 3
}"#;

        fs::write(temp_file, config_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(
            result.is_err(),
            "Should fail when scheduler_type is not a string"
        );
    }

    #[test]
    fn test_wrong_type_step_size() {
        let temp_file = "test_temp_wrong_step_size.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": "three",
  "gamma": 0.5
}"#;

        fs::write(temp_file, config_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(
            result.is_err(),
            "Should fail when step_size is not a number"
        );
    }

    #[test]
    fn test_wrong_type_gamma() {
        let temp_file = "test_temp_wrong_gamma.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": "zero point five"
}"#;

        fs::write(temp_file, config_json).unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(result.is_err(), "Should fail when gamma is not a number");
    }

    #[test]
    fn test_empty_file() {
        let temp_file = "test_temp_empty.json";
        fs::write(temp_file, "").unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

        assert!(result.is_err(), "Should fail on empty file");
    }

    #[test]
    fn test_empty_json_object() {
        let temp_file = "test_temp_empty_obj.json";
        fs::write(temp_file, "{}").unwrap();
        let result = load_config(temp_file);
        fs::remove_file(temp_file).unwrap();

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
        let temp_file = "test_temp_optional.json";

        // Create a config with only required field
        let config_json = r#"{
  "scheduler_type": "custom"
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

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
        let temp_file = "test_temp_zeros.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 0,
  "gamma": 0.0
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.step_size, Some(0));
        assert_eq!(config.gamma, Some(0.0));
    }

    #[test]
    fn test_large_values() {
        let temp_file = "test_temp_large.json";
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 1000000,
  "gamma": 0.999999
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.step_size, Some(1000000));
        assert!((config.gamma.unwrap() - 0.999999).abs() < 1e-6);
    }

    #[test]
    fn test_negative_float_values() {
        let temp_file = "test_temp_negative.json";
        let config_json = r#"{
  "scheduler_type": "test",
  "gamma": -0.5,
  "min_lr": -0.001
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        // JSON allows negative numbers, even if they might not make sense for the application
        assert_eq!(config.gamma, Some(-0.5));
        assert_eq!(config.min_lr, Some(-0.001));
    }

    #[test]
    fn test_extra_whitespace() {
        let temp_file = "test_temp_whitespace.json";
        let config_json = r#"

        {
            "scheduler_type"   :   "step_decay"   ,
            "step_size"        :   3               ,
            "gamma"            :   0.5
        }

        "#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.scheduler_type, "step_decay");
        assert_eq!(config.step_size, Some(3));
        assert_eq!(config.gamma, Some(0.5));
    }

    #[test]
    fn test_unicode_in_strings() {
        let temp_file = "test_temp_unicode.json";
        let config_json = r#"{
  "scheduler_type": "step_decay_🚀",
  "step_size": 3,
  "gamma": 0.5
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert_eq!(config.scheduler_type, "step_decay_🚀");
    }

    #[test]
    fn test_scientific_notation() {
        let temp_file = "test_temp_scientific.json";
        let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": 1e-3,
  "min_lr": 1.5e-4
}"#;

        fs::write(temp_file, config_json).unwrap();
        let config = load_config(temp_file).unwrap();
        fs::remove_file(temp_file).unwrap();

        assert!((config.decay_rate.unwrap() - 0.001).abs() < 1e-6);
        assert!((config.min_lr.unwrap() - 0.00015).abs() < 1e-6);
    }
}
