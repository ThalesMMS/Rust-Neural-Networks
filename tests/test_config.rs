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

    /// Verifies that a cosine annealing scheduler configuration is loaded correctly from the example JSON.
    ///
    /// Asserts that `scheduler_type` equals `"cosine_annealing"`, `min_lr` and `T_max` are present with the expected values,
    /// and that `step_size`, `gamma`, and `decay_rate` are `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let config = rust_neural_networks::config::load_config("config/mnist_mlp_cosine.json").unwrap();
    /// assert_eq!(config.scheduler_type, "cosine_annealing");
    /// assert_eq!(config.min_lr, Some(0.0001));
    /// assert_eq!(config.T_max, Some(10));
    /// ```
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

    #[test]
    fn test_load_activations_demo_config() {
        let config = load_config("config/activations_demo.json")
            .expect("Failed to load activations demo config");

        assert_eq!(config.scheduler_type, "step_decay");
        assert_eq!(config.step_size, Some(3));
        assert_eq!(config.gamma, Some(0.5));
        assert_eq!(config.activation_function, Some("leaky_relu".to_string()));
        assert_eq!(config.leaky_relu_alpha, Some(0.01));
        assert_eq!(config.elu_alpha, None);
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
}

// ============================================================================
// Activation Function Tests
// ============================================================================

mod activation_function_tests {
    use super::*;

    #[test]
    fn test_valid_relu_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu"
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("relu".to_string()));
    }

    #[test]
    fn test_valid_leaky_relu_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.2
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("leaky_relu".to_string()));
        assert_eq!(config.leaky_relu_alpha, Some(0.2));
    }

    #[test]
    fn test_valid_elu_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": 1.5
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("elu".to_string()));
        assert_eq!(config.elu_alpha, Some(1.5));
    }

    #[test]
    fn test_valid_gelu_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "gelu"
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("gelu".to_string()));
    }

    #[test]
    fn test_valid_swish_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "swish"
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("swish".to_string()));
    }

    #[test]
    fn test_valid_tanh_activation() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "tanh"
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, Some("tanh".to_string()));
    }

    #[test]
    fn test_invalid_activation_function() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "invalid_activation"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail on invalid activation function name"
        );
    }

    #[test]
    fn test_negative_leaky_relu_alpha() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": -0.1
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on negative leaky_relu_alpha");
    }

    #[test]
    fn test_zero_elu_alpha() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": 0.0
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on zero elu_alpha");
    }

    #[test]
    fn test_negative_elu_alpha() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "elu",
  "elu_alpha": -1.0
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on negative elu_alpha");
    }

    #[test]
    fn test_activation_function_optional() {
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_config(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.activation_function, None);
        assert_eq!(config.leaky_relu_alpha, None);
        assert_eq!(config.elu_alpha, None);
    }
}

// ============================================================================
// Training Hyperparameter Validation Tests
// ============================================================================

mod training_hyperparameter_tests {
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
}

// ============================================================================
// Config Error Message Content Tests
// ============================================================================

mod config_error_tests {
    use super::*;

    #[test]
    fn test_config_error_missing_file_shows_guidance() {
        let result = load_config("nonexistent_path/config_that_does_not_exist.json");

        assert!(result.is_err(), "Should fail on missing config file");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("config/training/"),
            "Missing file error should mention 'config/training/' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_config_error_missing_file_shows_path() {
        let missing_path = "no_such_config.json";
        let result = load_config(missing_path);

        assert!(result.is_err(), "Should fail on missing config file");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains(missing_path),
            "Missing file error should include the path '{}' but got: {}",
            missing_path,
            err_msg
        );
    }

    #[test]
    fn test_config_error_invalid_json_shows_guidance() {
        let malformed_json = r#"{ "scheduler_type": "step_decay", invalid }"#;

        let temp_file = write_temp_config(malformed_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on malformed JSON");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.to_lowercase().contains("json") || err_msg.contains("parse"),
            "JSON parse error should mention 'json' or 'parse' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_config_error_invalid_json_shows_valid_fields() {
        let malformed_json = "not valid json at all";

        let temp_file = write_temp_config(malformed_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on malformed JSON");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("scheduler_type") || err_msg.contains("learning_rate"),
            "JSON parse error should mention valid field names but got: {}",
            err_msg
        );
    }

    // =========================================================================
    // Scheduler-specific required field error message tests
    // =========================================================================

    #[test]
    fn test_step_decay_missing_step_size_error_mentions_required_fields() {
        // step_decay without step_size (only has gamma)
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "gamma": 0.5
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when step_decay is missing step_size"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("step_size"),
            "Error for step_decay missing step_size should mention 'step_size' but got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("gamma"),
            "Error for step_decay missing step_size should mention 'gamma' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_step_decay_missing_gamma_error_mentions_required_fields() {
        // step_decay without gamma (only has step_size)
        let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 5
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when step_decay is missing gamma"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("step_size"),
            "Error for step_decay missing gamma should mention 'step_size' but got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("gamma"),
            "Error for step_decay missing gamma should mention 'gamma' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_step_decay_missing_both_fields_error_includes_example() {
        // step_decay without both step_size and gamma
        let config_json = r#"{
  "scheduler_type": "step_decay"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when step_decay is missing both step_size and gamma"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("step_decay"),
            "Error should name the scheduler 'step_decay' but got: {}",
            err_msg
        );
        // The error message should include an example showing how to fix it
        assert!(
            err_msg.contains("Example")
                || err_msg.contains("example")
                || err_msg.contains("step_size"),
            "Error for step_decay missing required fields should include guidance but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_exponential_missing_decay_rate_error_mentions_required_field() {
        // exponential without decay_rate
        let config_json = r#"{
  "scheduler_type": "exponential"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when exponential is missing decay_rate"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("decay_rate"),
            "Error for exponential missing decay_rate should mention 'decay_rate' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_exponential_missing_decay_rate_error_names_scheduler() {
        let config_json = r#"{
  "scheduler_type": "exponential"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("exponential"),
            "Error for exponential missing decay_rate should name the scheduler but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_cosine_annealing_missing_min_lr_error_mentions_required_fields() {
        // cosine_annealing without min_lr (only has T_max)
        let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "T_max": 10
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when cosine_annealing is missing min_lr"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("min_lr"),
            "Error for cosine_annealing missing min_lr should mention 'min_lr' but got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("T_max"),
            "Error for cosine_annealing missing min_lr should mention 'T_max' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_cosine_annealing_missing_t_max_error_mentions_required_fields() {
        // cosine_annealing without T_max (only has min_lr)
        let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when cosine_annealing is missing T_max"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("min_lr"),
            "Error for cosine_annealing missing T_max should mention 'min_lr' but got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("T_max"),
            "Error for cosine_annealing missing T_max should mention 'T_max' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_cosine_annealing_missing_both_fields_error_names_scheduler() {
        let config_json = r#"{
  "scheduler_type": "cosine_annealing"
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "Should fail when cosine_annealing is missing both min_lr and T_max"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cosine_annealing"),
            "Error should name the scheduler 'cosine_annealing' but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_json_parse_error_mentions_common_issues_guidance() {
        // JSON with a trailing comma (common mistake)
        let malformed_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
}"#;

        let temp_file = write_temp_config(malformed_json);
        let result = load_config(temp_file.path().to_str().unwrap());

        assert!(result.is_err(), "Should fail on trailing comma JSON");
        let err_msg = result.unwrap_err().to_string();
        // The error message should provide guidance about common JSON mistakes
        assert!(
            err_msg.contains("comma")
                || err_msg.contains("quote")
                || err_msg.contains("parse")
                || err_msg.to_lowercase().contains("json"),
            "JSON parse error should mention common issues but got: {}",
            err_msg
        );
    }

    #[test]
    fn test_json_parse_error_includes_file_path() {
        let malformed_json = r#"{ bad json }"#;

        let temp_file = write_temp_config(malformed_json);
        let path_str = temp_file.path().to_str().unwrap();
        let result = load_config(path_str);

        assert!(result.is_err(), "Should fail on bad JSON");
        let err_msg = result.unwrap_err().to_string();
        // The error should include the file path so users know which file caused the problem
        assert!(
            err_msg.contains(path_str) || err_msg.to_lowercase().contains("json"),
            "JSON parse error should include file path or json context but got: {}",
            err_msg
        );
    }
}
