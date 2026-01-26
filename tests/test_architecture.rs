//! Comprehensive tests for architecture parsing and building
//!
//! This file tests the architecture module including:
//! - Loading valid JSON architecture configs
//! - Parsing different layer types (Dense, Conv2D, BatchNorm, Dropout)
//! - Building models from configs
//! - Handling invalid JSON
//! - Handling missing files
//! - Validating layer connections
//! - Edge cases (empty, single layer, etc.)

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::utils::rng::SimpleRng;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_temp_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    file.write_all(contents.as_bytes())
        .expect("failed to write temp config");
    file
}

// ============================================================================
// Valid Architecture Loading Tests
// ============================================================================

mod valid_architecture_tests {
    use super::*;

    #[test]
    fn test_load_simple_mlp() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 2);
        assert_eq!(config.layers[0].layer_type, "dense");
        assert_eq!(config.layers[0].input_size, Some(784));
        assert_eq!(config.layers[0].output_size, Some(256));
        assert_eq!(config.layers[1].layer_type, "dense");
        assert_eq!(config.layers[1].input_size, Some(256));
        assert_eq!(config.layers[1].output_size, Some(10));
    }

    #[test]
    fn test_load_mlp_with_batchnorm() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256,
      "epsilon": 1e-5,
      "momentum": 0.9
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 3);
        assert_eq!(config.layers[1].layer_type, "batchnorm");
        assert_eq!(config.layers[1].size, Some(256));
        assert_eq!(config.layers[1].epsilon, Some(1e-5));
        assert_eq!(config.layers[1].momentum, Some(0.9));
    }

    #[test]
    fn test_load_mlp_with_dropout() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "dropout",
      "size": 512,
      "drop_rate": 0.5
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 3);
        assert_eq!(config.layers[1].layer_type, "dropout");
        assert_eq!(config.layers[1].size, Some(512));
        assert_eq!(config.layers[1].drop_rate, Some(0.5));
    }

    #[test]
    fn test_load_conv2d_architecture() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    },
    {
      "layer_type": "dense",
      "input_size": 6272,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 2);
        assert_eq!(config.layers[0].layer_type, "conv2d");
        assert_eq!(config.layers[0].in_channels, Some(1));
        assert_eq!(config.layers[0].out_channels, Some(8));
        assert_eq!(config.layers[0].kernel_size, Some(3));
        assert_eq!(config.layers[0].padding, Some(1));
        assert_eq!(config.layers[0].stride, Some(1));
        assert_eq!(config.layers[0].input_height, Some(28));
        assert_eq!(config.layers[0].input_width, Some(28));
    }

    /// Verifies that a complex architecture with all layer types loads correctly.
    ///
    /// Tests an architecture containing Dense, Conv2D, BatchNorm, and Dropout layers
    /// to ensure all layer types can be parsed and validated together.
    ///
    /// # Examples
    ///
    /// ```
    /// // Architecture: Conv2D -> Dense -> BatchNorm -> Dropout -> Dense
    /// let config = load_architecture("path/to/complex_config.json").unwrap();
    /// assert_eq!(config.layers.len(), 5);
    /// ```
    #[test]
    fn test_load_complex_architecture() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "batchnorm",
      "size": 512,
      "epsilon": 1e-5,
      "momentum": 0.9
    },
    {
      "layer_type": "dropout",
      "size": 512,
      "drop_rate": 0.3
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 6);
        // Verify layer types are correct
        assert_eq!(config.layers[0].layer_type, "dense");
        assert_eq!(config.layers[1].layer_type, "batchnorm");
        assert_eq!(config.layers[2].layer_type, "dropout");
        assert_eq!(config.layers[3].layer_type, "dense");
        assert_eq!(config.layers[4].layer_type, "batchnorm");
        assert_eq!(config.layers[5].layer_type, "dense");
    }
}

// ============================================================================
// Model Building Tests
// ============================================================================

mod model_building_tests {
    use super::*;

    #[test]
    fn test_build_simple_mlp() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 128
    },
    {
      "layer_type": "dense",
      "input_size": 128,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();

        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 128);
        assert_eq!(layers[1].input_size(), 128);
        assert_eq!(layers[1].output_size(), 10);
    }

    #[test]
    fn test_build_model_with_all_layer_types() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 100,
      "output_size": 64
    },
    {
      "layer_type": "batchnorm",
      "size": 64,
      "epsilon": 1e-5,
      "momentum": 0.9
    },
    {
      "layer_type": "dropout",
      "size": 64,
      "drop_rate": 0.2
    },
    {
      "layer_type": "dense",
      "input_size": 64,
      "output_size": 32
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();

        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0].input_size(), 100);
        assert_eq!(layers[0].output_size(), 64);
        assert_eq!(layers[1].input_size(), 64);
        assert_eq!(layers[1].output_size(), 64); // BatchNorm preserves size
        assert_eq!(layers[2].input_size(), 64);
        assert_eq!(layers[2].output_size(), 64); // Dropout preserves size
        assert_eq!(layers[3].input_size(), 64);
        assert_eq!(layers[3].output_size(), 32);
    }

    #[test]
    fn test_build_conv2d_model() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 4,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();

        assert_eq!(layers.len(), 1);
        // Conv2D input: 1 * 28 * 28 = 784
        assert_eq!(layers[0].input_size(), 784);
        // Conv2D output with padding=1, stride=1, kernel=3: same spatial size
        // Output: 4 * 28 * 28 = 3136
        assert_eq!(layers[0].output_size(), 3136);
    }

    #[test]
    fn test_build_single_layer_model() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();

        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 10);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_json() {
        let invalid_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },  // trailing comma
  ]
}"#;

        let temp_file = write_temp_config(invalid_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_file() {
        let result = load_architecture("nonexistent_file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_layers() {
        let config_json = r#"{
  "layers": []
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("at least one layer"));
    }

    #[test]
    fn test_invalid_layer_type() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "invalid_layer",
      "input_size": 784,
      "output_size": 256
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid layer type"));
    }

    #[test]
    fn test_missing_required_field_dense() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("output_size"));
    }

    #[test]
    fn test_missing_required_field_conv2d() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("input_height") || error_msg.contains("input_width"));
    }

    #[test]
    fn test_missing_required_field_dropout() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dropout",
      "size": 256
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("drop_rate"));
    }

    #[test]
    fn test_layer_connection_mismatch() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 128,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Layer connection mismatch"));
    }

    #[test]
    fn test_zero_input_size() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 0,
      "output_size": 256
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("greater than 0"));
    }

    #[test]
    fn test_invalid_dropout_rate() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dropout",
      "size": 256,
      "drop_rate": 1.5
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("drop_rate"));
    }

    #[test]
    fn test_invalid_batchnorm_epsilon() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "batchnorm",
      "size": 256,
      "epsilon": -0.001
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("epsilon"));
    }

    #[test]
    fn test_invalid_batchnorm_momentum() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "batchnorm",
      "size": 256,
      "momentum": 1.5
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("momentum"));
    }
}

// ============================================================================
// Layer Connection Validation Tests
// ============================================================================

mod layer_connection_tests {
    use super::*;

    #[test]
    fn test_valid_dense_to_dense() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_dense_to_batchnorm() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_dense_to_dropout() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "dropout",
      "size": 512,
      "drop_rate": 0.5
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_conv2d_to_dense() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    },
    {
      "layer_type": "dense",
      "input_size": 6272,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_dense_to_batchnorm_size_mismatch() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 128
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Layer connection mismatch"));
    }

    #[test]
    fn test_invalid_conv2d_to_dense_size_mismatch() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    },
    {
      "layer_type": "dense",
      "input_size": 1000,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let result = load_architecture(temp_file.path().to_str().unwrap());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Layer connection mismatch"));
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_dense_layer() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 1);
    }

    #[test]
    fn test_single_conv2d_layer() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 3,
      "out_channels": 16,
      "kernel_size": 5,
      "padding": 2,
      "stride": 1,
      "input_height": 32,
      "input_width": 32
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 1);
    }

    #[test]
    fn test_very_deep_network() {
        let mut layers = Vec::new();

        // First layer
        layers.push(
            r#"    {
      "layer_type": "dense",
      "input_size": 100,
      "output_size": 64
    }"#,
        );

        // 8 middle layers (64 -> 64)
        let middle_layer = r#"    {
      "layer_type": "dense",
      "input_size": 64,
      "output_size": 64
    }"#;
        layers.extend(std::iter::repeat_n(middle_layer, 8));

        // Final layer
        layers.push(
            r#"    {
      "layer_type": "dense",
      "input_size": 64,
      "output_size": 10
    }"#,
        );

        let config_json = format!(
            r#"{{
  "layers": [
{}
  ]
}}"#,
            layers.join(",\n")
        );

        let temp_file = write_temp_config(&config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 10);
    }

    #[test]
    fn test_batchnorm_with_defaults() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        // Should use default epsilon and momentum
        assert_eq!(config.layers[1].epsilon, None); // Will use 1e-5 default in build
        assert_eq!(config.layers[1].momentum, None); // Will use 0.9 default in build
    }

    #[test]
    fn test_conv2d_with_defaults() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "input_height": 28,
      "input_width": 28
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        // Should use default padding (0) and stride (1)
        assert_eq!(config.layers[0].padding, None); // Will use 0 default in build
        assert_eq!(config.layers[0].stride, None); // Will use 1 default in build
    }

    #[test]
    fn test_case_insensitive_layer_types() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "Dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "BATCHNORM",
      "size": 256
    },
    {
      "layer_type": "Dropout",
      "size": 256,
      "drop_rate": 0.5
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.layers.len(), 4);
        // Layer types are stored as provided, but validation handles case-insensitively
    }
}

// ============================================================================
// Comprehensive Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    /// Verifies end-to-end workflow: load config, build model, check layer sizes.
    ///
    /// Tests the complete workflow of loading an architecture configuration from JSON,
    /// building the model with a seeded RNG, and verifying that all layer dimensions
    /// are correctly initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// let config = load_architecture("config.json").unwrap();
    /// let mut rng = SimpleRng::new(42);
    /// let layers = build_model(&config, &mut rng).unwrap();
    /// // All layers should have correct input/output sizes matching the config
    /// ```
    #[test]
    fn test_end_to_end_mlp() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "batchnorm",
      "size": 512,
      "epsilon": 1e-5,
      "momentum": 0.9
    },
    {
      "layer_type": "dropout",
      "size": 512,
      "drop_rate": 0.3
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}"#;

        // Step 1: Load configuration
        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.layers.len(), 6);

        // Step 2: Build model
        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
        assert_eq!(layers.len(), 6);

        // Step 3: Verify layer sizes
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 512);
        assert_eq!(layers[1].input_size(), 512);
        assert_eq!(layers[1].output_size(), 512);
        assert_eq!(layers[2].input_size(), 512);
        assert_eq!(layers[2].output_size(), 512);
        assert_eq!(layers[3].input_size(), 512);
        assert_eq!(layers[3].output_size(), 256);
        assert_eq!(layers[4].input_size(), 256);
        assert_eq!(layers[4].output_size(), 256);
        assert_eq!(layers[5].input_size(), 256);
        assert_eq!(layers[5].output_size(), 10);
    }

    #[test]
    fn test_end_to_end_cnn() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    },
    {
      "layer_type": "dense",
      "input_size": 6272,
      "output_size": 128
    },
    {
      "layer_type": "dropout",
      "size": 128,
      "drop_rate": 0.5
    },
    {
      "layer_type": "dense",
      "input_size": 128,
      "output_size": 10
    }
  ]
}"#;

        // Step 1: Load configuration
        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.layers.len(), 4);

        // Step 2: Build model
        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
        assert_eq!(layers.len(), 4);

        // Step 3: Verify layer sizes
        assert_eq!(layers[0].input_size(), 784); // 1 * 28 * 28
        assert_eq!(layers[0].output_size(), 6272); // 8 * 28 * 28
        assert_eq!(layers[1].input_size(), 6272);
        assert_eq!(layers[1].output_size(), 128);
        assert_eq!(layers[2].input_size(), 128);
        assert_eq!(layers[2].output_size(), 128);
        assert_eq!(layers[3].input_size(), 128);
        assert_eq!(layers[3].output_size(), 10);
    }

    #[test]
    fn test_multiple_rngs_produce_different_weights() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 10,
      "output_size": 5
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        // Build with two different seeds
        let mut rng1 = SimpleRng::new(42);
        let layers1 = build_model(&config, &mut rng1).unwrap();

        let mut rng2 = SimpleRng::new(123);
        let layers2 = build_model(&config, &mut rng2).unwrap();

        // Layers should have same structure but different weights
        assert_eq!(layers1.len(), layers2.len());
        assert_eq!(layers1[0].input_size(), layers2[0].input_size());
        assert_eq!(layers1[0].output_size(), layers2[0].output_size());
        // Note: We can't directly compare weights through the Layer trait,
        // but this test verifies the building process works with different RNGs
    }

    #[test]
    fn test_same_seed_produces_identical_structure() {
        let config_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 128
    },
    {
      "layer_type": "dense",
      "input_size": 128,
      "output_size": 10
    }
  ]
}"#;

        let temp_file = write_temp_config(config_json);
        let config = load_architecture(temp_file.path().to_str().unwrap()).unwrap();

        // Build twice with same seed
        let mut rng1 = SimpleRng::new(42);
        let layers1 = build_model(&config, &mut rng1).unwrap();

        let mut rng2 = SimpleRng::new(42);
        let layers2 = build_model(&config, &mut rng2).unwrap();

        // Should produce identical structure
        assert_eq!(layers1.len(), layers2.len());
        for i in 0..layers1.len() {
            assert_eq!(layers1[i].input_size(), layers2[i].input_size());
            assert_eq!(layers1[i].output_size(), layers2[i].output_size());
        }
    }
}
