//! Architecture configuration structures
//!
//! This module provides configuration structures for defining neural network architectures
//! via JSON configuration files. This enables architecture experimentation without code changes.

mod builder;
mod config;
mod validation;

pub use self::builder::build_model;
pub use self::config::{ArchitectureConfig, LayerConfig};

use self::validation::validate_architecture;
use std::error::Error;
use std::fs;

/// Loads an architecture configuration from a JSON file.
///
/// Reads the file at `path`, deserializes it as JSON into an `ArchitectureConfig`,
/// and validates the resulting configuration.
///
/// # Parameters
///
/// - `path` — Filesystem path to a JSON file containing an architecture configuration.
///
/// # Returns
///
/// `ArchitectureConfig` on success.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::architecture::load_architecture;
///
/// let arch = load_architecture("config/architectures/mlp_simple.json").unwrap();
/// assert!(!arch.layers.is_empty());
/// ```
pub fn load_architecture(path: &str) -> Result<ArchitectureConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: ArchitectureConfig = serde_json::from_str(&contents)?;
    validate_architecture(&config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_architecture() {
        let json_content = r#"{
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

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        let result = load_architecture(temp_path);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.layers.len(), 2);
        assert_eq!(config.layers[0].layer_type, "dense");
        assert_eq!(config.layers[0].input_size, Some(784));
        assert_eq!(config.layers[0].output_size, Some(256));
        assert_eq!(config.layers[1].layer_type, "dense");
        assert_eq!(config.layers[1].input_size, Some(256));
        assert_eq!(config.layers[1].output_size, Some(10));
    }

    /// Loads repository example architecture files from `config/architectures/*.json`.
    ///
    /// This test depends on `config/architectures/mlp_simple.json`,
    /// `config/architectures/mlp_medium.json`, and `config/architectures/cnn_simple.json`
    /// being present in the working tree, so it is ignored by default.
    #[test]
    #[ignore = "depends on external example files under config/architectures"]
    fn test_example_configs() {
        let mlp_simple = load_architecture("config/architectures/mlp_simple.json");
        assert!(
            mlp_simple.is_ok(),
            "Failed to load mlp_simple.json: {:?}",
            mlp_simple.err()
        );
        let mlp_simple_config = mlp_simple.unwrap();
        assert_eq!(mlp_simple_config.layers.len(), 2);
        assert_eq!(mlp_simple_config.layers[0].layer_type, "dense");
        assert_eq!(mlp_simple_config.layers[0].input_size, Some(784));
        assert_eq!(mlp_simple_config.layers[0].output_size, Some(256));
        assert_eq!(mlp_simple_config.layers[1].layer_type, "dense");
        assert_eq!(mlp_simple_config.layers[1].input_size, Some(256));
        assert_eq!(mlp_simple_config.layers[1].output_size, Some(10));

        let mlp_medium = load_architecture("config/architectures/mlp_medium.json");
        assert!(
            mlp_medium.is_ok(),
            "Failed to load mlp_medium.json: {:?}",
            mlp_medium.err()
        );
        let mlp_medium_config = mlp_medium.unwrap();
        assert_eq!(mlp_medium_config.layers.len(), 3);
        assert_eq!(mlp_medium_config.layers[0].layer_type, "dense");
        assert_eq!(mlp_medium_config.layers[0].input_size, Some(784));
        assert_eq!(mlp_medium_config.layers[0].output_size, Some(512));
        assert_eq!(mlp_medium_config.layers[1].layer_type, "dense");
        assert_eq!(mlp_medium_config.layers[1].input_size, Some(512));
        assert_eq!(mlp_medium_config.layers[1].output_size, Some(256));
        assert_eq!(mlp_medium_config.layers[2].layer_type, "dense");
        assert_eq!(mlp_medium_config.layers[2].input_size, Some(256));
        assert_eq!(mlp_medium_config.layers[2].output_size, Some(10));

        let cnn_simple = load_architecture("config/architectures/cnn_simple.json");
        assert!(
            cnn_simple.is_ok(),
            "Failed to load cnn_simple.json: {:?}",
            cnn_simple.err()
        );
        let cnn_simple_config = cnn_simple.unwrap();
        assert_eq!(cnn_simple_config.layers.len(), 3);
        assert_eq!(cnn_simple_config.layers[0].layer_type, "conv2d");
        assert_eq!(cnn_simple_config.layers[0].in_channels, Some(1));
        assert_eq!(cnn_simple_config.layers[0].out_channels, Some(8));
        assert_eq!(cnn_simple_config.layers[0].kernel_size, Some(3));
        assert_eq!(cnn_simple_config.layers[0].padding, Some(1));
        assert_eq!(cnn_simple_config.layers[0].stride, Some(1));
        assert_eq!(cnn_simple_config.layers[0].input_height, Some(28));
        assert_eq!(cnn_simple_config.layers[0].input_width, Some(28));
        assert_eq!(cnn_simple_config.layers[1].layer_type, "dense");
        assert_eq!(cnn_simple_config.layers[1].input_size, Some(6272));
        assert_eq!(cnn_simple_config.layers[1].output_size, Some(128));
        assert_eq!(cnn_simple_config.layers[2].layer_type, "dense");
        assert_eq!(cnn_simple_config.layers[2].input_size, Some(128));
        assert_eq!(cnn_simple_config.layers[2].output_size, Some(10));
    }
}
