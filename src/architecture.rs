//! Architecture configuration structures
//!
//! This module provides configuration structures for defining neural network architectures
//! via JSON configuration files. This enables architecture experimentation without code changes.

use crate::layers::{BatchNormLayer, Conv2DLayer, DenseLayer, DropoutLayer, Layer};
use crate::utils::rng::SimpleRng;
use serde::Deserialize;
use std::error::Error;
use std::fs;

/// Configuration for a single layer in the neural network.
///
/// Defines the layer type and its parameters. Different layer types require different fields:
///
/// - **Dense**: Requires `input_size` and `output_size`
/// - **Conv2D**: Requires `in_channels`, `out_channels`, `kernel_size`, `input_height`, `input_width`,
///   and optional `padding` (default 0), `stride` (default 1)
/// - **BatchNorm**: Requires `size`, and optional `epsilon` (default 1e-5), `momentum` (default 0.9)
/// - **Dropout**: Requires `size` and `drop_rate` (probability of dropping units, range [0.0, 1.0))
///
/// # Examples
///
/// ```json
/// {
///   "layer_type": "dense",
///   "input_size": 784,
///   "output_size": 512
/// }
/// ```
///
/// ```json
/// {
///   "layer_type": "conv2d",
///   "in_channels": 1,
///   "out_channels": 8,
///   "kernel_size": 3,
///   "padding": 1,
///   "stride": 1,
///   "input_height": 28,
///   "input_width": 28
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct LayerConfig {
    /// Type of layer: "dense", "conv2d", "batchnorm", or "dropout"
    pub layer_type: String,

    // Dense layer parameters
    /// Input size for Dense layer
    pub input_size: Option<usize>,
    /// Output size for Dense layer
    pub output_size: Option<usize>,

    // Conv2D layer parameters
    /// Number of input channels for Conv2D layer
    pub in_channels: Option<usize>,
    /// Number of output channels (filters) for Conv2D layer
    pub out_channels: Option<usize>,
    /// Kernel size for Conv2D layer (assumes square kernel)
    pub kernel_size: Option<usize>,
    /// Zero-padding for Conv2D layer (default: 0)
    pub padding: Option<isize>,
    /// Stride for Conv2D layer (default: 1)
    pub stride: Option<usize>,
    /// Input height for Conv2D layer
    pub input_height: Option<usize>,
    /// Input width for Conv2D layer
    pub input_width: Option<usize>,

    // BatchNorm layer parameters
    /// Size (number of features) for BatchNorm and Dropout layers
    pub size: Option<usize>,
    /// Epsilon for BatchNorm layer (default: 1e-5)
    pub epsilon: Option<f32>,
    /// Momentum for BatchNorm layer (default: 0.9)
    pub momentum: Option<f32>,

    // Dropout layer parameters
    /// Drop rate for Dropout layer (probability of dropping units)
    pub drop_rate: Option<f32>,
}

/// Configuration for the entire neural network architecture.
///
/// Contains a sequence of layer configurations that define the network structure.
/// Layers are applied in the order they appear in the configuration.
///
/// # Example
///
/// ```json
/// {
///   "layers": [
///     {
///       "layer_type": "dense",
///       "input_size": 784,
///       "output_size": 256
///     },
///     {
///       "layer_type": "batchnorm",
///       "size": 256,
///       "epsilon": 1e-5,
///       "momentum": 0.9
///     },
///     {
///       "layer_type": "dropout",
///       "size": 256,
///       "drop_rate": 0.2
///     },
///     {
///       "layer_type": "dense",
///       "input_size": 256,
///       "output_size": 10
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ArchitectureConfig {
    /// Sequence of layer configurations defining the network structure
    pub layers: Vec<LayerConfig>,
}

/// Loads an architecture configuration from a JSON file.
///
/// Reads the file at `path` and deserializes its JSON contents into an `ArchitectureConfig`.
/// Performs basic validation on the configuration structure.
///
/// # Returns
///
/// `Ok(ArchitectureConfig)` on success, or an error if the file cannot be read or the JSON is invalid.
///
/// # Examples
///
/// ```no_run
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

/// Gets the input size of a layer configuration.
///
/// Calculates the input size based on the layer type.
fn get_layer_input_size(layer: &LayerConfig) -> Result<usize, Box<dyn Error>> {
    let layer_type = layer.layer_type.to_lowercase();

    match layer_type.as_str() {
        "dense" => layer.input_size.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Dense layer missing input_size",
            )) as Box<dyn Error>
        }),
        "conv2d" => {
            let in_channels = layer.in_channels.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing in_channels",
                ))
            })?;
            let input_height = layer.input_height.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing input_height",
                ))
            })?;
            let input_width = layer.input_width.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing input_width",
                ))
            })?;
            Ok(in_channels * input_height * input_width)
        }
        "batchnorm" | "dropout" => layer.size.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} layer missing size", layer.layer_type),
            )) as Box<dyn Error>
        }),
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown layer type: {}", layer.layer_type),
        )) as Box<dyn Error>),
    }
}

/// Gets the output size of a layer configuration.
///
/// Calculates the output size based on the layer type and parameters.
fn get_layer_output_size(layer: &LayerConfig) -> Result<usize, Box<dyn Error>> {
    let layer_type = layer.layer_type.to_lowercase();

    match layer_type.as_str() {
        "dense" => layer.output_size.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Dense layer missing output_size",
            )) as Box<dyn Error>
        }),
        "conv2d" => {
            let out_channels = layer.out_channels.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing out_channels",
                ))
            })?;
            let input_height = layer.input_height.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing input_height",
                ))
            })?;
            let input_width = layer.input_width.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing input_width",
                ))
            })?;
            let kernel_size = layer.kernel_size.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Conv2D layer missing kernel_size",
                ))
            })?;
            let padding = layer.padding.unwrap_or(0);
            let stride = layer.stride.unwrap_or(1);

            // Calculate output dimensions
            let out_height = ((input_height as isize + 2 * padding - kernel_size as isize)
                / stride as isize
                + 1) as usize;
            let out_width = ((input_width as isize + 2 * padding - kernel_size as isize)
                / stride as isize
                + 1) as usize;

            Ok(out_channels * out_height * out_width)
        }
        "batchnorm" | "dropout" => {
            // BatchNorm and Dropout don't change the size
            layer.size.ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{} layer missing size", layer.layer_type),
                )) as Box<dyn Error>
            })
        }
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown layer type: {}", layer.layer_type),
        )) as Box<dyn Error>),
    }
}

/// Validates an architecture configuration.
///
/// Checks that:
/// - Architecture has at least one layer
/// - Each layer has the required fields for its type
/// - Parameter values are within valid ranges
/// - Layer connections are valid (output size of layer i matches input size of layer i+1)
///
/// # Errors
///
/// Returns an error if validation fails with a descriptive message.
fn validate_architecture(config: &ArchitectureConfig) -> Result<(), Box<dyn Error>> {
    if config.layers.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Architecture must have at least one layer",
        )));
    }

    // Validate each layer individually
    for (i, layer) in config.layers.iter().enumerate() {
        validate_layer(layer, i)?;
    }

    // Validate layer connections (output of layer i must match input of layer i+1)
    for i in 0..config.layers.len() - 1 {
        let current_layer = &config.layers[i];
        let next_layer = &config.layers[i + 1];

        let current_output = get_layer_output_size(current_layer)?;
        let next_input = get_layer_input_size(next_layer)?;

        if current_output != next_input {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Layer connection mismatch: Layer {} output size ({}) does not match Layer {} input size ({})",
                    i, current_output, i + 1, next_input
                ),
            )));
        }
    }

    Ok(())
}

/// Validates a single layer configuration.
///
/// Checks that the layer has all required fields for its type and that
/// parameter values are within valid ranges.
fn validate_layer(layer: &LayerConfig, index: usize) -> Result<(), Box<dyn Error>> {
    let layer_type = layer.layer_type.to_lowercase();

    match layer_type.as_str() {
        "dense" => {
            if layer.input_size.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Dense layer requires 'input_size'", index),
                )));
            }
            if layer.output_size.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Dense layer requires 'output_size'", index),
                )));
            }
            if let Some(input_size) = layer.input_size {
                if input_size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: input_size must be greater than 0", index),
                    )));
                }
            }
            if let Some(output_size) = layer.output_size {
                if output_size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: output_size must be greater than 0", index),
                    )));
                }
            }
        }
        "conv2d" => {
            if layer.in_channels.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Conv2D layer requires 'in_channels'", index),
                )));
            }
            if layer.out_channels.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Conv2D layer requires 'out_channels'", index),
                )));
            }
            if layer.kernel_size.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Conv2D layer requires 'kernel_size'", index),
                )));
            }
            if layer.input_height.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Conv2D layer requires 'input_height'", index),
                )));
            }
            if layer.input_width.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Conv2D layer requires 'input_width'", index),
                )));
            }

            // Validate positive values
            if let Some(in_channels) = layer.in_channels {
                if in_channels == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: in_channels must be greater than 0", index),
                    )));
                }
            }
            if let Some(out_channels) = layer.out_channels {
                if out_channels == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: out_channels must be greater than 0", index),
                    )));
                }
            }
            if let Some(kernel_size) = layer.kernel_size {
                if kernel_size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: kernel_size must be greater than 0", index),
                    )));
                }
            }
            if let Some(stride) = layer.stride {
                if stride == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: stride must be greater than 0", index),
                    )));
                }
            }
        }
        "batchnorm" => {
            if layer.size.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: BatchNorm layer requires 'size'", index),
                )));
            }
            if let Some(size) = layer.size {
                if size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: size must be greater than 0", index),
                    )));
                }
            }
            if let Some(epsilon) = layer.epsilon {
                if epsilon <= 0.0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: epsilon must be positive", index),
                    )));
                }
            }
            if let Some(momentum) = layer.momentum {
                if !(0.0..=1.0).contains(&momentum) {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: momentum must be in range [0.0, 1.0]", index),
                    )));
                }
            }
        }
        "dropout" => {
            if layer.size.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Dropout layer requires 'size'", index),
                )));
            }
            if layer.drop_rate.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Layer {}: Dropout layer requires 'drop_rate'", index),
                )));
            }
            if let Some(size) = layer.size {
                if size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: size must be greater than 0", index),
                    )));
                }
            }
            if let Some(drop_rate) = layer.drop_rate {
                if !(0.0..1.0).contains(&drop_rate) {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: drop_rate must be in range [0.0, 1.0)", index),
                    )));
                }
            }
        }
        _ => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Layer {}: Invalid layer type '{}'. Must be one of: dense, conv2d, batchnorm, dropout",
                    index, layer.layer_type
                ),
            )));
        }
    }

    Ok(())
}

/// Builds a neural network model from architecture configuration.
///
/// Creates a vector of layers based on the provided architecture configuration.
/// Each layer is initialized with appropriate parameters from the config and
/// uses the provided RNG for weight initialization.
///
/// # Arguments
///
/// * `config` - Architecture configuration defining the layer sequence
/// * `rng` - Random number generator for weight initialization
///
/// # Returns
///
/// A vector of boxed trait objects implementing the Layer trait, ordered as specified in config.
///
/// # Errors
///
/// Returns an error if a layer configuration is invalid or if layer construction fails.
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::architecture::{load_architecture, build_model};
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let config = load_architecture("config/architectures/mlp_simple.json").unwrap();
/// let mut rng = SimpleRng::new(42);
/// let layers = build_model(&config, &mut rng).unwrap();
/// assert_eq!(layers.len(), config.layers.len());
/// ```
pub fn build_model(
    config: &ArchitectureConfig,
    rng: &mut SimpleRng,
) -> Result<Vec<Box<dyn Layer>>, Box<dyn Error>> {
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();

    for (i, layer_config) in config.layers.iter().enumerate() {
        let layer_type = layer_config.layer_type.to_lowercase();

        match layer_type.as_str() {
            "dense" => {
                let input_size = layer_config.input_size.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Dense layer missing input_size", i),
                    ))
                })?;
                let output_size = layer_config.output_size.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Dense layer missing output_size", i),
                    ))
                })?;
                let layer = DenseLayer::new(input_size, output_size, rng);
                layers.push(Box::new(layer));
            }
            "conv2d" => {
                let in_channels = layer_config.in_channels.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Conv2D layer missing in_channels", i),
                    ))
                })?;
                let out_channels = layer_config.out_channels.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Conv2D layer missing out_channels", i),
                    ))
                })?;
                let kernel_size = layer_config.kernel_size.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Conv2D layer missing kernel_size", i),
                    ))
                })?;
                let input_height = layer_config.input_height.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Conv2D layer missing input_height", i),
                    ))
                })?;
                let input_width = layer_config.input_width.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Conv2D layer missing input_width", i),
                    ))
                })?;
                let padding = layer_config.padding.unwrap_or(0);
                let stride = layer_config.stride.unwrap_or(1);

                let layer = Conv2DLayer::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    stride,
                    input_height,
                    input_width,
                    rng,
                );
                layers.push(Box::new(layer));
            }
            "batchnorm" => {
                let size = layer_config.size.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: BatchNorm layer missing size", i),
                    ))
                })?;
                let epsilon = layer_config.epsilon.unwrap_or(1e-5);
                let momentum = layer_config.momentum.unwrap_or(0.9);

                let layer = BatchNormLayer::new(size, epsilon, momentum);
                layers.push(Box::new(layer));
            }
            "dropout" => {
                let size = layer_config.size.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Dropout layer missing size", i),
                    ))
                })?;
                let drop_rate = layer_config.drop_rate.ok_or_else(|| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Layer {}: Dropout layer missing drop_rate", i),
                    ))
                })?;

                let layer = DropoutLayer::new(size, drop_rate, rng);
                layers.push(Box::new(layer));
            }
            _ => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Layer {}: Invalid layer type '{}'. Must be one of: dense, conv2d, batchnorm, dropout",
                        i, layer_config.layer_type
                    ),
                )));
            }
        }
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_dense_layer() {
        let layer = LayerConfig {
            layer_type: "dense".to_string(),
            input_size: Some(784),
            output_size: Some(256),
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: None,
            stride: None,
            input_height: None,
            input_width: None,
            size: None,
            epsilon: None,
            momentum: None,
            drop_rate: None,
        };

        assert!(validate_layer(&layer, 0).is_ok());
    }

    #[test]
    fn test_validate_dense_layer_missing_fields() {
        let layer = LayerConfig {
            layer_type: "dense".to_string(),
            input_size: None,
            output_size: Some(256),
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: None,
            stride: None,
            input_height: None,
            input_width: None,
            size: None,
            epsilon: None,
            momentum: None,
            drop_rate: None,
        };

        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_invalid_layer_type() {
        let layer = LayerConfig {
            layer_type: "invalid".to_string(),
            input_size: None,
            output_size: None,
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: None,
            stride: None,
            input_height: None,
            input_width: None,
            size: None,
            epsilon: None,
            momentum: None,
            drop_rate: None,
        };

        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_empty_architecture() {
        let config = ArchitectureConfig { layers: vec![] };
        assert!(validate_architecture(&config).is_err());
    }

    #[test]
    fn test_validate_dropout_layer() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            input_size: None,
            output_size: None,
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: None,
            stride: None,
            input_height: None,
            input_width: None,
            size: Some(256),
            epsilon: None,
            momentum: None,
            drop_rate: Some(0.5),
        };

        assert!(validate_layer(&layer, 0).is_ok());
    }

    #[test]
    fn test_validate_dropout_layer_invalid_rate() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            input_size: None,
            output_size: None,
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: None,
            stride: None,
            input_height: None,
            input_width: None,
            size: Some(256),
            epsilon: None,
            momentum: None,
            drop_rate: Some(1.5),
        };

        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_load_architecture() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary JSON file with a valid architecture config
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

        // Load the architecture
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

    #[test]
    fn test_build_model() {
        use crate::utils::rng::SimpleRng;

        // Create a simple config with two dense layers
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(256),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(256),
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };

        let mut rng = SimpleRng::new(42);
        let result = build_model(&config, &mut rng);
        assert!(result.is_ok());

        let layers = result.unwrap();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 256);
        assert_eq!(layers[1].input_size(), 256);
        assert_eq!(layers[1].output_size(), 10);
    }

    #[test]
    fn test_build_model_with_batchnorm_and_dropout() {
        use crate::utils::rng::SimpleRng;

        // Create a config with dense, batchnorm, and dropout layers
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(256),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "batchnorm".to_string(),
                    input_size: None,
                    output_size: None,
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: Some(256),
                    epsilon: Some(1e-5),
                    momentum: Some(0.9),
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dropout".to_string(),
                    input_size: None,
                    output_size: None,
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: Some(256),
                    epsilon: None,
                    momentum: None,
                    drop_rate: Some(0.5),
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(256),
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };

        let mut rng = SimpleRng::new(42);
        let result = build_model(&config, &mut rng);
        assert!(result.is_ok());

        let layers = result.unwrap();
        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 256);
        assert_eq!(layers[1].input_size(), 256);
        assert_eq!(layers[1].output_size(), 256);
        assert_eq!(layers[2].input_size(), 256);
        assert_eq!(layers[2].output_size(), 256);
        assert_eq!(layers[3].input_size(), 256);
        assert_eq!(layers[3].output_size(), 10);
    }

    #[test]
    fn test_build_model_conv2d() {
        use crate::utils::rng::SimpleRng;

        // Create a config with a conv2d layer
        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "conv2d".to_string(),
                input_size: None,
                output_size: None,
                in_channels: Some(1),
                out_channels: Some(8),
                kernel_size: Some(3),
                padding: Some(1),
                stride: Some(1),
                input_height: Some(28),
                input_width: Some(28),
                size: None,
                epsilon: None,
                momentum: None,
                drop_rate: None,
            }],
        };

        let mut rng = SimpleRng::new(42);
        let result = build_model(&config, &mut rng);
        assert!(result.is_ok());

        let layers = result.unwrap();
        assert_eq!(layers.len(), 1);
        // Conv2D input size is flattened (in_channels * height * width)
        assert_eq!(layers[0].input_size(), 28 * 28);
        // Conv2D output size is flattened (out_channels * out_height * out_width)
        // With padding=1, stride=1, kernel=3: output_size = input_size
        assert_eq!(layers[0].output_size(), 8 * 28 * 28);
    }

    #[test]
    fn test_build_model_invalid_layer_type() {
        use crate::utils::rng::SimpleRng;

        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "invalid".to_string(),
                input_size: None,
                output_size: None,
                in_channels: None,
                out_channels: None,
                kernel_size: None,
                padding: None,
                stride: None,
                input_height: None,
                input_width: None,
                size: None,
                epsilon: None,
                momentum: None,
                drop_rate: None,
            }],
        };

        let mut rng = SimpleRng::new(42);
        let result = build_model(&config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_model_missing_fields() {
        use crate::utils::rng::SimpleRng;

        // Dense layer missing output_size
        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "dense".to_string(),
                input_size: Some(784),
                output_size: None, // Missing required field
                in_channels: None,
                out_channels: None,
                kernel_size: None,
                padding: None,
                stride: None,
                input_height: None,
                input_width: None,
                size: None,
                epsilon: None,
                momentum: None,
                drop_rate: None,
            }],
        };

        let mut rng = SimpleRng::new(42);
        let result = build_model(&config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_layer_connection_mismatch() {
        // Create a config with mismatched layer connections
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(256), // Output is 256
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(128), // Input is 128 - MISMATCH!
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };

        let result = validate_architecture(&config);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Layer connection mismatch"));
    }

    #[test]
    fn test_validate_layer_connection_valid() {
        // Create a config with valid layer connections
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(256),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "batchnorm".to_string(),
                    input_size: None,
                    output_size: None,
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: Some(256), // Matches previous layer output
                    epsilon: Some(1e-5),
                    momentum: Some(0.9),
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(256), // Matches batchnorm size
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };

        let result = validate_architecture(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation() {
        // Test that validation catches various error conditions

        // Valid config should pass
        let valid_config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(128),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(128),
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };
        assert!(validate_architecture(&valid_config).is_ok());

        // Empty layers should fail
        let empty_config = ArchitectureConfig { layers: vec![] };
        assert!(validate_architecture(&empty_config).is_err());

        // Invalid layer type should fail
        let invalid_type_config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "invalid".to_string(),
                input_size: None,
                output_size: None,
                in_channels: None,
                out_channels: None,
                kernel_size: None,
                padding: None,
                stride: None,
                input_height: None,
                input_width: None,
                size: None,
                epsilon: None,
                momentum: None,
                drop_rate: None,
            }],
        };
        assert!(validate_architecture(&invalid_type_config).is_err());

        // Mismatched layer connections should fail
        let mismatch_config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(784),
                    output_size: Some(256),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
                LayerConfig {
                    layer_type: "dense".to_string(),
                    input_size: Some(128), // Mismatch!
                    output_size: Some(10),
                    in_channels: None,
                    out_channels: None,
                    kernel_size: None,
                    padding: None,
                    stride: None,
                    input_height: None,
                    input_width: None,
                    size: None,
                    epsilon: None,
                    momentum: None,
                    drop_rate: None,
                },
            ],
        };
        assert!(validate_architecture(&mismatch_config).is_err());
    }

    #[test]
    fn test_example_configs() {
        // Test that all example config files parse successfully

        // Test mlp_simple.json
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

        // Test mlp_medium.json
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

        // Test cnn_simple.json
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
