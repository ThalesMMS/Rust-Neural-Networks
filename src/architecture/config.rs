use serde::Deserialize;

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
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct ArchitectureConfig {
    /// Sequence of layer configurations defining the network structure
    pub layers: Vec<LayerConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_config_rejects_unknown_top_level_field() {
        let json = r#"{
            "layers": [],
            "unexpected": true
        }"#;

        let result = serde_json::from_str::<ArchitectureConfig>(json);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unexpected"));
    }

    #[test]
    fn test_layer_config_rejects_unknown_field() {
        let json = r#"{
            "layers": [
                {
                    "layer_type": "dense",
                    "input_size": 784,
                    "output_size": 10,
                    "outpt_size": 20
                }
            ]
        }"#;

        let result = serde_json::from_str::<ArchitectureConfig>(json);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outpt_size"));
    }
}
