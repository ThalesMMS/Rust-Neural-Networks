use serde::Deserialize;

/// Configuration for a single layer in the neural network.
///
/// Defines the layer type and its parameters. Different layer types require different fields:
///
/// - **Dense**: Requires `input_size` and `output_size`
/// - **Conv2D**: Requires `in_channels`, `out_channels`, `kernel_size`, `input_height`, `input_width`,
///   and optional `padding` (default 0), `stride` (default 1)
/// - **ResidualBlock**: Requires `in_channels`, `out_channels`, `input_height`, `input_width`,
///   and optional `stride` (default 1). Uses an internal identity or projection shortcut.
/// - **BatchNorm**: Requires `size`, and optional `epsilon` (default 1e-5), `momentum` (default 0.9)
/// - **Dropout**: Requires `size` and `drop_rate` (probability of dropping units, range [0.0, 1.0))
/// - **GlobalAvgPool**: Requires `pool_input_height`, `pool_input_width`, and `pool_channels`
/// - **Pooling**: Supported via:
///   - `layer_type`: "maxpool" or "avgpool" and pooling fields (see below), or
///   - `layer_type`: "pool" with `pool_mode`: "max" or "avg" and pooling fields
///
///   Pooling requires `pool_size`, `pool_input_height`, `pool_input_width`, and `pool_channels`, and
///   supports optional `pool_stride` (default: `pool_size`) and `pool_padding` (default: 0).
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
///
/// ```json
/// {
///   "layer_type": "residual_block",
///   "in_channels": 16,
///   "out_channels": 32,
///   "stride": 2,
///   "input_height": 32,
///   "input_width": 32
/// }
/// ```
///
/// ```json
/// {
///   "layer_type": "globalavgpool",
///   "pool_input_height": 4,
///   "pool_input_width": 4,
///   "pool_channels": 128
/// }
/// ```
///
/// ```json
/// {
///   "layer_type": "maxpool",
///   "pool_size": 2,
///   "pool_stride": 2,
///   "pool_padding": 0,
///   "pool_input_height": 28,
///   "pool_input_width": 28,
///   "pool_channels": 8
/// }
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerConfig {
    /// Type of layer: "dense", "conv2d", "residual_block", "batchnorm", "dropout",
    /// "globalavgpool", "maxpool", "avgpool", or "pool"
    pub layer_type: String,

    // Dense layer parameters
    /// Input size for Dense layer
    pub input_size: Option<usize>,
    /// Output size for Dense layer
    pub output_size: Option<usize>,

    // Conv2D and ResidualBlock layer parameters
    /// Number of input channels for Conv2D and ResidualBlock layers
    pub in_channels: Option<usize>,
    /// Number of output channels for Conv2D and ResidualBlock layers
    pub out_channels: Option<usize>,
    /// Kernel size for Conv2D layer (assumes square kernel)
    pub kernel_size: Option<usize>,
    /// Zero-padding for Conv2D layer (default: 0)
    pub padding: Option<isize>,
    /// Stride for Conv2D and ResidualBlock layers (default: 1)
    pub stride: Option<usize>,
    /// Input height for Conv2D and ResidualBlock layers
    pub input_height: Option<usize>,
    /// Input width for Conv2D and ResidualBlock layers
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

    // Pooling layer parameters
    /// Window size for pooling layers (assumes square window)
    pub pool_size: Option<usize>,
    /// Stride for pooling layers (default: pool_size)
    pub pool_stride: Option<usize>,
    /// Input height for pooling layers
    pub pool_input_height: Option<usize>,
    /// Input width for pooling layers
    pub pool_input_width: Option<usize>,
    /// Number of channels for pooling layers
    pub pool_channels: Option<usize>,
    /// Padding for pooling layers (default: 0)
    pub pool_padding: Option<isize>,
    /// Pooling mode for pooling layers: "max" or "avg" (optional; alternative to layer_type variants)
    pub pool_mode: Option<String>,
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
