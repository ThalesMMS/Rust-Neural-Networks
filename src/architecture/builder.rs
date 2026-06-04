use super::config::ArchitectureConfig;
use super::validation::validate_architecture;
use crate::layers::pooling::{AvgPoolLayer, MaxPoolLayer};
use crate::layers::{
    BatchNormLayer, Conv2DLayer, DenseLayer, DropoutLayer, GlobalAvgPoolLayer, Layer, ResidualBlock,
};
use crate::utils::rng::SimpleRng;
use std::error::Error;

fn invalid_data(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
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
    validate_architecture(config)?;
    let mut layers: Vec<Box<dyn Layer>> = Vec::new();

    for (i, layer_config) in config.layers.iter().enumerate() {
        let missing = |field: &str| {
            invalid_data(format!(
                "Layer {}: {} layer missing {}",
                i, layer_config.layer_type, field
            ))
        };

        match layer_config.layer_type.to_lowercase().as_str() {
            "dense" => {
                let input_size = layer_config
                    .input_size
                    .ok_or_else(|| missing("input_size"))?;
                let output_size = layer_config
                    .output_size
                    .ok_or_else(|| missing("output_size"))?;
                layers.push(Box::new(DenseLayer::new(input_size, output_size, rng)));
            }
            "conv2d" => {
                let in_channels = layer_config
                    .in_channels
                    .ok_or_else(|| missing("in_channels"))?;
                let out_channels = layer_config
                    .out_channels
                    .ok_or_else(|| missing("out_channels"))?;
                let kernel_size = layer_config
                    .kernel_size
                    .ok_or_else(|| missing("kernel_size"))?;
                let input_height = layer_config
                    .input_height
                    .ok_or_else(|| missing("input_height"))?;
                let input_width = layer_config
                    .input_width
                    .ok_or_else(|| missing("input_width"))?;
                let padding = layer_config.padding.unwrap_or(0);
                let stride = layer_config.stride.unwrap_or(1);

                layers.push(Box::new(Conv2DLayer::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    stride,
                    input_height,
                    input_width,
                    rng,
                )));
            }
            "residual_block" => {
                let in_channels = layer_config
                    .in_channels
                    .ok_or_else(|| missing("in_channels"))?;
                let out_channels = layer_config
                    .out_channels
                    .ok_or_else(|| missing("out_channels"))?;
                let input_height = layer_config
                    .input_height
                    .ok_or_else(|| missing("input_height"))?;
                let input_width = layer_config
                    .input_width
                    .ok_or_else(|| missing("input_width"))?;
                let stride = layer_config.stride.unwrap_or(1);

                layers.push(Box::new(ResidualBlock::new(
                    in_channels,
                    out_channels,
                    stride,
                    input_height,
                    input_width,
                    rng,
                )));
            }
            "batchnorm" => {
                let size = layer_config.size.ok_or_else(|| missing("size"))?;
                layers.push(Box::new(BatchNormLayer::new(
                    size,
                    layer_config.epsilon.unwrap_or(1e-5),
                    layer_config.momentum.unwrap_or(0.9),
                )));
            }
            "dropout" => {
                let size = layer_config.size.ok_or_else(|| missing("size"))?;
                let drop_rate = layer_config.drop_rate.ok_or_else(|| missing("drop_rate"))?;
                layers.push(Box::new(DropoutLayer::new(size, drop_rate, rng)));
            }
            "globalavgpool" => {
                let input_height = layer_config
                    .pool_input_height
                    .ok_or_else(|| missing("pool_input_height"))?;
                let input_width = layer_config
                    .pool_input_width
                    .ok_or_else(|| missing("pool_input_width"))?;
                let channels = layer_config
                    .pool_channels
                    .ok_or_else(|| missing("pool_channels"))?;

                layers.push(Box::new(GlobalAvgPoolLayer::new(
                    input_height,
                    input_width,
                    channels,
                )));
            }
            "maxpool" | "avgpool" | "pool" => {
                let pool_size = layer_config.pool_size.ok_or_else(|| missing("pool_size"))?;
                let pool_stride = layer_config
                    .pool_stride
                    .ok_or_else(|| missing("pool_stride"))?;
                let pool_padding = layer_config.pool_padding.unwrap_or(0);
                let input_height = layer_config
                    .pool_input_height
                    .ok_or_else(|| missing("pool_input_height"))?;
                let input_width = layer_config
                    .pool_input_width
                    .ok_or_else(|| missing("pool_input_width"))?;
                let channels = layer_config
                    .pool_channels
                    .ok_or_else(|| missing("pool_channels"))?;

                let mode = match layer_config.layer_type.to_lowercase().as_str() {
                    "maxpool" => "max",
                    "avgpool" => "avg",
                    "pool" => layer_config
                        .pool_mode
                        .as_deref()
                        .ok_or_else(|| missing("pool_mode"))?,
                    _ => unreachable!(),
                };

                match mode {
                    "max" => layers.push(Box::new(MaxPoolLayer::new(
                        channels,
                        input_height,
                        input_width,
                        pool_size,
                        pool_stride,
                        pool_padding,
                    ))),
                    "avg" => layers.push(Box::new(AvgPoolLayer::new(
                        channels,
                        input_height,
                        input_width,
                        pool_size,
                        pool_stride,
                        pool_padding,
                    ))),
                    _ => {
                        return Err(invalid_data(format!(
                            "Layer {}: Invalid pool_mode '{}'. Must be one of: max, avg",
                            i, mode
                        )));
                    }
                };
            }
            _ => {
                return Err(invalid_data(format!(
                    "Layer {}: Invalid layer type '{}'. Must be one of: dense, conv2d, residual_block, batchnorm, dropout, globalavgpool, maxpool, avgpool, pool",
                    i, layer_config.layer_type
                )));
            }
        }
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::super::config::{ArchitectureConfig, LayerConfig};
    use super::*;
    use crate::utils::rng::SimpleRng;

    fn dense(input_size: usize, output_size: usize) -> LayerConfig {
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: Some(input_size),
            output_size: Some(output_size),
            ..Default::default()
        }
    }

    #[test]
    fn test_build_model() {
        let config = ArchitectureConfig {
            layers: vec![dense(784, 256), dense(256, 10)],
        };

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].input_size(), 784);
        assert_eq!(layers[0].output_size(), 256);
        assert_eq!(layers[1].input_size(), 256);
        assert_eq!(layers[1].output_size(), 10);
    }

    #[test]
    fn test_build_model_with_batchnorm_and_dropout() {
        let config = ArchitectureConfig {
            layers: vec![
                dense(784, 256),
                LayerConfig {
                    layer_type: "batchnorm".to_string(),
                    size: Some(256),
                    epsilon: Some(1e-5),
                    momentum: Some(0.9),
                    ..Default::default()
                },
                LayerConfig {
                    layer_type: "dropout".to_string(),
                    size: Some(256),
                    drop_rate: Some(0.5),
                    ..Default::default()
                },
                dense(256, 10),
            ],
        };

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
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
        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "conv2d".to_string(),
                in_channels: Some(1),
                out_channels: Some(8),
                kernel_size: Some(3),
                padding: Some(1),
                stride: Some(1),
                input_height: Some(28),
                input_width: Some(28),
                ..Default::default()
            }],
        };

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].input_size(), 28 * 28);
        assert_eq!(layers[0].output_size(), 8 * 28 * 28);
    }

    #[test]
    fn test_build_model_uses_conv2d_and_batchnorm_defaults() {
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "conv2d".to_string(),
                    in_channels: Some(1),
                    out_channels: Some(2),
                    kernel_size: Some(3),
                    input_height: Some(5),
                    input_width: Some(5),
                    ..Default::default()
                },
                LayerConfig {
                    layer_type: "batchnorm".to_string(),
                    size: Some(18),
                    ..Default::default()
                },
            ],
        };

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();

        let conv = layers[0]
            .as_any()
            .downcast_ref::<crate::layers::Conv2DLayer>()
            .expect("expected Conv2DLayer");
        assert_eq!(conv.padding(), 0);
        assert_eq!(conv.stride(), 1);

        let batchnorm = layers[1]
            .as_any()
            .downcast_ref::<crate::layers::BatchNormLayer>()
            .expect("expected BatchNormLayer");
        assert!((batchnorm.epsilon() - 1e-5f32).abs() < f32::EPSILON);
        assert!((batchnorm.momentum() - 0.9f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_build_model_with_pooling_layers() {
        let config = ArchitectureConfig {
            layers: vec![
                LayerConfig {
                    layer_type: "conv2d".to_string(),
                    in_channels: Some(1),
                    out_channels: Some(2),
                    kernel_size: Some(3),
                    padding: Some(1),
                    stride: Some(1),
                    input_height: Some(8),
                    input_width: Some(8),
                    ..Default::default()
                },
                LayerConfig {
                    layer_type: "maxpool".to_string(),
                    pool_size: Some(2),
                    pool_stride: Some(2),
                    pool_padding: Some(0),
                    pool_input_height: Some(8),
                    pool_input_width: Some(8),
                    pool_channels: Some(2),
                    ..Default::default()
                },
                LayerConfig {
                    layer_type: "pool".to_string(),
                    pool_mode: Some("avg".to_string()),
                    pool_size: Some(2),
                    pool_stride: Some(2),
                    pool_padding: Some(0),
                    pool_input_height: Some(4),
                    pool_input_width: Some(4),
                    pool_channels: Some(2),
                    ..Default::default()
                },
            ],
        };

        let mut rng = SimpleRng::new(42);
        let layers = build_model(&config, &mut rng).unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].input_size(), 8 * 8);
        assert_eq!(layers[0].output_size(), 2 * 8 * 8);
        assert_eq!(layers[1].input_size(), 2 * 8 * 8);
        assert_eq!(layers[1].output_size(), 2 * 4 * 4);
        assert_eq!(layers[2].input_size(), 2 * 4 * 4);
        assert_eq!(layers[2].output_size(), 2 * 2 * 2);

        assert!(layers[1].as_any().downcast_ref::<MaxPoolLayer>().is_some());
        assert!(layers[2].as_any().downcast_ref::<AvgPoolLayer>().is_some());
    }

    #[test]
    fn test_build_model_invalid_layer_type() {
        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "invalid".to_string(),
                ..Default::default()
            }],
        };

        let mut rng = SimpleRng::new(42);
        assert!(build_model(&config, &mut rng).is_err());
    }

    #[test]
    fn test_build_model_missing_fields() {
        let config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "dense".to_string(),
                input_size: Some(784),
                ..Default::default()
            }],
        };

        let mut rng = SimpleRng::new(42);
        assert!(build_model(&config, &mut rng).is_err());
    }
}
