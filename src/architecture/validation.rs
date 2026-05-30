use super::config::{ArchitectureConfig, LayerConfig};
use std::error::Error;

fn invalid_data(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

/// Gets the input size of a layer configuration.
///
/// Calculates the input size based on the layer type.
fn get_layer_input_size(layer: &LayerConfig) -> Result<usize, Box<dyn Error>> {
    match layer.layer_type.to_lowercase().as_str() {
        "dense" => layer
            .input_size
            .ok_or_else(|| invalid_data("Dense layer missing input_size")),
        "conv2d" => {
            let in_channels = layer
                .in_channels
                .ok_or_else(|| invalid_data("Conv2D layer missing in_channels"))?;
            let input_height = layer
                .input_height
                .ok_or_else(|| invalid_data("Conv2D layer missing input_height"))?;
            let input_width = layer
                .input_width
                .ok_or_else(|| invalid_data("Conv2D layer missing input_width"))?;
            in_channels
                .checked_mul(input_height)
                .and_then(|v| v.checked_mul(input_width))
                .ok_or_else(|| invalid_data("Conv2D input size overflow"))
        }
        "maxpool" | "avgpool" | "pool" => {
            let channels = layer
                .pool_channels
                .ok_or_else(|| invalid_data("Pooling layer missing pool_channels"))?;
            let input_height = layer
                .pool_input_height
                .ok_or_else(|| invalid_data("Pooling layer missing pool_input_height"))?;
            let input_width = layer
                .pool_input_width
                .ok_or_else(|| invalid_data("Pooling layer missing pool_input_width"))?;

            channels
                .checked_mul(input_height)
                .and_then(|v| v.checked_mul(input_width))
                .ok_or_else(|| invalid_data("Pooling input size overflow"))
        }
        "batchnorm" | "dropout" => layer
            .size
            .ok_or_else(|| invalid_data(format!("{} layer missing size", layer.layer_type))),
        _ => Err(invalid_data(format!(
            "Unknown layer type: {}",
            layer.layer_type
        ))),
    }
}

/// Gets the output size of a layer configuration.
///
/// Calculates the output size based on the layer type and parameters.
fn get_layer_output_size(layer: &LayerConfig) -> Result<usize, Box<dyn Error>> {
    match layer.layer_type.to_lowercase().as_str() {
        "dense" => layer
            .output_size
            .ok_or_else(|| invalid_data("Dense layer missing output_size")),
        "conv2d" => {
            let out_channels = layer
                .out_channels
                .ok_or_else(|| invalid_data("Conv2D layer missing out_channels"))?;
            let input_height = layer
                .input_height
                .ok_or_else(|| invalid_data("Conv2D layer missing input_height"))?;
            let input_width = layer
                .input_width
                .ok_or_else(|| invalid_data("Conv2D layer missing input_width"))?;
            let kernel_size = layer
                .kernel_size
                .ok_or_else(|| invalid_data("Conv2D layer missing kernel_size"))?;
            let padding = layer.padding.unwrap_or(0);
            let stride = layer.stride.unwrap_or(1);

            let out_height_isize =
                (input_height as isize + 2 * padding - kernel_size as isize) / stride as isize + 1;
            if out_height_isize < 0 {
                return Err(invalid_data("Conv2D output height is negative"));
            }
            let out_width_isize =
                (input_width as isize + 2 * padding - kernel_size as isize) / stride as isize + 1;
            if out_width_isize < 0 {
                return Err(invalid_data("Conv2D output width is negative"));
            }
            let out_height = out_height_isize as usize;
            let out_width = out_width_isize as usize;

            out_channels
                .checked_mul(out_height)
                .and_then(|v| v.checked_mul(out_width))
                .ok_or_else(|| invalid_data("Conv2D output size overflow"))
        }
        "maxpool" | "avgpool" | "pool" => {
            let channels = layer
                .pool_channels
                .ok_or_else(|| invalid_data("Pooling layer missing pool_channels"))?;
            let input_height = layer
                .pool_input_height
                .ok_or_else(|| invalid_data("Pooling layer missing pool_input_height"))?;
            let input_width = layer
                .pool_input_width
                .ok_or_else(|| invalid_data("Pooling layer missing pool_input_width"))?;
            let pool_size = layer
                .pool_size
                .ok_or_else(|| invalid_data("Pooling layer missing pool_size"))?;
            let pool_padding = layer.pool_padding.unwrap_or(0);
            let pool_stride = layer.pool_stride.unwrap_or(pool_size);

            let out_height_isize = (input_height as isize + 2 * pool_padding - pool_size as isize)
                / pool_stride as isize
                + 1;
            if out_height_isize < 0 {
                return Err(invalid_data("Pooling output height is negative"));
            }
            let out_width_isize = (input_width as isize + 2 * pool_padding - pool_size as isize)
                / pool_stride as isize
                + 1;
            if out_width_isize < 0 {
                return Err(invalid_data("Pooling output width is negative"));
            }
            let out_height = out_height_isize as usize;
            let out_width = out_width_isize as usize;

            channels
                .checked_mul(out_height)
                .and_then(|v| v.checked_mul(out_width))
                .ok_or_else(|| invalid_data("Pooling output size overflow"))
        }
        "batchnorm" | "dropout" => layer
            .size
            .ok_or_else(|| invalid_data(format!("{} layer missing size", layer.layer_type))),
        _ => Err(invalid_data(format!(
            "Unknown layer type: {}",
            layer.layer_type
        ))),
    }
}

fn validate_layer_connection(
    current: &LayerConfig,
    next: &LayerConfig,
    index: usize,
) -> Result<(), Box<dyn Error>> {
    // If both layers declare full spatial shapes, validate H/W/C compatibility.
    // Otherwise fall back to the scalar size check.
    if let (Some(curr_h), Some(curr_w), Some(curr_c)) = (
        get_layer_output_height(current),
        get_layer_output_width(current),
        get_layer_output_channels(current),
    ) {
        if let (Some(next_h), Some(next_w), Some(next_c)) = (
            get_layer_input_height(next),
            get_layer_input_width(next),
            get_layer_input_channels(next),
        ) {
            if curr_c != next_c {
                return Err(invalid_data(format!(
                    "Layer connection mismatch: Layer {} output channels ({}) does not match Layer {} input channels ({})",
                    index,
                    curr_c,
                    index + 1,
                    next_c
                )));
            }
            if curr_h != next_h {
                return Err(invalid_data(format!(
                    "Layer connection mismatch: Layer {} output height ({}) does not match Layer {} input height ({})",
                    index,
                    curr_h,
                    index + 1,
                    next_h
                )));
            }
            if curr_w != next_w {
                return Err(invalid_data(format!(
                    "Layer connection mismatch: Layer {} output width ({}) does not match Layer {} input width ({})",
                    index,
                    curr_w,
                    index + 1,
                    next_w
                )));
            }

            return Ok(());
        }
    }

    let current_output = get_layer_output_size(current)?;
    let next_input = get_layer_input_size(next)?;

    if current_output != next_input {
        return Err(invalid_data(format!(
            "Layer connection mismatch: Layer {} output size ({}) does not match Layer {} input size ({})",
            index,
            current_output,
            index + 1,
            next_input
        )));
    }

    Ok(())
}

fn get_layer_input_height(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => layer.input_height,
        "maxpool" | "avgpool" | "pool" => layer.pool_input_height,
        _ => None,
    }
}

fn get_layer_input_width(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => layer.input_width,
        "maxpool" | "avgpool" | "pool" => layer.pool_input_width,
        _ => None,
    }
}

fn get_layer_input_channels(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => layer.in_channels,
        "maxpool" | "avgpool" | "pool" => layer.pool_channels,
        _ => None,
    }
}

fn get_layer_output_height(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => {
            let input_height = layer.input_height?;
            let kernel_size = layer.kernel_size?;
            let padding = layer.padding.unwrap_or(0);
            let stride = layer.stride.unwrap_or(1);

            let out_height_isize =
                (input_height as isize + 2 * padding - kernel_size as isize) / stride as isize + 1;
            (out_height_isize >= 0).then_some(out_height_isize as usize)
        }
        "maxpool" | "avgpool" | "pool" => {
            let input_height = layer.pool_input_height?;
            let pool_size = layer.pool_size?;
            let pool_padding = layer.pool_padding.unwrap_or(0);
            let pool_stride = layer.pool_stride.unwrap_or(pool_size);

            let out_height_isize = (input_height as isize + 2 * pool_padding - pool_size as isize)
                / pool_stride as isize
                + 1;
            (out_height_isize >= 0).then_some(out_height_isize as usize)
        }
        _ => None,
    }
}

fn get_layer_output_width(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => {
            let input_width = layer.input_width?;
            let kernel_size = layer.kernel_size?;
            let padding = layer.padding.unwrap_or(0);
            let stride = layer.stride.unwrap_or(1);

            let out_width_isize =
                (input_width as isize + 2 * padding - kernel_size as isize) / stride as isize + 1;
            (out_width_isize >= 0).then_some(out_width_isize as usize)
        }
        "maxpool" | "avgpool" | "pool" => {
            let input_width = layer.pool_input_width?;
            let pool_size = layer.pool_size?;
            let pool_padding = layer.pool_padding.unwrap_or(0);
            let pool_stride = layer.pool_stride.unwrap_or(pool_size);

            let out_width_isize = (input_width as isize + 2 * pool_padding - pool_size as isize)
                / pool_stride as isize
                + 1;
            (out_width_isize >= 0).then_some(out_width_isize as usize)
        }
        _ => None,
    }
}

fn get_layer_output_channels(layer: &LayerConfig) -> Option<usize> {
    match layer.layer_type.to_lowercase().as_str() {
        "conv2d" => layer.out_channels,
        "maxpool" | "avgpool" | "pool" => layer.pool_channels,
        _ => None,
    }
}

/// Validates an architecture configuration for required fields, parameter ranges, and adjacent layer compatibility.
///
/// Ensures the config contains at least one layer, that each layer provides the fields required by its type and that numeric parameters lie in valid ranges, and that the output size of each layer matches the input size of the next layer.
///
/// # Errors
///
/// Returns an `InvalidData`-style error with a descriptive message if the configuration is empty, a layer is missing required fields or has invalid parameter values, or if two adjacent layers have incompatible sizes.
///
/// # Examples
///
/// ```ignore
/// // Construct a valid ArchitectureConfig with compatible layers, then validate it.
/// // (Replace the following placeholder with a real ArchitectureConfig value.)
/// # use super::super::config::{ArchitectureConfig};
/// # let config = ArchitectureConfig { layers: Vec::new() }; // placeholder
/// let _ = crate::architecture::validation::validate_architecture(&config);
/// ```
pub(super) fn validate_architecture(config: &ArchitectureConfig) -> Result<(), Box<dyn Error>> {
    if config.layers.is_empty() {
        return Err(invalid_data("Architecture must have at least one layer"));
    }

    for (i, layer) in config.layers.iter().enumerate() {
        validate_layer(layer, i)?;
    }

    for i in 0..config.layers.len() - 1 {
        validate_layer_connection(&config.layers[i], &config.layers[i + 1], i)?;
    }

    Ok(())
}

/// Validate a single layer configuration and return an error describing any missing or invalid fields.
///
/// This checks that the layer contains all fields required for its declared type and that numeric
/// parameters fall within the allowed ranges (e.g., sizes > 0, momentum in [0.0, 1.0], drop_rate in
/// [0.0, 1.0), convolution geometry constraints, etc.). Error messages include the provided `index`
/// to identify the layer in higher-level configurations.
///
/// # Parameters
///
/// - `layer`: The layer configuration to validate.
/// - `index`: The index of the layer within the architecture; included in error messages.
///
/// # Returns
///
/// `Ok(())` if the layer is valid; otherwise `Err` with an `InvalidData`-style message describing the
/// missing or invalid field.
///
/// # Examples
///
/// ```ignore
/// use super::config::LayerConfig;
/// use super::validation::validate_layer;
///
/// let dense = LayerConfig {
///     layer_type: "dense".to_string(),
///     input_size: Some(4),
///     output_size: Some(2),
///     in_channels: None,
///     out_channels: None,
///     kernel_size: None,
///     input_height: None,
///     input_width: None,
///     padding: None,
///     stride: None,
///     size: None,
///     epsilon: None,
///     momentum: None,
///     drop_rate: None,
/// };
///
/// assert!(validate_layer(&dense, 0).is_ok());
/// ```
fn validate_layer(layer: &LayerConfig, index: usize) -> Result<(), Box<dyn Error>> {
    let require = |cond: bool, msg: &str| -> Result<(), Box<dyn Error>> {
        if !cond {
            Err(invalid_data(format!("Layer {}: {}", index, msg)))
        } else {
            Ok(())
        }
    };

    match layer.layer_type.to_lowercase().as_str() {
        "dense" => {
            require(
                layer.input_size.is_some(),
                "Dense layer requires 'input_size'",
            )?;
            require(
                layer.output_size.is_some(),
                "Dense layer requires 'output_size'",
            )?;
            require(
                layer.input_size.unwrap() > 0,
                "input_size must be greater than 0",
            )?;
            require(
                layer.output_size.unwrap() > 0,
                "output_size must be greater than 0",
            )?;
        }
        "conv2d" => {
            require(
                layer.in_channels.is_some(),
                "Conv2D layer requires 'in_channels'",
            )?;
            require(
                layer.out_channels.is_some(),
                "Conv2D layer requires 'out_channels'",
            )?;
            require(
                layer.kernel_size.is_some(),
                "Conv2D layer requires 'kernel_size'",
            )?;
            require(
                layer.input_height.is_some(),
                "Conv2D layer requires 'input_height'",
            )?;
            require(
                layer.input_width.is_some(),
                "Conv2D layer requires 'input_width'",
            )?;
            require(
                layer.in_channels.unwrap() > 0,
                "in_channels must be greater than 0",
            )?;
            require(
                layer.out_channels.unwrap() > 0,
                "out_channels must be greater than 0",
            )?;
            require(
                layer.kernel_size.unwrap() > 0,
                "kernel_size must be greater than 0",
            )?;
            require(
                layer.stride.unwrap_or(1) > 0,
                "stride must be greater than 0",
            )?;
            require(
                layer.input_height.unwrap() > 0,
                "input_height must be greater than 0",
            )?;
            require(
                layer.input_width.unwrap() > 0,
                "input_width must be greater than 0",
            )?;

            let padding = layer.padding.unwrap_or(0);
            require(
                padding >= 0,
                "invalid Conv2D configuration: padding must be >= 0",
            )?;
            let kernel_size = layer.kernel_size.unwrap();
            let h_num = layer.input_height.unwrap() as isize + 2 * padding - kernel_size as isize;
            require(
                h_num >= 0,
                "invalid Conv2D configuration: input_height + 2*padding - kernel_size must be >= 0",
            )?;
            let w_num = layer.input_width.unwrap() as isize + 2 * padding - kernel_size as isize;
            require(
                w_num >= 0,
                "invalid Conv2D configuration: input_width + 2*padding - kernel_size must be >= 0",
            )?;
        }
        "batchnorm" => {
            require(layer.size.is_some(), "BatchNorm layer requires 'size'")?;
            require(layer.size.unwrap() > 0, "size must be greater than 0")?;
            if let Some(epsilon) = layer.epsilon {
                require(epsilon > 0.0, "epsilon must be positive")?;
            }
            if let Some(momentum) = layer.momentum {
                require(
                    (0.0..=1.0).contains(&momentum),
                    "momentum must be in range [0.0, 1.0]",
                )?;
            }
        }
        "dropout" => {
            require(layer.size.is_some(), "Dropout layer requires 'size'")?;
            require(layer.size.unwrap() > 0, "size must be greater than 0")?;
            require(
                layer.drop_rate.is_some(),
                "Dropout layer requires 'drop_rate'",
            )?;
            let rate = layer.drop_rate.unwrap();
            require(
                (0.0..1.0).contains(&rate),
                "drop_rate must be in range [0.0, 1.0)",
            )?;
        }
        "maxpool" | "avgpool" | "pool" => {
            require(
                layer.pool_size.is_some(),
                "Pooling layer requires 'pool_size'",
            )?;
            require(
                layer.pool_input_height.is_some(),
                "Pooling layer requires 'pool_input_height'",
            )?;
            require(
                layer.pool_input_width.is_some(),
                "Pooling layer requires 'pool_input_width'",
            )?;
            require(
                layer.pool_channels.is_some(),
                "Pooling layer requires 'pool_channels'",
            )?;

            require(
                layer.pool_size.unwrap() > 0,
                "pool_size must be greater than 0",
            )?;
            require(
                layer.pool_stride.unwrap_or(layer.pool_size.unwrap()) > 0,
                "pool_stride must be greater than 0",
            )?;
            require(
                layer.pool_input_height.unwrap() > 0,
                "pool_input_height must be greater than 0",
            )?;
            require(
                layer.pool_input_width.unwrap() > 0,
                "pool_input_width must be greater than 0",
            )?;
            require(
                layer.pool_channels.unwrap() > 0,
                "pool_channels must be greater than 0",
            )?;

            let padding = layer.pool_padding.unwrap_or(0);
            require(
                padding >= 0,
                "invalid pooling configuration: pool_padding must be >= 0",
            )?;

            let pool_size = layer.pool_size.unwrap();
            let h_num =
                layer.pool_input_height.unwrap() as isize + 2 * padding - pool_size as isize;
            require(
                h_num >= 0,
                "invalid pooling configuration: pool_input_height + 2*pool_padding - pool_size must be >= 0",
            )?;
            let w_num = layer.pool_input_width.unwrap() as isize + 2 * padding - pool_size as isize;
            require(
                w_num >= 0,
                "invalid pooling configuration: pool_input_width + 2*pool_padding - pool_size must be >= 0",
            )?;

            if layer.layer_type.to_lowercase() == "pool" {
                require(
                    layer.pool_mode.is_some(),
                    "Pooling layer with layer_type='pool' requires 'pool_mode'",
                )?;
                let mode = layer.pool_mode.as_ref().unwrap().to_lowercase();
                require(
                    mode == "max" || mode == "avg",
                    "pool_mode must be either 'max' or 'avg'",
                )?;
            }
        }
        _ => {
            return Err(invalid_data(format!(
                "Layer {}: Invalid layer type '{}'. Must be one of: dense, conv2d, batchnorm, dropout, maxpool, avgpool, pool",
                index, layer.layer_type
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(input_size: usize, output_size: usize) -> LayerConfig {
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: Some(input_size),
            output_size: Some(output_size),
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_dense_layer() {
        assert!(validate_layer(&dense(784, 256), 0).is_ok());
    }

    #[test]
    fn test_validate_dense_layer_missing_fields() {
        let layer = LayerConfig {
            layer_type: "dense".to_string(),
            output_size: Some(256),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_conv2d_layer_rejects_negative_padding() {
        let layer = LayerConfig {
            layer_type: "conv2d".to_string(),
            input_size: None,
            output_size: None,
            in_channels: Some(1),
            out_channels: Some(8),
            kernel_size: Some(3),
            padding: Some(-1),
            stride: Some(1),
            input_height: Some(5),
            input_width: Some(5),
            size: None,
            epsilon: None,
            momentum: None,
            drop_rate: None,
            ..Default::default()
        };

        let err = validate_layer(&layer, 0).unwrap_err();
        assert!(err.to_string().contains("padding must be >= 0"));
    }

    #[test]
    fn test_get_layer_input_size_rejects_conv2d_overflow() {
        let layer = LayerConfig {
            layer_type: "conv2d".to_string(),
            in_channels: Some(usize::MAX),
            input_height: Some(2),
            input_width: Some(2),
            ..Default::default()
        };

        let err = get_layer_input_size(&layer).unwrap_err();
        assert!(err.to_string().contains("Conv2D input size overflow"));
    }

    #[test]
    fn test_get_layer_output_size_rejects_conv2d_overflow() {
        let layer = LayerConfig {
            layer_type: "conv2d".to_string(),
            out_channels: Some(usize::MAX),
            kernel_size: Some(1),
            input_height: Some(2),
            input_width: Some(2),
            stride: Some(1),
            padding: Some(0),
            ..Default::default()
        };

        let err = get_layer_output_size(&layer).unwrap_err();
        assert!(err.to_string().contains("Conv2D output size overflow"));
    }

    #[test]
    fn test_get_layer_output_size_rejects_negative_conv2d_height() {
        let layer = LayerConfig {
            layer_type: "conv2d".to_string(),
            out_channels: Some(8),
            kernel_size: Some(5),
            input_height: Some(2),
            input_width: Some(5),
            stride: Some(1),
            padding: Some(0),
            ..Default::default()
        };

        let err = get_layer_output_size(&layer).unwrap_err();
        assert!(err.to_string().contains("output height is negative"));
    }

    #[test]
    fn test_validate_invalid_layer_type() {
        let layer = LayerConfig {
            layer_type: "invalid".to_string(),
            ..Default::default()
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
            size: Some(256),
            drop_rate: Some(0.5),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_ok());
    }

    #[test]
    fn test_validate_dropout_layer_missing_rate() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: Some(256),
            drop_rate: None,
            ..Default::default()
        };
        let err = validate_layer(&layer, 0).unwrap_err();
        assert!(err.to_string().contains("drop_rate"));
    }

    #[test]
    fn test_validate_dropout_layer_invalid_rate() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: Some(256),
            drop_rate: Some(1.5),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_dropout_layer_rate_zero_is_valid() {
        // 0.0 is in the valid range [0.0, 1.0)
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: Some(128),
            drop_rate: Some(0.0),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_ok());
    }

    #[test]
    fn test_validate_dropout_layer_rate_one_is_invalid() {
        // 1.0 is NOT in [0.0, 1.0) - the range is exclusive of 1.0
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: Some(128),
            drop_rate: Some(1.0),
            ..Default::default()
        };
        let err = validate_layer(&layer, 0).unwrap_err();
        assert!(err.to_string().contains("drop_rate"));
    }

    #[test]
    fn test_validate_dropout_layer_negative_rate_is_invalid() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: Some(128),
            drop_rate: Some(-0.1),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_dropout_layer_missing_size_is_invalid() {
        let layer = LayerConfig {
            layer_type: "dropout".to_string(),
            size: None,
            drop_rate: Some(0.5),
            ..Default::default()
        };
        assert!(validate_layer(&layer, 0).is_err());
    }

    #[test]
    fn test_validate_architecture_with_dropout_layer() {
        // Dropout layer as part of a valid architecture chain
        let config = ArchitectureConfig {
            layers: vec![
                dense(784, 256),
                LayerConfig {
                    layer_type: "dropout".to_string(),
                    size: Some(256),
                    drop_rate: Some(0.5),
                    ..Default::default()
                },
                dense(256, 10),
            ],
        };
        assert!(validate_architecture(&config).is_ok());
    }

    #[test]
    fn test_validate_architecture_dropout_without_drop_rate_fails() {
        // Dropout layer missing drop_rate in architecture context
        let config = ArchitectureConfig {
            layers: vec![
                dense(784, 256),
                LayerConfig {
                    layer_type: "dropout".to_string(),
                    size: Some(256),
                    drop_rate: None,
                    ..Default::default()
                },
            ],
        };
        assert!(validate_architecture(&config).is_err());
    }

    #[test]
    fn test_validate_layer_connection_mismatch() {
        let config = ArchitectureConfig {
            layers: vec![dense(784, 256), dense(128, 10)],
        };
        let result = validate_architecture(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Layer connection mismatch"));
    }

    #[test]
    fn test_validate_layer_connection_valid() {
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
                dense(256, 10),
            ],
        };
        assert!(validate_architecture(&config).is_ok());
    }

    #[test]
    fn test_validation() {
        let valid_config = ArchitectureConfig {
            layers: vec![dense(784, 128), dense(128, 10)],
        };
        assert!(validate_architecture(&valid_config).is_ok());

        let empty_config = ArchitectureConfig { layers: vec![] };
        assert!(validate_architecture(&empty_config).is_err());

        let invalid_type_config = ArchitectureConfig {
            layers: vec![LayerConfig {
                layer_type: "invalid".to_string(),
                ..Default::default()
            }],
        };
        assert!(validate_architecture(&invalid_type_config).is_err());

        let mismatch_config = ArchitectureConfig {
            layers: vec![dense(784, 256), dense(128, 10)],
        };
        assert!(validate_architecture(&mismatch_config).is_err());
    }
}
