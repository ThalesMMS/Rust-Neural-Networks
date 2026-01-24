//! 2D Convolutional layer implementation
//!
//! This module provides a Conv2DLayer that performs 2D convolution operations,
//! commonly used in computer vision tasks like image classification.

use crate::layers::Layer;
use crate::utils::SimpleRng;
use std::cell::RefCell;

/// 2D Convolutional layer with learnable filters.
///
/// Performs 2D convolution: slides filters over input to produce feature maps.
/// Supports zero-padding and configurable stride.
///
/// # Fields
///
/// * `in_channels` - Number of input channels (e.g., 1 for grayscale, 3 for RGB)
/// * `out_channels` - Number of output feature maps (number of filters)
/// * `kernel_size` - Size of the convolutional kernel (assumed square: kernel_size × kernel_size)
/// * `padding` - Zero-padding applied to input (symmetric on all sides)
/// * `stride` - Stride for the convolution operation
/// * `input_height` - Height of input feature map
/// * `input_width` - Width of input feature map
/// * `weights` - Convolutional filters (out_channels × in_channels × kernel_size × kernel_size)
/// * `biases` - Bias for each output channel (out_channels)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::Conv2DLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// // 1 input channel (grayscale), 8 output channels, 3x3 kernel, padding=1
/// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
/// assert_eq!(layer.out_channels(), 8);
/// ```
pub struct Conv2DLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: isize,
    stride: usize,
    input_height: usize,
    input_width: usize,
    weights: Vec<f32>,      // [out_channels * in_channels * kernel_size * kernel_size]
    biases: Vec<f32>,       // [out_channels]
    // Gradient accumulators (mutable interior via RefCell for trait compatibility)
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}

impl Conv2DLayer {
    /// Create a new Conv2DLayer with Xavier initialization.
    ///
    /// Weights are initialized using Xavier/Glorot initialization adapted for convolutions:
    /// randomly sampled from uniform distribution [-limit, limit]
    /// where limit = sqrt(6 / (fan_in + fan_out)).
    /// For convolutions: fan_in = in_channels × kernel_size², fan_out = out_channels × kernel_size²
    ///
    /// Biases are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output feature maps (filters)
    /// * `kernel_size` - Size of square kernel (e.g., 3 for 3×3)
    /// * `padding` - Zero-padding to apply
    /// * `stride` - Stride for convolution
    /// * `input_height` - Height of input feature map
    /// * `input_width` - Width of input feature map
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// A new Conv2DLayer with randomly initialized weights and zero biases
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// // 1 input channel, 8 output channels, 3x3 kernel, padding=1, stride=1, 28x28 input
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: isize,
        stride: usize,
        input_height: usize,
        input_width: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        // Xavier initialization for convolutional layers
        // fan_in = in_channels * kernel_size * kernel_size
        // fan_out = out_channels * kernel_size * kernel_size
        let fan_in = (in_channels * kernel_size * kernel_size) as f32;
        let fan_out = (out_channels * kernel_size * kernel_size) as f32;
        let limit = (6.0f32 / (fan_in + fan_out)).sqrt();

        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let mut weights = vec![0.0f32; weight_count];

        for value in &mut weights {
            *value = rng.gen_range_f32(-limit, limit);
        }

        Self {
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            input_height,
            input_width,
            weights,
            biases: vec![0.0f32; out_channels],
            grad_weights: RefCell::new(vec![0.0f32; weight_count]),
            grad_biases: RefCell::new(vec![0.0f32; out_channels]),
        }
    }

    /// Get the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels (filters).
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the padding amount.
    pub fn padding(&self) -> isize {
        self.padding
    }

    /// Get the stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get the output height after convolution.
    ///
    /// Calculated as: (input_height + 2*padding - kernel_size) / stride + 1
    pub fn output_height(&self) -> usize {
        ((self.input_height as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize + 1) as usize
    }

    /// Get the output width after convolution.
    ///
    /// Calculated as: (input_width + 2*padding - kernel_size) / stride + 1
    pub fn output_width(&self) -> usize {
        ((self.input_width as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize + 1) as usize
    }

    /// Get the total number of trainable parameters.
    ///
    /// Returns weights count + biases count
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_initialization() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

        assert_eq!(layer.in_channels(), 1);
        assert_eq!(layer.out_channels(), 8);
        assert_eq!(layer.kernel_size(), 3);
        assert_eq!(layer.padding(), 1);
        assert_eq!(layer.stride(), 1);
    }

    #[test]
    fn test_conv2d_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

        // weights: 8 * 1 * 3 * 3 = 72
        // biases: 8
        // total: 80
        assert_eq!(layer.parameter_count(), 80);
    }

    #[test]
    fn test_conv2d_output_dimensions() {
        let mut rng = SimpleRng::new(42);
        // With padding=1 and stride=1, 3x3 kernel maintains spatial dimensions
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

        assert_eq!(layer.output_height(), 28);
        assert_eq!(layer.output_width(), 28);
    }

    #[test]
    fn test_conv2d_output_dimensions_no_padding() {
        let mut rng = SimpleRng::new(42);
        // Without padding, 3x3 kernel reduces dimensions by 2 on each side
        let layer = Conv2DLayer::new(1, 8, 3, 0, 1, 28, 28, &mut rng);

        assert_eq!(layer.output_height(), 26); // 28 - 3 + 1 = 26
        assert_eq!(layer.output_width(), 26);
    }

    #[test]
    fn test_conv2d_xavier_initialization_bounds() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);

        // Xavier limit for this configuration
        let fan_in = (1 * 3 * 3) as f32;
        let fan_out = (8 * 3 * 3) as f32;
        let limit = (6.0f32 / (fan_in + fan_out)).sqrt();

        // All weights should be within [-limit, limit]
        for &weight in &layer.weights {
            assert!(
                weight >= -limit && weight <= limit,
                "Weight {} outside Xavier bounds [{}, {}]",
                weight,
                -limit,
                limit
            );
        }

        // All biases should be initialized to zero
        for &bias in &layer.biases {
            assert_eq!(bias, 0.0);
        }
    }

    #[test]
    fn test_conv2d_deterministic_initialization() {
        let mut rng1 = SimpleRng::new(12345);
        let layer1 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng1);

        let mut rng2 = SimpleRng::new(12345);
        let layer2 = Conv2DLayer::new(3, 16, 5, 2, 1, 32, 32, &mut rng2);

        // Same seed should produce identical weights
        assert_eq!(layer1.weights, layer2.weights);
        assert_eq!(layer1.biases, layer2.biases);
    }
}
