//! 2D Convolutional layer implementation
//!
//! This module provides a Conv2DLayer that performs 2D convolution operations,
//! commonly used in computer vision tasks like image classification.
//! Inputs are expected in channels-last (pixel-interleaved) order.

use crate::layers::gradient::GradientAccumulator;
use crate::utils::rng::SimpleRng;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::gpu::backend::GpuBackend;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use std::sync::Arc;

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
/// use rust_neural_networks::layers::conv2d::Conv2DLayer;
/// use rust_neural_networks::utils::rng::SimpleRng;
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
    weights: Vec<f32>, // [out_channels * in_channels * kernel_size * kernel_size]
    biases: Vec<f32>,  // [out_channels]
    // Gradient accumulators
    grad_weights: GradientAccumulator,
    grad_biases: GradientAccumulator,
    // Optional GPU backend for accelerated computation
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    gpu_backend: Option<Arc<dyn GpuBackend>>,
}

impl Conv2DLayer {
    /// Creates a Conv2DLayer initialized with Xavier (Glorot) weights and zero biases.
    ///
    /// Weights are sampled uniformly from [-limit, limit] where
    /// limit = sqrt(6 / (fan_in + fan_out)) and, for convolutions,
    /// fan_in = in_channels × kernel_size², fan_out = out_channels × kernel_size².
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// // 1 input channel, 8 output channels, 3x3 kernel, padding=1, stride=1, 28x28 input
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// assert_eq!(layer.in_channels(), 1);
    /// assert_eq!(layer.out_channels(), 8);
    /// assert_eq!(layer.kernel_size(), 3);
    /// ```
    #[allow(clippy::too_many_arguments)]
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
        assert!(stride > 0, "Stride must be greater than 0");

        // Validate output dimensions to prevent underflow/invalid configuration
        let h_num = input_height as isize + 2 * padding - kernel_size as isize;
        let w_num = input_width as isize + 2 * padding - kernel_size as isize;

        if h_num < 0 {
            panic!(
                "Invalid Conv2D configuration: Output height would be negative. \
                 Input H: {}, Kernel: {}, Padding: {}, Stride: {}. \
                 Formula: Output = (input + 2*padding - kernel) / stride + 1 = ({} + {} - {}) / {} + 1. \
                 Fix: increase padding or reduce kernel size.",
                input_height,
                kernel_size,
                padding,
                stride,
                input_height,
                2 * padding,
                kernel_size,
                stride
            );
        }
        if w_num < 0 {
            panic!(
                "Invalid Conv2D configuration: Output width would be negative. \
                 Input W: {}, Kernel: {}, Padding: {}, Stride: {}. \
                 Formula: Output = (input + 2*padding - kernel) / stride + 1 = ({} + {} - {}) / {} + 1. \
                 Fix: increase padding or reduce kernel size.",
                input_width,
                kernel_size,
                padding,
                stride,
                input_width,
                2 * padding,
                kernel_size,
                stride
            );
        }

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
            grad_weights: GradientAccumulator::new(weight_count),
            grad_biases: GradientAccumulator::new(out_channels),
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            gpu_backend: None,
        }
    }

    /// Number of input channels.
    ///
    /// # Returns
    ///
    /// `usize` number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels (filters).
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Provides the size (side length) of the square convolution kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// assert_eq!(layer.kernel_size(), 3);
    /// ```
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the padding amount.
    pub fn padding(&self) -> isize {
        self.padding
    }

    /// Number of input pixels the kernel moves between consecutive applications.
    ///
    /// # Returns
    ///
    /// The stride value (step size in pixels) used when sliding the convolutional kernel.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Computes the output height of the layer's feature maps after applying the convolution.
    ///
    /// The result is floor((input_height + 2*padding - kernel_size) / stride) + 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = Conv2DLayer::new(
    ///     1, // in_channels
    ///     8, // out_channels
    ///     3, // kernel_size
    ///     1, // padding
    ///     1, // stride
    ///     28, // input_height
    ///     28, // input_width
    ///     &mut rng,
    /// );
    /// assert_eq!(layer.output_height(), 28);
    /// ```
    pub fn output_height(&self) -> usize {
        ((self.input_height as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize
            + 1) as usize
    }

    /// Computes the spatial width of the output feature map produced by this layer.
    ///
    /// The result is computed from the layer's input width, padding, kernel size, and stride:
    /// (input_width + 2*padding - kernel_size) / stride + 1.
    ///
    /// # Examples
    ///
    /// ```
    /// // Construct a layer with input width 28, kernel 3, padding 1 and stride 1.
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// assert_eq!(layer.output_width(), 28);
    /// ```
    pub fn output_width(&self) -> usize {
        ((self.input_width as isize + 2 * self.padding - self.kernel_size as isize)
            / self.stride as isize
            + 1) as usize
    }

    /// Returns the input height configured for this convolution layer.
    pub fn input_height(&self) -> usize {
        self.input_height
    }

    /// Returns the input width configured for this convolution layer.
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// Immutable view of the convolution filter weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Immutable view of the convolution bias values.
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    fn validate_parameter_shapes(&self, context: &str) {
        let expected_weights =
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;
        assert_eq!(
            self.weights.len(),
            expected_weights,
            "{}: weights length mismatch, expected {}, got {}",
            context,
            expected_weights,
            self.weights.len()
        );
        assert_eq!(
            self.biases.len(),
            self.out_channels,
            "{}: biases length mismatch, expected {}, got {}",
            context,
            self.out_channels,
            self.biases.len()
        );
    }

    /// Total number of trainable parameters in the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(0);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// assert_eq!(layer.parameter_count(), 1 * 8 * 3 * 3 + 8);
    /// ```
    ///
    /// # Returns
    ///
    /// The total number of trainable parameters (weights + biases).
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Computes the L2 norm of the layer's accumulated weight and bias gradients.
    ///
    /// # Returns
    ///
    /// A tuple `(weight_grad_norm, bias_grad_norm)` where `weight_grad_norm` is the L2
    /// norm of the weight gradients and `bias_grad_norm` is the L2 norm of the bias gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);
    ///
    /// // after a forward/backward step gradients may be non-zero
    /// let input = vec![1.0; layer.input_size()];
    /// let mut output = vec![0.0; layer.output_size()];
    /// layer.forward(&input, &mut output, 1);
    /// let grad_output = vec![0.1; layer.output_size()];
    /// let mut grad_input = vec![0.0; layer.input_size()];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// let (w_norm, b_norm) = layer.get_gradient_magnitude();
    /// assert!(w_norm >= 0.0);
    /// assert!(b_norm >= 0.0);
    /// ```
    pub fn get_gradient_magnitude(&self) -> (f32, f32) {
        (self.grad_weights.l2_norm(), self.grad_biases.l2_norm())
    }

    /// Create a Conv2DLayer using the provided weights and biases.
    ///
    /// Intended for reconstructing a layer from saved parameters; gradient accumulators
    /// are initialized to zero and ready for the next training pass.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len()` != `out_channels * in_channels * kernel_size * kernel_size`
    /// or if `biases.len()` != `out_channels`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    ///
    /// let in_channels = 1;
    /// let out_channels = 2;
    /// let kernel_size = 3;
    /// let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    /// let weights = vec![0.1f32; weight_count];
    /// let biases = vec![0.0f32; out_channels];
    /// let layer = Conv2DLayer::new_with_weights(
    ///     in_channels, out_channels, kernel_size, 1, 1, 28, 28, weights.clone(), biases.clone()
    /// );
    /// assert_eq!(layer.weights(), weights.as_slice());
    /// assert_eq!(layer.biases(), biases.as_slice());
    /// assert_eq!(layer.in_channels(), 1);
    /// assert_eq!(layer.out_channels(), 2);
    /// ```
    pub fn new_with_weights(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: isize,
        stride: usize,
        input_height: usize,
        input_width: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) -> Self {
        let expected_weight_count = out_channels * in_channels * kernel_size * kernel_size;
        assert_eq!(
            weights.len(),
            expected_weight_count,
            "weights length {} does not match out_channels * in_channels * kernel_size * kernel_size = {}",
            weights.len(),
            expected_weight_count
        );
        assert_eq!(
            biases.len(),
            out_channels,
            "biases length {} does not match out_channels = {}",
            biases.len(),
            out_channels
        );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            input_height,
            input_width,
            grad_weights: GradientAccumulator::new(weights.len()),
            grad_biases: GradientAccumulator::new(biases.len()),
            weights,
            biases,
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            gpu_backend: None,
        }
    }
}

mod cpu;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
mod gpu;

#[cfg(test)]
mod tests;
