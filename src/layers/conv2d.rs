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

// Layer trait implementation

impl Layer for Conv2DLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let out_h = self.output_height();
        let out_w = self.output_width();
        let out_spatial = out_h * out_w;
        let in_spatial = self.input_height * self.input_width;

        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let out_base_b = b * (self.out_channels * out_spatial);

            for oc in 0..self.out_channels {
                let bias = self.biases[oc];
                let out_base = out_base_b + oc * out_spatial;

                // For each output pixel, accumulate over input channels and kernel window
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = bias;

                        // Accumulate over input channels
                        for ic in 0..self.in_channels {
                            let w_base = (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;
                            let in_base_c = in_base + ic * in_spatial;

                            // Convolve kernel over input
                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize - self.padding;

                                    if iy >= 0 && iy < self.input_height as isize
                                        && ix >= 0 && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base_c + iyy * self.input_width + ixx;
                                        let w_idx = w_base + ky * self.kernel_size + kx;
                                        sum += input[in_idx] * self.weights[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = out_base + oy * out_w + ox;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    fn backward(&self, input: &[f32], grad_output: &[f32], grad_input: &mut [f32], batch_size: usize) {
        let scale = 1.0f32 / batch_size as f32;
        let out_h = self.output_height();
        let out_w = self.output_width();
        let out_spatial = out_h * out_w;
        let in_spatial = self.input_height * self.input_width;

        // Clear gradient accumulators
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        // Accumulate gradients for weights and biases
        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let g_base_b = b * (self.out_channels * out_spatial);

            for oc in 0..self.out_channels {
                let g_base = g_base_b + oc * out_spatial;

                // Accumulate bias gradient
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let g = grad_output[g_base + oy * out_w + ox];
                        grad_b[oc] += g;
                    }
                }

                // Accumulate weight gradients
                for ic in 0..self.in_channels {
                    let w_base = (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;
                    let in_base_c = in_base + ic * in_spatial;

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let g = grad_output[g_base + oy * out_w + ox];

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize - self.padding;

                                    if iy >= 0 && iy < self.input_height as isize
                                        && ix >= 0 && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base_c + iyy * self.input_width + ixx;
                                        let w_idx = w_base + ky * self.kernel_size + kx;
                                        grad_w[w_idx] += g * input[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Scale gradients by batch size
        for g in grad_w.iter_mut() {
            *g *= scale;
        }
        for g in grad_b.iter_mut() {
            *g *= scale;
        }

        // Compute gradient with respect to input
        for v in grad_input.iter_mut() {
            *v = 0.0;
        }

        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let g_base_b = b * (self.out_channels * out_spatial);

            for ic in 0..self.in_channels {
                let in_base_c = in_base + ic * in_spatial;

                for oc in 0..self.out_channels {
                    let g_base = g_base_b + oc * out_spatial;
                    let w_base = (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let g = grad_output[g_base + oy * out_w + ox];

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize - self.padding;

                                    if iy >= 0 && iy < self.input_height as isize
                                        && ix >= 0 && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base_c + iyy * self.input_width + ixx;
                                        let w_idx = w_base + ky * self.kernel_size + kx;
                                        grad_input[in_idx] += g * self.weights[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        let grad_w = self.grad_weights.borrow();
        let grad_b = self.grad_biases.borrow();

        // Update weights: weight = weight - learning_rate * gradient
        for (weight, &gradient) in self.weights.iter_mut().zip(grad_w.iter()) {
            *weight -= learning_rate * gradient;
        }

        // Update biases: bias = bias - learning_rate * gradient
        for (bias, &gradient) in self.biases.iter_mut().zip(grad_b.iter()) {
            *bias -= learning_rate * gradient;
        }

        // Clear gradients for next iteration
        drop(grad_w);
        drop(grad_b);
        self.grad_weights.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
        self.grad_biases.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
    }

    fn input_size(&self) -> usize {
        self.in_channels * self.input_height * self.input_width
    }

    fn output_size(&self) -> usize {
        self.out_channels * self.output_height() * self.output_width()
    }

    fn parameter_count(&self) -> usize {
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
        let fan_in = (3 * 3) as f32;
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
