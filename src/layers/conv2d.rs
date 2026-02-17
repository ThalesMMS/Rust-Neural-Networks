//! 2D Convolutional layer implementation
//!
//! This module provides a Conv2DLayer that performs 2D convolution operations,
//! commonly used in computer vision tasks like image classification.
//! Inputs are expected in channels-last (pixel-interleaved) order.

use crate::layers::gradient::GradientAccumulator;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;

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

    /// Computes the L2 norm (magnitude) of the layer's weight and bias gradients.
    ///
    /// This is useful for monitoring gradient flow during training and detecting
    /// vanishing or exploding gradients. The L2 norm is computed as sqrt(sum(g_i^2))
    /// for each gradient component.
    ///
    /// # Returns
    ///
    /// A tuple `(weight_grad_norm, bias_grad_norm)` where:
    /// - `weight_grad_norm` is the L2 norm of the weight gradients
    /// - `bias_grad_norm` is the L2 norm of the bias gradients
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);
    ///
    /// // Perform forward and backward pass to accumulate gradients
    /// let input = vec![1.0; layer.input_size()];
    /// let mut output = vec![0.0; layer.output_size()];
    /// layer.forward(&input, &mut output, 1);
    /// let grad_output = vec![0.1; layer.output_size()];
    /// let mut grad_input = vec![0.0; layer.input_size()];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// // Get gradient magnitudes
    /// let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
    /// assert!(weight_norm >= 0.0);
    /// assert!(bias_norm >= 0.0);
    /// ```
    pub fn get_gradient_magnitude(&self) -> (f32, f32) {
        (self.grad_weights.l2_norm(), self.grad_biases.l2_norm())
    }

    /// Creates a Conv2DLayer with pre-existing weights and biases (no random initialization).
    ///
    /// This constructor is used when loading a model from disk, allowing the layer to be
    /// reconstructed with previously trained parameters. Gradient accumulators are
    /// zero-initialized, ready for the next training pass.
    ///
    /// # Panics
    ///
    /// Panics if the length of `weights` does not equal
    /// `out_channels * in_channels * kernel_size * kernel_size` or
    /// the length of `biases` does not equal `out_channels`.
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
    #[allow(clippy::too_many_arguments)]
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
        }
    }
}

// Layer trait implementation

impl Layer for Conv2DLayer {
    /// Applies this convolutional layer to `input` and writes the computed feature maps into `output`.
    ///
    /// # Forward Pass Formula
    ///
    /// The 2D convolution operation computes output feature maps by sliding filters over the input:
    /// ```text
    /// y[b,i,j,c_out] = Σ(x[b, i*s + di - p, j*s + dj - p, c_in] × W[c_out, c_in, di, dj]) + b[c_out]
    /// ```
    ///
    /// where the sum is over all kernel positions (di ∈ [0, K), dj ∈ [0, K)) and input channels (c_in ∈ [0, C_in)).
    ///
    /// where:
    /// - `x` is the input tensor (batch_size × input_height × input_width × in_channels)
    /// - `W` is the filter weights (out_channels × in_channels × kernel_size × kernel_size)
    /// - `b` is the bias vector (out_channels)
    /// - `y` is the output tensor (batch_size × output_height × output_width × out_channels)
    /// - `s` is the stride (step size)
    /// - `p` is the padding (zero-padding applied to input borders)
    /// - `K` is the kernel_size (assumed square)
    ///
    /// # Dimension Transformations
    ///
    /// **Input dimensions:**
    /// - Input: (batch_size × input_height × input_width × in_channels)
    /// - Memory layout: channels-last (pixel-interleaved RGBRGB...)
    ///
    /// **Output dimensions:**
    /// - Output height: floor((input_height + 2×padding - kernel_size) / stride) + 1
    /// - Output width: floor((input_width + 2×padding - kernel_size) / stride) + 1
    /// - Output: (batch_size × output_height × output_width × out_channels)
    ///
    /// **Convolution kernel:**
    /// - Weights: (out_channels × in_channels × kernel_size × kernel_size)
    /// - For each output position (i, j), the kernel slides over a (kernel_size × kernel_size) window
    ///   of the input, computing a weighted sum across all input channels
    ///
    /// # Implementation Details
    ///
    /// This method uses nested loops to implement the convolution:
    /// 1. Iterate over batch samples
    /// 2. Iterate over output channels (filters)
    /// 3. Iterate over spatial output positions (i, j)
    /// 4. For each output position, accumulate contributions from:
    ///    - All input channels (c_in)
    ///    - All kernel positions (di, dj)
    /// 5. Add bias for the output channel
    ///
    /// Zero-padding is applied by checking bounds during convolution - positions outside
    /// the input dimensions contribute 0 to the output sum.
    ///
    /// # Parameters
    ///
    /// - `input`: Flattened input tensor with layout [batch, input_height, input_width, in_channels].
    /// - `output`: Mutable flattened output buffer with layout [batch, out_channels, output_height, output_width].
    /// - `batch_size`: Number of examples in the batch.
    ///
    /// # Examples
    ///
    /// ```
    /// // Construct a layer (rng provided by the surrounding crate/test harness)
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// let batch_size = 2;
    /// let input = vec![0.0f32; batch_size * layer.input_size()];
    /// let mut output = vec![0.0f32; batch_size * layer.output_size()];
    /// layer.forward(&input, &mut output, batch_size);
    /// assert_eq!(output.len(), batch_size * layer.output_size());
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Calculate output spatial dimensions using the convolution formula:
        // out_dim = floor((in_dim + 2*padding - kernel_size) / stride) + 1
        let out_h = self.output_height();
        let out_w = self.output_width();

        // Dimension checks: verify input and output buffers have the expected sizes.
        // Input layout: (batch_size × input_height × input_width × in_channels) — channels-last
        // Output layout: (batch_size × out_channels × output_height × output_width)
        assert_eq!(
            input.len(),
            batch_size * self.input_size(),
            "Conv2DLayer forward: input shape mismatch. \
             Expected (batch={}, H={}, W={}, C_in={}) = {} elements, got {}. \
             Ensure input has batch_size * in_channels * input_height * input_width elements.",
            batch_size,
            self.input_height,
            self.input_width,
            self.in_channels,
            batch_size * self.input_size(),
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size(),
            "Conv2DLayer forward: output shape mismatch. \
             Expected (batch={}, C_out={}, H_out={}, W_out={}) = {} elements, got {}. \
             Ensure output buffer has batch_size * out_channels * output_height * output_width elements.",
            batch_size,
            self.out_channels,
            out_h,
            out_w,
            batch_size * self.output_size(),
            output.len()
        );
        let out_spatial = out_h * out_w; // Total spatial elements per output channel
        let in_spatial = self.input_height * self.input_width; // Total spatial elements per input channel

        // Process each sample in the batch
        for b in 0..batch_size {
            // Input base offset: (batch_index × in_channels × H × W)
            // Input dimensions: (batch_size × input_height × input_width × in_channels)
            let in_base = b * (self.in_channels * in_spatial);

            // Output base offset for this batch sample
            // Output dimensions: (batch_size × out_channels × output_height × output_width)
            let out_base_b = b * (self.out_channels * out_spatial);

            // Process each output channel (filter)
            for oc in 0..self.out_channels {
                let bias = self.biases[oc]; // Bias for this output channel
                let out_base = out_base_b + oc * out_spatial;

                // For each output spatial position (oy, ox), compute convolution result
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = bias; // Start with bias

                        // Accumulate contributions from all input channels
                        for ic in 0..self.in_channels {
                            // Weight index base for filter[oc, ic, :, :]
                            // Weights dimension: (out_channels × in_channels × kernel_size × kernel_size)
                            let w_base =
                                (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;

                            // Slide kernel window over input spatial positions
                            // For each kernel position (ky, kx), compute corresponding input position
                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    // Input position: output_pos × stride + kernel_offset - padding
                                    // This implements: i = oy*stride + ky - padding
                                    let iy = oy as isize * self.stride as isize + ky as isize
                                        - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize
                                        - self.padding;

                                    // Zero-padding: only accumulate if position is within input bounds
                                    if iy >= 0
                                        && iy < self.input_height as isize
                                        && ix >= 0
                                        && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;

                                        // Input index in channels-last format: [H, W, C]
                                        // Index: base + (row × width + col) × channels + channel_index
                                        let in_idx = in_base
                                            + (iyy * self.input_width + ixx) * self.in_channels
                                            + ic;

                                        // Weight index: [oc, ic, ky, kx]
                                        let w_idx = w_base + ky * self.kernel_size + kx;

                                        // Accumulate: sum += input[i,j,c_in] × weight[c_out,c_in,ky,kx]
                                        sum += input[in_idx] * self.weights[w_idx];
                                    }
                                }
                            }
                        }

                        // Write final convolution result: y[b, oy, ox, oc] = sum
                        let out_idx = out_base + oy * out_w + ox;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    /// Computes and accumulates gradients for this convolutional layer and writes the input gradients.
    ///
    /// This method updates the layer's internal gradient accumulators (`grad_weights` and `grad_biases`) by
    /// accumulating gradients from `grad_output` across the batch and spatial locations, and writes the gradient
    /// with respect to the layer input into `grad_input`.
    ///
    /// Parameters are expected in contiguous row-major layout with ordering (batch, height, width, channel).
    /// `input`/`grad_input` use NHWC (channels-last) layout, while `grad_output` follows the layer output
    /// layout [batch, out_channels, out_h, out_w].
    /// - `input`: length `batch_size * layer.input_size()`.
    /// - `grad_output`: length `batch_size * layer.output_size()`.
    /// - `grad_input`: mutable buffer with the same length and layout as `input`; it is overwritten with computed gradients.
    /// - `batch_size`: number of examples in the first dimension of `input` and `grad_output`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 1, 3, 1, 1, 5, 5, &mut rng);
    /// let batch = 1;
    /// let input = vec![0.0f32; batch * layer.input_size()];
    /// let grad_out = vec![1.0f32; batch * layer.output_size()];
    /// let mut grad_in = vec![0.0f32; batch * layer.input_size()];
    /// layer.backward(&input, &grad_out, &mut grad_in, batch);
    /// assert!(grad_in.iter().any(|&v| v != 0.0));
    /// ```
    /// # Backward Pass
    ///
    /// Computes gradients via the chain rule for backpropagation.
    ///
    /// Given gradient w.r.t. output: ∂L/∂y (batch_size × out_height × out_width × out_channels)
    ///
    /// **Step 1: Weight gradients**
    /// - ∂L/∂W[c_out, c_in, ky, kx] = Σ (∂L/∂y[b, oy, ox, c_out] × x[b, oy*s + ky - p, ox*s + kx - p, c_in])
    /// - Summation over all batch samples (b), output spatial positions (oy, ox)
    /// - Dimension check: For each filter position (c_out, c_in, ky, kx), correlate output gradient with corresponding input patch
    /// - Chain rule: Since y[b,i,j,c_out] = Σ(x[...] × W[c_out, c_in, di, dj]) + b[c_out], we have ∂y/∂W = x at corresponding positions
    ///
    /// **Step 2: Bias gradients**
    /// - ∂L/∂b[c_out] = Σ (∂L/∂y[b, oy, ox, c_out])
    /// - Summation over all batch samples (b) and output spatial positions (oy, ox)
    /// - Dimension check: Sum over (batch_size × out_height × out_width × out_channels) → (out_channels)
    /// - Chain rule: Since y = convolution + b, we have ∂y/∂b = 1 (for each output channel)
    ///
    /// **Step 3: Input gradients (for backprop to previous layer)**
    /// - ∂L/∂x[b, iy, ix, c_in] = Σ (∂L/∂y[b, oy, ox, c_out] × W[c_out, c_in, ky, kx])
    /// - Summation over all output channels (c_out) and kernel positions (ky, kx) where iy = oy*s + ky - p and ix = ox*s + kx - p
    /// - Dimension check: For each input position, accumulate gradients from all output positions that used it in the forward pass
    /// - Chain rule: Since y = convolution(x, W) + b, we have ∂y/∂x = W^T (transposed/rotated convolution)
    ///
    /// where:
    /// - `s` is the stride
    /// - `p` is the padding
    /// - `ky, kx` are kernel spatial indices ∈ [0, kernel_size)
    /// - `oy, ox` are output spatial indices ∈ [0, output_height), [0, output_width)
    /// - `iy, ix` are input spatial indices ∈ [0, input_height), [0, input_width)
    ///
    /// All gradients are reset at the start and accumulated across the batch to support
    /// mini-batch gradient descent.
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let out_h = self.output_height();
        let out_w = self.output_width();

        // Dimension checks: verify all buffers have the expected sizes before computing gradients.
        // Input layout: (batch_size × input_height × input_width × in_channels) — channels-last
        // grad_output layout: (batch_size × out_channels × output_height × output_width)
        // grad_input layout: same as input
        assert_eq!(
            input.len(),
            batch_size * self.input_size(),
            "Conv2DLayer backward: input shape mismatch. \
             Expected (batch={}, H={}, W={}, C_in={}) = {} elements, got {}. \
             This input must match the input used in the corresponding forward pass.",
            batch_size,
            self.input_height,
            self.input_width,
            self.in_channels,
            batch_size * self.input_size(),
            input.len()
        );
        assert_eq!(
            grad_output.len(),
            batch_size * self.output_size(),
            "Conv2DLayer backward: grad_output shape mismatch. \
             Expected (batch={}, C_out={}, H_out={}, W_out={}) = {} elements, got {}. \
             Ensure grad_output has batch_size * out_channels * output_height * output_width elements.",
            batch_size,
            self.out_channels,
            out_h,
            out_w,
            batch_size * self.output_size(),
            grad_output.len()
        );
        assert_eq!(
            grad_input.len(),
            batch_size * self.input_size(),
            "Conv2DLayer backward: grad_input shape mismatch. \
             Expected (batch={}, H={}, W={}, C_in={}) = {} elements, got {}. \
             Ensure grad_input buffer has batch_size * in_channels * input_height * input_width elements.",
            batch_size,
            self.input_height,
            self.input_width,
            self.in_channels,
            batch_size * self.input_size(),
            grad_input.len()
        );

        let out_spatial = out_h * out_w;
        let in_spatial = self.input_height * self.input_width;

        // Zero out accumulators before processing the current batch
        self.grad_weights.zero();
        self.grad_biases.zero();

        // Borrow gradient accumulators for direct index-based accumulation
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

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let g = grad_output[g_base + oy * out_w + ox];

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize
                                        - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize
                                        - self.padding;

                                    if iy >= 0
                                        && iy < self.input_height as isize
                                        && ix >= 0
                                        && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base
                                            + (iyy * self.input_width + ixx) * self.in_channels
                                            + ic;
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

        // Compute gradient with respect to input
        for v in grad_input.iter_mut() {
            *v = 0.0;
        }

        for b in 0..batch_size {
            let in_base = b * (self.in_channels * in_spatial);
            let g_base_b = b * (self.out_channels * out_spatial);

            for ic in 0..self.in_channels {
                for oc in 0..self.out_channels {
                    let g_base = g_base_b + oc * out_spatial;
                    let w_base = (oc * self.in_channels + ic) * self.kernel_size * self.kernel_size;

                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let g = grad_output[g_base + oy * out_w + ox];

                            for ky in 0..self.kernel_size {
                                for kx in 0..self.kernel_size {
                                    let iy = oy as isize * self.stride as isize + ky as isize
                                        - self.padding;
                                    let ix = ox as isize * self.stride as isize + kx as isize
                                        - self.padding;

                                    if iy >= 0
                                        && iy < self.input_height as isize
                                        && ix >= 0
                                        && ix < self.input_width as isize
                                    {
                                        let iyy = iy as usize;
                                        let ixx = ix as usize;
                                        let in_idx = in_base
                                            + (iyy * self.input_width + ixx) * self.in_channels
                                            + ic;
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

    /// Applies a gradient-descent step to the layer's parameters and resets accumulated gradients.
    ///
    /// Updates each weight and bias by subtracting `learning_rate * gradient` using the values
    /// stored in the layer's internal gradient accumulators, then zeroes those accumulators.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Create a tiny layer and a deterministic RNG (types must be in scope).
    /// let mut rng = SimpleRng::new(0);
    /// let mut layer = Conv2DLayer::new(1, 1, 1, 0, 1, 1, 1, &mut rng);
    ///
    /// // Simulate accumulated gradients
    /// {
    ///     let mut gw = layer.grad_weights.borrow_mut();
    ///     gw[0] = 0.5;
    ///     let mut gb = layer.grad_biases.borrow_mut();
    ///     gb[0] = 0.25;
    /// }
    ///
    /// let old_w = layer.weights[0];
    /// let old_b = layer.biases[0];
    /// layer.update_parameters(0.1);
    /// assert_eq!(layer.grad_weights.borrow()[0], 0.0);
    /// assert_eq!(layer.grad_biases.borrow()[0], 0.0);
    /// assert_eq!(layer.weights[0], old_w - 0.1 * 0.5);
    /// assert_eq!(layer.biases[0], old_b - 0.1 * 0.25);
    /// ```
    fn update_parameters(&mut self, learning_rate: f32) {
        self.grad_weights
            .apply_sgd_update(&mut self.weights, learning_rate);
        self.grad_biases
            .apply_sgd_update(&mut self.biases, learning_rate);
    }

    fn update_with_optimizer(&mut self, optimizer: &mut dyn crate::optimizers::Optimizer) {
        self.grad_weights
            .apply_optimizer_update(&mut self.weights, optimizer);
        self.grad_biases
            .apply_optimizer_update(&mut self.biases, optimizer);
    }

    /// Computes the total number of values in a single input example.
    ///
    /// # Examples
    ///
    /// ```
    /// // Demonstrates the calculation for a layer with 3 channels and 28x28 spatial size.
    /// # struct Dummy { in_channels: usize, input_height: usize, input_width: usize }
    /// # impl Dummy { fn input_size(&self) -> usize { self.in_channels * self.input_height * self.input_width } }
    /// let layer = Dummy { in_channels: 3, input_height: 28, input_width: 28 };
    /// assert_eq!(layer.input_size(), 3 * 28 * 28);
    /// ```
    fn input_size(&self) -> usize {
        self.in_channels * self.input_height * self.input_width
    }

    /// Compute the total number of scalar elements in a single output feature map.
    ///
    /// This is the product of the number of output channels and the spatial dimensions:
    /// out_channels * output_height() * output_width().
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// use rust_neural_networks::layers::Layer;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 10, 10, &mut rng);
    /// let n = layer.output_size();
    /// assert_eq!(n, layer.out_channels() * layer.output_height() * layer.output_width());
    /// ```
    fn output_size(&self) -> usize {
        self.out_channels * self.output_height() * self.output_width()
    }

    /// Return the total number of trainable parameters (weights and biases).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
    /// assert_eq!(layer.parameter_count(), 1 * 8 * 3 * 3 + 8);
    /// ```
    fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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

    #[test]
    fn test_conv2d_forward() {
        let mut rng = SimpleRng::new(42);
        // 1 input channel, 2 output channels, 3x3 kernel, padding=1, stride=1, 4x4 input
        let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

        // Single sample: 1 channel × 4 × 4 = 16 values
        let input = vec![1.0f32; 16];
        // Output: 2 channels × 4 × 4 = 32 values
        let mut output = vec![0.0f32; 32];

        layer.forward(&input, &mut output, 1);

        // Output should be computed (finite values)
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_conv2d_forward_batch() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

        // Batch of 2 samples
        let input = vec![1.0f32; 32]; // 2 × 16
        let mut output = vec![0.0f32; 64]; // 2 × 32

        layer.forward(&input, &mut output, 2);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_conv2d_backward() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

        let input = vec![1.0f32; 16];
        let mut output = vec![0.0f32; 32];
        layer.forward(&input, &mut output, 1);

        // Gradient from loss
        let grad_output = vec![1.0f32; 32];
        let mut grad_input = vec![0.0f32; 16];

        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Gradients should be finite
        assert!(grad_input.iter().all(|&x| x.is_finite()));
        // At least some gradients should be non-zero
        assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_conv2d_update_parameters() {
        let mut rng = SimpleRng::new(42);
        let mut layer = Conv2DLayer::new(1, 2, 3, 1, 1, 4, 4, &mut rng);

        let original_weights = layer.weights.clone();

        let input = vec![1.0f32; 16];
        let mut output = vec![0.0f32; 32];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0f32; 32];
        let mut grad_input = vec![0.0f32; 16];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        layer.update_parameters(0.1);

        // Weights should have changed
        let weights_changed = layer
            .weights
            .iter()
            .zip(original_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(weights_changed, "Weights should change after update");
    }

    #[test]
    fn test_conv2d_input_output_size() {
        let mut rng = SimpleRng::new(42);
        let layer = Conv2DLayer::new(3, 8, 3, 1, 1, 28, 28, &mut rng);

        // Input size: 3 channels × 28 × 28
        assert_eq!(layer.input_size(), 3 * 28 * 28);
        // Output size: 8 channels × 28 × 28 (with padding=1, same dimensions)
        assert_eq!(layer.output_size(), 8 * 28 * 28);
    }

    #[test]
    fn test_conv2d_stride_2() {
        let mut rng = SimpleRng::new(42);
        // Stride 2 should halve the output dimensions
        let layer = Conv2DLayer::new(1, 4, 3, 1, 2, 8, 8, &mut rng);

        // (8 + 2*1 - 3) / 2 + 1 = 4
        assert_eq!(layer.output_height(), 4);
        assert_eq!(layer.output_width(), 4);
    }

    #[test]
    fn test_conv2d_new_with_weights_stores_parameters() {
        let in_channels = 1;
        let out_channels = 2;
        let kernel_size = 3;
        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let weights = vec![0.1f32; weight_count];
        let biases = vec![0.0f32, 0.5];

        let layer = Conv2DLayer::new_with_weights(
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            28,
            28,
            weights.clone(),
            biases.clone(),
        );

        assert_eq!(layer.in_channels(), in_channels);
        assert_eq!(layer.out_channels(), out_channels);
        assert_eq!(layer.kernel_size(), kernel_size);
        assert_eq!(layer.padding(), 1);
        assert_eq!(layer.stride(), 1);
        assert_eq!(layer.weights(), weights.as_slice());
        assert_eq!(layer.biases(), biases.as_slice());
    }

    #[test]
    fn test_conv2d_new_with_weights_gradient_initially_zero() {
        let in_channels = 1;
        let out_channels = 2;
        let kernel_size = 3;
        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let weights = vec![0.1f32; weight_count];
        let biases = vec![0.0f32; out_channels];

        let layer = Conv2DLayer::new_with_weights(
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            28,
            28,
            weights,
            biases,
        );

        // Gradient accumulators should start at zero
        let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
        assert_eq!(weight_norm, 0.0);
        assert_eq!(bias_norm, 0.0);
    }

    #[test]
    fn test_conv2d_new_with_weights_parameter_count() {
        let in_channels = 3;
        let out_channels = 8;
        let kernel_size = 3;
        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let weights = vec![0.0f32; weight_count];
        let biases = vec![0.0f32; out_channels];

        let layer = Conv2DLayer::new_with_weights(
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            32,
            32,
            weights,
            biases,
        );

        // 8 * 3 * 3 * 3 weights + 8 biases = 216 + 8 = 224
        assert_eq!(layer.parameter_count(), weight_count + out_channels);
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn test_conv2d_new_with_weights_wrong_weight_length_panics() {
        // 1 in, 2 out, 3×3 kernel: expects 2*1*3*3 = 18 weights, give 10
        let weights = vec![0.1f32; 10];
        let biases = vec![0.0f32; 2];
        let _layer =
            Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
    }

    #[test]
    #[should_panic(expected = "biases length")]
    fn test_conv2d_new_with_weights_wrong_bias_length_panics() {
        // 1 in, 2 out, 3×3 kernel: expects 2 biases, give 5
        let weights = vec![0.1f32; 2 * 1 * 3 * 3];
        let biases = vec![0.0f32; 5];
        let _layer =
            Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
    }
}
