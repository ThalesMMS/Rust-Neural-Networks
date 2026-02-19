//! 2D Convolutional layer implementation
//!
//! This module provides a Conv2DLayer that performs 2D convolution operations,
//! commonly used in computer vision tasks like image classification.
//! Inputs are expected in channels-last (pixel-interleaved) order.

use crate::layers::gradient::GradientAccumulator;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::gpu::backend::{GpuBackend, GpuError};
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

// GPU-accelerated forward and backward implementations
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
impl Conv2DLayer {
    /// Constructs a Conv2DLayer and attaches the provided GPU backend for accelerated forward and backward computations.
    ///
    /// The layer will attempt to use the backend for GPU-accelerated operations and may fall back to the CPU path if a GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// // let mut rng = SimpleRng::new(seed);
    /// // let backend: Arc<dyn GpuBackend> = Arc::new(MyGpuBackend::new(...));
    /// // let layer = Conv2DLayer::new_with_gpu(3, 16, 3, 1, 1, 32, 32, &mut rng, backend);
    /// ```
    pub fn new_with_gpu(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: isize,
        stride: usize,
        input_height: usize,
        input_width: usize,
        rng: &mut SimpleRng,
        gpu_backend: Arc<dyn GpuBackend>,
    ) -> Self {
        let mut layer = Self::new(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            input_height,
            input_width,
            rng,
        );
        layer.gpu_backend = Some(gpu_backend);
        layer
    }

    /// Attach or replace the GPU backend used by this layer.
    ///
    /// After calling this, the layer will attempt to use the provided backend for GPU-accelerated
    /// forward and backward paths when available; the previous backend (if any) is replaced.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use rust_neural_networks::gpu::GpuBackend;
    /// // `backend` must implement the `GpuBackend` trait.
    /// let backend: Arc<dyn GpuBackend> = /* create backend */;
    /// layer.set_gpu_backend(backend);
    /// ```
    pub fn set_gpu_backend(&mut self, backend: Arc<dyn GpuBackend>) {
        self.gpu_backend = Some(backend);
    }

    /// Checks whether a GPU backend is attached to the layer.
    ///
    /// # Returns
    ///
    /// `true` if a GPU backend is attached, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // given a `Conv2DLayer` instance `layer`
    /// let has_gpu = layer.has_gpu_backend();
    /// ```
    pub fn has_gpu_backend(&self) -> bool {
        self.gpu_backend.is_some()
    }

    /// Performs the forward convolution using the attached GPU backend.
    ///
    /// Attempts to compute this layer's forward pass on the GPU. If the layer's
    /// padding is negative or the backend fails, an error is returned so the caller
    /// may fall back to a CPU implementation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `layer` is a Conv2DLayer and `backend` implements GpuBackend.
    /// let batch_size = 1;
    /// let input = vec![0.0f32; layer.input_size() * batch_size];
    /// let mut output = vec![0.0f32; layer.output_size() * batch_size];
    /// layer.forward_gpu(&input, &mut output, batch_size, backend.as_ref())?;
    /// ```
    ///
    /// # Returns
    ///
    /// `Ok(())` if the GPU convolution completed successfully, `Err(GpuError)` if the
    /// GPU backend cannot perform the operation (for example, when padding is
    /// negative) or if the backend call fails.
    fn forward_gpu(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        backend: &dyn GpuBackend,
    ) -> Result<(), GpuError> {
        // Padding must be non-negative for GPU backend (uses usize)
        let padding = if self.padding >= 0 {
            self.padding as usize
        } else {
            return Err(GpuError::Unsupported(
                "Negative padding not supported on GPU".to_string(),
            ));
        };

        backend.conv2d_forward(
            input,
            &self.weights,
            &self.biases,
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            self.input_height,
            self.input_width,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            padding,
        )
    }

    /// Performs the convolution backward pass on the GPU, accumulating gradients for weights and biases and writing input gradients into `grad_input`.
    ///
    /// On success, GPU-computed gradients for filters and biases are accumulated into the layer's internal
    /// gradient accumulators and `grad_input` is populated with gradients w.r.t. the input.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Unsupported(_))` if the layer's padding is negative (GPU path requires non-negative padding).
    /// Returns other `GpuError` variants if the GPU backend reports a failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `layer` is a Conv2DLayer and `backend` implements GpuBackend.
    /// // `input`, `grad_output`, and `grad_input` must have appropriate lengths.
    /// let res = layer.backward_gpu(&input, &grad_output, &mut grad_input, batch_size, &backend);
    /// match res {
    ///     Ok(()) => { /* gradients accumulated into layer */ }
    ///     Err(e) => { /* handle GPU failure or fallback to CPU */ }
    /// }
    /// ```
    fn backward_gpu(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
        backend: &dyn GpuBackend,
    ) -> Result<(), GpuError> {
        let padding = if self.padding >= 0 {
            self.padding as usize
        } else {
            return Err(GpuError::Unsupported(
                "Negative padding not supported on GPU".to_string(),
            ));
        };

        let weight_count =
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;
        let mut grad_filters = vec![0.0f32; weight_count];
        let mut grad_bias = vec![0.0f32; self.out_channels];

        backend.conv2d_backward(
            input,
            &self.weights,
            grad_output,
            grad_input,
            &mut grad_filters,
            &mut grad_bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            self.input_height,
            self.input_width,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            padding,
        )?;

        // Accumulate GPU-computed gradients into the gradient accumulators
        self.grad_weights.accumulate(&grad_filters);
        self.grad_biases.accumulate(&grad_bias);

        Ok(())
    }
}

// Layer trait implementation

impl Layer for Conv2DLayer {
    /// Computes this layer's convolution over a batch of inputs and writes feature maps into `output`.
    ///
    /// The `input` slice must be laid out channels-last per sample: (H × W × C_in). The `output`
    /// slice must be laid out per sample as (C_out × H_out × W_out). `batch_size` is the number of
    /// samples packed consecutively in each buffer (so total lengths are `batch_size * input_size()`
    /// and `batch_size * output_size()` respectively).
    ///
    /// This method will panic if the provided buffer lengths do not match the expected sizes for the
    /// configured layer dimensions. If a GPU backend is attached and its forward call succeeds, the
    /// GPU-accelerated path is used; otherwise the CPU implementation is executed.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
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
        // Try GPU-accelerated path if backend is available
        #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
        {
            if let Some(ref backend) = self.gpu_backend {
                if self
                    .forward_gpu(input, output, batch_size, backend.as_ref())
                    .is_ok()
                {
                    return;
                }
                // GPU failed, fall through to CPU path
            }
        }

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

    /// Accumulates gradients for the layer's parameters from `grad_output` and writes the
    /// gradient with respect to the layer input into `grad_input`.
    ///
    /// Input and output buffer layouts:
    /// - `input` and `grad_input` use NHWC (batch, height, width, channels).
    /// - `grad_output` uses the layer output layout (batch, out_channels, out_height, out_width).
    ///
    /// Buffer length requirements:
    /// - `input.len() == batch_size * self.input_size()`
    /// - `grad_output.len() == batch_size * self.output_size()`
    /// - `grad_input.len() == batch_size * self.input_size()`
    ///
    /// This method zeros `self.grad_weights` and `self.grad_biases` before accumulating
    /// gradients across the batch and spatial dimensions. `grad_input` is overwritten
    /// with the computed input gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::conv2d::Conv2DLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = Conv2DLayer::new(1, 1, 3, 1, 1, 5, 5, &mut rng);
    /// let batch = 1;
    /// let input = vec![0.0f32; batch * layer.input_size()];
    /// let grad_out = vec![1.0f32; batch * layer.output_size()];
    /// let mut grad_in = vec![0.0f32; batch * layer.input_size()];
    ///
    /// layer.backward(&input, &grad_out, &mut grad_in, batch);
    /// assert!(grad_in.iter().any(|&v| v != 0.0));
    /// ```
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

        // Try GPU-accelerated path if backend is available
        #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
        {
            if let Some(ref backend) = self.gpu_backend {
                // Zero accumulators before GPU backward
                self.grad_weights.zero();
                self.grad_biases.zero();
                // Zero grad_input
                for v in grad_input.iter_mut() {
                    *v = 0.0;
                }
                if self
                    .backward_gpu(input, grad_output, grad_input, batch_size, backend.as_ref())
                    .is_ok()
                {
                    return;
                }
                // GPU failed, fall through to CPU path (accumulators already zeroed, that's fine)
            }
        }

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

    /// Estimated FLOPS for a Conv2D forward pass.
    ///
    /// Each output spatial location requires computing
    /// `in_channels * kernel_size * kernel_size` multiply-add operations per
    /// output filter, so total FLOPS are:
    ///
    /// `2 * batch_size * in_channels * kernel_h * kernel_w * out_channels * out_h * out_w`
    fn flops_forward(&self, batch_size: usize) -> u64 {
        let out_h = self.output_height();
        let out_w = self.output_width();
        2 * batch_size as u64
            * self.in_channels as u64
            * self.kernel_size as u64
            * self.kernel_size as u64
            * self.out_channels as u64
            * out_h as u64
            * out_w as u64
    }

    /// Estimated FLOPS for a Conv2D backward pass.
    ///
    /// Both the gradient with respect to the input and the gradient with
    /// respect to the weights involve convolution operations of similar
    /// complexity to the forward pass.  A conservative estimate of 2× the
    /// forward FLOPS is used.
    fn flops_backward(&self, batch_size: usize) -> u64 {
        2 * self.flops_forward(batch_size)
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
        let _layer = Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
    }

    #[test]
    #[should_panic(expected = "biases length")]
    fn test_conv2d_new_with_weights_wrong_bias_length_panics() {
        // 1 in, 2 out, 3×3 kernel: expects 2 biases, give 5
        let weights = vec![0.1f32; 2 * 3 * 3];
        let biases = vec![0.0f32; 5];
        let _layer = Conv2DLayer::new_with_weights(1, 2, 3, 1, 1, 28, 28, weights, biases);
    }
}