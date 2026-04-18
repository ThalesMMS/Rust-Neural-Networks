use super::Conv2DLayer;
use crate::layers::Layer;

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
        assert!(
            batch_size > 0,
            "Conv2DLayer forward: batch_size must be > 0"
        );
        self.validate_parameter_shapes("Conv2DLayer forward");

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
        assert!(
            batch_size > 0,
            "Conv2DLayer backward: batch_size must be > 0"
        );
        self.validate_parameter_shapes("Conv2DLayer backward");

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
