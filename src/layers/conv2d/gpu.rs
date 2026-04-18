use super::Conv2DLayer;
use crate::gpu::backend::{GpuBackend, GpuError};
use crate::utils::rng::SimpleRng;
use std::sync::Arc;

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
    pub(super) fn forward_gpu(
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
    pub(super) fn backward_gpu(
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
