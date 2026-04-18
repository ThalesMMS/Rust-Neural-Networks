use super::CudaBackend;
use crate::gpu::backend::GpuError;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

impl CudaBackend {
    /// Executes the forward pass of a 2D convolution using the CUDA backend.
    ///
    /// Validates convolution parameters and buffer lengths, then runs the CUDA kernel to compute
    /// the convolution output, writing results into the provided `output` slice.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(GpuError)` when validation, transfer, allocation, or kernel execution fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use crate::gpu::cuda_backend::CudaBackend;
    /// # use crate::gpu::GpuError;
    /// # fn example(backend: &CudaBackend) -> Result<(), GpuError> {
    /// let batch = 1;
    /// let ic = 3;
    /// let oc = 8;
    /// let ih = 32;
    /// let iw = 32;
    /// let kh = 3;
    /// let kw = 3;
    /// let stride = 1;
    /// let padding = 1;
    ///
    /// let input = vec![0.0f32; batch * ic * ih * iw];
    /// let filters = vec![0.0f32; oc * ic * kh * kw];
    /// let bias = vec![0.0f32; oc];
    /// let out_h = (ih + 2 * padding - kh) / stride + 1;
    /// let out_w = (iw + 2 * padding - kw) / stride + 1;
    /// let mut output = vec![0.0f32; batch * oc * out_h * out_w];
    ///
    /// backend.conv2d_forward_impl(
    ///     &input,
    ///     &filters,
    ///     &bias,
    ///     &mut output,
    ///     batch,
    ///     ic,
    ///     oc,
    ///     ih,
    ///     iw,
    ///     kh,
    ///     kw,
    ///     stride,
    ///     padding,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub(super) fn conv2d_forward_impl(
        &self,
        input: &[f32],
        filters: &[f32],
        bias: &[f32],
        output: &mut [f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<(), GpuError> {
        if stride == 0 {
            return Err(GpuError::DimensionMismatch(
                "conv2d_forward: stride must be > 0".into(),
            ));
        }
        if kernel_h > input_h + 2 * padding || kernel_w > input_w + 2 * padding {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_forward: kernel ({}x{}) exceeds padded input ({}x{})",
                kernel_h,
                kernel_w,
                input_h + 2 * padding,
                input_w + 2 * padding
            )));
        }

        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

        let expected_input = batch_size * in_channels * input_h * input_w;
        let expected_filters = out_channels * in_channels * kernel_h * kernel_w;
        let expected_output = batch_size * out_channels * out_h * out_w;

        if input.len() < expected_input {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_forward: input len {} < expected {}",
                input.len(),
                expected_input
            )));
        }
        if filters.len() < expected_filters {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_forward: filters len {} < expected {}",
                filters.len(),
                expected_filters
            )));
        }
        if bias.len() < out_channels {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_forward: bias len {} < expected {}",
                bias.len(),
                out_channels
            )));
        }
        if output.len() < expected_output {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_forward: output len {} < expected {}",
                output.len(),
                expected_output
            )));
        }

        let stream = self.stream();
        let dev_input = stream
            .clone_htod(input)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        let dev_filters = stream
            .clone_htod(filters)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        let dev_bias = stream
            .clone_htod(bias)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        let mut dev_output: CudaSlice<f32> = stream
            .alloc_zeros(expected_output)
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        let f = self.conv2d_fn("conv2d_forward")?;

        let block_x = 16u32.min(out_w as u32);
        let block_y = 16u32.min(out_h as u32);
        let cfg = LaunchConfig {
            block_dim: (block_x, block_y, 1),
            grid_dim: (
                (out_w as u32 + block_x - 1) / block_x,
                (out_h as u32 + block_y - 1) / block_y,
                (batch_size * out_channels) as u32,
            ),
            shared_mem_bytes: 0,
        };

        let (bs, ic, oc) = (batch_size as i32, in_channels as i32, out_channels as i32);
        let (ih, iw, kh, kw) = (
            input_h as i32,
            input_w as i32,
            kernel_h as i32,
            kernel_w as i32,
        );
        let (s, p, oh_i, ow_i) = (stride as i32, padding as i32, out_h as i32, out_w as i32);

        let mut builder = stream.launch_builder(&f);
        builder.arg(&dev_input);
        builder.arg(&dev_filters);
        builder.arg(&dev_bias);
        builder.arg(&mut dev_output);
        builder.arg(&bs);
        builder.arg(&ic);
        builder.arg(&oc);
        builder.arg(&ih);
        builder.arg(&iw);
        builder.arg(&kh);
        builder.arg(&kw);
        builder.arg(&s);
        builder.arg(&p);
        builder.arg(&oh_i);
        builder.arg(&ow_i);
        unsafe { builder.launch(cfg) }.map_err(|e| GpuError::OperationFailed(e.to_string()))?;

        stream
            .memcpy_dtoh(&dev_output, &mut output[..expected_output])
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        Ok(())
    }

    /// Performs the CUDA-backed backward pass for a 2D convolution, computing gradients for
    /// the input, filters, and bias and writing them into the provided mutable slices.
    ///
    /// The function expects tensors in the following memory layout:
    /// - `input`: batch_size × in_channels × input_h × input_w
    /// - `filters`: out_channels × in_channels × kernel_h × kernel_w
    /// - `grad_output`: batch_size × out_channels × out_h × out_w
    ///
    /// Validates convolution parameters (stride > 0; kernel fits inside padded input) and
    /// that the provided output slices are large enough. On success this launches three CUDA
    /// kernels in sequence to compute `grad_input`, `grad_filters`, and `grad_bias`, copying
    /// each result back into the corresponding host slice.
    ///
    /// # Errors
    ///
    /// Returns an Err(GpuError) for any of:
    /// - invalid convolution parameters (stride == 0 or kernel larger than padded input),
    /// - mismatched slice lengths for any provided buffer,
    /// - device transfer failures (mapped to `GpuError::TransferError`),
    /// - device allocation failures (mapped to `GpuError::AllocationFailed`),
    /// - kernel launch failures (mapped to `GpuError::OperationFailed`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Construct a backend and buffers appropriate for your environment.
    /// # use crate::gpu::cuda_backend::CudaBackend;
    /// let backend: CudaBackend = unimplemented!();
    /// let input = vec![0.0f32; 1*1*5*5];
    /// let filters = vec![0.0f32; 1*1*3*3];
    /// let grad_output = vec![0.0f32; 1*1*3*3];
    /// let mut grad_input = vec![0.0f32; 1*1*5*5];
    /// let mut grad_filters = vec![0.0f32; 1*1*3*3];
    /// let mut grad_bias = vec![0.0f32; 1];
    /// let result = backend.conv2d_backward_impl(
    ///     &input,
    ///     &filters,
    ///     &grad_output,
    ///     &mut grad_input,
    ///     &mut grad_filters,
    ///     &mut grad_bias,
    ///     1, // batch_size
    ///     1, // in_channels
    ///     1, // out_channels
    ///     5, // input_h
    ///     5, // input_w
    ///     3, // kernel_h
    ///     3, // kernel_w
    ///     1, // stride
    ///     0, // padding
    /// );
    /// assert!(result.is_ok());
    /// ```
    pub(super) fn conv2d_backward_impl(
        &self,
        input: &[f32],
        filters: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_filters: &mut [f32],
        grad_bias: &mut [f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Result<(), GpuError> {
        if stride == 0 {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: stride must be > 0".into(),
            ));
        }
        if kernel_h > input_h + 2 * padding || kernel_w > input_w + 2 * padding {
            return Err(GpuError::DimensionMismatch(format!(
                "conv2d_backward: kernel ({}x{}) exceeds padded input ({}x{})",
                kernel_h,
                kernel_w,
                input_h + 2 * padding,
                input_w + 2 * padding
            )));
        }

        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

        let input_size = batch_size * in_channels * input_h * input_w;
        let filter_size = out_channels * in_channels * kernel_h * kernel_w;
        let output_size = batch_size * out_channels * out_h * out_w;

        if input.len() < input_size {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: input size mismatch".into(),
            ));
        }
        if filters.len() < filter_size {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: filters size mismatch".into(),
            ));
        }
        if grad_output.len() < output_size {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: grad_output size mismatch".into(),
            ));
        }
        if grad_input.len() < input_size {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: grad_input size mismatch".into(),
            ));
        }
        if grad_filters.len() < filter_size {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: grad_filters size mismatch".into(),
            ));
        }
        if grad_bias.len() < out_channels {
            return Err(GpuError::DimensionMismatch(
                "conv2d_backward: grad_bias size mismatch".into(),
            ));
        }

        let stream = self.stream();
        let dev_grad_output = stream
            .clone_htod(grad_output)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        let dev_filters = stream
            .clone_htod(filters)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;
        let dev_input = stream
            .clone_htod(input)
            .map_err(|e| GpuError::TransferError(e.to_string()))?;

        let (bs, ic, oc) = (batch_size as i32, in_channels as i32, out_channels as i32);
        let (ih, iw, kh, kw) = (
            input_h as i32,
            input_w as i32,
            kernel_h as i32,
            kernel_w as i32,
        );
        let (s, p, oh_i, ow_i) = (stride as i32, padding as i32, out_h as i32, out_w as i32);

        // ── 1. Backward input gradient ──────────────────────────────────
        {
            let mut dev_grad_input: CudaSlice<f32> = stream
                .alloc_zeros(input_size)
                .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

            let f = self.conv2d_fn("conv2d_backward_input")?;
            let block_x = 16u32.min(input_w as u32);
            let block_y = 16u32.min(input_h as u32);
            let cfg = LaunchConfig {
                block_dim: (block_x, block_y, 1),
                grid_dim: (
                    (input_w as u32 + block_x - 1) / block_x,
                    (input_h as u32 + block_y - 1) / block_y,
                    (batch_size * in_channels) as u32,
                ),
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&f);
            builder.arg(&dev_grad_output);
            builder.arg(&dev_filters);
            builder.arg(&mut dev_grad_input);
            builder.arg(&bs);
            builder.arg(&ic);
            builder.arg(&oc);
            builder.arg(&ih);
            builder.arg(&iw);
            builder.arg(&kh);
            builder.arg(&kw);
            builder.arg(&s);
            builder.arg(&p);
            builder.arg(&oh_i);
            builder.arg(&ow_i);
            unsafe { builder.launch(cfg) }.map_err(|e| GpuError::OperationFailed(e.to_string()))?;

            stream
                .memcpy_dtoh(&dev_grad_input, &mut grad_input[..input_size])
                .map_err(|e| GpuError::TransferError(e.to_string()))?;
        }

        // ── 2. Backward filter gradient ─────────────────────────────────
        {
            let mut dev_grad_filters: CudaSlice<f32> = stream
                .alloc_zeros(filter_size)
                .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

            let f = self.conv2d_fn("conv2d_backward_filters")?;
            let block_x = 16u32.min(kernel_w as u32);
            let block_y = 16u32.min(kernel_h as u32);
            let cfg = LaunchConfig {
                block_dim: (block_x, block_y, 1),
                grid_dim: (
                    (kernel_w as u32 + block_x - 1) / block_x,
                    (kernel_h as u32 + block_y - 1) / block_y,
                    (out_channels * in_channels) as u32,
                ),
                shared_mem_bytes: 0,
            };

            let mut builder = stream.launch_builder(&f);
            builder.arg(&dev_input);
            builder.arg(&dev_grad_output);
            builder.arg(&mut dev_grad_filters);
            builder.arg(&bs);
            builder.arg(&ic);
            builder.arg(&oc);
            builder.arg(&ih);
            builder.arg(&iw);
            builder.arg(&kh);
            builder.arg(&kw);
            builder.arg(&s);
            builder.arg(&p);
            builder.arg(&oh_i);
            builder.arg(&ow_i);
            unsafe { builder.launch(cfg) }.map_err(|e| GpuError::OperationFailed(e.to_string()))?;

            stream
                .memcpy_dtoh(&dev_grad_filters, &mut grad_filters[..filter_size])
                .map_err(|e| GpuError::TransferError(e.to_string()))?;
        }

        // ── 3. Backward bias gradient ───────────────────────────────────
        {
            let mut dev_grad_bias: CudaSlice<f32> = stream
                .alloc_zeros(out_channels)
                .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

            let f = self.conv2d_fn("conv2d_backward_bias")?;
            let cfg = Self::launch_cfg_1d(out_channels);

            let mut builder = stream.launch_builder(&f);
            builder.arg(&dev_grad_output);
            builder.arg(&mut dev_grad_bias);
            builder.arg(&bs);
            builder.arg(&oc);
            builder.arg(&oh_i);
            builder.arg(&ow_i);
            unsafe { builder.launch(cfg) }.map_err(|e| GpuError::OperationFailed(e.to_string()))?;

            stream
                .memcpy_dtoh(&dev_grad_bias, &mut grad_bias[..out_channels])
                .map_err(|e| GpuError::TransferError(e.to_string()))?;
        }

        Ok(())
    }
}
