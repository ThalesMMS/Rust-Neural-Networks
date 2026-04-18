use super::CudaBackend;
use crate::gpu::backend::GpuError;
use cudarc::driver::{LaunchConfig, PushKernelArg};

impl CudaBackend {
    /// Applies the ReLU activation elementwise to the provided slice in-place using the CUDA backend.
    ///
    /// Performs no work if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// // Obtain a backend appropriate for your environment.
    /// let backend = CudaBackend::new();
    /// let mut data = vec![-1.0f32, 0.5, -0.2];
    /// backend.relu_impl(&mut data).unwrap();
    /// assert_eq!(data[0], 0.0);
    /// assert!(data[1] > 0.0);
    /// ```
    ///
    /// # Returns
    /// `Ok(())` on success; `Err(GpuError)` when a host↔device transfer or kernel launch fails.
    pub(super) fn relu_impl(&self, data: &mut [f32]) -> Result<(), GpuError> {
        let len = data.len();
        if len == 0 {
            return Ok(());
        }

        let stream = self.stream();
        let mut dev_data = stream
            .clone_htod(data)
            .map_err(|e| GpuError::TransferError(format!("relu: host→device: {}", e)))?;

        let f = self.elementwise_fn("relu")?;
        let n = len as i32;
        let cfg = Self::launch_cfg_1d(len);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut dev_data);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("relu kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_data, data)
            .map_err(|e| GpuError::TransferError(format!("relu: device→host: {}", e)))?;
        Ok(())
    }

    /// Compute the gradient of a ReLU activation on the GPU and store it in `grad_input`.
    ///
    /// On success, writes the elementwise ReLU-backward result into `grad_input` and returns `Ok(())`.
    /// Returns `GpuError::DimensionMismatch` if `input`, `grad_output`, and `grad_input` do not have the same length.
    /// Transfer failures are returned as `GpuError::TransferError(...)` and kernel launch failures as `GpuError::OperationFailed(...)`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use your_crate::gpu::cuda_backend::CudaBackend;
    /// # use your_crate::gpu::GpuError;
    /// let backend: CudaBackend = /* obtain backend */;
    /// let input = vec![0.5_f32, -1.0, 2.0];
    /// let grad_output = vec![1.0_f32, 1.0, 1.0];
    /// let mut grad_input = vec![0.0_f32; input.len()];
    /// backend.relu_backward_impl(&input, &grad_output, &mut grad_input).unwrap();
    /// ```
    pub(super) fn relu_backward_impl(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        let len = input.len();
        if len == 0 {
            return Ok(());
        }
        if grad_output.len() != len || grad_input.len() != len {
            return Err(GpuError::DimensionMismatch(
                "relu_backward: input, grad_output, and grad_input must have the same length"
                    .into(),
            ));
        }

        let stream = self.stream();
        let dev_input = stream
            .clone_htod(input)
            .map_err(|e| GpuError::TransferError(format!("relu_backward: {}", e)))?;
        let dev_grad_output = stream
            .clone_htod(grad_output)
            .map_err(|e| GpuError::TransferError(format!("relu_backward: {}", e)))?;
        let mut dev_grad_input = stream
            .clone_htod(grad_input)
            .map_err(|e| GpuError::TransferError(format!("relu_backward: {}", e)))?;

        let f = self.elementwise_fn("relu_backward")?;
        let n = len as i32;
        let cfg = Self::launch_cfg_1d(len);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&dev_input);
        builder.arg(&dev_grad_output);
        builder.arg(&mut dev_grad_input);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("relu_backward kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_grad_input, grad_input)
            .map_err(|e| GpuError::TransferError(format!("relu_backward: device→host: {}", e)))?;
        Ok(())
    }

    /// Applies the elementwise sigmoid function to `data` in-place using the CUDA backend.
    ///
    /// Transfers `data` to the device, runs the `sigmoid` CUDA kernel over all elements, and
    /// copies the results back into `data`. Returns `Ok(())` if the operation completed and
    /// `Err(GpuError)` if any host↔device transfer or kernel launch failed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use your_crate::gpu::CudaBackend;
    /// # fn get_backend() -> CudaBackend { unimplemented!() }
    /// let mut data = [0.0f32, 1.0, -1.0];
    /// let backend = get_backend();
    /// backend.sigmoid_impl(&mut data).unwrap();
    /// ```
    pub(super) fn sigmoid_impl(&self, data: &mut [f32]) -> Result<(), GpuError> {
        let len = data.len();
        if len == 0 {
            return Ok(());
        }

        let stream = self.stream();
        let mut dev_data = stream
            .clone_htod(data)
            .map_err(|e| GpuError::TransferError(format!("sigmoid: {}", e)))?;

        let f = self.elementwise_fn("sigmoid")?;
        let n = len as i32;
        let cfg = Self::launch_cfg_1d(len);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut dev_data);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("sigmoid kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_data, data)
            .map_err(|e| GpuError::TransferError(format!("sigmoid: device→host: {}", e)))?;
        Ok(())
    }

    /// Computes the gradient with respect to the input of a sigmoid activation by applying
    /// the sigmoid-backward CUDA kernel over `sigmoid_output` and `grad_output`, writing results into `grad_input`.
    ///
    /// Validates that all three slices have the same length and returns early on empty input.
    /// Host↔device transfer failures and kernel launch failures are reported through `GpuError`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `Err(GpuError::DimensionMismatch)` if the input/output slices differ in length;
    /// `Err(GpuError::TransferError)` for host↔device or device→host transfer failures;
    /// `Err(GpuError::OperationFailed)` if the CUDA kernel launch fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let sigmoid_output = vec![0.5_f32, 0.8, 0.2];
    /// let grad_output = vec![0.1_f32, 0.2, 0.3];
    /// let mut grad_input = vec![0.0_f32; 3];
    /// // `backend` is a CudaBackend instance available in scope.
    /// // backend.sigmoid_backward_impl(&sigmoid_output, &grad_output, &mut grad_input).unwrap();
    /// ```
    pub(super) fn sigmoid_backward_impl(
        &self,
        sigmoid_output: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        let len = sigmoid_output.len();
        if len == 0 {
            return Ok(());
        }
        if grad_output.len() != len || grad_input.len() != len {
            return Err(GpuError::DimensionMismatch(
                "sigmoid_backward: sigmoid_output, grad_output, and grad_input must have the same length".into(),
            ));
        }

        let stream = self.stream();
        let dev_sigmoid = stream
            .clone_htod(sigmoid_output)
            .map_err(|e| GpuError::TransferError(format!("sigmoid_backward: {}", e)))?;
        let dev_grad_output = stream
            .clone_htod(grad_output)
            .map_err(|e| GpuError::TransferError(format!("sigmoid_backward: {}", e)))?;
        let mut dev_grad_input = stream
            .clone_htod(grad_input)
            .map_err(|e| GpuError::TransferError(format!("sigmoid_backward: {}", e)))?;

        let f = self.elementwise_fn("sigmoid_backward")?;
        let n = len as i32;
        let cfg = Self::launch_cfg_1d(len);
        let mut builder = stream.launch_builder(&f);
        builder.arg(&dev_sigmoid);
        builder.arg(&dev_grad_output);
        builder.arg(&mut dev_grad_input);
        builder.arg(&n);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("sigmoid_backward kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_grad_input, grad_input)
            .map_err(|e| {
                GpuError::TransferError(format!("sigmoid_backward: device→host: {}", e))
            })?;
        Ok(())
    }

    /// Adds a bias vector to each row of a batched data matrix using the CUDA backend.
    ///
    /// The function adds `bias[j]` to every element `data[i * n + j]` for each batch row `i` in
    /// 0..`batch_size`, updating `data` in place on success.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DimensionMismatch` if `data.len() < batch_size * n` or `bias.len() < n`.
    /// Returns `GpuError::TransferError` for host↔device transfer failures and
    /// `GpuError::OperationFailed` if the kernel launch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let batch_size = 4;
    /// let n = 3;
    /// let mut data = vec![0.0f32; batch_size * n];
    /// let bias = vec![1.0f32; n];
    /// // `backend` is a configured `CudaBackend` instance.
    /// // backend.add_bias_impl(&mut data, &bias, batch_size, n).unwrap();
    /// ```
    pub(super) fn add_bias_impl(
        &self,
        data: &mut [f32],
        bias: &[f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        if data.len() < batch_size * n {
            return Err(GpuError::DimensionMismatch(
                "add_bias: data length mismatch".into(),
            ));
        }
        if bias.len() < n {
            return Err(GpuError::DimensionMismatch(
                "add_bias: bias length mismatch".into(),
            ));
        }
        if batch_size == 0 || n == 0 {
            return Ok(());
        }

        let stream = self.stream();
        let mut dev_data = stream
            .clone_htod(data)
            .map_err(|e| GpuError::TransferError(format!("add_bias: {}", e)))?;
        let dev_bias = stream
            .clone_htod(bias)
            .map_err(|e| GpuError::TransferError(format!("add_bias: {}", e)))?;

        let f = self.elementwise_fn("add_bias")?;
        let bs = batch_size as i32;
        let ni = n as i32;

        let block_x = 256u32.min(n as u32);
        let block_y = 1u32;
        let grid_x = (n as u32 + block_x - 1) / block_x;
        let grid_y = batch_size as u32;
        let cfg = LaunchConfig {
            block_dim: (block_x, block_y, 1),
            grid_dim: (grid_x, grid_y, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&f);
        builder.arg(&mut dev_data);
        builder.arg(&dev_bias);
        builder.arg(&bs);
        builder.arg(&ni);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("add_bias kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_data, &mut data[..batch_size * n])
            .map_err(|e| GpuError::TransferError(format!("add_bias: device→host: {}", e)))?;
        Ok(())
    }

    /// Computes per-column sums over `batch_size` rows of `data` and writes the results into `out`.
    ///
    /// `data` is interpreted as `batch_size` consecutive rows of length `n` (row-major);
    /// `out` must have length at least `n` and will be updated with the sum of each column across all rows.
    ///
    /// # Examples
    ///
    /// ```
    /// // Given two rows of 3 elements:
    /// // data = [a, b, c, d, e, f]  (batch_size = 2, n = 3)
    /// // out will become [a + d, b + e, c + f]
    /// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut out = [0.0f32; 3];
    /// // backend.sum_rows_impl(&data, &mut out, 2, 3).unwrap();
    /// // assert_eq!(out, [5.0, 7.0, 9.0]);
    /// ```
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError)` on failure (e.g., dimension mismatch, transfer or kernel errors).
    pub(super) fn sum_rows_impl(
        &self,
        data: &[f32],
        out: &mut [f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        if data.len() < batch_size * n {
            return Err(GpuError::DimensionMismatch(
                "sum_rows: data length mismatch".into(),
            ));
        }
        if out.len() < n {
            return Err(GpuError::DimensionMismatch(
                "sum_rows: out length mismatch".into(),
            ));
        }
        if batch_size == 0 || n == 0 {
            return Ok(());
        }

        let stream = self.stream();
        let dev_data = stream
            .clone_htod(data)
            .map_err(|e| GpuError::TransferError(format!("sum_rows: {}", e)))?;
        let mut dev_out = stream
            .clone_htod(out)
            .map_err(|e| GpuError::TransferError(format!("sum_rows: {}", e)))?;

        let f = self.elementwise_fn("sum_rows")?;
        let bs = batch_size as i32;
        let ni = n as i32;
        let cfg = Self::launch_cfg_1d(n);

        let mut builder = stream.launch_builder(&f);
        builder.arg(&dev_data);
        builder.arg(&mut dev_out);
        builder.arg(&bs);
        builder.arg(&ni);
        unsafe { builder.launch(cfg) }
            .map_err(|e| GpuError::OperationFailed(format!("sum_rows kernel: {}", e)))?;

        stream
            .memcpy_dtoh(&dev_out, out)
            .map_err(|e| GpuError::TransferError(format!("sum_rows: device→host: {}", e)))?;
        Ok(())
    }
}
