use super::MetalBackend;
use crate::gpu::backend::GpuError;
use metal::{MTLResourceOptions, MTLSize};

impl MetalBackend {
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

        let pipeline = self.get_pipeline("conv2d_forward")?;

        let buf_input = self.create_buffer(input);
        let buf_filters = self.create_buffer(filters);
        let buf_bias = self.create_buffer(bias);
        let buf_output = self.create_buffer(output);

        // params: [batch_size, in_channels, out_channels, input_h, input_w,
        //          kernel_h, kernel_w, stride, padding, out_h, out_w]
        let params: [u32; 11] = [
            batch_size as u32,
            in_channels as u32,
            out_channels as u32,
            input_h as u32,
            input_w as u32,
            kernel_h as u32,
            kernel_w as u32,
            stride as u32,
            padding as u32,
            out_h as u32,
            out_w as u32,
        ];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_input), 0);
        encoder.set_buffer(1, Some(&buf_filters), 0);
        encoder.set_buffer(2, Some(&buf_bias), 0);
        encoder.set_buffer(3, Some(&buf_output), 0);
        encoder.set_buffer(4, Some(&buf_params), 0);

        // grid: (out_w, out_h, batch_size * out_channels)
        let grid = MTLSize::new(
            out_w as u64,
            out_h as u64,
            (batch_size * out_channels) as u64,
        );
        let tw = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let group = MTLSize::new(tw.min(out_w as u64), 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_output.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), expected_output);
        }
        Ok(())
    }

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

        // Shared params for backward_input and backward_filters kernels
        // [batch_size, in_channels, out_channels, input_h, input_w,
        //  kernel_h, kernel_w, stride, padding, out_h, out_w]
        let params: [u32; 11] = [
            batch_size as u32,
            in_channels as u32,
            out_channels as u32,
            input_h as u32,
            input_w as u32,
            kernel_h as u32,
            kernel_w as u32,
            stride as u32,
            padding as u32,
            out_h as u32,
            out_w as u32,
        ];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buf_grad_output = self.create_buffer(grad_output);
        let buf_filters = self.create_buffer(filters);
        let buf_input = self.create_buffer(input);

        // ── 1. Backward input gradient ──────────────────────────────────
        {
            let pipeline = self.get_pipeline("conv2d_backward_input")?;
            let buf_grad_input = self.create_empty_buffer(input_size);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&buf_grad_output), 0);
            encoder.set_buffer(1, Some(&buf_filters), 0);
            encoder.set_buffer(2, Some(&buf_grad_input), 0);
            encoder.set_buffer(3, Some(&buf_params), 0);

            // grid: (input_w, input_h, batch_size * in_channels)
            let grid = MTLSize::new(
                input_w as u64,
                input_h as u64,
                (batch_size * in_channels) as u64,
            );
            let tw = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
            let group = MTLSize::new(tw.min(input_w as u64), 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = buf_grad_input.contents() as *const f32;
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, grad_input.as_mut_ptr(), input_size);
            }
        }

        // ── 2. Backward filter gradient ─────────────────────────────────
        {
            let pipeline = self.get_pipeline("conv2d_backward_filters")?;
            let buf_grad_filters = self.create_empty_buffer(filter_size);

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&buf_input), 0);
            encoder.set_buffer(1, Some(&buf_grad_output), 0);
            encoder.set_buffer(2, Some(&buf_grad_filters), 0);
            encoder.set_buffer(3, Some(&buf_params), 0);

            // grid: (kernel_w, kernel_h, out_channels * in_channels)
            let grid = MTLSize::new(
                kernel_w as u64,
                kernel_h as u64,
                (out_channels * in_channels) as u64,
            );
            let tw = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
            let group = MTLSize::new(tw.min(kernel_w as u64), 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = buf_grad_filters.contents() as *const f32;
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, grad_filters.as_mut_ptr(), filter_size);
            }
        }

        // ── 3. Backward bias gradient ───────────────────────────────────
        {
            let pipeline = self.get_pipeline("conv2d_backward_bias")?;
            let buf_grad_bias = self.create_empty_buffer(out_channels);

            // bias params: [batch_size, out_channels, out_h, out_w]
            let bias_params: [u32; 4] = [
                batch_size as u32,
                out_channels as u32,
                out_h as u32,
                out_w as u32,
            ];
            let buf_bias_params = self.device.new_buffer_with_data(
                bias_params.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(&bias_params) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&buf_grad_output), 0);
            encoder.set_buffer(1, Some(&buf_grad_bias), 0);
            encoder.set_buffer(2, Some(&buf_bias_params), 0);

            let tw = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
            let grid = MTLSize::new(out_channels as u64, 1, 1);
            let group = MTLSize::new(tw.min(out_channels as u64), 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = buf_grad_bias.contents() as *const f32;
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, grad_bias.as_mut_ptr(), out_channels);
            }
        }

        Ok(())
    }
}
