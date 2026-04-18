use super::MetalBackend;
use crate::gpu::backend::GpuError;
use metal::{MTLResourceOptions, MTLSize};

impl MetalBackend {
    pub(super) fn relu_impl(&self, data: &mut [f32]) -> Result<(), GpuError> {
        let len = data.len();
        if len == 0 {
            return Ok(());
        }

        let pipeline = self.get_pipeline("relu")?;
        let buf_data = self.create_buffer(data);
        let params: [u32; 1] = [len as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_data), 0);
        encoder.set_buffer(1, Some(&buf_params), 0);

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid = MTLSize::new(len as u64, 1, 1);
        let group = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_data.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), len);
        }
        Ok(())
    }

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
        if grad_output.len() < len || grad_input.len() < len {
            return Err(GpuError::DimensionMismatch(
                "relu_backward: input, grad_output, and grad_input must have the same length"
                    .into(),
            ));
        }

        let pipeline = self.get_pipeline("relu_backward")?;
        let buf_input = self.create_buffer(input);
        let buf_grad_output = self.create_buffer(grad_output);
        let buf_grad_input = self.create_buffer(grad_input);
        let params: [u32; 1] = [len as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_input), 0);
        encoder.set_buffer(1, Some(&buf_grad_output), 0);
        encoder.set_buffer(2, Some(&buf_grad_input), 0);
        encoder.set_buffer(3, Some(&buf_params), 0);

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid = MTLSize::new(len as u64, 1, 1);
        let group = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_grad_input.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, grad_input.as_mut_ptr(), len);
        }
        Ok(())
    }

    pub(super) fn sigmoid_impl(&self, data: &mut [f32]) -> Result<(), GpuError> {
        let len = data.len();
        if len == 0 {
            return Ok(());
        }

        let pipeline = self.get_pipeline("sigmoid")?;
        let buf_data = self.create_buffer(data);
        let params: [u32; 1] = [len as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_data), 0);
        encoder.set_buffer(1, Some(&buf_params), 0);

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid = MTLSize::new(len as u64, 1, 1);
        let group = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_data.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), len);
        }
        Ok(())
    }

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
        if grad_output.len() < len || grad_input.len() < len {
            return Err(GpuError::DimensionMismatch(
                "sigmoid_backward: sigmoid_output, grad_output, and grad_input must have the same length".into(),
            ));
        }

        let pipeline = self.get_pipeline("sigmoid_backward")?;
        let buf_sigmoid = self.create_buffer(sigmoid_output);
        let buf_grad_output = self.create_buffer(grad_output);
        let buf_grad_input = self.create_buffer(grad_input);
        let params: [u32; 1] = [len as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_sigmoid), 0);
        encoder.set_buffer(1, Some(&buf_grad_output), 0);
        encoder.set_buffer(2, Some(&buf_grad_input), 0);
        encoder.set_buffer(3, Some(&buf_params), 0);

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid = MTLSize::new(len as u64, 1, 1);
        let group = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_grad_input.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, grad_input.as_mut_ptr(), len);
        }
        Ok(())
    }

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

        let pipeline = self.get_pipeline("add_bias")?;
        let buf_data = self.create_buffer(data);
        let buf_bias = self.create_buffer(bias);
        let params: [u32; 2] = [batch_size as u32, n as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_data), 0);
        encoder.set_buffer(1, Some(&buf_bias), 0);
        encoder.set_buffer(2, Some(&buf_params), 0);

        let grid = MTLSize::new(n as u64, batch_size as u64, 1);
        let tw = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let group = MTLSize::new(tw.min(n as u64), 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_data.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), batch_size * n);
        }
        Ok(())
    }

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

        let pipeline = self.get_pipeline("sum_rows")?;
        let buf_data = self.create_buffer(data);
        let buf_out = self.create_buffer(out);
        let params: [u32; 2] = [batch_size as u32, n as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(&params) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_data), 0);
        encoder.set_buffer(1, Some(&buf_out), 0);
        encoder.set_buffer(2, Some(&buf_params), 0);

        let threads_per_group = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(threads_per_group.min(n as u64), 1, 1);
        encoder.dispatch_threads(grid, group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = buf_out.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Ok(())
    }
}
