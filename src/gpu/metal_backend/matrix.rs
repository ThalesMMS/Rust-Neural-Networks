use super::MetalBackend;
use crate::gpu::backend::GpuError;
use metal::{MTLResourceOptions, MTLSize};

impl MetalBackend {
    /// Tile size used by the tiled sgemm shaders. Must match TILE_SIZE in sgemm.metal.
    const TILE_SIZE: usize = 16;

    /// Dispatches a tiled SGEMM kernel by name and writes the matrix product into `c`.
    ///
    /// The kernel computes C = op(A) * op(B); choose `kernel_name` to select the transpose
    /// variant: `"sgemm_nn"` (A and B not transposed), `"sgemm_tn"` (A transposed),
    /// or `"sgemm_nt"` (B transposed). The function uploads `a`, `b`, and `c` to GPU buffers,
    /// dispatches the compute kernel, waits for completion, and copies the result back into `c`.
    ///
    /// # Errors
    ///
    /// Returns an error if the named pipeline cannot be found or if GPU pipeline dispatch/compilation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Given an initialized MetalBackend `backend`, compute 2x2 matrix multiply:
    /// let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 (row-major)
    /// let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 (row-major)
    /// let mut c = vec![0.0f32; 4];
    /// // kernel_name selects transpose behavior; here neither A nor B are transposed
    /// backend.dispatch_sgemm("sgemm_nn", 2, 2, 2, &a, &b, &mut c).unwrap();
    /// // c now contains the product of A and B (row-major)
    /// ```
    fn dispatch_sgemm(
        &self,
        kernel_name: &str,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let pipeline = self.get_pipeline(kernel_name)?;

        let buf_a = self.create_buffer(a);
        let buf_b = self.create_buffer(b);
        let buf_c = self.create_buffer(c);
        let params: [u32; 3] = [m as u32, n as u32, k as u32];
        let buf_params = self.device.new_buffer_with_data(
            params.as_ptr() as *const std::ffi::c_void,
            (params.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_params), 0);

        let tile = Self::TILE_SIZE as u64;
        let grid_size = MTLSize::new(
            ((n as u64 + tile - 1) / tile) * tile,
            ((m as u64 + tile - 1) / tile) * tile,
            1,
        );
        let threadgroup_size = MTLSize::new(tile, tile, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back from GPU buffer
        let result_ptr = buf_c.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(result_ptr, c.as_mut_ptr(), m * n);
        }

        Ok(())
    }

    pub(super) fn sgemm_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                m * k,
                b.len(),
                k * n,
                c.len(),
                m * n,
            )));
        }
        self.dispatch_sgemm("sgemm_nn", m, n, k, a, b, c)
    }

    pub(super) fn sgemm_at_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        // A is stored as (k × m), transposed to (m × k)
        if a.len() < k * m || b.len() < k * n || c.len() < m * n {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm_at: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                k * m,
                b.len(),
                k * n,
                c.len(),
                m * n,
            )));
        }
        self.dispatch_sgemm("sgemm_tn", m, n, k, a, b, c)
    }

    pub(super) fn sgemm_bt_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        // B is stored as (n × k), transposed to (k × n)
        if a.len() < m * k || b.len() < n * k || c.len() < m * n {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm_bt: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                m * k,
                b.len(),
                n * k,
                c.len(),
                m * n,
            )));
        }
        self.dispatch_sgemm("sgemm_nt", m, n, k, a, b, c)
    }
}
