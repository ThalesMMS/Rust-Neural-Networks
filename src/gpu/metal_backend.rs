//! Metal GPU backend for Apple Silicon.
//!
//! This module implements the [`GpuBackend`] trait using Apple's Metal framework,
//! providing GPU-accelerated matrix operations, element-wise activations, and
//! convolution kernels on macOS devices with Metal-capable GPUs.

use std::collections::HashMap;

use metal::{
    Buffer as MetalBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

use crate::gpu::backend::{BackendType, GpuBackend, GpuDevice, GpuError};

/// Metal Shading Language source for compute kernels.
const SGEMM_SHADER: &str = include_str!("shaders/sgemm.metal");
const ELEMENTWISE_SHADER: &str = include_str!("shaders/elementwise.metal");
const CONV2D_SHADER: &str = include_str!("shaders/conv2d.metal");

/// Metal-based GPU backend for neural network operations.
///
/// Discovers the system default Metal device, compiles MSL shader sources,
/// and creates compute pipeline states for each kernel function. All GPU
/// work is dispatched through a single command queue.
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::gpu::metal_backend::MetalBackend;
///
/// let backend = MetalBackend::new()?;
/// println!("GPU: {}", backend.device_info().name);
/// ```
pub struct MetalBackend {
    /// The Metal device (GPU).
    device: Device,
    /// Command queue for submitting GPU work.
    command_queue: CommandQueue,
    /// Compiled compute pipeline states keyed by kernel function name.
    pipelines: HashMap<String, ComputePipelineState>,
    /// Cached device information.
    device_info: GpuDevice,
}

impl MetalBackend {
    /// Create a MetalBackend configured with the system default Metal device, a command queue, and compiled GPU shader pipelines.
    ///
    /// On success returns a ready-to-use MetalBackend that can dispatch compute kernels.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DeviceNotFound` if no Metal-capable GPU is available on the system.
    /// Returns `GpuError::KernelError` if shader compilation or pipeline creation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::gpu::metal_backend::MetalBackend;
    /// let backend = MetalBackend::new().expect("failed to create Metal backend");
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        let device = Device::system_default()
            .ok_or_else(|| GpuError::DeviceNotFound("No Metal-capable GPU found".into()))?;

        let command_queue = device.new_command_queue();

        let device_info = GpuDevice {
            name: device.name().to_string(),
            backend: BackendType::Metal,
            memory_bytes: Some(device.recommended_max_working_set_size()),
            is_supported: true,
        };

        let mut backend = Self {
            device,
            command_queue,
            pipelines: HashMap::new(),
            device_info,
        };

        backend.compile_shaders()?;

        Ok(backend)
    }

    /// Compiles the bundled MSL shader sources and registers their compute pipelines.
    ///
    /// This compiles the sgemm, elementwise, and conv2d shader sources into Metal libraries,
    /// creates a compute pipeline for each function, and stores the pipelines in the backend's
    /// pipeline map keyed by function name.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError::KernelError)` if shader compilation, function lookup,
    /// or pipeline creation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Construction of the backend compiles shaders; this illustrates successful creation.
    /// let _backend = MetalBackend::new().expect("Metal device and shader compilation required");
    /// ```
    fn compile_shaders(&mut self) -> Result<(), GpuError> {
        let shader_sources = [
            ("sgemm", SGEMM_SHADER),
            ("elementwise", ELEMENTWISE_SHADER),
            ("conv2d", CONV2D_SHADER),
        ];

        for (name, source) in &shader_sources {
            let library = self.compile_library(source).map_err(|e| {
                GpuError::KernelError(format!("Failed to compile {} shaders: {}", name, e))
            })?;

            let function_names = library.function_names();
            for fn_name in function_names.iter() {
                let function = library.get_function(fn_name, None).map_err(|e| {
                    GpuError::KernelError(format!("Failed to get function '{}': {}", fn_name, e))
                })?;

                let pipeline = self
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| {
                        GpuError::KernelError(format!(
                            "Failed to create pipeline for '{}': {}",
                            fn_name, e
                        ))
                    })?;

                self.pipelines.insert(fn_name.to_string(), pipeline);
            }
        }

        Ok(())
    }

    /// Compile Metal Shading Language (MSL) source into a Metal library.
    ///
    /// Attempts to compile the given MSL `source` for the backend's device and produces a
    /// `Library` usable to create compute functions and pipelines.
    ///
    /// On success returns a compiled Metal `Library`; on failure returns an error string
    /// describing the compilation failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Given a `backend: MetalBackend`
    /// let src = r#"
    /// kernel void noop(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    ///     // no-op
    /// }
    /// "#;
    ///
    /// let lib = backend.compile_library(src).expect("MSL compilation failed");
    /// assert!(lib.function_names().iter().any(|n| n.contains("noop")));
    /// ```
    fn compile_library(&self, source: &str) -> Result<Library, String> {
        let options = CompileOptions::new();
        self.device.new_library_with_source(source, &options)
    }

    /// Retrieve a compiled compute pipeline by its function name.
    ///
    /// # Returns
    ///
    /// A reference to the `ComputePipelineState` for the requested kernel.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::KernelError` if no pipeline with the given name exists.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pipeline = backend.get_pipeline("relu").unwrap();
    /// ```
    pub fn get_pipeline(&self, name: &str) -> Result<&ComputePipelineState, GpuError> {
        self.pipelines
            .get(name)
            .ok_or_else(|| GpuError::KernelError(format!("Kernel '{}' not found", name)))
    }

    /// Creates a Metal buffer containing the provided `data` in shared CPU/GPU memory.
    ///
    /// The returned buffer is initialized from `data` and uses shared storage so both CPU and GPU can access the same memory without explicit host↔device copies.
    ///
    /// # Parameters
    /// - `data`: Host slice of `f32` values used to initialize the buffer.
    ///
    /// # Returns
    /// A `MetalBuffer` wrapping the allocated buffer populated with `data`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is an instance of `MetalBackend`.
    /// let src = [1.0f32, 2.0, 3.0];
    /// let buf = backend.create_buffer(&src);
    /// // `buf` now contains the values from `src` and is accessible to GPU kernels.
    /// ```
    pub fn create_buffer(&self, data: &[f32]) -> MetalBuffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a shared Metal buffer capable of holding `len` f32 values.
    ///
    /// The returned buffer is uninitialized and uses Metal's shared CPU/GPU storage; its capacity is
    /// `len * std::mem::size_of::<f32>()` bytes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let backend = MetalBackend::new().unwrap();
    /// let buf = backend.create_empty_buffer(1024);
    /// // `buf` can be used as a device buffer for 1024 f32 elements.
    /// ```
    pub fn create_empty_buffer(&self, len: usize) -> MetalBuffer {
        self.device.new_buffer(
            (len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Accesses the backend's internal Metal command queue.
    ///
    /// # Returns
    ///
    /// A reference to the backend's `CommandQueue`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::gpu::metal_backend::MetalBackend;
    ///
    /// fn example(backend: &MetalBackend) {
    ///     let _queue = backend.command_queue();
    /// }
    /// ```
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Access the backend's Metal device.
    ///
    /// Returns a reference to the underlying Metal device for issuing commands or creating resources.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let backend = MetalBackend::new().unwrap();
    /// let device = backend.device();
    /// // use `device` to create buffers or pipelines
    /// ```
    pub fn device(&self) -> &Device {
        &self.device
    }

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
}

impl GpuBackend for MetalBackend {
    /// Access the backend's cached GPU device information.
    ///
    /// # Returns
    ///
    /// A reference to the cached `GpuDevice` containing the device name, backend type, and recommended working set size.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is an instance of `MetalBackend`
    /// let info = backend.device_info();
    /// let _name = &info.name;
    /// ```
    fn device_info(&self) -> &GpuDevice {
        &self.device_info
    }

    /// Performs single-precision matrix multiplication C = A * B for matrices with the given dimensions.
    ///
    /// A is interpreted as an m-by-k matrix, B as a k-by-n matrix, and C as an m-by-n matrix. All matrices are provided as contiguous slices in row-major order; lengths must be exactly m*k, k*n, and m*n respectively.
    ///
    /// # Arguments
    ///
    /// * `m` - number of rows in A and C.
    /// * `n` - number of columns in B and C.
    /// * `k` - number of columns in A and rows in B.
    /// * `a` - elements of A in row-major order (length m * k).
    /// * `b` - elements of B in row-major order (length k * n).
    /// * `c` - output buffer for C in row-major order (length m * n); overwritten with the result.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError::DimensionMismatch(..))` if any input slice length is smaller than required, or another `GpuError` if GPU pipeline dispatch or execution fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // A (2x3) * B (3x2) = C (2x2)
    /// let m = 2usize;
    /// let k = 3usize;
    /// let n = 2usize;
    /// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2*3
    /// let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3*2
    /// let mut c = vec![0.0f32; m * n];
    /// backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();
    /// ```
    fn sgemm(
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

    /// Performs single-precision matrix multiplication with A transposed: computes C += A^T * B.
    ///
    /// A is provided in memory as shape (k × m) (i.e. stored transposed) and is treated as m × k
    /// for the multiplication. B is shape (k × n) and C is shape (m × n).
    ///
    /// Returns `Ok(())` on success. Returns `Err(GpuError::DimensionMismatch)` if any input or output
    /// slice is smaller than the expected size (A: k*m, B: k*n, C: m*n), or another `GpuError` if
    /// the underlying kernel dispatch or pipeline lookup fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Compute C += A^T * B with m=2, n=3, k=4:
    /// let backend = MetalBackend::new().unwrap();
    /// let m = 2usize;
    /// let n = 3usize;
    /// let k = 4usize;
    /// // A stored as (k × m) => length k*m = 8
    /// let a: Vec<f32> = vec![0.0; k * m];
    /// let b: Vec<f32> = vec![0.0; k * n];
    /// let mut c: Vec<f32> = vec![0.0; m * n];
    /// backend.sgemm_at(m, n, k, &a, &b, &mut c).unwrap();
    /// ```
    fn sgemm_at(
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

    /// Performs single-precision matrix multiplication where B is supplied in transposed layout.
    ///
    /// A is interpreted as an m×k matrix stored in row-major order. B must be supplied as an n×k
    /// buffer (i.e., each row of `b` corresponds to a column of the logical k×n matrix). The result
    /// A × B is written into `c`, which is treated as an m×n row-major output buffer.
    ///
    /// # Parameters
    ///
    /// - `m`: number of rows in A and C.
    /// - `n`: number of columns in B and C.
    /// - `k`: number of columns in A and rows in the logical B.
    /// - `a`: input buffer containing A with length at least `m * k` (row-major).
    /// - `b`: input buffer containing B in n×k layout with length at least `n * k` (transposed relative to k×n).
    /// - `c`: output buffer for C with length at least `m * n` (row-major).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. Returns `Err(GpuError::DimensionMismatch)` if any buffer is too small,
    /// or another `GpuError` if kernel compilation/dispatch or GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Assume `backend` implements the MetalBackend/GpuBackend interface.
    /// let m = 2;
    /// let n = 3;
    /// let k = 4;
    /// let a = vec![1.0f32; m * k]; // A: 2×4
    /// let b = vec![1.0f32; n * k]; // B provided as 3×4 (transposed for kernel)
    /// let mut c = vec![0.0f32; m * n]; // C: 2×3
    /// backend.sgemm_bt(m, n, k, &a, &b, &mut c).unwrap();
    /// ```
    fn sgemm_bt(
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

    /// Applies the ReLU activation to each element of `data` in-place using the Metal GPU backend.
    ///
    /// Each element is replaced with max(0.0, x).
    ///
    /// Returns `Ok(())` on success, or an `Err(GpuError)` if a GPU/kernel error or other failure occurs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is a configured MetalBackend instance.
    /// let mut vals = [-1.0f32, 0.0, 2.5];
    /// backend.relu(&mut vals).unwrap();
    /// assert_eq!(vals, [0.0, 0.0, 2.5]);
    /// ```
    fn relu(&self, data: &mut [f32]) -> Result<(), GpuError> {
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

    /// Computes the ReLU backward pass, writing elementwise gradients into `grad_input`.
    ///
    /// Validates that `input`, `grad_output`, and `grad_input` have the same length and performs
    /// the GPU kernel invocation that computes grad_input[i] = grad_output[i] * (input[i] > 0 ? 1 : 0).
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DimensionMismatch` if the slices do not have the same length.
    /// Returns other `GpuError` variants if the ReLU-backward kernel is not available or GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::gpu::metal_backend::MetalBackend;
    /// use rust_neural_networks::gpu::GpuBackend;
    /// // This example requires a Metal-capable device at runtime.
    /// let backend = MetalBackend::new().expect("Metal device required for example");
    /// let input = [1.0f32, -2.0, 0.5];
    /// let grad_output = [0.1f32, 0.2, 0.3];
    /// let mut grad_input = [0.0f32; 3];
    ///
    /// backend.relu_backward(&input, &grad_output, &mut grad_input).unwrap();
    ///
    /// assert_eq!(grad_input, [0.1, 0.0, 0.3]);
    /// ```
    fn relu_backward(
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

    /// Applies the element-wise sigmoid function to `data` in place.
    ///
    /// Performs a no-op when `data` is empty. On success the slice is overwritten so that
    /// each element becomes sigmoid(x) = 1 / (1 + e^-x); on failure an error is returned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is a configured MetalBackend instance.
    /// let mut values = [0.0f32, 1.0, -1.0];
    /// backend.sigmoid(&mut values).unwrap();
    /// // After a successful call: values ~= [0.5, 0.7310586, 0.26894143]
    /// ```
    ///
    /// On success returns `Ok(())`, on failure returns `Err(GpuError)`.
    fn sigmoid(&self, data: &mut [f32]) -> Result<(), GpuError> {
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

    /// Computes gradients for a sigmoid activation and writes them into `grad_input`.
    ///
    /// The gradient is computed element-wise as grad_input[i] = grad_output[i] * sigmoid_output[i] * (1.0 - sigmoid_output[i]).
    /// Returns an error if the input slices do not all have the same length or if the backend cannot find or run the kernel.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DimensionMismatch` when the three buffers have differing lengths. Other GPU-related errors
    /// (e.g., missing pipeline or kernel failures) are returned as `GpuError` variants propagated from the backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is an initialized MetalBackend
    /// let sigmoid_output = vec![0.5f32, 0.8];
    /// let grad_output = vec![1.0f32, 0.5];
    /// let mut grad_input = vec![0.0f32; sigmoid_output.len()];
    /// backend.sigmoid_backward(&sigmoid_output, &grad_output, &mut grad_input).unwrap();
    /// // after call: grad_input contains element-wise gradients
    /// ```
    fn sigmoid_backward(
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

    /// Adds the bias vector to each row of `data` in-place.
    ///
    /// The `data` slice is treated as a matrix with `batch_size` rows and `n` columns;
    /// for each column `j` this adds `bias[j]` to every row element in that column.
    ///
    /// Returns `Err(GpuError::DimensionMismatch)` if `data.len() < batch_size * n` or
    /// `bias.len() < n`. If `batch_size == 0` or `n == 0` the function returns `Ok(())`
    /// without performing work. Other `GpuError` variants may be returned when acquiring
    /// or dispatching the compute kernel.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is a configured MetalBackend
    /// let mut data = vec![0.0f32; 6]; // batch_size = 2, n = 3
    /// let bias = vec![1.0f32, 2.0f32, 3.0f32];
    /// backend.add_bias(&mut data, &bias, 2, 3).unwrap();
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// ```
    fn add_bias(
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

    /// Sums each column over the batch dimension and writes the per-column sums into `out`.
    ///
    /// The input `data` is laid out as `batch_size` consecutive rows of length `n` (row-major).
    /// For each column `j` in `0..n`, the result is:
    /// `out[j] = sum_{b=0..batch_size-1} data[b * n + j]`.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::DimensionMismatch)` if `data.len() < batch_size * n` or `out.len() < n`.
    /// Other `GpuError` variants may be returned for GPU or kernel failures.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // `backend` is an initialized MetalBackend implementing the GPU backend API.
    /// let batch_size = 2;
    /// let n = 3;
    /// let data = vec![1.0f32, 2.0, 3.0,  // batch 0
    ///                 4.0, 5.0, 6.0];    // batch 1
    /// let mut out = vec![0.0f32; n];
    /// backend.sum_rows(&data, &mut out, batch_size, n).unwrap();
    /// assert_eq!(out, vec![5.0f32, 7.0, 9.0]);
    /// ```
    fn sum_rows(
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

    /// Compute a 2D convolution over `input` and write the results into `output`.
    ///
    /// Validates buffer sizes, uploads input, filter and bias data plus convolution
    /// parameters to the GPU, dispatches the `conv2d_forward` kernel, and copies the
    /// computed output back into the provided `output` slice.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DimensionMismatch` if `input`, `filters`, `bias`, or
    /// `output` are smaller than the sizes implied by the provided dimensions.
    /// Propagates other `GpuError` variants for missing pipelines or GPU dispatch/transfer failures.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let backend = MetalBackend::new().unwrap();
    /// let batch_size = 1;
    /// let in_channels = 1;
    /// let out_channels = 1;
    /// let input_h = 3;
    /// let input_w = 3;
    /// let kernel_h = 2;
    /// let kernel_w = 2;
    /// let stride = 1;
    /// let padding = 0;
    ///
    /// let input = vec![1.0f32; batch_size * in_channels * input_h * input_w];
    /// let filters = vec![1.0f32; out_channels * in_channels * kernel_h * kernel_w];
    /// let bias = vec![0.0f32; out_channels];
    /// let mut output = vec![0.0f32; batch_size * out_channels * ((input_h + 2*padding - kernel_h) / stride + 1) * ((input_w + 2*padding - kernel_w) / stride + 1)];
    ///
    /// backend.conv2d_forward(
    ///     &input,
    ///     &filters,
    ///     &bias,
    ///     &mut output,
    ///     batch_size,
    ///     in_channels,
    ///     out_channels,
    ///     input_h,
    ///     input_w,
    ///     kernel_h,
    ///     kernel_w,
    ///     stride,
    ///     padding,
    /// ).unwrap();
    ///
    /// assert_eq!(output.len(), 4);
    /// ```
    fn conv2d_forward(
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

    /// Computes gradients for a 2D convolution and writes them into the provided output slices.
    ///
    /// The function computes, in sequence, the gradient with respect to the input (`grad_input`),
    /// the filters (`grad_filters`), and the bias (`grad_bias`) using GPU kernels. Output spatial
    /// dimensions are derived from the input size, kernel size, stride, and padding; all output
    /// slices must have sufficient length for the computed sizes or the call returns
    /// `GpuError::DimensionMismatch`. Kernel or pipeline failures propagate as `GpuError::KernelError`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::gpu::{GpuBackend, metal_backend::MetalBackend};
    /// let backend = MetalBackend::new().unwrap();
    /// let batch_size = 1;
    /// let in_channels = 1;
    /// let out_channels = 1;
    /// let input_h = 5;
    /// let input_w = 5;
    /// let kernel_h = 3;
    /// let kernel_w = 3;
    /// let stride = 1;
    /// let padding = 0;
    /// let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    /// let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    /// let input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
    /// let filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
    /// let grad_output = vec![0.0f32; batch_size * out_channels * out_h * out_w];
    /// let mut grad_input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
    /// let mut grad_filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
    /// let mut grad_bias = vec![0.0f32; out_channels];
    /// backend.conv2d_backward(
    ///     &input, &filters, &grad_output,
    ///     &mut grad_input, &mut grad_filters, &mut grad_bias,
    ///     batch_size, in_channels, out_channels,
    ///     input_h, input_w, kernel_h, kernel_w,
    ///     stride, padding,
    /// ).unwrap();
    /// ```
    fn conv2d_backward(
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