//! Metal GPU backend for Apple Silicon.
//!
//! This module implements the [`GpuBackend`] trait using Apple's Metal framework,
//! providing GPU-accelerated matrix operations, element-wise activations, and
//! convolution kernels on macOS devices with Metal-capable GPUs.

use std::collections::HashMap;

use metal::{
    Buffer as MetalBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions,
};

use crate::gpu::backend::{BackendType, GpuBackend, GpuDevice, GpuError};

/// Metal Shading Language source for compute kernels.
const SGEMM_SHADER: &str = include_str!("../shaders/sgemm.metal");
const ELEMENTWISE_SHADER: &str = include_str!("../shaders/elementwise.metal");
const CONV2D_SHADER: &str = include_str!("../shaders/conv2d.metal");

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
}

mod conv2d;
mod elementwise;
mod matrix;

impl GpuBackend for MetalBackend {
    fn device_info(&self) -> &GpuDevice {
        &self.device_info
    }

    fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        self.sgemm_impl(m, n, k, a, b, c)
    }

    fn sgemm_at(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        self.sgemm_at_impl(m, n, k, a, b, c)
    }

    fn sgemm_bt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        self.sgemm_bt_impl(m, n, k, a, b, c)
    }

    fn relu(&self, data: &mut [f32]) -> Result<(), GpuError> {
        self.relu_impl(data)
    }

    fn relu_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        self.relu_backward_impl(input, grad_output, grad_input)
    }

    fn sigmoid(&self, data: &mut [f32]) -> Result<(), GpuError> {
        self.sigmoid_impl(data)
    }

    fn sigmoid_backward(
        &self,
        sigmoid_output: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        self.sigmoid_backward_impl(sigmoid_output, grad_output, grad_input)
    }

    fn add_bias(
        &self,
        data: &mut [f32],
        bias: &[f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        self.add_bias_impl(data, bias, batch_size, n)
    }

    fn sum_rows(
        &self,
        data: &[f32],
        out: &mut [f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        self.sum_rows_impl(data, out, batch_size, n)
    }

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
        self.conv2d_forward_impl(
            input,
            filters,
            bias,
            output,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
    }

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
        self.conv2d_backward_impl(
            input,
            filters,
            grad_output,
            grad_input,
            grad_filters,
            grad_bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
    }
}
