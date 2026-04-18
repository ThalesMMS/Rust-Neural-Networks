//! CUDA GPU backend for NVIDIA GPUs.
//!
//! This module implements the [`GpuBackend`] trait using NVIDIA's CUDA toolkit
//! via the `cudarc` crate, providing GPU-accelerated matrix operations,
//! element-wise activations, and convolution kernels on NVIDIA GPUs.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{result, CudaContext, CudaFunction, CudaModule, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::gpu::backend::{BackendType, GpuBackend, GpuDevice, GpuError};

/// CUDA kernel sources compiled at runtime via NVRTC.
const ELEMENTWISE_KERNEL_SRC: &str = include_str!("../shaders/elementwise.cu");
const CONV2D_KERNEL_SRC: &str = include_str!("../shaders/conv2d.cu");

/// CUDA-based GPU backend for neural network operations.
///
/// Discovers the first CUDA-capable GPU, creates a CUDA context via `cudarc`,
/// initialises a cuBLAS handle for matrix operations, and compiles PTX kernels
/// for element-wise and convolution operations.
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::gpu::cuda_backend::CudaBackend;
///
/// let backend = CudaBackend::new()?;
/// println!("GPU: {}", backend.device_info().name);
/// ```
pub struct CudaBackend {
    /// CUDA context wrapping the selected GPU device.
    ctx: Arc<CudaContext>,
    /// cuBLAS handle for matrix multiply operations.
    blas: CudaBlas,
    /// Compiled element-wise kernel module.
    elementwise_module: Arc<CudaModule>,
    /// Compiled conv2d kernel module.
    conv2d_module: Arc<CudaModule>,
    /// Cached device information.
    device_info: GpuDevice,
}

impl CudaBackend {
    /// Creates a new CUDA backend by selecting the first CUDA-capable NVIDIA GPU,
    /// initializing cuBLAS, and compiling/loading the elementwise and conv2d kernels.
    ///
    /// The returned backend is ready to run CUDA-accelerated operations (SGEMM,
    /// elementwise kernels, and conv2d) using the compiled modules and a default
    /// CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DeviceNotFound` if no CUDA-capable GPU is available or a
    /// CUDA context cannot be created.
    ///
    /// Returns `GpuError::KernelError` if NVRTC compilation of the elementwise or
    /// conv2d kernels fails, or if a compiled module cannot be loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let backend = CudaBackend::new().expect("CUDA backend initialization failed");
    /// // Use backend for GPU operations...
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        let ctx = CudaContext::new(0).map_err(|e| {
            GpuError::DeviceNotFound(format!("Failed to create CUDA context: {}", e))
        })?;

        let stream = ctx.default_stream();

        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            GpuError::OperationFailed(format!("Failed to create cuBLAS handle: {}", e))
        })?;

        // Compile kernels via NVRTC
        let elementwise_ptx = compile_ptx(ELEMENTWISE_KERNEL_SRC).map_err(|e| {
            GpuError::KernelError(format!("Failed to compile elementwise kernels: {}", e))
        })?;
        let conv2d_ptx = compile_ptx(CONV2D_KERNEL_SRC).map_err(|e| {
            GpuError::KernelError(format!("Failed to compile conv2d kernels: {}", e))
        })?;

        let elementwise_module = ctx.load_module(elementwise_ptx).map_err(|e| {
            GpuError::KernelError(format!("Failed to load elementwise module: {}", e))
        })?;
        let conv2d_module = ctx
            .load_module(conv2d_ptx)
            .map_err(|e| GpuError::KernelError(format!("Failed to load conv2d module: {}", e)))?;

        let cu_device = result::device::get(0).map_err(|e| {
            GpuError::OperationFailed(format!("Failed to query CUDA device handle: {}", e))
        })?;
        let device_name = result::device::get_name(cu_device).map_err(|e| {
            GpuError::OperationFailed(format!("Failed to query CUDA device name: {}", e))
        })?;
        let memory_bytes = unsafe { result::device::total_mem(cu_device) }.map_err(|e| {
            GpuError::OperationFailed(format!("Failed to query CUDA device memory: {}", e))
        })? as u64;

        let device_info = GpuDevice {
            name: device_name,
            backend: BackendType::Cuda,
            memory_bytes: Some(memory_bytes),
            is_supported: true,
        };

        Ok(Self {
            ctx,
            blas,
            elementwise_module,
            conv2d_module,
            device_info,
        })
    }

    /// Accesses the backend's default CUDA stream.
    ///
    /// Returns an `Arc` pointing to the context's default `CudaStream`, suitable for kernel launches and cuBLAS operations.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = CudaBackend::new().unwrap();
    /// let stream = backend.stream();
    /// // `stream` can be cloned and shared for submissions to the CUDA context.
    /// assert!(std::sync::Arc::strong_count(&stream) >= 1);
    /// ```
    fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }

    /// Loads an elementwise CUDA kernel by symbol name from the compiled elementwise module.
    ///
    /// # Parameters
    /// - `name`: Kernel function symbol name to load.
    ///
    /// # Returns
    /// The loaded `CudaFunction` on success.
    ///
    /// # Errors
    /// Returns `GpuError::KernelError` if the function symbol cannot be found or loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// // `backend` is a `CudaBackend`
    /// let relu = backend.elementwise_fn("relu").expect("failed to load relu kernel");
    /// ```
    fn elementwise_fn(&self, name: &str) -> Result<CudaFunction, GpuError> {
        self.elementwise_module
            .load_function(name)
            .map_err(|e| GpuError::KernelError(format!("Function '{}': {}", name, e)))
    }

    /// Load a kernel function named `name` from the compiled conv2d CUDA module.
    ///
    /// # Returns
    ///
    /// The loaded `CudaFunction`.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::KernelError` if the named function cannot be found or loaded from the module.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let backend = CudaBackend::new().unwrap();
    /// let func = backend.conv2d_fn("conv2d_forward").unwrap();
    /// ```
    fn conv2d_fn(&self, name: &str) -> Result<CudaFunction, GpuError> {
        self.conv2d_module
            .load_function(name)
            .map_err(|e| GpuError::KernelError(format!("Function '{}': {}", name, e)))
    }

    /// Computes a `LaunchConfig` for a 1D kernel that covers `n` threads.
    ///
    /// The returned `LaunchConfig` uses a fixed block size of 256 threads and a grid size
    /// computed by ceiling division so that `grid_dim.x * block_dim.x >= n`. `shared_mem_bytes`
    /// is set to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// let cfg = launch_cfg_1d(1000);
    /// // block_dim.x == 256 and grid_dim.x * block_dim.x >= 1000
    /// assert_eq!(cfg.block_dim.0, 256);
    /// assert!(cfg.grid_dim.0 * cfg.block_dim.0 >= 1000);
    /// ```
    fn launch_cfg_1d(n: usize) -> LaunchConfig {
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (grid, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

mod conv2d;
mod elementwise;
mod matrix;

impl GpuBackend for CudaBackend {
    /// Accesses the cached GPU device metadata for this backend.
    ///
    /// Returns a reference to the stored `GpuDevice` describing the backend.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use crate::gpu::cuda_backend::CudaBackend;
    /// # use crate::gpu::GpuBackend;
    /// # fn example(backend: &CudaBackend) {
    /// let info = backend.device_info();
    /// let _name = &info.name;
    /// # }
    /// ```
    fn device_info(&self) -> &GpuDevice {
        &self.device_info
    }

    /// Performs single-precision general matrix-matrix multiplication (SGEMM): C := A * B + C.
    ///
    /// - `m`: number of rows of A and C.
    /// - `n`: number of columns of B and C.
    /// - `k`: number of columns of A and rows of B.
    /// - `a`: row-major slice containing A with length `m * k`.
    /// - `b`: row-major slice containing B with length `k * n`.
    /// - `c`: row-major slice containing C with length `m * n`; updated in place with the result.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError)` if the underlying GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use your_crate::gpu::cuda_backend::CudaBackend;
    /// # use your_crate::gpu::GpuError;
    /// fn example(backend: &CudaBackend) -> Result<(), GpuError> {
    ///     let m = 2usize;
    ///     let k = 3usize;
    ///     let n = 2usize;
    ///     let a = vec![1.0_f32, 2.0, 3.0,   // row 0
    ///                  4.0, 5.0, 6.0];      // row 1  (2x3)
    ///     let b = vec![7.0_f32, 8.0,        // row 0
    ///                  9.0, 10.0,          // row 1
    ///                  11.0, 12.0];        // row 2  (3x2)
    ///     let mut c = vec![0.0_f32; m * n]; // 2x2
    ///     backend.sgemm(m, n, k, &a, &b, &mut c)?;
    ///     Ok(())
    /// }
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
        self.sgemm_impl(m, n, k, a, b, c)
    }

    /// Computes single-precision matrix multiplication where A is transposed (Aᵀ) and B is not, writing the result into `c`.
    ///
    /// The operation computes C := Aᵀ * B and stores the m-by-n result in `c`.
    /// - `m`, `n`, `k` are the output and reduction dimensions such that A has shape (k, m) before transpose, B has shape (k, n), and C has shape (m, n).
    /// - `a` and `b` are input slices containing `k * m` and `k * n` elements respectively.
    /// - `c` is an output slice containing `m * n` elements which will be overwritten with the result.
    ///
    /// # Examples
    ///
    /// ```
    /// // Compute C = Aᵀ * B with m=2, n=2, k=3
    /// let m = 2;
    /// let n = 2;
    /// let k = 3;
    /// // A is k x m = 3 x 2 (column-major view not required here; layout is row-major linear)
    /// let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
    /// // B is k x n = 3 x 2
    /// let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
    /// let mut c = vec![0.0f32; m * n];
    /// // backend.sgemm_at(m, n, k, &a, &b, &mut c).unwrap();
    /// // After a successful call, `c` will contain the 2x2 result of Aᵀ * B.
    /// ```
    ///
    /// @returns `Ok(())` on success, or an `Err(GpuError)` if the GPU operation or transfer fails.
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

    /// Performs single-precision general matrix multiplication where the second operand is transposed.
    ///
    /// Multiplies A (m × k) by B^T (k × n) and writes the result into C (m × n). All matrices are
    /// expected in row-major order: `a.len() == m * k`, `b.len() == n * k`, and `c.len() == m * n`.
    ///
    /// # Errors
    ///
    /// Returns a `GpuError` if the underlying CUDA/cuBLAS operation fails or if kernel dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let backend = CudaBackend::new().unwrap();
    /// let m = 2;
    /// let n = 3;
    /// let k = 4;
    /// let a = vec![1f32; m * k]; // A: 2x4
    /// let b = vec![2f32; n * k]; // B: 3x4 (will be used as B^T)
    /// let mut c = vec![0f32; m * n]; // C: 2x3
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
        self.sgemm_bt_impl(m, n, k, a, b, c)
    }

    /// Applies the ReLU activation in-place to the provided slice of floats.
    ///
    /// The slice is modified so that every negative value becomes `0.0` while non-negative values are left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut data = [-1.0f32, 0.5, -0.2];
    /// let backend = CudaBackend::new().unwrap();
    /// backend.relu(&mut data).unwrap();
    /// assert_eq!(data, [0.0, 0.5, 0.0]);
    /// ```
    fn relu(&self, data: &mut [f32]) -> Result<(), GpuError> {
        self.relu_impl(data)
    }

    /// Computes the element-wise backward pass for a ReLU activation and writes the result into `grad_input`.
    ///
    /// For each index i, `grad_input[i]` is set to `grad_output[i]` when `input[i] > 0.0`, otherwise `0.0`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an `Err(GpuError)` if the GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Given a `CudaBackend` instance `backend`, compute ReLU backward:
    /// let input = vec![1.0_f32, -0.5, 2.0];
    /// let grad_output = vec![0.1_f32, 0.2, 0.3];
    /// let mut grad_input = vec![0.0_f32; input.len()];
    /// backend.relu_backward(&input, &grad_output, &mut grad_input).unwrap();
    /// // grad_input == [0.1, 0.0, 0.3]
    /// ```
    fn relu_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        self.relu_backward_impl(input, grad_output, grad_input)
    }

    /// Applies the sigmoid activation function elementwise to `data` in place.
    ///
    /// The operation is performed by the backend and may offload computation to the GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut buf = [0.0f32, 1.0, -1.0];
    /// // let backend = CudaBackend::new().unwrap();
    /// // backend.sigmoid(&mut buf).unwrap();
    /// ```
    ///
    /// Returns `Ok(())` on success, `Err(GpuError)` if the operation fails.
    fn sigmoid(&self, data: &mut [f32]) -> Result<(), GpuError> {
        self.sigmoid_impl(data)
    }

    /// Computes the gradient of the sigmoid activation and writes the per-element results into `grad_input`.
    ///
    /// The function expects three slices of equal length: `sigmoid_output` (the forward sigmoid outputs),
    /// `grad_output` (the gradient w.r.t. the sigmoid outputs), and `grad_input` (destination for the computed gradients).
    /// On success, `grad_input[i]` will contain the gradient for the corresponding element.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError)` if the underlying GPU operation fails (e.g., kernel launch or resource error).
    ///
    /// # Examples
    ///
    /// ```
    /// let sigmoid_output = vec![0.5f32, 0.8, 0.2];
    /// let grad_output = vec![1.0f32, 0.5, -0.3];
    /// let mut grad_input = vec![0.0f32; 3];
    /// // assume `backend` is a ready CudaBackend
    /// // backend.sigmoid_backward(&sigmoid_output, &grad_output, &mut grad_input).unwrap();
    /// // After call, grad_input contains the per-element sigmoid gradients.
    /// ```
    fn sigmoid_backward(
        &self,
        sigmoid_output: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError> {
        self.sigmoid_backward_impl(sigmoid_output, grad_output, grad_input)
    }

    /// Adds a bias vector to each row of a batched matrix stored in `data`, modifying `data` in-place.
    ///
    /// `data` is interpreted as `batch_size` consecutive rows each of length `n` (row-major).
    /// The `bias` slice must have length `n`; its j-th element is added to every row's j-th column.
    ///
    /// # Parameters
    ///
    /// - `data`: Mutable slice containing `batch_size * n` elements arranged as `batch_size` rows of `n` columns.
    /// - `bias`: Slice of length `n` whose values are added to each row of `data`.
    /// - `batch_size`: Number of rows in the batch.
    /// - `n`: Number of columns per row; must equal `bias.len()`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError)` if the operation fails (e.g., kernel launch or validation error).
    ///
    /// # Examples
    ///
    /// ```
    /// // layout: 2 rows (batch_size = 2) × 3 columns (n = 3)
    /// let mut data = vec![1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0];
    /// let bias = vec![0.5f32, 1.0, -1.0];
    /// // After adding bias: each row has bias added elementwise
    /// // row0: [1.5, 3.0, 2.0], row1: [4.5, 6.0, 5.0]
    /// // (The actual call would be `backend.add_bias(&mut data, &bias, 2, 3)`)
    /// for row in 0..2 {
    ///     for col in 0..3 {
    ///         let idx = row * 3 + col;
    ///         data[idx] += bias[col];
    ///     }
    /// }
    /// assert_eq!(data, vec![1.5f32, 3.0, 2.0, 4.5, 6.0, 5.0]);
    /// ```
    fn add_bias(
        &self,
        data: &mut [f32],
        bias: &[f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        self.add_bias_impl(data, bias, batch_size, n)
    }

    /// Compute per-column sums across a batch of row vectors.
    ///
    /// The input `data` is interpreted as `batch_size` rows of length `n` in row-major order
    /// (i.e., element (i, j) is at index `i * n + j`). The result is written into `out`,
    /// which must have length `n`; after the call `out[j]` equals the sum of `data[i * n + j]`
    /// for `i` in `0..batch_size`.
    ///
    /// # Parameters
    ///
    /// - `data`: Flattened batch of row vectors in row-major order.
    /// - `out`: Mutable slice of length `n` receiving per-column sums.
    /// - `batch_size`: Number of rows in the batch.
    /// - `n`: Number of columns (length of each row).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an `Err(GpuError)` if the backend operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Given a backend `b`, two rows [1,2,3] and [4,5,6]:
    /// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut out = [0.0f32; 3];
    /// // b.sum_rows(&data, &mut out, 2, 3)?;
    /// // After call: out == [5.0, 7.0, 9.0]
    /// ```
    fn sum_rows(
        &self,
        data: &[f32],
        out: &mut [f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        self.sum_rows_impl(data, out, batch_size, n)
    }

    /// Computes the forward pass of a 2D convolution and writes the result into `output`.
    ///
    /// Expects `input` to be laid out as NCHW (batch_size, in_channels, input_h, input_w)
    /// and `filters` as (out_channels, in_channels, kernel_h, kernel_w). `bias` must have
    /// length `out_channels`. `output` is written in NCHW layout with spatial dimensions
    /// computed as:
    /// output_h = (input_h + 2*padding - kernel_h) / stride + 1
    /// output_w = (input_w + 2*padding - kernel_w) / stride + 1
    ///
    /// # Parameters
    ///
    /// - `input`: Flattened input tensor in NCHW order.
    /// - `filters`: Flattened convolution kernels.
    /// - `bias`: Per-output-channel bias values.
    /// - `output`: Destination buffer for the convolution result (NCHW layout).
    /// - `batch_size`, `in_channels`, `out_channels`: Tensor channel and batch dimensions.
    /// - `input_h`, `input_w`: Height and width of the input feature maps.
    /// - `kernel_h`, `kernel_w`: Height and width of the convolution kernel.
    /// - `stride`: Stride applied to both spatial dimensions.
    /// - `padding`: Zero-padding applied equally to both spatial dimensions.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the operation was launched successfully on the GPU, `Err(GpuError)` if
    /// an error occurred preparing or launching the kernel.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use my_crate::gpu::{CudaBackend, GpuBackend};
    /// // Prepare shapes and buffers (filled with valid floats)
    /// let batch = 1;
    /// let in_ch = 3;
    /// let out_ch = 8;
    /// let ih = 32;
    /// let iw = 32;
    /// let kh = 3;
    /// let kw = 3;
    /// let stride = 1;
    /// let padding = 1;
    /// let input = vec![0.0f32; batch * in_ch * ih * iw];
    /// let filters = vec![0.0f32; out_ch * in_ch * kh * kw];
    /// let bias = vec![0.0f32; out_ch];
    /// let oh = (ih + 2*padding - kh) / stride + 1;
    /// let ow = (iw + 2*padding - kw) / stride + 1;
    /// let mut output = vec![0.0f32; batch * out_ch * oh * ow];
    ///
    /// let backend = CudaBackend::new().unwrap();
    /// backend.conv2d_forward(
    ///     &input,
    ///     &filters,
    ///     &bias,
    ///     &mut output,
    ///     batch,
    ///     in_ch,
    ///     out_ch,
    ///     ih,
    ///     iw,
    ///     kh,
    ///     kw,
    ///     stride,
    ///     padding,
    /// ).unwrap();
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

    /// Computes gradients for a 2D convolution layer (backward pass) and writes them into the provided gradient buffers.
    ///
    /// This computes:
    /// - `grad_input`: gradient with respect to the input, same shape and layout as `input`.
    /// - `grad_filters`: gradient with respect to the convolution filters, same shape and layout as `filters`.
    /// - `grad_bias`: gradient with respect to the per-output-channel bias (length `out_channels`).
    ///
    /// Parameters use NCHW layout:
    /// - `input`: shape (batch_size, in_channels, input_h, input_w).
    /// - `filters`: shape (out_channels, in_channels, kernel_h, kernel_w).
    /// - `grad_output`: shape (batch_size, out_channels, out_h, out_w), where out_h/out_w are determined by input size, kernel size, stride, and padding.
    /// - `grad_input`: output buffer shaped like `input` (written to).
    /// - `grad_filters`: output buffer shaped like `filters` (written to).
    /// - `grad_bias`: output buffer of length `out_channels` (written to).
    /// - `stride`: convolution stride (pixels).
    /// - `padding`: zero-padding applied to input (pixels).
    ///
    /// All buffers must be suitably sized for the described layouts. The method delegates computation to the CUDA backend and will return an error if GPU execution or resource setup fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use my_crate::gpu::CudaBackend;
    /// # use my_crate::gpu::GpuBackend;
    /// let backend = CudaBackend::new().unwrap();
    /// let batch = 1;
    /// let in_c = 1;
    /// let out_c = 1;
    /// let ih = 5;
    /// let iw = 5;
    /// let kh = 3;
    /// let kw = 3;
    /// let stride = 1;
    /// let padding = 0;
    /// let input = vec![0.0f32; batch * in_c * ih * iw];
    /// let filters = vec![0.0f32; out_c * in_c * kh * kw];
    /// let grad_output = vec![0.0f32; batch * out_c * ( (ih + 2*padding - kh)/stride + 1 ) * ( (iw + 2*padding - kw)/stride + 1 )];
    /// let mut grad_input = vec![0.0f32; input.len()];
    /// let mut grad_filters = vec![0.0f32; filters.len()];
    /// let mut grad_bias = vec![0.0f32; out_c];
    /// backend.conv2d_backward(
    ///     &input,
    ///     &filters,
    ///     &grad_output,
    ///     &mut grad_input,
    ///     &mut grad_filters,
    ///     &mut grad_bias,
    ///     batch,
    ///     in_c,
    ///     out_c,
    ///     ih,
    ///     iw,
    ///     kh,
    ///     kw,
    ///     stride,
    ///     padding,
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

#[cfg(test)]
mod tests;
