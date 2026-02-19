//! CUDA GPU backend for NVIDIA GPUs.
//!
//! This module implements the [`GpuBackend`] trait using NVIDIA's CUDA toolkit
//! via the `cudarc` crate, providing GPU-accelerated matrix operations,
//! element-wise activations, and convolution kernels on NVIDIA GPUs.

use std::sync::Arc;

use cudarc::cublas::{sys, CudaBlas, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::gpu::backend::{BackendType, GpuBackend, GpuDevice, GpuError};

/// CUDA kernel sources compiled at runtime via NVRTC.
const ELEMENTWISE_KERNEL_SRC: &str = r#"
extern "C" __global__ void relu(float *data, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

extern "C" __global__ void relu_backward(const float *input, const float *grad_output,
                                          float *grad_input, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
}

extern "C" __global__ void sigmoid(float *data, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

extern "C" __global__ void sigmoid_backward(const float *sigmoid_output,
                                             const float *grad_output,
                                             float *grad_input, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid_output[i];
        grad_input[i] = grad_output[i] * s * (1.0f - s);
    }
}

extern "C" __global__ void add_bias(float *data, const float *bias, int batch_size, int n) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < n) {
        data[row * n + col] += bias[col];
    }
}

extern "C" __global__ void sum_rows(const float *data, float *out, int batch_size, int n) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += data[b * n + col];
        }
        out[col] = sum;
    }
}
"#;

const CONV2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void conv2d_forward(
    const float *input, const float *filters, const float *bias, float *output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int b = idx / out_channels;
    int oc = idx % out_channels;

    if (ow >= out_w || oh >= out_h || b >= batch_size) return;

    float sum = bias[oc];
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    float iv = input[((b * in_channels + ic) * input_h + ih) * input_w + iw];
                    float fv = filters[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                    sum += iv * fv;
                }
            }
        }
    }
    output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
}

extern "C" __global__ void conv2d_backward_input(
    const float *grad_output, const float *filters, float *grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int iw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ih_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int b = idx / in_channels;
    int ic = idx % in_channels;

    if (iw_idx >= input_w || ih_idx >= input_h || b >= batch_size) return;

    float sum = 0.0f;
    for (int oc = 0; oc < out_channels; oc++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int oh = ih_idx + padding - kh;
                int ow = iw_idx + padding - kw;
                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        float gov = grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                        float fv = filters[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                        sum += gov * fv;
                    }
                }
            }
        }
    }
    grad_input[((b * in_channels + ic) * input_h + ih_idx) * input_w + iw_idx] = sum;
}

extern "C" __global__ void conv2d_backward_filters(
    const float *input, const float *grad_output, float *grad_filters,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int kw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kh_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int oc = idx / in_channels;
    int ic = idx % in_channels;

    if (kw_idx >= kernel_w || kh_idx >= kernel_h || oc >= out_channels) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int ih = oh * stride - padding + kh_idx;
                int iw = ow * stride - padding + kw_idx;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    float iv = input[((b * in_channels + ic) * input_h + ih) * input_w + iw];
                    float gov = grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                    sum += iv * gov;
                }
            }
        }
    }
    grad_filters[((oc * in_channels + ic) * kernel_h + kh_idx) * kernel_w + kw_idx] = sum;
}

extern "C" __global__ void conv2d_backward_bias(
    const float *grad_output, float *grad_bias,
    int batch_size, int out_channels, int out_h, int out_w)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_channels) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                sum += grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
            }
        }
    }
    grad_bias[oc] = sum;
}
"#;

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

        let device_info = GpuDevice {
            name: format!("CUDA Device 0"),
            backend: BackendType::Cuda,
            memory_bytes: None,
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

    /// Get the default CUDA stream associated with this backend's context.
    ///
    /// The returned `Arc<CudaStream>` is the context's default stream used for kernel launches and cuBLAS operations.
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

    /// Load an elementwise CUDA kernel by symbol name from the compiled elementwise module.
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

    /// Load a conv2d kernel function by name from the compiled conv2d CUDA module.
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

    /// Performs single-precision matrix multiplication, producing row-major C = A × B.
    ///
    /// The function adapts row-major inputs to cuBLAS's column-major convention so the device computes the correct row-major result. On success, `c` is overwritten with the product; failures return a `GpuError`.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the GEMM completed and `c` contains the result, `Err(GpuError)` on failure.
    ///
    /// # Examples
    ///
    /// ```
    /// # // Setup omitted: obtain a CudaBackend and valid cuBLAS op enums in real code.
    /// # let backend = CudaBackend::new().unwrap();
    /// # let m = 2usize; let n = 2usize; let k = 2usize;
    /// # let a = vec![1.0f32; m * k];
    /// # let b = vec![1.0f32; k * n];
    /// # let mut c = vec![0.0f32; m * n];
    /// backend
    ///     .dispatch_sgemm(
    ///         sys::cublasOperation_t::CUBLAS_OP_N,
    ///         sys::cublasOperation_t::CUBLAS_OP_N,
    ///         m,
    ///         n,
    ///         k,
    ///         &a,
    ///         &b,
    ///         &mut c,
    ///     )
    ///     .unwrap();
    /// assert_eq!(c.len(), m * n);
    /// ```
    fn dispatch_sgemm(
        &self,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let stream = self.stream();

        let a_dev = stream
            .clone_htod(a)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy A to device: {}", e)))?;
        let b_dev = stream
            .clone_htod(b)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy B to device: {}", e)))?;
        let mut c_dev = stream
            .clone_htod(c)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy C to device: {}", e)))?;

        // cuBLAS is column-major. To compute row-major C = A * B, we use:
        //   C^T = B^T * A^T
        // So we swap A and B and swap m/n, then transpose flags are inverted.
        //
        // For C = A * B (no transpose): cuBLAS does C_col = B_col * A_col with NN
        // For C = A^T * B: cuBLAS does C_col = B_col * A_col^T with N,T
        // For C = A * B^T: cuBLAS does C_col = B_col^T * A_col with T,N

        let (lda_cublas, ldb_cublas) = match (transa, transb) {
            (sys::cublasOperation_t::CUBLAS_OP_N, sys::cublasOperation_t::CUBLAS_OP_N) => (n, k),
            (sys::cublasOperation_t::CUBLAS_OP_T, sys::cublasOperation_t::CUBLAS_OP_N) => (n, k),
            (sys::cublasOperation_t::CUBLAS_OP_N, sys::cublasOperation_t::CUBLAS_OP_T) => (n, n),
            _ => (n, k),
        };

        let cfg = GemmConfig {
            transa: transb, // swap for row-major trick
            transb: transa,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: lda_cublas as i32,
            ldb: ldb_cublas as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                .map_err(|e| GpuError::OperationFailed(format!("cuBLAS sgemm failed: {}", e)))?;
        }

        stream
            .memcpy_dtoh(&c_dev, c)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy C from device: {}", e)))?;

        Ok(())
    }
}

impl GpuBackend for CudaBackend {
    /// Access the backend's cached GPU device metadata.
    ///
    /// The returned `GpuDevice` contains observed device properties such as the device
    /// name, backend type, memory hints, and the `is_supported` flag.
    ///
    /// # Returns
    ///
    /// A reference to the cached `GpuDevice`.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = backend.device_info();
    /// assert!(!info.name.is_empty());
    /// assert!(matches!(info.backend_type, crate::gpu::GpuBackendType::Cuda | crate::gpu::GpuBackendType::Other));
    /// ```
    fn device_info(&self) -> &GpuDevice {
        &self.device_info
    }

    /// Performs single-precision matrix multiplication of row-major matrices A and B and writes the result into C.
    ///
    /// The matrices are interpreted as:
    /// - A: m × k, stored row-major in `a`
    /// - B: k × n, stored row-major in `b`
    /// - C: m × n, stored row-major in `c`
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. Returns `GpuError::DimensionMismatch` if the provided slices are smaller than the expected sizes (m*k, k*n, m*n), or other `GpuError` variants on failure.
    ///
    /// # Examples
    ///
    /// ```
    /// // assuming `backend` is an initialized CUDA backend implementing this method
    /// let backend = get_cuda_backend();
    /// let a = [1.0f32, 2.0, 3.0, 4.0]; // 2×2: [[1,2],[3,4]]
    /// let b = [5.0f32, 6.0, 7.0, 8.0]; // 2×2: [[5,6],[7,8]]
    /// let mut c = [0.0f32; 4];
    /// backend.sgemm(2, 2, 2, &a, &b, &mut c).unwrap();
    /// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]); // C = A × B
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
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }

    /// Computes the single-precision matrix product C = A^T × B and writes the result into `c`.
    ///
    /// Validates that `a`, `b`, and `c` have lengths A: k×m, B: k×n, C: m×n and returns
    /// `GpuError::DimensionMismatch` when they do not. Other failures are returned as their
    /// corresponding `GpuError` variants.
    ///
    /// # Examples
    ///
    /// ```
    /// // Given a CUDA backend `backend`, compute C = A^T × B where:
    /// // A is k×m, B is k×n, and C is m×n.
    /// let m = 2usize;
    /// let n = 3usize;
    /// let k = 4usize;
    /// let a = vec![0f32; k * m]; // A (k x m)
    /// let b = vec![0f32; k * n]; // B (k x n)
    /// let mut c = vec![0f32; m * n]; // C (m x n)
    /// // backend.sgemm_at(m, n, k, &a, &b, &mut c).unwrap();
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
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }

    /// Performs matrix multiplication C = A × B^T for single-precision floats.
    ///
    /// A is interpreted as an m×k matrix stored in row-major order, B is interpreted as an n×k
    /// matrix whose transpose (k×n) will be used, and C is an m×n output matrix stored in
    /// row-major order. Validates input buffer lengths before dispatching the cuBLAS GEMM
    /// operation and returns a mapped GpuError on failure.
    ///
    /// # Arguments
    ///
    /// - `m`: number of rows in A and C.
    /// - `n`: number of columns in C (and rows in B^T).
    /// - `k`: number of columns in A and columns in B (before transposition).
    /// - `a`: flattened row-major buffer for the m×k matrix A (length >= m*k).
    /// - `b`: flattened row-major buffer for the n×k matrix B (length >= n*k); the function uses B^T.
    /// - `c`: flattened row-major buffer for the m×n output matrix C (length >= m*n).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an appropriate `Err(GpuError)` (for example `DimensionMismatch`)
    /// if input sizes are invalid or a GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // `backend` is an initialized CudaBackend implementing the same method.
    /// let m = 2;
    /// let n = 3;
    /// let k = 4;
    /// let a = vec![0f32; m * k];
    /// let b = vec![0f32; n * k];
    /// let mut c = vec![0f32; m * n];
    /// // Use the backend to compute C = A × B^T
    /// // backend.sgemm_bt(m, n, k, &a, &b, &mut c).unwrap();
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
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_T,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }

    /// Applies the ReLU activation to each element of `data` in place.
    ///
    /// The slice is mutated so that each element becomes `max(0.0, original)`.
    /// An empty slice is a no-op.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError)` if the GPU operation (transfer, kernel launch,
    /// or device→host copy) fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let backend = CudaBackend::new().unwrap();
    /// let mut vals = [-1.0f32, 0.0, 2.5];
    /// backend.relu(&mut vals).unwrap();
    /// assert_eq!(vals, [0.0, 0.0, 2.5]);
    /// ```
    fn relu(&self, data: &mut [f32]) -> Result<(), GpuError> {
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

    /// Computes the gradient of the ReLU activation with respect to its input and writes the result into `grad_input`.
    ///
    /// For each element i:
    /// `grad_input[i] = grad_output[i] if input[i] > 0, otherwise 0`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or a `GpuError` if dimensions mismatch, transfers fail, or the kernel launch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Requires a CUDA-capable device and the CUDA backend.
    /// let backend = CudaBackend::new().unwrap();
    /// let input = vec![-1.0_f32, 0.5, 2.0];
    /// let grad_output = vec![0.1_f32, 0.2, 0.3];
    /// let mut grad_input = vec![0.0_f32; input.len()];
    /// backend.relu_backward(&input, &grad_output, &mut grad_input).unwrap();
    /// assert_eq!(grad_input, vec![0.0_f32, 0.2_f32, 0.3_f32]);
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

    /// Applies the sigmoid function elementwise to `data` in place.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the operation succeeds, `Err(GpuError)` if a transfer, kernel, or launch error occurs.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let mut data = vec![-1.0_f32, 0.0, 1.0];
    /// let backend = CudaBackend::new().unwrap();
    /// backend.sigmoid(&mut data).unwrap();
    /// // `data` now contains the sigmoid of the original values.
    /// ```
    fn sigmoid(&self, data: &mut [f32]) -> Result<(), GpuError> {
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

    /// Computes the elementwise gradient of the sigmoid activation and writes it into `grad_input`.
    ///
    /// For each element i this computes:
    /// `grad_input[i] = sigmoid_output[i] * (1.0 - sigmoid_output[i]) * grad_output[i]`.
    ///
    /// `sigmoid_output`, `grad_output`, and `grad_input` must have the same length; an error is
    /// returned if they do not.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(GpuError)` if the input/output lengths mismatch or if a GPU transfer
    /// or kernel operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let sigmoid_output = vec![0.5f32, 0.7310586, 0.26894143];
    /// let grad_output = vec![1.0f32, 0.5, -0.2];
    /// let mut grad_input = vec![0.0f32; sigmoid_output.len()];
    ///
    /// // Assuming `cuda` implements the CudaBackend and is available:
    /// cuda.sigmoid_backward(&sigmoid_output, &grad_output, &mut grad_input).unwrap();
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

    /// Adds a per-column bias vector to a batched 2D activation buffer in-place.
    ///
    /// The `data` slice is interpreted as `batch_size × n` row-major elements; the same `bias`
    /// vector of length `n` is added to every row. Operates in-place on `data`.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::DimensionMismatch` if `data.len() < batch_size * n` or `bias.len() < n`.
    /// Returns other `GpuError` variants for device transfer, allocation, or kernel launch failures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // `backend` is an initialized `CudaBackend`.
    /// let mut data = vec![0.0f32; 2 * 3]; // batch_size = 2, n = 3
    /// let bias = vec![1.0f32, 2.0f32, 3.0f32];
    /// backend.add_bias(&mut data, &bias, 2, 3).unwrap();
    /// assert_eq!(data, vec![1.0f32, 2.0f32, 3.0f32, 1.0f32, 2.0f32, 3.0f32]);
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

    /// Computes the column-wise sum across a batched 2D array and writes the result into `out`.
    ///
    /// The input `data` is interpreted as `batch_size` rows each with `n` columns (row-major),
    /// and `out` receives the per-column sums (length `n`). If `batch_size == 0` or `n == 0` this
    /// function returns early with `Ok(())` and does nothing.
    ///
    /// Errors:
    /// - Returns `GpuError::DimensionMismatch` if `data.len() < batch_size * n` or `out.len() < n`.
    /// - May return `GpuError::TransferError` or `GpuError::OperationFailed` for device transfer or kernel failures.
    ///
    /// # Examples
    ///
    /// ```
    /// # use your_crate::gpu::CudaBackend;
    /// # use your_crate::gpu::GpuBackend;
    /// let backend = CudaBackend::new().unwrap();
    /// let batch_size = 2;
    /// let n = 3;
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3 matrix: rows [1,2,3], [4,5,6]
    /// let mut out = vec![0.0f32; n];
    /// backend.sum_rows(&data, &mut out, batch_size, n).unwrap();
    /// assert_eq!(out, vec![5.0f32, 7.0, 9.0]); // column sums
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

    /// Performs a 2D convolution (forward pass) over a batched input tensor and writes results into `output`.
    ///
    /// Accepts input in row-major NCHW layout:
    /// - `input` length must be batch_size * in_channels * input_h * input_w
    /// - `filters` length must be out_channels * in_channels * kernel_h * kernel_w
    /// - `bias` length must be out_channels
    /// - `output` length must be batch_size * out_channels * out_h * out_w, where
    ///   out_h = (input_h + 2*padding - kernel_h) / stride + 1 and
    ///   out_w = (input_w + 2*padding - kernel_w) / stride + 1
    ///
    /// The function validates input sizes, transfers buffers to the device, launches the CUDA
    /// conv2d_forward kernel, and copies the computed output back into `output`.
    ///
    /// Errors:
    /// - Returns `GpuError::DimensionMismatch` when any buffer is smaller than the expected size.
    /// - Returns `GpuError::TransferError`, `GpuError::AllocationFailed`, or `GpuError::OperationFailed`
    ///   on device transfer, allocation, or kernel-launch failures respectively.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Single-batch, single-channel example with 3x3 input and 2x2 kernel:
    /// let backend = CudaBackend::new().unwrap();
    /// let batch = 1;
    /// let ic = 1;
    /// let oc = 1;
    /// let ih = 3;
    /// let iw = 3;
    /// let kh = 2;
    /// let kw = 2;
    /// let stride = 1;
    /// let padding = 0;
    ///
    /// let input = vec![1.0f32; batch * ic * ih * iw];
    /// let filters = vec![1.0f32; oc * ic * kh * kw];
    /// let bias = vec![0.0f32; oc];
    /// let mut output = vec![0.0f32; batch * oc * ((ih + 2*padding - kh)/stride + 1) * ((iw + 2*padding - kw)/stride + 1)];
    ///
    /// backend.conv2d_forward(
    ///     &input, &filters, &bias, &mut output,
    ///     batch, ic, oc, ih, iw, kh, kw, stride, padding
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

    /// Computes gradients for a 2D convolution with respect to the input, filters, and bias.
    ///
    /// Validates buffer sizes and spatial dimensions derived from the provided shape, kernel size,
    /// stride, and padding, then fills `grad_input`, `grad_filters`, and `grad_bias`. Returns an
    /// error if dimension validation, device memory allocation, data transfer, kernel lookup, or
    /// kernel launch fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Create backend and compute convolutional gradients (example; requires CUDA device).
    /// let backend = CudaBackend::new().unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;
    use cblas::{sgemm as cpu_sgemm, Layout, Transpose};

    /// Compute an m × n matrix product using the CPU BLAS `sgemm` reference implementation.
    ///
    /// The inputs `a` and `b` are interpreted according to `trans_a` and `trans_b`:
    /// - `a`: shape (m, k) if `trans_a == Transpose::None`, otherwise (k, m).
    /// - `b`: shape (k, n) if `trans_b == Transpose::None`, otherwise (n, k).
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the result matrix in row-major order with shape (m, n).
    ///
    /// # Examples
    ///
    /// ```
    /// // C = A * B where A is 2x3 and B is 3x2
    /// let a = vec![1.0f32, 2.0, 3.0,
    ///              4.0, 5.0, 6.0]; // 2 x 3
    /// let b = vec![7.0f32, 8.0,
    ///              9.0, 10.0,
    ///              11.0, 12.0]; // 3 x 2
    /// let c = cpu_matmul(2, 2, 3, &a, &b, Transpose::None, Transpose::None);
    /// assert_eq!(c.len(), 4);
    /// // manual check: first row = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    /// assert!((c[0] - 58.0).abs() < 1e-6 && (c[1] - 64.0).abs() < 1e-6);
    /// ```
    fn cpu_matmul(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        trans_a: Transpose,
        trans_b: Transpose,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let lda = if trans_a == Transpose::None { k } else { m };
        let ldb = if trans_b == Transpose::None { n } else { k };
        unsafe {
            cpu_sgemm(
                Layout::RowMajor,
                trans_a,
                trans_b,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                lda as i32,
                b,
                ldb as i32,
                0.0,
                &mut c,
                n as i32,
            );
        }
        c
    }

    /// Asserts that two f32 slices are equal elementwise within a given absolute tolerance.
    ///
    /// Panics if the slices have different lengths or if any corresponding elements differ by
    /// greater than or equal to `tol`.
    ///
    /// # Parameters
    ///
    /// - `a`: first slice to compare (reference / expected values).
    /// - `b`: second slice to compare (actual values).
    /// - `tol`: maximum allowed absolute difference for each element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1.0_f32, 2.0, 3.0];
    /// let b = [1.001_f32, 1.999, 3.0];
    /// assert_approx_eq(&a, &b, 0.01);
    /// ```
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {}: cpu={} gpu={} (diff={})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_cuda_sgemm_basic() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        // A (2x3) * B (3x4) = C (2x4)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::None, Transpose::None);
        let mut c = vec![0.0f32; 8];
        backend.sgemm(2, 4, 3, &a, &b, &mut c).unwrap();
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_cuda_sgemm_identity() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 9];
        backend.sgemm(3, 3, 3, &a, &identity, &mut c).unwrap();
        assert_approx_eq(&c, &a, 1e-5);
    }

    #[test]
    fn test_cuda_sgemm_at() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        // A stored as (k=3 x m=2), transposed to (2x3), times B (3x4) = C (2x4)
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::Ordinary, Transpose::None);
        let mut c = vec![0.0f32; 8];
        backend.sgemm_at(2, 4, 3, &a, &b, &mut c).unwrap();
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_cuda_sgemm_bt() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        // A (2x3) * B^T where B stored as (n=4 x k=3)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ];
        let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::None, Transpose::Ordinary);
        let mut c = vec![0.0f32; 8];
        backend.sgemm_bt(2, 4, 3, &a, &b, &mut c).unwrap();
        assert_approx_eq(&c, &expected, 1e-4);
    }

    #[test]
    fn test_cuda_sgemm_large() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let (m, n, k) = (64, 32, 128);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let expected = cpu_matmul(m, n, k, &a, &b, Transpose::None, Transpose::None);
        let mut c = vec![0.0f32; m * n];
        backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_cuda_sgemm_dimension_mismatch() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let a = vec![1.0; 4]; // too small for 2x3
        let b = vec![1.0; 12];
        let mut c = vec![0.0; 8];
        let result = backend.sgemm(2, 4, 3, &a, &b, &mut c);
        assert!(result.is_err());
    }

    // ── Element-wise kernel tests ──────────────────────────────────────

    #[test]
    fn test_cuda_elementwise_relu() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5];
        backend.relu(&mut data).unwrap();
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5];
        assert_approx_eq(&data, &expected, 1e-6);
    }

    #[test]
    fn test_cuda_elementwise_relu_empty() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let mut data: Vec<f32> = vec![];
        backend.relu(&mut data).unwrap();
        assert!(data.is_empty());
    }

    /// Verifies that the CUDA ReLU backward kernel writes the upstream gradient to positions where the input is greater than zero and writes zero elsewhere.
    ///
    /// The test constructs sample input and upstream gradients, runs `relu_backward`, and asserts that
    /// `grad_input[i] == grad_output[i]` when `input[i] > 0`, otherwise `grad_input[i] == 0`.
    #[test]
    fn test_cuda_elementwise_relu_backward() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5];
        let grad_output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut grad_input = vec![0.0; 7];
        backend
            .relu_backward(&input, &grad_output, &mut grad_input)
            .unwrap();
        // grad_input[i] = input[i] > 0 ? grad_output[i] : 0
        let expected = vec![0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0];
        assert_approx_eq(&grad_input, &expected, 1e-6);
    }

    #[test]
    fn test_cuda_elementwise_sigmoid() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        let mut data = vec![0.0, 1.0, -1.0, 10.0, -10.0];
        backend.sigmoid(&mut data).unwrap();

        // sigmoid(0) = 0.5
        assert!((data[0] - 0.5).abs() < 1e-5);
        // sigmoid(x) in (0, 1)
        for &v in &data {
            assert!(v > 0.0 && v < 1.0);
        }
        // sigmoid(1) ≈ 0.7310586
        assert!((data[1] - 0.7310586).abs() < 1e-4);
        // sigmoid(-1) ≈ 0.2689414
        assert!((data[2] - 0.2689414).abs() < 1e-4);
    }

    #[test]
    fn test_cuda_elementwise_sigmoid_backward() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        let sigmoid_output = vec![0.5, 0.7310586, 0.2689414];
        let grad_output = vec![1.0, 1.0, 1.0];
        let mut grad_input = vec![0.0; 3];
        backend
            .sigmoid_backward(&sigmoid_output, &grad_output, &mut grad_input)
            .unwrap();

        // grad = s * (1 - s) * grad_output
        // s=0.5: 0.5 * 0.5 = 0.25
        assert!((grad_input[0] - 0.25).abs() < 1e-5);
        // s=0.731: 0.731 * 0.269 ≈ 0.1966
        assert!((grad_input[1] - 0.7310586 * 0.2689414).abs() < 1e-4);
    }

    #[test]
    fn test_cuda_elementwise_add_bias() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        // 2 rows x 3 columns
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![10.0, 20.0, 30.0];
        backend.add_bias(&mut data, &bias, 2, 3).unwrap();
        let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_approx_eq(&data, &expected, 1e-5);
    }

    #[test]
    fn test_cuda_elementwise_sum_rows() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        // 3 rows x 4 columns
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut out = vec![0.0; 4];
        backend.sum_rows(&data, &mut out, 3, 4).unwrap();
        let expected = vec![15.0, 18.0, 21.0, 24.0];
        assert_approx_eq(&out, &expected, 1e-5);
    }

    // ── Conv2D kernel tests ────────────────────────────────────────────

    /// Compute a CPU reference for a batched 2D convolution in NCHW layout.
    ///
    /// Produces an output buffer of shape (batch_size, out_channels, out_h, out_w) in NCHW order,
    /// where out_h and out_w are computed from input dimensions, kernel size, stride, and padding.
    /// The provided `bias` is added per output channel; `stride` and `padding` are applied to the input
    /// before convolution.
    ///
    /// # Examples
    ///
    /// ```
    /// let input = vec![1.0f32, 2.0, 3.0, 4.0]; // 1×1×2×2 (batch=1, in_ch=1, H=2, W=2)
    /// let filters = vec![1.0f32]; // 1×1×1×1 (out_ch=1, in_ch=1, kh=1, kw=1)
    /// let bias = vec![0.0f32]; // out_ch=1
    /// let out = cpu_conv2d_forward(&input, &filters, &bias, 1, 1, 1, 2, 2, 1, 1, 1, 0);
    /// assert_eq!(out, input);
    /// ```
    fn cpu_conv2d_forward(
        input: &[f32],
        filters: &[f32],
        bias: &[f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Vec<f32> {
        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
        let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias[oc];
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = (oh * stride + kh) as isize - padding as isize;
                                    let iw = (ow * stride + kw) as isize - padding as isize;
                                    if ih >= 0
                                        && ih < input_h as isize
                                        && iw >= 0
                                        && iw < input_w as isize
                                    {
                                        let iv = input[((b * in_channels + ic) * input_h
                                            + ih as usize)
                                            * input_w
                                            + iw as usize];
                                        let fv = filters[((oc * in_channels + ic) * kernel_h + kh)
                                            * kernel_w
                                            + kw];
                                        sum += iv * fv;
                                    }
                                }
                            }
                        }
                        output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn test_cuda_conv2d_forward_basic() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("No CUDA device available, skipping test");
                return;
            }
        };

        // 1 batch, 1 in_channel, 2 out_channels, 4x4 input, 3x3 kernel, stride=1, padding=0
        let batch_size = 1;
        let in_channels = 1;
        let out_channels = 2;
        let (input_h, input_w) = (4, 4);
        let (kernel_h, kernel_w) = (3, 3);
        let stride = 1;
        let padding = 0;
        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1; // 2
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1; // 2

        let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let bias = vec![0.1, -0.1];

        let expected = cpu_conv2d_forward(
            &input,
            &filters,
            &bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        );

        let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];
        backend
            .conv2d_forward(
                &input,
                &filters,
                &bias,
                &mut output,
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
            .unwrap();

        assert_approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_cuda_conv2d_forward_with_padding() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let batch_size = 2;
        let in_channels = 1;
        let out_channels = 1;
        let (input_h, input_w) = (3, 3);
        let (kernel_h, kernel_w) = (3, 3);
        let stride = 1;
        let padding = 1;
        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1; // 3
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1; // 3

        let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let filters = vec![1.0; out_channels * in_channels * kernel_h * kernel_w];
        let bias = vec![0.0];

        let expected = cpu_conv2d_forward(
            &input,
            &filters,
            &bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        );

        let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];
        backend
            .conv2d_forward(
                &input,
                &filters,
                &bias,
                &mut output,
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
            .unwrap();

        assert_approx_eq(&output, &expected, 1e-4);
    }

    /// Verifies that `conv2d_forward` produces the same output as a CPU reference for a multi-channel input.
    ///
    /// This integration test constructs a 3-channel input, two output channels, 3×3 kernels with padding,
    /// and compares the GPU backend's forward convolution result against `cpu_conv2d_forward` within 1e-3.
    ///
    /// # Examples
    ///
    /// ```
    /// // Setup backend (test skips if CUDA is unavailable)
    /// let backend = match CudaBackend::new() {
    ///     Ok(b) => b,
    ///     Err(_) => return,
    /// };
    ///
    /// // Shapes and parameters
    /// let batch_size = 1;
    /// let in_channels = 3;
    /// let out_channels = 2;
    /// let (input_h, input_w) = (5, 5);
    /// let (kernel_h, kernel_w) = (3, 3);
    /// let stride = 1;
    /// let padding = 1;
    ///
    /// // Randomized example data (deterministic pattern for test)
    /// let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
    ///     .map(|i| ((i % 7) as f32) * 0.1)
    ///     .collect();
    /// let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
    ///     .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
    ///     .collect();
    /// let bias = vec![0.5, -0.3];
    ///
    /// // Reference and GPU outputs
    /// let expected = cpu_conv2d_forward(
    ///     &input, &filters, &bias,
    ///     batch_size, in_channels, out_channels,
    ///     input_h, input_w, kernel_h, kernel_w, stride, padding,
    /// );
    /// let mut output = vec![0.0f32; batch_size * out_channels * ((input_h + 2 * padding - kernel_h) / stride + 1) * ((input_w + 2 * padding - kernel_w) / stride + 1)];
    ///
    /// backend.conv2d_forward(
    ///     &input, &filters, &bias, &mut output,
    ///     batch_size, in_channels, out_channels,
    ///     input_h, input_w, kernel_h, kernel_w, stride, padding,
    /// ).unwrap();
    ///
    /// assert_approx_eq(&output, &expected, 1e-3);
    /// ```
    #[test]
    fn test_cuda_conv2d_forward_multichannel() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 2;
        let (input_h, input_w) = (5, 5);
        let (kernel_h, kernel_w) = (3, 3);
        let stride = 1;
        let padding = 1;
        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

        let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
            .map(|i| ((i % 7) as f32) * 0.1)
            .collect();
        let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let bias = vec![0.5, -0.3];

        let expected = cpu_conv2d_forward(
            &input,
            &filters,
            &bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        );

        let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];
        backend
            .conv2d_forward(
                &input,
                &filters,
                &bias,
                &mut output,
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
            .unwrap();

        assert_approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_cuda_conv2d_backward_bias() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let batch_size = 2;
        let in_channels = 1;
        let out_channels = 2;
        let (input_h, input_w) = (4, 4);
        let (kernel_h, kernel_w) = (3, 3);
        let stride = 1;
        let padding = 0;
        let out_h = 2;
        let out_w = 2;

        let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let grad_output: Vec<f32> = (0..batch_size * out_channels * out_h * out_w)
            .map(|i| (i as f32) * 0.1)
            .collect();

        let mut grad_input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
        let mut grad_filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
        let mut grad_bias = vec![0.0f32; out_channels];

        backend
            .conv2d_backward(
                &input,
                &filters,
                &grad_output,
                &mut grad_input,
                &mut grad_filters,
                &mut grad_bias,
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
            .unwrap();

        // CPU reference for bias gradient: sum over batch and spatial dims
        let mut expected_grad_bias = vec![0.0f32; out_channels];
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        expected_grad_bias[oc] +=
                            grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                    }
                }
            }
        }
        assert_approx_eq(&grad_bias, &expected_grad_bias, 1e-4);
    }

    #[test]
    fn test_cuda_conv2d_backward_filters() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        // Simple case: 1 batch, 1 in/out channel, 3x3 input, 2x2 kernel, no padding
        let batch_size = 1;
        let in_channels = 1;
        let out_channels = 1;
        let (input_h, input_w) = (3, 3);
        let (kernel_h, kernel_w) = (2, 2);
        let stride = 1;
        let padding = 0;
        let out_h = 2;
        let out_w = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let filters = vec![0.1, 0.2, 0.3, 0.4];
        let grad_output = vec![1.0, 1.0, 1.0, 1.0];

        let mut grad_input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
        let mut grad_filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
        let mut grad_bias = vec![0.0f32; out_channels];

        backend
            .conv2d_backward(
                &input,
                &filters,
                &grad_output,
                &mut grad_input,
                &mut grad_filters,
                &mut grad_bias,
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
            .unwrap();

        // grad_filters[kh][kw] = sum over (b, oh, ow) of input[oh*s+kh][ow*s+kw] * grad_output[oh][ow]
        // kh=0,kw=0: 1*1+2*1+4*1+5*1 = 12
        // kh=0,kw=1: 2*1+3*1+5*1+6*1 = 16
        // kh=1,kw=0: 4*1+5*1+7*1+8*1 = 24
        // kh=1,kw=1: 5*1+6*1+8*1+9*1 = 28
        let expected_grad_filters = vec![12.0, 16.0, 24.0, 28.0];
        assert_approx_eq(&grad_filters, &expected_grad_filters, 1e-4);
    }

    #[test]
    fn test_cuda_conv2d_dimension_mismatch() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let input = vec![1.0; 4]; // too small
        let filters = vec![1.0; 9];
        let bias = vec![0.0];
        let mut output = vec![0.0; 1];

        let result = backend.conv2d_forward(
            &input,
            &filters,
            &bias,
            &mut output,
            1,
            1,
            1,
            4,
            4,
            3,
            3,
            1,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_elementwise_relu_large() {
        let backend = match CudaBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let n = 10000;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
        backend.relu(&mut data).unwrap();
        for (i, &v) in data.iter().enumerate() {
            let orig = (i as f32) - (n as f32 / 2.0);
            let expected = if orig > 0.0 { orig } else { 0.0 };
            assert!((v - expected).abs() < 1e-5, "mismatch at {}", i);
        }
    }
}