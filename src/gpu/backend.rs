//! GPU backend trait and common types for GPU-accelerated operations.
//!
//! This module defines the `GpuBackend` trait that all GPU backends (Metal, CUDA)
//! must implement, along with supporting types for device discovery and error handling.

use std::fmt;

/// Errors that can occur during GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// Device not found or not available.
    DeviceNotFound(String),
    /// Failed to allocate GPU memory.
    AllocationFailed(String),
    /// Failed to compile or load a GPU kernel/shader.
    KernelError(String),
    /// Data transfer between host and device failed.
    TransferError(String),
    /// A GPU operation returned an error.
    OperationFailed(String),
    /// Dimension mismatch in matrix/tensor operation.
    DimensionMismatch(String),
    /// The requested feature is not supported by this backend.
    Unsupported(String),
}

impl fmt::Display for GpuError {
    /// Formats a `GpuError` into a human-readable message.
    ///
    /// Produces a concise, user-facing description for each `GpuError` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::GpuError;
    /// let err = GpuError::DeviceNotFound("PCIe device not present".into());
    /// assert_eq!(format!("{}", err), "GPU device not found: PCIe device not present");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceNotFound(msg) => write!(f, "GPU device not found: {}", msg),
            GpuError::AllocationFailed(msg) => write!(f, "GPU allocation failed: {}", msg),
            GpuError::KernelError(msg) => write!(f, "GPU kernel error: {}", msg),
            GpuError::TransferError(msg) => write!(f, "GPU transfer error: {}", msg),
            GpuError::OperationFailed(msg) => write!(f, "GPU operation failed: {}", msg),
            GpuError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            GpuError::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// Information about a discovered GPU device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Human-readable device name (e.g., "Apple M1 Pro", "NVIDIA RTX 4090").
    pub name: String,
    /// Backend type that manages this device.
    pub backend: BackendType,
    /// Total device memory in bytes, if known.
    pub memory_bytes: Option<u64>,
    /// Whether this device supports the required operations.
    pub is_supported: bool,
}

/// Identifies which GPU backend implementation is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Apple Metal backend (macOS).
    Metal,
    /// NVIDIA CUDA backend.
    Cuda,
}

impl fmt::Display for BackendType {
    /// Format `BackendType` as a human-readable name.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::BackendType;
    ///
    /// assert_eq!(format!("{}", BackendType::Metal), "Metal");
    /// assert_eq!(format!("{}", BackendType::Cuda), "CUDA");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Cuda => write!(f, "CUDA"),
        }
    }
}

/// Core trait for GPU-accelerated neural network operations.
///
/// Implementations of this trait provide GPU-accelerated versions of the
/// fundamental operations used in neural network training and inference.
/// Each backend (Metal, CUDA) provides its own implementation.
///
/// All data is passed as host-side `&[f32]` / `&mut [f32]` slices.
/// Implementations are responsible for transferring data to/from the GPU.
///
/// # Example
///
/// ```ignore
/// let backend = MetalBackend::new()?;
/// println!("Using device: {}", backend.device_info().name);
///
/// // Matrix multiply: C = A × B
/// // A is (m × k), B is (k × n), C is (m × n)
/// backend.sgemm(m, n, k, &a, &b, &mut c)?;
/// ```
pub trait GpuBackend: Send {
    /// Returns information about the GPU device used by this backend.
    fn device_info(&self) -> &GpuDevice;

    // ── Matrix operations ───────────────────────────────────────────────

    /// General matrix multiply: C = A × B.
    ///
    /// Computes the product of two row-major matrices:
    /// - `a` has shape (m × k)
    /// - `b` has shape (k × n)
    /// - `c` has shape (m × n)
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    /// * `a` - Input matrix A, row-major (length m*k)
    /// * `b` - Input matrix B, row-major (length k*n)
    /// * `c` - Output matrix C, row-major (length m*n)
    fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError>;

    /// Transposed matrix multiply: C = Aᵀ × B.
    ///
    /// - `a` has shape (k × m), treated as Aᵀ so effective shape is (m × k)
    /// - `b` has shape (k × n)
    /// - `c` has shape (m × n)
    fn sgemm_at(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError>;

    /// Transposed matrix multiply: C = A × Bᵀ.
    ///
    /// - `a` has shape (m × k)
    /// - `b` has shape (n × k), treated as Bᵀ so effective shape is (k × n)
    /// - `c` has shape (m × n)
    fn sgemm_bt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError>;

    // ── Element-wise activation operations ──────────────────────────────

    /// Apply ReLU activation in-place: x[i] = max(0, x[i]).
    fn relu(&self, data: &mut [f32]) -> Result<(), GpuError>;

    /// Compute ReLU backward pass: grad_input[i] = grad_output[i] if input[i] > 0, else 0.
    fn relu_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError>;

    /// Apply sigmoid activation in-place: x[i] = 1 / (1 + exp(-x[i])).
    fn sigmoid(&self, data: &mut [f32]) -> Result<(), GpuError>;

    /// Compute sigmoid backward pass: grad_input[i] = grad_output[i] * s[i] * (1 - s[i]).
    ///
    /// `sigmoid_output` contains the forward-pass sigmoid outputs (not the pre-activation inputs).
    fn sigmoid_backward(
        &self,
        sigmoid_output: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), GpuError>;

    /// Add bias to each row of a batch: output[b][j] += bias[j].
    ///
    /// - `data` has shape (batch_size × n), row-major
    /// - `bias` has shape (n,)
    fn add_bias(
        &self,
        data: &mut [f32],
        bias: &[f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError>;

    /// Sum rows to produce a vector: out[j] = Σ_b data[b][j].
    ///
    /// - `data` has shape (batch_size × n), row-major
    /// - `out` has shape (n,)
    fn sum_rows(
        &self,
        data: &[f32],
        out: &mut [f32],
        batch_size: usize,
        n: usize,
    ) -> Result<(), GpuError>;

    // ── Convolution operations ──────────────────────────────────────────

    /// 2D convolution forward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor, shape (batch × in_channels × h × w)
    /// * `filters` - Filter weights, shape (out_channels × in_channels × kh × kw)
    /// * `bias` - Bias per output channel, shape (out_channels,)
    /// * `output` - Output tensor, shape (batch × out_channels × out_h × out_w)
    /// * `batch_size` - Number of images in the batch
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels (filters)
    /// * `input_h` - Input height
    /// * `input_w` - Input width
    /// * `kernel_h` - Filter kernel height
    /// * `kernel_w` - Filter kernel width
    /// * `stride` - Convolution stride
    /// * `padding` - Zero-padding applied to each side
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), GpuError>;

    /// 2D convolution backward pass (computes grad_input, grad_filters, grad_bias).
    ///
    /// # Arguments
    ///
    /// * `input` - Original input from forward pass
    /// * `filters` - Filter weights from forward pass
    /// * `grad_output` - Gradient w.r.t. output, shape (batch × out_channels × out_h × out_w)
    /// * `grad_input` - Gradient w.r.t. input (to be computed)
    /// * `grad_filters` - Gradient w.r.t. filters (to be computed)
    /// * `grad_bias` - Gradient w.r.t. bias (to be computed)
    /// * `batch_size` - Number of images in the batch
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `input_h` - Input height
    /// * `input_w` - Input width
    /// * `kernel_h` - Filter kernel height
    /// * `kernel_w` - Filter kernel width
    /// * `stride` - Convolution stride
    /// * `padding` - Zero-padding applied to each side
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), GpuError>;
}