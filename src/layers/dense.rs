//! Dense (fully connected) layer implementation
//!
//! This module provides a DenseLayer (also known as Linear or Fully Connected layer)
//! that performs the transformation: output = input × weights + biases

use crate::layers::gradient::GradientAccumulator;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::gpu::backend::{GpuBackend, GpuError};
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use std::sync::Arc;

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;
use cblas::{sgemm, Layout, Transpose};

/// Dense (fully connected) layer with weights and biases.
///
/// Performs the linear transformation: y = xW + b
/// where x is the input (batch_size × input_size),
/// W is the weight matrix (input_size × output_size),
/// and b is the bias vector (output_size).
///
/// # Fields
///
/// * `input_size` - Number of input features
/// * `output_size` - Number of output features
/// * `weights` - Weight matrix stored in row-major format (input_size × output_size)
/// * `biases` - Bias vector (output_size)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::DenseLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(784, 512, &mut rng);
/// assert_eq!(layer.input_size(), 784);
/// assert_eq!(layer.output_size(), 512);
/// ```
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    // Gradient accumulators
    grad_weights: GradientAccumulator,
    grad_biases: GradientAccumulator,
    // Optional GPU backend for accelerated computation
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    gpu_backend: Option<Arc<dyn GpuBackend>>,
}

impl DenseLayer {
    /// Creates a dense (fully connected) layer with Xavier-initialized weights and zero biases.
    ///
    /// Weights are sampled uniformly from [-limit, limit], where
    /// `limit = sqrt(6.0 / (input_size + output_size))`. Biases and gradient accumulators
    /// are initialized to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::DenseLayer;
    /// use rust_neural_networks::utils::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(128, 64, &mut rng);
    /// assert_eq!(layer.input_size(), 128);
    /// assert_eq!(layer.output_size(), 64);
    /// ```
    pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
        // Xavier initialization: limit = sqrt(6 / (fan_in + fan_out))
        let mut weights = vec![0.0f32; input_size * output_size];
        let limit = (6.0f32 / (input_size + output_size) as f32).sqrt();

        for value in &mut weights {
            *value = rng.gen_range_f32(-limit, limit);
        }

        Self {
            input_size,
            output_size,
            weights,
            biases: vec![0.0f32; output_size],
            grad_weights: GradientAccumulator::new(input_size * output_size),
            grad_biases: GradientAccumulator::new(output_size),
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            gpu_backend: None,
        }
    }

    /// Creates a DenseLayer with the provided GPU backend attached.
    ///
    /// The backend will be used for accelerated matrix operations during forward and backward
    /// passes; if a GPU operation fails at runtime the implementation falls back to the CPU BLAS path.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use rust_neural_networks::layers::DenseLayer;
    /// use rust_neural_networks::utils::SimpleRng;
    /// let backend = Arc::new(MetalBackend::new().unwrap());
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new_with_gpu(784, 512, &mut rng, backend);
    /// assert_eq!(layer.input_size(), 784);
    /// assert_eq!(layer.output_size(), 512);
    /// ```
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    pub fn new_with_gpu(
        input_size: usize,
        output_size: usize,
        rng: &mut SimpleRng,
        gpu_backend: Arc<dyn GpuBackend>,
    ) -> Self {
        let mut layer = Self::new(input_size, output_size, rng);
        layer.gpu_backend = Some(gpu_backend);
        layer
    }

    /// Attach or replace the GPU backend for this layer.
    ///
    /// When a backend is attached, the layer will attempt GPU-accelerated forward and backward
    /// paths if a GPU backend is present and the corresponding features are enabled.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use rust_neural_networks::layers::DenseLayer;
    /// use rust_neural_networks::utils::SimpleRng;
    /// use rust_neural_networks::gpu::GpuBackend;
    ///
    /// let mut rng = SimpleRng::new(0);
    /// let mut layer = DenseLayer::new(4, 8, &mut rng);
    /// let backend: Arc<dyn GpuBackend> = /* create backend */;
    /// layer.set_gpu_backend(backend);
    /// ```
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    pub fn set_gpu_backend(&mut self, backend: Arc<dyn GpuBackend>) {
        self.gpu_backend = Some(backend);
    }

    /// Indicates whether a GPU backend is attached to this layer.
    ///
    /// # Returns
    ///
    /// `true` if a GPU backend is attached, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Construct a DenseLayer (details omitted) and query the backend presence:
    /// let layer = /* DenseLayer::new(...) */ ;
    /// assert!(!layer.has_gpu_backend());
    /// ```
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    pub fn has_gpu_backend(&self) -> bool {
        self.gpu_backend.is_some()
    }

    /// Configured number of input features for the layer.
    ///
    /// # Returns
    ///
    /// The number of input features configured for this layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::DenseLayer;
    /// use rust_neural_networks::utils::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(4, 8, &mut rng);
    /// assert_eq!(layer.input_size(), 4);
    /// ```
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Reports the number of output features produced by the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(4, 8, &mut rng);
    /// assert_eq!(layer.output_size(), 8);
    /// ```
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Return the total number of trainable parameters in the layer.
    ///
    /// This equals input_size × output_size (weights) plus output_size (biases).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(3, 4, &mut rng);
    /// assert_eq!(layer.parameter_count(), 3 * 4 + 4);
    /// ```
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Immutable view of the layer's weight values.
    ///
    /// The returned slice contains weights in row-major order with length equal to
    /// `input_size * output_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(2, 3, &mut rng);
    /// assert_eq!(layer.weights().len(), 2 * 3);
    /// ```
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Provides a slice view of the layer's bias vector.
    ///
    /// # Returns
    /// A slice containing the bias for each output feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(4, 3, &mut rng);
    /// let b = layer.biases();
    /// assert_eq!(b.len(), 3);
    /// ```
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    /// Computes the L2 norm (magnitude) of the layer's weight and bias gradients.
    ///
    /// This is useful for monitoring gradient flow during training and detecting
    /// vanishing or exploding gradients. The L2 norm is computed as sqrt(sum(g_i^2))
    /// for each gradient component.
    ///
    /// # Returns
    ///
    /// A tuple `(weight_grad_norm, bias_grad_norm)` where:
    /// - `weight_grad_norm` is the L2 norm of the weight gradients
    /// - `bias_grad_norm` is the L2 norm of the bias gradients
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(2, 3, &mut rng);
    ///
    /// // Perform forward and backward pass to accumulate gradients
    /// let input = vec![1.0, 2.0];
    /// let mut output = vec![0.0; 3];
    /// layer.forward(&input, &mut output, 1);
    /// let grad_output = vec![0.1, 0.2, 0.3];
    /// let mut grad_input = vec![0.0; 2];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// // Get gradient magnitudes
    /// let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
    /// assert!(weight_norm >= 0.0);
    /// assert!(bias_norm >= 0.0);
    /// ```
    pub fn get_gradient_magnitude(&self) -> (f32, f32) {
        (self.grad_weights.l2_norm(), self.grad_biases.l2_norm())
    }

    /// Creates a dense layer with pre-existing weights and biases (no random initialization).
    ///
    /// This constructor is used when loading a model from disk, allowing the layer to be
    /// reconstructed with previously trained parameters. Gradient accumulators are
    /// zero-initialized, ready for the next training pass.
    ///
    /// # Panics
    ///
    /// Panics if the length of `weights` does not equal `input_size * output_size` or
    /// the length of `biases` does not equal `output_size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::layers::Layer;
    ///
    /// let input_size = 2;
    /// let output_size = 3;
    /// let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 × 3
    /// let biases = vec![0.1, 0.2, 0.3];
    /// let layer = DenseLayer::new_with_weights(input_size, output_size, weights.clone(), biases.clone());
    /// assert_eq!(layer.weights(), weights.as_slice());
    /// assert_eq!(layer.biases(), biases.as_slice());
    /// assert_eq!(layer.input_size(), 2);
    /// assert_eq!(layer.output_size(), 3);
    /// ```
    pub fn new_with_weights(
        input_size: usize,
        output_size: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) -> Self {
        assert_eq!(
            weights.len(),
            input_size * output_size,
            "weights length {} does not match input_size * output_size = {}",
            weights.len(),
            input_size * output_size
        );
        assert_eq!(
            biases.len(),
            output_size,
            "biases length {} does not match output_size = {}",
            biases.len(),
            output_size
        );

        Self {
            input_size,
            output_size,
            weights,
            biases,
            grad_weights: GradientAccumulator::new(input_size * output_size),
            grad_biases: GradientAccumulator::new(output_size),
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            gpu_backend: None,
        }
    }
}

// GPU-accelerated forward and backward implementations
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
impl DenseLayer {
    /// Perform a GPU-accelerated forward pass computing output = input × weights + biases.
    ///
    /// Attempts to run the matrix multiplication and bias addition on the provided GPU backend.
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful GPU computation, `Err(GpuError)` if the GPU backend reports an error.
    ///
    /// # Examples
    ///
    /// ```
    /// // Given `layer: &DenseLayer`, `input`, `output`, `batch_size` and `backend` already prepared:
    /// // layer.forward_gpu(&input, &mut output, batch_size, backend)?;
    /// ```
    fn forward_gpu(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        backend: &dyn GpuBackend,
    ) -> Result<(), GpuError> {
        // Step 1: Matrix multiply: output = input × weights
        // input: (batch_size × input_size), weights: (input_size × output_size)
        backend.sgemm(
            batch_size,
            self.output_size,
            self.input_size,
            input,
            &self.weights,
            output,
        )?;

        // Step 2: Add bias to each row
        backend.add_bias(output, &self.biases, batch_size, self.output_size)?;

        Ok(())
    }

    /// Compute and accumulate gradients for weights and biases and write input gradients using the GPU backend.
    ///
    /// Performs three operations for the provided batch:
    /// 1. Accumulates weight gradients computed as input^T × grad_output, scaled by 1 / batch_size.
    /// 2. Accumulates bias gradients computed by summing grad_output across the batch, scaled by 1 / batch_size.
    /// 3. Writes input gradients computed as grad_output × weights^T into `grad_input`.
    ///
    /// Returns `Ok(())` on success, `Err(GpuError)` if the GPU backend reports an error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Assume `layer` is a DenseLayer with a GPU backend attached.
    /// let batch_size = 4;
    /// let input = vec![0.0f32; batch_size * layer.input_size()];
    /// let grad_output = vec![0.0f32; batch_size * layer.output_size()];
    /// let mut grad_input = vec![0.0f32; batch_size * layer.input_size()];
    /// layer.backward_gpu(&input, &grad_output, &mut grad_input, batch_size, backend.as_ref()).unwrap();
    /// ```
    fn backward_gpu(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
        backend: &dyn GpuBackend,
    ) -> Result<(), GpuError> {
        let scale = 1.0f32 / batch_size as f32;

        // Step 1: Weight gradients: dW = input^T × grad_output / batch_size
        let mut grad_w_batch = vec![0.0f32; self.input_size * self.output_size];
        backend.sgemm_at(
            self.input_size,
            self.output_size,
            batch_size,
            input,
            grad_output,
            &mut grad_w_batch,
        )?;

        // Step 2: Bias gradients: db = sum_rows(grad_output) / batch_size
        let mut batch_bias_grad = vec![0.0f32; self.output_size];
        backend.sum_rows(
            grad_output,
            &mut batch_bias_grad,
            batch_size,
            self.output_size,
        )?;

        // Step 3: Input gradients: dX = grad_output × weights^T
        backend.sgemm_bt(
            batch_size,
            self.input_size,
            self.output_size,
            grad_output,
            &self.weights,
            grad_input,
        )?;

        // All GPU operations succeeded — now accumulate gradients.
        // Deferred to avoid partial accumulation if a GPU step fails mid-way.
        self.grad_weights.accumulate_scaled(&grad_w_batch, scale);
        self.grad_biases.accumulate_scaled(&batch_bias_grad, scale);

        Ok(())
    }
}

// Helper functions for BLAS operations

/// Performs a single-precision general matrix-matrix multiplication using the BLAS `sgemm`
/// routine with row-major layout.
///
/// The function computes: C := alpha * op(A) * op(B) + beta * C, where `op(X)` is either
/// the matrix `X` or its transpose depending on the corresponding transpose flag.
///
/// # Examples
///
/// ```ignore
/// // Multiply 2x2 matrices: result = A * B
/// let m = 2usize;
/// let n = 2usize;
/// let k = 2usize;
/// let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; // row-major 2x2: [[1,2],[3,4]]
/// let b: [f32; 4] = [5.0, 6.0, 7.0, 8.0]; // row-major 2x2: [[5,6],[7,8]]
/// let mut c: [f32; 4] = [0.0; 4];
/// // leading dimensions for row-major layout are the number of columns
/// sgemm_wrapper(m, n, k, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 0.0);
/// // Expected C = [[19,22],[43,50]]
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
#[allow(clippy::too_many_arguments)]
fn sgemm_wrapper(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
) {
    let trans_a = if transpose_a {
        Transpose::Ordinary
    } else {
        Transpose::None
    };
    let trans_b = if transpose_b {
        Transpose::Ordinary
    } else {
        Transpose::None
    };

    unsafe {
        sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

/// Adds a bias vector to each row of a row-major matrix in place.
///
/// The `data` slice represents a matrix with `rows` rows and `cols` columns in row-major order.
/// Each element of `bias` is added to the corresponding column of every row. `bias.len()` must
/// equal `cols`.
///
/// # Examples
///
/// ```ignore
/// let mut data = vec![0.0f32, 1.0, 2.0,   // row 0
///                     3.0, 4.0, 5.0];  // row 1
/// let bias = vec![1.0f32, 10.0, 100.0];
/// add_bias(&mut data, 2, 3, &bias);
/// assert_eq!(data, vec![1.0, 11.0, 102.0,  4.0, 14.0, 105.0]);
/// ```
fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

/// Sums each column of a row-major matrix and stores the column-wise sums in `out`.
///
/// `data` is interpreted as a matrix with `rows` rows and `cols` columns in row-major order.
/// The function overwrites the first `cols` elements of `out` with the sum of each column.
/// `out` must have length at least `cols`.
///
/// # Examples
///
/// ```ignore
/// let data: Vec<f32> = vec![
///     1.0, 2.0, 3.0, // row 0
///     4.0, 5.0, 6.0, // row 1
/// ];
/// let mut out = vec![0.0; 3];
/// sum_rows(&data, 2, 3, &mut out);
/// assert_eq!(out, vec![5.0, 7.0, 9.0]);
/// ```
fn sum_rows(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }

    for row in data.chunks_exact(cols).take(rows) {
        for (value, sum) in row.iter().zip(out.iter_mut()) {
            *sum += *value;
        }
    }
}

// Layer trait implementation

impl Layer for DenseLayer {
    /// Compute the dense layer's linear output for a batch: y = x * W + b.
    ///
    /// The input slice represents a row-major matrix with shape (batch_size, input_size).
    /// The output slice is written as a row-major matrix with shape (batch_size, output_size).
    /// The weight matrix has shape (input_size, output_size) and the bias has length output_size.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(2, 3, &mut rng);
    /// let input = [0.5f32, -1.0f32]; // batch_size = 1, input_size = 2
    /// let mut output = vec![0f32; 3]; // batch_size = 1, output_size = 3
    /// layer.forward(&input, &mut output, 1);
    /// assert_eq!(output.len(), 3);
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Step 1: Perform matrix multiplication: y = x × W
        // Dimension check:
        //   - x (input): (batch_size × input_size)
        //   - W (weights): (input_size × output_size)
        //   - y (output): (batch_size × output_size)
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "DenseLayer forward: input shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             This check prevents a BLAS segfault: incorrect input dimensions \
             passed to sgemm() cause undefined behavior in the unsafe BLAS call. \
             Verify input dimensions match the layer configuration \
             (see CLAUDE.md \u{2192} BLAS Safety for details).",
            batch_size,
            self.input_size,
            batch_size * self.input_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "DenseLayer forward: output shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             This check prevents a BLAS segfault: incorrect output buffer dimensions \
             passed to sgemm() cause undefined behavior in the unsafe BLAS call. \
             Ensure the output buffer has batch_size * output_size elements \
             (see CLAUDE.md \u{2192} BLAS Safety for details).",
            batch_size,
            self.output_size,
            batch_size * self.output_size,
            output.len()
        );
        assert_eq!(
            self.weights.len(),
            self.input_size * self.output_size,
            "DenseLayer forward: weights shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             This check prevents a BLAS segfault: incorrect weight matrix dimensions \
             passed to sgemm() cause undefined behavior in the unsafe BLAS call. \
             This indicates an internal layer state inconsistency \
             (see CLAUDE.md \u{2192} BLAS Safety for details).",
            self.input_size,
            self.output_size,
            self.input_size * self.output_size,
            self.weights.len()
        );

        // Try GPU-accelerated path if backend is available
        #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
        {
            if let Some(ref backend) = self.gpu_backend {
                if self
                    .forward_gpu(input, output, batch_size, backend.as_ref())
                    .is_ok()
                {
                    return;
                }
                // GPU failed, fall through to CPU path
            }
        }

        // CPU path: BLAS sgemm computes: output = 1.0 * (input × weights) + 0.0 * output
        // Using row-major layout with no transpose
        sgemm_wrapper(
            batch_size,
            self.output_size,
            self.input_size,
            input,
            self.input_size,
            &self.weights,
            self.output_size,
            output,
            self.output_size,
            false,
            false,
            1.0,
            0.0,
        );

        // Step 2: Add bias vector to each row of the output matrix
        assert_eq!(
            self.biases.len(),
            self.output_size,
            "DenseLayer forward: biases length mismatch. \
             Expected {} elements (one per output), got {}. \
             This indicates an internal layer state inconsistency.",
            self.output_size,
            self.biases.len()
        );
        add_bias(output, batch_size, self.output_size, &self.biases);
    }

    /// Accumulates this layer's gradients from a batch and writes the gradient with respect to the inputs into `grad_input`.
    ///
    /// Updates the internal gradient accumulators:
    /// - accumulates weight gradients averaged by `batch_size`,
    /// - accumulates bias gradients averaged by `batch_size`,
    /// and computes `grad_input` corresponding to the provided `grad_output`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::layers::Layer;
    ///
    /// let batch_size = 2usize;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(4, 3, &mut rng);
    /// let input = vec![0.0f32; batch_size * layer.input_size()];
    /// let grad_output = vec![0.0f32; batch_size * layer.output_size()];
    /// let mut grad_input = vec![0.0f32; batch_size * layer.input_size()];
    ///
    /// layer.backward(&input, &grad_output, &mut grad_input, batch_size);
    /// ```
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        if batch_size == 0 {
            panic!("batch_size cannot be zero in Dense::backward");
        }

        // Try GPU-accelerated path if backend is available
        #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
        {
            if let Some(ref backend) = self.gpu_backend {
                if self
                    .backward_gpu(input, grad_output, grad_input, batch_size, backend.as_ref())
                    .is_ok()
                {
                    return;
                }
                // GPU failed, fall through to CPU path
            }
        }

        let scale = 1.0f32 / batch_size as f32;

        // ===== Step 1: Weight gradients =====
        // Compute: ∂L/∂W = x^T × ∂L/∂y / batch_size
        // Input dimensions: x (batch_size × input_size)
        // Gradient w.r.t. output: ∂L/∂y (batch_size × output_size)
        // Result: ∂L/∂W (input_size × output_size)
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "DenseLayer backward: input shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             This input must match the input used in the corresponding forward pass.",
            batch_size,
            self.input_size,
            batch_size * self.input_size,
            input.len()
        );
        assert_eq!(
            grad_output.len(),
            batch_size * self.output_size,
            "DenseLayer backward: grad_output shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             Ensure grad_output has batch_size * output_size elements.",
            batch_size,
            self.output_size,
            batch_size * self.output_size,
            grad_output.len()
        );

        {
            let mut grad_w = self.grad_weights.borrow_mut();
            assert_eq!(
                grad_w.len(),
                self.input_size * self.output_size,
                "DenseLayer backward: grad_weights shape mismatch. \
                 Expected ({}, {}) = {} elements, got {}. \
                 This indicates an internal gradient accumulator inconsistency.",
                self.input_size,
                self.output_size,
                self.input_size * self.output_size,
                grad_w.len()
            );

            sgemm_wrapper(
                self.input_size,
                self.output_size,
                batch_size,
                input,
                self.input_size,
                grad_output,
                self.output_size,
                &mut grad_w,
                self.output_size,
                true,
                false,
                scale,
                1.0, // Accumulate gradients
            );
        }

        // ===== Step 2: Bias gradients =====
        // Compute: ∂L/∂b = Σ(∂L/∂y) / batch_size (sum along batch dimension)
        // Gradient w.r.t. output: ∂L/∂y (batch_size × output_size)
        // Result: ∂L/∂b (output_size) - one gradient per output neuron
        // We use a temporary buffer to sum the batch, then accumulate scaled into the persistent gradient
        let mut batch_bias_grad = vec![0.0; self.output_size];
        sum_rows(
            grad_output,
            batch_size,
            self.output_size,
            &mut batch_bias_grad,
        );
        self.grad_biases.accumulate_scaled(&batch_bias_grad, scale);

        // ===== Step 3: Input gradients (backprop to previous layer) =====
        // Compute: ∂L/∂x = ∂L/∂y × W^T
        // Gradient w.r.t. output: ∂L/∂y (batch_size × output_size)
        // Weights transposed: W^T (output_size × input_size)
        // Result: ∂L/∂x (batch_size × input_size)
        assert_eq!(
            grad_input.len(),
            batch_size * self.input_size,
            "DenseLayer backward: grad_input shape mismatch. \
             Expected ({}, {}) = {} elements, got {}. \
             Ensure grad_input buffer has batch_size * input_size elements.",
            batch_size,
            self.input_size,
            batch_size * self.input_size,
            grad_input.len()
        );
        sgemm_wrapper(
            batch_size,
            self.input_size,
            self.output_size,
            grad_output,
            self.output_size,
            &self.weights,
            self.output_size,
            grad_input,
            self.input_size,
            false,
            true,
            1.0,
            0.0,
        );
    }

    /// Applies a gradient-descent update to the layer's parameters and clears accumulated gradients.
    ///
    /// The stored weight and bias gradients are scaled by `learning_rate` and subtracted from the
    /// corresponding parameters. After the update, gradient accumulators are reset to zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let mut layer = DenseLayer::new(2, 3, &mut rng);
    ///
    /// // simulate accumulated gradients
    /// {
    ///     let mut gw = layer.grad_weights.borrow_mut();
    ///     for g in gw.iter_mut() { *g = 0.1; }
    ///     let mut gb = layer.grad_biases.borrow_mut();
    ///     for g in gb.iter_mut() { *g = 0.2; }
    /// }
    ///
    /// let before_w = layer.weights()[0];
    /// let before_b = layer.biases()[0];
    /// layer.update_parameters(0.5);
    /// assert_eq!(layer.weights()[0], before_w - 0.5 * 0.1);
    /// assert_eq!(layer.biases()[0], before_b - 0.5 * 0.2);
    /// ```
    fn update_parameters(&mut self, learning_rate: f32) {
        self.grad_weights
            .apply_sgd_update(&mut self.weights, learning_rate);
        self.grad_biases
            .apply_sgd_update(&mut self.biases, learning_rate);
    }

    fn update_with_optimizer(&mut self, optimizer: &mut dyn crate::optimizers::Optimizer) {
        self.grad_weights
            .apply_optimizer_update(&mut self.weights, optimizer);
        self.grad_biases
            .apply_optimizer_update(&mut self.biases, optimizer);
    }

    /// Number of input features expected by the layer.
    ///
    /// # Returns
    ///
    /// The number of input features.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(4, 2, &mut rng);
    /// assert_eq!(layer.input_size(), 4);
    /// ```
    fn input_size(&self) -> usize {
        self.input_size
    }

    /// Number of output features produced by the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(4, 8, &mut rng);
    /// assert_eq!(layer.output_size(), 8);
    /// ```
    fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the total number of trainable parameters (weights plus biases) in the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(2, 3, &mut rng);
    /// assert_eq!(layer.parameter_count(), 2 * 3 + 3);
    /// ```
    fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Estimated FLOPS for a dense forward pass: `Y = X * W + b`.
    ///
    /// Each of the `batch_size * output_size` output neurons requires
    /// `input_size` multiply-add operations, giving
    /// `2 * batch_size * input_size * output_size` total FLOPS.
    fn flops_forward(&self, batch_size: usize) -> u64 {
        2 * batch_size as u64 * self.input_size as u64 * self.output_size as u64
    }

    /// Estimated FLOPS for a dense backward pass.
    ///
    /// Both `dL/dX = dL/dY * Wᵀ` and `dL/dW = Xᵀ * dL/dY` are matrix
    /// multiplications with the same dimensions as the forward pass, so the
    /// backward FLOPS equal the forward FLOPS.
    fn flops_backward(&self, batch_size: usize) -> u64 {
        2 * batch_size as u64 * self.input_size as u64 * self.output_size as u64
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_creation() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(10, 5, &mut rng);

        assert_eq!(layer.input_size(), 10);
        assert_eq!(layer.output_size(), 5);
        assert_eq!(layer.weights.len(), 50); // 10 × 5
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_dense_layer_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(784, 512, &mut rng);

        // 784 × 512 weights + 512 biases = 401,408 + 512 = 401,920
        assert_eq!(layer.parameter_count(), 784 * 512 + 512);
    }

    #[test]
    fn test_xavier_initialization() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(100, 50, &mut rng);

        // Xavier limit = sqrt(6 / (100 + 50)) = sqrt(6 / 150) ≈ 0.2
        let limit = (6.0f32 / 150.0).sqrt();

        // Check that all weights are within the expected range
        for &weight in &layer.weights {
            assert!(
                weight >= -limit && weight <= limit,
                "Weight {} outside Xavier range [{}, {}]",
                weight,
                -limit,
                limit
            );
        }

        // Check that biases are initialized to zero
        for &bias in &layer.biases {
            assert_eq!(bias, 0.0);
        }
    }

    #[test]
    fn test_deterministic_initialization() {
        let mut rng1 = SimpleRng::new(42);
        let layer1 = DenseLayer::new(10, 5, &mut rng1);

        let mut rng2 = SimpleRng::new(42);
        let layer2 = DenseLayer::new(10, 5, &mut rng2);

        // Same seed should produce identical weights
        assert_eq!(layer1.weights, layer2.weights);
        assert_eq!(layer1.biases, layer2.biases);
    }

    #[test]
    fn test_add_bias() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 cols
        let bias = vec![0.1, 0.2, 0.3];
        add_bias(&mut data, 2, 3, &bias);

        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 2.2).abs() < 1e-6);
        assert!((data[2] - 3.3).abs() < 1e-6);
        assert!((data[3] - 4.1).abs() < 1e-6);
        assert!((data[4] - 5.2).abs() < 1e-6);
        assert!((data[5] - 6.3).abs() < 1e-6);
    }

    #[test]
    fn test_sum_rows() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 cols
        let mut out = vec![0.0; 3];
        sum_rows(&data, 2, 3, &mut out);

        // Column 0: 1 + 4 = 5
        // Column 1: 2 + 5 = 7
        // Column 2: 3 + 6 = 9
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_forward() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(3, 2, &mut rng);

        // Single sample forward pass
        let input = vec![1.0, 0.5, -0.5];
        let mut output = vec![0.0; 2];

        layer.forward(&input, &mut output, 1);

        // Output should be input × weights + biases
        // Verify output is computed (not zeros and finite)
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output.iter().any(|&x| x != 0.0) || layer.biases.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_dense_forward_batch() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(2, 3, &mut rng);

        // Batch of 2 samples
        let input = vec![1.0, 0.0, 0.0, 1.0]; // 2 samples × 2 features
        let mut output = vec![0.0; 6]; // 2 samples × 3 outputs

        layer.forward(&input, &mut output, 2);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dense_backward() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(3, 2, &mut rng);

        let input = vec![1.0, 0.5, -0.5];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output, 1);

        // Create gradient of output
        let grad_output = vec![1.0, -1.0];
        let mut grad_input = vec![0.0; 3];

        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Gradient should propagate back
        assert!(grad_input.iter().all(|&x| x.is_finite()));
        // At least some gradients should be non-zero
        assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_dense_update_parameters() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(3, 2, &mut rng);

        let original_weights = layer.weights.clone();
        let _original_biases = layer.biases.clone();

        // Do a forward and backward pass to accumulate gradients
        let input = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0, 1.0];
        let mut grad_input = vec![0.0; 3];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Update parameters
        layer.update_parameters(0.1);

        // Weights should have changed
        let weights_changed = layer
            .weights
            .iter()
            .zip(original_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(weights_changed, "Weights should change after update");
    }

    #[test]
    fn test_weights_and_biases_accessors() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        assert_eq!(layer.weights().len(), 12); // 4 × 3
        assert_eq!(layer.biases().len(), 3);
    }

    #[test]
    fn test_gradient_magnitude_initially_zero() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(3, 2, &mut rng);

        // Initially, gradients should be zero
        let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
        assert_eq!(weight_norm, 0.0);
        assert_eq!(bias_norm, 0.0);
    }

    #[test]
    fn test_gradient_magnitude_after_backward() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(3, 2, &mut rng);

        // Perform forward pass
        let input = vec![1.0, 0.5, -0.5];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output, 1);

        // Perform backward pass to accumulate gradients
        let grad_output = vec![1.0, -1.0];
        let mut grad_input = vec![0.0; 3];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Get gradient magnitudes
        let (weight_norm, bias_norm) = layer.get_gradient_magnitude();

        // Gradients should be non-zero and non-negative
        assert!(weight_norm >= 0.0);
        assert!(bias_norm >= 0.0);
        assert!(weight_norm > 0.0 || bias_norm > 0.0);
    }

    #[test]
    fn test_gradient_magnitude_after_update() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(3, 2, &mut rng);

        // Perform forward and backward pass
        let input = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0, 1.0];
        let mut grad_input = vec![0.0; 3];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Verify gradients are non-zero
        let (weight_norm_before, bias_norm_before) = layer.get_gradient_magnitude();
        assert!(weight_norm_before > 0.0);
        assert!(bias_norm_before > 0.0);

        // Update parameters (this should clear gradients)
        layer.update_parameters(0.1);

        // After update, gradients should be cleared to zero
        let (weight_norm_after, bias_norm_after) = layer.get_gradient_magnitude();
        assert_eq!(weight_norm_after, 0.0);
        assert_eq!(bias_norm_after, 0.0);
    }

    #[test]
    fn test_gradient_magnitude_accumulation() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(2, 2, &mut rng);

        // First backward pass
        let input = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![0.1, 0.1];
        let mut grad_input = vec![0.0; 2];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        let (weight_norm_first, bias_norm_first) = layer.get_gradient_magnitude();

        // Second backward pass (accumulates gradients)
        layer.forward(&input, &mut output, 1);
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        let (weight_norm_second, bias_norm_second) = layer.get_gradient_magnitude();

        // Gradients should have accumulated (increased)
        assert!(weight_norm_second > weight_norm_first);
        assert!(bias_norm_second > bias_norm_first);
    }

    #[test]
    fn test_dense_new_with_weights_stores_parameters() {
        let input_size = 2;
        let output_size = 3;
        let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 × 3
        let biases = vec![0.1f32, 0.2, 0.3];

        let layer =
            DenseLayer::new_with_weights(input_size, output_size, weights.clone(), biases.clone());

        assert_eq!(layer.input_size(), input_size);
        assert_eq!(layer.output_size(), output_size);
        assert_eq!(layer.weights(), weights.as_slice());
        assert_eq!(layer.biases(), biases.as_slice());
    }

    #[test]
    fn test_dense_new_with_weights_gradient_initially_zero() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 × 2
        let biases = vec![0.5f32, -0.5];

        let layer = DenseLayer::new_with_weights(2, 2, weights, biases);

        // Gradient accumulators should start at zero
        let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
        assert_eq!(weight_norm, 0.0);
        assert_eq!(bias_norm, 0.0);
    }

    #[test]
    fn test_dense_new_with_weights_correct_forward_output() {
        // Use a known weight matrix to verify forward computation
        // Input: [1.0, 0.0], Weights: [[1.0, 2.0], [3.0, 4.0]], Biases: [0.5, -0.5]
        // Expected output: [1.0*1.0 + 0.0*3.0 + 0.5, 1.0*2.0 + 0.0*4.0 + (-0.5)]
        //                = [1.5, 1.5]
        let weights = vec![1.0f32, 2.0, 3.0, 4.0]; // row-major 2×2
        let biases = vec![0.5f32, -0.5];

        let layer = DenseLayer::new_with_weights(2, 2, weights, biases);

        let input = vec![1.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        layer.forward(&input, &mut output, 1);

        assert!((output[0] - 1.5).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 1.5).abs() < 1e-6, "output[1] = {}", output[1]);
    }

    #[test]
    #[should_panic(expected = "weights length")]
    fn test_dense_new_with_weights_wrong_weight_length_panics() {
        // 2×2 layer but only 3 weights provided (should be 4)
        let weights = vec![0.1f32, 0.2, 0.3];
        let biases = vec![0.0f32, 0.0];
        let _layer = DenseLayer::new_with_weights(2, 2, weights, biases);
    }

    #[test]
    #[should_panic(expected = "biases length")]
    fn test_dense_new_with_weights_wrong_bias_length_panics() {
        // 2×2 layer but 3 biases provided (should be 2)
        let weights = vec![0.1f32, 0.2, 0.3, 0.4];
        let biases = vec![0.0f32, 0.0, 0.0];
        let _layer = DenseLayer::new_with_weights(2, 2, weights, biases);
    }
}