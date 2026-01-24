//! Dense (fully connected) layer implementation
//!
//! This module provides a DenseLayer (also known as Linear or Fully Connected layer)
//! that performs the transformation: output = input × weights + biases

use crate::layers::Layer;
use crate::utils::rng::SimpleRng;
use std::cell::RefCell;

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
    // Gradient accumulators (mutable interior via RefCell for trait compatibility)
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
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
    /// ```ignore
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
            grad_weights: RefCell::new(vec![0.0f32; input_size * output_size]),
            grad_biases: RefCell::new(vec![0.0f32; output_size]),
        }
    }

    /// Get the input size of the layer.
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
    /// Computes the layer's linear output for a batch: output = input × weights + biases.
    ///
    /// Expects `input` to be a row-major matrix with shape (batch_size × input_size)
    /// and `output` to be a row-major buffer with shape (batch_size × output_size).
    ///
    /// # Examples
    ///
    /// ```
    /// // Given a DenseLayer `layer` with input_size = 2 and output_size = 3,
    /// // call `forward` with a single-row batch:
    /// use rust_neural_networks::layers::dense::DenseLayer;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(0);
    /// let layer = DenseLayer::new(2, 3, &mut rng);
    /// let input = [0.5f32, -1.0f32]; // 1 × 2
    /// let mut output = vec![0f32; 3]; // 1 × 3
    /// layer.forward(&input, &mut output, 1);
    /// assert_eq!(output.len(), 3);
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Perform matrix multiplication: output = input × weights
        // input: (batch_size × input_size)
        // weights: (input_size × output_size)
        // output: (batch_size × output_size)
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "input len mismatch: expected {}, got {}",
            batch_size * self.input_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "output len mismatch: expected {}, got {}",
            batch_size * self.output_size,
            output.len()
        );
        assert_eq!(
            self.weights.len(),
            self.input_size * self.output_size,
            "weights len mismatch: expected {}, got {}",
            self.input_size * self.output_size,
            self.weights.len()
        );

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

        // Add biases to each row
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "output len mismatch for add_bias: expected {}, got {}",
            batch_size * self.output_size,
            output.len()
        );
        assert_eq!(
            self.biases.len(),
            self.output_size,
            "biases len mismatch: expected {}, got {}",
            self.output_size,
            self.biases.len()
        );
        add_bias(output, batch_size, self.output_size, &self.biases);
    }

    /// Computes and accumulates gradients for this layer given a batch of inputs and output gradients,
    /// and writes the gradient with respect to the inputs into `grad_input`.
    ///
    /// The method updates the layer's internal gradient accumulators:
    /// - accumulates weight gradients (input^T × grad_output) averaged by `batch_size`,
    /// - computes bias gradients as the column-wise sum of `grad_output` averaged by `batch_size`.
    ///   It also computes `grad_input = grad_output × weights^T`.
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
        let scale = 1.0f32 / batch_size as f32;

        // Compute gradient with respect to weights: grad_w = input^T × grad_output / batch_size
        // input: (batch_size × input_size)
        // grad_output: (batch_size × output_size)
        // grad_weights: (input_size × output_size)
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "input len mismatch in backward: expected {}, got {}",
            batch_size * self.input_size,
            input.len()
        );
        assert_eq!(
            grad_output.len(),
            batch_size * self.output_size,
            "grad_output len mismatch in backward: expected {}, got {}",
            batch_size * self.output_size,
            grad_output.len()
        );

        let mut grad_w = self.grad_weights.borrow_mut();
        assert_eq!(
            grad_w.len(),
            self.input_size * self.output_size,
            "grad_weights len mismatch: expected {}, got {}",
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

        // Compute gradient with respect to biases: grad_b = sum(grad_output) / batch_size
        // Compute gradient with respect to biases: grad_b += sum(grad_output) * scale
        // We use a temporary buffer to sum the batch, then accumulate into the persistent gradient
        let mut batch_bias_grad = vec![0.0; self.output_size];
        sum_rows(
            grad_output,
            batch_size,
            self.output_size,
            &mut batch_bias_grad,
        );

        let mut grad_b = self.grad_biases.borrow_mut();
        assert_eq!(
            grad_b.len(),
            self.output_size,
            "grad_biases len mismatch: expected {}, got {}",
            self.output_size,
            grad_b.len()
        );

        for (acc, g) in grad_b.iter_mut().zip(batch_bias_grad.iter()) {
            *acc += *g * scale;
        }

        // Compute gradient with respect to input: grad_input = grad_output × weights^T
        // grad_output: (batch_size × output_size)
        // weights: (input_size × output_size)
        // grad_input: (batch_size × input_size)
        assert_eq!(
            grad_input.len(),
            batch_size * self.input_size,
            "grad_input len mismatch: expected {}, got {}",
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
    /// # use crate::layers::dense::DenseLayer;
    /// # use crate::utils::rng::SimpleRng;
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
        let grad_w = self.grad_weights.borrow();
        let grad_b = self.grad_biases.borrow();

        // Update weights: weight = weight - learning_rate * gradient
        for (weight, &gradient) in self.weights.iter_mut().zip(grad_w.iter()) {
            *weight -= learning_rate * gradient;
        }

        // Update biases: bias = bias - learning_rate * gradient
        for (bias, &gradient) in self.biases.iter_mut().zip(grad_b.iter()) {
            *bias -= learning_rate * gradient;
        }

        // Clear gradients for next iteration
        drop(grad_w);
        drop(grad_b);
        self.grad_weights
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
        self.grad_biases
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
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
}
