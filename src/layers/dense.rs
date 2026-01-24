//! Dense (fully connected) layer implementation
//!
//! This module provides a DenseLayer (also known as Linear or Fully Connected layer)
//! that performs the transformation: output = input × weights + biases

use crate::layers::Layer;
use crate::utils::SimpleRng;
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
    /// Create a new DenseLayer with Xavier initialization.
    ///
    /// Weights are initialized using Xavier/Glorot initialization:
    /// randomly sampled from uniform distribution [-limit, limit]
    /// where limit = sqrt(6 / (input_size + output_size)).
    ///
    /// Biases are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// A new DenseLayer with randomly initialized weights and zero biases
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(128, 64, &mut rng);
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

    /// Get the output size of the layer.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get the number of trainable parameters.
    ///
    /// Returns input_size × output_size (weights) + output_size (biases).
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Get a reference to the layer's weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get a reference to the layer's biases.
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }
}

// Helper functions for BLAS operations

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

fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

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
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Perform matrix multiplication: output = input × weights
        // input: (batch_size × input_size)
        // weights: (input_size × output_size)
        // output: (batch_size × output_size)
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
        add_bias(output, batch_size, self.output_size, &self.biases);
    }

    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let scale = 1.0f32 / batch_size as f32;

        // Compute gradient with respect to weights: grad_w = input^T × grad_output / batch_size
        // input: (batch_size × input_size)
        // grad_output: (batch_size × output_size)
        // grad_weights: (input_size × output_size)
        let mut grad_w = self.grad_weights.borrow_mut();
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
        let mut grad_b = self.grad_biases.borrow_mut();
        sum_rows(grad_output, batch_size, self.output_size, &mut grad_b);
        for bias_grad in grad_b.iter_mut() {
            *bias_grad *= scale;
        }

        // Compute gradient with respect to input: grad_input = grad_output × weights^T
        // grad_output: (batch_size × output_size)
        // weights: (input_size × output_size)
        // grad_input: (batch_size × input_size)
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

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

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
}
