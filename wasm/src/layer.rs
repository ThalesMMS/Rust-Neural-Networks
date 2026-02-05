//! Dense (fully connected) layer implementation for WebAssembly
//!
//! This module provides a DenseLayer that works in WASM environments.
//! Unlike the main library version, this uses pure Rust matrix operations
//! instead of BLAS for cross-platform compatibility.

use crate::matrix_ops::{add_bias, matrix_multiply};

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
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl DenseLayer {
    /// Creates a dense layer with pre-initialized weights and biases.
    ///
    /// This constructor is used when loading a trained model from binary data.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    /// * `weights` - Pre-trained weight matrix (input_size × output_size)
    /// * `biases` - Pre-trained bias vector (output_size)
    ///
    /// # Panics
    ///
    /// Panics if weights or biases have incorrect dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::layer::DenseLayer;
    ///
    /// let weights = vec![0.1; 784 * 512];
    /// let biases = vec![0.0; 512];
    /// let layer = DenseLayer::new(784, 512, weights, biases);
    /// assert_eq!(layer.input_size(), 784);
    /// assert_eq!(layer.output_size(), 512);
    /// ```
    pub fn new(input_size: usize, output_size: usize, weights: Vec<f32>, biases: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            input_size * output_size,
            "Weights must have length input_size * output_size"
        );
        assert_eq!(
            biases.len(),
            output_size,
            "Biases must have length output_size"
        );

        Self {
            input_size,
            output_size,
            weights,
            biases,
        }
    }

    /// Get the input size of the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::layer::DenseLayer;
    ///
    /// let layer = DenseLayer::new(128, 64, vec![0.0; 128 * 64], vec![0.0; 64]);
    /// assert_eq!(layer.input_size(), 128);
    /// ```
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the output size of the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::layer::DenseLayer;
    ///
    /// let layer = DenseLayer::new(128, 64, vec![0.0; 128 * 64], vec![0.0; 64]);
    /// assert_eq!(layer.output_size(), 64);
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
    /// use mnist_wasm::layer::DenseLayer;
    ///
    /// let layer = DenseLayer::new(3, 4, vec![0.0; 12], vec![0.0; 4]);
    /// assert_eq!(layer.parameter_count(), 12 + 4);
    /// ```
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Forward propagation through the layer.
    ///
    /// Computes the layer output: output = input × weights + biases
    ///
    /// # Arguments
    ///
    /// * `input` - Input data flattened as a 1D array (batch_size × input_size)
    /// * `output` - Output buffer to store results (batch_size × output_size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Panics
    ///
    /// Panics if input or output dimensions don't match expected sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::layer::DenseLayer;
    ///
    /// let weights = vec![0.5, 0.5, 0.5, 0.5]; // 2x2 matrix
    /// let biases = vec![0.1, 0.2];
    /// let layer = DenseLayer::new(2, 2, weights, biases);
    ///
    /// let input = vec![1.0, 2.0]; // Single sample
    /// let mut output = vec![0.0; 2];
    ///
    /// layer.forward(&input, &mut output, 1);
    ///
    /// // Expected: [1.0*0.5 + 2.0*0.5 + 0.1, 1.0*0.5 + 2.0*0.5 + 0.2]
    /// //         = [1.6, 1.7]
    /// assert!((output[0] - 1.6).abs() < 1e-5);
    /// assert!((output[1] - 1.7).abs() < 1e-5);
    /// ```
    pub fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "Input size mismatch"
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "Output size mismatch"
        );

        // Compute: output = input × weights
        // input: (batch_size × input_size)
        // weights: (input_size × output_size)
        // output: (batch_size × output_size)
        matrix_multiply(
            batch_size,
            self.output_size,
            self.input_size,
            input,
            self.input_size,
            &self.weights,
            self.output_size,
            output,
            self.output_size,
            false, // no transpose on input
            false, // no transpose on weights
            1.0,   // alpha
            0.0,   // beta (overwrite output)
        );

        // Add biases to each row
        add_bias(output, batch_size, self.output_size, &self.biases);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_creation() {
        let weights = vec![0.1; 6]; // 2x3 matrix
        let biases = vec![0.0; 3];
        let layer = DenseLayer::new(2, 3, weights, biases);

        assert_eq!(layer.input_size(), 2);
        assert_eq!(layer.output_size(), 3);
        assert_eq!(layer.parameter_count(), 9); // 6 weights + 3 biases
    }

    #[test]
    fn test_dense_layer_forward() {
        // Create a simple 2x2 layer with known weights
        let weights = vec![
            1.0, 2.0, // First input feature weights
            3.0, 4.0, // Second input feature weights
        ];
        let biases = vec![0.5, 1.0];
        let layer = DenseLayer::new(2, 2, weights, biases);

        // Single sample: [1.0, 2.0]
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];

        layer.forward(&input, &mut output, 1);

        // Expected computation:
        // output[0] = 1.0*1.0 + 2.0*3.0 + 0.5 = 1.0 + 6.0 + 0.5 = 7.5
        // output[1] = 1.0*2.0 + 2.0*4.0 + 1.0 = 2.0 + 8.0 + 1.0 = 11.0
        assert!((output[0] - 7.5).abs() < 1e-5);
        assert!((output[1] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_layer_forward_batch() {
        // 2x2 layer
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let biases = vec![0.0, 0.0];
        let layer = DenseLayer::new(2, 2, weights, biases);

        // Two samples: [[1, 0], [0, 1]]
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 4];

        layer.forward(&input, &mut output, 2);

        // First sample [1, 0]: output = [1, 2]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);

        // Second sample [0, 1]: output = [3, 4]
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_layer_identity() {
        // Identity transformation with zero bias
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix
        let biases = vec![0.0, 0.0];
        let layer = DenseLayer::new(2, 2, weights, biases);

        let input = vec![5.0, 7.0];
        let mut output = vec![0.0; 2];

        layer.forward(&input, &mut output, 1);

        // Identity transformation should preserve input
        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_layer_with_bias() {
        // Zero weights, non-zero bias (should just output bias)
        let weights = vec![0.0; 4]; // 2x2 zeros
        let biases = vec![3.0, 7.0];
        let layer = DenseLayer::new(2, 2, weights, biases);

        let input = vec![100.0, 200.0]; // Input shouldn't matter
        let mut output = vec![0.0; 2];

        layer.forward(&input, &mut output, 1);

        // With zero weights, output should equal bias
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "Weights must have length input_size * output_size")]
    fn test_dense_layer_invalid_weights() {
        let weights = vec![0.0; 5]; // Should be 6 for 2x3
        let biases = vec![0.0; 3];
        DenseLayer::new(2, 3, weights, biases);
    }

    #[test]
    #[should_panic(expected = "Biases must have length output_size")]
    fn test_dense_layer_invalid_biases() {
        let weights = vec![0.0; 6];
        let biases = vec![0.0; 2]; // Should be 3
        DenseLayer::new(2, 3, weights, biases);
    }
}
