//! Dense (fully connected) layer implementation
//!
//! This module provides a DenseLayer (also known as Linear or Fully Connected layer)
//! that performs the transformation: output = input × weights + biases

use crate::utils::SimpleRng;

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
            assert!(weight >= -limit && weight <= limit,
                   "Weight {} outside Xavier range [{}, {}]", weight, -limit, limit);
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
