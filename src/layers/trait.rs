//! Layer trait definition for neural network layers
//!
//! This module defines the core Layer trait that all layer types must implement.
//! The trait provides a common interface for forward propagation, backward propagation,
//! and parameter updates.

/// Core trait for neural network layers.
///
/// All layer types (Dense, Conv2D, etc.) implement this trait to provide
/// a uniform interface for forward and backward propagation.
///
/// # Type Parameters
///
/// Layers work with f32 data for compatibility with BLAS operations and GPU acceleration.
///
/// # Example
///
/// ```ignore
/// // Forward pass through a layer
/// let mut output = vec![0.0f32; batch_size * output_size];
/// layer.forward(&input, &mut output, batch_size);
///
/// // Backward pass to compute gradients
/// let mut grad_input = vec![0.0f32; batch_size * input_size];
/// layer.backward(&input, &grad_output, &mut grad_input, batch_size);
/// ```
pub trait Layer {
    /// Forward propagation through the layer.
    ///
    /// Computes the layer output given input data. The layer applies its transformation
    /// (e.g., matrix multiplication + bias for dense layers, convolution for conv layers)
    /// and stores any necessary intermediate values for backward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data flattened as a 1D array (batch_size × input_size)
    /// * `output` - Output buffer to store results (batch_size × output_size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Panics
    ///
    /// Implementations may panic if input/output dimensions don't match expected sizes.
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);

    /// Backward propagation through the layer.
    ///
    /// Computes gradients with respect to inputs and parameters given the gradient
    /// of the loss with respect to outputs. This method:
    /// 1. Computes grad_input (gradient with respect to layer inputs)
    /// 2. Accumulates gradients for weights and biases internally
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass (batch_size × input_size)
    /// * `grad_output` - Gradient of loss w.r.t. layer output (batch_size × output_size)
    /// * `grad_input` - Buffer to store gradient w.r.t. input (batch_size × input_size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Notes
    ///
    /// - The input should be the same data used in the corresponding forward pass
    /// - Some layers may need to cache activations from forward pass for gradient computation
    /// - Weight and bias gradients are accumulated internally and applied via `update_parameters`
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    );

    /// Update layer parameters using accumulated gradients.
    ///
    /// Applies the gradient descent update rule to weights and biases:
    /// weight = weight - learning_rate * gradient
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Notes
    ///
    /// - This should be called after one or more backward passes
    /// - Implementations should clear accumulated gradients after updating
    fn update_parameters(&mut self, learning_rate: f32);

    /// Get the input size of the layer.
    ///
    /// Returns the expected number of input features per sample.
    fn input_size(&self) -> usize;

    /// Get the output size of the layer.
    ///
    /// Returns the number of output features per sample.
    fn output_size(&self) -> usize;

    /// Get the number of trainable parameters in the layer.
    ///
    /// Returns the total count of weights and biases.
    /// For example, a dense layer has input_size × output_size weights
    /// plus output_size biases.
    fn parameter_count(&self) -> usize;
}
