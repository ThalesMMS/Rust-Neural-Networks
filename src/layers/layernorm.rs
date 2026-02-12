//! Layer normalization layer implementation
//!
//! This module provides a LayerNormLayer that normalizes activations across the feature dimension,
//! improving training stability especially for sequence models like transformers.
//!
//! # Layer Normalization Theory
//!
//! Layer normalization normalizes inputs to have zero mean and unit variance across features
//! (the last dimension) for each sample independently, then applies learnable scale (gamma)
//! and shift (beta) parameters:
//!
//! 1. Compute statistics: mean μ and variance σ² across features for each sample
//! 2. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
//! 3. Scale and shift: y = γ * x_norm + β
//!
//! Unlike batch normalization which normalizes across the batch dimension, layer normalization
//! normalizes across features. This makes it independent of batch size and deterministic
//! (no difference between training and inference modes).
//!
//! # Benefits
//!
//! - **Batch-size independent**: Works with any batch size, even batch_size=1
//! - **Deterministic**: Same behavior in training and inference (no running statistics)
//! - **Effective for sequences**: Better suited for RNNs and transformers than batch norm
//! - **Training stability**: Reduces internal covariate shift within each sample
//!
//! # References
//!
//! Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.

use std::cell::RefCell;

/// Layer normalization layer with learnable scale and shift parameters.
///
/// Normalizes inputs to have zero mean and unit variance per sample across features,
/// then applies learnable affine transformation. Unlike batch normalization, layer
/// normalization has no running statistics and behaves identically during training
/// and inference.
///
/// # Fields
///
/// * `size` - Number of input/output features (layer norm doesn't change dimensions)
/// * `epsilon` - Small constant for numerical stability (prevents division by zero)
/// * `gamma` - Learnable scale parameter (initialized to 1.0)
/// * `beta` - Learnable shift parameter (initialized to 0.0)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::LayerNormLayer;
///
/// let mut layer = LayerNormLayer::new(512, 1e-5);
/// assert_eq!(layer.input_size(), 512);
/// assert_eq!(layer.output_size(), 512);
/// assert_eq!(layer.parameter_count(), 1024);  // 512 gamma + 512 beta
/// ```
pub struct LayerNormLayer {
    size: usize,
    epsilon: f32,

    // Learnable parameters
    gamma: Vec<f32>,
    beta: Vec<f32>,

    // Gradient accumulators (mutable interior via RefCell for trait compatibility)
    grad_gamma: RefCell<Vec<f32>>,
    grad_beta: RefCell<Vec<f32>>,

    // Cached values from forward pass (needed for backward pass)
    cached_input: RefCell<Vec<f32>>,
    cached_mean: RefCell<Vec<f32>>,
    cached_var: RefCell<Vec<f32>>,
    cached_normalized: RefCell<Vec<f32>>,
    cached_std: RefCell<Vec<f32>>,
}

impl LayerNormLayer {
    /// Creates a new layer normalization layer with specified size and epsilon.
    ///
    /// Initializes gamma to 1.0 (no scaling) and beta to 0.0 (no shift). Gradient
    /// accumulators are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of input/output features
    /// * `epsilon` - Small constant for numerical stability (typical: 1e-5)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(256, 1e-5);
    /// assert_eq!(layer.input_size(), 256);
    /// assert_eq!(layer.output_size(), 256);
    /// ```
    pub fn new(size: usize, epsilon: f32) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");

        Self {
            size,
            epsilon,

            // Initialize gamma to 1.0 (identity scaling), beta to 0.0 (no shift)
            gamma: vec![1.0f32; size],
            beta: vec![0.0f32; size],

            // Zero-initialize gradients
            grad_gamma: RefCell::new(vec![0.0f32; size]),
            grad_beta: RefCell::new(vec![0.0f32; size]),

            // Initialize caches (will be resized during forward pass)
            cached_input: RefCell::new(Vec::new()),
            cached_mean: RefCell::new(Vec::new()),
            cached_var: RefCell::new(Vec::new()),
            cached_normalized: RefCell::new(Vec::new()),
            cached_std: RefCell::new(Vec::new()),
        }
    }

    /// Get the epsilon value used for numerical stability.
    ///
    /// # Returns
    ///
    /// The small constant added to variance before taking square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(128, 1e-5);
    /// assert_eq!(layer.epsilon(), 1e-5);
    /// ```
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get the input size of the layer.
    ///
    /// # Returns
    ///
    /// Number of input features.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(256, 1e-5);
    /// assert_eq!(layer.input_size(), 256);
    /// ```
    pub fn input_size(&self) -> usize {
        self.size
    }

    /// Get the output size of the layer.
    ///
    /// For layer normalization, input and output sizes are always the same since
    /// layer norm doesn't change the dimensionality of the data.
    ///
    /// # Returns
    ///
    /// Number of output features (equal to input_size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(512, 1e-5);
    /// assert_eq!(layer.output_size(), 512);
    /// assert_eq!(layer.input_size(), layer.output_size());
    /// ```
    pub fn output_size(&self) -> usize {
        self.size
    }

    /// Get the number of trainable parameters.
    ///
    /// Layer normalization has 2 * size parameters: size gamma (scale) parameters
    /// plus size beta (shift) parameters.
    ///
    /// # Returns
    ///
    /// Total number of trainable parameters (2 * size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(64, 1e-5);
    /// assert_eq!(layer.parameter_count(), 128);  // 64 gamma + 64 beta
    /// ```
    pub fn parameter_count(&self) -> usize {
        2 * self.size // gamma + beta
    }

    /// Immutable view of the layer's gamma (scale) parameters.
    ///
    /// # Returns
    ///
    /// A slice containing the scale parameter for each feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(4, 1e-5);
    /// let gamma = layer.gamma();
    /// assert_eq!(gamma.len(), 4);
    /// // Gamma initialized to 1.0
    /// assert_eq!(gamma[0], 1.0);
    /// ```
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Immutable view of the layer's beta (shift) parameters.
    ///
    /// # Returns
    ///
    /// A slice containing the shift parameter for each feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::layernorm::LayerNormLayer;
    /// let layer = LayerNormLayer::new(4, 1e-5);
    /// let beta = layer.beta();
    /// assert_eq!(beta.len(), 4);
    /// // Beta initialized to 0.0
    /// assert_eq!(beta[0], 0.0);
    /// ```
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }
}

use crate::layers::Layer;
use crate::optimizers::Optimizer;
use std::any::Any;

impl Layer for LayerNormLayer {
    /// Forward propagation through the layer normalization layer.
    ///
    /// Computes per-sample statistics (mean and variance across features), normalizes
    /// the input, and applies learnable scale (gamma) and shift (beta). Behavior is
    /// identical in training and inference modes.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × size)
    /// * `output` - Output buffer (batch_size × size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, layernorm::LayerNormLayer};
    /// let mut layer = LayerNormLayer::new(128, 1e-5);
    /// let input = vec![1.0f32; 128 * 4];  // batch_size = 4
    /// let mut output = vec![0.0f32; 128 * 4];
    /// layer.forward(&input, &mut output, 4);
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let total_size = batch_size * self.size;
        assert_eq!(
            input.len(),
            total_size,
            "input len mismatch: expected {}, got {}",
            total_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            total_size,
            "output len mismatch: expected {}, got {}",
            total_size,
            output.len()
        );

        // Allocate storage for per-sample statistics
        let mut sample_mean = vec![0.0f32; batch_size];
        let mut sample_var = vec![0.0f32; batch_size];
        let mut sample_std = vec![0.0f32; batch_size];
        let mut normalized = vec![0.0f32; total_size];

        // Compute mean for each sample across features
        for i in 0..batch_size {
            let mut sum = 0.0f32;
            for j in 0..self.size {
                sum += input[i * self.size + j];
            }
            sample_mean[i] = sum / self.size as f32;
        }

        // Compute variance for each sample across features
        for i in 0..batch_size {
            let mut sum_sq = 0.0f32;
            for j in 0..self.size {
                let diff = input[i * self.size + j] - sample_mean[i];
                sum_sq += diff * diff;
            }
            sample_var[i] = sum_sq / self.size as f32;
            sample_std[i] = (sample_var[i] + self.epsilon).sqrt();
        }

        // Normalize and apply scale/shift
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                normalized[idx] = (input[idx] - sample_mean[i]) / sample_std[i];
                output[idx] = self.gamma[j] * normalized[idx] + self.beta[j];
            }
        }

        // Cache values needed for backward pass
        *self.cached_input.borrow_mut() = input.to_vec();
        *self.cached_mean.borrow_mut() = sample_mean;
        *self.cached_var.borrow_mut() = sample_var;
        *self.cached_std.borrow_mut() = sample_std;
        *self.cached_normalized.borrow_mut() = normalized;
    }

    /// Backward propagation through the layer normalization layer.
    ///
    /// Computes gradients with respect to inputs, gamma, and beta using cached values
    /// from the forward pass. Accumulates gradients for gamma and beta internally.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass (batch_size × size)
    /// * `grad_output` - Gradient of loss w.r.t. layer output (batch_size × size)
    /// * `grad_input` - Buffer to store gradient w.r.t. input (batch_size × size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, layernorm::LayerNormLayer};
    /// let mut layer = LayerNormLayer::new(128, 1e-5);
    ///
    /// let input = vec![1.0f32; 128 * 4];
    /// let mut output = vec![0.0f32; 128 * 4];
    /// layer.forward(&input, &mut output, 4);
    ///
    /// let grad_output = vec![1.0f32; 128 * 4];
    /// let mut grad_input = vec![0.0f32; 128 * 4];
    /// layer.backward(&input, &grad_output, &mut grad_input, 4);
    /// ```
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let total_size = batch_size * self.size;
        assert_eq!(
            grad_output.len(),
            total_size,
            "grad_output len mismatch: expected {}, got {}",
            total_size,
            grad_output.len()
        );
        assert_eq!(
            grad_input.len(),
            total_size,
            "grad_input len mismatch: expected {}, got {}",
            total_size,
            grad_input.len()
        );

        // Retrieve cached values from forward pass
        let normalized = self.cached_normalized.borrow();
        let std = self.cached_std.borrow();
        let mean = self.cached_mean.borrow();
        let input = self.cached_input.borrow();

        let mut grad_gamma = self.grad_gamma.borrow_mut();
        let mut grad_beta = self.grad_beta.borrow_mut();

        // Compute gradients for gamma and beta (accumulated across batch)
        let scale = 1.0 / batch_size as f32;
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                grad_gamma[j] += grad_output[idx] * normalized[idx] * scale;
                grad_beta[j] += grad_output[idx] * scale;
            }
        }

        // For each sample, compute gradient with respect to input
        // This requires computing gradients through the normalization operation
        for i in 0..batch_size {
            // Compute gradient with respect to normalized values
            let mut grad_normalized = vec![0.0f32; self.size];
            for j in 0..self.size {
                let idx = i * self.size + j;
                grad_normalized[j] = grad_output[idx] * self.gamma[j];
            }

            // Compute gradient with respect to variance
            let mut grad_var = 0.0f32;
            for j in 0..self.size {
                let idx = i * self.size + j;
                let x_centered = input[idx] - mean[i];
                grad_var += grad_normalized[j] * x_centered * (-0.5) * (std[i].powi(3)).recip();
            }

            // Compute gradient with respect to mean
            let mut grad_mean = 0.0f32;
            for j in 0..self.size {
                grad_mean += grad_normalized[j] * (-1.0 / std[i]);
            }

            // Add contribution from variance gradient
            let mut sum_centered = 0.0f32;
            for j in 0..self.size {
                let idx = i * self.size + j;
                sum_centered += input[idx] - mean[i];
            }
            grad_mean += grad_var * (-2.0 * sum_centered / self.size as f32);

            // Compute gradient with respect to input
            for j in 0..self.size {
                let idx = i * self.size + j;
                let x_centered = input[idx] - mean[i];
                grad_input[idx] = grad_normalized[j] / std[i]
                    + grad_var * 2.0 * x_centered / self.size as f32
                    + grad_mean / self.size as f32;
            }
        }
    }

    /// Update layer parameters using accumulated gradients.
    ///
    /// Applies gradient descent to gamma and beta parameters and clears
    /// accumulated gradients.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, layernorm::LayerNormLayer};
    /// let mut layer = LayerNormLayer::new(64, 1e-5);
    /// // ... forward and backward passes ...
    /// layer.update_parameters(0.01);
    /// ```
    fn update_parameters(&mut self, learning_rate: f32) {
        let grad_gamma = self.grad_gamma.borrow();
        let grad_beta = self.grad_beta.borrow();

        // Update gamma: gamma = gamma - learning_rate * gradient
        for (param, &gradient) in self.gamma.iter_mut().zip(grad_gamma.iter()) {
            *param -= learning_rate * gradient;
        }

        // Update beta: beta = beta - learning_rate * gradient
        for (param, &gradient) in self.beta.iter_mut().zip(grad_beta.iter()) {
            *param -= learning_rate * gradient;
        }

        // Clear gradients for next iteration
        drop(grad_gamma);
        drop(grad_beta);
        self.grad_gamma
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
        self.grad_beta
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
    }

    /// Update layer parameters using an optimizer.
    ///
    /// Applies the optimizer's update rule to gamma and beta parameters and clears
    /// accumulated gradients.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an optimizer implementing the Optimizer trait
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, layernorm::LayerNormLayer};
    /// use rust_neural_networks::optimizers::Adam;
    /// let mut layer = LayerNormLayer::new(64, 1e-5);
    /// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// // ... forward and backward passes ...
    /// layer.update_with_optimizer(&mut optimizer);
    /// ```
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        let grad_gamma = self.grad_gamma.borrow();
        let grad_beta = self.grad_beta.borrow();

        // Update gamma using optimizer
        optimizer.update(&mut self.gamma, &grad_gamma);

        // Update beta using optimizer
        optimizer.update(&mut self.beta, &grad_beta);

        // Clear gradients for next iteration
        drop(grad_gamma);
        drop(grad_beta);
        self.grad_gamma
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
        self.grad_beta
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
    }

    fn input_size(&self) -> usize {
        self.size
    }

    fn output_size(&self) -> usize {
        self.size
    }

    fn parameter_count(&self) -> usize {
        2 * self.size
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
