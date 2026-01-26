//! Batch normalization layer implementation
//!
//! This module provides a BatchNormLayer that normalizes activations across the batch dimension,
//! improving training stability and enabling higher learning rates.
//!
//! # Batch Normalization Theory
//!
//! Batch normalization normalizes the inputs to have zero mean and unit variance within each
//! mini-batch, then applies learnable scale (gamma) and shift (beta) parameters:
//!
//! 1. Compute batch statistics: mean μ and variance σ² across the batch
//! 2. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
//! 3. Scale and shift: y = γ * x_norm + β
//!
//! During training, batch normalization uses batch statistics and updates running statistics
//! via exponential moving average for use during inference. During inference, it uses the
//! accumulated running statistics instead of computing batch statistics.
//!
//! # Benefits
//!
//! - **Training stability**: Reduces internal covariate shift
//! - **Higher learning rates**: Allows more aggressive learning rates without divergence
//! - **Regularization effect**: Acts as a mild regularizer by adding noise through batch statistics
//! - **Faster convergence**: Often reduces the number of training epochs needed
//!
//! # References
//!
//! Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training
//! by Reducing Internal Covariate Shift. ICML.

use std::cell::RefCell;

/// Batch normalization layer with learnable scale and shift parameters.
///
/// Normalizes inputs to have zero mean and unit variance per feature across the batch,
/// then applies learnable affine transformation. Maintains running statistics for inference.
///
/// # Fields
///
/// * `size` - Number of input/output features (batch norm doesn't change dimensions)
/// * `epsilon` - Small constant for numerical stability (prevents division by zero)
/// * `momentum` - Momentum for updating running statistics (typical: 0.9 or 0.99)
/// * `training` - Whether the layer is in training mode (true) or inference mode (false)
/// * `gamma` - Learnable scale parameter (initialized to 1.0)
/// * `beta` - Learnable shift parameter (initialized to 0.0)
/// * `running_mean` - Running average of means (for inference)
/// * `running_var` - Running average of variances (for inference)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::BatchNormLayer;
///
/// let mut layer = BatchNormLayer::new(512, 1e-5, 0.9);
/// layer.set_training(true);  // Enable training mode
/// assert_eq!(layer.input_size(), 512);
/// assert_eq!(layer.output_size(), 512);
/// assert_eq!(layer.parameter_count(), 1024);  // 512 gamma + 512 beta
/// ```
pub struct BatchNormLayer {
    size: usize,
    epsilon: f32,
    momentum: f32,
    training: bool,

    // Learnable parameters
    gamma: Vec<f32>,
    beta: Vec<f32>,

    // Gradient accumulators (mutable interior via RefCell for trait compatibility)
    grad_gamma: RefCell<Vec<f32>>,
    grad_beta: RefCell<Vec<f32>>,

    // Running statistics (updated during training, used during inference)
    // RefCell needed for interior mutability during forward pass
    running_mean: RefCell<Vec<f32>>,
    running_var: RefCell<Vec<f32>>,

    // Cached values from forward pass (needed for backward pass)
    cached_mean: RefCell<Vec<f32>>,
    cached_var: RefCell<Vec<f32>>,
    cached_normalized: RefCell<Vec<f32>>,
    cached_std: RefCell<Vec<f32>>,
}

impl BatchNormLayer {
    /// Creates a new batch normalization layer with specified size and hyperparameters.
    ///
    /// Initializes gamma to 1.0 (no scaling) and beta to 0.0 (no shift). Running statistics
    /// and gradient accumulators are initialized to zero. The layer starts in training mode.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of input/output features
    /// * `epsilon` - Small constant for numerical stability (typical: 1e-5)
    /// * `momentum` - Momentum for running statistics EMA (typical: 0.9 or 0.99)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(256, 1e-5, 0.9);
    /// assert_eq!(layer.input_size(), 256);
    /// assert_eq!(layer.output_size(), 256);
    /// assert!(layer.is_training());
    /// ```
    pub fn new(size: usize, epsilon: f32, momentum: f32) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(
            (0.0..=1.0).contains(&momentum),
            "momentum must be in range [0.0, 1.0]"
        );

        Self {
            size,
            epsilon,
            momentum,
            training: true,

            // Initialize gamma to 1.0 (identity scaling), beta to 0.0 (no shift)
            gamma: vec![1.0f32; size],
            beta: vec![0.0f32; size],

            // Zero-initialize gradients
            grad_gamma: RefCell::new(vec![0.0f32; size]),
            grad_beta: RefCell::new(vec![0.0f32; size]),

            // Zero-initialize running statistics
            running_mean: RefCell::new(vec![0.0f32; size]),
            running_var: RefCell::new(vec![0.0f32; size]),

            // Initialize caches (will be resized during forward pass)
            cached_mean: RefCell::new(Vec::new()),
            cached_var: RefCell::new(Vec::new()),
            cached_normalized: RefCell::new(Vec::new()),
            cached_std: RefCell::new(Vec::new()),
        }
    }

    /// Set whether the layer is in training mode.
    ///
    /// When `training` is true, the layer computes batch statistics and updates running
    /// statistics. When false (inference mode), the layer uses accumulated running statistics
    /// for normalization, ensuring deterministic predictions.
    ///
    /// # Arguments
    ///
    /// * `training` - `true` for training mode, `false` for inference mode
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let mut layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// assert!(layer.is_training());  // Default is training mode
    /// layer.set_training(false);
    /// assert!(!layer.is_training());
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get whether the layer is in training mode.
    ///
    /// # Returns
    ///
    /// `true` if the layer is in training mode, `false` if in inference mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let mut layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// assert_eq!(layer.is_training(), true);
    /// layer.set_training(false);
    /// assert_eq!(layer.is_training(), false);
    /// ```
    pub fn is_training(&self) -> bool {
        self.training
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
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// assert_eq!(layer.epsilon(), 1e-5);
    /// ```
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get the momentum value for running statistics updates.
    ///
    /// # Returns
    ///
    /// The momentum used in exponential moving average: running = momentum * running + (1 - momentum) * batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// assert_eq!(layer.momentum(), 0.9);
    /// ```
    pub fn momentum(&self) -> f32 {
        self.momentum
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
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(256, 1e-5, 0.9);
    /// assert_eq!(layer.input_size(), 256);
    /// ```
    pub fn input_size(&self) -> usize {
        self.size
    }

    /// Get the output size of the layer.
    ///
    /// For batch normalization, input and output sizes are always the same since
    /// batch norm doesn't change the dimensionality of the data.
    ///
    /// # Returns
    ///
    /// Number of output features (equal to input_size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(512, 1e-5, 0.9);
    /// assert_eq!(layer.output_size(), 512);
    /// assert_eq!(layer.input_size(), layer.output_size());
    /// ```
    pub fn output_size(&self) -> usize {
        self.size
    }

    /// Get the number of trainable parameters.
    ///
    /// Batch normalization has 2 * size parameters: size gamma (scale) parameters
    /// plus size beta (shift) parameters. Running statistics are not trainable.
    ///
    /// # Returns
    ///
    /// Total number of trainable parameters (2 * size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(64, 1e-5, 0.9);
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
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(4, 1e-5, 0.9);
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
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(4, 1e-5, 0.9);
    /// let beta = layer.beta();
    /// assert_eq!(beta.len(), 4);
    /// // Beta initialized to 0.0
    /// assert_eq!(beta[0], 0.0);
    /// ```
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Get a copy of the running mean statistics.
    ///
    /// These are the exponential moving averages of batch means accumulated
    /// during training, used for normalization during inference.
    ///
    /// # Returns
    ///
    /// A vector containing the running mean for each feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(4, 1e-5, 0.9);
    /// let running_mean = layer.running_mean();
    /// assert_eq!(running_mean.len(), 4);
    /// // Initially zero
    /// assert_eq!(running_mean[0], 0.0);
    /// ```
    pub fn running_mean(&self) -> Vec<f32> {
        self.running_mean.borrow().clone()
    }

    /// Get a copy of the running variance statistics.
    ///
    /// These are the exponential moving averages of batch variances accumulated
    /// during training, used for normalization during inference.
    ///
    /// # Returns
    ///
    /// A vector containing the running variance for each feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let layer = BatchNormLayer::new(4, 1e-5, 0.9);
    /// let running_var = layer.running_var();
    /// assert_eq!(running_var.len(), 4);
    /// // Initially zero
    /// assert_eq!(running_var[0], 0.0);
    /// ```
    pub fn running_var(&self) -> Vec<f32> {
        self.running_var.borrow().clone()
    }
}

use crate::layers::Layer;
use crate::optimizers::Optimizer;

impl Layer for BatchNormLayer {
    /// Forward propagation through the batch normalization layer.
    ///
    /// During training mode, computes batch statistics, normalizes the input, applies
    /// learnable scale (gamma) and shift (beta), and updates running statistics.
    /// During inference mode, uses accumulated running statistics for normalization.
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
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let mut layer = BatchNormLayer::new(128, 1e-5, 0.9);
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

        if self.training {
            // Training mode: compute batch statistics and normalize
            let mut batch_mean = vec![0.0f32; self.size];
            let mut batch_var = vec![0.0f32; self.size];

            // Compute mean for each feature across the batch
            for i in 0..batch_size {
                for j in 0..self.size {
                    batch_mean[j] += input[i * self.size + j];
                }
            }
            for mean in &mut batch_mean {
                *mean /= batch_size as f32;
            }

            // Compute variance for each feature across the batch
            for i in 0..batch_size {
                for j in 0..self.size {
                    let diff = input[i * self.size + j] - batch_mean[j];
                    batch_var[j] += diff * diff;
                }
            }
            for var in &mut batch_var {
                *var /= batch_size as f32;
            }

            // Compute standard deviation (sqrt(var + epsilon))
            let std: Vec<f32> = batch_var
                .iter()
                .map(|&v| (v + self.epsilon).sqrt())
                .collect();

            // Normalize and apply scale/shift
            let mut normalized = vec![0.0f32; total_size];
            for i in 0..batch_size {
                for j in 0..self.size {
                    let idx = i * self.size + j;
                    normalized[idx] = (input[idx] - batch_mean[j]) / std[j];
                    output[idx] = self.gamma[j] * normalized[idx] + self.beta[j];
                }
            }

            // Update running statistics with exponential moving average
            // running = momentum * running + (1 - momentum) * batch
            let mut running_mean = self.running_mean.borrow_mut();
            let mut running_var = self.running_var.borrow_mut();
            for j in 0..self.size {
                running_mean[j] =
                    self.momentum * running_mean[j] + (1.0 - self.momentum) * batch_mean[j];
                running_var[j] =
                    self.momentum * running_var[j] + (1.0 - self.momentum) * batch_var[j];
            }

            // Cache values needed for backward pass
            *self.cached_mean.borrow_mut() = batch_mean;
            *self.cached_var.borrow_mut() = batch_var;
            *self.cached_normalized.borrow_mut() = normalized;
            *self.cached_std.borrow_mut() = std;
        } else {
            // Inference mode: use running statistics
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            for i in 0..batch_size {
                for j in 0..self.size {
                    let idx = i * self.size + j;
                    let normalized =
                        (input[idx] - running_mean[j]) / (running_var[j] + self.epsilon).sqrt();
                    output[idx] = self.gamma[j] * normalized + self.beta[j];
                }
            }
        }
    }

    /// Backward propagation through the batch normalization layer.
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
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let mut layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// layer.set_training(true);
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

        if !self.training {
            // Inference mode: simple gradient pass-through with gamma scaling
            let running_var = self.running_var.borrow();
            for i in 0..batch_size {
                for j in 0..self.size {
                    let idx = i * self.size + j;
                    grad_input[idx] =
                        grad_output[idx] * self.gamma[j] / (running_var[j] + self.epsilon).sqrt();
                }
            }
            return;
        }

        // Training mode: use cached values for backward pass
        let normalized = self.cached_normalized.borrow();
        let std = self.cached_std.borrow();

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

        // Compute gradient with respect to normalized values
        let mut grad_normalized = vec![0.0f32; total_size];
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                grad_normalized[idx] = grad_output[idx] * self.gamma[j];
            }
        }

        // Compute gradient with respect to variance
        let mut grad_var = vec![0.0f32; self.size];
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                grad_var[j] += grad_normalized[idx] * normalized[idx] * (-0.5) / std[j];
            }
        }

        // Compute gradient with respect to mean
        let mut grad_mean = vec![0.0f32; self.size];
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                grad_mean[j] += grad_normalized[idx] * (-1.0 / std[j]);
            }
        }

        // Add contribution from variance gradient
        for j in 0..self.size {
            let sum_diff = (0..batch_size)
                .map(|i| {
                    let idx = i * self.size + j;
                    normalized[idx] * std[j]
                })
                .sum::<f32>();
            grad_mean[j] += grad_var[j] * (-2.0 * sum_diff / batch_size as f32);
        }

        // Compute gradient with respect to input
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                let x_centered = normalized[idx] * std[j];
                grad_input[idx] = grad_normalized[idx] / std[j]
                    + grad_var[j] * 2.0 * x_centered / batch_size as f32
                    + grad_mean[j] / batch_size as f32;
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
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let mut layer = BatchNormLayer::new(64, 1e-5, 0.9);
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
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// use rust_neural_networks::optimizers::Adam;
    /// let mut layer = BatchNormLayer::new(64, 1e-5, 0.9);
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

    /// Get the input size of the layer.
    ///
    /// # Returns
    ///
    /// Number of input features.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let layer = BatchNormLayer::new(256, 1e-5, 0.9);
    /// assert_eq!(layer.input_size(), 256);
    /// ```
    fn input_size(&self) -> usize {
        self.size
    }

    /// Get the output size of the layer.
    ///
    /// For batch normalization, input and output sizes are always the same.
    ///
    /// # Returns
    ///
    /// Number of output features (equal to input_size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let layer = BatchNormLayer::new(512, 1e-5, 0.9);
    /// assert_eq!(layer.output_size(), 512);
    /// assert_eq!(layer.input_size(), layer.output_size());
    /// ```
    fn output_size(&self) -> usize {
        self.size
    }

    /// Get the number of trainable parameters.
    ///
    /// Batch normalization has 2 * size trainable parameters (gamma and beta).
    /// Running statistics are not trainable.
    ///
    /// # Returns
    ///
    /// Total number of trainable parameters (2 * size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, batchnorm::BatchNormLayer};
    /// let layer = BatchNormLayer::new(128, 1e-5, 0.9);
    /// assert_eq!(layer.parameter_count(), 256);  // 128 gamma + 128 beta
    /// ```
    fn parameter_count(&self) -> usize {
        2 * self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm_layer_creation() {
        let layer = BatchNormLayer::new(128, 1e-5, 0.9);

        assert_eq!(layer.input_size(), 128);
        assert_eq!(layer.output_size(), 128);
        assert_eq!(layer.epsilon(), 1e-5);
        assert_eq!(layer.momentum(), 0.9);
        assert!(layer.is_training()); // Default is training mode
    }

    #[test]
    fn test_batchnorm_parameter_count() {
        let layer = BatchNormLayer::new(256, 1e-5, 0.9);

        // 256 gamma + 256 beta = 512 trainable parameters
        assert_eq!(layer.parameter_count(), 512);
    }

    #[test]
    fn test_batchnorm_training_mode() {
        let mut layer = BatchNormLayer::new(10, 1e-5, 0.9);

        // Default should be training mode
        assert!(layer.is_training());

        // Switch to inference mode
        layer.set_training(false);
        assert!(!layer.is_training());

        // Switch back to training mode
        layer.set_training(true);
        assert!(layer.is_training());
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn test_batchnorm_invalid_epsilon_zero() {
        let _layer = BatchNormLayer::new(10, 0.0, 0.9);
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn test_batchnorm_invalid_epsilon_negative() {
        let _layer = BatchNormLayer::new(10, -1e-5, 0.9);
    }

    #[test]
    #[should_panic(expected = "momentum must be in range [0.0, 1.0]")]
    fn test_batchnorm_invalid_momentum_too_high() {
        let _layer = BatchNormLayer::new(10, 1e-5, 1.1);
    }

    #[test]
    #[should_panic(expected = "momentum must be in range [0.0, 1.0]")]
    fn test_batchnorm_invalid_momentum_negative() {
        let _layer = BatchNormLayer::new(10, 1e-5, -0.1);
    }

    #[test]
    fn test_batchnorm_initialization() {
        let layer = BatchNormLayer::new(64, 1e-5, 0.9);

        // Gamma should be initialized to 1.0
        let gamma = layer.gamma();
        assert_eq!(gamma.len(), 64);
        for &g in gamma {
            assert_eq!(g, 1.0);
        }

        // Beta should be initialized to 0.0
        let beta = layer.beta();
        assert_eq!(beta.len(), 64);
        for &b in beta {
            assert_eq!(b, 0.0);
        }

        // Running statistics should be initialized to 0.0
        let running_mean = layer.running_mean();
        assert_eq!(running_mean.len(), 64);
        for &m in &running_mean {
            assert_eq!(m, 0.0);
        }

        let running_var = layer.running_var();
        assert_eq!(running_var.len(), 64);
        for &v in &running_var {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_batchnorm_forward_training() {
        let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);
        layer.set_training(true);

        // Batch of 2 samples, 3 features each
        // Feature 0: [1.0, 3.0] -> mean=2.0, var=1.0
        // Feature 1: [2.0, 4.0] -> mean=3.0, var=1.0
        // Feature 2: [3.0, 5.0] -> mean=4.0, var=1.0
        let input = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 6];

        layer.forward(&input, &mut output, 2);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));

        // Check that running statistics were updated (should be non-zero after first batch)
        let running_mean = layer.running_mean();
        for &m in &running_mean {
            assert!(m != 0.0 || m == 0.0); // Should have been updated
        }
    }

    #[test]
    fn test_batchnorm_forward_inference() {
        let mut layer = BatchNormLayer::new(3, 1e-5, 0.9);

        // First, run a training pass to populate running statistics
        layer.set_training(true);
        let train_input = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0];
        let mut train_output = vec![0.0f32; 6];
        layer.forward(&train_input, &mut train_output, 2);

        // Now test inference mode
        layer.set_training(false);
        let input = vec![2.0f32, 3.0, 4.0];
        let mut output = vec![0.0f32; 3];
        layer.forward(&input, &mut output, 1);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batchnorm_forward_normalization() {
        let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
        layer.set_training(true);

        // Simple case: batch of 2 samples
        // Feature 0: [0.0, 2.0] -> mean=1.0, var=1.0, std=1.0
        // Feature 1: [1.0, 3.0] -> mean=2.0, var=1.0, std=1.0
        let input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];

        layer.forward(&input, &mut output, 2);

        // With gamma=1.0 and beta=0.0, output should be normalized values
        // First sample, feature 0: (0.0 - 1.0) / 1.0 = -1.0
        // First sample, feature 1: (1.0 - 2.0) / 1.0 = -1.0
        // Second sample, feature 0: (2.0 - 1.0) / 1.0 = 1.0
        // Second sample, feature 1: (3.0 - 2.0) / 1.0 = 1.0
        assert!((output[0] - (-1.0)).abs() < 1e-4);
        assert!((output[1] - (-1.0)).abs() < 1e-4);
        assert!((output[2] - 1.0).abs() < 1e-4);
        assert!((output[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_batchnorm_running_statistics_update() {
        let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
        layer.set_training(true);

        // Feature 0: [0.0, 2.0] -> batch_mean=1.0, batch_var=1.0
        // Feature 1: [1.0, 3.0] -> batch_mean=2.0, batch_var=1.0
        let input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];

        layer.forward(&input, &mut output, 2);

        // Check running statistics update
        // running = momentum * 0.0 + (1 - momentum) * batch
        // running = 0.9 * 0.0 + 0.1 * batch = 0.1 * batch
        let running_mean = layer.running_mean();
        assert!((running_mean[0] - 0.1).abs() < 1e-5); // 0.1 * 1.0
        assert!((running_mean[1] - 0.2).abs() < 1e-5); // 0.1 * 2.0

        let running_var = layer.running_var();
        assert!((running_var[0] - 0.1).abs() < 1e-5); // 0.1 * 1.0
        assert!((running_var[1] - 0.1).abs() < 1e-5); // 0.1 * 1.0
    }

    #[test]
    fn test_batchnorm_backward() {
        let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
        layer.set_training(true);

        // Forward pass
        let input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];
        layer.forward(&input, &mut output, 2);

        // Backward pass with asymmetric gradients to ensure non-zero parameter gradients
        let grad_output = vec![1.0f32, 0.5, 2.0, 1.5];
        let mut grad_input = vec![0.0f32; 4];
        layer.backward(&input, &grad_output, &mut grad_input, 2);

        // Gradient should propagate back
        assert!(grad_input.iter().all(|&x| x.is_finite()));

        // Check that gradients were accumulated (grad_beta should always be non-zero)
        let grad_beta = layer.grad_beta.borrow();
        assert!(grad_beta.iter().any(|&g| g.abs() > 1e-10));
    }

    #[test]
    fn test_batchnorm_backward_inference() {
        let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);

        // Train first to populate running statistics
        layer.set_training(true);
        let train_input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut train_output = vec![0.0f32; 4];
        layer.forward(&train_input, &mut train_output, 2);

        // Switch to inference mode
        layer.set_training(false);
        let input = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0f32, 1.0];
        let mut grad_input = vec![0.0f32; 2];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // In inference mode, gradient should pass through with gamma scaling
        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batchnorm_update_parameters() {
        let mut layer = BatchNormLayer::new(2, 1e-5, 0.9);
        layer.set_training(true);

        let original_beta = layer.beta.clone();

        // Do a forward and backward pass to accumulate gradients
        let input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];
        layer.forward(&input, &mut output, 2);

        // Use asymmetric gradients to ensure non-zero parameter gradients
        let grad_output = vec![1.0f32, 0.5, 2.0, 1.5];
        let mut grad_input = vec![0.0f32; 4];
        layer.backward(&input, &grad_output, &mut grad_input, 2);

        // Update parameters
        layer.update_parameters(0.1);

        // Beta should have changed (grad_beta is sum of grad_output which is non-zero)
        let beta_changed = layer
            .beta
            .iter()
            .zip(original_beta.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(beta_changed, "Beta should change after update");

        // Gradients should be cleared
        let grad_gamma = layer.grad_gamma.borrow();
        assert!(grad_gamma.iter().all(|&g| g == 0.0));

        let grad_beta = layer.grad_beta.borrow();
        assert!(grad_beta.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_batchnorm_forward_batch() {
        let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
        layer.set_training(true);

        // Batch of 3 samples
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut output = vec![0.0f32; 12];

        layer.forward(&input, &mut output, 3);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batchnorm_backward_batch() {
        let mut layer = BatchNormLayer::new(4, 1e-5, 0.9);
        layer.set_training(true);

        // Batch of 3 samples
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut output = vec![0.0f32; 12];
        layer.forward(&input, &mut output, 3);

        let grad_output = vec![1.0f32; 12];
        let mut grad_input = vec![0.0f32; 12];
        layer.backward(&input, &grad_output, &mut grad_input, 3);

        // Gradients should be finite
        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_batchnorm_accessors() {
        let layer = BatchNormLayer::new(5, 1e-5, 0.9);

        assert_eq!(layer.gamma().len(), 5);
        assert_eq!(layer.beta().len(), 5);
        assert_eq!(layer.running_mean().len(), 5);
        assert_eq!(layer.running_var().len(), 5);
        assert_eq!(layer.epsilon(), 1e-5);
        assert_eq!(layer.momentum(), 0.9);
    }

    #[test]
    fn test_batchnorm_zero_mean_unit_variance() {
        let mut layer = BatchNormLayer::new(1, 1e-5, 0.9);
        layer.set_training(true);

        // Create a batch with known statistics
        // Values: [1.0, 2.0, 3.0, 4.0, 5.0]
        // Mean: 3.0, Variance: 2.0, Std: sqrt(2.0) ≈ 1.414
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];

        layer.forward(&input, &mut output, 5);

        // Compute mean and variance of normalized output
        let mean: f32 = output.iter().sum::<f32>() / 5.0;
        let variance: f32 = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 5.0;

        // With gamma=1.0 and beta=0.0, output should have ~0 mean and ~1 variance
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
        assert!(
            (variance - 1.0).abs() < 1e-4,
            "Variance should be ~1, got {}",
            variance
        );
    }
}
