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

use crate::layers::gradient::GradientAccumulator;
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

    // Gradient accumulators
    grad_gamma: GradientAccumulator,
    grad_beta: GradientAccumulator,

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
            grad_gamma: GradientAccumulator::new(size),
            grad_beta: GradientAccumulator::new(size),

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

    /// Creates a new batch normalization layer with pre-loaded parameters.
    ///
    /// Used when loading a saved model from disk. Accepts all learnable parameters
    /// and running statistics directly, skipping the default initialization.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of input/output features
    /// * `epsilon` - Small constant for numerical stability (typical: 1e-5)
    /// * `momentum` - Momentum for running statistics EMA (typical: 0.9 or 0.99)
    /// * `gamma` - Learnable scale parameters (length must equal `size`)
    /// * `beta` - Learnable shift parameters (length must equal `size`)
    /// * `running_mean` - Accumulated running mean from training (length must equal `size`)
    /// * `running_var` - Accumulated running variance from training (length must equal `size`)
    ///
    /// # Panics
    ///
    /// Panics if any of the provided vectors have a length that does not match `size`,
    /// or if `epsilon` is not positive, or if `momentum` is outside `[0.0, 1.0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::batchnorm::BatchNormLayer;
    /// let gamma = vec![1.5f32, 2.0, 0.5];
    /// let beta = vec![0.1f32, -0.2, 0.3];
    /// let running_mean = vec![0.5f32, 1.0, -0.5];
    /// let running_var = vec![1.0f32, 0.8, 1.2];
    /// let layer = BatchNormLayer::new_with_params(3, 1e-5, 0.9, gamma, beta, running_mean, running_var);
    /// assert_eq!(layer.input_size(), 3);
    /// assert_eq!(layer.gamma()[0], 1.5);
    /// assert_eq!(layer.beta()[1], -0.2);
    /// assert_eq!(layer.running_mean()[2], -0.5);
    /// ```
    pub fn new_with_params(
        size: usize,
        epsilon: f32,
        momentum: f32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        running_mean: Vec<f32>,
        running_var: Vec<f32>,
    ) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(
            (0.0..=1.0).contains(&momentum),
            "momentum must be in range [0.0, 1.0]"
        );
        assert_eq!(
            gamma.len(),
            size,
            "gamma length {} does not match size = {}",
            gamma.len(),
            size
        );
        assert_eq!(
            beta.len(),
            size,
            "beta length {} does not match size = {}",
            beta.len(),
            size
        );
        assert_eq!(
            running_mean.len(),
            size,
            "running_mean length {} does not match size = {}",
            running_mean.len(),
            size
        );
        assert_eq!(
            running_var.len(),
            size,
            "running_var length {} does not match size = {}",
            running_var.len(),
            size
        );

        Self {
            size,
            epsilon,
            momentum,
            training: false,

            gamma,
            beta,

            // Zero-initialize gradients
            grad_gamma: GradientAccumulator::new(size),
            grad_beta: GradientAccumulator::new(size),

            // Use provided running statistics
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),

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
    /// # Normalization Formula
    ///
    /// **Training Mode:**
    ///
    /// 1. **Compute batch mean per feature:**
    ///    - `μ_j = (1/m) × Σ(x_ij)` for i = 1 to m
    ///
    /// 2. **Compute batch variance per feature:**
    ///    - `σ²_j = (1/m) × Σ(x_ij - μ_j)²` for i = 1 to m
    ///
    /// 3. **Normalize:**
    ///    - `x̂_ij = (x_ij - μ_j) / sqrt(σ²_j + ε)`
    ///
    /// 4. **Scale and shift:**
    ///    - `y_ij = γ_j × x̂_ij + β_j`
    ///
    /// 5. **Update running statistics (exponential moving average):**
    ///    - `running_μ = α × running_μ + (1 - α) × μ`
    ///    - `running_σ² = α × running_σ² + (1 - α) × σ²`
    ///
    /// where:
    /// - `x` is the input (batch_size × size)
    /// - `y` is the output (batch_size × size)
    /// - `m` is the batch_size
    /// - `j` indexes features (0 to size-1)
    /// - `i` indexes samples in the batch (0 to batch_size-1)
    /// - `μ_j` is the mean of feature j across the batch
    /// - `σ²_j` is the variance of feature j across the batch
    /// - `ε` is epsilon for numerical stability
    /// - `γ_j` is the learnable scale parameter for feature j
    /// - `β_j` is the learnable shift parameter for feature j
    /// - `α` is the momentum (typical: 0.9 or 0.99)
    /// - `x̂_ij` is the normalized value
    ///
    /// **Inference Mode:**
    ///
    /// Uses accumulated running statistics instead of batch statistics:
    /// - `x̂_ij = (x_ij - running_μ_j) / sqrt(running_σ²_j + ε)`
    /// - `y_ij = γ_j × x̂_ij + β_j`
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
    /// # Gradient Formulas
    ///
    /// Given gradient w.r.t. output: `∂L/∂y` (batch_size × size)
    ///
    /// **Training Mode:**
    ///
    /// The backward pass computes gradients through the normalization chain rule:
    ///
    /// **Step 1: Parameter gradients (accumulated across batch)**
    ///
    /// - **Gamma gradient:**
    ///   - `∂L/∂γ_j = (1/m) × Σ(∂L/∂y_ij × x̂_ij)`
    ///
    /// - **Beta gradient:**
    ///   - `∂L/∂β_j = (1/m) × Σ(∂L/∂y_ij)`
    ///
    /// **Step 2: Gradient w.r.t. normalized values**
    ///
    /// - `∂L/∂x̂_ij = ∂L/∂y_ij × γ_j`
    ///
    /// **Step 3: Gradient w.r.t. variance**
    ///
    /// - `∂L/∂σ²_j = Σ(∂L/∂x̂_ij × (x_ij - μ_j) × (-0.5) × (σ²_j + ε)^(-3/2))`
    /// - Simplified: `∂L/∂σ²_j = Σ(∂L/∂x̂_ij × x̂_ij × (-0.5) / sqrt(σ²_j + ε))`
    ///
    /// **Step 4: Gradient w.r.t. mean**
    ///
    /// - `∂L/∂μ_j = Σ(∂L/∂x̂_ij × (-1 / sqrt(σ²_j + ε)))`
    /// - Plus contribution from variance: `∂L/∂μ_j += ∂L/∂σ²_j × (-2/m) × Σ(x_ij - μ_j)`
    ///
    /// **Step 5: Gradient w.r.t. input (chain rule combination)**
    ///
    /// - `∂L/∂x_ij = ∂L/∂x̂_ij / sqrt(σ²_j + ε)`
    ///   - `+ ∂L/∂σ²_j × (2/m) × (x_ij - μ_j)`
    ///   - `+ ∂L/∂μ_j / m`
    ///
    /// where:
    /// - `∂L/∂y` is the gradient w.r.t. output (batch_size × size)
    /// - `∂L/∂x` is the gradient w.r.t. input (batch_size × size)
    /// - `m` is the batch_size
    /// - `x̂_ij` is the normalized value (cached from forward pass)
    /// - `μ_j`, `σ²_j` are the batch mean and variance (cached from forward pass)
    /// - `ε` is epsilon for numerical stability
    /// - `γ_j`, `β_j` are the learnable scale and shift parameters
    ///
    /// **Inference Mode:**
    ///
    /// Simplified gradient pass-through using running statistics:
    /// - `∂L/∂x_ij = ∂L/∂y_ij × γ_j / sqrt(running_σ²_j + ε)`
    ///
    /// Parameter gradients are not accumulated in inference mode as the layer
    /// should not be trained during inference.
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

        // Compute gradients for gamma and beta (accumulated across batch)
        let scale = 1.0 / batch_size as f32;
        let mut dg = vec![0.0f32; self.size];
        let mut db = vec![0.0f32; self.size];
        for i in 0..batch_size {
            for j in 0..self.size {
                let idx = i * self.size + j;
                dg[j] += grad_output[idx] * normalized[idx];
                db[j] += grad_output[idx];
            }
        }
        self.grad_gamma.accumulate_scaled(&dg, scale);
        self.grad_beta.accumulate_scaled(&db, scale);

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
        self.grad_gamma
            .apply_sgd_update(&mut self.gamma, learning_rate);
        self.grad_beta
            .apply_sgd_update(&mut self.beta, learning_rate);
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
        self.grad_gamma
            .apply_optimizer_update(&mut self.gamma, optimizer);
        self.grad_beta
            .apply_optimizer_update(&mut self.beta, optimizer);
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
mod tests;
