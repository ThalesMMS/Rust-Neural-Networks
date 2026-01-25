//! Adam (Adaptive Moment Estimation) optimizer implementation
//!
//! This module provides the Adam optimizer, which combines momentum and
//! adaptive learning rates with bias correction for improved convergence.

use crate::optimizers::Optimizer;

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Adam combines ideas from momentum optimization and RMSprop to provide
/// adaptive learning rates for each parameter. It maintains two moving
/// averages for each parameter:
///
/// 1. First moment (mean) of gradients (momentum)
/// 2. Second moment (uncentered variance) of gradients (adaptive learning rate)
///
/// The update rule is:
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * gradient
/// v_t = β2 * v_{t-1} + (1 - β2) * gradient²
/// m_hat = m_t / (1 - β1^t)
/// v_hat = v_t / (1 - β2^t)
/// parameter = parameter - α * m_hat / (√v_hat + ε)
/// ```
///
/// where:
/// - α (alpha) is the learning rate
/// - β1 (beta1) is the exponential decay rate for first moment estimates
/// - β2 (beta2) is the exponential decay rate for second moment estimates
/// - ε (epsilon) is a small constant for numerical stability
/// - t is the time step
///
/// # Fields
///
/// * `learning_rate` - The step size for parameter updates (α)
/// * `beta1` - Exponential decay rate for first moment estimates (typically 0.9)
/// * `beta2` - Exponential decay rate for second moment estimates (typically 0.999)
/// * `epsilon` - Small constant for numerical stability (typically 1e-8)
/// * `m` - First moment estimates (momentum) for each parameter
/// * `v` - Second moment estimates (adaptive learning rate) for each parameter
/// * `t` - Time step counter for bias correction
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::optimizers::{Optimizer, Adam};
///
/// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
/// let mut weights = vec![1.0, 2.0, 3.0];
/// let gradients = vec![0.1, 0.2, 0.3];
///
/// optimizer.update(&mut weights, &gradients);
/// // weights are updated using adaptive learning rates
/// ```
///
/// # Advantages
///
/// - Adaptive learning rates per parameter
/// - Works well with sparse gradients
/// - Requires little tuning of hyperparameters
/// - Combines benefits of momentum and RMSprop
///
/// # Reference
///
/// Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
/// arXiv preprint arXiv:1412.6980.
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// First moment estimates (momentum)
    m: Vec<f32>,
    /// Second moment estimates (adaptive learning rate)
    v: Vec<f32>,
    /// Time step counter for bias correction
    t: usize,
}

impl Adam {
    /// Creates a new Adam optimizer with the specified hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The step size for parameter updates (α, must be positive)
    /// * `beta1` - Exponential decay rate for first moment estimates (0 < β1 < 1)
    /// * `beta2` - Exponential decay rate for second moment estimates (0 < β2 < 1)
    /// * `epsilon` - Small constant for numerical stability (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::adam::Adam;
    /// use rust_neural_networks::optimizers::Optimizer;
    ///
    /// // Default Adam hyperparameters from the paper
    /// let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    ///
    /// # Typical Values
    ///
    /// The original Adam paper recommends:
    /// - learning_rate: 0.001
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: 1e-8
    ///
    /// These defaults work well for a wide range of problems.
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    /// Update parameters using Adam optimizer rule.
    ///
    /// Applies the Adam update with bias correction:
    /// 1. Update biased first moment estimate (momentum)
    /// 2. Update biased second moment estimate (adaptive learning rate)
    /// 3. Compute bias-corrected first moment estimate
    /// 4. Compute bias-corrected second moment estimate
    /// 5. Update parameters using corrected estimates
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameters to update (weights or biases)
    /// * `gradients` - Gradient of loss with respect to each parameter
    ///
    /// # Panics
    ///
    /// Panics if `parameters` and `gradients` have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, adam::Adam};
    ///
    /// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// let mut params = vec![1.0, 2.0, 3.0];
    /// let grads = vec![0.1, 0.2, 0.3];
    ///
    /// optimizer.update(&mut params, &grads);
    /// // Parameters updated with adaptive learning rates
    /// ```
    fn update(&mut self, parameters: &mut [f32], gradients: &[f32]) {
        assert_eq!(
            parameters.len(),
            gradients.len(),
            "Parameters and gradients must have the same length"
        );

        // Initialize moment vectors on first use
        if self.m.is_empty() {
            self.m = vec![0.0; parameters.len()];
            self.v = vec![0.0; parameters.len()];
        }

        // Ensure moment vectors have correct size
        if self.m.len() != parameters.len() {
            self.m.resize(parameters.len(), 0.0);
            self.v.resize(parameters.len(), 0.0);
        }

        // Increment time step
        self.t += 1;

        // Compute bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Update parameters
        for i in 0..parameters.len() {
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / bias_correction2;

            // Update parameters
            parameters[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    /// Reset optimizer state.
    ///
    /// Clears all momentum and adaptive learning rate statistics,
    /// and resets the time step counter. Useful when starting a new
    /// training run or switching between different datasets.
    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    /// Get the current learning rate.
    ///
    /// Returns the base learning rate (α). Note that Adam applies different
    /// effective learning rates to different parameters based on their gradient history.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, adam::Adam};
    ///
    /// let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set a new learning rate.
    ///
    /// Updates the base learning rate (α). Useful for implementing learning rate
    /// schedules or decay strategies.
    ///
    /// # Arguments
    ///
    /// * `lr` - New learning rate value (should be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, adam::Adam};
    ///
    /// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// optimizer.set_learning_rate(0.0001);
    /// assert_eq!(optimizer.learning_rate(), 0.0001);
    /// ```
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_new() {
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_update() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        let original_params = params.clone();
        optimizer.update(&mut params, &grads);

        // Parameters should have changed
        assert_ne!(params[0], original_params[0]);
        assert_ne!(params[1], original_params[1]);
        assert_ne!(params[2], original_params[2]);

        // Parameters should have decreased (gradients are positive)
        assert!(params[0] < original_params[0]);
        assert!(params[1] < original_params[1]);
        assert!(params[2] < original_params[2]);
    }

    #[test]
    fn test_adam_multiple_updates() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 1.0];
        let grads1 = vec![1.0, -1.0];
        let grads2 = vec![0.5, -0.5];

        // First update
        optimizer.update(&mut params, &grads1);
        let params_after_first = params.clone();

        // Second update with different gradients
        optimizer.update(&mut params, &grads2);

        // Parameters should continue to change
        assert_ne!(params[0], params_after_first[0]);
        assert_ne!(params[1], params_after_first[1]);

        // Time step should have incremented
        assert_eq!(optimizer.t, 2);
    }

    #[test]
    fn test_adam_bias_correction() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0];
        let grads = vec![1.0];

        // First update - bias correction should have large effect
        optimizer.update(&mut params, &grads);

        // Reset and do many updates - bias correction effect should diminish
        optimizer.reset();
        params = vec![1.0];
        for _ in 0..1000 {
            optimizer.update(&mut params, &grads);
        }

        // After many updates, bias correction for beta1 should be close to 1
        let bias_correction1 = 1.0 - optimizer.beta1.powi(1000);
        assert!(bias_correction1 > 0.99);

        // For beta2=0.999, even after 1000 iterations, bias correction grows more slowly
        // but should still be significant
        let bias_correction2 = 1.0 - optimizer.beta2.powi(1000);
        assert!(bias_correction2 > 0.63); // 1 - 0.999^1000 ≈ 0.632
    }

    #[test]
    fn test_adam_reset() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        // Perform some updates
        optimizer.update(&mut params, &grads);
        optimizer.update(&mut params, &grads);

        assert_eq!(optimizer.t, 2);
        assert!(!optimizer.m.is_empty());
        assert!(!optimizer.v.is_empty());

        // Reset optimizer
        optimizer.reset();

        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_empty());
        assert!(optimizer.v.is_empty());
    }

    #[test]
    fn test_adam_learning_rate_update() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(optimizer.learning_rate(), 0.001);

        optimizer.set_learning_rate(0.0001);
        assert_eq!(optimizer.learning_rate(), 0.0001);
    }

    #[test]
    #[should_panic(expected = "Parameters and gradients must have the same length")]
    fn test_adam_mismatched_lengths() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2];

        optimizer.update(&mut params, &grads);
    }

    #[test]
    fn test_adam_state_persistence() {
        // Test that Adam maintains internal state across updates
        let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 2.0];

        // First update
        optimizer.update(&mut params, &[0.1, 0.2]);
        assert_eq!(optimizer.t, 1);
        assert_eq!(optimizer.m.len(), 2);
        assert_eq!(optimizer.v.len(), 2);

        // Save state
        let m_after_first = optimizer.m.clone();
        let v_after_first = optimizer.v.clone();

        // Second update - should use accumulated state
        optimizer.update(&mut params, &[0.1, 0.2]);
        assert_eq!(optimizer.t, 2);

        // State should have changed (momentum accumulated)
        assert_ne!(optimizer.m, m_after_first);
        assert_ne!(optimizer.v, v_after_first);
    }

    #[test]
    fn test_adam_adaptive_learning_rates() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0, 1.0];

        // One parameter gets large gradients, one gets small gradients
        for _ in 0..5 {
            let grads = vec![10.0, 0.1];
            optimizer.update(&mut params, &grads);
        }

        // Both parameters should have moved despite very different gradient magnitudes
        // This demonstrates adaptive learning rates
        assert!(params[0] < 1.0);
        assert!(params[1] < 1.0);
    }
}
