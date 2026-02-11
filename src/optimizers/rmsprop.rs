//! RMSprop (Root Mean Square Propagation) optimizer implementation
//!
//! This module provides the RMSprop optimizer, which uses adaptive learning
//! rates based on a moving average of squared gradients for improved convergence.

use crate::optimizers::Optimizer;

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// RMSprop is an adaptive learning rate optimization algorithm that addresses
/// the diminishing learning rates problem of AdaGrad. It maintains a moving
/// average of squared gradients for each parameter and uses this to adapt
/// the learning rate.
///
/// Unlike Adam, RMSprop does not use momentum (first moment), only the
/// second moment (moving average of squared gradients).
///
/// The update rule is:
///
/// ```text
/// v_t = ρ * v_{t-1} + (1 - ρ) * gradient²
/// parameter = parameter - α * gradient / (√v_t + ε)
/// ```
///
/// where:
/// - α (alpha) is the learning rate
/// - ρ (rho/decay_rate) is the exponential decay rate for the moving average
/// - ε (epsilon) is a small constant for numerical stability
/// - v is the moving average of squared gradients
///
/// # Fields
///
/// * `learning_rate` - The step size for parameter updates (α)
/// * `decay_rate` - Exponential decay rate for squared gradient moving average (typically 0.9)
/// * `epsilon` - Small constant for numerical stability (typically 1e-8)
/// * `v` - Moving average of squared gradients for each parameter
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::optimizers::{Optimizer, RMSprop};
///
/// let mut optimizer = RMSprop::new(0.001, 0.9, 1e-8);
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
/// - Works well with non-stationary objectives
/// - Simpler than Adam (no momentum term)
/// - Uses less memory than Adam (only one moving average)
///
/// # Compared to Adam
///
/// - RMSprop: Only second moment, no bias correction
/// - Adam: Both first and second moments with bias correction
/// - RMSprop is often faster and uses less memory
/// - Adam may converge better in some scenarios
///
/// # Reference
///
/// Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient
/// by a running average of its recent magnitude. COURSERA: Neural networks for
/// machine learning, 4(2), 26-31.
pub struct RMSprop {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    /// Moving average of squared gradients
    v: Vec<f32>,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer with the specified hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The step size for parameter updates (α, must be positive)
    /// * `decay_rate` - Exponential decay rate for squared gradient average (0 < ρ < 1)
    /// * `epsilon` - Small constant for numerical stability (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::rmsprop::RMSprop;
    /// use rust_neural_networks::optimizers::Optimizer;
    ///
    /// // Default RMSprop hyperparameters
    /// let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    ///
    /// # Typical Values
    ///
    /// Common RMSprop hyperparameters:
    /// - learning_rate: 0.001
    /// - decay_rate: 0.9 (some use 0.99)
    /// - epsilon: 1e-8
    ///
    /// These defaults work well for a wide range of problems.
    pub fn new(learning_rate: f32, decay_rate: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            decay_rate,
            epsilon,
            v: Vec::new(),
        }
    }

    /// Get the decay rate parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::rmsprop::RMSprop;
    ///
    /// let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// assert_eq!(optimizer.decay_rate(), 0.9);
    /// ```
    pub fn decay_rate(&self) -> f32 {
        self.decay_rate
    }

    /// Get the epsilon parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::rmsprop::RMSprop;
    ///
    /// let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// assert_eq!(optimizer.epsilon(), 1e-8);
    /// ```
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl Optimizer for RMSprop {
    /// Update parameters using RMSprop optimizer rule.
    ///
    /// Applies the RMSprop update:
    /// 1. Update moving average of squared gradients
    /// 2. Update parameters using adaptive learning rate
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
    /// use rust_neural_networks::optimizers::{Optimizer, rmsprop::RMSprop};
    ///
    /// let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
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

        // Initialize v vector on first use
        if self.v.is_empty() {
            self.v = vec![0.0; parameters.len()];
        }

        // Ensure v vector has correct size
        if self.v.len() != parameters.len() {
            self.v.resize(parameters.len(), 0.0);
        }

        // Update parameters
        for i in 0..parameters.len() {
            // Update moving average of squared gradients
            // v = decay_rate * v + (1 - decay_rate) * gradient²
            self.v[i] =
                self.decay_rate * self.v[i] + (1.0 - self.decay_rate) * gradients[i] * gradients[i];

            // Update parameters with adaptive learning rate
            // parameter = parameter - learning_rate * gradient / (√v + epsilon)
            parameters[i] -= self.learning_rate * gradients[i] / (self.v[i].sqrt() + self.epsilon);
        }
    }

    /// Reset optimizer state.
    ///
    /// Clears the moving average of squared gradients, effectively resetting
    /// the optimizer to its initial state.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, rmsprop::RMSprop};
    ///
    /// let mut optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// let mut params = vec![1.0, 2.0, 3.0];
    /// let grads = vec![0.1, 0.2, 0.3];
    ///
    /// optimizer.update(&mut params, &grads);
    /// optimizer.reset();
    /// // Internal state (v) is cleared
    /// ```
    fn reset(&mut self) {
        self.v.clear();
    }

    /// Get the current learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, rmsprop::RMSprop};
    ///
    /// let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set a new learning rate.
    ///
    /// Useful for implementing learning rate schedules or decay strategies.
    ///
    /// # Arguments
    ///
    /// * `lr` - New learning rate value (should be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, rmsprop::RMSprop};
    ///
    /// let mut optimizer = RMSprop::new(0.001, 0.9, 1e-8);
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
    fn test_rmsprop_new() {
        let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.decay_rate(), 0.9);
        assert_eq!(optimizer.epsilon(), 1e-8);
    }

    #[test]
    fn test_rmsprop_update() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        let initial_params = params.clone();
        optimizer.update(&mut params, &grads);

        // Parameters should have changed
        assert_ne!(params[0], initial_params[0]);
        assert_ne!(params[1], initial_params[1]);
        assert_ne!(params[2], initial_params[2]);

        // Parameters should decrease (gradient descent)
        assert!(params[0] < initial_params[0]);
        assert!(params[1] < initial_params[1]);
        assert!(params[2] < initial_params[2]);
    }

    #[test]
    fn test_rmsprop_adaptive_learning_rate() {
        // RMSprop normalizes step size by gradient magnitude
        // Demonstrates that RMSprop adapts to gradient scale
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0];

        // First update with small gradient
        let small_grad = vec![0.01];
        optimizer.update(&mut params, &small_grad);
        let step1 = (1.0 - params[0]).abs();

        // Reset for fair comparison
        params[0] = 1.0;
        let mut optimizer2 = RMSprop::new(0.1, 0.9, 1e-8);

        // Second update with large gradient
        let large_grad = vec![1.0];
        optimizer2.update(&mut params, &large_grad);
        let step2 = (1.0 - params[0]).abs();

        // Large gradient gets normalized by sqrt(grad²), resulting in larger step
        // but not proportionally larger (adaptive behavior)
        // step1 ≈ lr * 0.01 / sqrt(0.0001) = 0.1 * 0.01 / 0.01 = 0.1
        // step2 ≈ lr * 1.0 / sqrt(0.1) = 0.1 * 1.0 / 0.316 ≈ 0.316
        assert!(step2 > step1);
        // But step2 is not 100x larger (it's only ~3.16x larger due to normalization)
        assert!(step2 < 10.0 * step1);
    }

    #[test]
    fn test_rmsprop_multiple_updates() {
        let mut optimizer = RMSprop::new(0.01, 0.9, 1e-8);
        let mut params = vec![1.0, 1.0];
        let grads = vec![1.0, -1.0];

        // First update
        optimizer.update(&mut params, &grads);
        let params_after_1 = params.clone();

        // Second update
        optimizer.update(&mut params, &grads);

        // Parameters should continue to change
        assert_ne!(params[0], params_after_1[0]);
        assert_ne!(params[1], params_after_1[1]);

        // First param should decrease, second should increase
        assert!(params[0] < params_after_1[0]);
        assert!(params[1] > params_after_1[1]);
    }

    #[test]
    fn test_rmsprop_reset() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        // Perform update to initialize state
        optimizer.update(&mut params, &grads);
        assert!(!optimizer.v.is_empty());

        // Reset should clear state
        optimizer.reset();
        assert!(optimizer.v.is_empty());
    }

    #[test]
    fn test_rmsprop_learning_rate_update() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        assert_eq!(optimizer.learning_rate(), 0.1);

        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_rmsprop_epsilon_prevents_division_by_zero() {
        // Test that epsilon prevents numerical issues with zero gradients
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.0, 0.0, 0.0];

        // Should not panic or produce NaN
        optimizer.update(&mut params, &grads);
        assert!(!params[0].is_nan());
        assert!(!params[1].is_nan());
        assert!(!params[2].is_nan());
    }

    #[test]
    fn test_rmsprop_decay_accumulation() {
        // Test that v accumulates squared gradients with decay
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0];
        let grads = vec![1.0];

        // First update: v = 0.9 * 0 + 0.1 * 1^2 = 0.1
        optimizer.update(&mut params, &grads);
        assert!((optimizer.v[0] - 0.1).abs() < 1e-6);

        // Second update: v = 0.9 * 0.1 + 0.1 * 1^2 = 0.19
        optimizer.update(&mut params, &grads);
        assert!((optimizer.v[0] - 0.19).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_different_decay_rates() {
        // Test that different decay rates produce different behaviors
        let mut opt_low_decay = RMSprop::new(0.1, 0.5, 1e-8);
        let mut opt_high_decay = RMSprop::new(0.1, 0.99, 1e-8);

        let mut params1 = vec![1.0];
        let mut params2 = vec![1.0];
        let grads = vec![1.0];

        // Apply same updates
        for _ in 0..5 {
            opt_low_decay.update(&mut params1, &grads);
            opt_high_decay.update(&mut params2, &grads);
        }

        // Different decay rates should produce different results
        assert_ne!(params1[0], params2[0]);
    }

    #[test]
    #[should_panic(expected = "Parameters and gradients must have the same length")]
    fn test_rmsprop_mismatched_lengths() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2];

        optimizer.update(&mut params, &grads);
    }

    #[test]
    fn test_rmsprop_resize_on_size_change() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);

        // First update with 3 parameters
        let mut params1 = vec![1.0, 2.0, 3.0];
        let grads1 = vec![0.1, 0.2, 0.3];
        optimizer.update(&mut params1, &grads1);
        assert_eq!(optimizer.v.len(), 3);

        // Second update with 5 parameters (should resize)
        let mut params2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let grads2 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        optimizer.update(&mut params2, &grads2);
        assert_eq!(optimizer.v.len(), 5);
    }
}
