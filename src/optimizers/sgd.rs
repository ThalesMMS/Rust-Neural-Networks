//! Stochastic Gradient Descent (SGD) optimizer implementation
//!
//! This module provides a vanilla SGD optimizer that performs the basic
//! gradient descent update: `parameter = parameter - learning_rate * gradient`

use crate::optimizers::Optimizer;

/// Stochastic Gradient Descent optimizer.
///
/// Implements the basic gradient descent update rule without momentum or
/// adaptive learning rates. This is the simplest optimization algorithm:
///
/// `w = w - η * ∇L/∂w`
///
/// where w is the parameter, η (eta) is the learning rate, and ∇L/∂w is the gradient.
///
/// # Fields
///
/// * `learning_rate` - The step size for parameter updates (η in the formula above)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::optimizers::{Optimizer, SGD};
///
/// let mut optimizer = SGD::new(0.01);
/// let mut weights = vec![1.0, 2.0, 3.0];
/// let gradients = vec![0.1, 0.2, 0.3];
///
/// optimizer.update(&mut weights, &gradients);
/// // weights are now: [0.999, 1.998, 2.997]
/// ```
///
/// # Limitations
///
/// Vanilla SGD can be slow to converge and may oscillate around minima.
/// For better convergence, consider using momentum-based optimizers like Adam.
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    /// Creates a new SGD optimizer with the specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The step size for parameter updates (must be positive)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::sgd::SGD;
    ///
    /// let optimizer = SGD::new(0.01);
    /// assert_eq!(optimizer.learning_rate(), 0.01);
    /// ```
    ///
    /// # Typical Values
    ///
    /// Common learning rates range from 0.001 to 0.1, depending on the problem:
    /// - 0.01: Good starting point for many problems
    /// - 0.001: More conservative, useful for fine-tuning
    /// - 0.1: Aggressive, may cause instability
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    /// Update parameters using vanilla SGD rule.
    ///
    /// Applies the update: `parameter[i] -= learning_rate * gradient[i]`
    /// for each parameter.
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
    /// use rust_neural_networks::optimizers::{Optimizer, sgd::SGD};
    ///
    /// let mut optimizer = SGD::new(0.1);
    /// let mut params = vec![1.0, 2.0, 3.0];
    /// let grads = vec![0.1, 0.2, 0.3];
    ///
    /// optimizer.update(&mut params, &grads);
    /// assert!((params[0] - 0.99).abs() < 1e-6);
    /// assert!((params[1] - 1.98).abs() < 1e-6);
    /// assert!((params[2] - 2.97).abs() < 1e-6);
    /// ```
    fn update(&mut self, parameters: &mut [f32], gradients: &[f32]) {
        assert_eq!(
            parameters.len(),
            gradients.len(),
            "Parameters and gradients must have the same length"
        );

        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            *param -= self.learning_rate * grad;
        }
    }

    /// Reset optimizer state.
    ///
    /// For vanilla SGD, this is a no-op since there is no internal state
    /// (no momentum or adaptive learning rate statistics).
    fn reset(&mut self) {
        // Vanilla SGD has no state to reset
    }

    /// Get the current learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, sgd::SGD};
    ///
    /// let optimizer = SGD::new(0.01);
    /// assert_eq!(optimizer.learning_rate(), 0.01);
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
    /// use rust_neural_networks::optimizers::{Optimizer, sgd::SGD};
    ///
    /// let mut optimizer = SGD::new(0.01);
    /// optimizer.set_learning_rate(0.001);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_new() {
        let optimizer = SGD::new(0.01);
        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_sgd_update() {
        let mut optimizer = SGD::new(0.1);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        optimizer.update(&mut params, &grads);

        assert!((params[0] - 0.99).abs() < 1e-6);
        assert!((params[1] - 1.98).abs() < 1e-6);
        assert!((params[2] - 2.97).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_multiple_updates() {
        let mut optimizer = SGD::new(0.01);
        let mut params = vec![1.0, 1.0];
        let grads = vec![1.0, -1.0];

        // First update
        optimizer.update(&mut params, &grads);
        assert!((params[0] - 0.99).abs() < 1e-6);
        assert!((params[1] - 1.01).abs() < 1e-6);

        // Second update
        optimizer.update(&mut params, &grads);
        assert!((params[0] - 0.98).abs() < 1e-6);
        assert!((params[1] - 1.02).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_learning_rate_update() {
        let mut optimizer = SGD::new(0.1);
        assert_eq!(optimizer.learning_rate(), 0.1);

        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.learning_rate(), 0.01);

        // Verify the new learning rate is used
        let mut params = vec![1.0];
        let grads = vec![1.0];
        optimizer.update(&mut params, &grads);
        assert!((params[0] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_reset() {
        let mut optimizer = SGD::new(0.01);
        optimizer.reset();
        // Reset should be a no-op for SGD, just verify it doesn't panic
        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    #[should_panic(expected = "Parameters and gradients must have the same length")]
    fn test_sgd_mismatched_lengths() {
        let mut optimizer = SGD::new(0.01);
        let mut params = vec![1.0, 2.0];
        let grads = vec![0.1, 0.2, 0.3];
        optimizer.update(&mut params, &grads);
    }

    #[test]
    fn test_sgd_zero_learning_rate() {
        let mut optimizer = SGD::new(0.0);
        let mut params = vec![1.0, 2.0, 3.0];
        let original = params.clone();
        let grads = vec![0.1, 0.2, 0.3];

        optimizer.update(&mut params, &grads);

        // With zero learning rate, parameters shouldn't change
        assert_eq!(params, original);
    }

    #[test]
    fn test_sgd_negative_gradients() {
        let mut optimizer = SGD::new(0.1);
        let mut params = vec![1.0, 2.0];
        let grads = vec![-0.5, -1.0];

        optimizer.update(&mut params, &grads);

        // Negative gradients should increase parameters
        assert!((params[0] - 1.05).abs() < 1e-6);
        assert!((params[1] - 2.1).abs() < 1e-6);
    }
}
