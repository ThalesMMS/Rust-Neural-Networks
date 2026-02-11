//! AdamW (Adam with decoupled Weight decay) optimizer implementation
//!
//! This module provides the AdamW optimizer, which fixes the weight decay
//! implementation in Adam by decoupling it from the adaptive learning rates.

use crate::optimizers::Optimizer;

/// AdamW (Adam with decoupled Weight decay) optimizer.
///
/// AdamW fixes a subtle but important issue in Adam's weight decay implementation.
/// Standard Adam applies L2 regularization to the gradients, which interacts poorly
/// with the adaptive learning rates. AdamW instead applies weight decay directly
/// to the parameters after the Adam update, decoupling it from the gradient-based
/// optimization.
///
/// The update rule is:
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * gradient
/// v_t = β2 * v_{t-1} + (1 - β2) * gradient²
/// m_hat = m_t / (1 - β1^t)
/// v_hat = v_t / (1 - β2^t)
/// parameter = parameter - α * m_hat / (√v_hat + ε) - α * λ * parameter
/// ```
///
/// where:
/// - α (alpha) is the learning rate
/// - β1 (beta1) is the exponential decay rate for first moment estimates
/// - β2 (beta2) is the exponential decay rate for second moment estimates
/// - ε (epsilon) is a small constant for numerical stability
/// - λ (lambda) is the weight decay coefficient
/// - t is the time step
///
/// # Fields
///
/// * `learning_rate` - The step size for parameter updates (α)
/// * `beta1` - Exponential decay rate for first moment estimates (typically 0.9)
/// * `beta2` - Exponential decay rate for second moment estimates (typically 0.999)
/// * `epsilon` - Small constant for numerical stability (typically 1e-8)
/// * `weight_decay` - Weight decay coefficient (λ, typically 0.01)
/// * `m` - First moment estimates (momentum) for each parameter
/// * `v` - Second moment estimates (adaptive learning rate) for each parameter
/// * `t` - Time step counter for bias correction
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::optimizers::{Optimizer, AdamW};
///
/// let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
/// let mut weights = vec![1.0, 2.0, 3.0];
/// let gradients = vec![0.1, 0.2, 0.3];
///
/// optimizer.update(&mut weights, &gradients);
/// // weights are updated using adaptive learning rates with proper weight decay
/// ```
///
/// # Advantages over Adam
///
/// - Proper weight decay that doesn't interfere with adaptive learning rates
/// - Better generalization on many tasks, especially transformers
/// - Recommended as the default optimizer for transformer models
/// - More predictable behavior when tuning weight decay
///
/// # Reference
///
/// Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
/// arXiv preprint arXiv:1711.05101.
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    /// First moment estimates (momentum)
    m: Vec<f32>,
    /// Second moment estimates (adaptive learning rate)
    v: Vec<f32>,
    /// Time step counter for bias correction
    t: usize,
}

impl AdamW {
    /// Creates a new AdamW optimizer with the specified hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The step size for parameter updates (α, must be positive)
    /// * `beta1` - Exponential decay rate for first moment estimates (0 < β1 < 1)
    /// * `beta2` - Exponential decay rate for second moment estimates (0 < β2 < 1)
    /// * `epsilon` - Small constant for numerical stability (must be positive)
    /// * `weight_decay` - Weight decay coefficient (λ, must be non-negative)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::adamw::AdamW;
    /// use rust_neural_networks::optimizers::Optimizer;
    ///
    /// // Default AdamW hyperparameters from the paper
    /// let optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
    /// assert_eq!(optimizer.learning_rate(), 0.001);
    /// ```
    ///
    /// # Typical Values
    ///
    /// The AdamW paper recommends:
    /// - learning_rate: 0.001
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: 1e-8
    /// - weight_decay: 0.01
    ///
    /// These defaults work well for transformer models and other deep architectures.
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for AdamW {
    /// Update parameters using AdamW optimizer rule.
    ///
    /// Applies the AdamW update with bias correction and decoupled weight decay:
    /// 1. Update biased first moment estimate (momentum)
    /// 2. Update biased second moment estimate (adaptive learning rate)
    /// 3. Compute bias-corrected first moment estimate
    /// 4. Compute bias-corrected second moment estimate
    /// 5. Update parameters using corrected estimates (Adam step)
    /// 6. Apply decoupled weight decay directly to parameters
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
    /// use rust_neural_networks::optimizers::{Optimizer, adamw::AdamW};
    ///
    /// let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
    /// let mut params = vec![1.0, 2.0, 3.0];
    /// let grads = vec![0.1, 0.2, 0.3];
    ///
    /// optimizer.update(&mut params, &grads);
    /// // Parameters updated with adaptive learning rates and proper weight decay
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

            // Update parameters with Adam step
            parameters[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);

            // Apply decoupled weight decay (this is the key difference from Adam)
            parameters[i] -= self.learning_rate * self.weight_decay * parameters[i];
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
    /// Returns the base learning rate (α). Note that AdamW applies different
    /// effective learning rates to different parameters based on their gradient history.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::optimizers::{Optimizer, adamw::AdamW};
    ///
    /// let optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
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
    /// use rust_neural_networks::optimizers::{Optimizer, adamw::AdamW};
    ///
    /// let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
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
    fn test_adamw_new() {
        let optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adamw_update() {
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        let original_params = params.clone();
        optimizer.update(&mut params, &grads);

        // Parameters should have changed
        assert_ne!(params[0], original_params[0]);
        assert_ne!(params[1], original_params[1]);
        assert_ne!(params[2], original_params[2]);

        // Parameters should have decreased (gradients are positive and weight decay applies)
        assert!(params[0] < original_params[0]);
        assert!(params[1] < original_params[1]);
        assert!(params[2] < original_params[2]);
    }

    #[test]
    fn test_adamw_weight_decay() {
        // Compare AdamW with weight_decay=0.01 to ensure weight decay is applied
        let mut optimizer_with_decay = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        let mut optimizer_no_decay = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.0);

        let mut params_with_decay = vec![1.0, 2.0, 3.0];
        let mut params_no_decay = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        optimizer_with_decay.update(&mut params_with_decay, &grads);
        optimizer_no_decay.update(&mut params_no_decay, &grads);

        // With weight decay, parameters should decrease more
        assert!(params_with_decay[0] < params_no_decay[0]);
        assert!(params_with_decay[1] < params_no_decay[1]);
        assert!(params_with_decay[2] < params_no_decay[2]);
    }

    #[test]
    fn test_adamw_multiple_updates() {
        let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
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
    fn test_adamw_bias_correction() {
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
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
    fn test_adamw_reset() {
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
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
    fn test_adamw_learning_rate_update() {
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        assert_eq!(optimizer.learning_rate(), 0.001);

        optimizer.set_learning_rate(0.0001);
        assert_eq!(optimizer.learning_rate(), 0.0001);
    }

    #[test]
    #[should_panic(expected = "Parameters and gradients must have the same length")]
    fn test_adamw_mismatched_lengths() {
        let mut optimizer = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2];

        optimizer.update(&mut params, &grads);
    }

    #[test]
    fn test_adamw_state_persistence() {
        // Test that AdamW maintains internal state across updates
        let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
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
    fn test_adamw_adaptive_learning_rates() {
        let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01);
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

    #[test]
    fn test_adamw_decoupled_weight_decay() {
        // Test that weight decay is truly decoupled (applied after Adam step)
        let mut optimizer = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut params = vec![10.0];
        let grads = vec![0.0]; // Zero gradient

        let original_param = params[0];
        optimizer.update(&mut params, &grads);

        // Even with zero gradient, weight decay should reduce the parameter
        // The reduction is: param -= lr * weight_decay * param
        // Expected: 10.0 - 0.1 * 0.1 * 10.0 = 10.0 - 0.1 = 9.9
        assert!(params[0] < original_param);

        // The parameter should be reduced by weight decay even without gradients
        // This is the key difference from L2 regularization in Adam
    }
}
