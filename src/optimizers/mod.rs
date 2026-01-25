//! Optimizer abstractions for neural network parameter updates
//!
//! This module provides the Optimizer trait and implementations for different
//! optimization algorithms used to update neural network parameters during training.
//!
//! # Overview
//!
//! Optimizers define how to use gradients to update model parameters. The basic
//! gradient descent update is `weight = weight - learning_rate * gradient`, but
//! modern optimizers like Adam use momentum and adaptive learning rates to improve
//! convergence.
//!
//! # Available Optimizers
//!
//! - SGD: Vanilla stochastic gradient descent
//! - Adam: Adaptive moment estimation with momentum and adaptive learning rates
//!
//! # Example
//!
//! ```ignore
//! use rust_neural_networks::optimizers::{Optimizer, Adam};
//!
//! // Create Adam optimizer with default parameters
//! let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
//!
//! // Update parameters after computing gradients
//! optimizer.update(&mut weights, &gradients);
//! ```

pub mod adam;
pub mod sgd;

pub use adam::Adam;
pub use sgd::SGD;

/// Core trait for neural network optimizers.
///
/// All optimizer types (SGD, Adam, etc.) implement this trait to provide
/// a uniform interface for parameter updates during training.
///
/// # Type Parameters
///
/// Optimizers work with f32 data for compatibility with BLAS operations and GPU acceleration.
///
/// # State Management
///
/// Some optimizers (like Adam) maintain internal state across updates:
/// - Momentum estimates
/// - Adaptive learning rate statistics
/// - Time step counters
///
/// The optimizer manages this state internally, so callers only need to
/// provide parameters and gradients.
///
/// # Example
///
/// ```ignore
/// // Create optimizer instance
/// let mut optimizer = Adam::new(learning_rate, beta1, beta2, epsilon);
///
/// // In training loop:
/// for epoch in 0..num_epochs {
///     // Forward pass and backward pass to compute gradients
///     layer.backward(&input, &grad_output, &mut grad_input, batch_size);
///
///     // Update parameters using optimizer
///     optimizer.update(&mut layer.weights, &layer.weight_gradients);
///     optimizer.update(&mut layer.biases, &layer.bias_gradients);
/// }
/// ```
pub trait Optimizer {
    /// Update parameters using gradients.
    ///
    /// Applies the optimizer's update rule to modify parameters in-place.
    /// The specific update rule depends on the optimizer:
    /// - SGD: `param = param - learning_rate * grad`
    /// - Adam: Uses momentum and adaptive learning rates with bias correction
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameters to update (weights or biases)
    /// * `gradients` - Gradient of loss with respect to each parameter
    ///
    /// # Panics
    ///
    /// Implementations may panic if parameters and gradients have different lengths.
    ///
    /// # Notes
    ///
    /// - Gradients should be computed via backpropagation before calling this method
    /// - For batch training, gradients are typically averaged over the batch
    /// - This method may update internal optimizer state (momentum, adaptive rates, etc.)
    fn update(&mut self, parameters: &mut [f32], gradients: &[f32]);

    /// Reset optimizer state.
    ///
    /// Clears any accumulated momentum, adaptive learning rate statistics,
    /// or other internal state. Useful when starting a new training run or
    /// switching between different datasets.
    ///
    /// For stateless optimizers like vanilla SGD, this may be a no-op.
    fn reset(&mut self);

    /// Get the learning rate for this optimizer.
    ///
    /// Returns the base learning rate. Note that adaptive optimizers may
    /// apply different effective learning rates to different parameters.
    fn learning_rate(&self) -> f32;

    /// Set the learning rate for this optimizer.
    ///
    /// Updates the base learning rate. Useful for implementing learning rate
    /// schedules or decay strategies.
    ///
    /// # Arguments
    ///
    /// * `lr` - New learning rate value (must be positive)
    fn set_learning_rate(&mut self, lr: f32);
}
