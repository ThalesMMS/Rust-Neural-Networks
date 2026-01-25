//! Learning rate scheduler trait and implementations
//!
//! This module defines the LRScheduler trait for adjusting learning rates during training.
//! Schedulers can implement various decay strategies (step decay, exponential, cosine annealing)
//! to improve convergence and final model performance.

/// Core trait for learning rate schedulers.
///
/// Schedulers adjust the learning rate during training based on the current epoch.
/// This allows for more sophisticated training strategies than a constant learning rate.
///
/// # Type Parameters
///
/// Schedulers work with f32 learning rates for compatibility with neural network training.
///
/// # Example
///
/// ```ignore
/// // Create a scheduler
/// let mut scheduler = StepDecay::new(0.1, 3, 0.5);
///
/// // During training loop
/// for epoch in 0..num_epochs {
///     let lr = scheduler.get_lr();
///     // ... train with current learning rate ...
///     scheduler.step();  // Update LR for next epoch
/// }
///
/// // Reset to initial learning rate if needed
/// scheduler.reset();
/// ```
pub trait LRScheduler {
    /// Get the current learning rate.
    ///
    /// Returns the learning rate that should be used for the current training step.
    /// This value is computed based on the scheduler's internal state (current epoch,
    /// configuration parameters, etc.).
    ///
    /// # Returns
    ///
    /// The current learning rate as an f32.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lr = scheduler.get_lr();
    /// layer.update_parameters(lr);
    /// ```
    fn get_lr(&self) -> f32;

    /// Advance the scheduler to the next epoch.
    ///
    /// Updates the scheduler's internal state (e.g., incrementing epoch counter)
    /// and recalculates the learning rate for the next training epoch.
    /// Call this method once per epoch, typically at the end of each epoch.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // At the end of each epoch
    /// scheduler.step();
    /// ```
    fn step(&mut self);

    /// Reset the scheduler to its initial state.
    ///
    /// Resets the learning rate to the initial value and clears any internal
    /// state (e.g., sets epoch counter back to 0). Useful when restarting
    /// training or running multiple training sessions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After training, reset for a new run
    /// scheduler.reset();
    /// assert_eq!(scheduler.get_lr(), initial_lr);
    /// ```
    fn reset(&mut self);
}
