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

/// Step decay learning rate scheduler.
///
/// Reduces the learning rate by a multiplicative factor (gamma) every `step_size` epochs.
/// This is one of the most common learning rate schedules, allowing the model to make
/// large updates early in training and fine-tune with smaller updates later.
///
/// Formula: lr = initial_lr * gamma^(epoch / step_size)
///
/// # Fields
///
/// * `initial_lr` - Starting learning rate
/// * `step_size` - Number of epochs between each decay step
/// * `gamma` - Multiplicative factor for decay (typically 0.1 to 0.5)
/// * `current_epoch` - Current training epoch (0-indexed)
/// * `current_lr` - Current learning rate value
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::utils::lr_scheduler::{LRScheduler, StepDecay};
///
/// let mut scheduler = StepDecay::new(0.1, 3, 0.5);
/// assert_eq!(scheduler.get_lr(), 0.1);
///
/// // After 3 epochs
/// for _ in 0..3 {
///     scheduler.step();
/// }
/// assert_eq!(scheduler.get_lr(), 0.05); // 0.1 * 0.5
/// ```
pub struct StepDecay {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
}

impl StepDecay {
    /// Creates a new step decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate (must be positive)
    /// * `step_size` - Number of epochs between decay steps (must be > 0)
    /// * `gamma` - Decay factor applied at each step (typically 0.1-0.5)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scheduler = StepDecay::new(0.01, 5, 0.1);
    /// // LR will be: 0.01 for epochs 0-4, 0.001 for epochs 5-9, etc.
    /// ```
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepDecay {
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;
        let num_decays = self.current_epoch / self.step_size;
        self.current_lr = self.initial_lr * self.gamma.powi(num_decays as i32);
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.initial_lr;
    }
}

/// Exponential decay learning rate scheduler.
///
/// Reduces the learning rate by a multiplicative factor (gamma) every epoch.
/// This provides smooth, continuous decay compared to step decay, allowing for
/// gradual reduction in learning rate throughout training.
///
/// Formula: lr = initial_lr * gamma^epoch
///
/// # Fields
///
/// * `initial_lr` - Starting learning rate
/// * `gamma` - Multiplicative decay factor applied each epoch (typically 0.95 to 0.99)
/// * `current_epoch` - Current training epoch (0-indexed)
/// * `current_lr` - Current learning rate value
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::utils::lr_scheduler::{LRScheduler, ExponentialDecay};
///
/// let mut scheduler = ExponentialDecay::new(0.1, 0.95);
/// assert_eq!(scheduler.get_lr(), 0.1);
///
/// // After 1 epoch
/// scheduler.step();
/// assert_eq!(scheduler.get_lr(), 0.095); // 0.1 * 0.95
///
/// // After 2 epochs total
/// scheduler.step();
/// assert_eq!(scheduler.get_lr(), 0.09025); // 0.1 * 0.95^2
/// ```
pub struct ExponentialDecay {
    initial_lr: f32,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
}

impl ExponentialDecay {
    /// Creates a new exponential decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate (must be positive)
    /// * `gamma` - Decay factor applied each epoch (typically 0.95-0.99)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scheduler = ExponentialDecay::new(0.01, 0.96);
    /// // LR will be: 0.01 at epoch 0, 0.0096 at epoch 1, 0.009216 at epoch 2, etc.
    /// ```
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for ExponentialDecay {
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;
        self.current_lr = self.initial_lr * self.gamma.powi(self.current_epoch as i32);
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.initial_lr;
    }
}
