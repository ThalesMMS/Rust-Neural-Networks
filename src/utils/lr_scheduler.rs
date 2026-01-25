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
    /// Creates a step-wise learning rate scheduler that multiplies the learning rate by `gamma` every `step_size` epochs.
    ///
    /// # Parameters
    ///
    /// - `initial_lr`: Starting learning rate; must be greater than 0.0.
    /// - `step_size`: Number of epochs between decay steps; must be greater than 0.
    /// - `gamma`: Decay factor applied at each step; typically between 0.0 and 1.0.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut scheduler = StepDecay::new(0.01, 5, 0.1);
    /// assert_eq!(scheduler.get_lr(), 0.01);
    /// // advance 5 epochs
    /// for _ in 0..5 { scheduler.step(); }
    /// assert_eq!(scheduler.get_lr(), 0.001);
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

    /// Create a StepDecay scheduler from a TrainingConfig.
    ///
    /// The `config` must provide `step_size` and `gamma`; those values are used to
    /// construct the scheduler. This function sets the scheduler's initial learning
    /// rate to 0.01 as a placeholder.
    ///
    /// # Panics
    ///
    /// Panics if `step_size` or `gamma` are missing from `config`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::config::load_config;
    ///
    /// let config = load_config("config/mnist_mlp_step.json").unwrap();
    /// let scheduler = rust_neural_networks::utils::lr_scheduler::StepDecay::from_config(&config);
    /// assert_eq!(scheduler.get_lr(), 0.01);
    /// ```
    pub fn from_config(config: &crate::config::TrainingConfig) -> Self {
        let step_size = config.step_size.expect("step_size required for StepDecay");
        let gamma = config.gamma.expect("gamma required for StepDecay");
        // Note: initial_lr is set to a placeholder (0.01), but binaries should use
        // their own LEARNING_RATE constant when creating schedulers
        Self::new(0.01, step_size, gamma)
    }
}

impl LRScheduler for StepDecay {
    /// Retrieve the current learning rate used for training.
    ///
    /// # Returns
    ///
    /// The current learning rate (`f32`).
    ///
    /// # Examples
    ///
    /// ```
    /// let sched = crate::utils::lr_scheduler::StepDecay::new(0.1, 10, 0.5);
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Advances the scheduler by one epoch and updates the current learning rate.
    ///
    /// After calling this method the scheduler's internal epoch counter is incremented
    /// and the current learning rate is recalculated according to the schedule.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = crate::utils::lr_scheduler::StepDecay::new(0.1, 2, 0.5);
    /// assert_eq!(sched.get_lr(), 0.1);
    /// sched.step(); // epoch 1 -> no decay yet
    /// assert_eq!(sched.get_lr(), 0.1);
    /// sched.step(); // epoch 2 -> one decay applied
    /// assert_eq!(sched.get_lr(), 0.05);
    /// ```
    fn step(&mut self) {
        self.current_epoch += 1;
        let num_decays = self.current_epoch / self.step_size;
        self.current_lr = self.initial_lr * self.gamma.powi(num_decays as i32);
    }

    /// Resets the scheduler to its initial state.
    ///
    /// After calling this method the current epoch is set to 0 and the current learning rate
    /// is restored to the scheduler's initial learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = StepDecay::new(0.1, 5, 0.5);
    /// // advance the scheduler so lr changes
    /// sched.step();
    /// assert_ne!(sched.get_lr(), 0.1);
    /// // reset to initial state
    /// sched.reset();
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
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
    /// Creates an exponential learning-rate scheduler that multiplies the learning rate by `gamma` each epoch.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate; should be greater than 0.0.
    /// * `gamma` - Decay factor applied each epoch (typical values are between 0.0 and 1.0, e.g., 0.95–0.99).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = ExponentialDecay::new(0.01, 0.96);
    /// // epoch 0
    /// assert!((sched.get_lr() - 0.01).abs() < 1e-8);
    /// sched.step(); // advance to epoch 1
    /// assert!((sched.get_lr() - 0.01 * 0.96).abs() < 1e-8);
    /// ```
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }

    /// Creates an ExponentialDecay scheduler from a TrainingConfig.
    ///
    /// # Panics
    /// Panics if `config.decay_rate` is `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::config::load_config;
    /// let config = load_config("config/mnist_mlp_exponential.json").unwrap();
    /// let scheduler = ExponentialDecay::from_config(&config);
    /// assert_eq!(scheduler.get_lr(), 0.01);
    /// ```
    pub fn from_config(config: &crate::config::TrainingConfig) -> Self {
        let gamma = config
            .decay_rate
            .expect("decay_rate required for ExponentialDecay");
        Self::new(0.01, gamma)
    }
}

impl LRScheduler for ExponentialDecay {
    /// Retrieve the current learning rate used for training.
    ///
    /// # Returns
    ///
    /// The current learning rate (`f32`).
    ///
    /// # Examples
    ///
    /// ```
    /// let sched = crate::utils::lr_scheduler::StepDecay::new(0.1, 10, 0.5);
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Advance the scheduler by one epoch and update the current learning rate.
    ///
    /// Increments the internal epoch counter and recalculates `current_lr` based on
    /// the initial learning rate and the decay factor.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = ExponentialDecay::new(0.1, 0.9);
    /// assert_eq!(sched.get_lr(), 0.1);
    /// sched.step();
    /// let expected = 0.1 * 0.9_f32.powi(1);
    /// assert!((sched.get_lr() - expected).abs() < 1e-6);
    /// ```
    fn step(&mut self) {
        self.current_epoch += 1;
        self.current_lr = self.initial_lr * self.gamma.powi(self.current_epoch as i32);
    }

    /// Resets the scheduler to its initial state.
    ///
    /// After calling this method the current epoch is set to 0 and the current learning rate
    /// is restored to the scheduler's initial learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = StepDecay::new(0.1, 5, 0.5);
    /// // advance the scheduler so lr changes
    /// sched.step();
    /// assert_ne!(sched.get_lr(), 0.1);
    /// // reset to initial state
    /// sched.reset();
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.initial_lr;
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Adjusts the learning rate following a cosine curve from the initial learning rate
/// down to a minimum learning rate over T_max epochs. This provides smooth decay that
/// starts fast and slows down, which can help with convergence.
///
/// Formula: lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(π * epoch / T_max))
///
/// # Fields
///
/// * `initial_lr` - Maximum learning rate at the start of the cycle
/// * `eta_min` - Minimum learning rate at the end of the cycle
/// * `t_max` - Number of epochs for one complete cosine cycle
/// * `current_epoch` - Current training epoch (0-indexed)
/// * `current_lr` - Current learning rate value
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::utils::lr_scheduler::{LRScheduler, CosineAnnealing};
///
/// let mut scheduler = CosineAnnealing::new(0.1, 0.0, 10);
/// assert_eq!(scheduler.get_lr(), 0.1);
///
/// // After 5 epochs (halfway through cycle)
/// for _ in 0..5 {
///     scheduler.step();
/// }
/// // LR will be close to eta_min (0.0)
///
/// // After 10 epochs (end of cycle)
/// for _ in 0..5 {
///     scheduler.step();
/// }
/// assert!(scheduler.get_lr() < 0.01); // Near eta_min
/// ```
pub struct CosineAnnealing {
    initial_lr: f32,
    eta_min: f32,
    t_max: usize,
    current_epoch: usize,
    current_lr: f32,
}

impl CosineAnnealing {
    /// Creates a cosine-annealing learning rate scheduler that decays the learning rate
    /// from `initial_lr` down to `eta_min` following a half-cosine curve over `t_max` epochs.
    ///
    /// `initial_lr` should be greater than 0. `t_max` must be greater than 0; `eta_min` may be
    /// equal to or greater than 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut sched = CosineAnnealing::new(0.01, 0.0001, 50);
    /// // initial learning rate is the starting value
    /// assert_eq!(sched.get_lr(), 0.01);
    /// // advancing the scheduler reduces the learning rate
    /// sched.step();
    /// assert!(sched.get_lr() < 0.01);
    /// ```
    pub fn new(initial_lr: f32, eta_min: f32, t_max: usize) -> Self {
        Self {
            initial_lr,
            eta_min,
            t_max,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }

    /// Constructs a CosineAnnealing scheduler from a TrainingConfig.
    ///
    /// Reads `min_lr` and `T_max` from the provided config and uses them as `eta_min`
    /// and `t_max` respectively. Panics if either `min_lr` or `T_max` is missing.
    ///
    /// # Panics
    ///
    /// Panics if `config.min_lr` or `config.T_max` is `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = crate::config::load_config("config/mnist_mlp_cosine.json").unwrap();
    /// let scheduler = crate::utils::lr_scheduler::CosineAnnealing::from_config(&config);
    /// ```
    pub fn from_config(config: &crate::config::TrainingConfig) -> Self {
        let eta_min = config.min_lr.expect("min_lr required for CosineAnnealing");
        let t_max = config.T_max.expect("T_max required for CosineAnnealing");
        Self::new(0.01, eta_min, t_max)
    }
}

impl LRScheduler for CosineAnnealing {
    /// Retrieve the current learning rate used for training.
    ///
    /// # Returns
    ///
    /// The current learning rate (`f32`).
    ///
    /// # Examples
    ///
    /// ```
    /// let sched = crate::utils::lr_scheduler::StepDecay::new(0.1, 10, 0.5);
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Advances the scheduler by one epoch and updates the current learning rate using cosine annealing.
    ///
    /// Increments the internal epoch counter and sets `current_lr` to a value between `initial_lr`
    /// and `eta_min` according to the cosine progression over `t_max` epochs.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = CosineAnnealing::new(0.1, 0.001, 100);
    /// let before = sched.get_lr();
    /// sched.step();
    /// let after = sched.get_lr();
    /// assert!(after >= 0.001 && after <= before);
    /// ```
    fn step(&mut self) {
        self.current_epoch += 1;
        let progress = (self.current_epoch as f32) / (self.t_max as f32);
        let cosine_term = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_term;
    }

    /// Resets the scheduler to its initial state.
    ///
    /// After calling this method the current epoch is set to 0 and the current learning rate
    /// is restored to the scheduler's initial learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut sched = StepDecay::new(0.1, 5, 0.5);
    /// // advance the scheduler so lr changes
    /// sched.step();
    /// assert_ne!(sched.get_lr(), 0.1);
    /// // reset to initial state
    /// sched.reset();
    /// assert_eq!(sched.get_lr(), 0.1);
    /// ```
    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.initial_lr;
    }
}