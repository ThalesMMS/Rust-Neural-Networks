//! Shared utilities for neural network implementations
//!
//! This module provides common utilities like random number generation,
//! activation functions, and other helper functions used across models.

pub mod activations;
pub mod gradient_clipping;
pub mod lr_scheduler;
pub mod rng;

// Re-export commonly used gradient clipping functions
pub use gradient_clipping::{clip_gradient_norm, clip_gradient_value};

// Re-export the learning rate scheduler trait and implementations
pub use lr_scheduler::{
    create_scheduler_from_config, ConstantLR, CosineAnnealing, ExponentialDecay, LRScheduler,
    StepDecay,
};

// Re-export the RNG for convenience
pub use rng::SimpleRng;
