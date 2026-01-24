//! Shared utilities for neural network implementations
//!
//! This module provides common utilities like random number generation,
//! activation functions, and other helper functions used across models.

pub mod activations;
pub mod rng;

// Re-export commonly used items for convenience
pub use activations::{relu_inplace, sigmoid, sigmoid_derivative, softmax_rows};
pub use rng::SimpleRng;
