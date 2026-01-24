//! Rust Neural Networks Library
//!
//! This library provides shared abstractions for neural network layers and utilities
//! to reduce code duplication between model implementations.
//!
//! # Modules
//!
//! - `layers`: Layer trait and implementations (Dense, Conv2D, etc.)
//! - `utils`: Shared utilities (RNG, activation functions, etc.)

pub mod layers;
pub mod utils;
