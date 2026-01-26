//! Rust Neural Networks Library
//!
//! This library provides shared abstractions for neural network layers and utilities
//! to reduce code duplication between model implementations.
//!
//! # Modules
//!
//! - `layers`: Layer trait and implementations (Dense, Conv2D, etc.)
//! - `optimizers`: Optimizer trait and implementations (SGD, Adam, etc.)
//! - `utils`: Shared utilities (RNG, activation functions, etc.)
//! - `config`: Training configuration structures
//! - `architecture`: Architecture configuration and model building

pub mod architecture;
pub mod config;
pub mod layers;
pub mod optimizers;
pub mod utils;
