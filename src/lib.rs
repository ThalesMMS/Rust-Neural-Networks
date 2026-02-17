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
//! - `data`: Data loading utilities for various datasets
//! - `sweep`: Hyperparameter sweep configuration structures

pub mod architecture;
pub mod benchmark;
pub mod config;
pub mod data;
pub mod layers;
pub mod optimizers;
pub mod sweep;
pub mod utils;
