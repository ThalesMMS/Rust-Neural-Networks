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
//! - `autoencoder`: Vanilla and variational autoencoder architectures
//! - `data`: Data loading utilities for various datasets
//! - `sweep`: Hyperparameter sweep configuration structures
//! - `profiling`: Performance profiling and metrics collection
//! - `step_debug`: Interactive step-through debugging for training
//! - `compression`: Model compression techniques (quantization, pruning)

pub mod architecture;
pub mod autoencoder;
pub mod autograd;
pub mod benchmark;
pub mod compression;
pub mod config;
pub mod data;
pub mod layers;
pub mod optimizers;
pub mod profiling;
pub mod step_debug;
pub mod sweep;
pub mod training;
pub mod utils;
