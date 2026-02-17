//! Model compression techniques for neural networks
//!
//! This module provides compression methods to reduce model size and improve
//! inference efficiency, including quantization and pruning techniques.
//!
//! # Submodules
//!
//! - `quantization`: INT8 quantization for reducing weight precision
//! - `pruning`: Magnitude-based pruning for removing low-importance weights

pub mod pruning;
pub mod quantization;
