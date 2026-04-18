//! Hyperparameter sweep configuration structures
//!
//! This module provides configuration structures for defining hyperparameter sweep ranges.
//! A sweep explores combinations of different parameter values to find optimal configurations.

mod config;
mod generation;
mod validation;

pub use self::config::{SweepConfig, SweepResult};
pub use self::generation::generate_configs;

use self::validation::validate_sweep_config;
use std::error::Error;
use std::fs;

/// Loads and validates a sweep configuration from a JSON file.
///
/// Reads the file at `path`, deserializes its contents into a `SweepConfig`,
/// and validates the resulting configuration.
///
/// # Returns
///
/// `Ok(SweepConfig)` containing the validated configuration on success; `Err` if the file
/// cannot be read, the JSON is invalid, or validation fails.
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::sweep::load_sweep_config;
///
/// let sweep = load_sweep_config("config/sweeps/mnist_mlp_sweep.json").unwrap();
/// assert_eq!(sweep.target_binary, "mnist_mlp");
/// ```
pub fn load_sweep_config(path: &str) -> Result<SweepConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: SweepConfig = serde_json::from_str(&contents)?;
    validate_sweep_config(&config)?;
    Ok(config)
}
