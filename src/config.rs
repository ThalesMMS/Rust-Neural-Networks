//! Configuration structures for training
//!
//! This module provides configuration structures for setting up training parameters,
//! particularly learning rate scheduling configurations.

use serde::Deserialize;
use std::error::Error;
use std::fs;

/// Configuration for training, including learning rate scheduler settings
///
/// This structure is used to parse training configuration from JSON files.
/// Different scheduler types require different optional fields:
///
/// - **StepDecay**: Requires `step_size` and `gamma`
/// - **ExponentialDecay**: Requires `decay_rate`
/// - **CosineAnnealing**: Requires `min_lr` and `T_max`
///
/// # Example
///
/// ```json
/// {
///   "scheduler_type": "step_decay",
///   "step_size": 3,
///   "gamma": 0.5
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
pub struct TrainingConfig {
    /// Type of learning rate scheduler: "step_decay", "exponential", or "cosine_annealing"
    pub scheduler_type: String,

    /// Step size for StepDecay scheduler (epochs between LR reductions)
    pub step_size: Option<usize>,

    /// Multiplicative factor for StepDecay scheduler (LR multiplier)
    pub gamma: Option<f32>,

    /// Decay rate for ExponentialDecay scheduler (per-epoch multiplier)
    pub decay_rate: Option<f32>,

    /// Minimum learning rate for CosineAnnealing scheduler
    pub min_lr: Option<f32>,

    /// Total number of epochs for CosineAnnealing scheduler
    pub T_max: Option<usize>,
}

/// Loads a training configuration from a JSON file.
///
/// This function reads a JSON file at the specified path and deserializes it into
/// a `TrainingConfig` struct. The file should contain valid JSON matching the
/// structure defined by `TrainingConfig`.
///
/// # Arguments
///
/// * `path` - Path to the JSON configuration file
///
/// # Returns
///
/// Returns `Ok(TrainingConfig)` if the file is successfully read and parsed,
/// or an error if the file cannot be read or contains invalid JSON.
///
/// # Errors
///
/// This function will return an error if:
/// - The file does not exist or cannot be read
/// - The file contains invalid JSON syntax
/// - The JSON structure does not match the expected `TrainingConfig` format
///
/// # Examples
///
/// ```ignore
/// use rust_neural_networks::config::load_config;
///
/// // Load a step decay scheduler configuration
/// let config = load_config("config/mnist_mlp_step.json").unwrap();
/// assert_eq!(config.scheduler_type, "step_decay");
/// assert_eq!(config.step_size, Some(3));
/// assert_eq!(config.gamma, Some(0.5));
/// ```
pub fn load_config(path: &str) -> Result<TrainingConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: TrainingConfig = serde_json::from_str(&contents)?;
    validate_config(&config)?;
    Ok(config)
}

fn validate_config(config: &TrainingConfig) -> Result<(), Box<dyn Error>> {
    if let Some(gamma) = config.gamma {
        if gamma < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "gamma must be non-negative",
            )));
        }
    }

    if let Some(decay_rate) = config.decay_rate {
        if decay_rate < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decay_rate must be non-negative",
            )));
        }
    }

    if let Some(min_lr) = config.min_lr {
        if min_lr < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "min_lr must be non-negative",
            )));
        }
    }

    Ok(())
}
