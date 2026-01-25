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
/// Reads the file at `path` and deserializes its JSON contents into a `TrainingConfig`.
///
/// # Returns
///
/// `Ok(TrainingConfig)` on success, or an error if the file cannot be read or the JSON is invalid.
///
/// # Examples
///
/// ```no_run
/// use crate::config::load_config;
///
/// let cfg = load_config("config/mnist_mlp_step.json").unwrap();
/// assert_eq!(cfg.scheduler_type, "step_decay");
/// ```
pub fn load_config(path: &str) -> Result<TrainingConfig, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let config: TrainingConfig = serde_json::from_str(&contents)?;
    Ok(config)
}