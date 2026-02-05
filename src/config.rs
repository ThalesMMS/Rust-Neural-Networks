//! Configuration structures for training
//!
//! This module provides configuration structures for setting up training parameters,
//! particularly learning rate scheduling configurations.

use serde::Deserialize;
use std::error::Error;
use std::fs;

/// Configuration for training, including learning rate scheduler and activation function settings
///
/// This structure is used to parse training configuration from JSON files.
/// Different scheduler types require different optional fields:
///
/// - **StepDecay**: Requires `step_size` and `gamma`
/// - **ExponentialDecay**: Requires `decay_rate`
/// - **CosineAnnealing**: Requires `min_lr` and `T_max`
///
/// Activation function can be specified with optional parameters:
///
/// - **ReLU**: No parameters (default)
/// - **LeakyReLU**: Optional `leaky_relu_alpha` (default 0.01)
/// - **ELU**: Optional `elu_alpha` (default 1.0)
/// - **GELU**: No parameters
/// - **Swish**: No parameters
/// - **Tanh**: No parameters
///
/// # Example
///
/// ```json
/// {
///   "scheduler_type": "step_decay",
///   "step_size": 3,
///   "gamma": 0.5,
///   "activation_function": "leaky_relu",
///   "leaky_relu_alpha": 0.01,
///   "learning_rate": 0.01,
///   "epochs": 10,
///   "batch_size": 64,
///   "validation_split": 0.1,
///   "early_stopping_patience": 3,
///   "early_stopping_min_delta": 0.001
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

    /// Activation function type: "relu", "leaky_relu", "elu", "gelu", "swish", or "tanh"
    pub activation_function: Option<String>,

    /// Alpha parameter for Leaky ReLU activation (default 0.01)
    pub leaky_relu_alpha: Option<f32>,

    /// Alpha parameter for ELU activation (default 1.0)
    pub elu_alpha: Option<f32>,

    /// Learning rate for training (must be positive if specified)
    pub learning_rate: Option<f32>,

    /// Number of training epochs (must be positive if specified)
    pub epochs: Option<usize>,

    /// Batch size for training (must be positive if specified)
    pub batch_size: Option<usize>,

    /// Fraction of training data to use for validation (must be in [0.0, 1.0] if specified)
    pub validation_split: Option<f32>,

    /// Number of epochs to wait for improvement before early stopping (optional)
    pub early_stopping_patience: Option<usize>,

    /// Minimum change in validation loss to qualify as improvement (optional)
    pub early_stopping_min_delta: Option<f32>,
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
/// use rust_neural_networks::config::load_config;
///
/// let cfg = load_config("config/mnist_mlp_step.json").unwrap();
/// assert_eq!(cfg.scheduler_type, "step_decay");
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

    // Validate activation function type
    if let Some(ref activation) = config.activation_function {
        let valid_activations = ["relu", "leaky_relu", "elu", "gelu", "swish", "tanh"];
        if !valid_activations.contains(&activation.as_str()) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid activation function '{}'. Must be one of: {}",
                    activation,
                    valid_activations.join(", ")
                ),
            )));
        }
    }

    // Validate activation function parameters
    if let Some(alpha) = config.leaky_relu_alpha {
        if alpha < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "leaky_relu_alpha must be non-negative",
            )));
        }
    }

    if let Some(alpha) = config.elu_alpha {
        if alpha <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "elu_alpha must be positive",
            )));
        }
    }

    // Validate training hyperparameters
    if let Some(learning_rate) = config.learning_rate {
        if learning_rate <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "learning_rate must be positive",
            )));
        }
    }

    if let Some(epochs) = config.epochs {
        if epochs == 0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "epochs must be positive",
            )));
        }
    }

    if let Some(batch_size) = config.batch_size {
        if batch_size == 0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "batch_size must be positive",
            )));
        }
    }

    if let Some(validation_split) = config.validation_split {
        if validation_split < 0.0 || validation_split > 1.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validation_split must be in the range [0.0, 1.0]",
            )));
        }
    }

    if let Some(min_delta) = config.early_stopping_min_delta {
        if min_delta < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "early_stopping_min_delta must be non-negative",
            )));
        }
    }

    Ok(())
}
