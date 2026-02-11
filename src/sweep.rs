//! Hyperparameter sweep configuration structures
//!
//! This module provides configuration structures for defining hyperparameter sweep ranges.
//! A sweep explores combinations of different parameter values to find optimal configurations.

use crate::config::{load_config, TrainingConfig};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

/// Configuration for hyperparameter sweeps
///
/// This structure defines parameter ranges to explore during a sweep.
/// Each parameter can have multiple values specified as a vector.
/// The sweep orchestrator will generate all combinations (Cartesian product) of these values.
///
/// All parameters are optional - specify only the parameters you want to vary.
/// Parameters not specified will use the base configuration values.
///
/// # Example
///
/// ```json
/// {
///   "base_config": "config/training/mnist_mlp_default.json",
///   "target_binary": "mnist_mlp",
///   "learning_rate": [0.001, 0.01, 0.1],
///   "batch_size": [32, 64, 128],
///   "epochs": [5, 10],
///   "description": "Learning rate and batch size sweep for MNIST MLP"
/// }
/// ```
///
/// This example would generate 3 × 3 × 2 = 18 different configurations to train.
#[derive(Debug, Clone, Deserialize)]
pub struct SweepConfig {
    /// Path to base training configuration file
    /// Parameters from this config are used as defaults, overridden by sweep values
    pub base_config: String,

    /// Target binary to run (e.g., "mnist_mlp", "mnist_cnn", "cifar10_cnn")
    pub target_binary: String,

    /// Optional description of the sweep experiment
    pub description: Option<String>,

    /// Learning rate values to explore
    pub learning_rate: Option<Vec<f32>>,

    /// Batch size values to explore
    pub batch_size: Option<Vec<usize>>,

    /// Epoch count values to explore
    pub epochs: Option<Vec<usize>>,

    /// Validation split values to explore (fraction in [0.0, 1.0])
    pub validation_split: Option<Vec<f32>>,

    /// Early stopping patience values to explore
    pub early_stopping_patience: Option<Vec<usize>>,

    /// Early stopping minimum delta values to explore
    pub early_stopping_min_delta: Option<Vec<f32>>,

    /// Scheduler type values to explore
    /// Valid values: "step_decay", "exponential", "cosine_annealing"
    pub scheduler_type: Option<Vec<String>>,

    /// Step size values to explore (for StepDecay scheduler)
    pub step_size: Option<Vec<usize>>,

    /// Gamma values to explore (for StepDecay scheduler)
    pub gamma: Option<Vec<f32>>,

    /// Decay rate values to explore (for ExponentialDecay scheduler)
    pub decay_rate: Option<Vec<f32>>,

    /// Minimum learning rate values to explore (for CosineAnnealing scheduler)
    pub min_lr: Option<Vec<f32>>,

    /// T_max values to explore (for CosineAnnealing scheduler - total epochs for cycle)
    #[allow(non_snake_case)]
    pub T_max: Option<Vec<usize>>,

    /// Activation function values to explore
    /// Valid values: "relu", "leaky_relu", "elu", "gelu", "swish", "tanh"
    pub activation_function: Option<Vec<String>>,

    /// Leaky ReLU alpha values to explore
    pub leaky_relu_alpha: Option<Vec<f32>>,

    /// ELU alpha values to explore
    pub elu_alpha: Option<Vec<f32>>,
}

/// Results from a single training run in a hyperparameter sweep
///
/// This structure captures the final metrics and configuration parameters
/// from a completed training run. It is used to aggregate results across
/// all configurations in a sweep for comparison and analysis.
///
/// The structure includes both the achieved metrics (losses, accuracy, timing)
/// and the configuration parameters that produced those results, enabling
/// correlation analysis between hyperparameters and performance.
///
/// # Example JSON Output
///
/// ```json
/// {
///   "config_id": 1,
///   "learning_rate": 0.01,
///   "batch_size": 64,
///   "epochs_completed": 10,
///   "scheduler_type": "step_decay",
///   "activation_function": "relu",
///   "final_train_loss": 0.1234,
///   "final_val_loss": 0.2345,
///   "final_val_accuracy": 0.9567,
///   "total_training_time": 123.45,
///   "log_file": "./logs/training_loss_20260211_123456.csv"
/// }
/// ```
///
/// # Serialization
///
/// This struct implements `Serialize` to enable JSON export for:
/// - Aggregated sweep results files (e.g., `sweep_results_<timestamp>.json`)
/// - Python visualization tools that compare configurations
/// - Long-term experiment tracking and reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// Configuration identifier (1-indexed)
    pub config_id: usize,

    /// Learning rate used for this training run
    pub learning_rate: f32,

    /// Batch size used for this training run
    pub batch_size: usize,

    /// Number of epochs completed (may be less than requested due to early stopping)
    pub epochs_completed: usize,

    /// Scheduler type used ("step_decay", "exponential", or "cosine_annealing")
    pub scheduler_type: String,

    /// Activation function used (e.g., "relu", "leaky_relu", "elu", etc.)
    pub activation_function: Option<String>,

    /// Final training loss from the last epoch
    pub final_train_loss: f32,

    /// Final validation loss from the last epoch
    pub final_val_loss: f32,

    /// Final validation accuracy from the last epoch (fraction in [0.0, 1.0])
    pub final_val_accuracy: f32,

    /// Total training time in seconds
    pub total_training_time: f32,

    /// Path to the training log file for this run
    pub log_file: String,

    /// Step size parameter (for StepDecay scheduler)
    pub step_size: Option<usize>,

    /// Gamma parameter (for StepDecay scheduler)
    pub gamma: Option<f32>,

    /// Decay rate parameter (for ExponentialDecay scheduler)
    pub decay_rate: Option<f32>,

    /// Minimum learning rate (for CosineAnnealing scheduler)
    pub min_lr: Option<f32>,

    /// T_max parameter (for CosineAnnealing scheduler)
    #[allow(non_snake_case)]
    pub T_max: Option<usize>,

    /// Validation split used (fraction of training data)
    pub validation_split: Option<f32>,

    /// Early stopping patience setting
    pub early_stopping_patience: Option<usize>,

    /// Early stopping minimum delta setting
    pub early_stopping_min_delta: Option<f32>,

    /// Leaky ReLU alpha parameter (if using LeakyReLU activation)
    pub leaky_relu_alpha: Option<f32>,

    /// ELU alpha parameter (if using ELU activation)
    pub elu_alpha: Option<f32>,
}

/// Loads a sweep configuration from a JSON file.
///
/// Reads the file at `path` and deserializes its JSON contents into a `SweepConfig`.
/// Validates the configuration to ensure all parameter values are valid.
///
/// # Returns
///
/// `Ok(SweepConfig)` on success, or an error if the file cannot be read,
/// the JSON is invalid, or validation fails.
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

/// Validates a sweep configuration
///
/// Checks that all parameter values are within valid ranges and that
/// combinations make sense (e.g., all learning rates are positive).
fn validate_sweep_config(config: &SweepConfig) -> Result<(), Box<dyn Error>> {
    // Validate base_config path is not empty
    if config.base_config.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "base_config path cannot be empty",
        )));
    }

    // Validate target_binary is not empty
    if config.target_binary.is_empty() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "target_binary cannot be empty",
        )));
    }

    // Validate learning_rate values (must be positive)
    if let Some(ref learning_rates) = config.learning_rate {
        if learning_rates.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "learning_rate array cannot be empty",
            )));
        }
        for &lr in learning_rates {
            if lr <= 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("learning_rate must be positive, found: {}", lr),
                )));
            }
        }
    }

    // Validate batch_size values (must be positive)
    if let Some(ref batch_sizes) = config.batch_size {
        if batch_sizes.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "batch_size array cannot be empty",
            )));
        }
        for &bs in batch_sizes {
            if bs == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("batch_size must be positive, found: {}", bs),
                )));
            }
        }
    }

    // Validate epochs values (must be positive)
    if let Some(ref epochs) = config.epochs {
        if epochs.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "epochs array cannot be empty",
            )));
        }
        for &e in epochs {
            if e == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("epochs must be positive, found: {}", e),
                )));
            }
        }
    }

    // Validate validation_split values (must be in [0.0, 1.0])
    if let Some(ref splits) = config.validation_split {
        if splits.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "validation_split array cannot be empty",
            )));
        }
        for &split in splits {
            if !(0.0..=1.0).contains(&split) {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("validation_split must be in [0.0, 1.0], found: {}", split),
                )));
            }
        }
    }

    // Validate gamma values (must be non-negative)
    if let Some(ref gammas) = config.gamma {
        if gammas.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "gamma array cannot be empty",
            )));
        }
        for &g in gammas {
            if g < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("gamma must be non-negative, found: {}", g),
                )));
            }
        }
    }

    // Validate decay_rate values (must be non-negative)
    if let Some(ref decay_rates) = config.decay_rate {
        if decay_rates.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decay_rate array cannot be empty",
            )));
        }
        for &dr in decay_rates {
            if dr < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("decay_rate must be non-negative, found: {}", dr),
                )));
            }
        }
    }

    // Validate min_lr values (must be non-negative)
    if let Some(ref min_lrs) = config.min_lr {
        if min_lrs.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "min_lr array cannot be empty",
            )));
        }
        for &lr in min_lrs {
            if lr < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("min_lr must be non-negative, found: {}", lr),
                )));
            }
        }
    }

    // Validate scheduler_type values
    if let Some(ref schedulers) = config.scheduler_type {
        if schedulers.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "scheduler_type array cannot be empty",
            )));
        }
        let valid_schedulers = ["step_decay", "exponential", "cosine_annealing"];
        for scheduler in schedulers {
            if !valid_schedulers.contains(&scheduler.as_str()) {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Invalid scheduler_type '{}'. Must be one of: {}",
                        scheduler,
                        valid_schedulers.join(", ")
                    ),
                )));
            }
        }
    }

    // Validate activation_function values
    if let Some(ref activations) = config.activation_function {
        if activations.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "activation_function array cannot be empty",
            )));
        }
        let valid_activations = ["relu", "leaky_relu", "elu", "gelu", "swish", "tanh"];
        for activation in activations {
            if !valid_activations.contains(&activation.as_str()) {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Invalid activation_function '{}'. Must be one of: {}",
                        activation,
                        valid_activations.join(", ")
                    ),
                )));
            }
        }
    }

    // Validate leaky_relu_alpha values (must be non-negative)
    if let Some(ref alphas) = config.leaky_relu_alpha {
        if alphas.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "leaky_relu_alpha array cannot be empty",
            )));
        }
        for &alpha in alphas {
            if alpha < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("leaky_relu_alpha must be non-negative, found: {}", alpha),
                )));
            }
        }
    }

    // Validate elu_alpha values (must be positive)
    if let Some(ref alphas) = config.elu_alpha {
        if alphas.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "elu_alpha array cannot be empty",
            )));
        }
        for &alpha in alphas {
            if alpha <= 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("elu_alpha must be positive, found: {}", alpha),
                )));
            }
        }
    }

    Ok(())
}

/// Generates all configuration combinations from a sweep config
///
/// This function performs a Cartesian product of all parameter values specified
/// in the sweep config. It loads the base configuration and creates one
/// `TrainingConfig` for each combination, overriding base values with sweep values.
///
/// # Arguments
///
/// * `sweep` - The sweep configuration specifying parameter ranges
///
/// # Returns
///
/// A vector of `TrainingConfig` instances, one for each parameter combination.
/// If no sweep parameters are specified, returns a single config (the base config).
///
/// # Errors
///
/// Returns an error if the base config file cannot be loaded or parsed.
///
/// # Example
///
/// ```no_run
/// use rust_neural_networks::sweep::{load_sweep_config, generate_configs};
///
/// let sweep = load_sweep_config("config/sweeps/mnist_mlp_sweep.json").unwrap();
/// let configs = generate_configs(&sweep).unwrap();
/// // If sweep has 3 learning rates × 2 batch sizes = 6 configs generated
/// assert_eq!(configs.len(), 6);
/// ```
pub fn generate_configs(sweep: &SweepConfig) -> Result<Vec<TrainingConfig>, Box<dyn Error>> {
    // Load base configuration
    let base_config = load_config(&sweep.base_config)?;

    // Collect all parameter arrays and their sizes
    let learning_rates = sweep.learning_rate.as_deref().unwrap_or(&[]);
    let batch_sizes = sweep.batch_size.as_deref().unwrap_or(&[]);
    let epochs_list = sweep.epochs.as_deref().unwrap_or(&[]);
    let validation_splits = sweep.validation_split.as_deref().unwrap_or(&[]);
    let early_stopping_patiences = sweep.early_stopping_patience.as_deref().unwrap_or(&[]);
    let early_stopping_min_deltas = sweep.early_stopping_min_delta.as_deref().unwrap_or(&[]);
    let scheduler_types = sweep.scheduler_type.as_deref().unwrap_or(&[]);
    let step_sizes = sweep.step_size.as_deref().unwrap_or(&[]);
    let gammas = sweep.gamma.as_deref().unwrap_or(&[]);
    let decay_rates = sweep.decay_rate.as_deref().unwrap_or(&[]);
    let min_lrs = sweep.min_lr.as_deref().unwrap_or(&[]);
    let t_maxes = sweep.T_max.as_deref().unwrap_or(&[]);
    let activation_functions = sweep.activation_function.as_deref().unwrap_or(&[]);
    let leaky_relu_alphas = sweep.leaky_relu_alpha.as_deref().unwrap_or(&[]);
    let elu_alphas = sweep.elu_alpha.as_deref().unwrap_or(&[]);

    // Calculate total number of combinations
    let num_learning_rates = if learning_rates.is_empty() {
        1
    } else {
        learning_rates.len()
    };
    let num_batch_sizes = if batch_sizes.is_empty() {
        1
    } else {
        batch_sizes.len()
    };
    let num_epochs = if epochs_list.is_empty() {
        1
    } else {
        epochs_list.len()
    };
    let num_validation_splits = if validation_splits.is_empty() {
        1
    } else {
        validation_splits.len()
    };
    let num_early_stopping_patiences = if early_stopping_patiences.is_empty() {
        1
    } else {
        early_stopping_patiences.len()
    };
    let num_early_stopping_min_deltas = if early_stopping_min_deltas.is_empty() {
        1
    } else {
        early_stopping_min_deltas.len()
    };
    let num_scheduler_types = if scheduler_types.is_empty() {
        1
    } else {
        scheduler_types.len()
    };
    let num_step_sizes = if step_sizes.is_empty() {
        1
    } else {
        step_sizes.len()
    };
    let num_gammas = if gammas.is_empty() { 1 } else { gammas.len() };
    let num_decay_rates = if decay_rates.is_empty() {
        1
    } else {
        decay_rates.len()
    };
    let num_min_lrs = if min_lrs.is_empty() { 1 } else { min_lrs.len() };
    let num_t_maxes = if t_maxes.is_empty() { 1 } else { t_maxes.len() };
    let num_activation_functions = if activation_functions.is_empty() {
        1
    } else {
        activation_functions.len()
    };
    let num_leaky_relu_alphas = if leaky_relu_alphas.is_empty() {
        1
    } else {
        leaky_relu_alphas.len()
    };
    let num_elu_alphas = if elu_alphas.is_empty() {
        1
    } else {
        elu_alphas.len()
    };

    let total_combinations = num_learning_rates
        * num_batch_sizes
        * num_epochs
        * num_validation_splits
        * num_early_stopping_patiences
        * num_early_stopping_min_deltas
        * num_scheduler_types
        * num_step_sizes
        * num_gammas
        * num_decay_rates
        * num_min_lrs
        * num_t_maxes
        * num_activation_functions
        * num_leaky_relu_alphas
        * num_elu_alphas;

    let mut configs = Vec::with_capacity(total_combinations);

    // Generate all combinations using nested loops
    // This is verbose but explicit and correct for Cartesian product
    for lr_idx in 0..num_learning_rates {
        for bs_idx in 0..num_batch_sizes {
            for ep_idx in 0..num_epochs {
                for vs_idx in 0..num_validation_splits {
                    for esp_idx in 0..num_early_stopping_patiences {
                        for esmd_idx in 0..num_early_stopping_min_deltas {
                            for st_idx in 0..num_scheduler_types {
                                for ss_idx in 0..num_step_sizes {
                                    for g_idx in 0..num_gammas {
                                        for dr_idx in 0..num_decay_rates {
                                            for mlr_idx in 0..num_min_lrs {
                                                for tm_idx in 0..num_t_maxes {
                                                    for af_idx in 0..num_activation_functions {
                                                        for lra_idx in 0..num_leaky_relu_alphas {
                                                            for ea_idx in 0..num_elu_alphas {
                                                                // Create config by cloning base and overriding sweep values
                                                                let mut config =
                                                                    base_config.clone();

                                                                // Override with sweep values if specified
                                                                if !learning_rates.is_empty() {
                                                                    config.learning_rate = Some(
                                                                        learning_rates[lr_idx],
                                                                    );
                                                                }
                                                                if !batch_sizes.is_empty() {
                                                                    config.batch_size =
                                                                        Some(batch_sizes[bs_idx]);
                                                                }
                                                                if !epochs_list.is_empty() {
                                                                    config.epochs =
                                                                        Some(epochs_list[ep_idx]);
                                                                }
                                                                if !validation_splits.is_empty() {
                                                                    config.validation_split = Some(
                                                                        validation_splits[vs_idx],
                                                                    );
                                                                }
                                                                if !early_stopping_patiences
                                                                    .is_empty()
                                                                {
                                                                    config.early_stopping_patience = Some(early_stopping_patiences[esp_idx]);
                                                                }
                                                                if !early_stopping_min_deltas
                                                                    .is_empty()
                                                                {
                                                                    config.early_stopping_min_delta = Some(early_stopping_min_deltas[esmd_idx]);
                                                                }
                                                                if !scheduler_types.is_empty() {
                                                                    config.scheduler_type =
                                                                        scheduler_types[st_idx]
                                                                            .clone();
                                                                }
                                                                if !step_sizes.is_empty() {
                                                                    config.step_size =
                                                                        Some(step_sizes[ss_idx]);
                                                                }
                                                                if !gammas.is_empty() {
                                                                    config.gamma =
                                                                        Some(gammas[g_idx]);
                                                                }
                                                                if !decay_rates.is_empty() {
                                                                    config.decay_rate =
                                                                        Some(decay_rates[dr_idx]);
                                                                }
                                                                if !min_lrs.is_empty() {
                                                                    config.min_lr =
                                                                        Some(min_lrs[mlr_idx]);
                                                                }
                                                                if !t_maxes.is_empty() {
                                                                    config.T_max =
                                                                        Some(t_maxes[tm_idx]);
                                                                }
                                                                if !activation_functions.is_empty()
                                                                {
                                                                    config.activation_function =
                                                                        Some(
                                                                            activation_functions
                                                                                [af_idx]
                                                                                .clone(),
                                                                        );
                                                                }
                                                                if !leaky_relu_alphas.is_empty() {
                                                                    config.leaky_relu_alpha = Some(
                                                                        leaky_relu_alphas[lra_idx],
                                                                    );
                                                                }
                                                                if !elu_alphas.is_empty() {
                                                                    config.elu_alpha =
                                                                        Some(elu_alphas[ea_idx]);
                                                                }

                                                                configs.push(config);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(configs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_base_config() {
        let config = SweepConfig {
            base_config: String::new(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_config"));
    }

    #[test]
    fn test_validate_empty_target_binary() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: String::new(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("target_binary"));
    }

    #[test]
    fn test_validate_negative_learning_rate() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: Some(vec![0.01, -0.001]),
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("learning_rate"));
    }

    #[test]
    fn test_validate_zero_batch_size() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: Some(vec![32, 0, 128]),
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("batch_size"));
    }

    #[test]
    fn test_validate_invalid_validation_split() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: Some(vec![0.1, 1.5]),
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("validation_split"));
    }

    #[test]
    fn test_validate_invalid_scheduler_type() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: Some(vec![
                "step_decay".to_string(),
                "invalid_scheduler".to_string(),
            ]),
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scheduler_type"));
    }

    #[test]
    fn test_validate_invalid_activation_function() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: Some(vec!["relu".to_string(), "invalid_activation".to_string()]),
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("activation_function"));
    }

    #[test]
    fn test_validate_valid_config() {
        let config = SweepConfig {
            base_config: "config/mnist_mlp_default.json".to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: Some("Test sweep".to_string()),
            learning_rate: Some(vec![0.001, 0.01, 0.1]),
            batch_size: Some(vec![32, 64, 128]),
            epochs: Some(vec![5, 10]),
            validation_split: Some(vec![0.1, 0.2]),
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: Some(vec!["step_decay".to_string(), "exponential".to_string()]),
            step_size: None,
            gamma: Some(vec![0.5, 0.9]),
            decay_rate: Some(vec![0.95, 0.99]),
            min_lr: Some(vec![0.0001]),
            T_max: None,
            activation_function: Some(vec!["relu".to_string(), "leaky_relu".to_string()]),
            leaky_relu_alpha: Some(vec![0.01, 0.1]),
            elu_alpha: Some(vec![1.0]),
        };

        let result = validate_sweep_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_configs_single_parameter() {
        use std::fs;

        // Create temporary base config file
        let temp_dir = std::env::temp_dir();
        let base_config_path = temp_dir.join("test_base_config.json");
        let base_config_content = r#"{
            "scheduler_type": "step_decay",
            "step_size": 3,
            "gamma": 0.5,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 64
        }"#;
        fs::write(&base_config_path, base_config_content).unwrap();

        let sweep = SweepConfig {
            base_config: base_config_path.to_str().unwrap().to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: Some(vec![0.001, 0.01, 0.1]),
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let configs = generate_configs(&sweep).unwrap();

        // Should generate 3 configs (one for each learning rate)
        assert_eq!(configs.len(), 3);

        // Check that learning rates are correctly set
        assert_eq!(configs[0].learning_rate, Some(0.001));
        assert_eq!(configs[1].learning_rate, Some(0.01));
        assert_eq!(configs[2].learning_rate, Some(0.1));

        // Check that other parameters are preserved from base config
        for config in &configs {
            assert_eq!(config.batch_size, Some(64));
            assert_eq!(config.epochs, Some(10));
            assert_eq!(config.scheduler_type, "step_decay");
        }

        // Cleanup
        fs::remove_file(base_config_path).ok();
    }

    #[test]
    fn test_generate_configs_multiple_parameters() {
        use std::fs;

        // Create temporary base config file
        let temp_dir = std::env::temp_dir();
        let base_config_path = temp_dir.join("test_base_config_multi.json");
        let base_config_content = r#"{
            "scheduler_type": "step_decay",
            "step_size": 3,
            "gamma": 0.5,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 64
        }"#;
        fs::write(&base_config_path, base_config_content).unwrap();

        let sweep = SweepConfig {
            base_config: base_config_path.to_str().unwrap().to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: Some(vec![0.001, 0.01]),
            batch_size: Some(vec![32, 64, 128]),
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let configs = generate_configs(&sweep).unwrap();

        // Should generate 2 × 3 = 6 configs
        assert_eq!(configs.len(), 6);

        // Verify Cartesian product: all combinations should exist
        // Check by iterating and verifying each expected combination
        let has_combination = |lr: f32, bs: usize| -> bool {
            configs
                .iter()
                .any(|c| c.learning_rate == Some(lr) && c.batch_size == Some(bs))
        };

        // Check all 6 combinations exist
        assert!(has_combination(0.001, 32));
        assert!(has_combination(0.001, 64));
        assert!(has_combination(0.001, 128));
        assert!(has_combination(0.01, 32));
        assert!(has_combination(0.01, 64));
        assert!(has_combination(0.01, 128));

        // Cleanup
        fs::remove_file(base_config_path).ok();
    }

    #[test]
    fn test_generate_configs_no_sweep_parameters() {
        use std::fs;

        // Create temporary base config file
        let temp_dir = std::env::temp_dir();
        let base_config_path = temp_dir.join("test_base_config_empty.json");
        let base_config_content = r#"{
            "scheduler_type": "step_decay",
            "step_size": 3,
            "gamma": 0.5,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 64
        }"#;
        fs::write(&base_config_path, base_config_content).unwrap();

        let sweep = SweepConfig {
            base_config: base_config_path.to_str().unwrap().to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let configs = generate_configs(&sweep).unwrap();

        // Should generate 1 config (just the base config)
        assert_eq!(configs.len(), 1);

        // Should match base config values
        assert_eq!(configs[0].learning_rate, Some(0.01));
        assert_eq!(configs[0].batch_size, Some(64));
        assert_eq!(configs[0].epochs, Some(10));

        // Cleanup
        fs::remove_file(base_config_path).ok();
    }

    #[test]
    fn test_generate_configs_scheduler_types() {
        use std::fs;

        // Create temporary base config file
        let temp_dir = std::env::temp_dir();
        let base_config_path = temp_dir.join("test_base_config_scheduler.json");
        let base_config_content = r#"{
            "scheduler_type": "step_decay",
            "step_size": 3,
            "gamma": 0.5,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 64
        }"#;
        fs::write(&base_config_path, base_config_content).unwrap();

        let sweep = SweepConfig {
            base_config: base_config_path.to_str().unwrap().to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: None,
            batch_size: None,
            epochs: None,
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: Some(vec!["step_decay".to_string(), "exponential".to_string()]),
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let configs = generate_configs(&sweep).unwrap();

        // Should generate 2 configs (one for each scheduler type)
        assert_eq!(configs.len(), 2);

        // Check scheduler types
        assert_eq!(configs[0].scheduler_type, "step_decay");
        assert_eq!(configs[1].scheduler_type, "exponential");

        // Cleanup
        fs::remove_file(base_config_path).ok();
    }

    #[test]
    fn test_generate_configs_three_way_cartesian() {
        use std::fs;

        // Create temporary base config file
        let temp_dir = std::env::temp_dir();
        let base_config_path = temp_dir.join("test_base_config_three_way.json");
        let base_config_content = r#"{
            "scheduler_type": "step_decay",
            "step_size": 3,
            "gamma": 0.5,
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 64
        }"#;
        fs::write(&base_config_path, base_config_content).unwrap();

        let sweep = SweepConfig {
            base_config: base_config_path.to_str().unwrap().to_string(),
            target_binary: "mnist_mlp".to_string(),
            description: None,
            learning_rate: Some(vec![0.001, 0.01]),
            batch_size: Some(vec![32, 64]),
            epochs: Some(vec![5, 10]),
            validation_split: None,
            early_stopping_patience: None,
            early_stopping_min_delta: None,
            scheduler_type: None,
            step_size: None,
            gamma: None,
            decay_rate: None,
            min_lr: None,
            T_max: None,
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        };

        let configs = generate_configs(&sweep).unwrap();

        // Should generate 2 × 2 × 2 = 8 configs
        assert_eq!(configs.len(), 8);

        // Verify all combinations exist
        let has_combination = |lr: f32, bs: usize, ep: usize| -> bool {
            configs.iter().any(|c| {
                c.learning_rate == Some(lr) && c.batch_size == Some(bs) && c.epochs == Some(ep)
            })
        };

        // Check all 8 combinations exist
        assert!(has_combination(0.001, 32, 5));
        assert!(has_combination(0.001, 32, 10));
        assert!(has_combination(0.001, 64, 5));
        assert!(has_combination(0.001, 64, 10));
        assert!(has_combination(0.01, 32, 5));
        assert!(has_combination(0.01, 32, 10));
        assert!(has_combination(0.01, 64, 5));
        assert!(has_combination(0.01, 64, 10));

        // Cleanup
        fs::remove_file(base_config_path).ok();
    }
}
