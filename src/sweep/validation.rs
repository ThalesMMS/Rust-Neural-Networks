use super::config::SweepConfig;
use crate::config::{ACTIVATION_ALLOWLIST, SCHEDULER_ALLOWLIST};
use std::error::Error;

fn invalid_data(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

/// Validate a sweep configuration's fields and value ranges.
///
/// Ensures required string fields are non-empty and that each present numeric or
/// enum vector is non-empty and its elements satisfy the parameter-specific
/// constraints (e.g., learning rates > 0, batch/epoch/step/T_max values > 0,
/// validation_split in [0.0, 1.0], non-negative decay/gamma/min_lr, allowed
/// scheduler and activation names, etc.). Returns the first encountered
/// descriptive `InvalidData` error when a check fails.
///
/// # Examples
///
/// ```ignore
/// let cfg = valid_config(); // returns a SweepConfig with required fields set
/// assert!(validate_sweep_config(&cfg).is_ok());
/// ```
pub(super) fn validate_sweep_config(config: &SweepConfig) -> Result<(), Box<dyn Error>> {
    if config.base_config.is_empty() {
        return Err(invalid_data("base_config path cannot be empty"));
    }

    if config.target_binary.is_empty() {
        return Err(invalid_data("target_binary cannot be empty"));
    }

    if let Some(ref learning_rates) = config.learning_rate {
        if learning_rates.is_empty() {
            return Err(invalid_data("learning_rate array cannot be empty"));
        }
        for &lr in learning_rates {
            if lr <= 0.0 {
                return Err(invalid_data(format!(
                    "learning_rate must be positive, found: {}",
                    lr
                )));
            }
        }
    }

    if let Some(ref batch_sizes) = config.batch_size {
        if batch_sizes.is_empty() {
            return Err(invalid_data("batch_size array cannot be empty"));
        }
        for &bs in batch_sizes {
            if bs == 0 {
                return Err(invalid_data(format!(
                    "batch_size must be positive, found: {}",
                    bs
                )));
            }
        }
    }

    if let Some(ref epochs) = config.epochs {
        if epochs.is_empty() {
            return Err(invalid_data("epochs array cannot be empty"));
        }
        for &e in epochs {
            if e == 0 {
                return Err(invalid_data(format!(
                    "epochs must be positive, found: {}",
                    e
                )));
            }
        }
    }

    if let Some(ref splits) = config.validation_split {
        if splits.is_empty() {
            return Err(invalid_data("validation_split array cannot be empty"));
        }
        for &split in splits {
            if !(0.0..=1.0).contains(&split) {
                return Err(invalid_data(format!(
                    "validation_split must be in [0.0, 1.0], found: {}",
                    split
                )));
            }
        }
    }

    if let Some(ref patiences) = config.early_stopping_patience {
        if patiences.is_empty() {
            return Err(invalid_data(
                "early_stopping_patience array cannot be empty",
            ));
        }
        for &patience in patiences {
            if patience == 0 {
                return Err(invalid_data(format!(
                    "early_stopping_patience must be positive, found: {}",
                    patience
                )));
            }
        }
    }

    if let Some(ref deltas) = config.early_stopping_min_delta {
        if deltas.is_empty() {
            return Err(invalid_data(
                "early_stopping_min_delta array cannot be empty",
            ));
        }
        for &delta in deltas {
            if delta < 0.0 {
                return Err(invalid_data(format!(
                    "early_stopping_min_delta must be non-negative, found: {}",
                    delta
                )));
            }
        }
    }

    if let Some(ref gammas) = config.gamma {
        if gammas.is_empty() {
            return Err(invalid_data("gamma array cannot be empty"));
        }
        for &g in gammas {
            if g < 0.0 {
                return Err(invalid_data(format!(
                    "gamma must be non-negative, found: {}",
                    g
                )));
            }
        }
    }

    if let Some(ref step_sizes) = config.step_size {
        if step_sizes.is_empty() {
            return Err(invalid_data("step_size array cannot be empty"));
        }
        for &step_size in step_sizes {
            if step_size == 0 {
                return Err(invalid_data(format!(
                    "step_size must be positive, found: {}",
                    step_size
                )));
            }
        }
    }

    if let Some(ref decay_rates) = config.decay_rate {
        if decay_rates.is_empty() {
            return Err(invalid_data("decay_rate array cannot be empty"));
        }
        for &dr in decay_rates {
            if dr < 0.0 {
                return Err(invalid_data(format!(
                    "decay_rate must be non-negative, found: {}",
                    dr
                )));
            }
        }
    }

    if let Some(ref min_lrs) = config.min_lr {
        if min_lrs.is_empty() {
            return Err(invalid_data("min_lr array cannot be empty"));
        }
        for &lr in min_lrs {
            if lr < 0.0 {
                return Err(invalid_data(format!(
                    "min_lr must be non-negative, found: {}",
                    lr
                )));
            }
        }
    }

    if let Some(ref t_maxes) = config.T_max {
        if t_maxes.is_empty() {
            return Err(invalid_data("T_max array cannot be empty"));
        }
        for &t_max in t_maxes {
            if t_max == 0 {
                return Err(invalid_data(format!(
                    "T_max must be positive, found: {}",
                    t_max
                )));
            }
        }
    }

    if let Some(ref schedulers) = config.scheduler_type {
        if schedulers.is_empty() {
            return Err(invalid_data("scheduler_type array cannot be empty"));
        }
        for scheduler in schedulers {
            if !SCHEDULER_ALLOWLIST.contains(&scheduler.as_str()) {
                return Err(invalid_data(format!(
                    "Invalid scheduler_type '{}'. Must be one of: {}",
                    scheduler,
                    SCHEDULER_ALLOWLIST.join(", ")
                )));
            }
        }
    }

    if let Some(ref activations) = config.activation_function {
        if activations.is_empty() {
            return Err(invalid_data("activation_function array cannot be empty"));
        }
        for activation in activations {
            if !ACTIVATION_ALLOWLIST.contains(&activation.as_str()) {
                return Err(invalid_data(format!(
                    "Invalid activation_function '{}'. Must be one of: {}",
                    activation,
                    ACTIVATION_ALLOWLIST.join(", ")
                )));
            }
        }
    }

    if let Some(ref alphas) = config.leaky_relu_alpha {
        if alphas.is_empty() {
            return Err(invalid_data("leaky_relu_alpha array cannot be empty"));
        }
        for &alpha in alphas {
            if alpha < 0.0 {
                return Err(invalid_data(format!(
                    "leaky_relu_alpha must be non-negative, found: {}",
                    alpha
                )));
            }
        }
    }

    if let Some(ref alphas) = config.elu_alpha {
        if alphas.is_empty() {
            return Err(invalid_data("elu_alpha array cannot be empty"));
        }
        for &alpha in alphas {
            if alpha <= 0.0 {
                return Err(invalid_data(format!(
                    "elu_alpha must be positive, found: {}",
                    alpha
                )));
            }
        }
    }

    Ok(())
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
    fn test_validate_empty_early_stopping_patience() {
        let mut config = valid_config();
        config.early_stopping_patience = Some(vec![]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("early_stopping_patience"));
    }

    #[test]
    fn test_validate_zero_early_stopping_patience() {
        let mut config = valid_config();
        config.early_stopping_patience = Some(vec![3, 0]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("early_stopping_patience"));
    }

    #[test]
    fn test_validate_empty_early_stopping_min_delta() {
        let mut config = valid_config();
        config.early_stopping_min_delta = Some(vec![]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("early_stopping_min_delta"));
    }

    #[test]
    fn test_validate_negative_early_stopping_min_delta() {
        let mut config = valid_config();
        config.early_stopping_min_delta = Some(vec![0.0, -0.1]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("early_stopping_min_delta"));
    }

    #[test]
    fn test_validate_empty_step_size() {
        let mut config = valid_config();
        config.step_size = Some(vec![]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("step_size"));
    }

    #[test]
    fn test_validate_zero_step_size() {
        let mut config = valid_config();
        config.step_size = Some(vec![1, 0]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("step_size"));
    }

    #[test]
    fn test_validate_empty_t_max() {
        let mut config = valid_config();
        config.T_max = Some(vec![]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("T_max"));
    }

    #[test]
    fn test_validate_zero_t_max() {
        let mut config = valid_config();
        config.T_max = Some(vec![10, 0]);

        let result = validate_sweep_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("T_max"));
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

    /// Create a baseline `SweepConfig` with required fields populated and all optional fields set to `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cfg = valid_config();
    /// assert_eq!(cfg.base_config, "config/mnist_mlp_default.json");
    /// assert_eq!(cfg.target_binary, "mnist_mlp");
    /// assert!(cfg.learning_rate.is_none());
    /// assert!(cfg.batch_size.is_none());
    /// ```
    fn valid_config() -> SweepConfig {
        SweepConfig {
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
            activation_function: None,
            leaky_relu_alpha: None,
            elu_alpha: None,
        }
    }
}
