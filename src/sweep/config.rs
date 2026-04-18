use serde::{Deserialize, Serialize};

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
#[serde(deny_unknown_fields)]
#[allow(non_snake_case)]
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
#[allow(non_snake_case)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_config_rejects_unknown_field() {
        let json = r#"{
            "base_config": "config/training/mnist_mlp_default.json",
            "target_binary": "mnist_mlp",
            "lerning_rate": [0.01]
        }"#;

        let result = serde_json::from_str::<SweepConfig>(json);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lerning_rate"));
    }
}
