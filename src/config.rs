//! Configuration structures for training
//!
//! This module provides configuration structures for setting up training parameters,
//! particularly learning rate scheduling configurations.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WarmupConfig {
    /// Linear warmup from `start_lr` to the base learning rate over `epochs` epochs.
    Linear {
        epochs: usize,
        start_lr: Option<f32>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CyclicalLRConfig {
    /// Triangular cyclical learning rate policy.
    /// `step_size` is the number of steps in a half-cycle (up then down).
    Triangular {
        base_lr: f32,
        max_lr: f32,
        step_size: usize,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RegularizationConfig {
    pub l1: Option<f32>,
    pub l2: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GradientClippingConfig {
    /// Clip gradients by global norm.
    Norm { max_norm: f32 },
    /// Clip gradients by absolute value.
    Value { max_value: f32 },
}

pub const SCHEDULER_ALLOWLIST: [&str; 3] = ["step_decay", "exponential", "cosine_annealing"];
pub const ACTIVATION_ALLOWLIST: [&str; 6] = ["relu", "leaky_relu", "elu", "gelu", "swish", "tanh"];

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
/// Optimizer can be specified with optional hyperparameters:
///
/// - **SGD**: No additional parameters (basic stochastic gradient descent)
/// - **Adam**: Optional `adam_beta1` (default 0.9), `adam_beta2` (default 0.999), `adam_epsilon` (default 1e-8)
/// - **AdamW**: Same as Adam, plus optional `adamw_weight_decay` (default 0.01)
/// - **RMSprop**: Optional `rmsprop_decay` (default 0.9), `rmsprop_epsilon` (default 1e-8)
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
///   "optimizer_type": "adamw",
///   "adam_beta1": 0.9,
///   "adam_beta2": 0.999,
///   "adam_epsilon": 1e-8,
///   "adamw_weight_decay": 0.01,
///   "learning_rate": 0.01,
///   "epochs": 10,
///   "batch_size": 64,
///   "validation_split": 0.1,
///   "early_stopping_patience": 3,
///   "early_stopping_min_delta": 0.001
/// }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[allow(non_snake_case)]
pub struct TrainingConfig {
    /// Type of learning rate scheduler: "step_decay", "exponential", or "cosine_annealing"
    ///
    /// Note: This is a required field. We intentionally do not provide a default so that
    /// empty/malformed configs fail fast (see tests/test_config/error_handling_tests.rs).
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

    /// Optimizer type: "sgd", "adam", "adamw", or "rmsprop"
    pub optimizer_type: Option<String>,

    /// Beta1 parameter for Adam/AdamW optimizer (default 0.9)
    pub adam_beta1: Option<f32>,

    /// Beta2 parameter for Adam/AdamW optimizer (default 0.999)
    pub adam_beta2: Option<f32>,

    /// Epsilon parameter for Adam/AdamW optimizer (default 1e-8)
    pub adam_epsilon: Option<f32>,

    /// Weight decay parameter for AdamW optimizer (default 0.01)
    pub adamw_weight_decay: Option<f32>,

    /// Decay rate parameter for RMSprop optimizer (default 0.9)
    pub rmsprop_decay: Option<f32>,

    /// Epsilon parameter for RMSprop optimizer (default 1e-8)
    pub rmsprop_epsilon: Option<f32>,

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

    /// Enable performance profiling during training (optional, default false)
    pub enable_profiling: Option<bool>,

    /// Enable data augmentation during training (optional, default false)
    pub enable_augmentation: Option<bool>,

    /// Probability of applying horizontal flip augmentation (must be in [0.0, 1.0] if specified)
    pub horizontal_flip_prob: Option<f32>,

    /// Padding size for random crop augmentation (pixels to pad on each side)
    pub random_crop_padding: Option<usize>,

    /// Brightness jitter factor for color augmentation (must be non-negative if specified)
    pub brightness_jitter: Option<f32>,

    /// Contrast jitter factor for color augmentation (must be non-negative if specified)
    pub contrast_jitter: Option<f32>,

    /// Saturation jitter factor for color augmentation (must be non-negative if specified)
    pub saturation_jitter: Option<f32>,

    // --- GAN-specific fields ---
    /// Dimension of the latent noise vector fed to the GAN generator (must be positive if specified)
    pub noise_dim: Option<usize>,

    /// Learning rate for the GAN generator (must be positive if specified)
    pub g_lr: Option<f32>,

    /// Learning rate for the GAN discriminator (must be positive if specified)
    pub d_lr: Option<f32>,

    /// Label smoothing factor for GAN training, e.g. 0.9 for one-sided smoothing
    /// (must be in [0.0, 1.0] if specified)
    pub label_smoothing: Option<f32>,

    // --- GPU acceleration fields ---
    /// Optional GPU backend selection: "auto", "metal", "cuda", or "cpu".
    /// When `None`, callers apply their own default (e.g., `unwrap_or("auto")`).
    pub gpu_backend: Option<String>,

    /// Optional GPU device ID for multi-GPU systems.
    /// When `None`, callers should default to device 0.
    pub gpu_device_id: Option<usize>,

    /// Enable interactive step-through debugging during training (optional, default false)
    pub step_debug: Option<bool>,

    // --- Training controls ---
    /// Optional warmup schedule configuration. When omitted, warmup is disabled.
    #[serde(default)]
    pub warmup: Option<WarmupConfig>,

    /// Optional cyclical learning rate configuration. When omitted, cyclical LR is disabled.
    #[serde(default)]
    pub cyclical_lr: Option<CyclicalLRConfig>,

    /// Optional regularization configuration (L1/L2). When omitted, regularization is disabled.
    #[serde(default)]
    pub regularization: Option<RegularizationConfig>,

    /// Optional gradient clipping configuration. When omitted, clipping is disabled.
    #[serde(default)]
    pub gradient_clipping: Option<GradientClippingConfig>,
}

/// Loads a training configuration from a JSON file.
///
/// Attempts to read and deserialize the JSON at `path` into a `TrainingConfig`.
///
/// Returns the deserialized `TrainingConfig` on success. Returns an error if the file cannot be read (including not found) or if the JSON is invalid.
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
    let contents = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Config file not found: {}. Default configs are in config/training/. \
                    Example: cargo run --bin mnist_mlp -- config/training/mnist_mlp_default.json",
                    path
                ),
            )));
        }
        Err(e) => return Err(Box::new(e)),
    };

    let config: TrainingConfig = match serde_json::from_str(&contents) {
        Ok(c) => c,
        Err(e) => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Failed to parse JSON config at {}: {}. \
                    Check for missing quotes, trailing commas, or incorrect field names. \
                    Valid fields: learning_rate, epochs, batch_size, scheduler_type, \
                    activation_function, optimizer_type, and more.",
                    path, e
                ),
            )));
        }
    };

    validate_config(&config)?;
    Ok(config)
}

/// Validates the semantic correctness of a TrainingConfig instance.
///
/// Performs scheduler-specific required-field checks and range/consistency validations
/// for scheduler, activation, optimizer, training, augmentation, GAN, and GPU settings.
///
/// # Errors
///
/// Returns an `Err` describing the first validation failure encountered.
///
/// # Examples
///
/// ```ignore
/// let cfg = TrainingConfig {
///     scheduler_type: "step_decay".to_string(),
///     step_size: Some(5),
///     gamma: Some(0.5),
///     decay_rate: None,
///     min_lr: None,
///     T_max: None,
///     activation_function: None,
///     leaky_relu_alpha: None,
///     elu_alpha: None,
///     optimizer_type: Some("adam".to_string()),
///     adam_beta1: Some(0.9),
///     adam_beta2: Some(0.999),
///     adam_epsilon: Some(1e-8),
///     adamw_weight_decay: None,
///     rmsprop_decay: None,
///     rmsprop_epsilon: None,
///     learning_rate: Some(0.001),
///     epochs: Some(10),
///     batch_size: Some(32),
///     validation_split: Some(0.1),
///     early_stopping_patience: None,
///     early_stopping_min_delta: Some(0.0),
///     enable_profiling: None,
///     enable_augmentation: None,
///     horizontal_flip_prob: None,
///     random_crop_padding: None,
///     brightness_jitter: None,
///     contrast_jitter: None,
///     saturation_jitter: None,
///     noise_dim: None,
///     g_lr: None,
///     d_lr: None,
///     label_smoothing: None,
///     gpu_backend: Some("auto".to_string()),
///     gpu_device_id: None,
///     step_debug: None,
/// };
///
/// assert!(validate_config(&cfg).is_ok());
/// ```
pub fn validate_config(config: &TrainingConfig) -> Result<(), Box<dyn Error>> {
    // Validate scheduler-specific required fields
    match config.scheduler_type.as_str() {
        "step_decay" => {
            if config.step_size.is_none() || config.gamma.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Scheduler '{}' requires both step_size and gamma. \
                        Example: {{ \"step_size\": 5, \"gamma\": 0.5 }}",
                        config.scheduler_type
                    ),
                )));
            }
        }
        "exponential" => {
            if config.decay_rate.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Scheduler '{}' requires decay_rate. \
                        Example: {{ \"decay_rate\": 0.95 }}",
                        config.scheduler_type
                    ),
                )));
            }
        }
        "cosine_annealing" => {
            if config.min_lr.is_none() || config.T_max.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Scheduler '{}' requires both min_lr and T_max. \
                        Example: {{ \"min_lr\": 0.0001, \"T_max\": 10 }}",
                        config.scheduler_type
                    ),
                )));
            }
        }
        _ => {} // Unknown/custom scheduler types have no required fields
    }

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

    // Validate optimizer type
    if let Some(ref optimizer) = config.optimizer_type {
        let valid_optimizers = ["sgd", "adam", "adamw", "rmsprop"];
        if !valid_optimizers.contains(&optimizer.as_str()) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid optimizer type '{}'. Must be one of: {}",
                    optimizer,
                    valid_optimizers.join(", ")
                ),
            )));
        }
    }

    // Validate optimizer hyperparameters
    if let Some(beta1) = config.adam_beta1 {
        if !(0.0..1.0).contains(&beta1) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "adam_beta1 must be in the range [0.0, 1.0)",
            )));
        }
    }

    if let Some(beta2) = config.adam_beta2 {
        if !(0.0..1.0).contains(&beta2) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "adam_beta2 must be in the range [0.0, 1.0)",
            )));
        }
    }

    if let Some(epsilon) = config.adam_epsilon {
        if epsilon <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "adam_epsilon must be positive",
            )));
        }
    }

    if let Some(weight_decay) = config.adamw_weight_decay {
        if weight_decay < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "adamw_weight_decay must be non-negative",
            )));
        }
    }

    if let Some(decay) = config.rmsprop_decay {
        if !(0.0..1.0).contains(&decay) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "rmsprop_decay must be in the range [0.0, 1.0)",
            )));
        }
    }

    if let Some(epsilon) = config.rmsprop_epsilon {
        if epsilon <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "rmsprop_epsilon must be positive",
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
        if !(0.0..=1.0).contains(&validation_split) {
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

    // Validate augmentation parameters
    if let Some(horizontal_flip_prob) = config.horizontal_flip_prob {
        if !(0.0..=1.0).contains(&horizontal_flip_prob) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "horizontal_flip_prob must be in the range [0.0, 1.0]",
            )));
        }
    }

    if let Some(brightness_jitter) = config.brightness_jitter {
        if brightness_jitter < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "brightness_jitter must be non-negative",
            )));
        }
    }

    if let Some(contrast_jitter) = config.contrast_jitter {
        if contrast_jitter < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "contrast_jitter must be non-negative",
            )));
        }
    }

    if let Some(saturation_jitter) = config.saturation_jitter {
        if saturation_jitter < 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "saturation_jitter must be non-negative",
            )));
        }
    }

    // Validate training controls
    if let Some(ref warmup) = config.warmup {
        match warmup {
            WarmupConfig::Linear { epochs, start_lr } => {
                if *epochs == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "warmup.epochs must be positive",
                    )));
                }
                if let Some(start_lr) = *start_lr {
                    if start_lr < 0.0 {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "warmup.start_lr must be non-negative",
                        )));
                    }
                    if let Some(base_lr) = config.learning_rate {
                        if start_lr > base_lr {
                            return Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "warmup.start_lr must be <= learning_rate",
                            )));
                        }
                    }
                }
                if let Some(total_epochs) = config.epochs {
                    if *epochs >= total_epochs {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "warmup.epochs must be < epochs",
                        )));
                    }
                }
            }
        }
    }

    if let Some(ref cyclical) = config.cyclical_lr {
        match cyclical {
            CyclicalLRConfig::Triangular {
                base_lr,
                max_lr,
                step_size,
            } => {
                if *base_lr <= 0.0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "cyclical_lr.base_lr must be positive",
                    )));
                }
                if *max_lr <= 0.0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "cyclical_lr.max_lr must be positive",
                    )));
                }
                if *max_lr <= *base_lr {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "cyclical_lr.max_lr must be > cyclical_lr.base_lr",
                    )));
                }
                if *step_size == 0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "cyclical_lr.step_size must be positive",
                    )));
                }
            }
        }

        // If cyclical LR is enabled, the base learning_rate should not be set,
        // to avoid ambiguity about which one is actually used.
        if config.learning_rate.is_some() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "learning_rate must be omitted when cyclical_lr is enabled (use cyclical_lr.base_lr/max_lr instead)",
            )));
        }

        // Likewise, the epoch-level scheduler_type shouldn't be set to a non-empty value.
        // (It is not optional today; keep allowing unknown scheduler types as 'constant',
        // but disallow mixing with cyclical to keep behavior deterministic.)
        if config.scheduler_type.as_str() != "constant" {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "scheduler_type must be 'constant' (or empty) when cyclical_lr is enabled",
            )));
        }
    }

    if let Some(ref regularization) = config.regularization {
        if let Some(l1) = regularization.l1 {
            if l1 < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "regularization.l1 must be non-negative",
                )));
            }
        }
        if let Some(l2) = regularization.l2 {
            if l2 < 0.0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "regularization.l2 must be non-negative",
                )));
            }
        }

        // Weight-decay style regularization is only implemented via AdamW today.
        // If users request L2 but are not using AdamW, fail early with guidance.
        if regularization.l2.unwrap_or(0.0) > 0.0 {
            let optimizer = config
                .optimizer_type
                .as_deref()
                .unwrap_or("adamw")
                .to_lowercase();
            if optimizer != "adamw" {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "regularization.l2 is currently only supported with optimizer_type='adamw' (uses adamw_weight_decay)",
                )));
            }
        }
    }

    if let Some(ref clipping) = config.gradient_clipping {
        match clipping {
            GradientClippingConfig::Norm { max_norm } => {
                if *max_norm <= 0.0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "gradient_clipping.max_norm must be positive",
                    )));
                }
            }
            GradientClippingConfig::Value { max_value } => {
                if *max_value <= 0.0 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "gradient_clipping.max_value must be positive",
                    )));
                }
            }
        }
    }

    // Validate GAN-specific parameters
    if let Some(noise_dim) = config.noise_dim {
        if noise_dim == 0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "noise_dim must be positive",
            )));
        }
    }

    if let Some(g_lr) = config.g_lr {
        if g_lr <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "g_lr (generator learning rate) must be positive",
            )));
        }
    }

    if let Some(d_lr) = config.d_lr {
        if d_lr <= 0.0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "d_lr (discriminator learning rate) must be positive",
            )));
        }
    }

    if let Some(label_smoothing) = config.label_smoothing {
        if !(0.0..=1.0).contains(&label_smoothing) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "label_smoothing must be in the range [0.0, 1.0]",
            )));
        }
    }

    // Validate GPU configuration
    if let Some(ref backend) = config.gpu_backend {
        let valid_backends = ["auto", "metal", "cuda", "cpu"];
        if !valid_backends.contains(&backend.as_str()) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid gpu_backend '{}'. Must be one of: {}",
                    backend,
                    valid_backends.join(", ")
                ),
            )));
        }
    }

    Ok(())
}
