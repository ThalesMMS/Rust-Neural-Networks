use crate::config::TrainingConfig;

/// Extracts the configuration file path from command-line arguments.
///
/// If `--config` is present and followed by an argument that does not start with `-`,
/// that following argument is returned. If `--config` is absent, at the end of the
/// arguments, or followed by a dash-prefixed token, `default_path` is returned.
///
/// # Returns
///
/// The selected configuration path: the value after `--config` when valid, otherwise `default_path`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::parse_config_path;
///
/// let args = vec!["binary".to_string(), "--config".to_string(), "my.json".to_string()];
/// assert_eq!(parse_config_path(&args, "default.json"), "my.json");
///
/// let args_empty: Vec<String> = vec!["binary".to_string()];
/// assert_eq!(parse_config_path(&args_empty, "default.json"), "default.json");
/// ```
pub fn parse_config_path(args: &[String], default_path: &str) -> String {
    let mut config = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                config = Some(args[i + 1].clone());
            }
        }
        i += 1;
    }
    config.unwrap_or_else(|| default_path.to_string())
}

/// Checks whether the `--step` flag is present in command-line arguments.
///
/// Returns `true` if any argument is exactly `"--step"`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::parse_step_flag;
///
/// let args = vec!["binary".to_string(), "--step".to_string()];
/// assert!(parse_step_flag(&args));
///
/// let args_empty: Vec<String> = vec!["binary".to_string()];
/// assert!(!parse_step_flag(&args_empty));
/// ```
pub fn parse_step_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--step")
}

/// Extracts an optional `--run-name` value from command-line arguments.
///
/// Accepts `--run-name <name>`. Returns `None` if not present or if the value is missing/invalid.
pub fn parse_run_name(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--run-name" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                return Some(args[i + 1].clone());
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Extracts an optional `--registry-dir` value from command-line arguments.
///
/// Accepts `--registry-dir <path>`. Returns `None` if not present or if the value is missing/invalid.
pub fn parse_registry_dir(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--registry-dir" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                return Some(args[i + 1].clone());
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Extracts an optional `--seed` value from command-line arguments.
///
/// Accepts `--seed <u64>`. Returns `None` if not present or if parsing fails.
pub fn parse_seed_override(args: &[String]) -> Option<u64> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--seed" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                return args[i + 1].parse::<u64>().ok();
            }
            return None;
        }
        i += 1;
    }
    None
}

/// Prints a human-readable training configuration to stdout.
///
/// Displays resolved hyperparameters (learning rate, epochs, batch size, validation split),
/// early stopping settings, scheduler type, optional activation function, and data augmentation
/// settings. Augmentation-specific fields are listed only when augmentation is enabled.
///
/// # Examples
///
/// ```ignore
/// let config = TrainingConfig { /* fields populated or loaded from file */ };
/// print_training_config(&config, 0.001, 50, 32, 0.1, 5, 0.0001);
/// ```
pub fn print_training_config(
    config: &TrainingConfig,
    learning_rate: f32,
    epochs: usize,
    batch_size: usize,
    validation_split: f32,
    early_stopping_patience: usize,
    early_stopping_min_delta: f32,
) {
    let enable_augmentation = config.enable_augmentation.unwrap_or(false);

    println!("\nConfiguration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Validation split: {:.1}%", validation_split * 100.0);
    println!("  Early stopping patience: {}", early_stopping_patience);
    println!("  Early stopping min delta: {}", early_stopping_min_delta);
    println!("  Scheduler type: {}", config.scheduler_type);

    if let Some(ref warmup) = config.warmup {
        println!("  Warmup: {:?}", warmup);
    } else {
        println!("  Warmup: disabled");
    }

    if let Some(ref cyclical) = config.cyclical_lr {
        println!("  Cyclical LR: {:?}", cyclical);
    } else {
        println!("  Cyclical LR: disabled");
    }

    if let Some(ref reg) = config.regularization {
        let l1 = reg.l1.unwrap_or(0.0);
        let l2 = reg.l2.unwrap_or(0.0);
        if l1 != 0.0 || l2 != 0.0 {
            println!("  Regularization: l1={}, l2={}", l1, l2);
        } else {
            println!("  Regularization: disabled");
        }
    } else {
        println!("  Regularization: disabled");
    }

    if let Some(ref clipping) = config.gradient_clipping {
        println!("  Gradient clipping: {:?}", clipping);
    } else {
        println!("  Gradient clipping: disabled");
    }

    if let Some(ref activation) = config.activation_function {
        println!("  Activation function: {}", activation);
    }
    println!(
        "  Data augmentation: {}",
        if enable_augmentation {
            "enabled"
        } else {
            "disabled"
        }
    );
    if enable_augmentation {
        if let Some(prob) = config.horizontal_flip_prob {
            println!("    Horizontal flip probability: {}", prob);
        }
        if let Some(padding) = config.random_crop_padding {
            println!("    Random crop padding: {}", padding);
        }
        if let Some(delta) = config.brightness_jitter {
            println!("    Brightness jitter: {}", delta);
        }
        if let Some(delta) = config.contrast_jitter {
            println!("    Contrast jitter: {}", delta);
        }
        if let Some(delta) = config.saturation_jitter {
            println!("    Saturation jitter: {}", delta);
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config_path_default() {
        let args: Vec<String> = vec!["binary".to_string()];
        assert_eq!(parse_config_path(&args, "default.json"), "default.json");
    }

    #[test]
    fn test_parse_config_path_flag() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "custom.json".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "custom.json");
    }

    #[test]
    fn test_parse_config_path_extra_args() {
        let args = vec![
            "binary".to_string(),
            "--verbose".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
            "--other".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "my.json");
    }

    #[test]
    fn test_parse_config_path_flag_at_end_ignored() {
        let args = vec!["binary".to_string(), "--config".to_string()];
        assert_eq!(parse_config_path(&args, "default.json"), "default.json");
    }

    #[test]
    fn test_parse_config_path_flag_value_ignored() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "--step".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "default.json");
    }

    #[test]
    fn test_parse_config_path_uses_later_valid_config_after_malformed_one() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "--step".to_string(),
            "--config".to_string(),
            "custom.json".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "custom.json");
    }

    #[test]
    fn test_parse_config_path_later_valid_config_overrides_earlier_one() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "first.json".to_string(),
            "--config".to_string(),
            "second.json".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "second.json");
    }

    #[test]
    fn test_parse_step_flag_present() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
            "--step".to_string(),
        ];
        assert!(parse_step_flag(&args));
    }

    #[test]
    fn test_parse_step_flag_absent() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
        ];
        assert!(!parse_step_flag(&args));
    }
}
