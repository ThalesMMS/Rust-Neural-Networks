use super::*;

/// Creates an LR scheduler configured from the given learning rate, epoch count, and optional config path.

///

/// The returned scheduler is configured according to the provided parameters and any settings present in the

/// configuration file if `config_path` is `Some`.

///

/// # Examples

///

/// ```

/// let lr = 0.001f32;

/// let epochs = 10usize;

/// let scheduler = scheduler_from_args(lr, epochs, None);

/// // `scheduler` implements the `LRScheduler` trait and can be used where that trait is expected.

/// ```
pub(crate) fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

/// Constructs an optimizer from the provided training configuration and learning rate.
///
/// Supported optimizer types (from `config.optimizer_type`): `"sgd"`, `"adam"`, `"adamw"`, and `"rmsprop"`.
/// Missing hyperparameters use sensible defaults (e.g., Adam beta1=0.9, beta2=0.999, epsilon=1e-8, AdamW weight_decay=0.01).
/// Unknown optimizer type prints a warning to stderr and falls back to `AdamW`.
///
/// # Returns
///
/// A boxed optimizer implementing the `Optimizer` trait corresponding to the selected type.
///
/// # Examples
///
/// ```
/// let config = TrainingConfig::default();
/// let opt = create_optimizer(&config, 1e-3);
/// // `opt` is a boxed `Optimizer` ready to be used for training
/// ```
pub(crate) fn create_optimizer(config: &TrainingConfig, lr: f32) -> Box<dyn Optimizer> {
    let optimizer_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    let beta1 = config.adam_beta1.unwrap_or(0.9);
    let beta2 = config.adam_beta2.unwrap_or(0.999);
    let epsilon = config.adam_epsilon.unwrap_or(1e-8);
    let weight_decay = config.adamw_weight_decay.unwrap_or(0.01);
    let rmsprop_decay = config.rmsprop_decay.unwrap_or(0.9);
    let rmsprop_epsilon = config.rmsprop_epsilon.unwrap_or(epsilon);

    match optimizer_type {
        "sgd" => Box::new(SGD::new(lr)),
        "adam" => Box::new(Adam::new(lr, beta1, beta2, epsilon)),
        "adamw" => Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay)),
        "rmsprop" => Box::new(RMSprop::new(lr, rmsprop_decay, rmsprop_epsilon)),
        _ => {
            eprintln!(
                "Unknown optimizer type: '{}', defaulting to AdamW",
                optimizer_type
            );
            Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay))
        }
    }
}

/// Parse command-line arguments and extract an optional architecture configuration file path.
///
/// Scans `args` for `--arch <path>` and returns `Some(path)` when present; returns `None` if the flag is absent.
/// If `--help` or `-h` is encountered, a usage message including default config and architecture paths is printed and the process exits.
///
/// # Examples
///
/// ```
/// let args = vec![
///     "prog".to_string(),
///     "--arch".to_string(),
///     "arch.yaml".to_string(),
/// ];
/// assert_eq!(parse_arch_path(&args), Some("arch.yaml".to_string()));
/// ```
pub(crate) fn parse_arch_path(args: &[String]) -> Option<String> {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: {} [OPTIONS]", args[0]);
            println!("\nOptions:");
            println!("  --config <path>   Path to training configuration file");
            println!("                    (default: {})", DEFAULT_CONFIG_PATH);
            println!("  --arch <path>     Path to architecture configuration file");
            println!(
                "                    (default: {})",
                DEFAULT_ARCHITECTURE_PATH
            );
            println!("  --help, -h        Show this help message");
            process::exit(0);
        } else if args[i] == "--arch" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

/// Parses command-line arguments and returns the resolved config path and optional architecture path.
///
/// The returned tuple is (config_path, arch_path). `config_path` is taken from `--config <path>` when provided;
/// otherwise `DEFAULT_CONFIG_PATH` is used. `arch_path` is `Some(path)` when `--arch <path>` is present, or `None` if absent.
///
/// # Examples
///
/// ```
/// let args = vec!["prog".to_string(), "--config".to_string(), "c.json".to_string()];
/// let (config, arch) = parse_config_paths(&args);
/// assert_eq!(config, "c.json");
/// assert_eq!(arch, None);
/// ```
pub(crate) fn parse_config_paths(args: &[String]) -> (String, Option<String>) {
    (
        parse_config_path(args, DEFAULT_CONFIG_PATH),
        parse_arch_path(args),
    )
}
