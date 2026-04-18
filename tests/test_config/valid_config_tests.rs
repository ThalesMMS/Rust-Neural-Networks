use super::*;

#[test]
fn test_load_step_decay_config() {
    let config =
        load_config("config/mnist_mlp_step.json").expect("Failed to load step decay config");

    assert_eq!(config.scheduler_type, "step_decay");
    assert_eq!(config.step_size, Some(3));
    assert_eq!(config.gamma, Some(0.5));
    assert_eq!(config.decay_rate, None);
    assert_eq!(config.min_lr, None);
    assert_eq!(config.T_max, None);
}

#[test]
fn test_load_exponential_config() {
    let config = load_config("config/mnist_mlp_exponential.json")
        .expect("Failed to load exponential config");

    assert_eq!(config.scheduler_type, "exponential");
    assert_eq!(config.decay_rate, Some(0.95));
    assert_eq!(config.step_size, None);
    assert_eq!(config.gamma, None);
    assert_eq!(config.min_lr, None);
    assert_eq!(config.T_max, None);
}

/// Verifies that a cosine annealing scheduler configuration is loaded correctly from the example JSON.
///
/// Asserts that `scheduler_type` equals `"cosine_annealing"`, `min_lr` and `T_max` are present with the expected values,
/// and that `step_size`, `gamma`, and `decay_rate` are `None`.
///
/// # Examples
///
/// ```
/// let config = rust_neural_networks::config::load_config("config/mnist_mlp_cosine.json").unwrap();
/// assert_eq!(config.scheduler_type, "cosine_annealing");
/// assert_eq!(config.min_lr, Some(0.0001));
/// assert_eq!(config.T_max, Some(10));
/// ```
#[test]
fn test_load_cosine_annealing_config() {
    let config = load_config("config/mnist_mlp_cosine.json")
        .expect("Failed to load cosine annealing config");

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.min_lr, Some(0.0001));
    assert_eq!(config.T_max, Some(10));
    assert_eq!(config.step_size, None);
    assert_eq!(config.gamma, None);
    assert_eq!(config.decay_rate, None);
}

#[test]
fn test_config_values_step_decay() {
    let config = load_config("config/mnist_mlp_step.json").unwrap();

    // Verify specific values
    assert_eq!(config.step_size.unwrap(), 3);
    assert!((config.gamma.unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn test_config_values_exponential() {
    let config = load_config("config/mnist_mlp_exponential.json").unwrap();

    // Verify specific values
    assert!((config.decay_rate.unwrap() - 0.95).abs() < 1e-6);
}

#[test]
fn test_config_values_cosine() {
    let config = load_config("config/mnist_mlp_cosine.json").unwrap();

    // Verify specific values
    assert!((config.min_lr.unwrap() - 0.0001).abs() < 1e-6);
    assert_eq!(config.T_max.unwrap(), 10);
}

#[test]
fn test_load_activations_demo_config() {
    let config = load_config("config/activations_demo.json")
        .expect("Failed to load activations demo config");

    assert_eq!(config.scheduler_type, "step_decay");
    assert_eq!(config.step_size, Some(3));
    assert_eq!(config.gamma, Some(0.5));
    assert_eq!(config.activation_function, Some("leaky_relu".to_string()));
    assert_eq!(config.leaky_relu_alpha, Some(0.01));
    assert_eq!(config.elu_alpha, None);
}
