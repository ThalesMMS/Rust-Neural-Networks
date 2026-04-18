use super::*;
use std::fs;

#[test]
fn test_generate_configs_single_parameter() {
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

    assert_eq!(configs.len(), 3);
    assert_eq!(configs[0].learning_rate, Some(0.001));
    assert_eq!(configs[1].learning_rate, Some(0.01));
    assert_eq!(configs[2].learning_rate, Some(0.1));

    for config in &configs {
        assert_eq!(config.batch_size, Some(64));
        assert_eq!(config.epochs, Some(10));
        assert_eq!(config.scheduler_type, "step_decay");
    }

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_multiple_parameters() {
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

    assert_eq!(configs.len(), 6);

    let has_combination = |lr: f32, bs: usize| -> bool {
        configs
            .iter()
            .any(|c| c.learning_rate == Some(lr) && c.batch_size == Some(bs))
    };

    assert!(has_combination(0.001, 32));
    assert!(has_combination(0.001, 64));
    assert!(has_combination(0.001, 128));
    assert!(has_combination(0.01, 32));
    assert!(has_combination(0.01, 64));
    assert!(has_combination(0.01, 128));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_no_sweep_parameters() {
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

    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0].learning_rate, Some(0.01));
    assert_eq!(configs[0].batch_size, Some(64));
    assert_eq!(configs[0].epochs, Some(10));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_errors_on_unknown_scheduler_type() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_unknown_scheduler.json");
    let base_config_content = r#"{
        "scheduler_type": "custom",
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

    let result = generate_configs(&sweep);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown scheduler_type"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_rejects_incompatible_fixed_scheduler_sweeps() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_fixed_step_decay.json");
    write_base_config(&base_config_path);

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
        decay_rate: Some(vec![0.95]),
        min_lr: None,
        T_max: None,
        activation_function: None,
        leaky_relu_alpha: None,
        elu_alpha: None,
    };

    let result = generate_configs(&sweep);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("decay_rate"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_scheduler_types() {
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
        step_size: Some(vec![3]),
        gamma: Some(vec![0.5]),
        decay_rate: Some(vec![0.95]),
        min_lr: None,
        T_max: None,
        activation_function: None,
        leaky_relu_alpha: None,
        elu_alpha: None,
    };

    let configs = generate_configs(&sweep).unwrap();

    assert_eq!(configs.len(), 2);
    assert_eq!(configs[0].scheduler_type, "step_decay");
    assert_eq!(configs[1].scheduler_type, "exponential");
    assert_eq!(configs[0].step_size, Some(3));
    assert_eq!(configs[0].gamma, Some(0.5));
    assert_eq!(configs[0].decay_rate, None);
    assert_eq!(configs[1].step_size, None);
    assert_eq!(configs[1].gamma, None);
    assert_eq!(configs[1].decay_rate, Some(0.95));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_only_iterates_matching_scheduler_parameters() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_scheduler_specific.json");
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
        scheduler_type: Some(vec![
            "step_decay".to_string(),
            "exponential".to_string(),
            "cosine_annealing".to_string(),
        ]),
        step_size: Some(vec![2, 4]),
        gamma: Some(vec![0.5, 0.9]),
        decay_rate: Some(vec![0.95, 0.99]),
        min_lr: Some(vec![0.0001, 0.0005]),
        T_max: Some(vec![5, 10]),
        activation_function: None,
        leaky_relu_alpha: None,
        elu_alpha: None,
    };

    let configs = generate_configs(&sweep).unwrap();

    assert_eq!(configs.len(), 10);
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.scheduler_type == "step_decay")
            .count(),
        4
    );
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.scheduler_type == "exponential")
            .count(),
        2
    );
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.scheduler_type == "cosine_annealing")
            .count(),
        4
    );
    assert!(configs.iter().any(|c| {
        c.scheduler_type == "step_decay" && c.step_size == Some(2) && c.gamma == Some(0.5)
    }));
    assert!(configs
        .iter()
        .any(|c| c.scheduler_type == "exponential" && c.decay_rate == Some(0.95)));
    assert!(configs.iter().any(|c| {
        c.scheduler_type == "cosine_annealing" && c.min_lr == Some(0.0001) && c.T_max == Some(5)
    }));
    for config in &configs {
        match config.scheduler_type.as_str() {
            "step_decay" => {
                assert!(config.step_size.is_some());
                assert!(config.gamma.is_some());
                assert_eq!(config.decay_rate, None);
                assert_eq!(config.min_lr, None);
                assert_eq!(config.T_max, None);
            }
            "exponential" => {
                assert_eq!(config.step_size, None);
                assert_eq!(config.gamma, None);
                assert!(config.decay_rate.is_some());
                assert_eq!(config.min_lr, None);
                assert_eq!(config.T_max, None);
            }
            "cosine_annealing" => {
                assert_eq!(config.step_size, None);
                assert_eq!(config.gamma, None);
                assert_eq!(config.decay_rate, None);
                assert!(config.min_lr.is_some());
                assert!(config.T_max.is_some());
            }
            other => panic!("unexpected scheduler type: {}", other),
        }
    }

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_rejects_incompatible_fixed_activation_sweeps() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_fixed_relu.json");
    let base_config_content = r#"{
        "scheduler_type": "step_decay",
        "step_size": 3,
        "gamma": 0.5,
        "activation_function": "relu",
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
        leaky_relu_alpha: Some(vec![0.01]),
        elu_alpha: None,
    };

    let result = generate_configs(&sweep);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("leaky_relu_alpha"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_only_iterates_matching_activation_alphas() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_activation_specific.json");
    write_base_config(&base_config_path);

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
        activation_function: Some(vec![
            "relu".to_string(),
            "leaky_relu".to_string(),
            "elu".to_string(),
            "tanh".to_string(),
        ]),
        leaky_relu_alpha: Some(vec![0.01, 0.02]),
        elu_alpha: Some(vec![1.0, 2.0]),
    };

    let configs = generate_configs(&sweep).unwrap();

    assert_eq!(configs.len(), 6);
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.activation_function.as_deref() == Some("leaky_relu"))
            .count(),
        2
    );
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.activation_function.as_deref() == Some("elu"))
            .count(),
        2
    );
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.activation_function.as_deref() == Some("relu"))
            .count(),
        1
    );
    assert_eq!(
        configs
            .iter()
            .filter(|c| c.activation_function.as_deref() == Some("tanh"))
            .count(),
        1
    );
    for config in &configs {
        match config.activation_function.as_deref() {
            Some("leaky_relu") => {
                assert!(config.leaky_relu_alpha.is_some());
                assert_eq!(config.elu_alpha, None);
            }
            Some("elu") => {
                assert_eq!(config.leaky_relu_alpha, None);
                assert!(config.elu_alpha.is_some());
            }
            Some("relu") | Some("tanh") => {
                assert_eq!(config.leaky_relu_alpha, None);
                assert_eq!(config.elu_alpha, None);
            }
            other => panic!("unexpected activation function: {:?}", other),
        }
    }

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_errors_when_swept_step_decay_missing_values() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_missing_step_decay.json");
    write_base_config(&base_config_path);

    let mut sweep = scheduler_sweep(base_config_path.to_str().unwrap(), "step_decay");
    sweep.step_size = Some(vec![2]);

    let result = generate_configs(&sweep);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("gamma"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_errors_when_swept_exponential_missing_values() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_missing_exponential.json");
    write_base_config(&base_config_path);

    let sweep = scheduler_sweep(base_config_path.to_str().unwrap(), "exponential");

    let result = generate_configs(&sweep);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("decay_rate"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_errors_when_swept_cosine_missing_values() {
    let temp_dir = std::env::temp_dir();
    let base_config_path = temp_dir.join("test_base_config_missing_cosine.json");
    write_base_config(&base_config_path);

    let mut sweep = scheduler_sweep(base_config_path.to_str().unwrap(), "cosine_annealing");
    sweep.min_lr = Some(vec![0.0001]);

    let result = generate_configs(&sweep);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("T_max"));

    fs::remove_file(base_config_path).ok();
}

#[test]
fn test_generate_configs_three_way_cartesian() {
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

    assert_eq!(configs.len(), 8);

    let has_combination = |lr: f32, bs: usize, ep: usize| -> bool {
        configs.iter().any(|c| {
            c.learning_rate == Some(lr) && c.batch_size == Some(bs) && c.epochs == Some(ep)
        })
    };

    assert!(has_combination(0.001, 32, 5));
    assert!(has_combination(0.001, 32, 10));
    assert!(has_combination(0.001, 64, 5));
    assert!(has_combination(0.001, 64, 10));
    assert!(has_combination(0.01, 32, 5));
    assert!(has_combination(0.01, 32, 10));
    assert!(has_combination(0.01, 64, 5));
    assert!(has_combination(0.01, 64, 10));

    fs::remove_file(base_config_path).ok();
}

fn write_base_config(path: &std::path::Path) {
    let base_config_content = r#"{
        "scheduler_type": "step_decay",
        "step_size": 3,
        "gamma": 0.5,
        "learning_rate": 0.01,
        "epochs": 10,
        "batch_size": 64
    }"#;
    fs::write(path, base_config_content).unwrap();
}

fn scheduler_sweep(base_config: &str, scheduler_type: &str) -> SweepConfig {
    SweepConfig {
        base_config: base_config.to_string(),
        target_binary: "mnist_mlp".to_string(),
        description: None,
        learning_rate: None,
        batch_size: None,
        epochs: None,
        validation_split: None,
        early_stopping_patience: None,
        early_stopping_min_delta: None,
        scheduler_type: Some(vec![scheduler_type.to_string()]),
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
