use super::*;

#[test]
fn test_load_cifar10_vit_minimal() {
    let config = load_config("config/cifar10_vit_minimal.json")
        .expect("Failed to load cifar10_vit_minimal config");

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.min_lr, Some(0.0001));
    assert_eq!(config.T_max, Some(1));
    assert_eq!(config.activation_function, Some("relu".to_string()));
    assert_eq!(config.optimizer_type, Some("adam".to_string()));
    assert_eq!(config.learning_rate, Some(0.001));
    assert_eq!(config.epochs, Some(1));
    assert_eq!(config.batch_size, Some(1));
    assert_eq!(config.validation_split, Some(0.999));
    assert_eq!(config.early_stopping_patience, Some(3));
    assert_eq!(config.early_stopping_min_delta, Some(0.001));
}

#[test]
fn test_load_cifar10_vit_quick_test() {
    let config = load_config("config/cifar10_vit_quick_test.json")
        .expect("Failed to load cifar10_vit_quick_test config");

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.batch_size, Some(64));
    assert_eq!(config.validation_split, Some(0.9));
}

#[test]
fn test_load_cifar10_vit_smoke_test() {
    let config = load_config("config/cifar10_vit_smoke_test.json")
        .expect("Failed to load cifar10_vit_smoke_test config");

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.batch_size, Some(5000));
    assert_eq!(config.validation_split, Some(0.1));
}

#[test]
fn test_load_cifar10_vit_test() {
    let config = load_config("config/cifar10_vit_test.json")
        .expect("Failed to load cifar10_vit_test config");

    assert_eq!(config.scheduler_type, "cosine_annealing");
    assert_eq!(config.batch_size, Some(512));
    assert_eq!(config.validation_split, Some(0.1));
}

#[test]
fn test_cifar10_vit_configs_share_common_fields() {
    let configs = vec![
        load_config("config/cifar10_vit_minimal.json").unwrap(),
        load_config("config/cifar10_vit_quick_test.json").unwrap(),
        load_config("config/cifar10_vit_smoke_test.json").unwrap(),
        load_config("config/cifar10_vit_test.json").unwrap(),
    ];

    for config in configs {
        assert_eq!(config.scheduler_type, "cosine_annealing");
        assert_eq!(config.min_lr, Some(0.0001));
        assert_eq!(config.T_max, Some(1));
        assert_eq!(config.activation_function, Some("relu".to_string()));
        assert_eq!(config.optimizer_type, Some("adam".to_string()));
        assert_eq!(config.learning_rate, Some(0.001));
        assert_eq!(config.epochs, Some(1));
    }
}
