use super::*;

#[test]
fn test_warmup_epochs_must_be_positive() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "warmup": { "type": "linear", "epochs": 0 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when warmup.epochs is 0");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("warmup.epochs"),
        "Error should mention warmup.epochs but got: {}",
        err_msg
    );
}

#[test]
fn test_warmup_start_lr_must_be_non_negative() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "learning_rate": 0.01,
  "warmup": { "type": "linear", "epochs": 2, "start_lr": -0.0001 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when warmup.start_lr is negative"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("warmup.start_lr"),
        "Error should mention warmup.start_lr but got: {}",
        err_msg
    );
}

#[test]
fn test_warmup_start_lr_must_not_exceed_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "learning_rate": 0.001,
  "warmup": { "type": "linear", "epochs": 2, "start_lr": 0.01 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when warmup.start_lr > learning_rate"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("warmup.start_lr") || err_msg.contains("learning_rate"),
        "Error should mention warmup.start_lr/learning_rate but got: {}",
        err_msg
    );
}

#[test]
fn test_warmup_epochs_must_be_less_than_total_epochs_when_epochs_provided() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "epochs": 5,
  "warmup": { "type": "linear", "epochs": 5 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when warmup.epochs >= epochs");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("warmup.epochs"),
        "Error should mention warmup.epochs but got: {}",
        err_msg
    );
}

#[test]
fn test_cyclical_lr_disallows_learning_rate() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "learning_rate": 0.001,
  "cyclical_lr": {
    "type": "triangular",
    "base_lr": 0.0001,
    "max_lr": 0.001,
    "step_size": 200
  }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when learning_rate is set alongside cyclical_lr"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("learning_rate") && err_msg.contains("cyclical_lr"),
        "Error should mention learning_rate and cyclical_lr but got: {}",
        err_msg
    );
}

#[test]
fn test_cyclical_lr_requires_constant_scheduler_type() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 1,
  "gamma": 0.5,
  "cyclical_lr": {
    "type": "triangular",
    "base_lr": 0.0001,
    "max_lr": 0.001,
    "step_size": 200
  }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when cyclical_lr is mixed with non-constant scheduler_type"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("scheduler_type") && err_msg.contains("cyclical_lr"),
        "Error should mention scheduler_type and cyclical_lr but got: {}",
        err_msg
    );
}

#[test]
fn test_cyclical_lr_base_lr_must_be_positive() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "cyclical_lr": {
    "type": "triangular",
    "base_lr": 0.0,
    "max_lr": 0.001,
    "step_size": 200
  }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when cyclical_lr.base_lr is 0");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cyclical_lr.base_lr"),
        "Error should mention cyclical_lr.base_lr but got: {}",
        err_msg
    );
}

#[test]
fn test_cyclical_lr_step_size_must_be_positive() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "cyclical_lr": {
    "type": "triangular",
    "base_lr": 0.0001,
    "max_lr": 0.001,
    "step_size": 0
  }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when cyclical_lr.step_size is 0"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cyclical_lr.step_size"),
        "Error should mention cyclical_lr.step_size but got: {}",
        err_msg
    );
}

#[test]
fn test_regularization_l2_requires_adamw() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "optimizer_type": "adam",
  "regularization": { "l2": 0.0001 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when regularization.l2 is requested without adamw"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("regularization.l2") && err_msg.contains("adamw"),
        "Error should mention regularization.l2 and adamw but got: {}",
        err_msg
    );
}

#[test]
fn test_gradient_clipping_norm_requires_positive_max_norm() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "gradient_clipping": { "type": "norm", "max_norm": 0.0 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when gradient_clipping.max_norm is 0"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("gradient_clipping.max_norm"),
        "Error should mention gradient_clipping.max_norm but got: {}",
        err_msg
    );
}

#[test]
fn test_gradient_clipping_value_requires_positive_max_value() {
    let config_json = r#"{
  "scheduler_type": "constant",
  "gradient_clipping": { "type": "value", "max_value": -1.0 }
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when gradient_clipping.max_value is negative"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("gradient_clipping.max_value"),
        "Error should mention gradient_clipping.max_value but got: {}",
        err_msg
    );
}
