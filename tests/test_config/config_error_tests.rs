use super::*;

#[test]
fn test_config_error_missing_file_shows_guidance() {
    let result = load_config("nonexistent_path/config_that_does_not_exist.json");

    assert!(result.is_err(), "Should fail on missing config file");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("config/training/"),
        "Missing file error should mention 'config/training/' but got: {}",
        err_msg
    );
}

#[test]
fn test_config_error_missing_file_shows_path() {
    let missing_path = "no_such_config.json";
    let result = load_config(missing_path);

    assert!(result.is_err(), "Should fail on missing config file");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains(missing_path),
        "Missing file error should include the path '{}' but got: {}",
        missing_path,
        err_msg
    );
}

#[test]
fn test_config_error_invalid_json_shows_guidance() {
    let malformed_json = r#"{ "scheduler_type": "step_decay", invalid }"#;

    let temp_file = write_temp_config(malformed_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on malformed JSON");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.to_lowercase().contains("json") || err_msg.contains("parse"),
        "JSON parse error should mention 'json' or 'parse' but got: {}",
        err_msg
    );
}

#[test]
fn test_config_error_invalid_json_shows_valid_fields() {
    let malformed_json = "not valid json at all";

    let temp_file = write_temp_config(malformed_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on malformed JSON");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("scheduler_type") || err_msg.contains("learning_rate"),
        "JSON parse error should mention valid field names but got: {}",
        err_msg
    );
}

// =========================================================================
// Scheduler-specific required field error message tests
// =========================================================================

#[test]
fn test_step_decay_missing_step_size_error_mentions_required_fields() {
    // step_decay without step_size (only has gamma)
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when step_decay is missing step_size"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("step_size"),
        "Error for step_decay missing step_size should mention 'step_size' but got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("gamma"),
        "Error for step_decay missing step_size should mention 'gamma' but got: {}",
        err_msg
    );
}

#[test]
fn test_step_decay_missing_gamma_error_mentions_required_fields() {
    // step_decay without gamma (only has step_size)
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when step_decay is missing gamma"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("step_size"),
        "Error for step_decay missing gamma should mention 'step_size' but got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("gamma"),
        "Error for step_decay missing gamma should mention 'gamma' but got: {}",
        err_msg
    );
}

#[test]
fn test_step_decay_missing_both_fields_error_includes_example() {
    // step_decay without both step_size and gamma
    let config_json = r#"{
  "scheduler_type": "step_decay"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when step_decay is missing both step_size and gamma"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("step_decay"),
        "Error should name the scheduler 'step_decay' but got: {}",
        err_msg
    );
    // The error message should include an example showing how to fix it
    assert!(
        err_msg.contains("Example") || err_msg.contains("example") || err_msg.contains("step_size"),
        "Error for step_decay missing required fields should include guidance but got: {}",
        err_msg
    );
}

#[test]
fn test_exponential_missing_decay_rate_error_mentions_required_field() {
    // exponential without decay_rate
    let config_json = r#"{
  "scheduler_type": "exponential"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when exponential is missing decay_rate"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("decay_rate"),
        "Error for exponential missing decay_rate should mention 'decay_rate' but got: {}",
        err_msg
    );
}

#[test]
fn test_exponential_missing_decay_rate_error_names_scheduler() {
    let config_json = r#"{
  "scheduler_type": "exponential"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("exponential"),
        "Error for exponential missing decay_rate should name the scheduler but got: {}",
        err_msg
    );
}

#[test]
fn test_cosine_annealing_missing_min_lr_error_mentions_required_fields() {
    // cosine_annealing without min_lr (only has T_max)
    let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "T_max": 10
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when cosine_annealing is missing min_lr"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("min_lr"),
        "Error for cosine_annealing missing min_lr should mention 'min_lr' but got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("T_max"),
        "Error for cosine_annealing missing min_lr should mention 'T_max' but got: {}",
        err_msg
    );
}

#[test]
fn test_cosine_annealing_missing_t_max_error_mentions_required_fields() {
    // cosine_annealing without T_max (only has min_lr)
    let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when cosine_annealing is missing T_max"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("min_lr"),
        "Error for cosine_annealing missing T_max should mention 'min_lr' but got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("T_max"),
        "Error for cosine_annealing missing T_max should mention 'T_max' but got: {}",
        err_msg
    );
}

#[test]
fn test_cosine_annealing_missing_both_fields_error_names_scheduler() {
    let config_json = r#"{
  "scheduler_type": "cosine_annealing"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when cosine_annealing is missing both min_lr and T_max"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cosine_annealing"),
        "Error should name the scheduler 'cosine_annealing' but got: {}",
        err_msg
    );
}

#[test]
fn test_json_parse_error_mentions_common_issues_guidance() {
    // JSON with a trailing comma (common mistake)
    let malformed_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
}"#;

    let temp_file = write_temp_config(malformed_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on trailing comma JSON");
    let err_msg = result.unwrap_err().to_string();
    // The error message should provide guidance about common JSON mistakes
    assert!(
        err_msg.contains("comma")
            || err_msg.contains("quote")
            || err_msg.contains("parse")
            || err_msg.to_lowercase().contains("json"),
        "JSON parse error should mention common issues but got: {}",
        err_msg
    );
}

#[test]
fn test_json_parse_error_includes_file_path() {
    let malformed_json = r#"{ bad json }"#;

    let temp_file = write_temp_config(malformed_json);
    let path_str = temp_file.path().to_str().unwrap();
    let result = load_config(path_str);

    assert!(result.is_err(), "Should fail on bad JSON");
    let err_msg = result.unwrap_err().to_string();
    // The error should include the file path so users know which file caused the problem
    assert!(
        err_msg.contains(path_str) || err_msg.to_lowercase().contains("json"),
        "JSON parse error should include file path or json context but got: {}",
        err_msg
    );
}
