use super::*;

#[test]
fn test_missing_file() {
    let result = load_config("nonexistent_config.json");
    assert!(result.is_err(), "Should fail on missing file");
}

#[test]
fn test_invalid_json_syntax() {
    let invalid_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5
  // Missing closing brace
"#;

    let temp_file = write_temp_config(invalid_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on invalid JSON");
}

#[test]
fn test_malformed_json() {
    let malformed_json = "not valid json at all";

    let temp_file = write_temp_config(malformed_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on malformed JSON");
}

#[test]
fn test_missing_scheduler_type() {
    let config_json = r#"{
  "step_size": 3,
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when scheduler_type is missing"
    );
}

#[test]
fn test_wrong_type_scheduler_type() {
    let config_json = r#"{
  "scheduler_type": 123,
  "step_size": 3
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when scheduler_type is not a string"
    );
}

#[test]
fn test_wrong_type_step_size() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": "three",
  "gamma": 0.5
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "Should fail when step_size is not a number"
    );
}

#[test]
fn test_wrong_type_gamma() {
    let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": "zero point five"
}"#;

    let temp_file = write_temp_config(config_json);
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail when gamma is not a number");
}

#[test]
fn test_empty_file() {
    let temp_file = write_temp_config("");
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on empty file");
}

#[test]
fn test_empty_json_object() {
    let temp_file = write_temp_config("{}");
    let result = load_config(temp_file.path().to_str().unwrap());

    assert!(result.is_err(), "Should fail on empty JSON object");
}
