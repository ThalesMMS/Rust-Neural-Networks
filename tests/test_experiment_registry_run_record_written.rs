use std::path::Path;

use rust_neural_networks::experiment_registry::{
    write_run_record, Artifacts, ConfigSnapshot, Metrics, RunRecord, RunStatus,
};
use rust_neural_networks::experiment_registry_run_id::generate_run_id;

#[test]
fn write_run_record_creates_run_json_in_run_dir() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let run_id = generate_run_id();
    let record = RunRecord {
        schema_version: "v0".to_string(),
        run_id: run_id.clone(),
        run_name: Some("smoke".to_string()),
        timestamp_start: "2026-01-01T00:00:00Z".to_string(),
        timestamp_end: Some("2026-01-01T00:00:01Z".to_string()),
        model_type: "test".to_string(),
        command: Some("unit-test".to_string()),
        status: RunStatus::Completed,
        seed: 123,
        config: ConfigSnapshot {
            config_path: None,
            config_format: Some("inline".to_string()),
            raw: Some("{}".to_string()),
            parsed: None,
        },
        dataset: None,
        metrics: Some(Metrics {
            epochs_completed: 1,
            final_train_loss: Some(0.1),
            final_val_loss: Some(0.2),
            final_val_accuracy: Some(99.0),
            total_training_time_seconds: Some(1.5),
        }),
        artifacts: Some(Artifacts {
            training_log_csv: None,
            checkpoints: Vec::new(),
            plots: Vec::new(),
            extra: None,
        }),
        environment: None,
    };

    write_run_record(tmp.path(), &record).expect("write_run_record");

    let run_json = tmp.path().join(&run_id).join("run.json");
    assert!(Path::new(&run_json).exists(), "run.json should exist");

    let loaded = rust_neural_networks::experiment_registry::load_run_record(&run_json)
        .expect("load_run_record");
    assert_eq!(loaded.run_id, record.run_id);
    assert_eq!(loaded.model_type, record.model_type);
    assert_eq!(loaded.seed, record.seed);
}
