use rust_neural_networks::experiment_registry::{
    collect_environment, dataset_placeholder, Artifacts, ConfigSnapshot, Metrics, RunRecord,
    RunStatus, RUN_RECORD_SCHEMA_VERSION,
};

#[test]
fn run_record_json_round_trip_preserves_required_fields() {
    let record = RunRecord {
        schema_version: RUN_RECORD_SCHEMA_VERSION.to_string(),
        run_id: "1700000000-deadbeefdeadbeef".to_string(),
        run_name: Some("unit-test".to_string()),
        timestamp_start: "2026-01-01T00:00:00Z".to_string(),
        timestamp_end: Some("2026-01-01T00:00:01Z".to_string()),
        model_type: "mnist_mlp".to_string(),
        command: Some("cargo run --bin mnist_mlp".to_string()),
        status: RunStatus::Completed,
        seed: 123,
        config: ConfigSnapshot {
            config_path: Some("configs/mnist_mlp.json".to_string()),
            config_format: Some("json".to_string()),
            raw: Some("{\"epochs\": 1}".to_string()),
            parsed: Some(serde_json::json!({"epochs": 1})),
        },
        dataset: Some(dataset_placeholder("mnist")),
        metrics: Some(Metrics {
            epochs_completed: 1,
            final_train_loss: Some(0.1),
            final_val_loss: Some(0.2),
            final_val_accuracy: Some(0.9),
            total_training_time_seconds: Some(0.01),
        }),
        artifacts: Some(Artifacts {
            training_log_csv: Some("./logs/training_loss_adam.csv".to_string()),
            checkpoints: vec!["./logs/model_final.bin".to_string()],
            plots: Vec::new(),
            extra: Some(serde_json::json!({"note": "extra"})),
        }),
        environment: Some(collect_environment()),
    };

    let json = serde_json::to_string_pretty(&record).expect("serialize RunRecord");
    let decoded: RunRecord = serde_json::from_str(&json).expect("deserialize RunRecord");

    assert_eq!(decoded.schema_version, RUN_RECORD_SCHEMA_VERSION);
    assert_eq!(decoded.run_id, record.run_id);
    assert_eq!(decoded.model_type, record.model_type);
    assert_eq!(decoded.seed, record.seed);

    let metrics = decoded.metrics.expect("metrics should be present");
    assert_eq!(metrics.epochs_completed, 1);
    assert_eq!(metrics.final_val_accuracy, Some(0.9));

    assert!(decoded.config.raw.is_some());
}
