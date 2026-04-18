// Tests for CsvGradientLogger: file creation, header format,
// write_layer() CSV rows, and multi-layer multi-epoch output.

use rust_neural_networks::training::CsvGradientLogger;
use tempfile::NamedTempFile;

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the path of a freshly-created temporary file as a String.
///
/// The file handle is returned alongside so the caller keeps it alive for the
/// duration of the test; dropping it would delete the file on some platforms.
fn tmp_path() -> (NamedTempFile, String) {
    let f = NamedTempFile::new().expect("create temp file");
    let path = f.path().to_string_lossy().to_string();
    (f, path)
}

// ─────────────────────────────────────────────────────────────────────────────
// File creation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_csv_gradient_logger_creates_file() {
    let (_guard, path) = tmp_path();
    let logger = CsvGradientLogger::new(&path);
    assert!(
        logger.is_ok(),
        "CsvGradientLogger::new should succeed for a valid path"
    );
}

#[test]
fn test_csv_gradient_logger_invalid_path_returns_error() {
    // A deeply nested path that does not exist should produce an Err.
    let result = CsvGradientLogger::new("/nonexistent/directory/gradients.csv");
    assert!(
        result.is_err(),
        "opening a file in a nonexistent directory should fail"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Header format
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_write_header_format() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let first_line = content.lines().next().expect("file should not be empty");

    assert_eq!(
        first_line, "epoch,layer_name,grad_norm_weights,grad_norm_biases",
        "header must match expected CSV columns"
    );
}

#[test]
fn test_write_header_only_one_line() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let line_count = content.lines().count();
    assert_eq!(
        line_count, 1,
        "write_header should produce exactly one line"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// write_layer() produces correct CSV rows
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_write_layer_single_row() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger
        .write_layer(1, "layer_0_dense", 0.042314, 0.003217)
        .expect("write layer");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    // There must be a header plus one data row.
    assert_eq!(lines.len(), 2, "expected header + 1 data row");

    let row = lines[1];
    let parts: Vec<&str> = row.split(',').collect();

    assert_eq!(
        parts.len(),
        4,
        "data row must have 4 comma-separated fields"
    );
    assert_eq!(parts[0], "1", "epoch field should be 1");
    assert_eq!(parts[1], "layer_0_dense", "layer_name field should match");

    // Numeric fields: parse them and check they are close to the expected values.
    let weight_norm: f32 = parts[2].parse().expect("weight_norm should be a valid f32");
    let bias_norm: f32 = parts[3].parse().expect("bias_norm should be a valid f32");

    assert!(
        (weight_norm - 0.042314_f32).abs() < 1e-5,
        "weight_norm value mismatch: got {}",
        weight_norm
    );
    assert!(
        (bias_norm - 0.003217_f32).abs() < 1e-5,
        "bias_norm value mismatch: got {}",
        bias_norm
    );
}

#[test]
fn test_write_layer_zero_norms() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger
        .write_layer(1, "layer_0_dense", 0.0, 0.0)
        .expect("write layer with zero norms");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines.len(), 2, "expected header + 1 data row");

    let parts: Vec<&str> = lines[1].split(',').collect();
    let weight_norm: f32 = parts[2].parse().expect("parse weight_norm");
    let bias_norm: f32 = parts[3].parse().expect("parse bias_norm");

    assert_eq!(weight_norm, 0.0_f32, "weight_norm should be 0.0");
    assert_eq!(bias_norm, 0.0_f32, "bias_norm should be 0.0");
}

#[test]
fn test_write_layer_epoch_field_correct() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger
        .write_layer(42, "some_layer", 0.1, 0.01)
        .expect("write layer");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let data_line = content.lines().nth(1).expect("data row should exist");
    let parts: Vec<&str> = data_line.split(',').collect();

    assert_eq!(parts[0], "42", "epoch field should be 42");
}

// ─────────────────────────────────────────────────────────────────────────────
// Multiple layers across multiple epochs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multiple_layers_single_epoch() {
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger
        .write_layer(1, "layer_0_dense", 0.05, 0.005)
        .expect("write layer 0");
    logger
        .write_layer(1, "layer_1_dense", 0.03, 0.003)
        .expect("write layer 1");
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    // Header + 2 data rows.
    assert_eq!(lines.len(), 3, "expected header + 2 data rows for 2 layers");

    // Both rows belong to epoch 1.
    for data_line in &lines[1..] {
        let parts: Vec<&str> = data_line.split(',').collect();
        assert_eq!(parts[0], "1", "both rows should have epoch=1");
    }

    // First row: layer_0_dense
    let parts0: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(
        parts0[1], "layer_0_dense",
        "first data row should be layer_0_dense"
    );

    // Second row: layer_1_dense
    let parts1: Vec<&str> = lines[2].split(',').collect();
    assert_eq!(
        parts1[1], "layer_1_dense",
        "second data row should be layer_1_dense"
    );
}

#[test]
fn test_multiple_layers_multiple_epochs() {
    let (_guard, path) = tmp_path();

    let layer_data = [
        (1usize, "layer_0_dense", 0.05_f32, 0.005_f32),
        (1, "layer_1_dense", 0.03, 0.003),
        (2, "layer_0_dense", 0.04, 0.004),
        (2, "layer_1_dense", 0.02, 0.002),
        (3, "layer_0_dense", 0.03, 0.003),
        (3, "layer_1_dense", 0.01, 0.001),
    ];

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    for &(epoch, name, w_norm, b_norm) in &layer_data {
        logger
            .write_layer(epoch, name, w_norm, b_norm)
            .expect("write layer");
    }
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    // 1 header + 6 data rows.
    assert_eq!(
        lines.len(),
        7,
        "expected header + 6 data rows (2 layers × 3 epochs)"
    );

    // Validate each data row against the source data.
    for (i, &(epoch, name, w_norm, b_norm)) in layer_data.iter().enumerate() {
        let row = lines[i + 1]; // skip header
        let parts: Vec<&str> = row.split(',').collect();

        assert_eq!(parts.len(), 4, "row {} must have 4 fields", i + 1);
        assert_eq!(parts[0], epoch.to_string(), "row {} epoch mismatch", i + 1);
        assert_eq!(parts[1], name, "row {} layer_name mismatch", i + 1);

        let parsed_w: f32 = parts[2].parse().expect("parse weight_norm");
        let parsed_b: f32 = parts[3].parse().expect("parse bias_norm");

        assert!(
            (parsed_w - w_norm).abs() < 1e-5,
            "row {} weight_norm mismatch: expected {}, got {}",
            i + 1,
            w_norm,
            parsed_w
        );
        assert!(
            (parsed_b - b_norm).abs() < 1e-5,
            "row {} bias_norm mismatch: expected {}, got {}",
            i + 1,
            b_norm,
            parsed_b
        );
    }
}

#[test]
fn test_gradient_norm_ordering_preserved() {
    // Verify that rows appear in insertion order (not reordered or buffered
    // out of sequence).
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");

    let epochs = 1..=3_usize;
    let layer_names = ["layer_0_dense", "layer_1_dense"];

    let mut expected_rows: Vec<(usize, &str, f32, f32)> = Vec::new();
    for epoch in epochs {
        for &name in &layer_names {
            let w = 0.1_f32 / epoch as f32;
            let b = 0.01_f32 / epoch as f32;
            logger.write_layer(epoch, name, w, b).expect("write layer");
            expected_rows.push((epoch, name, w, b));
        }
    }
    logger.flush().expect("flush");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(
        lines.len(),
        expected_rows.len() + 1,
        "line count should be 1 (header) + number of written rows"
    );

    for (i, &(epoch, name, w_norm, b_norm)) in expected_rows.iter().enumerate() {
        let parts: Vec<&str> = lines[i + 1].split(',').collect();
        assert_eq!(
            parts[0],
            epoch.to_string(),
            "epoch ordering preserved at row {}",
            i + 1
        );
        assert_eq!(
            parts[1],
            name,
            "layer_name ordering preserved at row {}",
            i + 1
        );

        let parsed_w: f32 = parts[2].parse().expect("parse weight_norm");
        let parsed_b: f32 = parts[3].parse().expect("parse bias_norm");
        assert!(
            (parsed_w - w_norm).abs() < 1e-5,
            "weight_norm at row {}",
            i + 1
        );
        assert!(
            (parsed_b - b_norm).abs() < 1e-5,
            "bias_norm at row {}",
            i + 1
        );
    }
}

#[test]
fn test_flush_does_not_corrupt_data() {
    // Call flush between writes and verify the content is still correct.
    let (_guard, path) = tmp_path();

    let mut logger = CsvGradientLogger::new(&path).expect("create logger");
    logger.write_header().expect("write header");
    logger.flush().expect("flush after header");

    logger
        .write_layer(1, "layer_0_dense", 0.05, 0.005)
        .expect("write epoch 1 layer 0");
    logger.flush().expect("flush after epoch 1 layer 0");

    logger
        .write_layer(2, "layer_0_dense", 0.04, 0.004)
        .expect("write epoch 2 layer 0");
    logger.flush().expect("flush after epoch 2 layer 0");

    let content = std::fs::read_to_string(&path).expect("read file");
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines.len(), 3, "expected header + 2 data rows");
    assert_eq!(
        lines[0], "epoch,layer_name,grad_norm_weights,grad_norm_biases",
        "header intact after mid-session flushes"
    );

    let parts1: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(parts1[0], "1", "first data row epoch should be 1");

    let parts2: Vec<&str> = lines[2].split(',').collect();
    assert_eq!(parts2[0], "2", "second data row epoch should be 2");
}
