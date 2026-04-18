//! Comprehensive tests for architecture parsing and building
//!
//! This file tests the architecture module including:
//! - Loading valid JSON architecture configs
//! - Parsing different layer types (Dense, Conv2D, BatchNorm, Dropout)
//! - Building models from configs
//! - Handling invalid JSON
//! - Handling missing files
//! - Validating layer connections
//! - Edge cases (empty, single layer, etc.)

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::utils::rng::SimpleRng;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_temp_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    file.write_all(contents.as_bytes())
        .expect("failed to write temp config");
    file
}

// ============================================================================
// Valid Architecture Loading Tests
// ============================================================================

#[path = "test_architecture/valid_architecture_tests.rs"]
mod valid_architecture_tests;

// ============================================================================
// Model Building Tests
// ============================================================================

#[path = "test_architecture/model_building_tests.rs"]
mod model_building_tests;

// ============================================================================
// Error Handling Tests
// ============================================================================

#[path = "test_architecture/error_handling_tests.rs"]
mod error_handling_tests;

// ============================================================================
// Layer Connection Validation Tests
// ============================================================================

#[path = "test_architecture/layer_connection_tests.rs"]
mod layer_connection_tests;

// ============================================================================
// Edge Case Tests
// ============================================================================

#[path = "test_architecture/edge_case_tests.rs"]
mod edge_case_tests;

// ============================================================================
// Comprehensive Integration Tests
// ============================================================================

#[path = "test_architecture/integration_tests.rs"]
mod integration_tests;
