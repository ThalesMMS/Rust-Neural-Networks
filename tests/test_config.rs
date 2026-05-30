//! Comprehensive tests for configuration parsing
//!
//! This file tests the config module including:
//! - Loading valid JSON config files
//! - Parsing different scheduler types (StepDecay, ExponentialDecay, CosineAnnealing)
//! - Handling invalid JSON
//! - Handling missing files
//! - Handling missing optional fields with defaults

use rust_neural_networks::config::load_config;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_temp_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp file");
    file.write_all(contents.as_bytes())
        .expect("failed to write temp config");
    file
}

// ============================================================================
// Valid Config Loading Tests
// ============================================================================

#[path = "test_config/valid_config_tests.rs"]
mod valid_config_tests;

// ============================================================================
// Temporary Config Creation Tests
// ============================================================================

#[path = "test_config/temp_config_tests.rs"]
mod temp_config_tests;

// ============================================================================
// Error Handling Tests
// ============================================================================

#[path = "test_config/error_handling_tests.rs"]
mod error_handling_tests;

// ============================================================================
// TrainingConfig Structure Tests
// ============================================================================

#[path = "test_config/structure_tests.rs"]
mod structure_tests;

// ============================================================================
// Edge Case Tests
// ============================================================================

#[path = "test_config/edge_case_tests.rs"]
mod edge_case_tests;

// ============================================================================
// Activation Function Tests
// ============================================================================

#[path = "test_config/activation_function_tests.rs"]
mod activation_function_tests;

// ============================================================================
// Training Hyperparameter Validation Tests
// ============================================================================

#[path = "test_config/training_hyperparameter_tests.rs"]
mod training_hyperparameter_tests;

// ============================================================================
// Config Error Message Content Tests
// ============================================================================

#[path = "test_config/config_error_tests.rs"]
mod config_error_tests;

// ============================================================================
// Training control validation tests (warmup/cyclical/regularization/clipping)
// ============================================================================

#[path = "test_config/training_controls_validation_tests.rs"]
mod training_controls_validation_tests;

// ============================================================================
// Optimizer Type Validation Tests
// ============================================================================

#[path = "test_config/optimizer_type_tests.rs"]
mod optimizer_type_tests;

// ============================================================================
// Adam Optimizer Hyperparameter Tests
// ============================================================================

#[path = "test_config/adam_optimizer_tests.rs"]
mod adam_optimizer_tests;

// ============================================================================
// AdamW Optimizer Hyperparameter Tests
// ============================================================================

#[path = "test_config/adamw_optimizer_tests.rs"]
mod adamw_optimizer_tests;

// ============================================================================
// RMSprop Optimizer Hyperparameter Tests
// ============================================================================

#[path = "test_config/rmsprop_optimizer_tests.rs"]
mod rmsprop_optimizer_tests;

// ============================================================================
// Data Augmentation Parameter Tests
// ============================================================================

#[path = "test_config/augmentation_tests.rs"]
mod augmentation_tests;

// ============================================================================
// GAN-Specific Parameter Tests
// ============================================================================

#[path = "test_config/gan_tests.rs"]
mod gan_tests;

// ============================================================================
// Boolean Configuration Field Tests
// ============================================================================

#[path = "test_config/boolean_field_tests.rs"]
mod boolean_field_tests;

// ============================================================================
// CIFAR10 ViT Config File Tests
// ============================================================================

#[path = "test_config/cifar10_vit_config_tests.rs"]
mod cifar10_vit_config_tests;

// ============================================================================
// Additional Edge Case and Regression Tests
// ============================================================================

#[path = "test_config/additional_edge_cases.rs"]
mod additional_edge_cases;
