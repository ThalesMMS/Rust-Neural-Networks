//! Comprehensive tests for image data augmentation functions
//!
//! This file tests the augmentation module including:
//! - Random horizontal flip
//! - Random crop with padding
//! - Random brightness adjustment
//! - Random contrast adjustment
//! - Random saturation adjustment
//! - Combination of multiple augmentations
//! - Edge cases and error conditions
//!
//! Tests verify that augmentations:
//! - Preserve valid pixel ranges [0.0, 1.0]
//! - Are deterministic with same RNG seed
//! - Handle edge cases correctly
//! - Validate input dimensions properly

use rust_neural_networks::data::augmentation::{
    random_brightness, random_contrast, random_crop, random_horizontal_flip, random_saturation,
};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Horizontal Flip Tests
// ============================================================================

#[path = "test_augmentation/horizontal_flip_tests.rs"]
mod horizontal_flip_tests;

// ============================================================================
// Random Crop Tests
// ============================================================================

#[path = "test_augmentation/random_crop_tests.rs"]
mod random_crop_tests;

// ============================================================================
// Brightness Adjustment Tests
// ============================================================================

#[path = "test_augmentation/brightness_tests.rs"]
mod brightness_tests;

// ============================================================================
// Contrast Adjustment Tests
// ============================================================================

#[path = "test_augmentation/contrast_tests.rs"]
mod contrast_tests;

// ============================================================================
// Saturation Adjustment Tests
// ============================================================================

#[path = "test_augmentation/saturation_tests.rs"]
mod saturation_tests;

// ============================================================================
// Combined Augmentation Tests
// ============================================================================

#[path = "test_augmentation/combined_augmentation_tests.rs"]
mod combined_augmentation_tests;

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

#[path = "test_augmentation/edge_case_tests.rs"]
mod edge_case_tests;
