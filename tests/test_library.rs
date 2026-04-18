//! Comprehensive tests for the rust_neural_networks library
//!
//! This file tests the public API of the library including:
//! - DenseLayer: creation, forward, backward, parameter updates
//! - Conv2DLayer: creation, forward, backward, parameter updates
//! - Activation functions: relu, softmax
//! - SimpleRng: random number generation

#[cfg(feature = "shared_activations")]
use approx::assert_relative_eq;
use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::activations::relu_inplace;
#[cfg(feature = "shared_activations")]
use rust_neural_networks::utils::activations::softmax_rows;
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// DenseLayer Tests
// ============================================================================

#[path = "test_library/dense_layer_tests.rs"]
mod dense_layer_tests;

// ============================================================================
// Conv2DLayer Tests
// ============================================================================

#[path = "test_library/conv2d_layer_tests.rs"]
mod conv2d_layer_tests;

// ============================================================================
// Activation Function Tests
// ============================================================================

#[cfg(feature = "shared_activations")]
#[path = "test_library/activation_tests.rs"]
mod activation_tests;

// ============================================================================
// SimpleRng Tests
// ============================================================================

#[path = "test_library/rng_tests.rs"]
mod rng_tests;

// ============================================================================
// Integration Tests - End-to-End Training
// ============================================================================

#[path = "test_library/integration_tests.rs"]
mod integration_tests;
