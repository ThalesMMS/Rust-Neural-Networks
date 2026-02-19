//! Tests for Metal element-wise kernels.
//!
//! Validates GPU-accelerated activation functions, bias operations, and row
//! reductions against CPU reference implementations.

#![cfg(feature = "gpu-metal")]

use rust_neural_networks::gpu::backend::GpuBackend;
use rust_neural_networks::gpu::metal_backend::MetalBackend;

/// Macro to skip tests when no Metal GPU is available.
macro_rules! require_metal {
    () => {
        match MetalBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping test: no Metal GPU available");
                return;
            }
        }
    };
}

/// Asserts that two f32 slices are equal element-wise within a tolerance.
///
/// Panics if the slices have different lengths or if any element pair differs by
/// greater than or equal to `tol`.
///
/// # Examples
///
/// ```
/// let a = [0.0_f32, 1.0, -2.5];
/// let b = [0.0_f32, 1.000001, -2.499999];
/// assert_approx_eq(&a, &b, 1e-5);
/// ```
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "Length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "Element {} differs: {} vs {} (diff={})",
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

// ── ReLU tests ──────────────────────────────────────────────────────────

/// Applies the rectified linear unit (ReLU) activation in-place to a flat vector, replacing each negative element with `0.0`.
///
/// # Examples
///
/// ```ignore
/// let backend = require_metal!();
/// let mut data = vec![-2.0, -1.0, 0.0, 1.0];
/// backend.relu(&mut data).unwrap();
/// assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0]);
/// ```
#[test]
fn test_metal_elementwise_relu_basic() {
    let backend = require_metal!();

    let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 3.0, -100.0];
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0];

    backend.relu(&mut data).unwrap();
    assert_approx_eq(&data, &expected, 1e-6);
}

/// Checks that applying ReLU to a vector of all-positive values leaves each element unchanged.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let expected = vec![1.0f32, 2.0, 3.0, 4.0];
///
/// backend.relu(&mut data).unwrap();
/// assert_approx_eq(&data, &expected, 1e-6);
/// ```
#[test]
fn test_metal_elementwise_relu_all_positive() {
    let backend = require_metal!();

    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    let expected = vec![1.0, 2.0, 3.0, 4.0];

    backend.relu(&mut data).unwrap();
    assert_approx_eq(&data, &expected, 1e-6);
}

#[test]
fn test_metal_elementwise_relu_all_negative() {
    let backend = require_metal!();

    let mut data = vec![-1.0, -2.0, -3.0, -4.0];
    let expected = vec![0.0, 0.0, 0.0, 0.0];

    backend.relu(&mut data).unwrap();
    assert_approx_eq(&data, &expected, 1e-6);
}

/// Verifies that applying ReLU to an empty vector preserves its emptiness.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let mut data: Vec<f32> = vec![];
/// backend.relu(&mut data).unwrap();
/// assert!(data.is_empty());
/// ```
#[test]
fn test_metal_elementwise_relu_empty() {
    let backend = require_metal!();
    let mut data: Vec<f32> = vec![];
    backend.relu(&mut data).unwrap();
    assert!(data.is_empty());
}

// ── ReLU backward tests ────────────────────────────────────────────────

/// Verifies that the Metal backend computes the ReLU backward pass correctly for a small vector.
///
/// The test checks that gradients are passed through where the input is greater than zero
/// and set to zero otherwise, comparing the backend result to an expected reference.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5];
/// let grad_output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut grad_input = vec![0.0f32; 6];
/// let expected = vec![0.0, 0.0, 0.0, 4.0, 5.0, 0.0];
///
/// backend.relu_backward(&input, &grad_output, &mut grad_input).unwrap();
/// assert_approx_eq(&grad_input, &expected, 1e-6);
/// ```
#[test]
fn test_metal_elementwise_relu_backward() {
    let backend = require_metal!();

    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5];
    let grad_output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut grad_input = vec![0.0f32; 6];
    let expected = vec![0.0, 0.0, 0.0, 4.0, 5.0, 0.0];

    backend
        .relu_backward(&input, &grad_output, &mut grad_input)
        .unwrap();
    assert_approx_eq(&grad_input, &expected, 1e-6);
}

/// Verifies that `relu_backward` returns an error when `grad_output` length does not match `input` length.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let input = vec![1.0, 2.0, 3.0];
/// let grad_output = vec![1.0]; // too short
/// let mut grad_input = vec![0.0f32; 3];
/// assert!(backend.relu_backward(&input, &grad_output, &mut grad_input).is_err());
/// ```
#[test]
fn test_metal_elementwise_relu_backward_dimension_mismatch() {
    let backend = require_metal!();

    let input = vec![1.0, 2.0, 3.0];
    let grad_output = vec![1.0]; // too short
    let mut grad_input = vec![0.0f32; 3];

    let result = backend.relu_backward(&input, &grad_output, &mut grad_input);
    assert!(result.is_err());
}

// ── Sigmoid tests ───────────────────────────────────────────────────────

/// Checks that the Metal backend's in-place sigmoid produces expected values for typical and extreme inputs.
///
/// Verifies sigmoid(0) == 0.5, sigmoid(1) ≈ 0.7310586, sigmoid(-1) ≈ 0.2689414, and that large-magnitude inputs saturate near 0 or 1.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let mut data = vec![0.0, 1.0, -1.0, 10.0, -10.0];
/// backend.sigmoid(&mut data).unwrap();
/// assert!((data[0] - 0.5).abs() < 1e-5);
/// ```
#[test]
fn test_metal_elementwise_sigmoid_basic() {
    let backend = require_metal!();

    let mut data = vec![0.0, 1.0, -1.0, 10.0, -10.0];
    backend.sigmoid(&mut data).unwrap();

    // sigmoid(0) = 0.5
    assert!((data[0] - 0.5).abs() < 1e-5);
    // sigmoid(1) ≈ 0.7311
    assert!((data[1] - 0.7310586).abs() < 1e-4);
    // sigmoid(-1) ≈ 0.2689
    assert!((data[2] - 0.2689414).abs() < 1e-4);
    // sigmoid(10) ≈ 1.0
    assert!(data[3] > 0.999);
    // sigmoid(-10) ≈ 0.0
    assert!(data[4] < 0.001);
}

/// Ensures the Metal backend's sigmoid kernel succeeds when given an empty input.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let mut data: Vec<f32> = vec![];
/// backend.sigmoid(&mut data).unwrap();
/// ```
#[test]
fn test_metal_elementwise_sigmoid_empty() {
    let backend = require_metal!();
    let mut data: Vec<f32> = vec![];
    backend.sigmoid(&mut data).unwrap();
}

// ── Add bias tests ──────────────────────────────────────────────────────

#[test]
fn test_metal_elementwise_add_bias() {
    let backend = require_metal!();

    // 3 rows x 4 columns
    let mut data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let bias = vec![0.1, 0.2, 0.3, 0.4];
    let expected = vec![
        1.1, 2.2, 3.3, 4.4, 5.1, 6.2, 7.3, 8.4, 9.1, 10.2, 11.3, 12.4,
    ];

    backend.add_bias(&mut data, &bias, 3, 4).unwrap();
    assert_approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_metal_elementwise_add_bias_single_row() {
    let backend = require_metal!();

    let mut data = vec![1.0, 2.0, 3.0];
    let bias = vec![10.0, 20.0, 30.0];
    let expected = vec![11.0, 22.0, 33.0];

    backend.add_bias(&mut data, &bias, 1, 3).unwrap();
    assert_approx_eq(&data, &expected, 1e-5);
}

/// Verifies that add_bias reports an error when the provided data buffer is too small for the specified matrix dimensions.
///
/// Attempts to add a bias vector to a 2×3 matrix stored in a too-short data slice and asserts that the backend returns an error.
///
/// # Examples
///
/// ```ignore
/// let backend = require_metal!();
/// let mut data = vec![1.0, 2.0]; // too small for 2x3
/// let bias = vec![0.1, 0.2, 0.3];
/// assert!(backend.add_bias(&mut data, &bias, 2, 3).is_err());
/// ```
#[test]
fn test_metal_elementwise_add_bias_dimension_mismatch() {
    let backend = require_metal!();

    let mut data = vec![1.0, 2.0]; // too small for 2x3
    let bias = vec![0.1, 0.2, 0.3];

    let result = backend.add_bias(&mut data, &bias, 2, 3);
    assert!(result.is_err());
}

// ── Sum rows tests ──────────────────────────────────────────────────────

/// Verifies that `sum_rows` computes column-wise sums across multiple rows.
///
/// This test creates a 3x4 matrix (row-major flattened), runs `sum_rows` to
/// reduce rows into per-column sums, and asserts the GPU-backed result matches
/// the expected column-wise totals within a tolerance.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// // 3 rows x 4 columns (row-major)
/// let data = vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
/// ];
/// let mut out = vec![0.0f32; 4];
/// let expected = vec![15.0, 18.0, 21.0, 24.0];
///
/// backend.sum_rows(&data, &mut out, 3, 4).unwrap();
/// assert_approx_eq(&out, &expected, 1e-5);
/// ```
#[test]
fn test_metal_elementwise_sum_rows() {
    let backend = require_metal!();

    // 3 rows x 4 columns
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut out = vec![0.0f32; 4];
    // Expected: column sums [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
    let expected = vec![15.0, 18.0, 21.0, 24.0];

    backend.sum_rows(&data, &mut out, 3, 4).unwrap();
    assert_approx_eq(&out, &expected, 1e-5);
}

/// Verifies that summing rows over a single-row matrix returns the original row values.
///
/// Ensures column-wise reduction with rows = 1 produces the same elements as the input row.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let data = vec![3.0, 5.0, 7.0];
/// let mut out = vec![0.0f32; 3];
/// backend.sum_rows(&data, &mut out, 1, 3).unwrap();
/// assert_eq!(out, vec![3.0, 5.0, 7.0]);
/// ```
#[test]
fn test_metal_elementwise_sum_rows_single_row() {
    let backend = require_metal!();

    let data = vec![3.0, 5.0, 7.0];
    let mut out = vec![0.0f32; 3];
    let expected = vec![3.0, 5.0, 7.0];

    backend.sum_rows(&data, &mut out, 1, 3).unwrap();
    assert_approx_eq(&out, &expected, 1e-5);
}

/// Verifies that `sum_rows` returns an error when the input buffer size does not match the specified dimensions.
///
/// The test constructs a data buffer that is too small for the given row/column shape (2 rows × 3 columns)
/// and asserts that the backend reports an error.
#[test]
fn test_metal_elementwise_sum_rows_dimension_mismatch() {
    let backend = require_metal!();

    let data = vec![1.0, 2.0]; // too small for 2x3
    let mut out = vec![0.0f32; 3];

    let result = backend.sum_rows(&data, &mut out, 2, 3);
    assert!(result.is_err());
}

// ── Larger test for non-trivial sizes ───────────────────────────────────

/// Validates that the Metal backend's in-place ReLU produces the same results as a CPU reference on a 1024-element input.
///
/// The test constructs a vector spanning negative and positive values, applies the backend's `relu` in place, and compares each element to the CPU-computed ReLU within a small tolerance.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let n = 1024;
/// let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.1).collect();
/// let expected: Vec<f32> = data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
/// backend.relu(&mut data).unwrap();
/// assert_approx_eq(&data, &expected, 1e-5);
/// ```
#[test]
fn test_metal_elementwise_relu_larger() {
    let backend = require_metal!();

    let n = 1024;
    let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.1).collect();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    backend.relu(&mut data).unwrap();
    assert_approx_eq(&data, &expected, 1e-5);
}

/// Validates that the Metal backend computes the sigmoid in-place for a 512-element input.
///
/// Creates 512 values centered near zero, computes the CPU reference sigmoid for each value,
/// runs the backend's in-place `sigmoid`, and asserts elementwise equality within a tolerance.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
///
/// let n = 512;
/// let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.02).collect();
/// let expected: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
///
/// backend.sigmoid(&mut data).unwrap();
/// assert_approx_eq(&data, &expected, 1e-4);
/// ```
fn test_metal_elementwise_sigmoid_larger() {
    let backend = require_metal!();

    let n = 512;
    let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.02).collect();
    let expected: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    backend.sigmoid(&mut data).unwrap();
    assert_approx_eq(&data, &expected, 1e-4);
}