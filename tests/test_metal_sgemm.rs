//! Tests for Metal sgemm kernel correctness.
//!
//! Compares Metal GPU sgemm output against CPU BLAS sgemm for numerical correctness.

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

/// Compute the single-precision matrix product C = A * B with row-major inputs and output.
///
/// A is interpreted as an m×k matrix stored in row-major order (length m * k).
/// B is interpreted as a k×n matrix stored in row-major order (length k * n).
/// The returned `Vec<f32>` contains C as an m×n matrix in row-major order (length m * n).
///
/// # Examples
///
/// ```
/// let m = 2;
/// let k = 3;
/// let n = 2;
/// // A = [[1,2,3],
/// //      [4,5,6]]
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// // B = [[7,8],
/// //      [9,10],
/// //      [11,12]]
/// let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
/// let c = cpu_sgemm_nn(m, n, k, &a, &b);
/// // C = [[58,64],
/// //      [139,154]]
/// assert_eq!(c, vec![58.0f32, 64.0, 139.0, 154.0]);
/// ```
fn cpu_sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Computes the matrix product C = A^T * B and returns C in row-major layout.
///
/// Interprets `a` as a k×m matrix (stored row-major as a flat slice where element A[p,i] = a[p * m + i]) so that A^T has shape m×k. Interprets `b` as a k×n matrix (element B[p,j] = b[p * n + j]). The result is an m×n matrix stored row-major in the returned Vec.
///
/// # Panics
///
/// This function does not perform bounds checks on the lengths of `a` and `b`; providing slices of incorrect length may panic or produce incorrect results.
///
/// # Examples
///
/// ```
/// let m = 2;
/// let n = 3;
/// let k = 4;
/// // a is k x m: 4x2
/// let a: Vec<f32> = vec![
///     1.0, 2.0,  // column i=0 values for p=0..3 are laid out by p*m + i
///     3.0, 4.0,
///     5.0, 6.0,
///     7.0, 8.0,
/// ];
/// // b is k x n: 4x3
/// let b: Vec<f32> = vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     0.5, 0.5, 0.5,
/// ];
/// let c = cpu_sgemm_tn(m, n, k, &a, &b);
/// assert_eq!(c.len(), m * n);
/// ```
fn cpu_sgemm_tn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[p * m + i] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Computes the matrix product C = A * B^T and returns C in row-major order.

///

/// A is interpreted as an m×k matrix stored row-major (length m*k). B is interpreted as an n×k matrix

/// where each row corresponds to a row of B (length n*k); the function multiplies A by B transposed

/// (so B's rows act as the columns in the multiplication).

///

/// # Examples

///

/// ```

/// let m = 2;

/// let n = 3;

/// let k = 2;

/// // A (2×2): [1, 2;

/// //           3, 4]

/// let a = vec![1.0f32, 2.0, 3.0, 4.0];

/// // B stored as n×k (3×2): rows [5,6], [7,8], [9,10]

/// // So B^T is 2×3: [5,7,9;

/// //                 6,8,10]

/// let b = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];

/// let c = cpu_sgemm_nt(m, n, k, &a, &b);

/// // C is 2×3:

/// // [1*5+2*6, 1*7+2*8, 1*9+2*10] = [17, 23, 29]

/// // [3*5+4*6, 3*7+4*8, 3*9+4*10] = [39, 53, 67]

/// assert_eq!(c, vec![17.0f32, 23.0, 29.0, 39.0, 53.0, 67.0]);

/// ```

///

/// # Returns

///

/// A vector of length `m * n` containing the product matrix C in row-major order (element at row `i`,

/// column `j` is stored at index `i * n + j`).
fn cpu_sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Asserts that two f32 slices are element-wise equal within a given tolerance.
///
/// # Panics
///
/// Panics if the slices have different lengths, or if any corresponding elements
/// have an absolute difference greater than or equal to `tol` (differences must be
/// strictly less than `tol`).
///
/// # Examples
///
/// ```
/// let a = [1.0f32, 2.0, 3.0];
/// let b = [1.0f32, 2.00001, 2.99999];
/// assert_approx_eq(&a, &b, 1e-4);
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

/// Verifies the Metal backend's `sgemm` (no transposes) against a CPU reference for a small 2×3 result.
///
/// This integration test constructs a 2×4 matrix A and a 4×3 matrix B, computes the reference
/// result with the CPU `cpu_sgemm_nn` helper, runs `backend.sgemm(m, n, k, &a, &b, &mut c)`,
/// and asserts element-wise approximate equality within a tolerance of 1e-4.
#[test]
fn test_metal_sgemm_nn_small() {
    let backend = require_metal!();

    let m = 2;
    let n = 3;
    let k = 4;
    // A: 2x4
    let a: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    // B: 4x3
    let b: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.5).collect();

    let expected = cpu_sgemm_nn(m, n, k, &a, &b);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();

    assert_approx_eq(&c, &expected, 1e-4);
}

/// Verifies that multiplying a matrix by a 4×4 identity matrix using the Metal backend's `sgemm` returns the original matrix.
///
/// # Examples
///
/// ```
/// // Requires a Metal backend; test is skipped if unavailable.
/// let backend = require_metal!();
/// let n = 4;
/// let mut eye = vec![0.0f32; n * n];
/// for i in 0..n { eye[i * n + i] = 1.0; }
/// let a: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
/// let mut c = vec![0.0f32; n * n];
/// backend.sgemm(n, n, n, &a, &eye, &mut c).unwrap();
/// assert_approx_eq(&c, &a, 1e-5);
/// ```
#[test]
fn test_metal_sgemm_nn_identity() {
    let backend = require_metal!();

    let n = 4;
    // Identity matrix
    let mut eye = vec![0.0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let a: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
    let mut c = vec![0.0f32; n * n];

    backend.sgemm(n, n, n, &a, &eye, &mut c).unwrap();
    assert_approx_eq(&c, &a, 1e-5);
}

/// Verifies the Metal backend computes C = A^T * B for small matrices where A is stored as k×m.
///
/// Uses a 3×2 result (m=3, n=2, k=4), constructs A as a k×m (4×3) buffer and B as k×n (4×2),
/// compares the Metal backend's `sgemm_at` output against the CPU reference `cpu_sgemm_tn` within 1e-4.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let m = 3;
/// let n = 2;
/// let k = 4;
/// let a: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect(); // 4×3 (k×m)
/// let b: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.3).collect(); // 4×2 (k×n)
/// let expected = cpu_sgemm_tn(m, n, k, &a, &b);
/// let mut c = vec![0.0f32; m * n];
/// backend.sgemm_at(m, n, k, &a, &b, &mut c).unwrap();
/// assert_approx_eq(&c, &expected, 1e-4);
/// ```
#[test]
fn test_metal_sgemm_at_small() {
    let backend = require_metal!();

    let m = 3;
    let n = 2;
    let k = 4;
    // A stored as k×m = 4×3
    let a: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();
    // B: k×n = 4×2
    let b: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.3).collect();

    let expected = cpu_sgemm_tn(m, n, k, &a, &b);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm_at(m, n, k, &a, &b, &mut c).unwrap();

    assert_approx_eq(&c, &expected, 1e-4);
}

#[test]
fn test_metal_sgemm_bt_small() {
    let backend = require_metal!();

    let m = 3;
    let n = 2;
    let k = 4;
    // A: m×k = 3×4
    let a: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();
    // B stored as n×k = 2×4
    let b: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.7).collect();

    let expected = cpu_sgemm_nt(m, n, k, &a, &b);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm_bt(m, n, k, &a, &b, &mut c).unwrap();

    assert_approx_eq(&c, &expected, 1e-4);
}

#[test]
fn test_metal_sgemm_nn_non_tile_aligned() {
    // Test with dimensions that don't align to tile size (16)
    let backend = require_metal!();

    let m = 7;
    let n = 5;
    let k = 9;
    let a: Vec<f32> = (0..(m * k))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let b: Vec<f32> = (0..(k * n))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.2)
        .collect();

    let expected = cpu_sgemm_nn(m, n, k, &a, &b);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();

    assert_approx_eq(&c, &expected, 1e-3);
}

#[test]
fn test_metal_sgemm_nn_larger() {
    // Test with a larger matrix to exercise multiple tiles
    let backend = require_metal!();

    let m = 64;
    let n = 48;
    let k = 32;
    let a: Vec<f32> = (0..(m * k))
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
        .collect();
    let b: Vec<f32> = (0..(k * n))
        .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01)
        .collect();

    let expected = cpu_sgemm_nn(m, n, k, &a, &b);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();

    assert_approx_eq(&c, &expected, 1e-2);
}

/// Ensures the Metal SGEMM implementation fails when input slice sizes do not match the specified matrix dimensions.
///
/// This test constructs input buffers that are too small for the declared m/n/k and asserts that `sgemm` returns an error.
///
/// # Examples
///
/// ```
/// // backend obtained via require_metal!() in test environment
/// let a = vec![1.0f32; 4]; // too small for 2x3
/// let b = vec![1.0f32; 6];
/// let mut c = vec![0.0f32; 6];
/// let result = backend.sgemm(2, 3, 3, &a, &b, &mut c);
/// assert!(result.is_err());
/// ```
#[test]
fn test_metal_sgemm_dimension_mismatch() {
    let backend = require_metal!();

    let a = vec![1.0f32; 4]; // too small for 2x3
    let b = vec![1.0f32; 6];
    let mut c = vec![0.0f32; 6];

    let result = backend.sgemm(2, 3, 3, &a, &b, &mut c);
    assert!(result.is_err());
}