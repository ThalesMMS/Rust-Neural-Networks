// Tests for GPU numerical equivalence: verifies that GPU operations produce
// results matching CPU reference implementations within f32 tolerance.
//
// Feature-gated: only compiled when gpu-metal or gpu-cuda is enabled.

#![cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

use rust_neural_networks::gpu::{create_gpu_backend, GpuBackend};
use std::sync::Arc;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Attempts to create a GPU backend and return it if available.
///
/// # Returns
///
/// `Some(Arc<dyn GpuBackend>)` when a GPU backend was successfully created, `None` otherwise.
///
/// # Examples
///
/// ```
/// if let Some(backend) = try_get_backend() {
///     // use `backend` to run GPU-backed operations
/// } else {
///     // no GPU available
/// }
/// ```
fn try_get_backend() -> Option<Arc<dyn GpuBackend>> {
    create_gpu_backend()
}

/// Macro to skip tests when no GPU is available.
macro_rules! require_gpu {
    () => {
        match try_get_backend() {
            Some(b) => b,
            None => {
                eprintln!("Skipping test: no GPU backend available");
                return;
            }
        }
    };
}

/// Generates a deterministic pseudo-random vector of f32 values using an xorshift PRNG.
///
/// Each element is approximately in the range [-1.0, 1.0]. The sequence is deterministic and
/// reproducible for a given `seed`; a `seed` of 0 is treated as 1 to avoid a zero state.
///
/// # Examples
///
/// ```
/// let v = make_data(4, 42);
/// assert_eq!(v.len(), 4);
/// for &x in &v {
///     assert!(x >= -1.0 && x <= 1.0);
/// }
/// ```
///
/// # Returns
///
/// A `Vec<f32>` of length `len` containing pseudo-random values approximately in [-1, 1].
fn make_data(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Map to roughly [-1, 1]
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Asserts element-wise numerical agreement between two f32 slices within a relative tolerance.
///
/// Panics if the slices have different lengths or if any corresponding pair of elements
/// has a relative difference greater than `rel_tol`. The relative difference for values
/// `c` (reference) and `g` (actual) is computed as `|c - g| / max(|c|, |g|, 1e-8)`.
///
/// # Examples
///
/// ```
/// let a = [1.0_f32, 0.0, -2.0];
/// let b = [1.00005_f32, 1e-9, -2.00005];
/// assert_approx_eq("example", &a, &b, 1e-3);
/// ```
fn assert_approx_eq(label: &str, cpu: &[f32], gpu: &[f32], rel_tol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{}: length mismatch", label);
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        let denom = c.abs().max(g.abs()).max(1e-8);
        assert!(
            diff / denom < rel_tol,
            "{} index {}: cpu={} gpu={} rel_diff={}",
            label,
            i,
            c,
            g,
            diff / denom
        );
    }
}

// ── CPU reference implementations ───────────────────────────────────────────

/// Computes the matrix product C = A × B and returns C in row-major order.
///
/// `a` must have length `m * k` and is interpreted row-major (row i at `a[i*k .. (i+1)*k]`).
/// `b` must have length `k * n` and is interpreted row-major (row p at `b[p*n .. (p+1)*n]`).
///
/// # Returns
///
/// A `Vec<f32>` of length `m * n` containing the product matrix C in row-major order.
///
/// # Examples
///
/// ```
/// let m = 2;
/// let k = 3;
/// let n = 2;
/// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
/// let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
/// let c = cpu_sgemm(m, n, k, &a, &b);
/// assert_eq!(c, vec![58.0f32, 64.0, 139.0, 154.0]);
/// ```
fn cpu_sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Compute the matrix product C = A × B where A is provided transposed.
///
/// A is stored as a (k × m) array and interpreted as the transpose of an (m × k) matrix. The result is a flattened row-major (m × n) matrix.
///
/// # Examples
///
/// ```
/// let m = 2;
/// let n = 2;
/// let k = 2;
/// // A (m x k) = [[1, 2],
/// //               [3, 4]]
/// // stored as (k x m): rows p=0..k-1 are [1,3], [2,4]
/// let a = vec![1.0f32, 3.0, 2.0, 4.0];
/// // B (k x n) = [[5, 6],
/// //               [7, 8]]
/// let b = vec![5.0f32, 6.0, 7.0, 8.0];
/// let c = cpu_sgemm_at(m, n, k, &a, &b);
/// assert_eq!(c, vec![19.0f32, 22.0, 43.0, 50.0]); // C = [[19,22],[43,50]]
/// ```
fn cpu_sgemm_at(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // A stored as (k x m), transposed to (m x k)
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[p * m + i] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Computes the matrix product C = A × B where:
/// - A has shape (m × k) stored row-major in `a`.
/// - B is provided in transposed storage as (n × k) (i.e., each row of `b` is a length-`k` row corresponding to a column of the logical B).
/// The result is returned as an m × n row-major vector.
///
/// # Parameters
///
/// - `m`: number of rows of A and C.
/// - `n`: number of columns of B and C.
/// - `k`: number of columns of A and rows of B (the reduction dimension).
/// - `a`: slice containing A in row-major order with length `m * k`.
/// - `b`: slice containing B stored as (n × k) with length `n * k`.
///
/// # Returns
///
/// A `Vec<f32>` of length `m * n` containing C in row-major order.
///
/// # Examples
///
/// ```rust
/// // A = [ [1, 2],
/// //       [3, 4] ]  (2x2)
/// // B (logical) = [ [5, 6, 7],
/// //                 [8, 9, 10] ] (2x3)
/// // b is stored as (n x k) = (3 x 2) rows: [5,8], [6,9], [7,10]
/// let a = vec![1.0f32, 2.0, 3.0, 4.0];
/// let b = vec![5.0f32, 8.0, 6.0, 9.0, 7.0, 10.0];
/// let c = cpu_sgemm_bt(2, 3, 2, &a, &b);
/// // Expected C (2x3): [1*5+2*8, 1*6+2*9, 1*7+2*10,
/// //                   3*5+4*8, 3*6+4*9, 3*7+4*10]
/// let expected = vec![21.0f32, 24.0, 27.0, 47.0, 54.0, 61.0];
/// assert_eq!(c, expected);
/// ```
fn cpu_sgemm_bt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // B stored as (n x k), transposed to (k x n)
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Applies the rectified linear unit (ReLU) element-wise to a slice of floats.

///

/// Each negative value is replaced with 0.0; non-negative values are left unchanged.

///

/// # Examples

///

/// ```

/// let input = [-1.0_f32, 0.0, 2.5];

/// let out = cpu_relu(&input);

/// assert_eq!(out, vec![0.0_f32, 0.0, 2.5]);

/// ```
fn cpu_relu(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| x.max(0.0)).collect()
}

/// Applies the sigmoid activation to each element of the input slice.
///
/// # Examples
///
/// ```
/// let input = vec![-1.0f32, 0.0, 1.0];
/// let out = cpu_sigmoid(&input);
/// assert!((out[0] - 0.26894143).abs() < 1e-6);
/// assert!((out[1] - 0.5).abs() < 1e-6);
/// assert!((out[2] - 0.7310586).abs() < 1e-6);
/// ```
///
/// # Returns
///
/// A `Vec<f32>` where each element is `1 / (1 + exp(-x))` for the corresponding input `x`.
fn cpu_sigmoid(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// Adds per-column bias to each row of a batched row-major 2D tensor.
///
/// Each of the `batch_size` rows (length `n`) in `data` has `bias[j]` added to its j-th element.
///
/// # Examples
///
/// ```
/// let data = vec![0.0_f32; 6]; // 2 rows × 3 cols
/// let bias = vec![1.0_f32, 2.0, 3.0];
/// let out = cpu_add_bias(&data, &bias, 2, 3);
/// assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
/// ```
fn cpu_add_bias(data: &[f32], bias: &[f32], batch_size: usize, n: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    for b in 0..batch_size {
        for j in 0..n {
            out[b * n + j] += bias[j];
        }
    }
    out
}

/// Computes per-column sums over a batched row-major 2D tensor.
///
/// The input `data` is interpreted as `batch_size` rows each with `n` columns (row-major).
/// The returned vector has length `n` where element `j` is the sum of `data[b * n + j]`
/// for all batches `b` in 0..`batch_size`.
///
/// # Examples
///
/// ```
/// let data = vec![
///     1.0f32, 2.0, 3.0, // row 0
///     4.0, 5.0, 6.0,    // row 1
/// ];
/// let sums = cpu_sum_rows(&data, 2, 3);
/// assert_eq!(sums, vec![5.0f32, 7.0, 9.0]);
/// ```
fn cpu_sum_rows(data: &[f32], batch_size: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for b in 0..batch_size {
        for j in 0..n {
            out[j] += data[b * n + j];
        }
    }
    out
}

/// Computes the forward pass of a 2D convolution over an NCHW-formatted input.
///
/// The `input` is expected flattened in NCHW order with shape (batch_size, in_ch, h, w).
/// `filters` are packed in (out_ch, in_ch, kh, kw) order and `bias` has length `out_ch`.
/// The output spatial dimensions are computed as:
/// `out_h = (h + 2*padding - kh) / stride + 1` and
/// `out_w = (w + 2*padding - kw) / stride + 1`.
/// The returned vector contains the output flattened in NCHW order
/// (batch_size, out_ch, out_h, out_w).
///
/// # Examples
///
/// ```
/// // Single-example: 1x1 output from a 3x3 input and 3x3 filter with no padding and stride 1.
/// let input = vec![
///     1.0f32, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// ]; // shape: (1, 1, 3, 3)
/// let filters = vec![1.0f32; 9]; // shape: (1, 1, 3, 3)
/// let bias = vec![0.0f32]; // shape: (1)
/// let out = cpu_conv2d_forward(&input, &filters, &bias, 1, 1, 1, 3, 3, 3, 3, 1, 0);
/// assert_eq!(out.len(), 1);
/// assert_eq!(out[0], 45.0f32);
/// ```
fn cpu_conv2d_forward(
    input: &[f32],
    filters: &[f32],
    bias: &[f32],
    batch_size: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    let mut output = vec![0.0f32; batch_size * out_ch * out_h * out_w];

    for b in 0..batch_size {
        for oc in 0..out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias[oc];
                    for ic in 0..in_ch {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * stride + fh;
                                let iw = ow * stride + fw;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let in_idx = b * (in_ch * h * w) + ic * (h * w) + ih * w + iw;
                                    let f_idx =
                                        oc * (in_ch * kh * kw) + ic * (kh * kw) + fh * kw + fw;
                                    sum += input[in_idx] * filters[f_idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        b * (out_ch * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
    output
}

// ── Tests ───────────────────────────────────────────────────────────────────

const TOL: f32 = 1e-4;

mod sgemm_tests {
    use super::*;

    #[test]
    fn test_sgemm_small() {
        let backend = require_gpu!();
        let (m, n, k) = (4, 4, 4);
        let a = make_data(m * k, 1);
        let b = make_data(k * n, 2);
        let cpu = cpu_sgemm(m, n, k, &a, &b);
        let mut gpu = vec![0.0f32; m * n];
        backend.sgemm(m, n, k, &a, &b, &mut gpu).unwrap();
        assert_approx_eq("sgemm_small", &cpu, &gpu, TOL);
    }

    #[test]
    fn test_sgemm_batch_sizes() {
        let backend = require_gpu!();
        for &batch in &[1, 16, 64, 256] {
            let (m, n, k) = (batch, 128, 64);
            let a = make_data(m * k, 10);
            let b = make_data(k * n, 20);
            let cpu = cpu_sgemm(m, n, k, &a, &b);
            let mut gpu = vec![0.0f32; m * n];
            backend.sgemm(m, n, k, &a, &b, &mut gpu).unwrap();
            assert_approx_eq(&format!("sgemm_batch_{}", batch), &cpu, &gpu, TOL);
        }
    }

    #[test]
    fn test_sgemm_at() {
        let backend = require_gpu!();
        let (m, n, k) = (16, 32, 64);
        let a = make_data(k * m, 3); // stored as (k x m)
        let b = make_data(k * n, 4);
        let cpu = cpu_sgemm_at(m, n, k, &a, &b);
        let mut gpu = vec![0.0f32; m * n];
        backend.sgemm_at(m, n, k, &a, &b, &mut gpu).unwrap();
        assert_approx_eq("sgemm_at", &cpu, &gpu, TOL);
    }

    #[test]
    fn test_sgemm_bt() {
        let backend = require_gpu!();
        let (m, n, k) = (16, 32, 64);
        let a = make_data(m * k, 5);
        let b = make_data(n * k, 6); // stored as (n x k)
        let cpu = cpu_sgemm_bt(m, n, k, &a, &b);
        let mut gpu = vec![0.0f32; m * n];
        backend.sgemm_bt(m, n, k, &a, &b, &mut gpu).unwrap();
        assert_approx_eq("sgemm_bt", &cpu, &gpu, TOL);
    }
}

mod activation_tests {
    use super::*;

    /// Verifies the GPU backend's ReLU implementation produces results that match the CPU reference for several input sizes.
    ///
    /// The test compares element-wise ReLU outputs between the CPU reference and the GPU backend within the relative tolerance `TOL`.
    ///
    /// # Examples
    ///
    /// ```
    /// // In test context where `backend` (from `require_gpu!()`), `make_data`, `cpu_relu`, and `assert_approx_eq` are available:
    /// let data = make_data(128, 0);
    /// let cpu = cpu_relu(&data);
    /// let mut gpu = data.clone();
    /// backend.relu(&mut gpu).unwrap();
    /// assert_approx_eq("relu_example", &cpu, &gpu, TOL);
    /// ```
    #[test]
    fn test_relu_equivalence() {
        let backend = require_gpu!();
        for &size in &[1, 16, 64, 256] {
            let n = size * 128;
            let data = make_data(n, 100);
            let cpu = cpu_relu(&data);
            let mut gpu = data.clone();
            backend.relu(&mut gpu).unwrap();
            assert_approx_eq(&format!("relu_{}", size), &cpu, &gpu, TOL);
        }
    }

    /// Verifies the GPU ReLU backward implementation produces the same per-element gradients as a CPU reference.
    ///
    /// The CPU reference passes the upstream gradient through where the input is greater than 0, otherwise yields 0. The test runs the backend's `relu_backward` and compares results with `assert_approx_eq` using `TOL`.
    ///
    /// # Examples
    ///
    /// ```
    /// // Prepare deterministic inputs
    /// let n = 1024;
    /// let input = make_data(n, 200);
    /// let grad_output = make_data(n, 201);
    ///
    /// // CPU reference: grad_input = grad_output where input > 0, else 0
    /// let cpu: Vec<f32> = input
    ///     .iter()
    ///     .zip(grad_output.iter())
    ///     .map(|(&i, &g)| if i > 0.0 { g } else { 0.0 })
    ///     .collect();
    ///
    /// // Run backend and compare
    /// let mut gpu = vec![0.0f32; n];
    /// backend.relu_backward(&input, &grad_output, &mut gpu).unwrap();
    /// assert_approx_eq("relu_backward", &cpu, &gpu, TOL);
    /// ```
    #[test]
    fn test_relu_backward_equivalence() {
        let backend = require_gpu!();
        let n = 1024;
        let input = make_data(n, 200);
        let grad_output = make_data(n, 201);
        // CPU reference
        let cpu: Vec<f32> = input
            .iter()
            .zip(grad_output.iter())
            .map(|(&i, &g)| if i > 0.0 { g } else { 0.0 })
            .collect();
        let mut gpu = vec![0.0f32; n];
        backend
            .relu_backward(&input, &grad_output, &mut gpu)
            .unwrap();
        assert_approx_eq("relu_backward", &cpu, &gpu, TOL);
    }

    #[test]
    fn test_sigmoid_equivalence() {
        let backend = require_gpu!();
        for &size in &[1, 16, 64, 256] {
            let n = size * 128;
            let data = make_data(n, 300);
            let cpu = cpu_sigmoid(&data);
            let mut gpu = data.clone();
            backend.sigmoid(&mut gpu).unwrap();
            assert_approx_eq(&format!("sigmoid_{}", size), &cpu, &gpu, TOL);
        }
    }

    /// Verifies that the GPU implementation of sigmoid backward produces results equivalent to a CPU reference within the test tolerance.
    ///
    /// The test builds deterministic sigmoid outputs and grad outputs, computes the CPU reference grad_input as
    /// `grad_output * s * (1 - s)`, runs `sigmoid_backward` on the GPU backend, and compares element-wise with a relative tolerance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // prepare inputs
    /// let sigmoid_output: Vec<f32> = vec![0.1, 0.5, 0.9];
    /// let grad_output: Vec<f32> = vec![0.2, 0.3, 0.4];
    /// let mut out = vec![0.0f32; sigmoid_output.len()];
    /// // call backend (assumes `backend: &dyn GpuBackend` is available)
    /// backend.sigmoid_backward(&sigmoid_output, &grad_output, &mut out).unwrap();
    /// ```
    #[test]
    fn test_sigmoid_backward_equivalence() {
        let backend = require_gpu!();
        let n = 1024;
        // sigmoid_output values should be in (0, 1)
        let raw = make_data(n, 400);
        let sigmoid_output: Vec<f32> = raw.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let grad_output = make_data(n, 401);
        // CPU: grad_input = grad_output * s * (1 - s)
        let cpu: Vec<f32> = sigmoid_output
            .iter()
            .zip(grad_output.iter())
            .map(|(&s, &g)| g * s * (1.0 - s))
            .collect();
        let mut gpu = vec![0.0f32; n];
        backend
            .sigmoid_backward(&sigmoid_output, &grad_output, &mut gpu)
            .unwrap();
        assert_approx_eq("sigmoid_backward", &cpu, &gpu, TOL);
    }
}

mod bias_tests {
    use super::*;

    /// Verifies that the GPU backend's `add_bias` produces the same results as the CPU reference across several batch sizes.
    ///
    /// Compares CPU and GPU outputs for batch sizes 1, 16, 64, and 256 with n = 128 using deterministic test data and the relative tolerance `TOL`.
    ///
    /// # Examples
    ///
    /// ```
    /// // Requires the surrounding test utilities: `require_gpu!`, `make_data`, `cpu_add_bias`, `assert_approx_eq`, and `TOL`.
    /// let backend = require_gpu!();
    /// let batch = 16;
    /// let n = 128;
    /// let data = make_data(batch * n, 0);
    /// let bias = make_data(n, 1);
    /// let cpu = cpu_add_bias(&data, &bias, batch, n);
    /// let mut gpu = data.clone();
    /// backend.add_bias(&mut gpu, &bias, batch, n).unwrap();
    /// assert_approx_eq("add_bias_example", &cpu, &gpu, TOL);
    /// ```
    #[test]
    fn test_add_bias_equivalence() {
        let backend = require_gpu!();
        for &batch in &[1, 16, 64, 256] {
            let n = 128;
            let data = make_data(batch * n, 500);
            let bias = make_data(n, 501);
            let cpu = cpu_add_bias(&data, &bias, batch, n);
            let mut gpu = data.clone();
            backend.add_bias(&mut gpu, &bias, batch, n).unwrap();
            assert_approx_eq(&format!("add_bias_{}", batch), &cpu, &gpu, TOL);
        }
    }

    #[test]
    fn test_sum_rows_equivalence() {
        let backend = require_gpu!();
        for &batch in &[1, 16, 64, 256] {
            let n = 128;
            let data = make_data(batch * n, 600);
            let cpu = cpu_sum_rows(&data, batch, n);
            let mut gpu = vec![0.0f32; n];
            backend.sum_rows(&data, &mut gpu, batch, n).unwrap();
            assert_approx_eq(&format!("sum_rows_{}", batch), &cpu, &gpu, TOL);
        }
    }
}

mod conv2d_tests {
    use super::*;

    #[test]
    fn test_conv2d_forward_equivalence() {
        let backend = require_gpu!();
        // Test various batch sizes with a simple conv config
        let (in_ch, out_ch, h, w, kh, kw, stride, padding) = (1, 4, 8, 8, 3, 3, 1, 1);
        for &batch in &[1, 16] {
            let input = make_data(batch * in_ch * h * w, 700);
            let filters = make_data(out_ch * in_ch * kh * kw, 701);
            let bias = make_data(out_ch, 702);
            let cpu = cpu_conv2d_forward(
                &input, &filters, &bias, batch, in_ch, out_ch, h, w, kh, kw, stride, padding,
            );
            let mut gpu = vec![0.0f32; cpu.len()];
            backend
                .conv2d_forward(
                    &input, &filters, &bias, &mut gpu, batch, in_ch, out_ch, h, w, kh, kw, stride,
                    padding,
                )
                .unwrap();
            assert_approx_eq(&format!("conv2d_fwd_b{}", batch), &cpu, &gpu, TOL);
        }
    }

    /// Verifies that the GPU implementation of 2D convolution produces the same outputs as the CPU reference for a multichannel input.
    ///
    /// This test constructs a small multichannel scenario (batch size 4, 3 input channels, 8 output channels,
    /// 16×16 spatial size, 3×3 kernels, stride 1, padding 1), computes a CPU reference via `cpu_conv2d_forward`,
    /// runs the backend's `conv2d_forward`, and asserts element-wise numerical agreement within tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// // Constructs inputs, filters, and bias, computes CPU reference, then runs GPU conv2d and compares.
    /// let backend = require_gpu!();
    /// let (batch, in_ch, out_ch, h, w, kh, kw, stride, padding) = (4, 3, 8, 16, 16, 3, 3, 1, 1);
    /// let input = make_data(batch * in_ch * h * w, 800);
    /// let filters = make_data(out_ch * in_ch * kh * kw, 801);
    /// let bias = make_data(out_ch, 802);
    /// let cpu = cpu_conv2d_forward(
    ///     &input, &filters, &bias, batch, in_ch, out_ch, h, w, kh, kw, stride, padding,
    /// );
    /// let mut gpu = vec![0.0f32; cpu.len()];
    /// backend
    ///     .conv2d_forward(
    ///         &input, &filters, &bias, &mut gpu, batch, in_ch, out_ch, h, w, kh, kw, stride,
    ///         padding,
    ///     )
    ///     .unwrap();
    /// assert_approx_eq("conv2d_fwd_multichannel", &cpu, &gpu, TOL);
    /// ```
    #[test]
    fn test_conv2d_forward_multichannel() {
        let backend = require_gpu!();
        let (batch, in_ch, out_ch, h, w, kh, kw, stride, padding) = (4, 3, 8, 16, 16, 3, 3, 1, 1);
        let input = make_data(batch * in_ch * h * w, 800);
        let filters = make_data(out_ch * in_ch * kh * kw, 801);
        let bias = make_data(out_ch, 802);
        let cpu = cpu_conv2d_forward(
            &input, &filters, &bias, batch, in_ch, out_ch, h, w, kh, kw, stride, padding,
        );
        let mut gpu = vec![0.0f32; cpu.len()];
        backend
            .conv2d_forward(
                &input, &filters, &bias, &mut gpu, batch, in_ch, out_ch, h, w, kh, kw, stride,
                padding,
            )
            .unwrap();
        assert_approx_eq("conv2d_fwd_multichannel", &cpu, &gpu, TOL);
    }

    #[test]
    fn test_conv2d_backward_equivalence() {
        let backend = require_gpu!();
        let (batch, in_ch, out_ch, h, w, kh, kw, stride, padding) = (2, 1, 4, 8, 8, 3, 3, 1, 1);
        let out_h = (h + 2 * padding - kh) / stride + 1;
        let out_w = (w + 2 * padding - kw) / stride + 1;

        let input = make_data(batch * in_ch * h * w, 900);
        let filters = make_data(out_ch * in_ch * kh * kw, 901);
        let grad_output = make_data(batch * out_ch * out_h * out_w, 902);

        // GPU backward
        let mut grad_input = vec![0.0f32; batch * in_ch * h * w];
        let mut grad_filters = vec![0.0f32; out_ch * in_ch * kh * kw];
        let mut grad_bias = vec![0.0f32; out_ch];

        backend
            .conv2d_backward(
                &input,
                &filters,
                &grad_output,
                &mut grad_input,
                &mut grad_filters,
                &mut grad_bias,
                batch,
                in_ch,
                out_ch,
                h,
                w,
                kh,
                kw,
                stride,
                padding,
            )
            .unwrap();

        // CPU reference for grad_bias: sum grad_output over batch and spatial dims
        let mut cpu_grad_bias = vec![0.0f32; out_ch];
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx =
                            b * (out_ch * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                        cpu_grad_bias[oc] += grad_output[idx];
                    }
                }
            }
        }
        assert_approx_eq("conv2d_bwd_grad_bias", &cpu_grad_bias, &grad_bias, TOL);

        // CPU reference for grad_filters
        let mut cpu_grad_filters = vec![0.0f32; out_ch * in_ch * kh * kw];
        for b in 0..batch {
            for oc in 0..out_ch {
                for ic in 0..in_ch {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let mut sum = 0.0f32;
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let ih = (oh * stride + fh) as isize - padding as isize;
                                    let iw = (ow * stride + fw) as isize - padding as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let in_idx = b * (in_ch * h * w)
                                            + ic * (h * w)
                                            + ih as usize * w
                                            + iw as usize;
                                        let go_idx = b * (out_ch * out_h * out_w)
                                            + oc * (out_h * out_w)
                                            + oh * out_w
                                            + ow;
                                        sum += input[in_idx] * grad_output[go_idx];
                                    }
                                }
                            }
                            let f_idx = oc * (in_ch * kh * kw) + ic * (kh * kw) + fh * kw + fw;
                            cpu_grad_filters[f_idx] += sum;
                        }
                    }
                }
            }
        }
        assert_approx_eq(
            "conv2d_bwd_grad_filters",
            &cpu_grad_filters,
            &grad_filters,
            TOL,
        );

        // CPU reference for grad_input
        let mut cpu_grad_input = vec![0.0f32; batch * in_ch * h * w];
        for b in 0..batch {
            for ic in 0..in_ch {
                for ih in 0..h {
                    for iw in 0..w {
                        let mut sum = 0.0f32;
                        for oc in 0..out_ch {
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let oh_val = ih + padding - fh;
                                    let ow_val = iw + padding - fw;
                                    if oh_val % stride == 0 && ow_val % stride == 0 {
                                        let oh = oh_val / stride;
                                        let ow = ow_val / stride;
                                        if oh < out_h && ow < out_w {
                                            let go_idx = b * (out_ch * out_h * out_w)
                                                + oc * (out_h * out_w)
                                                + oh * out_w
                                                + ow;
                                            let f_idx = oc * (in_ch * kh * kw)
                                                + ic * (kh * kw)
                                                + fh * kw
                                                + fw;
                                            sum += grad_output[go_idx] * filters[f_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let in_idx = b * (in_ch * h * w) + ic * (h * w) + ih * w + iw;
                        cpu_grad_input[in_idx] = sum;
                    }
                }
            }
        }
        assert_approx_eq("conv2d_bwd_grad_input", &cpu_grad_input, &grad_input, TOL);
    }
}
