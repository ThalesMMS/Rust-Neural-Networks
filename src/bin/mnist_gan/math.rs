use super::*;

/// Computes the logistic sigmoid function, mapping any real value to (0, 1).
///
/// # Returns
///
/// `y` in (0.0, 1.0) equal to `1.0 / (1.0 + (-x).exp())`.
///
/// # Examples
///
/// ```
/// let s = sigmoid(0.0);
/// assert!((s - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid activation in-place to every element of `v`.
#[inline]
pub(crate) fn sigmoid_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = sigmoid(*x);
    }
}

/// Applies the hyperbolic tangent (tanh) to each element of the slice in place.
///
/// # Examples
///
/// ```
/// let mut v = [-1.0f32, 0.0, 1.0];
/// tanh_inplace(&mut v);
/// assert!(v[0] < 0.0 && v[1] == 0.0 && v[2] > 0.0);
/// ```
#[inline]
pub(crate) fn tanh_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

/// Applies the leaky ReLU activation to each element of the slice in place.
///
/// Negative elements are scaled by `alpha`; non-negative elements are left unchanged.
///
/// # Parameters
///
/// - `v`: slice of values to transform in place.
/// - `alpha`: slope applied to values less than zero (e.g., `0.01`).
///
/// # Examples
///
/// ```
/// let mut data = [1.0_f32, -2.0, 0.0, -0.5];
/// leaky_relu_inplace(&mut data, 0.1);
/// assert_eq!(data, [1.0, -0.2, 0.0, -0.05]);
/// ```
#[inline]
pub(crate) fn leaky_relu_inplace(v: &mut [f32], alpha: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= alpha;
        }
    }
}

/// Computes the binary cross-entropy loss for a single prediction/target pair.
///
/// The loss is
/// L = −[target · ln(pred + ε) + (1 − target) · ln(1 − pred + ε)]
/// where ε = 1e-7 is added to prevent taking ln(0).
///
/// `pred` is interpreted as the predicted probability (typically in [0, 1]) and
/// `target` as the binary label (0.0 or 1.0).
///
/// # Examples
///
/// ```
/// let loss = bce_loss(0.9_f32, 1.0_f32);
/// // expected ≈ -ln(0.9) ≈ 0.1053605
/// assert!((loss - 0.1053605).abs() < 1e-4);
/// ```
pub(crate) fn bce_loss(pred: f32, target: f32) -> f32 {
    -(target * (pred + 1e-7).ln() + (1.0 - target) * (1.0 - pred + 1e-7).ln())
}

// ============================================================================
// BLAS Helper
// ============================================================================

/// Performs row-major single-precision matrix multiply-add: C ← alpha · A · op(B) + beta · C.
///
/// Matrices are stored in row-major order. A is m×k and C is m×n. The contents of `c` are
/// scaled by `beta` before adding the product; if `beta` is 0.0 the previous contents of `c` are overwritten.
///
/// # Parameters
///
/// * `m` — number of rows in A and C.
/// * `n` — number of columns in B (after op) and C.
/// * `k` — inner dimension (columns of A, rows of B when not transposed).
/// * `trans_b` — if `false`, `b` is interpreted as k×n; if `true`, `b` is interpreted as n×k and transposed before multiplication.
///
/// # Examples
///
/// ```
/// // Compute C = A · B for 2×2 matrices
/// let m = 2;
/// let n = 2;
/// let k = 2;
/// let alpha = 1.0_f32;
/// let beta = 0.0_f32;
/// let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2 row-major
/// let b = [5.0_f32, 6.0, 7.0, 8.0]; // 2×2 row-major
/// let mut c = [0.0_f32; 4];
/// sgemm_row(m, n, k, alpha, &a, &b, beta, &mut c, false);
/// // C == [[19, 22], [43, 50]]
/// assert_eq!(c, [19.0_f32, 22.0, 43.0, 50.0]);
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_row(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    trans_b: bool,
) {
    assert_eq!(
        a.len(),
        m * k,
        "sgemm_row: a length {} must equal m * k {}",
        a.len(),
        m * k
    );
    let expected_b = if trans_b { n * k } else { k * n };
    assert_eq!(
        b.len(),
        expected_b,
        "sgemm_row: b length {} must equal expected {}",
        b.len(),
        expected_b
    );
    assert_eq!(
        c.len(),
        m * n,
        "sgemm_row: c length {} must equal m * n {}",
        c.len(),
        m * n
    );

    let (trans_b_flag, ldb) = if trans_b {
        (Transpose::Ordinary, k as i32) // B stored n×k → after transpose k×n
    } else {
        (Transpose::None, n as i32) // B stored k×n
    };
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            trans_b_flag,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32, // lda = cols of A (k)
            b,
            ldb,
            beta,
            c,
            n as i32, // ldc = cols of C (n)
        );
    }
}

// ============================================================================
// Generator
// ============================================================================
