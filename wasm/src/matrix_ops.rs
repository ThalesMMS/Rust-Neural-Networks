//! Pure Rust matrix operations for WebAssembly
//!
//! This module provides BLAS-free matrix operations suitable for WASM environments.
//! All operations use row-major layout and are optimized for readability and correctness
//! rather than maximum performance (which would require SIMD and other low-level optimizations).

/// Performs a single-precision general matrix-matrix multiplication.
///
/// Computes: C := alpha * op(A) * op(B) + beta * C
/// where op(X) is either the matrix X or its transpose.
///
/// All matrices use row-major layout:
/// - A: m × k (or k × m if transposed)
/// - B: k × n (or n × k if transposed)
/// - C: m × n
///
/// # Parameters
///
/// * `m` - Number of rows in op(A) and C
/// * `n` - Number of columns in op(B) and C
/// * `k` - Number of columns in op(A) and rows in op(B)
/// * `a` - Input matrix A in row-major format
/// * `lda` - Leading dimension of A (number of columns in the stored matrix)
/// * `b` - Input matrix B in row-major format
/// * `ldb` - Leading dimension of B (number of columns in the stored matrix)
/// * `c` - Output matrix C in row-major format (modified in place)
/// * `ldc` - Leading dimension of C (number of columns)
/// * `transpose_a` - If true, use A^T instead of A
/// * `transpose_b` - If true, use B^T instead of B
/// * `alpha` - Scalar multiplier for A*B
/// * `beta` - Scalar multiplier for C (use 0.0 to overwrite C)
///
/// # Examples
///
/// ```
/// use mnist_wasm::matrix_ops::matrix_multiply;
///
/// // Multiply 2x2 matrices: C = A * B
/// let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
/// let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
/// let mut c = vec![0.0; 4];
///
/// matrix_multiply(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 0.0);
///
/// // Expected: [[19,22],[43,50]]
/// assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn matrix_multiply(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
) {
    // Validate dimensions
    assert!(m > 0 && n > 0 && k > 0, "Matrix dimensions must be positive");
    assert!(
        lda >= if transpose_a { m } else { k },
        "Leading dimension lda too small"
    );
    assert!(
        ldb >= if transpose_b { k } else { n },
        "Leading dimension ldb too small"
    );
    assert!(ldc >= n, "Leading dimension ldc too small");

    // Scale existing C values by beta
    if beta == 0.0 {
        // Overwrite C with zeros
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] = 0.0;
            }
        }
    } else if beta != 1.0 {
        // Scale C by beta
        for i in 0..m {
            for j in 0..n {
                c[i * ldc + j] *= beta;
            }
        }
    }

    // Compute C += alpha * op(A) * op(B)
    // This is a naive O(m*n*k) implementation
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                // Get A[i, p] with optional transpose
                let a_val = if transpose_a {
                    a[p * lda + i]
                } else {
                    a[i * lda + p]
                };

                // Get B[p, j] with optional transpose
                let b_val = if transpose_b {
                    b[j * ldb + p]
                } else {
                    b[p * ldb + j]
                };

                sum += a_val * b_val;
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}

/// Adds a bias vector to each row of a row-major matrix in place.
///
/// The matrix `data` has `rows` rows and `cols` columns in row-major order.
/// Each element of `bias` is added to the corresponding column of every row.
///
/// # Parameters
///
/// * `data` - Matrix data in row-major format (modified in place)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `bias` - Bias vector with length equal to `cols`
///
/// # Panics
///
/// Panics if `bias.len() != cols` or if `data.len() < rows * cols`.
///
/// # Examples
///
/// ```
/// use mnist_wasm::matrix_ops::add_bias;
///
/// let mut data = vec![0.0, 1.0, 2.0,    // row 0
///                     3.0, 4.0, 5.0];   // row 1
/// let bias = vec![1.0, 10.0, 100.0];
///
/// add_bias(&mut data, 2, 3, &bias);
///
/// assert_eq!(data, vec![1.0, 11.0, 102.0, 4.0, 14.0, 105.0]);
/// ```
pub fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    assert_eq!(
        bias.len(),
        cols,
        "Bias length must equal number of columns"
    );
    assert!(
        data.len() >= rows * cols,
        "Data buffer too small for given dimensions"
    );

    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

/// Sums each column of a row-major matrix.
///
/// Computes column-wise sums and stores them in `out`.
/// The input `data` is interpreted as a matrix with `rows` rows and `cols` columns.
///
/// # Parameters
///
/// * `data` - Input matrix in row-major format
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
/// * `out` - Output buffer for column sums (must have length >= `cols`)
///
/// # Panics
///
/// Panics if `out.len() < cols` or if `data.len() < rows * cols`.
///
/// # Examples
///
/// ```
/// use mnist_wasm::matrix_ops::sum_columns;
///
/// let data = vec![
///     1.0, 2.0, 3.0,  // row 0
///     4.0, 5.0, 6.0,  // row 1
/// ];
/// let mut out = vec![0.0; 3];
///
/// sum_columns(&data, 2, 3, &mut out);
///
/// assert_eq!(out, vec![5.0, 7.0, 9.0]);
/// ```
pub fn sum_columns(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    assert!(
        out.len() >= cols,
        "Output buffer too small for column sums"
    );
    assert!(
        data.len() >= rows * cols,
        "Data buffer too small for given dimensions"
    );

    // Initialize output to zero
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }

    // Sum each column
    for row in data.chunks_exact(cols).take(rows) {
        for (value, sum) in row.iter().zip(out.iter_mut()) {
            *sum += *value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiply_basic() {
        // Test basic 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let mut c = vec![0.0; 4];

        matrix_multiply(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 0.0);

        // Expected: [[19,22],[43,50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        // Test multiplication with identity matrix
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let identity = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
        let mut c = vec![0.0; 4];

        matrix_multiply(
            2,
            2,
            2,
            &a,
            2,
            &identity,
            2,
            &mut c,
            2,
            false,
            false,
            1.0,
            0.0,
        );

        // A * I = A
        assert_eq!(c, a);
    }

    #[test]
    fn test_matrix_multiply_transpose_a() {
        // Test with transposed A
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let mut c = vec![0.0; 4];

        // Multiply A^T * B where A^T = [[1,3],[2,4]]
        matrix_multiply(2, 2, 2, &a, 2, &b, 2, &mut c, 2, true, false, 1.0, 0.0);

        // Expected: [[26,30],[38,44]]
        assert_eq!(c, vec![26.0, 30.0, 38.0, 44.0]);
    }

    #[test]
    fn test_matrix_multiply_transpose_b() {
        // Test with transposed B
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
        let mut c = vec![0.0; 4];

        // Multiply A * B^T where B^T = [[5,7],[6,8]]
        matrix_multiply(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, true, 1.0, 0.0);

        // Expected: [[17,23],[39,53]]
        assert_eq!(c, vec![17.0, 23.0, 39.0, 53.0]);
    }

    #[test]
    fn test_matrix_multiply_non_square() {
        // Test non-square matrices: (2x3) * (3x2) = (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [[7,8],[9,10],[11,12]]
        let mut c = vec![0.0; 4];

        matrix_multiply(2, 2, 3, &a, 3, &b, 2, &mut c, 2, false, false, 1.0, 0.0);

        // Expected: [[58,64],[139,154]]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matrix_multiply_alpha_beta() {
        // Test alpha and beta scaling
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        // C = 2.0 * (A * B) + 3.0 * C
        matrix_multiply(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 2.0, 3.0);

        // A * B = [[1,2],[3,4]]
        // 2.0 * [[1,2],[3,4]] + 3.0 * [[1,1],[1,1]] = [[5,7],[9,11]]
        assert_eq!(c, vec![5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_add_bias_basic() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let bias = vec![1.0, 10.0, 100.0];

        add_bias(&mut data, 2, 3, &bias);

        assert_eq!(data, vec![1.0, 11.0, 102.0, 4.0, 14.0, 105.0]);
    }

    #[test]
    fn test_add_bias_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        let bias = vec![0.5, 0.5, 0.5];

        add_bias(&mut data, 1, 3, &bias);

        assert_eq!(data, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_sum_columns_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 3];

        sum_columns(&data, 2, 3, &mut out);

        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_columns_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];

        sum_columns(&data, 1, 3, &mut out);

        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sum_columns_zeros() {
        let data = vec![0.0; 6];
        let mut out = vec![1.0; 3];

        sum_columns(&data, 2, 3, &mut out);

        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }
}
