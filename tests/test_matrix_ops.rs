// Tests for matrix operations: GEMM (sgemm_wrapper), add_bias, and sum_rows.
// These functions are copied from mnist_mlp.rs for testing purposes.

extern crate blas_src;

use approx::assert_relative_eq;
use cblas::{sgemm, Layout, Transpose};

// GEMM wrapper from mnist_mlp.rs.
fn sgemm_wrapper(
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
    let trans_a = if transpose_a {
        Transpose::Ordinary
    } else {
        Transpose::None
    };
    let trans_b = if transpose_b {
        Transpose::Ordinary
    } else {
        Transpose::None
    };

    unsafe {
        sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

// Add bias to each row from mnist_mlp.rs.
fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

// Sum rows from mnist_mlp.rs.
fn sum_rows(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }

    for row in data.chunks_exact(cols).take(rows) {
        for (value, sum) in row.iter().zip(out.iter_mut()) {
            *sum += *value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GEMM tests.
    #[test]
    fn test_gemm_basic_multiplication() {
        // 2x3 * 3x2 = 2x2
        let a = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        let b = vec![
            1.0, 2.0, // Row 1
            3.0, 4.0, // Row 2
            5.0, 6.0, // Row 3
        ];
        let mut c = vec![0.0; 4];

        sgemm_wrapper(2, 2, 3, &a, 3, &b, 2, &mut c, 2, false, false, 1.0, 0.0);

        // Expected: [[22, 28], [49, 64]]
        assert_relative_eq!(c[0], 22.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 28.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 49.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 64.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_identity_multiplication() {
        // 2x2 * I2 = 2x2 (identity matrix)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
        let mut c = vec![0.0; 4];

        sgemm_wrapper(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 0.0);

        assert_relative_eq!(c[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 2.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 3.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_with_transpose_a() {
        // A (2x3) transposed to A^T (3x2) * B (2x2) = C (3x2)
        let a = vec![
            1.0, 2.0, 3.0, // Row 1 of A (2x3)
            4.0, 5.0, 6.0, // Row 2 of A (2x3)
        ];
        let b = vec![
            1.0, 0.0, // Row 1 of B (2x2)
            0.0, 1.0, // Row 2 of B (2x2)
        ];
        let mut c = vec![0.0; 6];

        // A^T is 3x2, B is 2x2, so C is 3x2
        sgemm_wrapper(3, 2, 2, &a, 3, &b, 2, &mut c, 2, true, false, 1.0, 0.0);

        // A^T = [[1, 4], [2, 5], [3, 6]]
        // B = [[1, 0], [0, 1]] (identity)
        // A^T * B = A^T
        assert_relative_eq!(c[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 4.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 5.0, epsilon = 1e-5);
        assert_relative_eq!(c[4], 3.0, epsilon = 1e-5);
        assert_relative_eq!(c[5], 6.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_with_transpose_b() {
        // A (2x3) * B^T (3x2) = C (2x2)
        let a = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        let b = vec![
            1.0, 3.0, 5.0, // Row 1
            2.0, 4.0, 6.0, // Row 2
        ];
        let mut c = vec![0.0; 4];

        sgemm_wrapper(2, 2, 3, &a, 3, &b, 3, &mut c, 2, false, true, 1.0, 0.0);

        // B^T = [[1, 2], [3, 4], [5, 6]]
        // Expected: [[22, 28], [49, 64]]
        assert_relative_eq!(c[0], 22.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 28.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 49.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 64.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_with_alpha() {
        // 2x2 * 2x2 with alpha = 2.0
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];

        sgemm_wrapper(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 2.0, 0.0);

        assert_relative_eq!(c[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 4.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 6.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 8.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_with_beta() {
        // Test C = alpha * A * B + beta * C
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![1.0, 1.0, 1.0, 1.0];

        sgemm_wrapper(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 1.0);

        // Result: [[1, 2], [3, 4]] + [[1, 1], [1, 1]] = [[2, 3], [4, 5]]
        assert_relative_eq!(c[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 3.0, epsilon = 1e-5);
        assert_relative_eq!(c[2], 4.0, epsilon = 1e-5);
        assert_relative_eq!(c[3], 5.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_zero_multiplication() {
        // Matrix with zeros
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![99.0; 4];

        sgemm_wrapper(2, 2, 2, &a, 2, &b, 2, &mut c, 2, false, false, 1.0, 0.0);

        for &val in &c {
            assert_relative_eq!(val, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_gemm_vector_multiplication() {
        // Matrix-vector multiplication: 2x3 * 3x1 = 2x1
        let a = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        let b = vec![1.0, 2.0, 3.0]; // Column vector
        let mut c = vec![0.0; 2];

        sgemm_wrapper(2, 1, 3, &a, 3, &b, 1, &mut c, 1, false, false, 1.0, 0.0);

        // Expected: [14, 32]
        assert_relative_eq!(c[0], 14.0, epsilon = 1e-5);
        assert_relative_eq!(c[1], 32.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gemm_single_element() {
        // 1x1 * 1x1 = 1x1
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c = vec![0.0];

        sgemm_wrapper(1, 1, 1, &a, 1, &b, 1, &mut c, 1, false, false, 1.0, 0.0);

        assert_relative_eq!(c[0], 12.0, epsilon = 1e-5);
    }

    // Add bias tests.
    #[test]
    fn test_add_bias_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        let bias = vec![0.5, 1.0, 1.5];

        add_bias(&mut data, 1, 3, &bias);

        assert_relative_eq!(data[0], 1.5, epsilon = 1e-6);
        assert_relative_eq!(data[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 4.5, epsilon = 1e-6);
    }

    #[test]
    fn test_add_bias_multiple_rows() {
        let mut data = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        let bias = vec![1.0, 2.0, 3.0];

        add_bias(&mut data, 2, 3, &bias);

        assert_relative_eq!(data[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 4.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 6.0, epsilon = 1e-6);
        assert_relative_eq!(data[3], 5.0, epsilon = 1e-6);
        assert_relative_eq!(data[4], 7.0, epsilon = 1e-6);
        assert_relative_eq!(data[5], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_add_bias_zero_bias() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        let bias = vec![0.0, 0.0];

        add_bias(&mut data, 2, 2, &bias);

        assert_eq!(data, original);
    }

    #[test]
    fn test_add_bias_negative_bias() {
        let mut data = vec![5.0, 10.0, 15.0];
        let bias = vec![-1.0, -2.0, -3.0];

        add_bias(&mut data, 1, 3, &bias);

        assert_relative_eq!(data[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 8.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_add_bias_single_element() {
        let mut data = vec![5.0];
        let bias = vec![3.0];

        add_bias(&mut data, 1, 1, &bias);

        assert_relative_eq!(data[0], 8.0, epsilon = 1e-6);
    }

    #[test]
    fn test_add_bias_large_matrix() {
        let rows = 10;
        let cols = 5;
        let mut data = vec![1.0; rows * cols];
        let bias = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        add_bias(&mut data, rows, cols, &bias);

        // Check first row
        assert_relative_eq!(data[0], 1.1, epsilon = 1e-6);
        assert_relative_eq!(data[4], 1.5, epsilon = 1e-6);

        // Check last row
        assert_relative_eq!(data[45], 1.1, epsilon = 1e-6);
        assert_relative_eq!(data[49], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_add_bias_broadcast_behavior() {
        // Test that bias is properly broadcast to all rows
        let mut data = vec![
            1.0, 2.0, // Row 1
            3.0, 4.0, // Row 2
            5.0, 6.0, // Row 3
        ];
        let bias = vec![10.0, 20.0];

        add_bias(&mut data, 3, 2, &bias);

        // Each row should have the same bias added
        assert_relative_eq!(data[0], 11.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 22.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 13.0, epsilon = 1e-6);
        assert_relative_eq!(data[3], 24.0, epsilon = 1e-6);
        assert_relative_eq!(data[4], 15.0, epsilon = 1e-6);
        assert_relative_eq!(data[5], 26.0, epsilon = 1e-6);
    }

    // Sum rows tests.
    #[test]
    fn test_sum_rows_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];

        sum_rows(&data, 1, 3, &mut out);

        assert_relative_eq!(out[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(out[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_multiple_rows() {
        let data = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        let mut out = vec![0.0; 3];

        sum_rows(&data, 2, 3, &mut out);

        assert_relative_eq!(out[0], 5.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 7.0, epsilon = 1e-6);
        assert_relative_eq!(out[2], 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_three_rows() {
        let data = vec![
            1.0, 2.0, // Row 1
            3.0, 4.0, // Row 2
            5.0, 6.0, // Row 3
        ];
        let mut out = vec![0.0; 2];

        sum_rows(&data, 3, 2, &mut out);

        assert_relative_eq!(out[0], 9.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 12.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let mut out = vec![99.0; 2];

        sum_rows(&data, 2, 2, &mut out);

        assert_relative_eq!(out[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_negative_values() {
        let data = vec![
            -1.0, -2.0, -3.0, // Row 1
            -4.0, -5.0, -6.0, // Row 2
        ];
        let mut out = vec![0.0; 3];

        sum_rows(&data, 2, 3, &mut out);

        assert_relative_eq!(out[0], -5.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], -7.0, epsilon = 1e-6);
        assert_relative_eq!(out[2], -9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_mixed_values() {
        let data = vec![
            1.0, -2.0, 3.0, // Row 1
            -4.0, 5.0, -6.0, // Row 2
        ];
        let mut out = vec![0.0; 3];

        sum_rows(&data, 2, 3, &mut out);

        assert_relative_eq!(out[0], -3.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(out[2], -3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_single_element() {
        let data = vec![42.0];
        let mut out = vec![0.0];

        sum_rows(&data, 1, 1, &mut out);

        assert_relative_eq!(out[0], 42.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_large_matrix() {
        let rows = 100;
        let cols = 10;
        let data = vec![1.0; rows * cols];
        let mut out = vec![0.0; cols];

        sum_rows(&data, rows, cols, &mut out);

        for &val in &out {
            assert_relative_eq!(val, 100.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_sum_rows_initializes_output() {
        // Test that sum_rows properly initializes the output vector
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![99.0, 99.0]; // Pre-filled with garbage

        sum_rows(&data, 2, 2, &mut out);

        assert_relative_eq!(out[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(out[1], 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_gradient_computation() {
        // Realistic use case from backpropagation
        let gradients = vec![
            0.1, 0.2, 0.3, // Sample 1 gradients
            0.4, 0.5, 0.6, // Sample 2 gradients
            0.7, 0.8, 0.9, // Sample 3 gradients
        ];
        let mut bias_grad = vec![0.0; 3];

        sum_rows(&gradients, 3, 3, &mut bias_grad);

        assert_relative_eq!(bias_grad[0], 1.2, epsilon = 1e-6);
        assert_relative_eq!(bias_grad[1], 1.5, epsilon = 1e-6);
        assert_relative_eq!(bias_grad[2], 1.8, epsilon = 1e-6);
    }
}
