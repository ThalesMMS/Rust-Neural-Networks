//! Activation functions for neural networks
//!
//! This module provides common activation functions used across different models:
//! - Sigmoid (f64 version for XOR/simple networks)
//! - ReLU (f32 version for MNIST networks)
//! - Softmax (f32 version for output layers)

/// Sigmoid activation function (f64 version).
///
/// Returns the sigmoid of the input: 1 / (1 + exp(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative assuming x = sigmoid(z).
///
/// Returns the derivative: x * (1 - x)
pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

/// ReLU activation function (f32 version) applied in-place.
///
/// Sets all negative values to 0.0, keeps positive values unchanged.
pub fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// Softmax activation function (f32 version) applied row-wise.
///
/// Converts logits to probabilities for each row. Uses the max-subtraction
/// trick for numerical stability to avoid overflow with large values.
///
/// # Arguments
/// * `outputs` - Flat array containing row-major matrix data
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
pub fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
    if cols == 0 {
        return; // Or panic, but return is safe for empty columns
    }
    assert_eq!(outputs.len(), rows * cols, "outputs length mismatch in softmax_rows");

    for row in outputs.chunks_exact_mut(cols).take(rows) {
        let mut max_value = row[0];
        for &value in row.iter().skip(1) {
            if value > max_value {
                max_value = value;
            }
        }

        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max_value).exp();
            sum += *value;
        }

        let inv_sum = 1.0f32 / sum;
        for value in row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;
    const EPSILON_F32: f32 = 1e-6;

    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!((result - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_sigmoid_positive() {
        let result = sigmoid(2.0);
        assert!(result > 0.5 && result < 1.0);
    }

    #[test]
    fn test_sigmoid_negative() {
        let result = sigmoid(-2.0);
        assert!(result > 0.0 && result < 0.5);
    }

    #[test]
    fn test_sigmoid_derivative_at_half() {
        let result = sigmoid_derivative(0.5);
        assert!((result - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_relu_negative() {
        let mut data = vec![-1.0f32];
        relu_inplace(&mut data);
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_relu_positive() {
        let mut data = vec![5.0f32];
        relu_inplace(&mut data);
        assert_eq!(data[0], 5.0);
    }

    #[test]
    fn test_relu_mixed() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax_single_row_sum() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < EPSILON_F32);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let mut data = vec![1.0, 1.0, 1.0];
        softmax_rows(&mut data, 1, 3);
        for &val in &data {
            assert!((val - 1.0 / 3.0).abs() < EPSILON_F32);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut data = vec![1000.0, 1001.0, 1002.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < EPSILON_F32);
        assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }
}
