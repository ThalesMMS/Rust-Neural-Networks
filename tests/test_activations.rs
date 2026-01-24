// Tests for activation functions: sigmoid, relu, and softmax.
// These functions are copied from the main binaries for testing purposes.

use approx::assert_relative_eq;

// Sigmoid activation function (f32 version).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Sigmoid derivative assuming x = sigmoid(z).
fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

// ReLU activation function (f32 version from mnist_mlp.rs).
fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

// Softmax activation function (f32 version from mnist_mlp.rs).
fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
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

    // Sigmoid tests.
    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_positive() {
        let result = sigmoid(2.0);
        assert!(result > 0.5 && result < 1.0);
        assert_relative_eq!(result, 0.880797, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_negative() {
        let result = sigmoid(-2.0);
        assert!(result > 0.0 && result < 0.5);
        assert_relative_eq!(result, 0.1192029, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let result = sigmoid(10.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let result = sigmoid(-10.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let x = 3.0;
        let pos = sigmoid(x);
        let neg = sigmoid(-x);
        assert_relative_eq!(pos + neg, 1.0, epsilon = 1e-6);
    }

    // Sigmoid derivative tests.
    #[test]
    fn test_sigmoid_derivative_at_half() {
        let result = sigmoid_derivative(0.5);
        assert_relative_eq!(result, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_derivative_at_extremes() {
        assert_relative_eq!(sigmoid_derivative(0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(sigmoid_derivative(1.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_derivative_range() {
        for i in 0..100 {
            let x = i as f32 / 100.0;
            let deriv = sigmoid_derivative(x);
            assert!((0.0..=0.25).contains(&deriv));
        }
    }

    // ReLU tests.
    #[test]
    fn test_relu_negative() {
        let mut data = vec![-1.0f32];
        relu_inplace(&mut data);
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_relu_zero() {
        let mut data = vec![0.0f32];
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
    fn test_relu_all_negative() {
        let mut data = vec![-5.0, -3.0, -1.0, -0.1];
        relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_all_positive() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = data.clone();
        relu_inplace(&mut data);
        assert_eq!(data, expected);
    }

    // Softmax tests.
    #[test]
    fn test_softmax_single_row_sum() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_single_row_values() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);
        assert!(data[0] < data[1] && data[1] < data[2]);
        for &val in &data {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_softmax_uniform_input() {
        let mut data = vec![1.0, 1.0, 1.0];
        softmax_rows(&mut data, 1, 3);
        for &val in &data {
            assert_relative_eq!(val, 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax_multiple_rows() {
        let mut data = vec![
            1.0, 2.0, 3.0, // Row 1
            4.0, 5.0, 6.0, // Row 2
        ];
        softmax_rows(&mut data, 2, 3);

        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();

        assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-6);
        assert_relative_eq!(row2_sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_large_values() {
        let mut data = vec![100.0, 200.0, 300.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut data = vec![-1.0, -2.0, -3.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(data[0] > data[1] && data[1] > data[2]);
    }

    #[test]
    fn test_softmax_zero_input() {
        let mut data = vec![0.0, 0.0, 0.0];
        softmax_rows(&mut data, 1, 3);
        for &val in &data {
            assert_relative_eq!(val, 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax_single_element() {
        let mut data = vec![5.0];
        softmax_rows(&mut data, 1, 1);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut data = vec![1000.0, 1001.0, 1002.0];
        softmax_rows(&mut data, 1, 3);
        let sum: f32 = data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
    }
}
