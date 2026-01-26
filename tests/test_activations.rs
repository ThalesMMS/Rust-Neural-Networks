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
    if cols == 0 {
        return;
    }
    assert_eq!(
        outputs.len(),
        rows * cols,
        "outputs length mismatch in softmax_rows"
    );
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

// Leaky ReLU activation function.
fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

// Leaky ReLU derivative.
fn leaky_relu_derivative(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

// ELU activation function.
fn elu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

// ELU derivative.
fn elu_derivative(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha * x.exp()
    }
}

// GELU activation function.
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044715;

    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

// GELU derivative.
fn gelu_derivative(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044715;

    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    let tanh_inner = inner.tanh();
    let sech_squared = 1.0 - tanh_inner * tanh_inner;

    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x)
}

// Swish activation function.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// Swish derivative.
fn swish_derivative(x: f32) -> f32 {
    let sigmoid_x = 1.0 / (1.0 + (-x).exp());
    sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
}

// Tanh activation function.
fn tanh(x: f32) -> f32 {
    x.tanh()
}

// Tanh derivative assuming y = tanh(x).
fn tanh_derivative(y: f32) -> f32 {
    1.0 - y * y
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

    // Leaky ReLU tests.
    #[test]
    fn test_leaky_relu_zero() {
        let result = leaky_relu(0.0, 0.01);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_leaky_relu_positive() {
        let result = leaky_relu(2.0, 0.01);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_leaky_relu_negative() {
        let result = leaky_relu(-2.0, 0.01);
        assert_relative_eq!(result, -0.02, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_large_positive() {
        let result = leaky_relu(100.0, 0.01);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_leaky_relu_large_negative() {
        let result = leaky_relu(-100.0, 0.01);
        assert_relative_eq!(result, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_different_alpha() {
        let result = leaky_relu(-2.0, 0.1);
        assert_relative_eq!(result, -0.2, epsilon = 1e-6);
    }

    // Leaky ReLU derivative tests.
    #[test]
    fn test_leaky_relu_derivative_positive() {
        let result = leaky_relu_derivative(2.0, 0.01);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_leaky_relu_derivative_negative() {
        let result = leaky_relu_derivative(-2.0, 0.01);
        assert_eq!(result, 0.01);
    }

    #[test]
    fn test_leaky_relu_derivative_zero() {
        let result = leaky_relu_derivative(0.0, 0.01);
        assert_eq!(result, 0.01);
    }

    // ELU tests.
    #[test]
    fn test_elu_zero() {
        let result = elu(0.0, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_elu_positive() {
        let result = elu(2.0, 1.0);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_elu_negative() {
        let result = elu(-1.0, 1.0);
        let expected = (-1.0f32).exp() - 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_elu_large_positive() {
        let result = elu(100.0, 1.0);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_elu_large_negative() {
        let result = elu(-10.0, 1.0);
        let expected = (-10.0f32).exp() - 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-6);
        assert!(result > -1.0 && result < 0.0);
    }

    #[test]
    fn test_elu_different_alpha() {
        let result = elu(-1.0, 2.0);
        let expected = 2.0 * ((-1.0f32).exp() - 1.0);
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    // ELU derivative tests.
    #[test]
    fn test_elu_derivative_positive() {
        let result = elu_derivative(2.0, 1.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_elu_derivative_zero() {
        let result = elu_derivative(0.0, 1.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_elu_derivative_negative() {
        let result = elu_derivative(-1.0, 1.0);
        let expected = (-1.0f32).exp();
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    // GELU tests.
    #[test]
    fn test_gelu_zero() {
        let result = gelu(0.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let result = gelu(1.0);
        assert!(result > 0.0 && result < 1.0);
        assert_relative_eq!(result, 0.841192, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_negative() {
        let result = gelu(-1.0);
        assert!(result < 0.0);
        assert_relative_eq!(result, -0.158808, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_large_positive() {
        let result = gelu(10.0);
        assert_relative_eq!(result, 10.0, epsilon = 1e-4);
    }

    #[test]
    fn test_gelu_large_negative() {
        let result = gelu(-10.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_gelu_symmetry_property() {
        // GELU is not symmetric, but verify it's continuous
        let pos = gelu(2.0);
        let neg = gelu(-2.0);
        assert!(pos > 0.0);
        assert!(neg < 0.0);
        assert!(pos > -neg); // GELU is not symmetric
    }

    // GELU derivative tests.
    #[test]
    fn test_gelu_derivative_zero() {
        let result = gelu_derivative(0.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_derivative_positive() {
        let result = gelu_derivative(1.0);
        assert!(result > 0.0);
        assert_relative_eq!(result, 1.082964, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_derivative_negative() {
        let result = gelu_derivative(-1.0);
        // GELU derivative can be negative for negative inputs
        assert_relative_eq!(result, -0.082964, epsilon = 1e-5);
    }

    // Swish tests.
    #[test]
    fn test_swish_zero() {
        let result = swish(0.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_swish_positive() {
        let result = swish(2.0);
        assert!(result > 0.0);
        let expected = 2.0 / (1.0 + (-2.0f32).exp());
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_swish_negative() {
        let result = swish(-2.0);
        assert!(result < 0.0);
        let expected = -2.0 / (1.0 + (2.0f32).exp());
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_swish_large_positive() {
        let result = swish(10.0);
        // Swish approaches x for large positive values
        assert!(result > 9.99);
        assert!(result < 10.01);
    }

    #[test]
    fn test_swish_large_negative() {
        let result = swish(-10.0);
        // Swish approaches 0 for large negative values
        assert!(result.abs() < 1e-3);
    }

    #[test]
    fn test_swish_small_values() {
        let result = swish(0.5);
        let expected = 0.5 / (1.0 + (-0.5f32).exp());
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    // Swish derivative tests.
    #[test]
    fn test_swish_derivative_zero() {
        let result = swish_derivative(0.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_swish_derivative_positive() {
        let result = swish_derivative(2.0);
        assert!(result > 0.0);
        let sigmoid = 1.0 / (1.0 + (-2.0f32).exp());
        let expected = sigmoid * (1.0 + 2.0 * (1.0 - sigmoid));
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_swish_derivative_negative() {
        let result = swish_derivative(-2.0);
        let sigmoid = 1.0 / (1.0 + (2.0f32).exp());
        let expected = sigmoid * (1.0 + (-2.0) * (1.0 - sigmoid));
        assert_relative_eq!(result, expected, epsilon = 1e-6);
        // Swish derivative can be negative for negative inputs
        assert!(result < 0.1);
    }

    // Tanh tests.
    #[test]
    fn test_tanh_zero() {
        let result = tanh(0.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_positive() {
        let result = tanh(1.0);
        let expected = 1.0f32.tanh();
        assert_relative_eq!(result, expected, epsilon = 1e-6);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_tanh_negative() {
        let result = tanh(-1.0);
        let expected = (-1.0f32).tanh();
        assert_relative_eq!(result, expected, epsilon = 1e-6);
        assert!(result < 0.0 && result > -1.0);
    }

    #[test]
    fn test_tanh_large_positive() {
        let result = tanh(10.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_large_negative() {
        let result = tanh(-10.0);
        assert_relative_eq!(result, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_symmetry() {
        let x = 2.0;
        let pos = tanh(x);
        let neg = tanh(-x);
        assert_relative_eq!(pos, -neg, epsilon = 1e-6);
    }

    // Tanh derivative tests.
    #[test]
    fn test_tanh_derivative_zero() {
        let result = tanh_derivative(0.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_derivative_at_half() {
        let result = tanh_derivative(0.5);
        assert_relative_eq!(result, 0.75, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_derivative_at_extremes() {
        let result_pos = tanh_derivative(1.0);
        assert_relative_eq!(result_pos, 0.0, epsilon = 1e-6);
        let result_neg = tanh_derivative(-1.0);
        assert_relative_eq!(result_neg, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_derivative_range() {
        for i in 0..100 {
            let y = (i as f32 / 100.0) * 2.0 - 1.0; // Range from -1 to 1
            let deriv = tanh_derivative(y);
            assert!((0.0..=1.0).contains(&deriv));
        }
    }
}
