/// Computes the logistic sigmoid of an input.
///
/// Returns the value 1 / (1 + exp(-x)), which maps any real-valued input to the range (0, 1).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::sigmoid;
/// let s = sigmoid(0.0);
/// assert!((s - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Computes the derivative of the logistic sigmoid given its output `x` (i.e., `x = sigmoid(z)`).
///
/// The derivative equals `x * (1.0 - x)`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::sigmoid_derivative;
/// let d = sigmoid_derivative(0.5);
/// assert!((d - 0.25).abs() < 1e-6);
/// ```
pub fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

/// Applies the Rectified Linear Unit (ReLU) activation to each element of the slice in place.
///
/// Each element less than 0.0 is set to 0.0; other values are left unchanged.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::relu_inplace;
/// let mut v = [-1.0f32, 0.0, 2.5];
/// relu_inplace(&mut v);
/// assert_eq!(v, [0.0, 0.0, 2.5]);
/// ```
pub fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// Applies the softmax function to each row of a row-major flat matrix in place.
///
/// Each row of `outputs` is transformed from logits to probabilities that sum to 1.
///
/// The function uses the max-subtraction trick for numerical stability before exponentiation.
/// `outputs` is expected to have at least `rows * cols` elements; extra elements after that are ignored.
///
/// # Arguments
///
/// * `outputs` - Mutable flat array containing row-major matrix data.
/// * `rows` - Number of rows to process from the start of `outputs`.
/// * `cols` - Number of columns per row.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::softmax_rows;
/// let mut data = vec![1.0f32, 2.0, 3.0];
/// softmax_rows(&mut data, 1, 3);
/// let sum: f32 = data.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// ```
pub fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
    if cols == 0 {
        return; // Or panic, but return is safe for empty columns
    }
    assert!(
        outputs.len() >= rows * cols,
        "outputs length mismatch in softmax_rows: len={}, rows={}, cols={}. Expected len >= {}",
        outputs.len(),
        rows,
        cols,
        rows * cols
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6; // Changed from f64 1e-10
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
