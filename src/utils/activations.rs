#[cfg(feature = "shared_activations")]
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

#[cfg(feature = "shared_activations")]
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

/// Computes the Leaky ReLU activation function.
///
/// Returns `x` if `x > 0.0`, otherwise returns `alpha * x`.
/// The default alpha is typically 0.01, allowing a small gradient when x < 0.
///
/// # Arguments
///
/// * `x` - Input value
/// * `alpha` - Slope for negative values (default 0.01)
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::leaky_relu;
/// let result = leaky_relu(2.0, 0.01);
/// assert_eq!(result, 2.0);
/// let result_neg = leaky_relu(-2.0, 0.01);
/// assert!((result_neg - (-0.02)).abs() < 1e-6);
/// ```
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Computes the derivative of Leaky ReLU with respect to the input `x`.
///
/// Returns `1.0` if `x > 0.0`, otherwise returns `alpha`.
///
/// # Arguments
///
/// * `x` - Input value
/// * `alpha` - Slope for negative values (same as used in leaky_relu)
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::leaky_relu_derivative;
/// let d_pos = leaky_relu_derivative(2.0, 0.01);
/// assert_eq!(d_pos, 1.0);
/// let d_neg = leaky_relu_derivative(-2.0, 0.01);
/// assert_eq!(d_neg, 0.01);
/// ```
pub fn leaky_relu_derivative(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

/// Computes the Exponential Linear Unit (ELU) activation function.
///
/// Returns `x` if `x > 0.0`, otherwise returns `alpha * (exp(x) - 1.0)`.
/// ELU can produce negative outputs, which helps with mean activation closer to zero.
///
/// # Arguments
///
/// * `x` - Input value
/// * `alpha` - Scale for negative values (typically 1.0)
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::elu;
/// let result = elu(2.0, 1.0);
/// assert_eq!(result, 2.0);
/// let result_neg = elu(0.0, 1.0);
/// assert_eq!(result_neg, 0.0);
/// ```
pub fn elu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

/// Computes the derivative of ELU with respect to the input `x`.
///
/// Returns `1.0` if `x > 0.0`, otherwise returns `alpha * exp(x)`.
///
/// # Arguments
///
/// * `x` - Input value
/// * `alpha` - Scale for negative values (same as used in elu)
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::elu_derivative;
/// let d_pos = elu_derivative(2.0, 1.0);
/// assert_eq!(d_pos, 1.0);
/// let d_neg = elu_derivative(0.0, 1.0);
/// assert_eq!(d_neg, 1.0);
/// ```
pub fn elu_derivative(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha * x.exp()
    }
}

/// Computes the Gaussian Error Linear Unit (GELU) activation function.
///
/// Uses the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
/// GELU is commonly used in transformer models like BERT and GPT.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::gelu;
/// let result = gelu(0.0);
/// assert!((result - 0.0).abs() < 1e-6);
/// let result_pos = gelu(1.0);
/// assert!(result_pos > 0.0 && result_pos < 1.0);
/// ```
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// Computes the derivative of GELU with respect to the input `x`.
///
/// Uses the derivative of the tanh approximation.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::gelu_derivative;
/// let d = gelu_derivative(0.0);
/// assert!(d > 0.0);
/// let d_pos = gelu_derivative(1.0);
/// assert!(d_pos > 0.0);
/// ```
pub fn gelu_derivative(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    let tanh_inner = inner.tanh();
    let sech_squared = 1.0 - tanh_inner * tanh_inner;

    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x)
}

/// Computes the Swish activation function (also known as SiLU).
///
/// Returns `x * sigmoid(x)`. Swish is a smooth, non-monotonic function
/// that has been shown to work better than ReLU in some deep networks.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::swish;
/// let result = swish(0.0);
/// assert!((result - 0.0).abs() < 1e-6);
/// let result_pos = swish(2.0);
/// assert!(result_pos > 0.0);
/// ```
pub fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Computes the derivative of Swish with respect to the input `x`.
///
/// The derivative equals `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::swish_derivative;
/// let d = swish_derivative(0.0);
/// assert!((d - 0.5).abs() < 1e-6);
/// let d_pos = swish_derivative(2.0);
/// assert!(d_pos > 0.0);
/// ```
pub fn swish_derivative(x: f32) -> f32 {
    let sigmoid_x = 1.0 / (1.0 + (-x).exp());
    sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
}

/// Computes the hyperbolic tangent (tanh) activation function.
///
/// Returns a value in the range (-1, 1) using the standard tanh implementation.
/// Tanh is a scaled and shifted version of sigmoid, with output centered at zero.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::tanh;
/// let result = tanh(0.0);
/// assert!((result - 0.0).abs() < 1e-6);
/// let result_pos = tanh(1.0);
/// assert!(result_pos > 0.0 && result_pos < 1.0);
/// ```
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Computes the derivative of tanh given its output `y` (i.e., `y = tanh(x)`).
///
/// The derivative equals `1.0 - y * y`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::activations::tanh_derivative;
/// let d = tanh_derivative(0.0);
/// assert!((d - 1.0).abs() < 1e-6);
/// let d_half = tanh_derivative(0.5);
/// assert!((d_half - 0.75).abs() < 1e-6);
/// ```
pub fn tanh_derivative(y: f32) -> f32 {
    1.0 - y * y
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

    const EPSILON_F32: f32 = 1e-6;

    #[cfg(feature = "shared_activations")]
    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[cfg(feature = "shared_activations")]
    #[test]
    fn test_sigmoid_positive() {
        let result = sigmoid(2.0);
        assert!(result > 0.5 && result < 1.0);
    }

    #[cfg(feature = "shared_activations")]
    #[test]
    fn test_sigmoid_negative() {
        let result = sigmoid(-2.0);
        assert!(result > 0.0 && result < 0.5);
    }

    #[cfg(feature = "shared_activations")]
    #[test]
    fn test_sigmoid_derivative_at_half() {
        let result = sigmoid_derivative(0.5);
        assert!((result - 0.25).abs() < 1e-6);
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

    #[test]
    fn test_leaky_relu_positive() {
        let result = leaky_relu(2.0, 0.01);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_leaky_relu_negative() {
        let result = leaky_relu(-2.0, 0.01);
        assert!((result - (-0.02)).abs() < EPSILON_F32);
    }

    #[test]
    fn test_leaky_relu_zero() {
        let result = leaky_relu(0.0, 0.01);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_elu_positive() {
        let result = elu(2.0, 1.0);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_elu_zero() {
        let result = elu(0.0, 1.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_elu_negative() {
        let result = elu(-1.0, 1.0);
        let expected = (-1.0f32).exp() - 1.0;
        assert!((result - expected).abs() < EPSILON_F32);
    }

    #[test]
    fn test_gelu_zero() {
        let result = gelu(0.0);
        assert!((result - 0.0).abs() < EPSILON_F32);
    }

    #[test]
    fn test_gelu_positive() {
        let result = gelu(1.0);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_gelu_negative() {
        let result = gelu(-1.0);
        assert!(result < 0.0);
    }

    #[test]
    fn test_swish_zero() {
        let result = swish(0.0);
        assert!((result - 0.0).abs() < EPSILON_F32);
    }

    #[test]
    fn test_swish_positive() {
        let result = swish(2.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_swish_negative() {
        let result = swish(-2.0);
        assert!(result < 0.0);
    }

    #[test]
    fn test_tanh_zero() {
        let result = tanh(0.0);
        assert!((result - 0.0).abs() < EPSILON_F32);
    }

    #[test]
    fn test_tanh_positive() {
        let result = tanh(1.0);
        let expected = 1.0f32.tanh();
        assert!((result - expected).abs() < EPSILON_F32);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_tanh_negative() {
        let result = tanh(-1.0);
        let expected = (-1.0f32).tanh();
        assert!((result - expected).abs() < EPSILON_F32);
        assert!(result < 0.0 && result > -1.0);
    }
}
