//! Pure Rust activation functions for WebAssembly
//!
//! This module provides activation functions suitable for WASM environments.
//! These are BLAS-free and optimized for inference use cases.

/// Computes the logistic sigmoid of an input.
///
/// Returns the value 1 / (1 + exp(-x)), which maps any real-valued input to the range (0, 1).
///
/// # Examples
///
/// ```
/// use mnist_wasm::activations::sigmoid;
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
/// use mnist_wasm::activations::sigmoid_derivative;
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
/// use mnist_wasm::activations::relu_inplace;
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
/// use mnist_wasm::activations::leaky_relu;
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
/// use mnist_wasm::activations::leaky_relu_derivative;
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
/// use mnist_wasm::activations::elu;
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
/// use mnist_wasm::activations::elu_derivative;
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
/// use mnist_wasm::activations::gelu;
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
/// use mnist_wasm::activations::gelu_derivative;
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

    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x);

    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * d_inner
}

/// Computes the Swish (SiLU) activation function.
///
/// Returns `x * sigmoid(x)`, which is a smooth, non-monotonic function.
/// Also known as SiLU (Sigmoid Linear Unit).
///
/// # Examples
///
/// ```
/// use mnist_wasm::activations::swish;
/// let result = swish(0.0);
/// assert!((result - 0.0).abs() < 1e-6);
/// let result_pos = swish(1.0);
/// assert!(result_pos > 0.0);
/// ```
pub fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Computes the derivative of Swish with respect to the input `x`.
///
/// Returns `sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`.
///
/// # Examples
///
/// ```
/// use mnist_wasm::activations::swish_derivative;
/// let d = swish_derivative(0.0);
/// assert!(d > 0.0);
/// ```
pub fn swish_derivative(x: f32) -> f32 {
    let sig = sigmoid(x);
    sig + x * sig * (1.0 - sig)
}

/// Applies the hyperbolic tangent activation function.
///
/// Returns `tanh(x)`, which maps inputs to the range (-1, 1).
///
/// # Examples
///
/// ```
/// use mnist_wasm::activations::tanh_activation;
/// let result = tanh_activation(0.0);
/// assert!((result - 0.0).abs() < 1e-6);
/// ```
pub fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

/// Applies softmax activation across each row of a matrix in place.
///
/// Each row is independently normalized so that its elements sum to 1.0 and all values
/// are in the range (0, 1). Uses the numerically stable formulation with max subtraction.
///
/// # Parameters
///
/// * `data` - Matrix data in row-major format (modified in place)
/// * `rows` - Number of rows in the matrix
/// * `cols` - Number of columns in the matrix
///
/// # Examples
///
/// ```
/// use mnist_wasm::activations::softmax_rows;
///
/// let mut data = vec![1.0, 2.0, 3.0,  // row 0
///                     4.0, 5.0, 6.0]; // row 1
/// softmax_rows(&mut data, 2, 3);
///
/// // Each row should sum to approximately 1.0
/// let row0_sum: f32 = data[0..3].iter().sum();
/// let row1_sum: f32 = data[3..6].iter().sum();
/// assert!((row0_sum - 1.0).abs() < 1e-5);
/// assert!((row1_sum - 1.0).abs() < 1e-5);
/// ```
pub fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    assert!(
        data.len() >= rows * cols,
        "Data buffer too small for given dimensions"
    );

    for row in data.chunks_exact_mut(cols).take(rows) {
        // Find max for numerical stability
        let max = row
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Compute exp(x - max) for each element
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }

        // Normalize by sum
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert!((sigmoid_derivative(0.5) - 0.25).abs() < 1e-6);
        assert!((sigmoid_derivative(0.0) - 0.0).abs() < 1e-6);
        assert!((sigmoid_derivative(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_inplace() {
        let mut data = vec![-1.0, 0.0, 1.0, -2.5, 3.5];
        relu_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.5]);
    }

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(2.0, 0.01), 2.0);
        assert!((leaky_relu(-2.0, 0.01) - (-0.02)).abs() < 1e-6);
        assert_eq!(leaky_relu(0.0, 0.01), 0.0);
    }

    #[test]
    fn test_leaky_relu_derivative() {
        assert_eq!(leaky_relu_derivative(2.0, 0.01), 1.0);
        assert_eq!(leaky_relu_derivative(-2.0, 0.01), 0.01);
    }

    #[test]
    fn test_elu() {
        assert_eq!(elu(2.0, 1.0), 2.0);
        assert_eq!(elu(0.0, 1.0), 0.0);
        assert!(elu(-1.0, 1.0) < 0.0);
        assert!(elu(-1.0, 1.0) > -1.0);
    }

    #[test]
    fn test_elu_derivative() {
        assert_eq!(elu_derivative(2.0, 1.0), 1.0);
        assert_eq!(elu_derivative(0.0, 1.0), 1.0);
        assert!(elu_derivative(-1.0, 1.0) < 1.0);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        let result_pos = gelu(1.0);
        assert!(result_pos > 0.0 && result_pos < 1.0);
        let result_neg = gelu(-1.0);
        assert!(result_neg < 0.0 && result_neg > -1.0);
    }

    #[test]
    fn test_gelu_derivative() {
        let d = gelu_derivative(0.0);
        assert!(d > 0.0);
        let d_pos = gelu_derivative(1.0);
        assert!(d_pos > 0.0);
    }

    #[test]
    fn test_swish() {
        assert!((swish(0.0) - 0.0).abs() < 1e-6);
        assert!(swish(1.0) > 0.0);
        assert!(swish(-1.0) < 0.0);
    }

    #[test]
    fn test_swish_derivative() {
        let d = swish_derivative(0.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_tanh_activation() {
        assert!((tanh_activation(0.0) - 0.0).abs() < 1e-6);
        assert!(tanh_activation(10.0) > 0.99);
        assert!(tanh_activation(-10.0) < -0.99);
    }

    #[test]
    fn test_softmax_rows_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        softmax_rows(&mut data, 2, 3);

        // Each row should sum to approximately 1.0
        let row0_sum: f32 = data[0..3].iter().sum();
        let row1_sum: f32 = data[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);

        // All values should be positive
        for value in &data {
            assert!(*value > 0.0);
        }
    }

    #[test]
    fn test_softmax_rows_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_rows(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_rows_numerical_stability() {
        // Test with large values that could cause overflow
        let mut data = vec![1000.0, 1001.0, 1002.0];
        softmax_rows(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data.iter().all(|&x| x.is_finite()));
    }
}
