// Tests for gradient checking of activation functions using finite differences.
// These tests verify that analytical derivatives match numerical approximations.
// Following the pattern from test_gradient_checking.rs.

use approx::assert_relative_eq;

// ============================================================================
// Activation Functions and Their Derivatives (f32 versions)
// ============================================================================

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

// Swish activation function (also known as SiLU).
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// Swish derivative.
fn swish_derivative(x: f32) -> f32 {
    let sigmoid_x = 1.0 / (1.0 + (-x).exp());
    sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
}

// Tanh activation function.
fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

// Tanh derivative given its output y (i.e., y = tanh(x)).
fn tanh_derivative(y: f32) -> f32 {
    1.0 - y * y
}

// ============================================================================
// Numerical Gradient Computation Helper
// ============================================================================

/// Compute numerical derivative using central difference formula.
/// d/dx f(x) ≈ (f(x + h) - f(x - h)) / (2h)
/// Using h = 1e-4 for better f32 numerical stability.
fn numerical_derivative<F>(f: F, x: f32, h: f32) -> f32
where
    F: Fn(f32) -> f32,
{
    let f_plus = f(x + h);
    let f_minus = f(x - h);
    (f_plus - f_minus) / (2.0 * h)
}

// ============================================================================
// Gradient Checking Tests for Leaky ReLU
// ============================================================================

#[test]
fn test_leaky_relu_gradient_positive() {
    let x = 2.0f32;
    let alpha = 0.01f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| leaky_relu(x, alpha), x, h);
    let analytical = leaky_relu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_leaky_relu_gradient_negative() {
    let x = -2.0f32;
    let alpha = 0.01f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| leaky_relu(x, alpha), x, h);
    let analytical = leaky_relu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_leaky_relu_gradient_near_zero() {
    // Test near the discontinuity point
    let x = 0.001f32;
    let alpha = 0.01f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| leaky_relu(x, alpha), x, h);
    let analytical = leaky_relu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_leaky_relu_gradient_different_alpha() {
    let x = -1.5f32;
    let alpha = 0.2f32; // Different alpha value
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| leaky_relu(x, alpha), x, h);
    let analytical = leaky_relu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

// ============================================================================
// Gradient Checking Tests for ELU
// ============================================================================

#[test]
fn test_elu_gradient_positive() {
    let x = 2.0f32;
    let alpha = 1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| elu(x, alpha), x, h);
    let analytical = elu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_elu_gradient_negative() {
    let x = -1.0f32;
    let alpha = 1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| elu(x, alpha), x, h);
    let analytical = elu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_elu_gradient_near_zero() {
    let x = 0.001f32;
    let alpha = 1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| elu(x, alpha), x, h);
    let analytical = elu_derivative(x, alpha);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_elu_gradient_large_negative() {
    let x = -5.0f32;
    let alpha = 1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(|x| elu(x, alpha), x, h);
    let analytical = elu_derivative(x, alpha);

    // At very large negative values, numerical precision is reduced
    assert_relative_eq!(numerical, analytical, max_relative = 0.03);
}

// ============================================================================
// Gradient Checking Tests for GELU
// ============================================================================

#[test]
fn test_gelu_gradient_zero() {
    let x = 0.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(gelu, x, h);
    let analytical = gelu_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_gelu_gradient_positive() {
    let x = 1.5f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(gelu, x, h);
    let analytical = gelu_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_gelu_gradient_negative() {
    let x = -1.5f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(gelu, x, h);
    let analytical = gelu_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_gelu_gradient_small_positive() {
    let x = 0.1f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(gelu, x, h);
    let analytical = gelu_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_gelu_gradient_small_negative() {
    let x = -0.1f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(gelu, x, h);
    let analytical = gelu_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

// ============================================================================
// Gradient Checking Tests for Swish
// ============================================================================

#[test]
fn test_swish_gradient_zero() {
    let x = 0.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(swish, x, h);
    let analytical = swish_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_swish_gradient_positive() {
    let x = 2.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(swish, x, h);
    let analytical = swish_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_swish_gradient_negative() {
    let x = -2.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(swish, x, h);
    let analytical = swish_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_swish_gradient_small_values() {
    let x = 0.5f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(swish, x, h);
    let analytical = swish_derivative(x);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

// ============================================================================
// Gradient Checking Tests for Tanh
// ============================================================================

#[test]
fn test_tanh_gradient_zero() {
    let x = 0.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(tanh_activation, x, h);

    // For tanh, the derivative takes the OUTPUT, not the input
    let y = tanh_activation(x);
    let analytical = tanh_derivative(y);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_tanh_gradient_positive() {
    let x = 1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(tanh_activation, x, h);

    // For tanh, the derivative takes the OUTPUT, not the input
    let y = tanh_activation(x);
    let analytical = tanh_derivative(y);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_tanh_gradient_negative() {
    let x = -1.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(tanh_activation, x, h);

    // For tanh, the derivative takes the OUTPUT, not the input
    let y = tanh_activation(x);
    let analytical = tanh_derivative(y);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_tanh_gradient_large_positive() {
    let x = 3.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(tanh_activation, x, h);

    // For tanh, the derivative takes the OUTPUT, not the input
    let y = tanh_activation(x);
    let analytical = tanh_derivative(y);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

#[test]
fn test_tanh_gradient_large_negative() {
    let x = -3.0f32;
    let h = 1e-4f32;

    let numerical = numerical_derivative(tanh_activation, x, h);

    // For tanh, the derivative takes the OUTPUT, not the input
    let y = tanh_activation(x);
    let analytical = tanh_derivative(y);

    assert_relative_eq!(numerical, analytical, max_relative = 0.01);
}

// ============================================================================
// Multi-point Gradient Checking Tests
// ============================================================================

#[test]
fn test_all_activations_at_multiple_points() {
    // Skip x=0.0 to avoid discontinuity issues with piecewise functions
    let test_points = vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];
    let h = 1e-4f32;
    let alpha_leaky = 0.01f32;
    let alpha_elu = 1.0f32;

    for &x in &test_points {
        // Test Leaky ReLU (piecewise function, derivative changes at x=0)
        let numerical_leaky = numerical_derivative(|x| leaky_relu(x, alpha_leaky), x, h);
        let analytical_leaky = leaky_relu_derivative(x, alpha_leaky);
        assert_relative_eq!(numerical_leaky, analytical_leaky, max_relative = 0.01);

        // Test ELU (piecewise function, derivative changes at x=0)
        let numerical_elu = numerical_derivative(|x| elu(x, alpha_elu), x, h);
        let analytical_elu = elu_derivative(x, alpha_elu);
        assert_relative_eq!(numerical_elu, analytical_elu, max_relative = 0.01);

        // Test GELU (smooth function)
        let numerical_gelu = numerical_derivative(gelu, x, h);
        let analytical_gelu = gelu_derivative(x);
        assert_relative_eq!(numerical_gelu, analytical_gelu, max_relative = 0.01);

        // Test Swish (smooth function)
        let numerical_swish = numerical_derivative(swish, x, h);
        let analytical_swish = swish_derivative(x);
        assert_relative_eq!(numerical_swish, analytical_swish, max_relative = 0.01);

        // Test Tanh (smooth function)
        let numerical_tanh = numerical_derivative(tanh_activation, x, h);
        let y = tanh_activation(x);
        let analytical_tanh = tanh_derivative(y);
        assert_relative_eq!(numerical_tanh, analytical_tanh, max_relative = 0.01);
    }
}
