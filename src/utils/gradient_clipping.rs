//! Gradient clipping utilities for preventing exploding gradients
//!
//! This module provides functions for clipping gradients during training, which is
//! particularly important for recurrent neural networks (RNNs and LSTMs) where
//! gradients can explode during backpropagation through time (BPTT).
//!
//! Two main approaches are provided:
//! - **Clipping by norm**: Scales the entire gradient vector if its L2 norm exceeds a threshold
//! - **Clipping by value**: Clamps individual gradient elements to a specified range

/// Clips gradients by their L2 norm to prevent exploding gradients.
///
/// If the L2 norm of the gradient vector exceeds `max_norm`, the entire gradient
/// is scaled down proportionally so that its norm equals `max_norm`. If the norm
/// is already below `max_norm`, the gradient is left unchanged.
///
/// This is the recommended approach for RNN/LSTM training as it preserves the
/// direction of the gradient vector while limiting its magnitude.
///
/// # Formula
///
/// ```text
/// norm = sqrt(sum(grad_i^2))
/// if norm > max_norm:
///     grad = grad * (max_norm / norm)
/// ```
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient values to clip
/// * `max_norm` - Maximum allowed L2 norm for the gradient vector
///
/// # Returns
///
/// The computed L2 norm before clipping, which can be useful for monitoring
/// gradient magnitudes during training.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::gradient_clipping::clip_gradient_norm;
///
/// // Gradient with norm = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
/// let mut grads = vec![3.0, 4.0, 5.0];
/// let norm = clip_gradient_norm(&mut grads, 5.0);
///
/// // Gradient should be scaled down to norm = 5.0
/// assert!((norm - 7.071).abs() < 0.01);
/// let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
/// assert!((new_norm - 5.0).abs() < 0.01);
/// ```
///
/// ```
/// use rust_neural_networks::utils::gradient_clipping::clip_gradient_norm;
///
/// // Gradient with small norm should be unchanged
/// let mut grads = vec![0.1, 0.2, 0.3];
/// let original = grads.clone();
/// clip_gradient_norm(&mut grads, 10.0);
/// assert_eq!(grads, original);
/// ```
pub fn clip_gradient_norm(gradients: &mut [f32], max_norm: f32) -> f32 {
    // Compute L2 norm: sqrt(sum(g_i^2))
    let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

    // Only clip if norm exceeds threshold
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }

    norm
}

/// Clips gradients by value, clamping each element to the range [-clip_value, clip_value].
///
/// This is a simpler but less sophisticated approach than norm-based clipping.
/// It clips each gradient element independently without considering the overall
/// gradient direction. This can change the direction of the gradient vector.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient values to clip
/// * `clip_value` - Maximum absolute value for any gradient element
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::gradient_clipping::clip_gradient_value;
///
/// let mut grads = vec![-5.0, 3.0, 10.0, -2.0];
/// clip_gradient_value(&mut grads, 4.0);
/// assert_eq!(grads, vec![-4.0, 3.0, 4.0, -2.0]);
/// ```
///
/// ```
/// use rust_neural_networks::utils::gradient_clipping::clip_gradient_value;
///
/// let mut grads = vec![1.0, -1.5, 0.5];
/// clip_gradient_value(&mut grads, 2.0);
/// assert_eq!(grads, vec![1.0, -1.5, 0.5]); // No change needed
/// ```
pub fn clip_gradient_value(gradients: &mut [f32], clip_value: f32) {
    let clip_value = clip_value.abs(); // Ensure positive
    for g in gradients.iter_mut() {
        if *g > clip_value {
            *g = clip_value;
        } else if *g < -clip_value {
            *g = -clip_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_by_norm_with_clipping() {
        // Gradient with norm sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.071
        let mut grads = vec![3.0, 4.0, 5.0];
        let norm = clip_gradient_norm(&mut grads, 5.0);

        // Check original norm was computed correctly
        assert!((norm - 7.071).abs() < 0.01);

        // Check new norm is approximately 5.0
        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 5.0).abs() < 0.01);

        // Check that gradient direction was preserved (scaled proportionally)
        let scale = 5.0 / 7.071;
        assert!((grads[0] - 3.0 * scale).abs() < 0.01);
        assert!((grads[1] - 4.0 * scale).abs() < 0.01);
        assert!((grads[2] - 5.0 * scale).abs() < 0.01);
    }

    #[test]
    fn test_clip_by_norm_no_clipping() {
        // Gradient with small norm should remain unchanged
        let mut grads = vec![0.1, 0.2, 0.3];
        let original = grads.clone();
        let norm = clip_gradient_norm(&mut grads, 10.0);

        // Norm should be computed correctly
        let expected_norm = (0.01 + 0.04 + 0.09_f32).sqrt();
        assert!((norm - expected_norm).abs() < 1e-6);

        // Gradients should be unchanged
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_by_norm_zero_gradients() {
        // Zero gradients should remain zero
        let mut grads = vec![0.0, 0.0, 0.0];
        let norm = clip_gradient_norm(&mut grads, 5.0);

        assert_eq!(norm, 0.0);
        assert_eq!(grads, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clip_by_norm_single_element() {
        // Single element should work correctly
        let mut grads = vec![10.0];
        let norm = clip_gradient_norm(&mut grads, 3.0);

        assert_eq!(norm, 10.0);
        assert_eq!(grads, vec![3.0]);
    }

    #[test]
    fn test_clip_by_norm_negative_values() {
        // Should handle negative values correctly
        let mut grads = vec![-3.0, -4.0];
        let norm = clip_gradient_norm(&mut grads, 2.0);

        assert_eq!(norm, 5.0);
        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 2.0).abs() < 1e-6);

        // Direction should be preserved
        assert!((grads[0] - (-3.0 * 2.0 / 5.0)).abs() < 1e-6);
        assert!((grads[1] - (-4.0 * 2.0 / 5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_clip_by_value_with_clipping() {
        let mut grads = vec![-5.0, 3.0, 10.0, -2.0];
        clip_gradient_value(&mut grads, 4.0);
        assert_eq!(grads, vec![-4.0, 3.0, 4.0, -2.0]);
    }

    #[test]
    fn test_clip_by_value_no_clipping() {
        let mut grads = vec![1.0, -1.5, 0.5];
        let original = grads.clone();
        clip_gradient_value(&mut grads, 2.0);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_by_value_zero() {
        let mut grads = vec![5.0, -3.0, 2.0];
        clip_gradient_value(&mut grads, 0.0);
        assert_eq!(grads, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clip_by_value_negative_clip_value() {
        // Should handle negative clip_value by taking absolute value
        let mut grads = vec![-5.0, 3.0, 10.0];
        clip_gradient_value(&mut grads, -4.0);
        assert_eq!(grads, vec![-4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_clip_by_value_large_clip_value() {
        // Large clip value should not change gradients
        let mut grads = vec![1.0, -2.0, 3.0];
        let original = grads.clone();
        clip_gradient_value(&mut grads, 100.0);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_both_methods_comparison() {
        // Compare the two methods on the same input
        let mut grads_norm = vec![3.0, 4.0, 5.0];
        let mut grads_value = vec![3.0, 4.0, 5.0];

        // Clip by norm (max norm = 5.0)
        clip_gradient_norm(&mut grads_norm, 5.0);

        // Clip by value (max value = 4.0)
        clip_gradient_value(&mut grads_value, 4.0);

        // Norm-based preserves direction, value-based doesn't
        // After norm clipping: all values scaled proportionally
        // After value clipping: only values > 4.0 are clamped
        assert_eq!(grads_value, vec![3.0, 4.0, 4.0]);

        // Norm clipping should scale all proportionally
        let scale = 5.0 / 7.071;
        assert!((grads_norm[0] - 3.0 * scale).abs() < 0.01);
        assert!((grads_norm[1] - 4.0 * scale).abs() < 0.01);
        assert!((grads_norm[2] - 5.0 * scale).abs() < 0.01);
    }
}
