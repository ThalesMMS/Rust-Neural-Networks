//! Dropout layer implementation for regularization
//!
//! This module provides a DropoutLayer that randomly drops (sets to zero) a fraction
//! of input units during training to prevent overfitting. During inference, all units
//! are kept and outputs are passed through unchanged.

use crate::utils::rng::SimpleRng;
use std::cell::RefCell;

/// Dropout layer for regularization.
///
/// During training, randomly sets a fraction of input units to zero with probability
/// `drop_rate`, and scales the remaining units by 1/(1-drop_rate) to maintain expected
/// values. During inference, passes inputs through unchanged.
///
/// # Fields
///
/// * `size` - Number of input/output features (dropout doesn't change dimensions)
/// * `drop_rate` - Probability of dropping each unit (0.0 = no dropout, 1.0 = drop all)
/// * `training` - Whether the layer is in training mode (true) or inference mode (false)
/// * `mask` - Binary mask indicating which units were kept in the last forward pass
/// * `rng` - Random number generator for dropout mask generation
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::DropoutLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let mut layer = DropoutLayer::new(512, 0.5, &mut rng);
/// layer.set_training(true);  // Enable dropout for training
/// assert_eq!(layer.input_size(), 512);
/// assert_eq!(layer.output_size(), 512);
/// ```
pub struct DropoutLayer {
    size: usize,
    drop_rate: f32,
    training: bool,
    mask: RefCell<Vec<f32>>,
    rng: RefCell<SimpleRng>,
}

impl DropoutLayer {
    /// Creates a new dropout layer with specified size and drop rate.
    ///
    /// The layer is initialized in training mode by default. The mask is allocated
    /// but not initialized until the first forward pass.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of input/output features
    /// * `drop_rate` - Probability of dropping each unit (should be in range [0.0, 1.0))
    /// * `rng` - Random number generator for reproducible dropout
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(256, 0.3, &mut rng);
    /// assert_eq!(layer.parameter_count(), 0);  // Dropout has no trainable parameters
    /// ```
    pub fn new(size: usize, drop_rate: f32, rng: &mut SimpleRng) -> Self {
        assert!(
            (0.0..1.0).contains(&drop_rate),
            "drop_rate must be in range [0.0, 1.0)"
        );

        Self {
            size,
            drop_rate,
            training: true,
            mask: RefCell::new(Vec::new()),
            rng: RefCell::new(rng.clone()),
        }
    }

    /// Set whether the layer is in training mode.
    ///
    /// When `training` is true, dropout is applied during forward passes with units
    /// randomly dropped according to `drop_rate`. When false (inference mode), inputs
    /// pass through unchanged, allowing deterministic predictions.
    ///
    /// # Arguments
    ///
    /// * `training` - `true` to enable dropout (training mode), `false` to disable (inference mode)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(128, 0.5, &mut rng);
    /// layer.set_training(false);  // Switch to inference mode
    /// assert_eq!(layer.is_training(), false);
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get whether the layer is in training mode.
    ///
    /// # Returns
    ///
    /// `true` if the layer is in training mode (dropout active), `false` if in inference mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(128, 0.5, &mut rng);
    /// assert_eq!(layer.is_training(), true);  // Default is training mode
    /// layer.set_training(false);
    /// assert_eq!(layer.is_training(), false);
    /// ```
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get the dropout rate.
    ///
    /// # Returns
    ///
    /// The probability (in range [0.0, 1.0)) that each unit will be dropped during training.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(128, 0.3, &mut rng);
    /// assert_eq!(layer.drop_rate(), 0.3);
    /// ```
    pub fn drop_rate(&self) -> f32 {
        self.drop_rate
    }

    /// Get the input size of the layer.
    ///
    /// # Returns
    ///
    /// Number of input features.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(256, 0.5, &mut rng);
    /// assert_eq!(layer.input_size(), 256);
    /// ```
    pub fn input_size(&self) -> usize {
        self.size
    }

    /// Get the output size of the layer.
    ///
    /// For dropout, input and output sizes are always the same since dropout
    /// doesn't change the dimensionality of the data.
    ///
    /// # Returns
    ///
    /// Number of output features (equal to input_size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(512, 0.5, &mut rng);
    /// assert_eq!(layer.output_size(), 512);
    /// assert_eq!(layer.input_size(), layer.output_size());
    /// ```
    pub fn output_size(&self) -> usize {
        self.size
    }

    /// Get the number of trainable parameters.
    ///
    /// Dropout has no trainable parameters, so this always returns 0.
    ///
    /// # Returns
    ///
    /// Always returns `0` since dropout is a non-parametric regularization technique.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::dropout::DropoutLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(1024, 0.5, &mut rng);
    /// assert_eq!(layer.parameter_count(), 0);
    /// ```
    pub fn parameter_count(&self) -> usize {
        0
    }
}

use crate::layers::Layer;
use crate::optimizers::Optimizer;

impl Layer for DropoutLayer {
    /// Forward propagation through the dropout layer.
    ///
    /// During training, randomly drops units with probability `drop_rate` and scales
    /// remaining units by 1/(1-drop_rate). During inference, passes input through unchanged.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × size)
    /// * `output` - Output buffer (batch_size × size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(128, 0.5, &mut rng);
    /// let input = vec![1.0f32; 128];
    /// let mut output = vec![0.0f32; 128];
    /// layer.forward(&input, &mut output, 1);
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let total_size = batch_size * self.size;
        assert_eq!(
            input.len(),
            total_size,
            "input len mismatch: expected {}, got {}",
            total_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            total_size,
            "output len mismatch: expected {}, got {}",
            total_size,
            output.len()
        );

        if !self.training {
            // Inference mode: pass through unchanged
            output.copy_from_slice(input);
        } else {
            // Training mode: apply dropout with mask
            let scale = 1.0 / (1.0 - self.drop_rate);
            let mut mask = self.mask.borrow_mut();
            let mut rng = self.rng.borrow_mut();

            // Resize mask if needed
            if mask.len() != total_size {
                mask.resize(total_size, 0.0);
            }

            // Generate dropout mask and apply to input
            for i in 0..total_size {
                let rand_val = rng.next_f32();
                if rand_val > self.drop_rate {
                    // Keep this unit, scale to maintain expected value
                    mask[i] = 1.0;
                    output[i] = input[i] * scale;
                } else {
                    // Drop this unit
                    mask[i] = 0.0;
                    output[i] = 0.0;
                }
            }
        }
    }

    /// Backward propagation through the dropout layer.
    ///
    /// Applies the saved dropout mask to the gradient, passing through gradients
    /// only for units that were kept during forward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass (unused for dropout)
    /// * `grad_output` - Gradient of loss w.r.t. layer output (batch_size × size)
    /// * `grad_input` - Buffer to store gradient w.r.t. input (batch_size × size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(128, 0.5, &mut rng);
    /// layer.set_training(true);
    ///
    /// let input = vec![1.0f32; 128];
    /// let mut output = vec![0.0f32; 128];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// let grad_output = vec![1.0f32; 128];
    /// let mut grad_input = vec![0.0f32; 128];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    /// ```
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let total_size = batch_size * self.size;
        assert_eq!(
            grad_output.len(),
            total_size,
            "grad_output len mismatch: expected {}, got {}",
            total_size,
            grad_output.len()
        );
        assert_eq!(
            grad_input.len(),
            total_size,
            "grad_input len mismatch: expected {}, got {}",
            total_size,
            grad_input.len()
        );

        if !self.training {
            // Inference mode: gradient passes through unchanged
            grad_input.copy_from_slice(grad_output);
        } else {
            // Training mode: apply mask to gradient
            let mask = self.mask.borrow();
            let scale = 1.0 / (1.0 - self.drop_rate);
            for i in 0..total_size {
                grad_input[i] = grad_output[i] * mask[i] * scale;
            }
        }
    }

    /// Update layer parameters (no-op for dropout layer).
    ///
    /// Dropout has no trainable parameters, so this method does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(64, 0.5, &mut rng);
    /// layer.update_parameters(0.01);  // No-op, but safe to call
    /// assert_eq!(layer.parameter_count(), 0);
    /// ```
    fn update_parameters(&mut self, _learning_rate: f32) {
        // No parameters to update
    }

    /// Update layer parameters using optimizer (no-op for dropout layer).
    ///
    /// Dropout has no trainable parameters, so this method does nothing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// use rust_neural_networks::optimizers::sgd::SGD;
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DropoutLayer::new(64, 0.5, &mut rng);
    /// let mut optimizer = SGD::new(0.01);
    /// layer.update_with_optimizer(&mut optimizer);  // No-op, but safe to call
    /// ```
    fn update_with_optimizer(&mut self, _optimizer: &mut dyn Optimizer) {
        // No parameters to update
    }

    /// Get the input size of the layer.
    ///
    /// # Returns
    ///
    /// Number of input features.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(256, 0.5, &mut rng);
    /// assert_eq!(layer.input_size(), 256);
    /// ```
    fn input_size(&self) -> usize {
        self.size
    }

    /// Get the output size of the layer.
    ///
    /// # Returns
    ///
    /// Number of output features (equal to input_size for dropout).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(512, 0.5, &mut rng);
    /// assert_eq!(layer.output_size(), 512);
    /// assert_eq!(layer.input_size(), layer.output_size());
    /// ```
    fn output_size(&self) -> usize {
        self.size
    }

    /// Get the number of trainable parameters.
    ///
    /// Dropout has no trainable parameters, so this always returns 0.
    ///
    /// # Returns
    ///
    /// Always returns `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::{Layer, dropout::DropoutLayer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DropoutLayer::new(1024, 0.5, &mut rng);
    /// assert_eq!(layer.parameter_count(), 0);
    /// ```
    fn parameter_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_layer_creation() {
        let mut rng = SimpleRng::new(42);
        let layer = DropoutLayer::new(128, 0.5, &mut rng);

        assert_eq!(layer.input_size(), 128);
        assert_eq!(layer.output_size(), 128);
        assert_eq!(layer.drop_rate(), 0.5);
        assert!(layer.is_training()); // Default is training mode
    }

    #[test]
    fn test_dropout_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = DropoutLayer::new(512, 0.3, &mut rng);

        // Dropout has no trainable parameters
        assert_eq!(layer.parameter_count(), 0);
    }

    #[test]
    fn test_dropout_training_mode() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(10, 0.5, &mut rng);

        // Default should be training mode
        assert!(layer.is_training());

        // Switch to inference mode
        layer.set_training(false);
        assert!(!layer.is_training());

        // Switch back to training mode
        layer.set_training(true);
        assert!(layer.is_training());
    }

    #[test]
    #[should_panic(expected = "drop_rate must be in range [0.0, 1.0)")]
    fn test_dropout_invalid_rate_too_high() {
        let mut rng = SimpleRng::new(42);
        let _layer = DropoutLayer::new(10, 1.0, &mut rng);
    }

    #[test]
    #[should_panic(expected = "drop_rate must be in range [0.0, 1.0)")]
    fn test_dropout_invalid_rate_negative() {
        let mut rng = SimpleRng::new(42);
        let _layer = DropoutLayer::new(10, -0.1, &mut rng);
    }

    #[test]
    fn test_dropout_zero_rate() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(10, 0.0, &mut rng);
        layer.set_training(true);

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = vec![0.0f32; 10];
        layer.forward(&input, &mut output, 1);

        // With drop_rate = 0.0, all values should be kept
        for i in 0..10 {
            assert_eq!(output[i], input[i]);
        }
    }

    #[test]
    fn test_dropout_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut layer1 = DropoutLayer::new(10, 0.5, &mut rng1);
        layer1.set_training(true);

        let mut rng2 = SimpleRng::new(42);
        let mut layer2 = DropoutLayer::new(10, 0.5, &mut rng2);
        layer2.set_training(true);

        let input = vec![1.0f32; 10];
        let mut output1 = vec![0.0f32; 10];
        let mut output2 = vec![0.0f32; 10];

        layer1.forward(&input, &mut output1, 1);
        layer2.forward(&input, &mut output2, 1);

        // Same seed should produce identical masks
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_dropout_forward() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(10, 0.5, &mut rng);

        // Test inference mode
        layer.set_training(false);
        let input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 10];
        layer.forward(&input, &mut output, 1);

        // In inference mode, output should equal input
        assert_eq!(output, input);

        // Test training mode
        layer.set_training(true);
        let mut output_train = vec![0.0f32; 10];
        layer.forward(&input, &mut output_train, 1);

        // In training mode, some outputs should be 0 (dropped)
        // and others should be scaled by 1/(1-0.5) = 2.0
        let mut dropped_count = 0;
        let mut kept_count = 0;
        for &val in &output_train {
            if val == 0.0 {
                dropped_count += 1;
            } else {
                kept_count += 1;
                // Kept values should be scaled
                assert!((val - 2.0).abs() < 1e-6);
            }
        }

        // Should have some dropped and some kept (statistically)
        assert!(dropped_count > 0);
        assert!(kept_count > 0);
    }

    #[test]
    fn test_dropout_forward_batch() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(5, 0.5, &mut rng);
        layer.set_training(true);

        // Batch of 3 samples
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        ];
        let mut output = vec![0.0f32; 15];

        layer.forward(&input, &mut output, 3);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));

        // Some values should be dropped (zero)
        assert!(output.contains(&0.0));

        // Some values should be kept (non-zero)
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_dropout_backward() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(10, 0.5, &mut rng);

        // Test inference mode
        layer.set_training(false);
        let input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 10];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0f32; 10];
        let mut grad_input = vec![0.0f32; 10];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // In inference mode, gradient should pass through unchanged
        assert_eq!(grad_input, grad_output);

        // Test training mode
        layer.set_training(true);
        let mut output_train = vec![0.0f32; 10];
        layer.forward(&input, &mut output_train, 1);

        let grad_output_train = vec![1.0f32; 10];
        let mut grad_input_train = vec![0.0f32; 10];
        layer.backward(&input, &grad_output_train, &mut grad_input_train, 1);

        // In training mode, gradient should be masked and scaled
        let scale = 1.0 / (1.0 - 0.5);
        for i in 0..10 {
            if output_train[i] == 0.0 {
                // Dropped units should have zero gradient
                assert_eq!(grad_input_train[i], 0.0);
            } else {
                // Kept units should have scaled gradient
                assert!((grad_input_train[i] - scale).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_dropout_backward_batch() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(5, 0.5, &mut rng);
        layer.set_training(true);

        // Batch of 2 samples
        let input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 10];
        layer.forward(&input, &mut output, 2);

        let grad_output = vec![1.0f32; 10];
        let mut grad_input = vec![0.0f32; 10];
        layer.backward(&input, &grad_output, &mut grad_input, 2);

        // Gradients should be finite
        assert!(grad_input.iter().all(|&x| x.is_finite()));

        // Gradient pattern should match forward pass
        let scale = 1.0 / (1.0 - 0.5);
        for i in 0..10 {
            if output[i] == 0.0 {
                assert_eq!(grad_input[i], 0.0);
            } else {
                assert!((grad_input[i] - scale).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_dropout_update_parameters() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(10, 0.5, &mut rng);

        // Dropout has no parameters, so update should be a no-op
        layer.update_parameters(0.1);

        // No assertion needed - just verify it doesn't panic
        assert_eq!(layer.parameter_count(), 0);
    }

    #[test]
    fn test_dropout_scaling_preserves_expected_value() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DropoutLayer::new(1000, 0.5, &mut rng);
        layer.set_training(true);

        // Use a constant input to verify expected value is preserved
        let input = vec![1.0f32; 1000];
        let mut output = vec![0.0f32; 1000];
        layer.forward(&input, &mut output, 1);

        // With drop_rate=0.5 and scaling=2.0, the sum should be approximately equal
        let input_sum: f32 = input.iter().sum();
        let output_sum: f32 = output.iter().sum();

        // Allow 10% tolerance due to randomness
        let tolerance = input_sum * 0.1;
        assert!(
            (output_sum - input_sum).abs() < tolerance,
            "Expected sum ~{}, got {}",
            input_sum,
            output_sum
        );
    }
}
