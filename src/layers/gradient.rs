//! Gradient accumulator for neural network layer parameter updates
//!
//! This module provides the [`GradientAccumulator`] struct, which encapsulates the common
//! pattern of `RefCell<Vec<f32>>` gradient storage used across all layer implementations.
//! It provides a uniform interface for zeroing, accumulating, and applying gradients.
//!
//! # Overview
//!
//! All layer types (DenseLayer, Conv2DLayer, RnnLayer, LstmLayer) share identical logic for:
//! - Storing gradients in a `RefCell<Vec<f32>>` for interior mutability
//! - Zeroing gradients after parameter updates
//! - Accumulating gradients across backward passes
//! - Applying SGD or optimizer-based updates
//!
//! `GradientAccumulator` centralises this repeated code into a single reusable component.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::layers::gradient::GradientAccumulator;
//!
//! let acc = GradientAccumulator::new(4);
//! assert_eq!(acc.len(), 4);
//!
//! // Accumulate gradients
//! acc.accumulate(&[0.1, 0.2, 0.3, 0.4]);
//! assert!((acc.l2_norm() - (0.1f32*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4).sqrt()).abs() < 1e-6);
//!
//! // Apply SGD update and zero
//! let mut params = vec![1.0f32; 4];
//! acc.apply_sgd_update(&mut params, 0.5);
//! assert!((params[0] - (1.0 - 0.5 * 0.1)).abs() < 1e-6);
//!
//! // Gradients should be zero after update
//! assert_eq!(acc.l2_norm(), 0.0);
//! ```

use crate::optimizers::Optimizer;
use std::cell::{Ref, RefCell, RefMut};

/// Encapsulates a gradient accumulation buffer with interior mutability.
///
/// Stores gradients in a `RefCell<Vec<f32>>`, providing methods for the common
/// operations all layer implementations need:
/// - Zeroing gradients after an update
/// - Adding gradients elementwise (accumulation)
/// - Adding scaled gradients elementwise
/// - Applying a vanilla SGD update to parameters, then zeroing
/// - Applying an optimizer update to parameters, then zeroing
/// - Computing the L2 norm for gradient monitoring
///
/// Using `RefCell` preserves interior mutability so layers can accumulate gradients
/// inside `&self` methods (as required by the `Layer::backward` signature).
///
/// # Panics
///
/// Methods that take gradient or parameter slices panic when the supplied slice
/// length differs from [`GradientAccumulator::len`].
///
/// # Examples
///
/// ```
/// use rust_neural_networks::layers::gradient::GradientAccumulator;
///
/// let acc = GradientAccumulator::new(3);
/// acc.accumulate(&[1.0, 2.0, 3.0]);
/// let norm = acc.l2_norm();
/// assert!((norm - (1.0f32 + 4.0 + 9.0).sqrt()).abs() < 1e-6);
/// acc.zero();
/// assert_eq!(acc.l2_norm(), 0.0);
/// ```
pub struct GradientAccumulator {
    gradients: RefCell<Vec<f32>>,
}

impl GradientAccumulator {
    /// Creates a new gradient accumulator initialised to zero.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of gradient elements (must match the corresponding parameter count)
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(10);
    /// assert_eq!(acc.len(), 10);
    /// assert_eq!(acc.l2_norm(), 0.0);
    /// ```
    pub fn new(size: usize) -> Self {
        Self {
            gradients: RefCell::new(vec![0.0f32; size]),
        }
    }

    /// Returns the number of gradient elements in this accumulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(8);
    /// assert_eq!(acc.len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.gradients.borrow().len()
    }

    /// Returns `true` if the accumulator holds zero gradient elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let empty = GradientAccumulator::new(0);
    /// assert!(empty.is_empty());
    /// let non_empty = GradientAccumulator::new(4);
    /// assert!(!non_empty.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.gradients.borrow().is_empty()
    }

    /// Resets all accumulated gradients to zero.
    ///
    /// Call this after a parameter update so that the next backward pass starts
    /// from a clean slate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(3);
    /// acc.accumulate(&[1.0, 2.0, 3.0]);
    /// assert!(acc.l2_norm() > 0.0);
    /// acc.zero();
    /// assert_eq!(acc.l2_norm(), 0.0);
    /// ```
    pub fn zero(&self) {
        self.gradients
            .borrow_mut()
            .iter_mut()
            .for_each(|g| *g = 0.0);
    }

    /// Adds a gradient vector elementwise to the accumulator.
    ///
    /// This is the standard accumulation step: repeated calls add gradients from
    /// multiple backward passes (gradient accumulation over mini-batches).
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient values to add; length must equal [`GradientAccumulator::len`]
    ///
    /// # Panics
    ///
    /// Panics if `grads.len()` differs from `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate(&[1.0, 2.0]);
    /// acc.accumulate(&[0.5, 0.5]);
    /// let grads = acc.borrow();
    /// assert!((grads[0] - 1.5).abs() < 1e-6);
    /// assert!((grads[1] - 2.5).abs() < 1e-6);
    /// ```
    pub fn accumulate(&self, grads: &[f32]) {
        let mut buf = self.gradients.borrow_mut();
        assert_eq!(
            buf.len(),
            grads.len(),
            "gradient accumulate: length mismatch: accumulator has {}, got {}",
            buf.len(),
            grads.len()
        );
        for (acc, &g) in buf.iter_mut().zip(grads.iter()) {
            *acc += g;
        }
    }

    /// Adds a scaled gradient vector elementwise to the accumulator.
    ///
    /// Equivalent to `accumulate`, but each incoming gradient value is multiplied
    /// by `scale` before being added. Useful for averaging gradients over a batch:
    /// `accumulate_scaled(grads, 1.0 / batch_size as f32)`.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient values; length must equal [`GradientAccumulator::len`]
    /// * `scale` - Scalar multiplier applied to each element of `grads`
    ///
    /// # Panics
    ///
    /// Panics if `grads.len()` differs from `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate_scaled(&[4.0, 6.0], 0.5);
    /// let grads = acc.borrow();
    /// assert!((grads[0] - 2.0).abs() < 1e-6);
    /// assert!((grads[1] - 3.0).abs() < 1e-6);
    /// ```
    pub fn accumulate_scaled(&self, grads: &[f32], scale: f32) {
        let mut buf = self.gradients.borrow_mut();
        assert_eq!(
            buf.len(),
            grads.len(),
            "gradient accumulate_scaled: length mismatch: accumulator has {}, got {}",
            buf.len(),
            grads.len()
        );
        for (acc, &g) in buf.iter_mut().zip(grads.iter()) {
            *acc += g * scale;
        }
    }

    /// Returns an immutable borrow of the gradient buffer.
    ///
    /// Useful when a layer needs to pass the raw gradient slice to an optimizer or
    /// to a BLAS routine via the `RefCell` borrow mechanism.
    ///
    /// # Panics
    ///
    /// Panics if the accumulator is already mutably borrowed (standard `RefCell` rules).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate(&[3.0, 4.0]);
    /// let grads = acc.borrow();
    /// assert_eq!(grads[0], 3.0);
    /// assert_eq!(grads[1], 4.0);
    /// ```
    pub fn borrow(&self) -> Ref<'_, Vec<f32>> {
        self.gradients.borrow()
    }

    /// Returns a mutable borrow of the gradient buffer.
    ///
    /// Gives direct write access to the underlying gradient vector. Used by layer
    /// implementations that accumulate gradients through BLAS calls (e.g. `sgemm`
    /// writes directly into this buffer with `beta = 1.0` for accumulation).
    ///
    /// # Panics
    ///
    /// Panics if the accumulator is already borrowed (standard `RefCell` rules).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// {
    ///     let mut grads = acc.borrow_mut();
    ///     grads[0] = 5.0;
    ///     grads[1] = 6.0;
    /// }
    /// assert!((acc.l2_norm() - (25.0f32 + 36.0).sqrt()).abs() < 1e-6);
    /// ```
    pub fn borrow_mut(&self) -> RefMut<'_, Vec<f32>> {
        self.gradients.borrow_mut()
    }

    /// Computes the L2 norm (Euclidean magnitude) of the accumulated gradients.
    ///
    /// Returns `sqrt(sum(g_i^2))`. Useful for monitoring gradient flow and detecting
    /// vanishing or exploding gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate(&[3.0, 4.0]);
    /// assert!((acc.l2_norm() - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    /// ```
    pub fn l2_norm(&self) -> f32 {
        self.gradients
            .borrow()
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt()
    }

    /// Applies a vanilla SGD update to `parameters` using the accumulated gradients,
    /// then zeros the accumulator.
    ///
    /// For each parameter `p` and corresponding accumulated gradient `g`:
    /// ```text
    /// p = p - learning_rate * g
    /// ```
    ///
    /// The accumulator is zeroed after the update so it is ready for the next iteration.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameters to update in place
    /// * `learning_rate` - Step size for gradient descent
    ///
    /// # Panics
    ///
    /// Panics if `parameters.len()` differs from [`GradientAccumulator::len`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate(&[0.2, 0.4]);
    /// let mut params = vec![1.0f32, 2.0f32];
    /// acc.apply_sgd_update(&mut params, 0.5);
    /// assert!((params[0] - 0.9).abs() < 1e-6); // 1.0 - 0.5 * 0.2
    /// assert!((params[1] - 1.8).abs() < 1e-6); // 2.0 - 0.5 * 0.4
    /// assert_eq!(acc.l2_norm(), 0.0);
    /// ```
    pub fn apply_sgd_update(&self, parameters: &mut [f32], learning_rate: f32) {
        let grads = self.gradients.borrow();
        assert_eq!(
            grads.len(),
            parameters.len(),
            "apply_sgd_update: length mismatch: gradients {}, parameters {}",
            grads.len(),
            parameters.len()
        );
        for (param, &grad) in parameters.iter_mut().zip(grads.iter()) {
            *param -= learning_rate * grad;
        }
        drop(grads);
        self.zero();
    }

    /// Applies an optimizer update to `parameters` using the accumulated gradients,
    /// then zeros the accumulator.
    ///
    /// Delegates to [`Optimizer::update`], which applies the optimizer-specific update
    /// rule (SGD with momentum, Adam, AdamW, RMSprop, etc.).
    ///
    /// The accumulator is zeroed after the update so it is ready for the next iteration.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameters to update in place
    /// * `optimizer` - Mutable reference to the optimizer managing update state
    ///
    /// # Panics
    ///
    /// Panics if `parameters.len()` differs from [`GradientAccumulator::len`], or if
    /// the underlying optimizer implementation panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::gradient::GradientAccumulator;
    /// use rust_neural_networks::optimizers::SGD;
    ///
    /// let acc = GradientAccumulator::new(2);
    /// acc.accumulate(&[0.1, 0.2]);
    /// let mut params = vec![1.0f32, 2.0f32];
    /// let mut optimizer = SGD::new(0.5);
    /// acc.apply_optimizer_update(&mut params, &mut optimizer);
    /// assert!((params[0] - 0.95).abs() < 1e-5); // 1.0 - 0.5 * 0.1
    /// assert!((params[1] - 1.9).abs() < 1e-5);  // 2.0 - 0.5 * 0.2
    /// assert_eq!(acc.l2_norm(), 0.0);
    /// ```
    pub fn apply_optimizer_update(&self, parameters: &mut [f32], optimizer: &mut dyn Optimizer) {
        let grads = self.gradients.borrow();
        assert_eq!(
            grads.len(),
            parameters.len(),
            "apply_optimizer_update: length mismatch: gradients {}, parameters {}",
            grads.len(),
            parameters.len()
        );
        optimizer.update(parameters, &grads);
        drop(grads);
        self.zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_gradient_accumulator_new() {
        let acc = GradientAccumulator::new(5);
        assert_eq!(acc.len(), 5);
        assert!(!acc.is_empty());
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_empty() {
        let acc = GradientAccumulator::new(0);
        assert_eq!(acc.len(), 0);
        assert!(acc.is_empty());
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_zero() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        assert!(acc.l2_norm() > 0.0);
        acc.zero();
        assert_eq!(acc.l2_norm(), 0.0);
        // All elements should be zero
        let grads = acc.borrow();
        assert!(grads.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_gradient_accumulator_accumulate() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        {
            let grads = acc.borrow();
            assert!((grads[0] - 1.0).abs() < 1e-6);
            assert!((grads[1] - 2.0).abs() < 1e-6);
            assert!((grads[2] - 3.0).abs() < 1e-6);
        }
        // Accumulate again - should add
        acc.accumulate(&[0.5, 0.5, 0.5]);
        {
            let grads = acc.borrow();
            assert!((grads[0] - 1.5).abs() < 1e-6);
            assert!((grads[1] - 2.5).abs() < 1e-6);
            assert!((grads[2] - 3.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_accumulator_accumulate_scaled() {
        let acc = GradientAccumulator::new(2);
        acc.accumulate_scaled(&[4.0, 6.0], 0.5);
        let grads = acc.borrow();
        assert!((grads[0] - 2.0).abs() < 1e-6);
        assert!((grads[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_accumulate_scaled_multiple() {
        let acc = GradientAccumulator::new(2);
        acc.accumulate_scaled(&[1.0, 2.0], 0.1);
        acc.accumulate_scaled(&[3.0, 4.0], 0.1);
        let grads = acc.borrow();
        // 0.1 + 0.3 = 0.4
        assert!((grads[0] - 0.4).abs() < 1e-6);
        // 0.2 + 0.4 = 0.6
        assert!((grads[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_l2_norm() {
        let acc = GradientAccumulator::new(2);
        // [3, 4] → norm = 5
        acc.accumulate(&[3.0, 4.0]);
        assert!((acc.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_l2_norm_zero() {
        let acc = GradientAccumulator::new(4);
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_borrow_mut() {
        let acc = GradientAccumulator::new(2);
        {
            let mut grads = acc.borrow_mut();
            grads[0] = 5.0;
            grads[1] = 12.0;
        }
        assert!((acc.l2_norm() - 13.0).abs() < 1e-6); // sqrt(25 + 144)
    }

    #[test]
    fn test_gradient_accumulator_apply_sgd_update() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[0.1, 0.2, 0.3]);
        let mut params = vec![1.0f32, 2.0f32, 3.0f32];
        acc.apply_sgd_update(&mut params, 0.5);

        assert!((params[0] - 0.95).abs() < 1e-6); // 1.0 - 0.5 * 0.1
        assert!((params[1] - 1.9).abs() < 1e-6); // 2.0 - 0.5 * 0.2
        assert!((params[2] - 2.85).abs() < 1e-6); // 3.0 - 0.5 * 0.3

        // Gradients should be zeroed after update
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_apply_sgd_update_zeros_grads() {
        let acc = GradientAccumulator::new(2);
        acc.accumulate(&[1.0, 1.0]);
        let mut params = vec![0.0f32; 2];
        acc.apply_sgd_update(&mut params, 0.1);
        // After update, gradients must be zero
        assert_eq!(acc.l2_norm(), 0.0);
        let grads = acc.borrow();
        assert!(grads.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_gradient_accumulator_apply_optimizer_update() {
        let acc = GradientAccumulator::new(2);
        acc.accumulate(&[0.2, 0.4]);
        let mut params = vec![1.0f32, 2.0f32];
        let mut optimizer = SGD::new(0.5);
        acc.apply_optimizer_update(&mut params, &mut optimizer);

        // SGD with lr=0.5: param = param - lr * grad
        assert!((params[0] - 0.9).abs() < 1e-5); // 1.0 - 0.5 * 0.2
        assert!((params[1] - 1.8).abs() < 1e-5); // 2.0 - 0.5 * 0.4

        // Gradients should be zeroed after update
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_apply_optimizer_zeros_grads() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[0.5, 0.5, 0.5]);
        let mut params = vec![0.0f32; 3];
        let mut optimizer = SGD::new(0.1);
        acc.apply_optimizer_update(&mut params, &mut optimizer);
        assert_eq!(acc.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_accumulator_multiple_updates() {
        let acc = GradientAccumulator::new(2);
        let mut params = vec![10.0f32, 10.0f32];

        // First iteration
        acc.accumulate(&[1.0, 1.0]);
        acc.apply_sgd_update(&mut params, 0.1);
        assert!((params[0] - 9.9).abs() < 1e-6);

        // Second iteration (gradients were cleared, accumulate fresh)
        assert_eq!(acc.l2_norm(), 0.0);
        acc.accumulate(&[2.0, 2.0]);
        acc.apply_sgd_update(&mut params, 0.1);
        assert!((params[0] - 9.7).abs() < 1e-6); // 9.9 - 0.1 * 2.0
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_gradient_accumulator_accumulate_wrong_length() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0]); // Wrong length: 2 vs 3
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_gradient_accumulator_accumulate_scaled_wrong_length() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate_scaled(&[1.0], 0.1); // Wrong length: 1 vs 3
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_gradient_accumulator_apply_sgd_update_wrong_length() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        let mut params = vec![0.0f32; 2]; // Wrong length: 2 vs 3
        acc.apply_sgd_update(&mut params, 0.1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_gradient_accumulator_apply_optimizer_update_wrong_length() {
        let acc = GradientAccumulator::new(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        let mut params = vec![0.0f32; 2]; // Wrong length: 2 vs 3
        let mut optimizer = SGD::new(0.1);
        acc.apply_optimizer_update(&mut params, &mut optimizer);
    }
}
