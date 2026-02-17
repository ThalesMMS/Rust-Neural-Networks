//! Layer trait definition for neural network layers
//!
//! This module defines the core Layer trait that all layer types must implement.
//! The trait provides a common interface for forward propagation, backward propagation,
//! and parameter updates.

use crate::optimizers::Optimizer;
use std::any::Any;

/// Core trait for neural network layers.
///
/// All layer types (Dense, Conv2D, etc.) implement this trait to provide
/// a uniform interface for forward and backward propagation.
///
/// # Type Parameters
///
/// Layers work with f32 data for compatibility with BLAS operations and GPU acceleration.
///
/// # Example
///
/// ```ignore
/// // Forward pass through a layer
/// let mut output = vec![0.0f32; batch_size * output_size];
/// layer.forward(&input, &mut output, batch_size);
///
/// // Backward pass to compute gradients
/// let mut grad_input = vec![0.0f32; batch_size * input_size];
/// layer.backward(&input, &grad_output, &mut grad_input, batch_size);
/// ```
pub trait Layer {
    /// Forward propagation through the layer.
    ///
    /// Computes the layer output given input data. The layer applies its transformation
    /// (e.g., matrix multiplication + bias for dense layers, convolution for conv layers)
    /// and stores any necessary intermediate values for backward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data flattened as a 1D array (batch_size × input_size)
    /// * `output` - Output buffer to store results (batch_size × output_size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Panics
    ///
    /// Implementations may panic if input/output dimensions don't match expected sizes.
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);

    /// Backward propagation through the layer.
    ///
    /// Computes gradients with respect to inputs and parameters given the gradient
    /// of the loss with respect to outputs. This method:
    /// 1. Computes grad_input (gradient with respect to layer inputs)
    /// 2. Accumulates gradients for weights and biases internally
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass (batch_size × input_size)
    /// * `grad_output` - Gradient of loss w.r.t. layer output (batch_size × output_size)
    /// * `grad_input` - Buffer to store gradient w.r.t. input (batch_size × input_size)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Notes
    ///
    /// - The input should be the same data used in the corresponding forward pass
    /// - Some layers may need to cache activations from forward pass for gradient computation
    /// - Weight and bias gradients are accumulated internally and applied via `update_parameters`
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    );

    /// Update layer parameters using accumulated gradients.
    ///
    /// Applies the gradient descent update rule to weights and biases:
    /// weight = weight - learning_rate * gradient
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Notes
    ///
    /// - This should be called after one or more backward passes
    /// - Implementations should clear accumulated gradients after updating
    fn update_parameters(&mut self, learning_rate: f32);

    /// Update layer parameters using an optimizer.
    ///
    /// Applies the optimizer's update rule to weights and biases. This method
    /// provides more flexibility than `update_parameters`, allowing the use of
    /// advanced optimizers like Adam that maintain momentum and adaptive learning rates.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an optimizer implementing the Optimizer trait
    ///
    /// # Notes
    ///
    /// - This should be called after one or more backward passes
    /// - The optimizer's update method is called separately for weights and biases
    /// - Implementations should clear accumulated gradients after updating
    /// - The optimizer manages its own internal state (momentum, adaptive rates, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rust_neural_networks::optimizers::{Optimizer, Adam};
    /// use rust_neural_networks::layers::Layer;
    ///
    /// let mut layer = DenseLayer::new(784, 512, &mut rng);
    /// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    ///
    /// // In training loop:
    /// layer.forward(&input, &mut output, batch_size);
    /// layer.backward(&input, &grad_output, &mut grad_input, batch_size);
    /// layer.update_with_optimizer(&mut optimizer);
    /// ```
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        // Default implementation uses the optimizer's learning rate with vanilla SGD
        // Layer implementations should override this to properly use optimizer state
        self.update_parameters(optimizer.learning_rate());
    }

    /// Get the input size of the layer.
    ///
    /// Returns the expected number of input features per sample.
    fn input_size(&self) -> usize;

    /// Get the output size of the layer.
    ///
    /// Returns the number of output features per sample.
    fn output_size(&self) -> usize;

    /// Get the number of trainable parameters in the layer.
    ///
    /// Returns the total count of weights and biases.
    /// For example, a dense layer has input_size × output_size weights
    /// plus output_size biases.
    fn parameter_count(&self) -> usize;

    /// Estimated floating-point operations for the forward pass.
    ///
    /// Returns the number of multiply-add operations performed during one
    /// forward pass over a mini-batch of `batch_size` samples.  The default
    /// implementation returns `0`; layer types with analytically known FLOP
    /// counts (e.g. `DenseLayer`, `Conv2DLayer`) override this.
    ///
    /// # Arguments
    ///
    /// * `batch_size` – number of samples in the mini-batch.
    fn flops_forward(&self, _batch_size: usize) -> u64 {
        0
    }

    /// Estimated floating-point operations for the backward pass.
    ///
    /// Returns the number of multiply-add operations performed during one
    /// backward pass over a mini-batch of `batch_size` samples.  The default
    /// implementation returns `0`; layer types with analytically known FLOP
    /// counts override this.
    ///
    /// # Arguments
    ///
    /// * `batch_size` – number of samples in the mini-batch.
    fn flops_backward(&self, _batch_size: usize) -> u64 {
        0
    }

    /// Memory occupied by trainable parameters in bytes.
    ///
    /// Defaults to `parameter_count() * 4` (i.e. one `f32` per parameter).
    /// Layers that store parameters in a different precision or format should
    /// override this method.
    fn parameter_memory_bytes(&self) -> usize {
        self.parameter_count() * 4
    }

    /// Convert the layer to a concrete type via downcasting.
    ///
    /// This method allows downcasting from `Box<dyn Layer>` to specific layer types
    /// like `Conv2DLayer` or `DenseLayer`. This is useful when architecture-based
    /// model construction needs to extract specific layer types.
    ///
    /// # Returns
    ///
    /// A boxed `Any` trait object that can be downcast to the concrete layer type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let layer_box: Box<dyn Layer> = Box::new(DenseLayer::new(784, 512, &mut rng));
    /// let dense_layer = *layer_box
    ///     .into_any()
    ///     .downcast::<DenseLayer>()
    ///     .expect("Failed to downcast to DenseLayer");
    /// ```
    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    /// Get a reference to the layer as an `Any` trait object for downcasting.
    ///
    /// This method allows downcasting from `&dyn Layer` to specific layer types
    /// without consuming or mutating the layer. This is useful for model serialization
    /// or reading layer-specific parameters.
    ///
    /// # Returns
    ///
    /// A reference to an `Any` trait object that can be downcast to the
    /// concrete layer type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let layer_box: Box<dyn Layer> = Box::new(DenseLayer::new(784, 512, &mut rng));
    /// if let Some(dense_layer) = layer_box.as_ref().as_any().downcast_ref::<DenseLayer>() {
    ///     println!("Weights size: {}", dense_layer.weights().len());
    /// }
    /// ```
    fn as_any(&self) -> &dyn Any;

    /// Get a mutable reference to the layer as an `Any` trait object for downcasting.
    ///
    /// This method allows downcasting from `&mut dyn Layer` to specific layer types
    /// like `BatchNormLayer` or `DropoutLayer` without consuming the layer. This is
    /// useful for setting training mode or accessing layer-specific methods.
    ///
    /// # Returns
    ///
    /// A mutable reference to an `Any` trait object that can be downcast to the
    /// concrete layer type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut layer_box: Box<dyn Layer> = Box::new(BatchNormLayer::new(128, 1e-5, 0.9));
    /// if let Some(bn_layer) = layer_box.as_mut().as_any_mut().downcast_mut::<BatchNormLayer>() {
    ///     bn_layer.set_training(false);  // Switch to inference mode
    /// }
    /// ```
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
