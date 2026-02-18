//! Patch Embedding layer for Vision Transformers
//!
//! This module provides a PatchEmbeddingLayer that projects flattened image patches
//! to the model dimension. It wraps a DenseLayer for the linear projection, reusing
//! BLAS-accelerated matrix multiplication and Xavier initialization.

use crate::layers::dense::DenseLayer;
use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::rng::SimpleRng;
use std::any::Any;

/// Patch embedding layer that projects flattened image patches to model dimension.
///
/// This is a thin wrapper around `DenseLayer` that performs the linear projection
/// `y = xW + b` where x is a flattened patch (e.g., 4x4x3 = 48 dims for CIFAR-10)
/// and y is the token embedding (e.g., 128 dims).
///
/// # Architecture
///
/// ```text
/// Input: [batch_size * num_patches, patch_dim]
///   → DenseLayer(patch_dim, d_model)
/// Output: [batch_size * num_patches, d_model]
/// ```
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::PatchEmbeddingLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);
/// assert_eq!(layer.patch_dim(), 48);
/// assert_eq!(layer.d_model(), 128);
/// ```
pub struct PatchEmbeddingLayer {
    patch_dim: usize,
    d_model: usize,
    dense: DenseLayer,
}

impl PatchEmbeddingLayer {
    /// Creates a new patch embedding layer.
    ///
    /// # Arguments
    ///
    /// * `patch_dim` - Flattened patch size (e.g., 4*4*3 = 48 for 4x4 RGB patches)
    /// * `d_model` - Model/embedding dimension
    /// * `rng` - Random number generator for Xavier weight initialization
    pub fn new(patch_dim: usize, d_model: usize, rng: &mut SimpleRng) -> Self {
        Self {
            patch_dim,
            d_model,
            dense: DenseLayer::new(patch_dim, d_model, rng),
        }
    }

    /// Returns the input patch dimension.
    pub fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    /// Returns the output model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl Layer for PatchEmbeddingLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        self.dense.forward(input, output, batch_size);
    }

    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        self.dense
            .backward(input, grad_output, grad_input, batch_size);
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        self.dense.update_parameters(learning_rate);
    }

    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        self.dense.update_with_optimizer(optimizer);
    }

    fn input_size(&self) -> usize {
        self.patch_dim
    }

    fn output_size(&self) -> usize {
        self.d_model
    }

    fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
    }

    fn flops_forward(&self, batch_size: usize) -> u64 {
        self.dense.flops_forward(batch_size)
    }

    fn flops_backward(&self, batch_size: usize) -> u64 {
        self.dense.flops_backward(batch_size)
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embedding_creation() {
        let mut rng = SimpleRng::new(42);
        let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

        assert_eq!(layer.patch_dim(), 48);
        assert_eq!(layer.d_model(), 128);
        assert_eq!(layer.input_size(), 48);
        assert_eq!(layer.output_size(), 128);
    }

    #[test]
    fn test_patch_embedding_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

        // 48 * 128 weights + 128 biases = 6272
        assert_eq!(layer.parameter_count(), 48 * 128 + 128);
    }

    #[test]
    fn test_patch_embedding_forward() {
        let mut rng = SimpleRng::new(42);
        let layer = PatchEmbeddingLayer::new(16, 32, &mut rng);

        let batch_size = 4;
        let input = vec![1.0f32; batch_size * 16];
        let mut output = vec![0.0f32; batch_size * 32];

        layer.forward(&input, &mut output, batch_size);

        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output.iter().any(|&x| x != 0.0));
    }
}
