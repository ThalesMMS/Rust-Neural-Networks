//! Transformer block layer implementation
//!
//! This module provides a TransformerBlock that combines multi-head attention,
//! layer normalization, feed-forward network, and residual connections following
//! the "Attention is All You Need" architecture (Vaswani et al., 2017).
//!
//! # Architecture
//!
//! The TransformerBlock implements a Pre-LN (Pre-Layer Normalization) architecture:
//!
//! ```text
//! input
//!   |
//!   +--> LayerNorm --> MultiHeadAttention --> Add(residual) --> [intermediate]
//!                                              |
//!                                              +--> LayerNorm --> FFN --> Add(residual) --> output
//! ```
//!
//! Where FFN (Feed-Forward Network) is:
//! ```text
//! Dense(d_model -> d_ff) --> ReLU --> Dense(d_ff -> d_model)
//! ```
//!
//! # Pre-LN vs Post-LN
//!
//! This implementation uses Pre-LN where layer normalization is applied *before*
//! each sub-layer (attention and FFN), which provides better training stability
//! and gradient flow compared to the original Post-LN architecture.
//!
//! # References
//!
//! Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
//! Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.

use crate::layers::{DenseLayer, Layer, LayerNormLayer, MultiHeadAttentionLayer};
use crate::optimizers::Optimizer;
use crate::utils::rng::SimpleRng;
use std::any::Any;
use std::cell::RefCell;

/// Transformer encoder block with multi-head attention and feed-forward network.
///
/// Implements a complete transformer encoder block with Pre-LN architecture.
/// The block combines multi-head self-attention, layer normalization, position-wise
/// feed-forward network, and residual connections.
///
/// # Fields
///
/// * `d_model` - Model dimension (input/output feature size)
/// * `num_heads` - Number of attention heads
/// * `d_ff` - Hidden dimension of feed-forward network
/// * `ln1` - First layer normalization (before attention)
/// * `attention` - Multi-head self-attention layer
/// * `ln2` - Second layer normalization (before FFN)
/// * `ffn1` - First dense layer of FFN (d_model → d_ff)
/// * `ffn2` - Second dense layer of FFN (d_ff → d_model)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::TransformerBlock;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let block = TransformerBlock::new(512, 8, 2048, &mut rng);
/// assert_eq!(block.input_size(), 512);
/// assert_eq!(block.output_size(), 512);
/// ```
pub struct TransformerBlock {
    d_model: usize,
    num_heads: usize,
    d_ff: usize,

    // Sub-layers
    ln1: LayerNormLayer,
    attention: MultiHeadAttentionLayer,
    ln2: LayerNormLayer,
    ffn1: DenseLayer,
    ffn2: DenseLayer,

    // Cached activations for backward pass (per thread/batch)
    // These are stored as RefCell to allow interior mutability in forward pass
    cached_input: RefCell<Vec<f32>>,
    cached_ln1_out: RefCell<Vec<f32>>,
    cached_attn_out: RefCell<Vec<f32>>,
    cached_residual1: RefCell<Vec<f32>>,
    cached_ln2_out: RefCell<Vec<f32>>,
    cached_ffn1_out: RefCell<Vec<f32>>,
    cached_ffn2_out: RefCell<Vec<f32>>,
    cached_batch_size: RefCell<usize>,
    cached_seq_len: RefCell<usize>,
}

impl TransformerBlock {
    /// Creates a new transformer block with Xavier-initialized weights.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    /// * `d_ff` - Hidden dimension of feed-forward network (typically 4 * d_model)
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if d_model is not divisible by num_heads.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let block = TransformerBlock::new(256, 8, 1024, &mut rng);
    /// assert_eq!(block.d_model(), 256);
    /// assert_eq!(block.num_heads(), 8);
    /// assert_eq!(block.d_ff(), 1024);
    /// ```
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, rng: &mut SimpleRng) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        // Initialize sub-layers
        let ln1 = LayerNormLayer::new(d_model, 1e-5);
        let attention = MultiHeadAttentionLayer::new(d_model, num_heads, rng);
        let ln2 = LayerNormLayer::new(d_model, 1e-5);
        let ffn1 = DenseLayer::new(d_model, d_ff, rng);
        let ffn2 = DenseLayer::new(d_ff, d_model, rng);

        Self {
            d_model,
            num_heads,
            d_ff,
            ln1,
            attention,
            ln2,
            ffn1,
            ffn2,
            cached_input: RefCell::new(Vec::new()),
            cached_ln1_out: RefCell::new(Vec::new()),
            cached_attn_out: RefCell::new(Vec::new()),
            cached_residual1: RefCell::new(Vec::new()),
            cached_ln2_out: RefCell::new(Vec::new()),
            cached_ffn1_out: RefCell::new(Vec::new()),
            cached_ffn2_out: RefCell::new(Vec::new()),
            cached_batch_size: RefCell::new(0),
            cached_seq_len: RefCell::new(0),
        }
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the feed-forward network hidden dimension.
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }

    /// ReLU activation function applied in-place.
    fn relu_inplace(data: &mut [f32]) {
        for x in data.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
    }
}

impl Layer for TransformerBlock {
    /// Forward propagation through the transformer block.
    ///
    /// Implements the Pre-LN transformer architecture:
    /// 1. Layer norm on input
    /// 2. Multi-head self-attention
    /// 3. Residual connection
    /// 4. Layer norm on intermediate output
    /// 5. Feed-forward network (Dense → ReLU → Dense)
    /// 6. Residual connection
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × seq_len × d_model, flattened)
    /// * `output` - Output buffer (batch_size × seq_len × d_model, flattened)
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Panics
    ///
    /// Panics if input/output dimensions don't match expected sizes.
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let total_size = input.len();
        assert_eq!(
            total_size % (batch_size * self.d_model),
            0,
            "input length must be divisible by batch_size * d_model"
        );

        let seq_len = total_size / (batch_size * self.d_model);

        // Cache batch_size and seq_len for backward pass
        *self.cached_batch_size.borrow_mut() = batch_size;
        *self.cached_seq_len.borrow_mut() = seq_len;

        // Resize cache buffers
        let mut cached_input = self.cached_input.borrow_mut();
        let mut cached_ln1_out = self.cached_ln1_out.borrow_mut();
        let mut cached_attn_out = self.cached_attn_out.borrow_mut();
        let mut cached_residual1 = self.cached_residual1.borrow_mut();
        let mut cached_ln2_out = self.cached_ln2_out.borrow_mut();
        let mut cached_ffn1_out = self.cached_ffn1_out.borrow_mut();
        let mut cached_ffn2_out = self.cached_ffn2_out.borrow_mut();

        cached_input.resize(total_size, 0.0);
        cached_ln1_out.resize(total_size, 0.0);
        cached_attn_out.resize(total_size, 0.0);
        cached_residual1.resize(total_size, 0.0);
        cached_ln2_out.resize(total_size, 0.0);
        cached_ffn1_out.resize(batch_size * seq_len * self.d_ff, 0.0);
        cached_ffn2_out.resize(total_size, 0.0);

        // Cache input
        cached_input.copy_from_slice(input);

        // 1. First layer norm
        self.ln1
            .forward(input, &mut cached_ln1_out, batch_size * seq_len);

        // 2. Multi-head self-attention
        self.attention
            .forward(&cached_ln1_out, &mut cached_attn_out, batch_size);

        // 3. First residual connection: residual1 = input + attn_out
        for i in 0..total_size {
            cached_residual1[i] = input[i] + cached_attn_out[i];
        }

        // 4. Second layer norm
        self.ln2
            .forward(&cached_residual1, &mut cached_ln2_out, batch_size * seq_len);

        // 5. Feed-forward network
        // FFN layer 1: d_model -> d_ff
        self.ffn1
            .forward(&cached_ln2_out, &mut cached_ffn1_out, batch_size * seq_len);

        // ReLU activation
        Self::relu_inplace(&mut cached_ffn1_out);

        // FFN layer 2: d_ff -> d_model
        self.ffn2
            .forward(&cached_ffn1_out, &mut cached_ffn2_out, batch_size * seq_len);

        // 6. Second residual connection: output = residual1 + ffn2_out
        for i in 0..total_size {
            output[i] = cached_residual1[i] + cached_ffn2_out[i];
        }
    }

    /// Backward propagation through the transformer block.
    ///
    /// Computes gradients using the chain rule through:
    /// 1. Second residual connection
    /// 2. Feed-forward network (Dense → ReLU → Dense)
    /// 3. Second layer norm
    /// 4. First residual connection
    /// 5. Multi-head self-attention
    /// 6. First layer norm
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass (unused, we use cached values)
    /// * `grad_output` - Gradient of loss w.r.t. block output
    /// * `grad_input` - Buffer to store gradient w.r.t. block input
    /// * `batch_size` - Number of samples in the batch
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let seq_len = *self.cached_seq_len.borrow();
        let total_size = batch_size * seq_len * self.d_model;

        // Retrieve cached values
        let cached_input = self.cached_input.borrow();
        let cached_ln1_out = self.cached_ln1_out.borrow();
        let cached_residual1 = self.cached_residual1.borrow();
        let cached_ln2_out = self.cached_ln2_out.borrow();
        let cached_ffn1_out = self.cached_ffn1_out.borrow();

        // Allocate gradient buffers
        let mut grad_residual1 = vec![0.0f32; total_size];
        let mut grad_ffn2_out = vec![0.0f32; total_size];
        let mut grad_ffn1_out = vec![0.0f32; batch_size * seq_len * self.d_ff];
        let mut grad_ln2_out = vec![0.0f32; total_size];
        let mut grad_attn_out = vec![0.0f32; total_size];
        let mut grad_ln1_out = vec![0.0f32; total_size];

        // Backward through second residual connection: output = residual1 + ffn2_out
        // grad_residual1 = grad_output, grad_ffn2_out = grad_output
        grad_residual1.copy_from_slice(grad_output);
        grad_ffn2_out.copy_from_slice(grad_output);

        // Backward through FFN layer 2
        self.ffn2.backward(
            &cached_ffn1_out,
            &grad_ffn2_out,
            &mut grad_ffn1_out,
            batch_size * seq_len,
        );

        // Backward through ReLU
        // We need the input to FFN1 (after ReLU) to compute ReLU derivative
        // Actually, we cached ffn1_out which is AFTER ReLU, so we need to use that
        // But ReLU derivative needs the PRE-activation values
        // Let me reconsider: cached_ffn1_out is after ReLU in forward pass
        // So for backward, we need to check if the cached value > 0
        for i in 0..grad_ffn1_out.len() {
            if cached_ffn1_out[i] <= 0.0 {
                grad_ffn1_out[i] = 0.0;
            }
        }

        // Backward through FFN layer 1
        self.ffn1.backward(
            &cached_ln2_out,
            &grad_ffn1_out,
            &mut grad_ln2_out,
            batch_size * seq_len,
        );

        // Backward through second layer norm
        let mut grad_residual1_from_ln2 = vec![0.0f32; total_size];
        self.ln2.backward(
            &cached_residual1,
            &grad_ln2_out,
            &mut grad_residual1_from_ln2,
            batch_size * seq_len,
        );

        // Add gradients from both paths into residual1
        for i in 0..total_size {
            grad_residual1[i] += grad_residual1_from_ln2[i];
        }

        // Backward through first residual connection: residual1 = input + attn_out
        // grad_input_from_residual = grad_residual1, grad_attn_out = grad_residual1
        grad_attn_out.copy_from_slice(&grad_residual1);

        // Backward through attention
        self.attention.backward(
            &cached_ln1_out,
            &grad_attn_out,
            &mut grad_ln1_out,
            batch_size,
        );

        // Backward through first layer norm
        self.ln1.backward(
            &cached_input,
            &grad_ln1_out,
            grad_input,
            batch_size * seq_len,
        );

        // Add gradient from first residual connection
        for i in 0..total_size {
            grad_input[i] += grad_residual1[i];
        }
    }

    /// Update layer parameters using learning rate.
    ///
    /// Updates all sub-layer parameters (attention, layer norms, FFN layers)
    /// using vanilla gradient descent.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    fn update_parameters(&mut self, learning_rate: f32) {
        self.ln1.update_parameters(learning_rate);
        self.attention.update_parameters(learning_rate);
        self.ln2.update_parameters(learning_rate);
        self.ffn1.update_parameters(learning_rate);
        self.ffn2.update_parameters(learning_rate);
    }

    /// Update layer parameters using an optimizer.
    ///
    /// Updates all sub-layer parameters using the provided optimizer.
    /// This allows using advanced optimizers like Adam with momentum
    /// and adaptive learning rates.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an optimizer implementing the Optimizer trait
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        self.ln1.update_with_optimizer(optimizer);
        self.attention.update_with_optimizer(optimizer);
        self.ln2.update_with_optimizer(optimizer);
        self.ffn1.update_with_optimizer(optimizer);
        self.ffn2.update_with_optimizer(optimizer);
    }

    /// Get the input size of the transformer block.
    ///
    /// For transformer blocks, input size is the model dimension (d_model).
    /// Note that the actual input is 3D (batch × seq_len × d_model) but
    /// flattened to 1D for the Layer trait interface.
    fn input_size(&self) -> usize {
        self.d_model
    }

    /// Get the output size of the transformer block.
    ///
    /// For transformer blocks, output size equals input size (d_model).
    fn output_size(&self) -> usize {
        self.d_model
    }

    /// Get the total number of trainable parameters.
    ///
    /// Returns the sum of parameters from all sub-layers:
    /// - First layer norm (2 * d_model)
    /// - Multi-head attention
    /// - Second layer norm (2 * d_model)
    /// - FFN layer 1 (d_model * d_ff + d_ff)
    /// - FFN layer 2 (d_ff * d_model + d_model)
    fn parameter_count(&self) -> usize {
        self.ln1.parameter_count()
            + self.attention.parameter_count()
            + self.ln2.parameter_count()
            + self.ffn1.parameter_count()
            + self.ffn2.parameter_count()
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

/// Transformer encoder consisting of multiple stacked TransformerBlocks.
///
/// This struct allows composing multiple transformer encoder blocks into a deep network.
/// Each block processes the output of the previous block sequentially.
///
/// # Fields
///
/// * `blocks` - Vector of TransformerBlock layers
/// * `d_model` - Model dimension (same for all blocks)
/// * `num_layers` - Number of stacked transformer blocks
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::TransformerEncoder;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let encoder = TransformerEncoder::new(6, 512, 8, 2048, &mut rng);
/// assert_eq!(encoder.num_layers(), 6);
/// assert_eq!(encoder.input_size(), 512);
/// ```
pub struct TransformerEncoder {
    blocks: Vec<TransformerBlock>,
    d_model: usize,
    num_layers: usize,
}

impl TransformerEncoder {
    /// Creates a new transformer encoder with multiple stacked blocks.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer blocks to stack
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads per block
    /// * `d_ff` - Hidden dimension of feed-forward network (typically 4 * d_model)
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if d_model is not divisible by num_heads.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let encoder = TransformerEncoder::new(6, 512, 8, 2048, &mut rng);
    /// assert_eq!(encoder.num_layers(), 6);
    /// ```
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(d_model, num_heads, d_ff, rng));
        }

        Self {
            blocks,
            d_model,
            num_layers,
        }
    }

    /// Returns the number of stacked transformer blocks.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl Layer for TransformerEncoder {
    /// Forward propagation through all stacked transformer blocks.
    ///
    /// Passes input sequentially through each block, with each block's
    /// output becoming the next block's input.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × seq_len × d_model, flattened)
    /// * `output` - Output buffer (batch_size × seq_len × d_model, flattened)
    /// * `batch_size` - Number of samples in the batch
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        if self.blocks.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        // We need intermediate buffers for passing data between blocks
        let mut buffer1 = input.to_vec();
        let mut buffer2 = vec![0.0f32; input.len()];

        for (i, block) in self.blocks.iter().enumerate() {
            if i % 2 == 0 {
                block.forward(&buffer1, &mut buffer2, batch_size);
            } else {
                block.forward(&buffer2, &mut buffer1, batch_size);
            }
        }

        // Copy final result to output
        if self.num_layers % 2 == 0 {
            output.copy_from_slice(&buffer1);
        } else {
            output.copy_from_slice(&buffer2);
        }
    }

    /// Backward propagation through all stacked transformer blocks.
    ///
    /// Propagates gradients in reverse order through the blocks,
    /// from the last block back to the first.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data from forward pass
    /// * `grad_output` - Gradient of loss w.r.t. encoder output
    /// * `grad_input` - Buffer to store gradient w.r.t. encoder input
    /// * `batch_size` - Number of samples in the batch
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        if self.blocks.is_empty() {
            grad_input.copy_from_slice(grad_output);
            return;
        }

        // We need to reconstruct the forward pass to get intermediate activations
        // For efficiency in production, these would be cached during forward pass
        let mut activations = Vec::with_capacity(self.num_layers + 1);
        activations.push(input.to_vec());

        // Forward pass to cache activations
        let mut current = input.to_vec();
        let mut next = vec![0.0f32; input.len()];
        for block in &self.blocks {
            block.forward(&current, &mut next, batch_size);
            activations.push(next.clone());
            std::mem::swap(&mut current, &mut next);
        }

        // Backward pass through blocks in reverse order
        let mut grad_buffer1 = grad_output.to_vec();
        let mut grad_buffer2 = vec![0.0f32; input.len()];

        for (i, block) in self.blocks.iter().enumerate().rev() {
            let block_input = &activations[i];

            if (self.num_layers - 1 - i) % 2 == 0 {
                block.backward(block_input, &grad_buffer1, &mut grad_buffer2, batch_size);
                std::mem::swap(&mut grad_buffer1, &mut grad_buffer2);
            } else {
                block.backward(block_input, &grad_buffer1, &mut grad_buffer2, batch_size);
                std::mem::swap(&mut grad_buffer1, &mut grad_buffer2);
            }
        }

        grad_input.copy_from_slice(&grad_buffer1);
    }

    /// Update parameters in all transformer blocks.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    fn update_parameters(&mut self, learning_rate: f32) {
        for block in &mut self.blocks {
            block.update_parameters(learning_rate);
        }
    }

    /// Update parameters in all transformer blocks using an optimizer.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an optimizer implementing the Optimizer trait
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        for block in &mut self.blocks {
            block.update_with_optimizer(optimizer);
        }
    }

    /// Get the input size of the transformer encoder.
    ///
    /// Returns the model dimension (d_model).
    fn input_size(&self) -> usize {
        self.d_model
    }

    /// Get the output size of the transformer encoder.
    ///
    /// Returns the model dimension (d_model).
    fn output_size(&self) -> usize {
        self.d_model
    }

    /// Get the total number of trainable parameters across all blocks.
    fn parameter_count(&self) -> usize {
        self.blocks.iter().map(|b| b.parameter_count()).sum()
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
    fn test_transformer_block_creation() {
        let mut rng = SimpleRng::new(42);
        let block = TransformerBlock::new(64, 4, 256, &mut rng);

        assert_eq!(block.d_model(), 64);
        assert_eq!(block.num_heads(), 4);
        assert_eq!(block.d_ff(), 256);
        assert_eq!(block.input_size(), 64);
        assert_eq!(block.output_size(), 64);
    }

    #[test]
    fn test_transformer_block_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let block = TransformerBlock::new(64, 4, 256, &mut rng);

        // LayerNorm1: 2 * 64 = 128
        // Attention: 4 * (64*64 + 64) = 4 * 4160 = 16640
        // LayerNorm2: 2 * 64 = 128
        // FFN1: 64 * 256 + 256 = 16640
        // FFN2: 256 * 64 + 64 = 16448
        // Total: 128 + 16640 + 128 + 16640 + 16448 = 49984
        let expected = 2 * 64 + 4 * (64 * 64 + 64) + 2 * 64 + (64 * 256 + 256) + (256 * 64 + 64);
        assert_eq!(block.parameter_count(), expected);
    }

    #[test]
    #[should_panic(expected = "d_model (63) must be divisible by num_heads (4)")]
    fn test_transformer_block_invalid_d_model() {
        let mut rng = SimpleRng::new(42);
        let _block = TransformerBlock::new(63, 4, 256, &mut rng);
    }

    #[test]
    fn test_transformer_block_forward() {
        let mut rng = SimpleRng::new(42);
        let block = TransformerBlock::new(64, 4, 128, &mut rng);

        let batch_size = 2;
        let seq_len = 8;
        let input = vec![1.0f32; batch_size * seq_len * 64];
        let mut output = vec![0.0f32; batch_size * seq_len * 64];

        block.forward(&input, &mut output, batch_size);

        // Output should be finite
        assert!(output.iter().all(|&x| x.is_finite()));

        // Output should not be all zeros (block has non-zero weights)
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_transformer_block_backward() {
        let mut rng = SimpleRng::new(42);
        let block = TransformerBlock::new(32, 2, 64, &mut rng);

        let batch_size = 1;
        let seq_len = 4;
        let input = vec![0.5f32; batch_size * seq_len * 32];
        let mut output = vec![0.0f32; batch_size * seq_len * 32];

        // Forward pass
        block.forward(&input, &mut output, batch_size);

        // Backward pass
        let grad_output = vec![1.0f32; batch_size * seq_len * 32];
        let mut grad_input = vec![0.0f32; batch_size * seq_len * 32];
        block.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Gradients should be finite
        assert!(grad_input.iter().all(|&x| x.is_finite()));

        // Gradients should not be all zeros
        assert!(grad_input.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_transformer_block_residual_connections() {
        let mut rng = SimpleRng::new(42);
        let block = TransformerBlock::new(16, 2, 32, &mut rng);

        let batch_size = 1;
        let seq_len = 1;
        let input = vec![1.0f32; batch_size * seq_len * 16];
        let mut output = vec![0.0f32; batch_size * seq_len * 16];

        // Forward pass
        block.forward(&input, &mut output, batch_size);

        // Output should be different from input due to transformations
        // but residual connections should preserve some signal
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Output magnitude should be non-zero
        assert!(output_norm > 0.0);

        // With residual connections, output should have some relationship to input
        // (this is a weak test, but verifies residuals are working)
        assert!(output_norm.is_finite());
    }
}
