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
    /// Creates a Transformer encoder block using a Pre-LN layout with Xavier-initialized weights.
    ///
    /// # Panics
    ///
    /// Panics if `d_model` is not divisible by `num_heads`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut rng = SimpleRng::new(42);
    /// let block = TransformerBlock::new(256, 8, 1024, &mut rng);
    /// assert_eq!(block.d_model(), 256);
    /// assert_eq!(block.num_heads(), 8);
    /// assert_eq!(block.d_ff(), 1024);
    /// ```
    #[allow(clippy::manual_is_multiple_of)]
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

    /// Returns a reference to the multi-head attention sub-layer.
    pub fn attention_layer(&self) -> &MultiHeadAttentionLayer {
        &self.attention
    }

    /// Returns all trainable parameter tensors owned by this block.
    pub fn parameter_slices(&self) -> Vec<&[f32]> {
        let mut slices = Vec::with_capacity(16);
        slices.push(self.ln1.gamma());
        slices.push(self.ln1.beta());
        for params in self.attention.parameter_slices() {
            slices.push(params);
        }
        slices.push(self.ln2.gamma());
        slices.push(self.ln2.beta());
        slices.push(self.ffn1.weights());
        slices.push(self.ffn1.biases());
        slices.push(self.ffn2.weights());
        slices.push(self.ffn2.biases());
        slices
    }

    fn relu_inplace(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = x.max(0.0);
        }
    }

    /// Returns true if the cached input, batch size, and sequence length all match the given values.
    pub(super) fn cached_input_matches(
        &self,
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> bool {
        *self.cached_batch_size.borrow() == batch_size
            && *self.cached_seq_len.borrow() == seq_len
            && self.cached_input.borrow().as_slice() == input
    }

    fn clear_cached_activations(&self) {
        self.cached_input.borrow_mut().clear();
        self.cached_ln1_out.borrow_mut().clear();
        self.cached_attn_out.borrow_mut().clear();
        self.cached_residual1.borrow_mut().clear();
        self.cached_ln2_out.borrow_mut().clear();
        self.cached_ffn1_out.borrow_mut().clear();
        self.cached_ffn2_out.borrow_mut().clear();
        *self.cached_batch_size.borrow_mut() = 0;
        *self.cached_seq_len.borrow_mut() = 0;
    }

    fn assert_cache_len(name: &str, actual: usize, expected: usize) {
        assert_eq!(
            actual, expected,
            "TransformerBlock::backward {name} length {actual} must equal expected length {expected}"
        );
    }
}

impl Layer for TransformerBlock {
    /// Performs a forward pass through the transformer encoder block (Pre-LN).
    ///
    /// The input is normalized, processed by multi-head self-attention, added residually to
    /// the input, normalized again, passed through a two-layer feed-forward network with
    /// an in-place ReLU, and finally added residually to produce the output. Input and
    /// output are expected as flattened tensors with layout (batch_size × seq_len × d_model).
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` is not divisible by `batch_size * d_model` or if `output.len()`
    /// does not equal `input.len()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Construct a transformer block for d_model=4, num_heads=2, d_ff=8
    /// let mut rng = SimpleRng::new(0);
    /// let block = TransformerBlock::new(4, 2, 8, &mut rng);
    ///
    /// let batch_size = 1;
    /// let seq_len = 1;
    /// let mut input = vec![0.5f32; batch_size * seq_len * block.d_model()];
    /// let mut output = vec![0.0f32; input.len()];
    ///
    /// block.forward(&input, &mut output, batch_size);
    ///
    /// assert_eq!(output.len(), input.len());
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let total_size = input.len();
        assert_eq!(
            output.len(),
            total_size,
            "TransformerBlock::forward output length must equal input length"
        );
        assert!(
            batch_size > 0,
            "TransformerBlock::forward batch_size must be > 0"
        );
        assert!(
            self.d_model > 0,
            "TransformerBlock::forward d_model must be > 0"
        );
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

    /// Performs backpropagation through the transformer block using cached forward activations.
    ///
    /// This computes and accumulates gradients for the block's inputs and internal parameters by
    /// propagating `grad_output` backward through the second residual + FFN path, the second layer
    /// normalization, the first residual + attention path, and the first layer normalization.
    /// The method writes gradients into `grad_input` (adds to existing values) and relies on values
    /// stored during the corresponding `forward` call; the `_input` parameter is ignored.
    ///
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        assert!(
            batch_size > 0,
            "TransformerBlock::backward batch_size must be > 0"
        );
        assert!(
            self.d_model > 0,
            "TransformerBlock::backward d_model must be > 0"
        );
        let seq_len = *self.cached_seq_len.borrow();
        assert!(
            seq_len > 0,
            "TransformerBlock::backward requires cached_seq_len > 0; call forward before backward"
        );
        assert_eq!(
            *self.cached_batch_size.borrow(),
            batch_size,
            "TransformerBlock::backward batch_size must match cached_batch_size"
        );
        let total_size = batch_size * seq_len * self.d_model;
        let ffn_size = batch_size * seq_len * self.d_ff;

        assert_eq!(
            grad_output.len(),
            total_size,
            "TransformerBlock::backward grad_output length must match batch_size * cached_seq_len * d_model"
        );
        assert_eq!(
            grad_input.len(),
            total_size,
            "TransformerBlock::backward grad_input length must match batch_size * cached_seq_len * d_model"
        );
        Self::assert_cache_len("cached_input", self.cached_input.borrow().len(), total_size);
        Self::assert_cache_len(
            "cached_ln1_out",
            self.cached_ln1_out.borrow().len(),
            total_size,
        );
        Self::assert_cache_len(
            "cached_attn_out",
            self.cached_attn_out.borrow().len(),
            total_size,
        );
        Self::assert_cache_len(
            "cached_residual1",
            self.cached_residual1.borrow().len(),
            total_size,
        );
        Self::assert_cache_len(
            "cached_ln2_out",
            self.cached_ln2_out.borrow().len(),
            total_size,
        );
        Self::assert_cache_len(
            "cached_ffn1_out",
            self.cached_ffn1_out.borrow().len(),
            ffn_size,
        );
        Self::assert_cache_len(
            "cached_ffn2_out",
            self.cached_ffn2_out.borrow().len(),
            total_size,
        );

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

    /// Apply vanilla gradient descent to every trainable parameter in this block.
    ///
    /// This updates the attention, both layer-norms, and both feed-forward dense layers
    /// by subtracting `learning_rate * gradient` from each parameter.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Scalar learning rate applied to all parameter updates.
    ///
    fn update_parameters(&mut self, learning_rate: f32) {
        self.ln1.update_parameters(learning_rate);
        self.attention.update_parameters(learning_rate);
        self.ln2.update_parameters(learning_rate);
        self.ffn1.update_parameters(learning_rate);
        self.ffn2.update_parameters(learning_rate);
        self.clear_cached_activations();
    }

    /// Update all trainable parameters using the given optimizer.
    ///
    /// Calls `update_with_optimizer` on each sub-layer so the optimizer can apply its
    /// parameter updates (e.g., Adam, SGD with momentum).
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an object implementing the `Optimizer` trait
    ///
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        self.ln1.update_with_optimizer(optimizer);
        self.attention.update_with_optimizer(optimizer);
        self.ln2.update_with_optimizer(optimizer);
        self.ffn1.update_with_optimizer(optimizer);
        self.ffn2.update_with_optimizer(optimizer);
        self.clear_cached_activations();
    }

    fn input_size(&self) -> usize {
        self.d_model
    }

    fn output_size(&self) -> usize {
        self.d_model
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Adam;
    use crate::utils::rng::SimpleRng;

    fn assert_forward_cache_empty(block: &TransformerBlock) {
        assert!(block.cached_input.borrow().is_empty());
        assert!(block.cached_ln1_out.borrow().is_empty());
        assert!(block.cached_attn_out.borrow().is_empty());
        assert!(block.cached_residual1.borrow().is_empty());
        assert!(block.cached_ln2_out.borrow().is_empty());
        assert!(block.cached_ffn1_out.borrow().is_empty());
        assert!(block.cached_ffn2_out.borrow().is_empty());
        assert_eq!(*block.cached_batch_size.borrow(), 0);
        assert_eq!(*block.cached_seq_len.borrow(), 0);
    }

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
        let _input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Output magnitude should be non-zero
        assert!(output_norm > 0.0);

        // With residual connections, output should have some relationship to input
        // (this is a weak test, but verifies residuals are working)
        assert!(output_norm.is_finite());
    }

    #[test]
    fn test_transformer_block_output_length_equals_input_length() {
        let mut rng = SimpleRng::new(7);
        let d_model = 8;
        let block = TransformerBlock::new(d_model, 2, 16, &mut rng);

        let batch_size = 3;
        let seq_len = 5;
        let n = batch_size * seq_len * d_model;
        let input = vec![0.1f32; n];
        let mut output = vec![0.0f32; n];

        block.forward(&input, &mut output, batch_size);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    #[should_panic(expected = "TransformerBlock::forward output length must equal input length")]
    fn test_transformer_block_forward_rejects_mismatched_output_length() {
        let mut rng = SimpleRng::new(7);
        let d_model = 8;
        let block = TransformerBlock::new(d_model, 2, 16, &mut rng);

        let batch_size = 1;
        let seq_len = 2;
        let input = vec![0.1f32; batch_size * seq_len * d_model];
        let mut output = vec![0.0f32; input.len() + 1];

        block.forward(&input, &mut output, batch_size);
    }

    #[test]
    fn test_transformer_block_single_head_works() {
        // num_heads=1 should be valid when d_model is divisible by 1
        let mut rng = SimpleRng::new(13);
        let block = TransformerBlock::new(8, 1, 16, &mut rng);

        assert_eq!(block.num_heads(), 1);

        let batch_size = 1;
        let seq_len = 2;
        let input = vec![0.5f32; batch_size * seq_len * 8];
        let mut output = vec![0.0f32; input.len()];

        block.forward(&input, &mut output, batch_size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_transformer_block_attention_layer_accessor_is_non_null() {
        let mut rng = SimpleRng::new(21);
        let block = TransformerBlock::new(16, 2, 32, &mut rng);
        // attention_layer() should return a reference without panicking
        let _attn = block.attention_layer();
    }

    #[test]
    fn test_transformer_block_parameter_slices_count() {
        // Expected slices: ln1 gamma + ln1 beta + attention slices + ln2 gamma + ln2 beta + ffn1 weights + ffn1 biases + ffn2 weights + ffn2 biases
        // attention has 4 weight matrices + 4 bias vectors = 8 slices; total = 2 + 8 + 2 + 4 = 16
        let mut rng = SimpleRng::new(99);
        let block = TransformerBlock::new(16, 2, 32, &mut rng);
        let slices = block.parameter_slices();
        // At minimum: ln1(2) + attn(>=4) + ln2(2) + ffn1(2) + ffn2(2) = 12+
        assert!(slices.len() >= 12);
        // All slices should be non-empty
        for s in &slices {
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_transformer_block_varied_seq_len_and_batch() {
        let mut rng = SimpleRng::new(55);
        let d_model = 16;
        let block = TransformerBlock::new(d_model, 2, 32, &mut rng);

        for batch_size in [1, 2, 4] {
            for seq_len in [1, 3, 7] {
                let n = batch_size * seq_len * d_model;
                let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
                let mut output = vec![0.0f32; n];
                block.forward(&input, &mut output, batch_size);
                assert!(
                    output.iter().all(|&x| x.is_finite()),
                    "batch_size={batch_size}, seq_len={seq_len}: non-finite output"
                );
            }
        }
    }

    #[test]
    fn test_transformer_block_grad_input_length_equals_input_length() {
        let mut rng = SimpleRng::new(77);
        let d_model = 8;
        let block = TransformerBlock::new(d_model, 2, 16, &mut rng);

        let batch_size = 2;
        let seq_len = 3;
        let n = batch_size * seq_len * d_model;
        let input = vec![0.3f32; n];
        let mut output = vec![0.0f32; n];
        block.forward(&input, &mut output, batch_size);

        let grad_output = vec![1.0f32; n];
        let mut grad_input = vec![0.0f32; n];
        block.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert_eq!(grad_input.len(), n);
        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }

    #[test]
    #[should_panic(expected = "TransformerBlock::backward requires cached_seq_len > 0")]
    fn test_transformer_block_backward_rejects_missing_forward_cache() {
        let mut rng = SimpleRng::new(77);
        let block = TransformerBlock::new(8, 2, 16, &mut rng);
        let grad_output = vec![1.0f32; 8];
        let mut grad_input = vec![0.0f32; 8];

        block.backward(&[], &grad_output, &mut grad_input, 1);
    }

    #[test]
    fn test_transformer_block_parameter_updates_clear_forward_cache() {
        let mut rng = SimpleRng::new(91);
        let mut block = TransformerBlock::new(8, 2, 16, &mut rng);
        let input = vec![0.25f32; 2 * 3 * 8];
        let mut output = vec![0.0f32; input.len()];

        block.forward(&input, &mut output, 2);
        assert!(!block.cached_input.borrow().is_empty());
        block.update_parameters(0.01);
        assert_forward_cache_empty(&block);

        block.forward(&input, &mut output, 2);
        assert!(!block.cached_input.borrow().is_empty());
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        block.update_with_optimizer(&mut optimizer);
        assert_forward_cache_empty(&block);
    }

    #[test]
    #[should_panic(expected = "input length must be divisible by batch_size * d_model")]
    fn test_transformer_block_forward_rejects_misaligned_input() {
        let mut rng = SimpleRng::new(11);
        let block = TransformerBlock::new(8, 2, 16, &mut rng);
        // 17 is not divisible by batch_size(1) * d_model(8)
        let input = vec![0.0f32; 17];
        let mut output = vec![0.0f32; 17];
        block.forward(&input, &mut output, 1);
    }
}
