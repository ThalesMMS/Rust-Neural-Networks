use super::TransformerBlock;
use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::rng::SimpleRng;
use std::any::Any;
use std::cell::RefCell;

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
    cached_activations: RefCell<Vec<Vec<f32>>>,
    cached_batch_size: RefCell<usize>,
    cached_seq_len: RefCell<usize>,
    cached_parameter_version: RefCell<Option<usize>>,
    parameter_version: usize,
}

impl TransformerEncoder {
    /// Constructs a TransformerEncoder composed of `num_layers` stacked TransformerBlock instances.
    ///
    /// Initializes per-layer activation caches to empty, cache metadata to zero/None, and sets the
    /// internal parameter version to 0.
    ///
    /// # Panics
    ///
    /// Panics if `d_model` is not divisible by `num_heads`.
    ///
    /// # Examples
    ///
    /// ```
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
            cached_activations: RefCell::new(Vec::new()),
            cached_batch_size: RefCell::new(0),
            cached_seq_len: RefCell::new(0),
            cached_parameter_version: RefCell::new(None),
            parameter_version: 0,
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

    /// Returns a slice of all stacked transformer blocks.
    pub fn blocks(&self) -> &[TransformerBlock] {
        &self.blocks
    }

    /// Returns all trainable parameter tensors from each contained block.
    pub fn parameter_slices(&self) -> Vec<&[f32]> {
        let mut slices = Vec::new();
        for block in &self.blocks {
            slices.extend(block.parameter_slices());
        }
        slices
    }

    /// Clears cached forward activations and resets cache metadata.
    fn invalidate_activation_cache(&self) {
        self.cached_activations.borrow_mut().clear();
        *self.cached_batch_size.borrow_mut() = 0;
        *self.cached_seq_len.borrow_mut() = 0;
        *self.cached_parameter_version.borrow_mut() = None;
    }
}

impl Layer for TransformerEncoder {
    /// Runs the input sequentially through all stacked transformer blocks.
    ///
    /// Panics if `input.len()` is not divisible by `batch_size * d_model`. Caches per-layer
    /// activations for reuse in the backward pass. If there are no blocks, copies input to output.
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        assert_eq!(
            output.len(),
            input.len(),
            "TransformerEncoder::forward output length must equal input length"
        );
        assert!(
            batch_size > 0,
            "TransformerEncoder::forward batch_size must be > 0"
        );
        assert!(
            self.d_model > 0,
            "TransformerEncoder::forward d_model must be > 0"
        );
        assert_eq!(
            input.len() % (batch_size * self.d_model),
            0,
            "input length must be divisible by batch_size * d_model before computing seq_len"
        );
        let seq_len = input.len() / (batch_size * self.d_model);

        {
            let mut cached_activations = self.cached_activations.borrow_mut();
            cached_activations.resize_with(self.num_layers + 1, Vec::new);
            cached_activations[0].resize(input.len(), 0.0);
            cached_activations[0].copy_from_slice(input);
        }
        *self.cached_batch_size.borrow_mut() = batch_size;
        *self.cached_seq_len.borrow_mut() = seq_len;
        *self.cached_parameter_version.borrow_mut() = Some(self.parameter_version);

        if self.blocks.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        // We need intermediate buffers for passing data between blocks
        let mut buffer1 = input.to_vec();
        let mut buffer2 = vec![0.0f32; input.len()];

        let mut cached_activations = self.cached_activations.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            if i % 2 == 0 {
                block.forward(&buffer1, &mut buffer2, batch_size);
                cached_activations[i + 1].resize(buffer2.len(), 0.0);
                cached_activations[i + 1].copy_from_slice(&buffer2);
            } else {
                block.forward(&buffer2, &mut buffer1, batch_size);
                cached_activations[i + 1].resize(buffer1.len(), 0.0);
                cached_activations[i + 1].copy_from_slice(&buffer1);
            }
        }
        drop(cached_activations);

        // Copy final result to output
        if self.num_layers % 2 == 0 {
            output.copy_from_slice(&buffer1);
        } else {
            output.copy_from_slice(&buffer2);
        }
    }

    /// Propagates gradients back through all stacked transformer blocks.
    ///
    /// Uses cached forward activations when available; otherwise replays a forward pass first.
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        assert!(
            batch_size > 0,
            "TransformerEncoder::backward batch_size must be > 0"
        );
        assert!(
            self.d_model > 0,
            "TransformerEncoder::backward d_model must be > 0"
        );
        assert_eq!(
            input.len() % (batch_size * self.d_model),
            0,
            "TransformerEncoder::backward input length must be divisible by batch_size * d_model before computing seq_len"
        );
        let seq_len = input.len() / (batch_size * self.d_model);
        let total_size = batch_size * seq_len * self.d_model;
        assert_eq!(
            grad_output.len(),
            total_size,
            "TransformerEncoder::backward grad_output length must match batch_size * seq_len * d_model"
        );
        assert_eq!(
            grad_input.len(),
            total_size,
            "TransformerEncoder::backward grad_input length must match batch_size * seq_len * d_model"
        );

        // Backward pass through blocks in reverse order
        let mut grad_buffer1 = grad_output.to_vec();
        let mut grad_buffer2 = vec![0.0f32; input.len()];

        if self.blocks.is_empty() {
            grad_input.copy_from_slice(grad_output);
            return;
        }

        let cached_activations = self.cached_activations.borrow();
        let cache_matches_forward = *self.cached_batch_size.borrow() == batch_size
            && *self.cached_seq_len.borrow() == seq_len
            && *self.cached_parameter_version.borrow() == Some(self.parameter_version)
            && cached_activations.len() == self.num_layers + 1
            && cached_activations
                .iter()
                .all(|activation| activation.len() == input.len())
            && self.blocks.iter().enumerate().all(|(i, block)| {
                block.cached_input_matches(&cached_activations[i], batch_size, seq_len)
            })
            && cached_activations[0] == input;

        if cache_matches_forward {
            for (i, block) in self.blocks.iter().enumerate().rev() {
                let block_input = &cached_activations[i];

                block.backward(block_input, &grad_buffer1, &mut grad_buffer2, batch_size);
                std::mem::swap(&mut grad_buffer1, &mut grad_buffer2);
            }
        } else {
            drop(cached_activations);

            let mut activations = Vec::with_capacity(self.num_layers + 1);
            activations.push(input.to_vec());

            let mut current = input.to_vec();
            let mut next = vec![0.0f32; input.len()];
            for block in &self.blocks {
                block.forward(&current, &mut next, batch_size);
                activations.push(next.clone());
                std::mem::swap(&mut current, &mut next);
            }

            for (i, block) in self.blocks.iter().enumerate().rev() {
                let block_input = &activations[i];

                block.backward(block_input, &grad_buffer1, &mut grad_buffer2, batch_size);
                std::mem::swap(&mut grad_buffer1, &mut grad_buffer2);
            }
        }

        grad_input.copy_from_slice(&grad_buffer1);
    }

    /// Updates all block parameters, increments the parameter version, and invalidates the activation cache.
    fn update_parameters(&mut self, learning_rate: f32) {
        for block in &mut self.blocks {
            block.update_parameters(learning_rate);
        }
        self.parameter_version = self.parameter_version.wrapping_add(1);
        self.invalidate_activation_cache();
    }

    /// Updates all block parameters with the optimizer, increments the parameter version, and invalidates the activation cache.
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        for block in &mut self.blocks {
            block.update_with_optimizer(optimizer);
        }
        self.parameter_version = self.parameter_version.wrapping_add(1);
        self.invalidate_activation_cache();
    }

    fn input_size(&self) -> usize {
        self.d_model
    }

    fn output_size(&self) -> usize {
        self.d_model
    }

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
    use crate::optimizers::Adam;

    /// Generates deterministic sinusoidal test values for unit tests.
    ///
    /// The returned vector has length `len` and element `k` equals `((k + 1) as f32 * scale).sin() * 0.5`.
    ///
    /// # Parameters
    ///
    /// - `len`: number of values to generate.
    /// - `scale`: frequency multiplier applied to each index before taking the sine.
    ///
    /// # Examples
    ///
    /// ```
    /// let vals = varied_values(4, std::f32::consts::PI / 2.0);
    /// assert_eq!(vals.len(), 4);
    /// // values are deterministic; check first element
    /// assert!((vals[0] - ((1.0 * std::f32::consts::PI / 2.0).sin() * 0.5)).abs() < 1e-6);
    /// ```
    fn varied_values(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * scale).sin() * 0.5)
            .collect()
    }

    /// Asserts that two slices have the same length and that their maximum absolute element-wise
    /// difference does not exceed `tol`.
    ///
    /// Panics if the slices differ in length or if the maximum absolute difference is greater than
    /// `tol`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0.0_f32, 1.0, 2.0];
    /// let b = [0.0_f32, 1.01, 1.99];
    /// // max absolute difference is 0.01
    /// assert_slices_close(&a, &b, 0.02);
    /// ```
    fn assert_slices_close(left: &[f32], right: &[f32], tol: f32) {
        assert_eq!(left.len(), right.len());
        let max_abs_diff = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= tol,
            "max_abs_diff {} exceeded tolerance {}",
            max_abs_diff,
            tol
        );
    }

    #[test]
    fn test_transformer_encoder_cached_backward_matches_uncached_replay() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 8;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.07);
        let grad_output = varied_values(len, 0.03);

        let mut cached_rng = SimpleRng::new(77);
        let cached_encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut cached_rng);
        let mut cached_output = vec![0.0f32; len];
        let mut cached_grad_input = vec![0.0f32; len];
        cached_encoder.forward(&input, &mut cached_output, batch_size);

        {
            let cached_activations = cached_encoder.cached_activations.borrow();
            assert_eq!(cached_activations.len(), cached_encoder.num_layers + 1);
            assert_eq!(cached_activations[0], input);
        }
        assert_eq!(
            *cached_encoder.cached_parameter_version.borrow(),
            Some(cached_encoder.parameter_version)
        );

        cached_encoder.backward(&input, &grad_output, &mut cached_grad_input, batch_size);

        {
            let cached_activations = cached_encoder.cached_activations.borrow();
            assert_eq!(cached_activations.len(), cached_encoder.num_layers + 1);
            assert_eq!(cached_activations[0], input);
        }

        let mut replay_rng = SimpleRng::new(77);
        let replay_encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut replay_rng);
        let mut replay_output = vec![0.0f32; len];
        let mut replay_grad_input = vec![0.0f32; len];
        replay_encoder.forward(&input, &mut replay_output, batch_size);
        replay_encoder.invalidate_activation_cache();
        replay_encoder.backward(&input, &grad_output, &mut replay_grad_input, batch_size);

        assert_slices_close(&cached_grad_input, &replay_grad_input, 1e-5);
    }

    #[test]
    #[should_panic(
        expected = "input length must be divisible by batch_size * d_model before computing seq_len"
    )]
    fn test_transformer_encoder_forward_rejects_invalid_input_length() {
        let mut rng = SimpleRng::new(88);
        let encoder = TransformerEncoder::new(1, 8, 2, 16, &mut rng);
        let input = vec![0.1f32; 9];
        let mut output = vec![0.0f32; input.len()];

        encoder.forward(&input, &mut output, 1);
    }

    #[test]
    fn test_transformer_encoder_backward_cache_miss_replays_for_different_input() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 8;
        let len = batch_size * seq_len * d_model;
        let cached_input = varied_values(len, 0.05);
        let backward_input = varied_values(len, 0.11);
        let grad_output = varied_values(len, 0.02);

        let mut miss_rng = SimpleRng::new(99);
        let miss_encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut miss_rng);
        let mut miss_output = vec![0.0f32; len];
        let mut miss_grad_input = vec![0.0f32; len];
        miss_encoder.forward(&cached_input, &mut miss_output, batch_size);
        miss_encoder.backward(
            &backward_input,
            &grad_output,
            &mut miss_grad_input,
            batch_size,
        );

        {
            let cached_activations = miss_encoder.cached_activations.borrow();
            assert_eq!(cached_activations[0], cached_input);
            assert_ne!(cached_activations[0], backward_input);
        }

        let mut baseline_rng = SimpleRng::new(99);
        let baseline_encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut baseline_rng);
        let mut baseline_output = vec![0.0f32; len];
        let mut baseline_grad_input = vec![0.0f32; len];
        baseline_encoder.forward(&backward_input, &mut baseline_output, batch_size);
        baseline_encoder.backward(
            &backward_input,
            &grad_output,
            &mut baseline_grad_input,
            batch_size,
        );

        assert_slices_close(&miss_grad_input, &baseline_grad_input, 1e-5);
    }

    #[test]
    fn test_transformer_encoder_backward_replays_when_block_cache_is_stale() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 8;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.08);
        let stale_input = varied_values(len, 0.17);
        let grad_output = varied_values(len, 0.025);

        let mut encoder_rng = SimpleRng::new(101);
        let encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut encoder_rng);
        let mut output = vec![0.0f32; len];
        let mut grad_input = vec![0.0f32; len];
        encoder.forward(&input, &mut output, batch_size);

        let mut stale_block_output = vec![0.0f32; len];
        encoder.blocks[0].forward(&stale_input, &mut stale_block_output, batch_size);
        assert_eq!(encoder.cached_activations.borrow()[0], input);
        assert!(encoder.blocks[0].cached_input_matches(&stale_input, batch_size, seq_len));

        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert!(encoder.blocks[0].cached_input_matches(&input, batch_size, seq_len));

        let mut baseline_rng = SimpleRng::new(101);
        let baseline_encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut baseline_rng);
        let mut baseline_output = vec![0.0f32; len];
        let mut baseline_grad_input = vec![0.0f32; len];
        baseline_encoder.forward(&input, &mut baseline_output, batch_size);
        baseline_encoder.backward(&input, &grad_output, &mut baseline_grad_input, batch_size);

        assert_slices_close(&grad_input, &baseline_grad_input, 1e-5);
    }

    #[test]
    fn test_transformer_encoder_parameter_updates_invalidate_cache() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 8;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.09);
        let grad_output = varied_values(len, 0.04);

        let mut rng = SimpleRng::new(123);
        let mut encoder = TransformerEncoder::new(2, d_model, 2, 16, &mut rng);
        let mut output = vec![0.0f32; len];
        let mut grad_input = vec![0.0f32; len];
        encoder.forward(&input, &mut output, batch_size);
        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert!(!encoder.cached_activations.borrow().is_empty());
        assert_eq!(*encoder.cached_parameter_version.borrow(), Some(0));

        encoder.update_parameters(0.01);

        assert!(encoder.cached_activations.borrow().is_empty());
        assert_eq!(*encoder.cached_batch_size.borrow(), 0);
        assert_eq!(*encoder.cached_seq_len.borrow(), 0);
        assert_eq!(*encoder.cached_parameter_version.borrow(), None);
        assert_eq!(encoder.parameter_version, 1);

        encoder.forward(&input, &mut output, batch_size);
        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);
        assert!(grad_input.iter().all(|value| value.is_finite()));
        assert_eq!(*encoder.cached_parameter_version.borrow(), Some(1));

        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        encoder.update_with_optimizer(&mut optimizer);

        assert!(encoder.cached_activations.borrow().is_empty());
        assert_eq!(*encoder.cached_batch_size.borrow(), 0);
        assert_eq!(*encoder.cached_seq_len.borrow(), 0);
        assert_eq!(*encoder.cached_parameter_version.borrow(), None);
        assert_eq!(encoder.parameter_version, 2);
    }

    #[test]
    fn test_transformer_encoder_zero_layers_forward_copies_input() {
        let mut rng = SimpleRng::new(5);
        let d_model = 8;
        let encoder = TransformerEncoder::new(0, d_model, 2, 16, &mut rng);
        assert_eq!(encoder.num_layers(), 0);

        let batch_size = 1;
        let seq_len = 4;
        let input = varied_values(batch_size * seq_len * d_model, 0.1);
        let mut output = vec![0.0f32; input.len()];

        encoder.forward(&input, &mut output, batch_size);
        assert_eq!(output, input);
    }

    #[test]
    fn test_transformer_encoder_zero_layers_backward_copies_grad() {
        let mut rng = SimpleRng::new(6);
        let d_model = 8;
        let encoder = TransformerEncoder::new(0, d_model, 2, 16, &mut rng);

        let batch_size = 1;
        let seq_len = 4;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.1);
        let grad_output = varied_values(len, 0.2);
        let mut grad_input = vec![0.0f32; len];

        // No forward needed for empty encoder
        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);
        assert_eq!(grad_input, grad_output);
    }

    #[test]
    fn test_transformer_encoder_single_layer_forward_backward() {
        let mut rng = SimpleRng::new(17);
        let d_model = 8;
        let encoder = TransformerEncoder::new(1, d_model, 2, 16, &mut rng);

        let batch_size = 2;
        let seq_len = 3;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.06);
        let mut output = vec![0.0f32; len];

        encoder.forward(&input, &mut output, batch_size);
        assert!(output.iter().all(|&x| x.is_finite()));

        let grad_output = varied_values(len, 0.04);
        let mut grad_input = vec![0.0f32; len];
        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);
        assert!(grad_input.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_transformer_encoder_blocks_accessor_returns_correct_count() {
        let mut rng = SimpleRng::new(33);
        let num_layers = 4;
        let encoder = TransformerEncoder::new(num_layers, 8, 2, 16, &mut rng);
        assert_eq!(encoder.blocks().len(), num_layers);
    }

    #[test]
    fn test_transformer_encoder_accessors() {
        let mut rng = SimpleRng::new(44);
        let d_model = 16;
        let num_layers = 3;
        let encoder = TransformerEncoder::new(num_layers, d_model, 2, 32, &mut rng);

        assert_eq!(encoder.num_layers(), num_layers);
        assert_eq!(encoder.d_model(), d_model);
        assert_eq!(encoder.input_size(), d_model);
        assert_eq!(encoder.output_size(), d_model);
    }

    #[test]
    fn test_transformer_encoder_parameter_count_is_sum_of_block_counts() {
        let mut rng_enc = SimpleRng::new(50);
        let d_model = 8;
        let num_layers = 3;
        let encoder = TransformerEncoder::new(num_layers, d_model, 2, 16, &mut rng_enc);

        let mut rng_block = SimpleRng::new(50);
        let single_block = super::TransformerBlock::new(d_model, 2, 16, &mut rng_block);
        let expected = single_block.parameter_count() * num_layers;

        assert_eq!(encoder.parameter_count(), expected);
    }

    #[test]
    fn test_transformer_encoder_parameter_slices_count() {
        let mut rng = SimpleRng::new(60);
        let num_layers = 2;
        let encoder = TransformerEncoder::new(num_layers, 8, 2, 16, &mut rng);

        // Each block should contribute the same number of slices
        let total_slices = encoder.parameter_slices();
        let mut rng2 = SimpleRng::new(60);
        let single_block = super::TransformerBlock::new(8, 2, 16, &mut rng2);
        let block_slices = single_block.parameter_slices().len();

        assert_eq!(total_slices.len(), block_slices * num_layers);
    }

    #[test]
    fn test_transformer_encoder_forward_output_length_matches_input() {
        let mut rng = SimpleRng::new(70);
        let d_model = 8;
        let encoder = TransformerEncoder::new(3, d_model, 2, 16, &mut rng);

        let batch_size = 2;
        let seq_len = 4;
        let n = batch_size * seq_len * d_model;
        let input = varied_values(n, 0.05);
        let mut output = vec![0.0f32; n];

        encoder.forward(&input, &mut output, batch_size);
        assert_eq!(output.len(), n);
    }

    #[test]
    #[should_panic(expected = "TransformerEncoder::forward output length must equal input length")]
    fn test_transformer_encoder_forward_rejects_mismatched_output_length() {
        let mut rng = SimpleRng::new(70);
        let d_model = 8;
        let encoder = TransformerEncoder::new(3, d_model, 2, 16, &mut rng);

        let batch_size = 2;
        let seq_len = 4;
        let input = varied_values(batch_size * seq_len * d_model, 0.05);
        let mut output = vec![0.0f32; input.len() + 1];

        encoder.forward(&input, &mut output, batch_size);
    }

    #[test]
    #[should_panic(expected = "TransformerEncoder::backward grad_output length must match")]
    fn test_transformer_encoder_backward_rejects_mismatched_grad_output_length() {
        let mut rng = SimpleRng::new(71);
        let d_model = 8;
        let encoder = TransformerEncoder::new(1, d_model, 2, 16, &mut rng);

        let batch_size = 2;
        let seq_len = 4;
        let input = varied_values(batch_size * seq_len * d_model, 0.05);
        let grad_output = vec![1.0f32; input.len() - 1];
        let mut grad_input = vec![0.0f32; input.len()];

        encoder.backward(&input, &grad_output, &mut grad_input, batch_size);
    }

    #[test]
    fn test_transformer_encoder_forward_is_deterministic() {
        // Same weights and input should produce same output
        let d_model = 8;
        let batch_size = 1;
        let seq_len = 3;
        let len = batch_size * seq_len * d_model;
        let input = varied_values(len, 0.12);

        let mut rng1 = SimpleRng::new(80);
        let enc1 = TransformerEncoder::new(2, d_model, 2, 16, &mut rng1);
        let mut out1 = vec![0.0f32; len];
        enc1.forward(&input, &mut out1, batch_size);

        let mut rng2 = SimpleRng::new(80);
        let enc2 = TransformerEncoder::new(2, d_model, 2, 16, &mut rng2);
        let mut out2 = vec![0.0f32; len];
        enc2.forward(&input, &mut out2, batch_size);

        assert_slices_close(&out1, &out2, 1e-6);
    }

    #[test]
    fn test_transformer_encoder_parameter_version_starts_at_zero() {
        let mut rng = SimpleRng::new(90);
        let encoder = TransformerEncoder::new(2, 8, 2, 16, &mut rng);
        assert_eq!(encoder.parameter_version, 0);
        assert_eq!(*encoder.cached_parameter_version.borrow(), None);
    }
}
