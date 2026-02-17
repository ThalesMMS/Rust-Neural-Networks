//! Residual block layer implementation
//!
//! This module provides a `ResidualBlock` that implements a residual (skip connection)
//! block as introduced in "Deep Residual Learning for Image Recognition" (He et al., 2016).
//!
//! # Architecture
//!
//! Each residual block applies two 3×3 convolution operations with batch normalization
//! and ReLU activations, then adds a shortcut connection from the block's input:
//!
//! ```text
//! input
//!   |
//!   +--> conv1(3x3) --> BN --> ReLU --> conv2(3x3) --> BN --> (+) --> ReLU --> output
//!   |                                                           |
//!   +------------ shortcut (identity or 1x1 projection) -------+
//! ```
//!
//! # Shortcut Types
//!
//! - **Identity shortcut**: Used when `in_channels == out_channels` AND `stride == 1`.
//!   The input is added directly to the main branch output.
//! - **Projection shortcut**: Used when channels differ OR `stride > 1`.
//!   A 1×1 convolution (followed by batch normalization) projects the input to match
//!   the output dimensions before the addition.
//!
//! # Data Format
//!
//! Inputs are expected in channels-last (NHWC) format per the shared library convention:
//! `flat_index = b * H * W * C + h * W * C + w * C + c`
//!
//! # References
//!
//! He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
//! Recognition. CVPR.

use crate::layers::batchnorm::BatchNormLayer;
use crate::layers::conv2d::Conv2DLayer;
use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::rng::SimpleRng;
use std::any::Any;
use std::cell::RefCell;

/// Residual block with two 3×3 convolutions and a skip connection.
///
/// Implements the basic ResNet building block. The block applies:
/// 1. `conv1` (3×3, stride, padding=1) → `bn1` → ReLU
/// 2. `conv2` (3×3, stride=1, padding=1) → `bn2`
/// 3. Shortcut (identity or 1×1 projection)
/// 4. Element-wise addition of main branch and shortcut
/// 5. Final ReLU
///
/// # Fields
///
/// * `in_channels` – Number of input channels
/// * `in_height` – Input spatial height
/// * `in_width` – Input spatial width
/// * `out_channels` – Number of output channels
/// * `out_height` – Output spatial height (= `in_height / stride` for padding=1, kernel=3)
/// * `out_width` – Output spatial width
/// * `conv1`, `bn1` – First conv-BN pair
/// * `conv2`, `bn2` – Second conv-BN pair
/// * `shortcut_conv`, `shortcut_bn` – Optional projection shortcut (None = identity)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::residual::ResidualBlock;
/// use rust_neural_networks::layers::Layer;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// // Identity shortcut: same channels and stride=1
/// let block = ResidualBlock::new(8, 8, 1, 4, 4, &mut rng);
/// assert_eq!(block.input_size(),  8 * 4 * 4);
/// assert_eq!(block.output_size(), 8 * 4 * 4);
/// ```
pub struct ResidualBlock {
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,

    // Main branch
    conv1: Conv2DLayer,
    bn1: BatchNormLayer,
    conv2: Conv2DLayer,
    bn2: BatchNormLayer,

    // Optional projection shortcut (when channels or spatial dims differ)
    shortcut_conv: Option<Conv2DLayer>,
    shortcut_bn: Option<BatchNormLayer>,

    // Cached activations for backward pass (interior mutability via RefCell)
    cached_input: RefCell<Vec<f32>>,           // block input
    cached_hidden: RefCell<Vec<f32>>,          // after conv1 → bn1 → ReLU (input to conv2)
    cached_after_conv2_bn2: RefCell<Vec<f32>>, // after conv2 → bn2 (before shortcut add)
    cached_shortcut: RefCell<Vec<f32>>,        // shortcut path output
    cached_batch_size: RefCell<usize>,
}

impl ResidualBlock {
    /// Creates a new residual block with Xavier-initialized weights.
    ///
    /// Automatically selects identity or projection shortcut based on whether
    /// `in_channels == out_channels && stride == 1`.
    ///
    /// # Arguments
    ///
    /// * `in_channels`  – Number of input channels
    /// * `out_channels` – Number of output channels
    /// * `stride`       – Stride for `conv1` (1 = same spatial dims, 2 = halve spatial dims)
    /// * `in_height`    – Input spatial height
    /// * `in_width`     – Input spatial width
    /// * `rng`          – Random number generator for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if `stride == 0` or if any computed spatial dimension is ≤ 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::residual::ResidualBlock;
    /// use rust_neural_networks::layers::Layer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// // Identity shortcut: same channels, stride=1
    /// let block = ResidualBlock::new(4, 4, 1, 8, 8, &mut rng);
    /// assert_eq!(block.input_size(),  4 * 8 * 8);
    /// assert_eq!(block.output_size(), 4 * 8 * 8);
    ///
    /// // Projection shortcut: stride=2 halves spatial dimensions
    /// let block2 = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
    /// assert_eq!(block2.input_size(),  4 * 8 * 8);
    /// assert_eq!(block2.output_size(), 8 * 4 * 4);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        in_height: usize,
        in_width: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        assert!(stride > 0, "stride must be > 0");

        // Output spatial dimensions for 3×3 conv with padding=1:
        // out_dim = floor((in_dim + 2*1 - 3) / stride) + 1 = floor((in_dim - 1) / stride) + 1
        let out_height = ((in_height as isize + 2 - 3) / stride as isize + 1) as usize;
        let out_width = ((in_width as isize + 2 - 3) / stride as isize + 1) as usize;

        assert!(out_height > 0, "Computed out_height must be positive");
        assert!(out_width > 0, "Computed out_width must be positive");

        let out_size = out_channels * out_height * out_width;

        // Main branch layers
        // conv1: in_channels → out_channels, 3×3, padding=1, stride=stride
        let conv1 = Conv2DLayer::new(
            in_channels,
            out_channels,
            3,
            1,
            stride,
            in_height,
            in_width,
            rng,
        );
        // bn1 normalises the conv1 output: size = out_channels * out_height * out_width
        let bn1 = BatchNormLayer::new(out_size, 1e-5, 0.9);

        // conv2: out_channels → out_channels, 3×3, padding=1, stride=1
        let conv2 = Conv2DLayer::new(
            out_channels,
            out_channels,
            3,
            1,
            1,
            out_height,
            out_width,
            rng,
        );
        // bn2 normalises the conv2 output: same size as conv1 output
        let bn2 = BatchNormLayer::new(out_size, 1e-5, 0.9);

        // Projection shortcut: used when spatial dims or channel count change
        let needs_projection = in_channels != out_channels || stride != 1;
        let (shortcut_conv, shortcut_bn) = if needs_projection {
            // 1×1 conv projects input to (out_channels, out_height, out_width)
            // padding=0, stride matches the main branch
            let sc = Conv2DLayer::new(
                in_channels,
                out_channels,
                1,
                0,
                stride,
                in_height,
                in_width,
                rng,
            );
            let sbn = BatchNormLayer::new(out_size, 1e-5, 0.9);
            (Some(sc), Some(sbn))
        } else {
            (None, None)
        };

        Self {
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut_conv,
            shortcut_bn,
            cached_input: RefCell::new(Vec::new()),
            cached_hidden: RefCell::new(Vec::new()),
            cached_after_conv2_bn2: RefCell::new(Vec::new()),
            cached_shortcut: RefCell::new(Vec::new()),
            cached_batch_size: RefCell::new(0),
        }
    }

    /// Returns the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Returns the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Returns the output spatial height.
    pub fn out_height(&self) -> usize {
        self.out_height
    }

    /// Returns the output spatial width.
    pub fn out_width(&self) -> usize {
        self.out_width
    }

    /// Returns `true` if this block uses a projection shortcut (1×1 conv).
    ///
    /// Identity shortcut is used when `in_channels == out_channels && stride == 1`.
    pub fn has_projection_shortcut(&self) -> bool {
        self.shortcut_conv.is_some()
    }

    /// Sets training mode for all internal BatchNorm layers.
    ///
    /// In training mode, batch statistics are used and running statistics are updated.
    /// In inference mode, accumulated running statistics are used instead.
    ///
    /// # Arguments
    ///
    /// * `training` – `true` for training mode, `false` for inference mode
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::residual::ResidualBlock;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let mut block = ResidualBlock::new(4, 4, 1, 8, 8, &mut rng);
    /// block.set_training(true);
    /// assert!(block.is_training());
    /// block.set_training(false);
    /// assert!(!block.is_training());
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.bn1.set_training(training);
        self.bn2.set_training(training);
        if let Some(ref mut sbn) = self.shortcut_bn {
            sbn.set_training(training);
        }
    }

    /// Returns `true` if the block is in training mode (reflects `bn1` state).
    pub fn is_training(&self) -> bool {
        self.bn1.is_training()
    }

    /// Applies ReLU in-place: sets all negative values to zero.
    fn relu_inplace(data: &mut [f32]) {
        for x in data.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
    }
}

impl Layer for ResidualBlock {
    /// Forward propagation through the residual block.
    ///
    /// Applies the two-conv main branch in parallel with the shortcut, then adds
    /// and applies the final ReLU:
    ///
    /// ```text
    /// output = ReLU( BN2(Conv2(ReLU(BN1(Conv1(input))))) + shortcut(input) )
    /// ```
    ///
    /// Intermediate activations are cached in `RefCell` fields for use in the
    /// backward pass.
    ///
    /// # Arguments
    ///
    /// * `input`      – Flat input (length = `batch_size × input_size()`)
    /// * `output`     – Flat output buffer (length = `batch_size × output_size()`)
    /// * `batch_size` – Number of samples in the batch
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let in_size = self.input_size();
        let out_size = self.output_size();

        assert_eq!(
            input.len(),
            batch_size * in_size,
            "ResidualBlock forward: input length mismatch (expected {}, got {})",
            batch_size * in_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * out_size,
            "ResidualBlock forward: output length mismatch (expected {}, got {})",
            batch_size * out_size,
            output.len()
        );

        *self.cached_batch_size.borrow_mut() = batch_size;

        // ── Cache input for backward ──────────────────────────────────────
        {
            let mut ci = self.cached_input.borrow_mut();
            ci.resize(batch_size * in_size, 0.0);
            ci.copy_from_slice(input);
        }

        // ── Main branch: conv1 → bn1 → ReLU → conv2 → bn2 ───────────────
        let mut conv1_out = vec![0.0f32; batch_size * out_size];
        self.conv1.forward(input, &mut conv1_out, batch_size);

        let mut bn1_out = vec![0.0f32; batch_size * out_size];
        self.bn1.forward(&conv1_out, &mut bn1_out, batch_size);

        // ReLU after bn1 (in-place); bn1_out is now the "hidden" activation
        Self::relu_inplace(&mut bn1_out);

        // Cache hidden = after conv1 → bn1 → ReLU (needed for conv2 backward)
        {
            let mut ch = self.cached_hidden.borrow_mut();
            ch.resize(batch_size * out_size, 0.0);
            ch.copy_from_slice(&bn1_out);
        }

        let mut conv2_out = vec![0.0f32; batch_size * out_size];
        self.conv2.forward(&bn1_out, &mut conv2_out, batch_size);

        let mut bn2_out = vec![0.0f32; batch_size * out_size];
        self.bn2.forward(&conv2_out, &mut bn2_out, batch_size);

        // Cache after_conv2_bn2 (needed for ReLU2 backward and bn2 backward)
        {
            let mut cab = self.cached_after_conv2_bn2.borrow_mut();
            cab.resize(batch_size * out_size, 0.0);
            cab.copy_from_slice(&bn2_out);
        }

        // ── Shortcut branch ───────────────────────────────────────────────
        let mut shortcut_out = vec![0.0f32; batch_size * out_size];

        match (&self.shortcut_conv, &self.shortcut_bn) {
            (Some(sc), Some(sbn)) => {
                // Projection shortcut: 1×1 conv → BN
                let mut sc_out = vec![0.0f32; batch_size * out_size];
                sc.forward(input, &mut sc_out, batch_size);
                sbn.forward(&sc_out, &mut shortcut_out, batch_size);
            }
            _ => {
                // Identity shortcut: pass input directly
                // in_size == out_size is guaranteed by the constructor for this case
                shortcut_out.copy_from_slice(input);
            }
        }

        // Cache shortcut output (needed for ReLU2 backward)
        {
            let mut cs = self.cached_shortcut.borrow_mut();
            cs.resize(batch_size * out_size, 0.0);
            cs.copy_from_slice(&shortcut_out);
        }

        // ── Add shortcut + final ReLU ─────────────────────────────────────
        for i in 0..batch_size * out_size {
            output[i] = bn2_out[i] + shortcut_out[i];
        }
        Self::relu_inplace(output);
    }

    /// Backward propagation through the residual block.
    ///
    /// Computes gradients via the chain rule through:
    /// 1. Final ReLU (using cached pre-activation to reconstruct the mask)
    /// 2. Gradient split at the skip-connection add junction
    /// 3. Main branch: bn2 → conv2 → ReLU1 → bn1 → conv1
    /// 4. Shortcut branch: identity passthrough or projection BN → conv
    /// 5. Gradient sum at the block input
    ///
    /// # Arguments
    ///
    /// * `_input`     – Unused (the block uses its own cached input)
    /// * `grad_output` – Gradient w.r.t. block output (length = `batch_size × output_size()`)
    /// * `grad_input`  – Output buffer for gradient w.r.t. block input
    /// * `batch_size`  – Number of samples in the batch
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let in_size = self.input_size();
        let out_size = self.output_size();

        // Borrow cached activations
        let cached_input = self.cached_input.borrow();
        let cached_hidden = self.cached_hidden.borrow();
        let cached_after_conv2_bn2 = self.cached_after_conv2_bn2.borrow();
        let cached_shortcut = self.cached_shortcut.borrow();

        // ── 1. ReLU2 backward ─────────────────────────────────────────────
        // pre_relu2 = after_conv2_bn2 + shortcut
        // grad_pre_relu2[i] = grad_output[i]  if pre_relu2[i] > 0
        //                    = 0              otherwise
        let mut grad_pre_relu2 = vec![0.0f32; batch_size * out_size];
        for i in 0..batch_size * out_size {
            let pre_relu2 = cached_after_conv2_bn2[i] + cached_shortcut[i];
            grad_pre_relu2[i] = if pre_relu2 > 0.0 { grad_output[i] } else { 0.0 };
        }

        // ── 2. Split gradient at the add junction ─────────────────────────
        // Both branches (main + shortcut) receive the same gradient
        let grad_add = &grad_pre_relu2;

        // ── 3. Main branch backward: bn2 → conv2 → ReLU1 → bn1 → conv1 ──

        // bn2 backward: BatchNormLayer uses its cached statistics (ignores _input)
        let mut grad_conv2_out = vec![0.0f32; batch_size * out_size];
        self.bn2.backward(
            &cached_after_conv2_bn2,
            grad_add,
            &mut grad_conv2_out,
            batch_size,
        );

        // conv2 backward: input to conv2 was cached_hidden
        let mut grad_hidden = vec![0.0f32; batch_size * out_size];
        self.conv2.backward(
            &cached_hidden,
            &grad_conv2_out,
            &mut grad_hidden,
            batch_size,
        );

        // ReLU1 backward: cached_hidden is the output of ReLU1
        // grad passes through where cached_hidden > 0 (was not zeroed by ReLU)
        for i in 0..batch_size * out_size {
            if cached_hidden[i] <= 0.0 {
                grad_hidden[i] = 0.0;
            }
        }

        // bn1 backward: uses cached statistics (ignores _input)
        let mut grad_conv1_out = vec![0.0f32; batch_size * out_size];
        let dummy_bn1_input = vec![0.0f32; batch_size * out_size];
        self.bn1.backward(
            &dummy_bn1_input,
            &grad_hidden,
            &mut grad_conv1_out,
            batch_size,
        );

        // conv1 backward: input to conv1 was cached_input
        let mut grad_input_main = vec![0.0f32; batch_size * in_size];
        self.conv1.backward(
            &cached_input,
            &grad_conv1_out,
            &mut grad_input_main,
            batch_size,
        );

        // ── 4. Shortcut branch backward ────────────────────────────────────
        let mut grad_input_shortcut = vec![0.0f32; batch_size * in_size];

        match (&self.shortcut_conv, &self.shortcut_bn) {
            (Some(sc), Some(sbn)) => {
                // Projection shortcut backward: shortcut_bn → shortcut_conv
                let mut grad_sc_out = vec![0.0f32; batch_size * out_size];
                let dummy_sbn_input = vec![0.0f32; batch_size * out_size];
                sbn.backward(&dummy_sbn_input, grad_add, &mut grad_sc_out, batch_size);
                sc.backward(
                    &cached_input,
                    &grad_sc_out,
                    &mut grad_input_shortcut,
                    batch_size,
                );
            }
            _ => {
                // Identity shortcut: gradient flows straight through
                // in_size == out_size for identity shortcut
                grad_input_shortcut.copy_from_slice(grad_add);
            }
        }

        // ── 5. Sum gradients from both branches ───────────────────────────
        for i in 0..batch_size * in_size {
            grad_input[i] = grad_input_main[i] + grad_input_shortcut[i];
        }
    }

    /// Update parameters in all sub-layers using vanilla SGD.
    fn update_parameters(&mut self, learning_rate: f32) {
        self.conv1.update_parameters(learning_rate);
        self.bn1.update_parameters(learning_rate);
        self.conv2.update_parameters(learning_rate);
        self.bn2.update_parameters(learning_rate);
        if let Some(ref mut sc) = self.shortcut_conv {
            sc.update_parameters(learning_rate);
        }
        if let Some(ref mut sbn) = self.shortcut_bn {
            sbn.update_parameters(learning_rate);
        }
    }

    /// Update parameters in all sub-layers using the provided optimizer.
    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        self.conv1.update_with_optimizer(optimizer);
        self.bn1.update_with_optimizer(optimizer);
        self.conv2.update_with_optimizer(optimizer);
        self.bn2.update_with_optimizer(optimizer);
        if let Some(ref mut sc) = self.shortcut_conv {
            sc.update_with_optimizer(optimizer);
        }
        if let Some(ref mut sbn) = self.shortcut_bn {
            sbn.update_with_optimizer(optimizer);
        }
    }

    /// Returns the per-sample input size: `in_channels × in_height × in_width`.
    fn input_size(&self) -> usize {
        self.in_channels * self.in_height * self.in_width
    }

    /// Returns the per-sample output size: `out_channels × out_height × out_width`.
    fn output_size(&self) -> usize {
        self.out_channels * self.out_height * self.out_width
    }

    /// Returns the total number of trainable parameters in all sub-layers.
    fn parameter_count(&self) -> usize {
        let mut count = self.conv1.parameter_count()
            + self.bn1.parameter_count()
            + self.conv2.parameter_count()
            + self.bn2.parameter_count();
        if let Some(ref sc) = self.shortcut_conv {
            count += sc.parameter_count();
        }
        if let Some(ref sbn) = self.shortcut_bn {
            count += sbn.parameter_count();
        }
        count
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
    use crate::layers::Layer;
    use crate::utils::rng::SimpleRng;

    /// Verify basic construction and dimension accessors.
    #[test]
    fn test_residual_block_creation_identity() {
        let mut rng = SimpleRng::new(42);
        // Identity shortcut: same channels, stride=1
        let block = ResidualBlock::new(4, 4, 1, 8, 8, &mut rng);

        assert_eq!(block.in_channels(), 4);
        assert_eq!(block.out_channels(), 4);
        assert_eq!(block.out_height(), 8);
        assert_eq!(block.out_width(), 8);
        assert_eq!(block.input_size(), 4 * 8 * 8);
        assert_eq!(block.output_size(), 4 * 8 * 8);
        assert!(!block.has_projection_shortcut());

        // Parameter count: conv1 + bn1 + conv2 + bn2 (no shortcut)
        let expected = block.conv1.parameter_count()
            + block.bn1.parameter_count()
            + block.conv2.parameter_count()
            + block.bn2.parameter_count();
        assert_eq!(block.parameter_count(), expected);
        assert!(block.parameter_count() > 0);
    }

    /// Verify projection shortcut is created when channels or stride differ.
    #[test]
    fn test_residual_block_creation_projection() {
        let mut rng = SimpleRng::new(42);
        // Projection shortcut: different channels
        let block = ResidualBlock::new(4, 8, 1, 8, 8, &mut rng);

        assert_eq!(block.input_size(), 4 * 8 * 8);
        assert_eq!(block.output_size(), 8 * 8 * 8);
        assert!(block.has_projection_shortcut());
        assert!(block.parameter_count() > 0);
    }

    /// Verify stride-2 block halves the spatial dimensions.
    #[test]
    fn test_residual_block_creation_stride2() {
        let mut rng = SimpleRng::new(42);
        let block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);

        // out_h = (8 + 2 - 3) / 2 + 1 = 7/2 + 1 = 3 + 1 = 4
        assert_eq!(block.out_height(), 4);
        assert_eq!(block.out_width(), 4);
        assert_eq!(block.input_size(), 4 * 8 * 8);
        assert_eq!(block.output_size(), 8 * 4 * 4);
        assert!(block.has_projection_shortcut());
    }

    /// Forward pass with identity shortcut: output has correct shape.
    #[test]
    fn test_residual_block_forward_identity() {
        let mut rng = SimpleRng::new(123);
        let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        // Output must have the correct length
        assert_eq!(output.len(), batch_size * block.output_size());
        // All outputs are ≥ 0 because of the final ReLU
        assert!(output.iter().all(|&v| v >= 0.0));
    }

    /// Forward pass with projection shortcut (different channels).
    #[test]
    fn test_residual_block_forward_projection() {
        let mut rng = SimpleRng::new(99);
        let mut block = ResidualBlock::new(4, 8, 1, 4, 4, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        assert_eq!(output.len(), batch_size * block.output_size());
        assert!(output.iter().all(|&v| v >= 0.0));
    }

    /// Forward pass with stride-2 block.
    #[test]
    fn test_residual_block_forward_stride2() {
        let mut rng = SimpleRng::new(77);
        let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        assert_eq!(output.len(), batch_size * block.output_size());
        assert!(output.iter().all(|&v| v >= 0.0));
    }

    /// Backward pass with identity shortcut: grad_input has correct shape
    /// and at least some non-zero gradients.
    #[test]
    fn test_residual_block_backward_identity() {
        let mut rng = SimpleRng::new(7);
        let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        let grad_output: Vec<f32> = vec![1.0f32; batch_size * block.output_size()];
        let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
        block.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert_eq!(grad_input.len(), batch_size * block.input_size());
        // At least some gradients must be non-zero
        assert!(
            grad_input.iter().any(|&v| v.abs() > 1e-10),
            "grad_input should have non-zero values"
        );
    }

    /// Backward pass with projection shortcut.
    #[test]
    fn test_residual_block_backward_projection() {
        let mut rng = SimpleRng::new(55);
        let mut block = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        let grad_output: Vec<f32> = vec![1.0f32; batch_size * block.output_size()];
        let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
        block.backward(&input, &grad_output, &mut grad_input, batch_size);

        assert_eq!(grad_input.len(), batch_size * block.input_size());
        assert!(
            grad_input.iter().any(|&v| v.abs() > 1e-10),
            "grad_input should have non-zero values"
        );
    }

    /// set_training() must propagate to BatchNorm layers.
    #[test]
    fn test_residual_block_set_training() {
        let mut rng = SimpleRng::new(1);
        let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);

        block.set_training(true);
        assert!(block.is_training());

        block.set_training(false);
        assert!(!block.is_training());
    }

    /// update_parameters() should not panic and should change weights.
    #[test]
    fn test_residual_block_update_parameters() {
        let mut rng = SimpleRng::new(9);
        let mut block = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
        block.set_training(true);

        let batch_size = 2;
        let input: Vec<f32> = (0..batch_size * block.input_size())
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut output = vec![0.0f32; batch_size * block.output_size()];
        block.forward(&input, &mut output, batch_size);

        let grad_output: Vec<f32> = vec![0.01f32; batch_size * block.output_size()];
        let mut grad_input = vec![0.0f32; batch_size * block.input_size()];
        block.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Should not panic
        block.update_parameters(0.01);
    }

    /// Combined test used by the verification command.
    #[test]
    fn test_residual_block() {
        let mut rng = SimpleRng::new(42);

        // ── Identity block ────────────────────────────────────────────────
        let mut block_id = ResidualBlock::new(4, 4, 1, 4, 4, &mut rng);
        block_id.set_training(true);

        assert_eq!(block_id.input_size(), 4 * 4 * 4);
        assert_eq!(block_id.output_size(), 4 * 4 * 4);
        assert!(!block_id.has_projection_shortcut());
        assert!(block_id.parameter_count() > 0);

        let batch = 2;
        let input_id: Vec<f32> = (0..batch * block_id.input_size())
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut out_id = vec![0.0f32; batch * block_id.output_size()];
        block_id.forward(&input_id, &mut out_id, batch);

        // Output shape and non-negativity (ReLU output)
        assert_eq!(out_id.len(), batch * block_id.output_size());
        assert!(out_id.iter().all(|&v| v >= 0.0));

        // Backward pass
        let grad_out_id = vec![1.0f32; batch * block_id.output_size()];
        let mut grad_in_id = vec![0.0f32; batch * block_id.input_size()];
        block_id.backward(&input_id, &grad_out_id, &mut grad_in_id, batch);

        assert_eq!(grad_in_id.len(), batch * block_id.input_size());
        assert!(
            grad_in_id.iter().any(|&v| v.abs() > 1e-10),
            "Identity block: expected non-zero grad_input"
        );

        // ── Projection block (stride=2) ──────────────────────────────────
        let mut block_proj = ResidualBlock::new(4, 8, 2, 8, 8, &mut rng);
        block_proj.set_training(true);

        assert_eq!(block_proj.input_size(), 4 * 8 * 8);
        assert_eq!(block_proj.output_size(), 8 * 4 * 4);
        assert!(block_proj.has_projection_shortcut());
        assert!(block_proj.parameter_count() > 0);

        let input_proj: Vec<f32> = (0..batch * block_proj.input_size())
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let mut out_proj = vec![0.0f32; batch * block_proj.output_size()];
        block_proj.forward(&input_proj, &mut out_proj, batch);

        assert_eq!(out_proj.len(), batch * block_proj.output_size());
        assert!(out_proj.iter().all(|&v| v >= 0.0));

        let grad_out_proj = vec![1.0f32; batch * block_proj.output_size()];
        let mut grad_in_proj = vec![0.0f32; batch * block_proj.input_size()];
        block_proj.backward(&input_proj, &grad_out_proj, &mut grad_in_proj, batch);

        assert_eq!(grad_in_proj.len(), batch * block_proj.input_size());
        assert!(
            grad_in_proj.iter().any(|&v| v.abs() > 1e-10),
            "Projection block: expected non-zero grad_input"
        );

        // ── Training mode toggle ─────────────────────────────────────────
        block_id.set_training(false);
        assert!(!block_id.is_training());
        block_id.set_training(true);
        assert!(block_id.is_training());
    }
}
