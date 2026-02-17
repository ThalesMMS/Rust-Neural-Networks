//! Global average pooling layer implementation
//!
//! This module provides a `GlobalAvgPoolLayer` that averages activations over all
//! spatial positions (height × width) for each channel independently, reducing a
//! spatial feature map to a single vector of per-channel averages.
//!
//! # Global Average Pooling Theory
//!
//! Global Average Pooling (GAP) replaces fully-connected layers at the end of
//! convolutional networks.  For each channel c, it computes:
//!
//! ```text
//! output[c] = (1 / (H * W)) * Σ_{h,w} input[h, w, c]
//! ```
//!
//! This dramatically reduces the number of parameters at the classifier head and
//! provides implicit spatial regularization, making the network more robust to
//! input spatial translations.
//!
//! # Data Format
//!
//! Input is laid out in channels-last / pixel-interleaved order:
//!
//! ```text
//! flat index = b * (H * W * C) + h * (W * C) + w * C + c
//! ```
//!
//! where `b` = batch index, `h` = row, `w` = column, `c` = channel.
//!
//! Output has shape `(batch_size, C)`, stored as:
//!
//! ```text
//! flat index = b * C + c
//! ```
//!
//! # References
//!
//! Lin, M., Chen, Q., & Yan, S. (2013). Network In Network. arXiv:1312.4400.
//! He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
//!   Image Recognition. CVPR.

use crate::layers::Layer;
use crate::optimizers::Optimizer;

/// Global average pooling layer.
///
/// Reduces a spatial feature map of shape `(H, W, C)` per sample to a vector of
/// shape `(C,)` by averaging all spatial positions for each channel.  This layer
/// has **no trainable parameters**.
///
/// # Fields
///
/// * `in_height`  – spatial height H of the input feature map
/// * `in_width`   – spatial width  W of the input feature map
/// * `channels`   – number of channels C
///
/// # Example
///
/// ```
/// use rust_neural_networks::layers::pooling::GlobalAvgPoolLayer;
/// use rust_neural_networks::layers::Layer;
///
/// // Feature map: H=4, W=4, C=8
/// let layer = GlobalAvgPoolLayer::new(4, 4, 8);
/// assert_eq!(layer.input_size(),  4 * 4 * 8); // 128
/// assert_eq!(layer.output_size(), 8);
/// assert_eq!(layer.parameter_count(), 0);
/// ```
pub struct GlobalAvgPoolLayer {
    in_height: usize,
    in_width: usize,
    channels: usize,
}

impl GlobalAvgPoolLayer {
    /// Creates a new global average pooling layer.
    ///
    /// # Arguments
    ///
    /// * `in_height` – spatial height H of the input feature map
    /// * `in_width`  – spatial width  W of the input feature map
    /// * `channels`  – number of input (and output) channels C
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::layers::pooling::GlobalAvgPoolLayer;
    /// let layer = GlobalAvgPoolLayer::new(8, 8, 16);
    /// assert_eq!(layer.in_height(), 8);
    /// assert_eq!(layer.in_width(),  8);
    /// assert_eq!(layer.channels(),  16);
    /// ```
    pub fn new(in_height: usize, in_width: usize, channels: usize) -> Self {
        assert!(in_height > 0, "in_height must be positive");
        assert!(in_width > 0, "in_width must be positive");
        assert!(channels > 0, "channels must be positive");
        Self {
            in_height,
            in_width,
            channels,
        }
    }

    /// Returns the spatial height of the input feature map.
    pub fn in_height(&self) -> usize {
        self.in_height
    }

    /// Returns the spatial width of the input feature map.
    pub fn in_width(&self) -> usize {
        self.in_width
    }

    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Layer for GlobalAvgPoolLayer {
    /// Forward pass: spatial average per channel.
    ///
    /// # Mathematical Operation
    ///
    /// For each sample `b` and channel `c`:
    ///
    /// ```text
    /// output[b * C + c] = (1 / (H * W)) * Σ_{h=0}^{H-1} Σ_{w=0}^{W-1}
    ///                       input[b * H*W*C + h * W*C + w * C + c]
    /// ```
    ///
    /// # Dimensions
    ///
    /// - Input:  `(batch_size × H × W × C)` channels-last
    /// - Output: `(batch_size × C)`
    ///
    /// # Arguments
    ///
    /// * `input`      – flat input slice  (length = `batch_size * H * W * C`)
    /// * `output`     – flat output slice (length = `batch_size * C`)
    /// * `batch_size` – number of samples
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let hw = self.in_height * self.in_width;
        let hwc = hw * self.channels;
        let scale = 1.0_f32 / hw as f32;

        assert_eq!(
            input.len(),
            batch_size * hwc,
            "GlobalAvgPoolLayer forward: input length mismatch: expected {}, got {}",
            batch_size * hwc,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.channels,
            "GlobalAvgPoolLayer forward: output length mismatch: expected {}, got {}",
            batch_size * self.channels,
            output.len()
        );

        for b in 0..batch_size {
            let in_base = b * hwc;
            let out_base = b * self.channels;

            // Zero-initialise output accumulators for this sample
            for c in 0..self.channels {
                output[out_base + c] = 0.0;
            }

            // Accumulate over all spatial positions
            for h in 0..self.in_height {
                for w in 0..self.in_width {
                    let spatial_base = in_base + (h * self.in_width + w) * self.channels;
                    for c in 0..self.channels {
                        output[out_base + c] += input[spatial_base + c];
                    }
                }
            }

            // Divide by H*W to get the mean
            for c in 0..self.channels {
                output[out_base + c] *= scale;
            }
        }
    }

    /// Backward pass: distribute gradient uniformly over spatial positions.
    ///
    /// # Mathematical Operation
    ///
    /// Since forward averages over H×W positions, each position receives an equal
    /// share of the output gradient:
    ///
    /// ```text
    /// grad_input[b * H*W*C + h * W*C + w * C + c]
    ///     = grad_output[b * C + c] / (H * W)
    /// ```
    ///
    /// # Dimensions
    ///
    /// - `grad_output`: `(batch_size × C)`
    /// - `grad_input`:  `(batch_size × H × W × C)` channels-last
    ///
    /// # Arguments
    ///
    /// * `_input`     – unused (forward values not needed for this backward pass)
    /// * `grad_output` – gradient w.r.t. layer output (length = `batch_size * C`)
    /// * `grad_input`  – gradient w.r.t. layer input  (length = `batch_size * H * W * C`)
    /// * `batch_size`  – number of samples
    fn backward(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let hw = self.in_height * self.in_width;
        let hwc = hw * self.channels;
        let scale = 1.0_f32 / hw as f32;

        assert_eq!(
            grad_output.len(),
            batch_size * self.channels,
            "GlobalAvgPoolLayer backward: grad_output length mismatch: expected {}, got {}",
            batch_size * self.channels,
            grad_output.len()
        );
        assert_eq!(
            grad_input.len(),
            batch_size * hwc,
            "GlobalAvgPoolLayer backward: grad_input length mismatch: expected {}, got {}",
            batch_size * hwc,
            grad_input.len()
        );

        for b in 0..batch_size {
            let in_base = b * hwc;
            let out_base = b * self.channels;

            for h in 0..self.in_height {
                for w in 0..self.in_width {
                    let spatial_base = in_base + (h * self.in_width + w) * self.channels;
                    for c in 0..self.channels {
                        grad_input[spatial_base + c] = grad_output[out_base + c] * scale;
                    }
                }
            }
        }
    }

    /// Update parameters (no-op: GlobalAvgPoolLayer has no trainable parameters).
    fn update_parameters(&mut self, _learning_rate: f32) {
        // No parameters to update
    }

    /// Update parameters via optimizer (no-op: GlobalAvgPoolLayer has no trainable parameters).
    fn update_with_optimizer(&mut self, _optimizer: &mut dyn Optimizer) {
        // No parameters to update
    }

    /// Returns the per-sample input size: `H * W * C`.
    fn input_size(&self) -> usize {
        self.in_height * self.in_width * self.channels
    }

    /// Returns the per-sample output size: `C`.
    fn output_size(&self) -> usize {
        self.channels
    }

    /// Returns the number of trainable parameters (always 0).
    fn parameter_count(&self) -> usize {
        0
    }

    /// Returns 0 bytes of parameter memory (no parameters).
    fn parameter_memory_bytes(&self) -> usize {
        0
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_avg_pool_creation() {
        let layer = GlobalAvgPoolLayer::new(4, 4, 8);
        assert_eq!(layer.in_height(), 4);
        assert_eq!(layer.in_width(), 4);
        assert_eq!(layer.channels(), 8);
        assert_eq!(layer.input_size(), 4 * 4 * 8);
        assert_eq!(layer.output_size(), 8);
        assert_eq!(layer.parameter_count(), 0);
        assert_eq!(layer.parameter_memory_bytes(), 0);
    }

    #[test]
    fn test_global_avg_pool_forward_uniform() {
        // H=2, W=2, C=3; all spatial values equal for each channel
        // channel 0 -> 1.0, channel 1 -> 2.0, channel 2 -> 3.0
        let layer = GlobalAvgPoolLayer::new(2, 2, 3);
        // Input layout: [h=0,w=0,c=0..2], [h=0,w=1,c=0..2], [h=1,w=0,c=0..2], [h=1,w=1,c=0..2]
        let input = vec![
            1.0_f32, 2.0, 3.0, // h=0, w=0
            1.0, 2.0, 3.0, // h=0, w=1
            1.0, 2.0, 3.0, // h=1, w=0
            1.0, 2.0, 3.0, // h=1, w=1
        ];
        let mut output = vec![0.0_f32; 3];
        layer.forward(&input, &mut output, 1);

        assert!(
            (output[0] - 1.0).abs() < 1e-6,
            "ch0 expected 1.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 2.0).abs() < 1e-6,
            "ch1 expected 2.0, got {}",
            output[1]
        );
        assert!(
            (output[2] - 3.0).abs() < 1e-6,
            "ch2 expected 3.0, got {}",
            output[2]
        );
    }

    #[test]
    fn test_global_avg_pool_forward_non_uniform() {
        // H=1, W=4, C=2
        // ch0 values: 2, 4, 6, 8  -> mean = 5.0
        // ch1 values: 1, 3, 5, 7  -> mean = 4.0
        let layer = GlobalAvgPoolLayer::new(1, 4, 2);
        let input = vec![
            2.0_f32, 1.0, // w=0
            4.0, 3.0, // w=1
            6.0, 5.0, // w=2
            8.0, 7.0, // w=3
        ];
        let mut output = vec![0.0_f32; 2];
        layer.forward(&input, &mut output, 1);

        assert!(
            (output[0] - 5.0).abs() < 1e-6,
            "ch0 expected 5.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 4.0).abs() < 1e-6,
            "ch1 expected 4.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_global_avg_pool_forward_batch() {
        // batch=2, H=2, W=2, C=2
        // sample 0: all positions -> ch0=1.0, ch1=2.0; expected output ch0=1.0, ch1=2.0
        // sample 1: all positions -> ch0=3.0, ch1=4.0; expected output ch0=3.0, ch1=4.0
        let layer = GlobalAvgPoolLayer::new(2, 2, 2);
        let input = vec![
            // sample 0
            1.0_f32, 2.0, // h=0,w=0
            1.0, 2.0, // h=0,w=1
            1.0, 2.0, // h=1,w=0
            1.0, 2.0, // h=1,w=1
            // sample 1
            3.0, 4.0, // h=0,w=0
            3.0, 4.0, // h=0,w=1
            3.0, 4.0, // h=1,w=0
            3.0, 4.0, // h=1,w=1
        ];
        let mut output = vec![0.0_f32; 4]; // batch=2, C=2
        layer.forward(&input, &mut output, 2);

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
        assert!((output[2] - 3.0).abs() < 1e-6);
        assert!((output[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_avg_pool_backward() {
        // H=2, W=3, C=2  -> HW=6
        // grad_output = [g0, g1] for one sample
        // Every spatial position should receive [g0/6, g1/6]
        let h = 2usize;
        let w = 3usize;
        let c = 2usize;
        let layer = GlobalAvgPoolLayer::new(h, w, c);

        let g0 = 6.0_f32;
        let g1 = 12.0_f32;
        let grad_output = vec![g0, g1];

        let input = vec![0.0_f32; h * w * c]; // unused in backward
        let mut grad_input = vec![0.0_f32; h * w * c];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        let expected_g0 = g0 / (h * w) as f32; // 1.0
        let expected_g1 = g1 / (h * w) as f32; // 2.0
        for pos in 0..(h * w) {
            assert!(
                (grad_input[pos * c] - expected_g0).abs() < 1e-6,
                "pos {} ch0: expected {}, got {}",
                pos,
                expected_g0,
                grad_input[pos * c]
            );
            assert!(
                (grad_input[pos * c + 1] - expected_g1).abs() < 1e-6,
                "pos {} ch1: expected {}, got {}",
                pos,
                expected_g1,
                grad_input[pos * c + 1]
            );
        }
    }

    #[test]
    fn test_global_avg_pool_backward_batch() {
        // batch=2, H=2, W=2, C=1
        // sample 0 grad_output = [4.0],  expected grad_input = [1.0, 1.0, 1.0, 1.0]
        // sample 1 grad_output = [8.0],  expected grad_input = [2.0, 2.0, 2.0, 2.0]
        let layer = GlobalAvgPoolLayer::new(2, 2, 1);
        let grad_output = vec![4.0_f32, 8.0];
        let input = vec![0.0_f32; 2 * 4];
        let mut grad_input = vec![0.0_f32; 2 * 4];
        layer.backward(&input, &grad_output, &mut grad_input, 2);

        for i in 0..4 {
            assert!(
                (grad_input[i] - 1.0).abs() < 1e-6,
                "s0 pos{}: got {}",
                i,
                grad_input[i]
            );
        }
        for i in 4..8 {
            assert!(
                (grad_input[i] - 2.0).abs() < 1e-6,
                "s1 pos{}: got {}",
                i,
                grad_input[i]
            );
        }
    }

    #[test]
    fn test_global_avg_pool() {
        // Combined test used by the verification command
        // Verify forward + backward consistency with a simple 1×1 spatial map
        // (trivial averaging: output == input, gradient == grad_output)
        let layer = GlobalAvgPoolLayer::new(1, 1, 4);
        assert_eq!(layer.input_size(), 4);
        assert_eq!(layer.output_size(), 4);
        assert_eq!(layer.parameter_count(), 0);

        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0_f32; 4];
        layer.forward(&input, &mut output, 1);

        // For 1×1 spatial map, output == input (mean of one element)
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-6,
                "channel {}: expected {}, got {}",
                i,
                inp,
                out
            );
        }

        // Backward: with HW=1, grad_input == grad_output
        let grad_output = vec![0.5_f32, 1.0, 1.5, 2.0];
        let mut grad_input = vec![0.0_f32; 4];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        for (i, (&g_out, &g_in)) in grad_output.iter().zip(grad_input.iter()).enumerate() {
            assert!(
                (g_out - g_in).abs() < 1e-6,
                "channel {}: expected grad_input {}, got {}",
                i,
                g_out,
                g_in
            );
        }
    }
}
