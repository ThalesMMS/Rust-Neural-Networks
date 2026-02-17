//! INT8 quantization for neural network weight compression
//!
//! Provides symmetric min-max quantization to reduce model weight precision
//! from f32 to i8, reducing memory footprint by approximately 4x.
//!
//! # Overview
//!
//! Symmetric INT8 quantization maps floating-point weights to the range [-127, 127]
//! using a scale factor derived from the maximum absolute weight value.
//!
//! The quantization formula is:
//! - `scale = max(|w|) / 127.0`
//! - `q = clamp(round(w / scale), -127, 127)`
//!
//! Dequantization reconstructs approximate floating-point values:
//! - `w ≈ q * scale`
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::compression::quantization::{
//!     quantize_weights, dequantize_weights, quantization_error,
//! };
//!
//! let weights = vec![0.5_f32, -0.25, 1.0, -1.0, 0.0];
//! let (quantized, params) = quantize_weights(&weights);
//!
//! // Scale should be 1.0 / 127.0 ≈ 0.00787
//! assert!((params.scale - 1.0 / 127.0).abs() < 1e-5);
//! assert_eq!(params.zero_point, 0);
//!
//! let dequantized = dequantize_weights(&quantized, &params);
//! let mse = quantization_error(&weights, &params, &quantized);
//! assert!(mse < 1e-4);
//! ```

use crate::layers::{DenseLayer, Layer};
use std::any::Any;

/// Parameters describing how floating-point weights are mapped to INT8.
///
/// For symmetric quantization, `zero_point` is always 0. The `scale` factor
/// converts between the quantized integer domain and the float domain.
///
/// # Fields
///
/// * `scale` - Scaling factor: `max(|w|) / 127.0`
/// * `zero_point` - Offset in the quantized domain (always 0 for symmetric quantization)
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationParams {
    /// Scaling factor mapping between float and INT8 domains.
    pub scale: f32,
    /// Zero-point offset in the quantized domain (0 for symmetric quantization).
    pub zero_point: i32,
}

/// Quantize a slice of f32 weights to INT8 using symmetric min-max quantization.
///
/// Computes a symmetric scale factor as `max(|w|) / 127` and maps each weight
/// to the nearest integer in `[-127, 127]`. The zero point is always 0.
///
/// If the input slice is empty or all weights are zero, the scale defaults to 1.0
/// to avoid division by zero.
///
/// # Arguments
///
/// * `weights` - Slice of floating-point weights to quantize
///
/// # Returns
///
/// A tuple of:
/// - `Vec<i8>` - Quantized weights in the range `[-127, 127]`
/// - `QuantizationParams` - The scale and zero point used for quantization
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::quantization::quantize_weights;
///
/// let weights = vec![1.0_f32, -1.0, 0.5, -0.5];
/// let (quantized, params) = quantize_weights(&weights);
///
/// assert_eq!(params.zero_point, 0);
/// assert!((params.scale - 1.0 / 127.0).abs() < 1e-6);
/// assert_eq!(quantized[0], 127);
/// assert_eq!(quantized[1], -127);
/// ```
pub fn quantize_weights(weights: &[f32]) -> (Vec<i8>, QuantizationParams) {
    // Find maximum absolute value to determine scale
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);

    // Avoid division by zero: use scale=1.0 when all weights are zero
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

    let params = QuantizationParams {
        scale,
        zero_point: 0,
    };

    // Quantize each weight: q = clamp(round(w / scale), -127, 127)
    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let q = (w / scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();

    (quantized, params)
}

/// Dequantize INT8 weights back to approximate f32 values.
///
/// Reconstructs floating-point weights by multiplying each quantized value by
/// the scale factor and adding the zero point (which is 0 for symmetric quantization).
///
/// # Arguments
///
/// * `quantized` - Slice of quantized INT8 weights
/// * `params` - The quantization parameters used during quantization
///
/// # Returns
///
/// A `Vec<f32>` of reconstructed floating-point values. These are approximations
/// of the original weights — rounding error is unavoidable.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::quantization::{
///     QuantizationParams, dequantize_weights,
/// };
///
/// let params = QuantizationParams { scale: 1.0 / 127.0, zero_point: 0 };
/// let quantized: Vec<i8> = vec![127, -127, 64, -64, 0];
/// let dequantized = dequantize_weights(&quantized, &params);
///
/// assert!((dequantized[0] - 1.0).abs() < 1e-5);
/// assert!((dequantized[1] + 1.0).abs() < 1e-5);
/// assert!((dequantized[4]).abs() < 1e-10);
/// ```
pub fn dequantize_weights(quantized: &[i8], params: &QuantizationParams) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as i32 - params.zero_point) as f32 * params.scale)
        .collect()
}

/// Compute the mean squared error (MSE) between original and quantized weights.
///
/// This measures the quantization error introduced by the INT8 approximation.
/// A lower MSE indicates better fidelity to the original weights.
///
/// MSE = (1/N) * sum((original_i - dequantized_i)^2)
///
/// # Arguments
///
/// * `original` - Original f32 weights before quantization
/// * `params` - Quantization parameters used during quantization
/// * `quantized` - Quantized INT8 weights
///
/// # Returns
///
/// The mean squared error as an f32. Returns 0.0 if the input slice is empty.
///
/// # Panics
///
/// Panics if `original` and `quantized` have different lengths.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::quantization::{
///     quantize_weights, quantization_error,
/// };
///
/// let weights = vec![1.0_f32, -1.0, 0.5, -0.5, 0.0];
/// let (quantized, params) = quantize_weights(&weights);
/// let mse = quantization_error(&weights, &params, &quantized);
///
/// // Quantization error should be very small for these values
/// assert!(mse < 1e-4);
/// ```
pub fn quantization_error(original: &[f32], params: &QuantizationParams, quantized: &[i8]) -> f32 {
    assert_eq!(
        original.len(),
        quantized.len(),
        "original and quantized slices must have the same length"
    );

    if original.is_empty() {
        return 0.0;
    }

    let dequantized = dequantize_weights(quantized, params);
    let sum_sq_err: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(&orig, &dq)| {
            let diff = orig - dq;
            diff * diff
        })
        .sum();

    sum_sq_err / original.len() as f32
}

/// A dense layer with INT8-quantized weights for memory-efficient inference.
///
/// Stores the weight matrix as `Vec<i8>` to achieve approximately 4× memory
/// reduction compared to `f32` weights. Biases are kept in `f32` precision
/// because they are few in number and benefit from higher precision.
///
/// This layer is **inference-only**. Calling `backward()` or
/// `update_parameters()` will panic with a clear error message.
///
/// # Construction
///
/// Use [`quantize_dense_layer`] to create a `QuantizedDenseLayer` from a
/// trained [`DenseLayer`]:
///
/// ```
/// use rust_neural_networks::layers::{DenseLayer, Layer};
/// use rust_neural_networks::utils::rng::SimpleRng;
/// use rust_neural_networks::compression::quantization::quantize_dense_layer;
///
/// let mut rng = SimpleRng::new(42);
/// let dense = DenseLayer::new(4, 2, &mut rng);
/// let quantized = quantize_dense_layer(&dense);
///
/// assert_eq!(quantized.input_size(), 4);
/// assert_eq!(quantized.output_size(), 2);
/// // INT8 weights: 1 byte per weight vs 4 bytes for f32
/// assert_eq!(quantized.parameter_memory_bytes(), quantized.parameter_count() * 1);
/// ```
pub struct QuantizedDenseLayer {
    input_size: usize,
    output_size: usize,
    /// INT8-quantized weight matrix in row-major format (input_size × output_size).
    weights: Vec<i8>,
    /// Quantization parameters used to encode/decode the weight matrix.
    weight_params: QuantizationParams,
    /// Bias vector stored in f32 precision (one value per output neuron).
    biases: Vec<f32>,
}

/// Construct a [`QuantizedDenseLayer`] by quantizing a trained [`DenseLayer`].
///
/// Copies the bias vector as-is and applies symmetric INT8 quantization to the
/// weight matrix using [`quantize_weights`].
///
/// # Arguments
///
/// * `dense` - A reference to the trained dense layer to be quantized
///
/// # Returns
///
/// A new `QuantizedDenseLayer` ready for inference.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::layers::{DenseLayer, Layer};
/// use rust_neural_networks::utils::rng::SimpleRng;
/// use rust_neural_networks::compression::quantization::quantize_dense_layer;
///
/// let mut rng = SimpleRng::new(0);
/// let dense = DenseLayer::new(3, 2, &mut rng);
/// let q_layer = quantize_dense_layer(&dense);
///
/// assert_eq!(q_layer.input_size(), dense.input_size());
/// assert_eq!(q_layer.output_size(), dense.output_size());
/// assert_eq!(q_layer.parameter_count(), dense.parameter_count());
/// ```
pub fn quantize_dense_layer(dense: &DenseLayer) -> QuantizedDenseLayer {
    let (quantized_weights, weight_params) = quantize_weights(dense.weights());

    QuantizedDenseLayer {
        input_size: dense.input_size(),
        output_size: dense.output_size(),
        weights: quantized_weights,
        weight_params,
        biases: dense.biases().to_vec(),
    }
}

impl Layer for QuantizedDenseLayer {
    /// Performs the forward pass by dequantizing weights to f32 then computing
    /// `output = input × weights + biases`.
    ///
    /// Weights are dequantized on-the-fly for each forward call. The computation
    /// uses a naive matrix multiply loop, which is intentional for an
    /// inference-only quantized layer (clarity over raw performance).
    ///
    /// # Arguments
    ///
    /// * `input`      - Input data in row-major layout (batch_size × input_size)
    /// * `output`     - Output buffer (batch_size × output_size)
    /// * `batch_size` - Number of samples in the batch
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "QuantizedDenseLayer forward: input shape mismatch. \
             Expected {} elements (batch_size={} × input_size={}), got {}.",
            batch_size * self.input_size,
            batch_size,
            self.input_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "QuantizedDenseLayer forward: output shape mismatch. \
             Expected {} elements (batch_size={} × output_size={}), got {}.",
            batch_size * self.output_size,
            batch_size,
            self.output_size,
            output.len()
        );

        // Dequantize INT8 weights to f32 for the matrix multiply.
        let dequant_weights = dequantize_weights(&self.weights, &self.weight_params);

        // Naive matrix multiply: output = input × dequant_weights + biases
        // Weights are stored row-major as (input_size × output_size).
        for b in 0..batch_size {
            for j in 0..self.output_size {
                let mut acc = self.biases[j];
                for i in 0..self.input_size {
                    acc +=
                        input[b * self.input_size + i] * dequant_weights[i * self.output_size + j];
                }
                output[b * self.output_size + j] = acc;
            }
        }
    }

    /// Not supported — this is an inference-only layer.
    ///
    /// # Panics
    ///
    /// Always panics with a message indicating that `QuantizedDenseLayer` is
    /// inference-only and does not support backpropagation.
    fn backward(
        &self,
        _input: &[f32],
        _grad_output: &[f32],
        _grad_input: &mut [f32],
        _batch_size: usize,
    ) {
        panic!(
            "QuantizedDenseLayer is inference-only and does not support backward(). \
             To train, use a DenseLayer and quantize it afterwards with quantize_dense_layer()."
        );
    }

    /// Not supported — this is an inference-only layer.
    ///
    /// # Panics
    ///
    /// Always panics with a message indicating that `QuantizedDenseLayer` is
    /// inference-only and does not support parameter updates.
    fn update_parameters(&mut self, _learning_rate: f32) {
        panic!(
            "QuantizedDenseLayer is inference-only and does not support update_parameters(). \
             To train, use a DenseLayer and quantize it afterwards with quantize_dense_layer()."
        );
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the total parameter count: `input_size × output_size + output_size`.
    ///
    /// This matches `DenseLayer::parameter_count()` so compression ratios can be
    /// computed by comparing `parameter_memory_bytes()` across layer types.
    fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Memory footprint of the quantized parameters in bytes.
    ///
    /// Returns `parameter_count()` because INT8 weights occupy 1 byte each.
    /// Compare with `DenseLayer`'s `parameter_count() * 4` (f32) to measure
    /// the 4× memory reduction provided by INT8 quantization.
    fn parameter_memory_bytes(&self) -> usize {
        self.parameter_count()
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
    fn test_quantization_params_symmetric_zero_point() {
        let weights = vec![0.3_f32, -0.7, 0.5, -0.1];
        let (_, params) = quantize_weights(&weights);
        assert_eq!(
            params.zero_point, 0,
            "symmetric quantization must have zero_point=0"
        );
    }

    #[test]
    fn test_quantization_scale_calculation() {
        let weights = vec![1.0_f32, -1.0, 0.5, -0.5];
        let (_, params) = quantize_weights(&weights);
        let expected_scale = 1.0_f32 / 127.0;
        assert!(
            (params.scale - expected_scale).abs() < 1e-6,
            "scale should be max(|w|)/127, got {}",
            params.scale
        );
    }

    #[test]
    fn test_quantize_extreme_values_map_to_127() {
        let weights = vec![1.0_f32, -1.0];
        let (quantized, _) = quantize_weights(&weights);
        assert_eq!(quantized[0], 127, "max weight should map to 127");
        assert_eq!(quantized[1], -127, "min weight should map to -127");
    }

    #[test]
    fn test_quantize_zero_maps_to_zero() {
        let weights = vec![0.0_f32, 1.0, -1.0];
        let (quantized, _) = quantize_weights(&weights);
        assert_eq!(quantized[0], 0, "zero weight should map to 0");
    }

    #[test]
    fn test_quantize_empty_slice() {
        let weights: Vec<f32> = vec![];
        let (quantized, params) = quantize_weights(&weights);
        assert!(quantized.is_empty());
        assert_eq!(params.scale, 1.0, "empty input should default scale to 1.0");
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let weights = vec![0.0_f32, 0.0, 0.0];
        let (quantized, params) = quantize_weights(&weights);
        assert_eq!(
            params.scale, 1.0,
            "all-zero input should default scale to 1.0"
        );
        for &q in &quantized {
            assert_eq!(q, 0);
        }
    }

    #[test]
    fn test_dequantize_roundtrip() {
        let weights = vec![0.5_f32, -0.25, 1.0, -1.0, 0.0, 0.75];
        let (quantized, params) = quantize_weights(&weights);
        let dequantized = dequantize_weights(&quantized, &params);

        assert_eq!(dequantized.len(), weights.len());
        // Dequantized values should be close to original (within ~1 LSB)
        for (orig, dq) in weights.iter().zip(dequantized.iter()) {
            assert!(
                (orig - dq).abs() < params.scale + 1e-6,
                "dequantized value {} should be close to original {}",
                dq,
                orig
            );
        }
    }

    #[test]
    fn test_dequantize_scale_applied() {
        let params = QuantizationParams {
            scale: 0.1,
            zero_point: 0,
        };
        let quantized: Vec<i8> = vec![10, -10, 0, 127, -127];
        let dequantized = dequantize_weights(&quantized, &params);

        assert!((dequantized[0] - 1.0).abs() < 1e-6);
        assert!((dequantized[1] + 1.0).abs() < 1e-6);
        assert!((dequantized[2]).abs() < 1e-10);
        assert!((dequantized[3] - 12.7).abs() < 1e-5);
        assert!((dequantized[4] + 12.7).abs() < 1e-5);
    }

    #[test]
    fn test_quantization_error_zero_for_exact() {
        // If scale perfectly represents the values, error should be near zero
        let params = QuantizationParams {
            scale: 1.0 / 127.0,
            zero_point: 0,
        };
        let quantized: Vec<i8> = vec![127, -127, 0];
        let original: Vec<f32> = quantized.iter().map(|&q| q as f32 * params.scale).collect();
        let mse = quantization_error(&original, &params, &quantized);
        assert!(
            mse < 1e-12,
            "MSE should be near zero for exact representation"
        );
    }

    #[test]
    fn test_quantization_error_is_small_for_typical_weights() {
        let weights = vec![1.0_f32, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0];
        let (quantized, params) = quantize_weights(&weights);
        let mse = quantization_error(&weights, &params, &quantized);
        assert!(
            mse < 1e-4,
            "MSE should be small for typical weights, got {}",
            mse
        );
    }

    #[test]
    fn test_quantization_error_empty() {
        let params = QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
        let mse = quantization_error(&[], &params, &[]);
        assert_eq!(mse, 0.0);
    }

    #[test]
    #[should_panic(expected = "original and quantized slices must have the same length")]
    fn test_quantization_error_mismatched_lengths_panics() {
        let params = QuantizationParams {
            scale: 1.0,
            zero_point: 0,
        };
        quantization_error(&[1.0], &params, &[]);
    }

    #[test]
    fn test_quantization_params_debug() {
        let params = QuantizationParams {
            scale: 0.5,
            zero_point: 0,
        };
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("scale"));
        assert!(debug_str.contains("zero_point"));
    }

    #[test]
    fn test_quantization_params_clone() {
        let params = QuantizationParams {
            scale: 0.5,
            zero_point: 0,
        };
        let cloned = params.clone();
        assert_eq!(params, cloned);
    }
}
