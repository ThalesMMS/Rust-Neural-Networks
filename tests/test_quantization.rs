//! Tests for model quantization functionality
//!
//! These tests verify the INT8 quantization implementation in the compression module,
//! including dequantization accuracy, memory size reduction, forward pass fidelity,
//! symmetric quantization parameter properties, and edge cases.

use approx::assert_relative_eq;
use rust_neural_networks::compression::quantization::{
    dequantize_weights, quantization_error, quantize_dense_layer, quantize_weights,
};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

/// Verify that dequantized values are close to the original floating-point weights.
///
/// Uses `approx::assert_relative_eq` with `epsilon = 0.01` (absolute tolerance)
/// to confirm that each dequantized value is within rounding error of its original.
#[test]
fn test_quantize_simple_weights() {
    let weights = vec![1.0_f32, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0];
    let (quantized, params) = quantize_weights(&weights);
    let dequantized = dequantize_weights(&quantized, &params);

    assert_eq!(dequantized.len(), weights.len());

    for (orig, dq) in weights.iter().zip(dequantized.iter()) {
        // INT8 quantization introduces at most scale/2 error per weight.
        // With scale = max(|w|)/127 ≈ 0.00787, all errors should be well within 0.01.
        assert_relative_eq!(*orig, *dq, epsilon = 0.01);
    }
}

/// Verify that a QuantizedDenseLayer uses exactly 4× less memory than DenseLayer.
///
/// INT8 weights occupy 1 byte each, while f32 weights occupy 4 bytes each.
/// Therefore `QuantizedDenseLayer::parameter_memory_bytes()` must equal
/// `DenseLayer::parameter_memory_bytes() / 4`.
#[test]
fn test_quantization_size_reduction() {
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(16, 8, &mut rng);
    let quantized = quantize_dense_layer(&dense);

    assert_eq!(
        quantized.parameter_memory_bytes(),
        dense.parameter_memory_bytes() / 4,
        "Quantized layer should use exactly 4× less memory than the f32 DenseLayer"
    );
}

/// Verify that quantized forward pass outputs are within 5% of the original DenseLayer outputs.
///
/// Creates a DenseLayer, quantizes it, runs the same input through both layers,
/// and checks that each output element differs by no more than 5% of the largest
/// output magnitude.
#[test]
fn test_quantized_forward_close_to_original() {
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(8, 4, &mut rng);
    let quantized = quantize_dense_layer(&dense);

    let input = vec![0.5_f32, -0.3, 0.8, -0.1, 0.2, -0.6, 0.4, -0.7];
    let mut orig_output = vec![0.0_f32; 4];
    let mut quant_output = vec![0.0_f32; 4];

    dense.forward(&input, &mut orig_output, 1);
    quantized.forward(&input, &mut quant_output, 1);

    // Compute a 5% tolerance based on the maximum absolute output magnitude.
    // Use a minimum floor of 1e-3 to guard against near-zero outputs.
    let max_magnitude = orig_output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let tolerance = (0.05 * max_magnitude).max(1e-3);

    for (i, (orig, quant)) in orig_output.iter().zip(quant_output.iter()).enumerate() {
        assert!(
            (orig - quant).abs() <= tolerance,
            "Output[{}] differs by more than 5%: orig={:.6}, quant={:.6}, \
             diff={:.6}, tolerance={:.6}",
            i,
            orig,
            quant,
            (orig - quant).abs(),
            tolerance
        );
    }
}

/// Verify that symmetric INT8 quantization always produces zero_point == 0 and scale > 0.
///
/// Symmetric quantization centres the integer range around zero, so the zero-point
/// offset must be zero. The scale factor converts between float and integer domains
/// and must always be strictly positive.
#[test]
fn test_quantization_params_symmetric() {
    let weights = vec![0.3_f32, -0.7, 0.5, -0.1, 0.9, -0.4, 0.2, -0.8];
    let (_, params) = quantize_weights(&weights);

    assert_eq!(
        params.zero_point, 0,
        "Symmetric quantization must have zero_point == 0, got {}",
        params.zero_point
    );
    assert!(
        params.scale > 0.0,
        "Quantization scale must be strictly positive, got {}",
        params.scale
    );
}

/// Verify that the quantization MSE is small (< 0.01) for Xavier-initialized weights.
///
/// Xavier initialization produces bounded weights, so the rounding error introduced
/// by INT8 quantization should be well below 0.01 mean squared error.
#[test]
fn test_quantization_error_small() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(16, 8, &mut rng);
    let weights = layer.weights();

    let (quantized, params) = quantize_weights(weights);
    let mse = quantization_error(weights, &params, &quantized);

    assert!(
        mse < 0.01,
        "MSE for Xavier-initialized weights should be < 0.01, got {:.8}",
        mse
    );
}

/// Edge case: all-zero weights should quantize cleanly.
///
/// When all weights are zero the maximum absolute value is 0, so the scale
/// defaults to 1.0 to avoid division by zero. All quantized values should be 0
/// and dequantization should recover 0.0 exactly.
#[test]
fn test_quantize_all_zeros() {
    let weights = vec![0.0_f32; 10];
    let (quantized, params) = quantize_weights(&weights);

    assert_eq!(
        params.scale, 1.0,
        "All-zero weights should default scale to 1.0 to avoid division by zero, got {}",
        params.scale
    );
    assert_eq!(
        params.zero_point, 0,
        "zero_point should be 0 for all-zero weights, got {}",
        params.zero_point
    );

    for (i, &q) in quantized.iter().enumerate() {
        assert_eq!(q, 0_i8, "quantized[{}] should be 0 for all-zero input", i);
    }

    let dequantized = dequantize_weights(&quantized, &params);
    for (i, &dq) in dequantized.iter().enumerate() {
        assert_eq!(
            dq, 0.0_f32,
            "dequantized[{}] should be 0.0 for all-zero input, got {}",
            i, dq
        );
    }
}
