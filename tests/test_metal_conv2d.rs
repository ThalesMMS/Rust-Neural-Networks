//! Tests for Metal conv2d kernels.
//!
//! Validates GPU-accelerated conv2d forward and backward passes against
//! CPU reference implementations.

#![cfg(feature = "gpu-metal")]

use rust_neural_networks::gpu::backend::GpuBackend;
use rust_neural_networks::gpu::metal_backend::MetalBackend;

/// Macro to skip tests when no Metal GPU is available.
macro_rules! require_metal {
    () => {
        match MetalBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping test: no Metal GPU available");
                return;
            }
        }
    };
}

/// Asserts two slices are element-wise approximately equal within a tolerance.
///
/// Panics if the slices have different lengths or if any corresponding pair of elements
/// differs by an absolute amount greater than or equal to `tol`. Comparison uses absolute
/// difference: |a[i] - b[i]| < tol.
///
/// # Examples
///
/// ```
/// let a = [1.0f32, 2.0, 3.0];
/// let b = [1.0f32, 2.001, 2.999];
/// // Passes because each element differs by less than 0.01
/// assert_approx_eq(&a, &b, 0.01);
/// ```
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "Length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "Element {} differs: {} vs {} (diff={})",
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

/// Compute a CPU reference conv2d forward pass for NCHW-formatted tensors.
///
/// Convolves `input` with `filters`, adds `bias`, and returns the resulting output tensor in NCHW order.
/// Output spatial dimensions are (input_h + 2*padding - kernel_h) / stride + 1 and (input_w + 2*padding - kernel_w) / stride + 1.
///
/// # Returns
///
/// `Vec<f32>` containing the output tensor in NCHW layout with shape (batch_size, out_channels, out_h, out_w).
///
/// # Examples
///
/// ```
/// let batch = 1;
/// let ic = 1;
/// let oc = 1;
/// let ih = 4;
/// let iw = 4;
/// let kh = 3;
/// let kw = 3;
/// let stride = 1;
/// let padding = 0;
/// let input: Vec<f32> = (0..(batch*ic*ih*iw)).map(|x| x as f32).collect();
/// let filters: Vec<f32> = vec![1.0f32; oc*ic*kh*kw];
/// let bias: Vec<f32> = vec![0.0f32; oc];
/// let out = cpu_conv2d_forward(&input, &filters, &bias, batch, ic, oc, ih, iw, kh, kw, stride, padding);
/// let out_h = (ih + 2*padding - kh) / stride + 1;
/// let out_w = (iw + 2*padding - kw) / stride + 1;
/// assert_eq!(out.len(), batch * oc * out_h * out_w);
/// ```
fn cpu_conv2d_forward(
    input: &[f32],
    filters: &[f32],
    bias: &[f32],
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = bias[oc];
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = (oy * stride + kh) as isize - padding as isize;
                                let iw = (ox * stride + kw) as isize - padding as isize;
                                if ih >= 0
                                    && ih < input_h as isize
                                    && iw >= 0
                                    && iw < input_w as isize
                                {
                                    let in_idx = b * (in_channels * input_h * input_w)
                                        + ic * (input_h * input_w)
                                        + ih as usize * input_w
                                        + iw as usize;
                                    let f_idx = oc * (in_channels * kernel_h * kernel_w)
                                        + ic * (kernel_h * kernel_w)
                                        + kh * kernel_w
                                        + kw;
                                    sum += input[in_idx] * filters[f_idx];
                                }
                            }
                        }
                    }
                    let out_idx =
                        b * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oy * out_w + ox;
                    output[out_idx] = sum;
                }
            }
        }
    }
    output
}

// ── Forward tests ──────────────────────────────────────────────────────

/// Verifies that the Metal backend's conv2d_forward produces the expected output for a basic 1x1→1x1 convolution.
///
/// Uses a 4x4 sequential input, a 3x3 kernel of all ones, zero bias, stride 1 and no padding, and compares the backend result
/// against the CPU reference implementation `cpu_conv2d_forward` with a tolerance of 1e-5.
///
/// # Examples
///
/// ```
/// // Requires a Metal-capable device; returns early if unavailable.
/// let backend = require_metal!();
/// let batch_size = 1;
/// let in_channels = 1;
/// let out_channels = 1;
/// let input_h = 4;
/// let input_w = 4;
/// let kernel_h = 3;
/// let kernel_w = 3;
/// let stride = 1;
/// let padding = 0;
///
/// let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
/// let filters = vec![1.0f32; 9];
/// let bias = vec![0.0f32];
/// let mut output = vec![0.0f32; batch_size * out_channels * 2 * 2];
///
/// backend
///     .conv2d_forward(
///         &input,
///         &filters,
///         &bias,
///         &mut output,
///         batch_size,
///         in_channels,
///         out_channels,
///         input_h,
///         input_w,
///         kernel_h,
///         kernel_w,
///         stride,
///         padding,
///     )
///     .unwrap();
/// ```
#[test]
fn test_metal_conv2d_forward_basic() {
    let backend = require_metal!();

    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_h = 4;
    let input_w = 4;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 0;
    let out_h = 2;
    let out_w = 2;

    // Simple input: sequential values
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // Simple filter: all ones
    let filters = vec![1.0f32; 9];
    let bias = vec![0.0f32];
    let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    backend
        .conv2d_forward(
            &input,
            &filters,
            &bias,
            &mut output,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    let expected = cpu_conv2d_forward(
        &input,
        &filters,
        &bias,
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
    );
    assert_approx_eq(&output, &expected, 1e-5);
}

/// Validates the Metal backend's conv2d forward pass with padding against a CPU reference implementation.
///
/// Constructs a 1×1×4×4 input, 2 output channels, 3×3 kernels, stride 1 and padding 1, runs the backend's
/// conv2d_forward, and asserts the produced output matches cpu_conv2d_forward within a tolerance of 1e-4.
///
/// # Examples
///
/// ```
/// // The test builds input, filters, and bias, calls conv2d_forward on the Metal backend,
/// // then compares the result to cpu_conv2d_forward using assert_approx_eq.
/// ```
#[test]
fn test_metal_conv2d_forward_with_padding() {
    let backend = require_metal!();

    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 2;
    let input_h = 4;
    let input_w = 4;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 1;

    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let filters: Vec<f32> = (0..18).map(|i| (i as f32) * 0.01 - 0.09).collect();
    let bias = vec![0.1, -0.1];

    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    backend
        .conv2d_forward(
            &input,
            &filters,
            &bias,
            &mut output,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    let expected = cpu_conv2d_forward(
        &input,
        &filters,
        &bias,
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
    );
    assert_approx_eq(&output, &expected, 1e-4);
}

/// Verifies the Metal backend's conv2d forward pass against a CPU reference for a multi-channel, batched input.
///
/// This integration test constructs a batch of inputs with multiple input and output channels, runs the Metal
/// conv2d forward implementation with padding and stride, and asserts the GPU output matches the CPU reference
/// within a small tolerance.
///
/// # Examples
///
/// ```
/// // Demonstrates typical usage performed by the test:
/// let backend = require_metal!();
/// let output = { /* prepare buffers and call backend.conv2d_forward(...) */ };
/// // Compare `output` to `cpu_conv2d_forward(...)`
/// ```
#[test]
fn test_metal_conv2d_forward_multi_channel() {
    let backend = require_metal!();

    let batch_size = 2;
    let in_channels = 3;
    let out_channels = 2;
    let input_h = 5;
    let input_w = 5;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 1;

    let input_len = batch_size * in_channels * input_h * input_w;
    let filter_len = out_channels * in_channels * kernel_h * kernel_w;

    let input: Vec<f32> = (0..input_len)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let filters: Vec<f32> = (0..filter_len)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();
    let bias = vec![0.1, -0.2];

    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    backend
        .conv2d_forward(
            &input,
            &filters,
            &bias,
            &mut output,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    let expected = cpu_conv2d_forward(
        &input,
        &filters,
        &bias,
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
    );
    assert_approx_eq(&output, &expected, 1e-4);
}

/// Verifies the Metal backend's conv2d forward pass with a stride of 2 matches the CPU reference.
///
/// Uses a single-batch, single-channel 6×6 input with a 3×3 kernel, stride 2, no padding,
/// filters filled with ones and a bias of 0.5. The GPU output is compared to `cpu_conv2d_forward`
/// with a tolerance of 1e-4.
///
/// # Examples
///
/// ```
/// let backend = require_metal!();
/// let batch_size = 1;
/// let in_channels = 1;
/// let out_channels = 1;
/// let input_h = 6;
/// let input_w = 6;
/// let kernel_h = 3;
/// let kernel_w = 3;
/// let stride = 2;
/// let padding = 0;
/// let input: Vec<f32> = (0..36).map(|i| i as f32 * 0.1).collect();
/// let filters = vec![1.0f32; 9];
/// let bias = vec![0.5];
/// let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
/// let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
/// let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];
///
/// backend
///     .conv2d_forward(
///         &input,
///         &filters,
///         &bias,
///         &mut output,
///         batch_size,
///         in_channels,
///         out_channels,
///         input_h,
///         input_w,
///         kernel_h,
///         kernel_w,
///         stride,
///         padding,
///     )
///     .unwrap();
/// let expected = cpu_conv2d_forward(
///     &input,
///     &filters,
///     &bias,
///     batch_size,
///     in_channels,
///     out_channels,
///     input_h,
///     input_w,
///     kernel_h,
///     kernel_w,
///     stride,
///     padding,
/// );
/// assert_approx_eq(&output, &expected, 1e-4);
/// ```
#[test]
fn test_metal_conv2d_forward_stride2() {
    let backend = require_metal!();

    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_h = 6;
    let input_w = 6;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 2;
    let padding = 0;

    let input: Vec<f32> = (0..36).map(|i| i as f32 * 0.1).collect();
    let filters = vec![1.0f32; 9];
    let bias = vec![0.5];

    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let mut output = vec![0.0f32; batch_size * out_channels * out_h * out_w];

    backend
        .conv2d_forward(
            &input,
            &filters,
            &bias,
            &mut output,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    let expected = cpu_conv2d_forward(
        &input,
        &filters,
        &bias,
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
    );
    assert_approx_eq(&output, &expected, 1e-4);
}

// ── Backward tests ─────────────────────────────────────────────────────

/// Checks that conv2d backward computes bias gradients equal to the sum of `grad_output`
/// over batch, height, and width for each output channel.
///
/// The test sets up a small convolution scenario, runs `conv2d_backward`, and compares
/// each element of `grad_bias` against the per-channel sum of `grad_output` within a
/// tolerance of 1e-3.
///
/// # Examples
///
/// ```rust
/// // Conceptual example: bias gradient equals sum of grad_output per output channel
/// let batch_size = 2;
/// let out_channels = 2;
/// let out_h = 4;
/// let out_w = 4;
/// let output_size = batch_size * out_channels * out_h * out_w;
/// let grad_output: Vec<f32> = (0..output_size).map(|i| (i % 5) as f32 * 0.1).collect();
///
/// // For each output channel, expected bias gradient is the sum of its slice in grad_output.
/// let mut expected = vec![0.0f32; out_channels];
/// for b in 0..batch_size {
///     for oc in 0..out_channels {
///         for oy in 0..out_h {
///             for ox in 0..out_w {
///                 let idx = b * (out_channels * out_h * out_w)
///                     + oc * (out_h * out_w)
///                     + oy * out_w
///                     + ox;
///                 expected[oc] += grad_output[idx];
///             }
///         }
///     }
/// }
/// ```
#[test]
fn test_metal_conv2d_backward_bias() {
    let backend = require_metal!();

    let batch_size = 2;
    let in_channels = 1;
    let out_channels = 2;
    let input_h = 4;
    let input_w = 4;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 1;
    let out_h = 4;
    let out_w = 4;

    let input_size = batch_size * in_channels * input_h * input_w;
    let filter_size = out_channels * in_channels * kernel_h * kernel_w;
    let output_size = batch_size * out_channels * out_h * out_w;

    let input = vec![1.0f32; input_size];
    let filters = vec![0.1f32; filter_size];
    let grad_output: Vec<f32> = (0..output_size).map(|i| (i % 5) as f32 * 0.1).collect();
    let mut grad_input = vec![0.0f32; input_size];
    let mut grad_filters = vec![0.0f32; filter_size];
    let mut grad_bias = vec![0.0f32; out_channels];

    backend
        .conv2d_backward(
            &input,
            &filters,
            &grad_output,
            &mut grad_input,
            &mut grad_filters,
            &mut grad_bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    // Verify bias gradient: sum of grad_output over batch and spatial dims per channel
    for oc in 0..out_channels {
        let mut expected_sum = 0.0f32;
        for b in 0..batch_size {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let idx =
                        b * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oy * out_w + ox;
                    expected_sum += grad_output[idx];
                }
            }
        }
        assert!(
            (grad_bias[oc] - expected_sum).abs() < 1e-3,
            "Bias gradient {} mismatch: got {} expected {}",
            oc,
            grad_bias[oc],
            expected_sum
        );
    }
}

/// Verifies that the Metal backend computes correct gradients for convolution filters in a simple single-channel scenario.
///
/// Runs conv2d_backward with a 1×1 batch, single input and output channel, a 4×4 input and a 3×3 kernel, stride 1 and no padding,
/// then checks that:
/// - all computed filter gradients are finite and at least one is non-zero, and
/// - each filter gradient matches the CPU reference value computed as the sum over output spatial positions of grad_output * corresponding input element (within a tolerance of 1e-3).
///
/// # Examples
///
/// ```
/// // Typical usage inside tests: prepare inputs, call conv2d_backward, then assert filter gradients.
/// let batch_size = 1;
/// let in_channels = 1;
/// let out_channels = 1;
/// let input_h = 4;
/// let input_w = 4;
/// let kernel_h = 3;
/// let kernel_w = 3;
/// let stride = 1;
/// let padding = 0;
///
/// let input_size = batch_size * in_channels * input_h * input_w;
/// let filter_size = out_channels * in_channels * kernel_h * kernel_w;
/// let out_h = 2;
/// let out_w = 2;
///
/// let input: Vec<f32> = (0..input_size).map(|i| i as f32).collect();
/// let filters = vec![0.1f32; filter_size];
/// let grad_output = vec![1.0f32; batch_size * out_channels * out_h * out_w];
/// let mut grad_input = vec![0.0f32; input_size];
/// let mut grad_filters = vec![0.0f32; filter_size];
/// let mut grad_bias = vec![0.0f32; out_channels];
///
/// let backend = require_metal!();
/// backend
///     .conv2d_backward(
///         &input,
///         &filters,
///         &grad_output,
///         &mut grad_input,
///         &mut grad_filters,
///         &mut grad_bias,
///         batch_size,
///         in_channels,
///         out_channels,
///         input_h,
///         input_w,
///         kernel_h,
///         kernel_w,
///         stride,
///         padding,
///     )
///     .unwrap();
///
/// // Assertions follow: finiteness, non-zero, and comparison to CPU reference.
/// ```
#[test]
fn test_metal_conv2d_backward_filters() {
    let backend = require_metal!();

    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_h = 4;
    let input_w = 4;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 0;
    let out_h = 2;
    let out_w = 2;

    let input_size = batch_size * in_channels * input_h * input_w;
    let filter_size = out_channels * in_channels * kernel_h * kernel_w;
    let output_size = batch_size * out_channels * out_h * out_w;

    let input: Vec<f32> = (0..input_size).map(|i| i as f32).collect();
    let filters = vec![0.1f32; filter_size];
    let grad_output = vec![1.0f32; output_size];
    let mut grad_input = vec![0.0f32; input_size];
    let mut grad_filters = vec![0.0f32; filter_size];
    let mut grad_bias = vec![0.0f32; out_channels];

    backend
        .conv2d_backward(
            &input,
            &filters,
            &grad_output,
            &mut grad_input,
            &mut grad_filters,
            &mut grad_bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    // Filter gradients should be non-zero and finite
    assert!(grad_filters.iter().all(|&x| x.is_finite()));
    assert!(grad_filters.iter().any(|&x| x.abs() > 1e-10));

    // CPU reference for filter gradients
    // grad_filters[kh, kw] = sum over oy, ox of grad_output[oy, ox] * input[oy+kh, ox+kw]
    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            let mut expected = 0.0f32;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let ih = oy * stride + kh;
                    let iw = ox * stride + kw;
                    expected += grad_output[oy * out_w + ox] * input[ih * input_w + iw];
                }
            }
            let idx = kh * kernel_w + kw;
            assert!(
                (grad_filters[idx] - expected).abs() < 1e-3,
                "Filter grad [{}, {}] mismatch: got {} expected {}",
                kh,
                kw,
                grad_filters[idx],
                expected
            );
        }
    }
}

#[test]
fn test_metal_conv2d_backward_input() {
    let backend = require_metal!();

    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_h = 4;
    let input_w = 4;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 0;

    let input_size = batch_size * in_channels * input_h * input_w;
    let filter_size = out_channels * in_channels * kernel_h * kernel_w;
    let out_h = 2;
    let out_w = 2;
    let output_size = batch_size * out_channels * out_h * out_w;

    let input = vec![1.0f32; input_size];
    let filters: Vec<f32> = (0..filter_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let grad_output = vec![1.0f32; output_size];
    let mut grad_input = vec![0.0f32; input_size];
    let mut grad_filters = vec![0.0f32; filter_size];
    let mut grad_bias = vec![0.0f32; out_channels];

    backend
        .conv2d_backward(
            &input,
            &filters,
            &grad_output,
            &mut grad_input,
            &mut grad_filters,
            &mut grad_bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    // Input gradients should be finite and some non-zero
    assert!(grad_input.iter().all(|&x| x.is_finite()));
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_metal_conv2d_backward_full() {
    // Test with multi-channel, batch, padding - verify all gradients are finite
    let backend = require_metal!();

    let batch_size = 2;
    let in_channels = 3;
    let out_channels = 2;
    let input_h = 5;
    let input_w = 5;
    let kernel_h = 3;
    let kernel_w = 3;
    let stride = 1;
    let padding = 1;

    let input_size = batch_size * in_channels * input_h * input_w;
    let filter_size = out_channels * in_channels * kernel_h * kernel_w;
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let output_size = batch_size * out_channels * out_h * out_w;

    let input: Vec<f32> = (0..input_size)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let filters: Vec<f32> = (0..filter_size)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
        .collect();
    let grad_output: Vec<f32> = (0..output_size)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect();
    let mut grad_input = vec![0.0f32; input_size];
    let mut grad_filters = vec![0.0f32; filter_size];
    let mut grad_bias = vec![0.0f32; out_channels];

    backend
        .conv2d_backward(
            &input,
            &filters,
            &grad_output,
            &mut grad_input,
            &mut grad_filters,
            &mut grad_bias,
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        )
        .unwrap();

    // All gradients should be finite
    assert!(
        grad_input.iter().all(|&x| x.is_finite()),
        "grad_input has non-finite values"
    );
    assert!(
        grad_filters.iter().all(|&x| x.is_finite()),
        "grad_filters has non-finite values"
    );
    assert!(
        grad_bias.iter().all(|&x| x.is_finite()),
        "grad_bias has non-finite values"
    );

    // All gradients should have some non-zero values
    assert!(
        grad_input.iter().any(|&x| x.abs() > 1e-10),
        "grad_input is all zeros"
    );
    assert!(
        grad_filters.iter().any(|&x| x.abs() > 1e-10),
        "grad_filters is all zeros"
    );
    assert!(
        grad_bias.iter().any(|&x| x.abs() > 1e-10),
        "grad_bias is all zeros"
    );
}

// ── Dimension error tests ──────────────────────────────────────────────

/// Verifies that conv2d_forward returns an error when the provided input buffer is too small for the specified dimensions.
///
/// # Examples
///
/// ```
/// // Prepare deliberately undersized input for a 1x1 batch with 4x4 spatial dimensions.
/// let input = vec![1.0f32; 4]; // too small
/// let filters = vec![1.0f32; 9];
/// let bias = vec![0.0f32];
/// let mut output = vec![0.0f32; 4];
///
/// // backend is assumed to be a MetalBackend obtained via the test's helper (e.g., require_metal!()).
/// let result = backend.conv2d_forward(
///     &input,
///     &filters,
///     &bias,
///     &mut output,
///     1, // batch
///     1, // in_channels
///     1, // out_channels
///     4, // in_h
///     4, // in_w
///     3, // kernel_h
///     3, // kernel_w
///     1, // stride
///     0, // padding
/// );
/// assert!(result.is_err());
/// ```
#[test]
fn test_metal_conv2d_forward_dimension_error() {
    let backend = require_metal!();

    let input = vec![1.0f32; 4]; // Too small
    let filters = vec![1.0f32; 9];
    let bias = vec![0.0f32];
    let mut output = vec![0.0f32; 4];

    let result = backend.conv2d_forward(
        &input,
        &filters,
        &bias,
        &mut output,
        1,
        1,
        1,
        4,
        4,
        3,
        3,
        1,
        0,
    );
    assert!(result.is_err());
}