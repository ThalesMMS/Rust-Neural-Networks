use super::*;
use cblas::{sgemm as cpu_sgemm, Layout, Transpose};

/// Multiply two matrices on the CPU using the BLAS `sgemm` reference and return the result in row-major order.
///
/// The inputs `a` and `b` are interpreted according to `trans_a` and `trans_b`:
/// - `a` has shape (m, k) if `trans_a == Transpose::None`, otherwise (k, m).
/// - `b` has shape (k, n) if `trans_b == Transpose::None`, otherwise (n, k).
///
/// # Returns
///
/// A `Vec<f32>` containing the result matrix in row-major order with shape (m, n).
///
/// # Examples
///
/// ```
/// // C = A * B where A is 2x3 and B is 3x2
/// let a = vec![1.0f32, 2.0, 3.0,
///              4.0, 5.0, 6.0]; // 2 x 3
/// let b = vec![7.0f32, 8.0,
///              9.0, 10.0,
///              11.0, 12.0]; // 3 x 2
/// let c = cpu_matmul(2, 2, 3, &a, &b, Transpose::None, Transpose::None);
/// assert_eq!(c.len(), 4);
/// // first row = [58, 64]
/// assert!((c[0] - 58.0).abs() < 1e-6 && (c[1] - 64.0).abs() < 1e-6);
/// ```
fn cpu_matmul(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_a: Transpose,
    trans_b: Transpose,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    let lda = if trans_a == Transpose::None { k } else { m };
    let ldb = if trans_b == Transpose::None { n } else { k };
    unsafe {
        cpu_sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            0.0,
            &mut c,
            n as i32,
        );
    }
    c
}

/// Asserts that two slices contain elementwise equal f32 values within a given absolute tolerance.
///
/// Panics if the slices have different lengths or if any corresponding elements differ by greater than or equal to `tol`.
///
/// # Examples
///
/// ```
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [1.001_f32, 1.999, 3.0];
/// assert_approx_eq(&a, &b, 0.01);
/// ```
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "mismatch at index {}: cpu={} gpu={} (diff={})",
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

#[test]
fn test_cuda_sgemm_basic() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    // A (2x3) * B (3x4) = C (2x4)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::None, Transpose::None);
    let mut c = vec![0.0f32; 8];
    backend.sgemm(2, 4, 3, &a, &b, &mut c).unwrap();
    assert_approx_eq(&c, &expected, 1e-4);
}

#[test]
fn test_cuda_sgemm_identity() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0f32; 9];
    backend.sgemm(3, 3, 3, &a, &identity, &mut c).unwrap();
    assert_approx_eq(&c, &a, 1e-5);
}

#[test]
fn test_cuda_sgemm_at() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // A stored as (k=3 x m=2), transposed to (2x3), times B (3x4) = C (2x4)
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::Ordinary, Transpose::None);
    let mut c = vec![0.0f32; 8];
    backend.sgemm_at(2, 4, 3, &a, &b, &mut c).unwrap();
    assert_approx_eq(&c, &expected, 1e-4);
}

#[test]
fn test_cuda_sgemm_bt() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // A (2x3) * B^T where B stored as (n=4 x k=3)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let expected = cpu_matmul(2, 4, 3, &a, &b, Transpose::None, Transpose::Ordinary);
    let mut c = vec![0.0f32; 8];
    backend.sgemm_bt(2, 4, 3, &a, &b, &mut c).unwrap();
    assert_approx_eq(&c, &expected, 1e-4);
}

/// Verifies large single-precision matrix multiplication on the CUDA backend against the CPU reference.
///
/// Uses m=64, n=32, k=128 with deterministic input patterns, computes a CPU reference via `cpu_matmul`,
/// runs `CudaBackend::sgemm`, and asserts the GPU result matches the reference within an absolute tolerance of 1e-2.
///
/// # Examples
///
/// ```
/// // Construct inputs and compare GPU sgemm against cpu_matmul reference.
/// let (m, n, k) = (64, 32, 128);
/// let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
/// let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
/// let expected = cpu_matmul(m, n, k, &a, &b, Transpose::None, Transpose::None);
/// let mut c = vec![0.0f32; m * n];
/// backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();
/// assert_approx_eq(&c, &expected, 1e-2);
/// ```
#[test]
fn test_cuda_sgemm_large() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let (m, n, k) = (64, 32, 128);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let expected = cpu_matmul(m, n, k, &a, &b, Transpose::None, Transpose::None);
    let mut c = vec![0.0f32; m * n];
    backend.sgemm(m, n, k, &a, &b, &mut c).unwrap();
    assert_approx_eq(&c, &expected, 1e-2);
}

#[test]
fn test_cuda_sgemm_dimension_mismatch() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let a = vec![1.0; 4]; // too small for 2x3
    let b = vec![1.0; 12];
    let mut c = vec![0.0; 8];
    let result = backend.sgemm(2, 4, 3, &a, &b, &mut c);
    assert!(result.is_err());
}

// ── Element-wise kernel tests ──────────────────────────────────────

#[test]
fn test_cuda_elementwise_relu() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5];
    backend.relu(&mut data).unwrap();
    let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5];
    assert_approx_eq(&data, &expected, 1e-6);
}

#[test]
fn test_cuda_elementwise_relu_empty() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let mut data: Vec<f32> = vec![];
    backend.relu(&mut data).unwrap();
    assert!(data.is_empty());
}

/// Verifies that the CUDA ReLU backward kernel writes the upstream gradient to positions where the input is greater than zero and writes zero elsewhere.
///
/// The test constructs sample input and upstream gradients, runs `relu_backward`, and asserts that
/// `grad_input[i] == grad_output[i]` when `input[i] > 0`, otherwise `grad_input[i] == 0`.
#[test]
fn test_cuda_elementwise_relu_backward() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5];
    let grad_output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut grad_input = vec![0.0; 7];
    backend
        .relu_backward(&input, &grad_output, &mut grad_input)
        .unwrap();
    // grad_input[i] = input[i] > 0 ? grad_output[i] : 0
    let expected = vec![0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0];
    assert_approx_eq(&grad_input, &expected, 1e-6);
}

#[test]
fn test_cuda_elementwise_sigmoid() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    let mut data = vec![0.0, 1.0, -1.0, 10.0, -10.0];
    backend.sigmoid(&mut data).unwrap();

    // sigmoid(0) = 0.5
    assert!((data[0] - 0.5).abs() < 1e-5);
    // sigmoid(x) in (0, 1)
    for &v in &data {
        assert!(v > 0.0 && v < 1.0);
    }
    // sigmoid(1) ≈ 0.7310586
    assert!((data[1] - 0.7310586).abs() < 1e-4);
    // sigmoid(-1) ≈ 0.2689414
    assert!((data[2] - 0.2689414).abs() < 1e-4);
}

#[test]
fn test_cuda_elementwise_sigmoid_backward() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    let sigmoid_output = vec![0.5, 0.7310586, 0.2689414];
    let grad_output = vec![1.0, 1.0, 1.0];
    let mut grad_input = vec![0.0; 3];
    backend
        .sigmoid_backward(&sigmoid_output, &grad_output, &mut grad_input)
        .unwrap();

    // grad = s * (1 - s) * grad_output
    // s=0.5: 0.5 * 0.5 = 0.25
    assert!((grad_input[0] - 0.25).abs() < 1e-5);
    // s=0.731: 0.731 * 0.269 ≈ 0.1966
    assert!((grad_input[1] - 0.7310586 * 0.2689414).abs() < 1e-4);
}

#[test]
fn test_cuda_elementwise_add_bias() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    // 2 rows x 3 columns
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias = vec![10.0, 20.0, 30.0];
    backend.add_bias(&mut data, &bias, 2, 3).unwrap();
    let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    assert_approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_cuda_elementwise_sum_rows() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    // 3 rows x 4 columns
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut out = vec![0.0; 4];
    backend.sum_rows(&data, &mut out, 3, 4).unwrap();
    let expected = vec![15.0, 18.0, 21.0, 24.0];
    assert_approx_eq(&out, &expected, 1e-5);
}

// ── Conv2D kernel tests ────────────────────────────────────────────

/// Computes a CPU reference for a batched 2D convolution in NCHW layout.
///
/// Produces an output buffer of shape (batch_size, out_channels, out_h, out_w) in NCHW order,
/// where out_h and out_w are derived from the input spatial dimensions, kernel size, stride,
/// and padding. The per-output-channel `bias` is added to each output element; `stride` and
/// `padding` are applied to the input before convolution. If a kernel element samples outside
/// the input bounds (after applying padding), that contribution is skipped.
///
/// # Examples
///
/// ```
/// let input = vec![1.0f32, 2.0, 3.0, 4.0]; // 1×1×2×2 (batch=1, in_ch=1, H=2, W=2)
/// let filters = vec![1.0f32]; // 1×1×1×1 (out_ch=1, in_ch=1, kh=1, kw=1)
/// let bias = vec![0.0f32]; // out_ch=1
/// let out = cpu_conv2d_forward(&input, &filters, &bias, 1, 1, 1, 2, 2, 1, 1, 1, 0);
/// assert_eq!(out, input);
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
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias[oc];
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = (oh * stride + kh) as isize - padding as isize;
                                let iw = (ow * stride + kw) as isize - padding as isize;
                                if ih >= 0
                                    && ih < input_h as isize
                                    && iw >= 0
                                    && iw < input_w as isize
                                {
                                    let iv = input[((b * in_channels + ic) * input_h
                                        + ih as usize)
                                        * input_w
                                        + iw as usize];
                                    let fv = filters
                                        [((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                                    sum += iv * fv;
                                }
                            }
                        }
                    }
                    output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    }
    output
}

#[test]
fn test_cuda_conv2d_forward_basic() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("No CUDA device available, skipping test");
            return;
        }
    };

    // 1 batch, 1 in_channel, 2 out_channels, 4x4 input, 3x3 kernel, stride=1, padding=0
    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 2;
    let (input_h, input_w) = (4, 4);
    let (kernel_h, kernel_w) = (3, 3);
    let stride = 1;
    let padding = 0;
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1; // 2
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1; // 2

    let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let bias = vec![0.1, -0.1];

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

    assert_approx_eq(&output, &expected, 1e-4);
}

#[test]
fn test_cuda_conv2d_forward_with_padding() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let batch_size = 2;
    let in_channels = 1;
    let out_channels = 1;
    let (input_h, input_w) = (3, 3);
    let (kernel_h, kernel_w) = (3, 3);
    let stride = 1;
    let padding = 1;
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1; // 3
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1; // 3

    let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let filters = vec![1.0; out_channels * in_channels * kernel_h * kernel_w];
    let bias = vec![0.0];

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

    assert_approx_eq(&output, &expected, 1e-4);
}

/// Verifies that `conv2d_forward` produces the same output as a CPU reference for a multi-channel input.
///
/// This integration test constructs a 3-channel input, two output channels, 3×3 kernels with padding,
/// and compares the GPU backend's forward convolution result against `cpu_conv2d_forward` within 1e-3.
///
/// # Examples
///
/// ```
/// // Setup backend (test skips if CUDA is unavailable)
/// let backend = match CudaBackend::new() {
///     Ok(b) => b,
///     Err(_) => return,
/// };
///
/// // Shapes and parameters
/// let batch_size = 1;
/// let in_channels = 3;
/// let out_channels = 2;
/// let (input_h, input_w) = (5, 5);
/// let (kernel_h, kernel_w) = (3, 3);
/// let stride = 1;
/// let padding = 1;
///
/// // Randomized example data (deterministic pattern for test)
/// let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
///     .map(|i| ((i % 7) as f32) * 0.1)
///     .collect();
/// let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
///     .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
///     .collect();
/// let bias = vec![0.5, -0.3];
///
/// // Reference and GPU outputs
/// let expected = cpu_conv2d_forward(
///     &input, &filters, &bias,
///     batch_size, in_channels, out_channels,
///     input_h, input_w, kernel_h, kernel_w, stride, padding,
/// );
/// let mut output = vec![0.0f32; batch_size * out_channels * ((input_h + 2 * padding - kernel_h) / stride + 1) * ((input_w + 2 * padding - kernel_w) / stride + 1)];
///
/// backend.conv2d_forward(
///     &input, &filters, &bias, &mut output,
///     batch_size, in_channels, out_channels,
///     input_h, input_w, kernel_h, kernel_w, stride, padding,
/// ).unwrap();
///
/// assert_approx_eq(&output, &expected, 1e-3);
/// ```
#[test]
fn test_cuda_conv2d_forward_multichannel() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let batch_size = 1;
    let in_channels = 3;
    let out_channels = 2;
    let (input_h, input_w) = (5, 5);
    let (kernel_h, kernel_w) = (3, 3);
    let stride = 1;
    let padding = 1;
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
        .map(|i| ((i % 7) as f32) * 0.1)
        .collect();
    let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    let bias = vec![0.5, -0.3];

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

    assert_approx_eq(&output, &expected, 1e-3);
}

#[test]
fn test_cuda_conv2d_backward_bias() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let batch_size = 2;
    let in_channels = 1;
    let out_channels = 2;
    let (input_h, input_w) = (4, 4);
    let (kernel_h, kernel_w) = (3, 3);
    let stride = 1;
    let padding = 0;
    let out_h = 2;
    let out_w = 2;

    let input: Vec<f32> = (0..batch_size * in_channels * input_h * input_w)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let filters: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let grad_output: Vec<f32> = (0..batch_size * out_channels * out_h * out_w)
        .map(|i| (i as f32) * 0.1)
        .collect();

    let mut grad_input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
    let mut grad_filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
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

    // CPU reference for bias gradient: sum over batch and spatial dims
    let mut expected_grad_bias = vec![0.0f32; out_channels];
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    expected_grad_bias[oc] +=
                        grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                }
            }
        }
    }
    assert_approx_eq(&grad_bias, &expected_grad_bias, 1e-4);
}

#[test]
fn test_cuda_conv2d_backward_filters() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Simple case: 1 batch, 1 in/out channel, 3x3 input, 2x2 kernel, no padding
    let batch_size = 1;
    let in_channels = 1;
    let out_channels = 1;
    let (input_h, input_w) = (3, 3);
    let (kernel_h, kernel_w) = (2, 2);
    let stride = 1;
    let padding = 0;
    let out_h = 2;
    let out_w = 2;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let filters = vec![0.1, 0.2, 0.3, 0.4];
    let grad_output = vec![1.0, 1.0, 1.0, 1.0];

    let mut grad_input = vec![0.0f32; batch_size * in_channels * input_h * input_w];
    let mut grad_filters = vec![0.0f32; out_channels * in_channels * kernel_h * kernel_w];
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

    // grad_filters[kh][kw] = sum over (b, oh, ow) of input[oh*s+kh][ow*s+kw] * grad_output[oh][ow]
    // kh=0,kw=0: 1*1+2*1+4*1+5*1 = 12
    // kh=0,kw=1: 2*1+3*1+5*1+6*1 = 16
    // kh=1,kw=0: 4*1+5*1+7*1+8*1 = 24
    // kh=1,kw=1: 5*1+6*1+8*1+9*1 = 28
    let expected_grad_filters = vec![12.0, 16.0, 24.0, 28.0];
    assert_approx_eq(&grad_filters, &expected_grad_filters, 1e-4);
}

#[test]
fn test_cuda_conv2d_dimension_mismatch() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let input = vec![1.0; 4]; // too small
    let filters = vec![1.0; 9];
    let bias = vec![0.0];
    let mut output = vec![0.0; 1];

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

#[test]
fn test_cuda_elementwise_relu_large() {
    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let n = 10000;
    let mut data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
    backend.relu(&mut data).unwrap();
    for (i, &v) in data.iter().enumerate() {
        let orig = (i as f32) - (n as f32 / 2.0);
        let expected = if orig > 0.0 { orig } else { 0.0 };
        assert!((v - expected).abs() < 1e-5, "mismatch at {}", i);
    }
}
