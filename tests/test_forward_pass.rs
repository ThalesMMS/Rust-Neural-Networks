// Tests for forward propagation: output dimensions and basic correctness.
// These functions are copied from the main binaries for testing purposes.

use approx::assert_relative_eq;

// ============================================================================
// Simple MLP (f64, sigmoid activation) - from mlp_simple.rs
// ============================================================================

// Sigmoid activation function (f64 version).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Dense layer: weights (input x output) and biases (output).
struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

// Layer forward: z = W*x + b, followed by sigmoid.
fn forward_propagation(layer: &LinearLayer, inputs: &[f64], outputs: &mut [f64]) {
    for (i, out) in outputs.iter_mut().enumerate().take(layer.output_size) {
        let mut activation = layer.biases[i];
        for (j, inp) in inputs.iter().enumerate().take(layer.input_size) {
            activation += inp * layer.weights[j][i];
        }
        *out = sigmoid(activation);
    }
}

// ============================================================================
// MNIST MLP (f32, GEMM-based) - from mnist_mlp.rs
// ============================================================================

extern crate blas_src;
use cblas::{sgemm, Layout, Transpose};

#[allow(clippy::too_many_arguments)]
fn sgemm_wrapper(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
) {
    let trans_a = if transpose_a {
        Transpose::Ordinary
    } else {
        Transpose::None
    };
    let trans_b = if transpose_b {
        Transpose::Ordinary
    } else {
        Transpose::None
    };

    unsafe {
        sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
    for row in outputs.chunks_exact_mut(cols).take(rows) {
        let mut max_value = row[0];
        for &value in row.iter().skip(1) {
            if value > max_value {
                max_value = value;
            }
        }

        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max_value).exp();
            sum += *value;
        }

        let inv_sum = 1.0f32 / sum;
        for value in row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Simple MLP forward pass tests (f64, sigmoid)
    // ========================================================================

    #[test]
    fn test_forward_propagation_output_dimensions() {
        let layer = LinearLayer {
            input_size: 3,
            output_size: 2,
            weights: vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
            biases: vec![0.1, 0.2],
        };

        let inputs = vec![1.0, 2.0, 3.0];
        let mut outputs = vec![0.0; 2];

        forward_propagation(&layer, &inputs, &mut outputs);

        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_forward_propagation_output_range() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 3,
            weights: vec![vec![0.5, -0.3, 0.2], vec![-0.4, 0.6, -0.1]],
            biases: vec![0.1, 0.2, 0.3],
        };

        let inputs = vec![1.0, 2.0];
        let mut outputs = vec![0.0; 3];

        forward_propagation(&layer, &inputs, &mut outputs);

        for &output in &outputs {
            assert!(output > 0.0 && output < 1.0);
        }
    }

    #[test]
    fn test_forward_propagation_zero_input() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 2,
            weights: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            biases: vec![0.0, 0.0],
        };

        let inputs = vec![0.0, 0.0];
        let mut outputs = vec![0.0; 2];

        forward_propagation(&layer, &inputs, &mut outputs);

        for &output in &outputs {
            assert_relative_eq!(output, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_forward_propagation_known_values() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![vec![1.0], vec![1.0]],
            biases: vec![0.0],
        };

        let inputs = vec![0.5, 0.5];
        let mut outputs = vec![0.0; 1];

        forward_propagation(&layer, &inputs, &mut outputs);

        let expected = sigmoid(1.0);
        assert_relative_eq!(outputs[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_propagation_single_neuron() {
        let layer = LinearLayer {
            input_size: 1,
            output_size: 1,
            weights: vec![vec![2.0]],
            biases: vec![1.0],
        };

        let inputs = vec![3.0];
        let mut outputs = vec![0.0; 1];

        forward_propagation(&layer, &inputs, &mut outputs);

        let expected = sigmoid(2.0 * 3.0 + 1.0);
        assert_relative_eq!(outputs[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_propagation_negative_inputs() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 2,
            weights: vec![vec![0.5, -0.5], vec![-0.3, 0.3]],
            biases: vec![0.1, -0.1],
        };

        let inputs = vec![-1.0, -2.0];
        let mut outputs = vec![0.0; 2];

        forward_propagation(&layer, &inputs, &mut outputs);

        for &output in &outputs {
            assert!(output > 0.0 && output < 1.0);
        }
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_forward_propagation_large_layer() {
        let layer = LinearLayer {
            input_size: 10,
            output_size: 5,
            weights: vec![vec![0.1; 5]; 10],
            biases: vec![0.0; 5],
        };

        let inputs = vec![1.0; 10];
        let mut outputs = vec![0.0; 5];

        forward_propagation(&layer, &inputs, &mut outputs);

        assert_eq!(outputs.len(), 5);
        for &output in &outputs {
            assert!(output > 0.0 && output < 1.0);
        }
    }

    #[test]
    fn test_forward_propagation_bias_effect() {
        let layer = LinearLayer {
            input_size: 2,
            output_size: 2,
            weights: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            biases: vec![5.0, -5.0],
        };

        let inputs = vec![1.0, 1.0];
        let mut outputs = vec![0.0; 2];

        forward_propagation(&layer, &inputs, &mut outputs);

        assert!(outputs[0] > 0.99);
        assert!(outputs[1] < 0.01);
    }

    // ========================================================================
    // GEMM-based forward pass tests (f32, batch processing)
    // ========================================================================

    #[test]
    fn test_gemm_forward_single_sample() {
        let batch_size = 1;
        let input_size = 4;
        let output_size = 3;

        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let biases = vec![0.1, 0.2, 0.3];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);

        assert_eq!(outputs.len(), 3);
        for &output in &outputs {
            assert!(!output.is_nan() && !output.is_infinite());
        }
    }

    #[test]
    fn test_gemm_forward_batch() {
        let batch_size = 4;
        let input_size = 3;
        let output_size = 2;

        let inputs = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let biases = vec![0.5, 1.0];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);

        assert_eq!(outputs.len(), batch_size * output_size);
        for &output in &outputs {
            assert!(!output.is_nan() && !output.is_infinite());
        }
    }

    #[test]
    fn test_gemm_forward_with_relu() {
        let batch_size = 2;
        let input_size = 3;
        let output_size = 4;

        let inputs = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let weights = vec![
            0.5, -0.3, 0.2, -0.1, -0.4, 0.6, -0.2, 0.3, 0.1, -0.5, 0.4, -0.2,
        ];
        let biases = vec![0.1, 0.2, 0.3, 0.4];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);
        relu_inplace(&mut outputs);

        assert_eq!(outputs.len(), batch_size * output_size);
        for &output in &outputs {
            assert!(output >= 0.0);
        }
    }

    #[test]
    fn test_gemm_forward_with_softmax() {
        let batch_size = 2;
        let input_size = 3;
        let output_size = 4;

        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let biases = vec![0.1, 0.2, 0.3, 0.4];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);
        softmax_rows(&mut outputs, batch_size, output_size);

        assert_eq!(outputs.len(), batch_size * output_size);

        for i in 0..batch_size {
            let row_start = i * output_size;
            let row_end = row_start + output_size;
            let row_sum: f32 = outputs[row_start..row_end].iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-5);

            for &output in &outputs[row_start..row_end] {
                assert!((0.0..=1.0).contains(&output));
            }
        }
    }

    #[test]
    fn test_gemm_forward_zero_weights() {
        let batch_size = 2;
        let input_size = 3;
        let output_size = 2;

        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = vec![0.0; input_size * output_size];
        let biases = vec![1.0, 2.0];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);

        for i in 0..batch_size {
            assert_relative_eq!(outputs[i * output_size], 1.0, epsilon = 1e-6);
            assert_relative_eq!(outputs[i * output_size + 1], 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gemm_forward_identity_weights() {
        let size = 3;
        let batch_size = 1;

        let inputs = vec![1.0, 2.0, 3.0];
        let mut weights = vec![0.0f32; size * size];
        for i in 0..size {
            weights[i * size + i] = 1.0;
        }
        let biases = vec![0.0; size];

        let mut outputs = vec![0.0f32; batch_size * size];

        sgemm_wrapper(
            batch_size,
            size,
            size,
            &inputs,
            size,
            &weights,
            size,
            &mut outputs,
            size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, size, &biases);

        for i in 0..size {
            assert_relative_eq!(outputs[i], inputs[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_full_forward_pass_two_layer() {
        let batch_size = 1;
        let input_size = 4;
        let hidden_size = 3;
        let output_size = 2;

        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let hidden_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let hidden_biases = vec![0.1, 0.2, 0.3];

        let output_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let output_biases = vec![0.1, 0.2];

        let mut hidden_outputs = vec![0.0f32; batch_size * hidden_size];

        sgemm_wrapper(
            batch_size,
            hidden_size,
            input_size,
            &inputs,
            input_size,
            &hidden_weights,
            hidden_size,
            &mut hidden_outputs,
            hidden_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut hidden_outputs, batch_size, hidden_size, &hidden_biases);
        relu_inplace(&mut hidden_outputs);

        let mut final_outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            hidden_size,
            &hidden_outputs,
            hidden_size,
            &output_weights,
            output_size,
            &mut final_outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut final_outputs, batch_size, output_size, &output_biases);
        softmax_rows(&mut final_outputs, batch_size, output_size);

        assert_eq!(final_outputs.len(), output_size);

        let sum: f32 = final_outputs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        for &output in &final_outputs {
            assert!((0.0..=1.0).contains(&output));
            assert!(!output.is_nan() && !output.is_infinite());
        }
    }

    #[test]
    fn test_forward_pass_output_consistency() {
        let batch_size = 3;
        let input_size = 2;
        let output_size = 4;

        let inputs = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let weights = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let biases = vec![0.1, 0.2, 0.3, 0.4];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);

        let row1 = &outputs[0..output_size];
        let row2 = &outputs[output_size..2 * output_size];
        let row3 = &outputs[2 * output_size..3 * output_size];

        for i in 0..output_size {
            assert_relative_eq!(row1[i], row2[i], epsilon = 1e-6);
            assert_relative_eq!(row2[i], row3[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_forward_pass_large_batch() {
        let batch_size = 64;
        let input_size = 10;
        let output_size = 5;

        let inputs = vec![0.5f32; batch_size * input_size];
        let weights = vec![0.1f32; input_size * output_size];
        let biases = vec![0.2f32; output_size];

        let mut outputs = vec![0.0f32; batch_size * output_size];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &inputs,
            input_size,
            &weights,
            output_size,
            &mut outputs,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );

        add_bias(&mut outputs, batch_size, output_size, &biases);
        relu_inplace(&mut outputs);

        assert_eq!(outputs.len(), batch_size * output_size);

        for &output in &outputs {
            assert!(output >= 0.0);
            assert!(!output.is_nan() && !output.is_infinite());
        }
    }
}
