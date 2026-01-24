// Tests for backward propagation: gradient shapes and numerical stability.
// These functions are copied from the main binaries for testing purposes.

use approx::assert_relative_eq;

// ============================================================================
// Simple MLP (f64, sigmoid activation) - from mlp_simple.rs
// ============================================================================

// Sigmoid activation function (f64 version).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Sigmoid derivative assuming x = sigmoid(z).
fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// Dense layer: weights (input x output) and biases (output).
struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: LinearLayer,
    output_layer: LinearLayer,
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

// Backprop: compute deltas for output and hidden layers.
fn backward(
    nn: &NeuralNetwork,
    _inputs: &[f64],
    hidden_outputs: &[f64],
    output_outputs: &[f64],
    errors: &[f64],
    delta_hidden: &mut [f64],
    delta_output: &mut [f64],
) {
    for (i, d_out) in delta_output.iter_mut().enumerate().take(nn.output_layer.output_size) {
        // delta_out = error * activation derivative.
        *d_out = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    for (i, d_hid) in delta_hidden.iter_mut().enumerate().take(nn.hidden_layer.output_size) {
        // Error backpropagated from output to hidden layer.
        let mut error = 0.0;
        for (j, d_out) in delta_output.iter().enumerate().take(nn.output_layer.output_size) {
            error += d_out * nn.output_layer.weights[i][j];
        }
        *d_hid = error * sigmoid_derivative(hidden_outputs[i]);
    }
}

// Update weights and biases with gradient descent (SGD).
fn update_weights_biases(
    layer: &mut LinearLayer,
    inputs: &[f64],
    deltas: &[f64],
    learning_rate: f64,
) {
    for (i, inp) in inputs.iter().enumerate().take(layer.input_size) {
        for (j, delta) in deltas.iter().enumerate().take(layer.output_size) {
            layer.weights[i][j] += learning_rate * delta * inp;
        }
    }

    for (i, delta) in deltas.iter().enumerate().take(layer.output_size) {
        layer.biases[i] += learning_rate * delta;
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

fn sum_rows(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }

    for row in data.chunks_exact(cols).take(rows) {
        for (value, sum) in row.iter().zip(out.iter_mut()) {
            *sum += *value;
        }
    }
}

fn compute_delta_and_loss(
    outputs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
) -> f32 {
    let mut total_loss = 0.0f32;
    let epsilon = 1e-9f32;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let prob = outputs[row_start + label as usize].max(epsilon);
        total_loss -= prob.ln();

        let row = &outputs[row_start..row_start + cols];
        let delta_row = &mut delta[row_start..row_start + cols];
        for (j, value) in row.iter().enumerate() {
            let mut v = *value;
            if j == label as usize {
                v -= 1.0;
            }
            delta_row[j] = v;
        }
    }

    total_loss
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Simple MLP backward pass tests (f64, sigmoid)
    // ========================================================================

    #[test]
    fn test_backward_gradient_dimensions_simple() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 3,
                output_size: 4,
                weights: vec![vec![0.1; 4]; 3],
                biases: vec![0.1; 4],
            },
            output_layer: LinearLayer {
                input_size: 4,
                output_size: 2,
                weights: vec![vec![0.1; 2]; 4],
                biases: vec![0.1; 2],
            },
        };

        let inputs = vec![1.0, 2.0, 3.0];
        let mut hidden_outputs = vec![0.0; 4];
        let mut output_outputs = vec![0.0; 2];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = vec![0.1, 0.2];
        let mut delta_hidden = vec![0.0; 4];
        let mut delta_output = vec![0.0; 2];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        assert_eq!(delta_output.len(), 2);
        assert_eq!(delta_hidden.len(), 4);
    }

    #[test]
    fn test_backward_no_nan_inf_simple() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 3,
                weights: vec![vec![0.5, -0.3, 0.2], vec![-0.4, 0.6, -0.1]],
                biases: vec![0.1, 0.2, 0.3],
            },
            output_layer: LinearLayer {
                input_size: 3,
                output_size: 2,
                weights: vec![vec![0.3, 0.4], vec![0.5, 0.6], vec![0.7, 0.8]],
                biases: vec![0.1, 0.2],
            },
        };

        let inputs = vec![1.0, 2.0];
        let mut hidden_outputs = vec![0.0; 3];
        let mut output_outputs = vec![0.0; 2];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = vec![0.5, -0.3];
        let mut delta_hidden = vec![0.0; 3];
        let mut delta_output = vec![0.0; 2];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        for &delta in &delta_output {
            assert!(!delta.is_nan() && !delta.is_infinite());
        }

        for &delta in &delta_hidden {
            assert!(!delta.is_nan() && !delta.is_infinite());
        }
    }

    #[test]
    fn test_backward_zero_error() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.5], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 2.0];
        let mut hidden_outputs = vec![0.0; 2];
        let mut output_outputs = vec![0.0; 1];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = vec![0.0];
        let mut delta_hidden = vec![0.0; 2];
        let mut delta_output = vec![0.0; 1];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        assert_relative_eq!(delta_output[0], 0.0, epsilon = 1e-10);

        for &delta in &delta_hidden {
            assert_relative_eq!(delta, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_backward_large_error() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
                biases: vec![0.0, 0.0],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.5], vec![0.6]],
                biases: vec![0.0],
            },
        };

        let inputs = vec![1.0, 2.0];
        let mut hidden_outputs = vec![0.0; 2];
        let mut output_outputs = vec![0.0; 1];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = vec![10.0];
        let mut delta_hidden = vec![0.0; 2];
        let mut delta_output = vec![0.0; 1];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        for &delta in &delta_output {
            assert!(!delta.is_nan() && !delta.is_infinite());
        }

        for &delta in &delta_hidden {
            assert!(!delta.is_nan() && !delta.is_infinite());
        }
    }

    #[test]
    fn test_backward_output_delta_calculation() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                biases: vec![0.0, 0.0],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![1.0], vec![1.0]],
                biases: vec![0.0],
            },
        };

        let inputs = vec![1.0, 1.0];
        let mut hidden_outputs = vec![0.0; 2];
        let mut output_outputs = vec![0.0; 1];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let error = 1.0;
        let errors = vec![error];
        let mut delta_hidden = vec![0.0; 2];
        let mut delta_output = vec![0.0; 1];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        let expected_delta_output = error * sigmoid_derivative(output_outputs[0]);
        assert_relative_eq!(delta_output[0], expected_delta_output, epsilon = 1e-10);
    }

    #[test]
    fn test_backward_hidden_delta_propagation() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 3,
                weights: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
                biases: vec![0.1, 0.2, 0.3],
            },
            output_layer: LinearLayer {
                input_size: 3,
                output_size: 2,
                weights: vec![vec![0.7, 0.8], vec![0.9, 1.0], vec![1.1, 1.2]],
                biases: vec![0.1, 0.2],
            },
        };

        let inputs = vec![1.0, 2.0];
        let mut hidden_outputs = vec![0.0; 3];
        let mut output_outputs = vec![0.0; 2];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = vec![0.5, 0.3];
        let mut delta_hidden = vec![0.0; 3];
        let mut delta_output = vec![0.0; 2];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        for (i, &hidden_output) in hidden_outputs.iter().enumerate().take(3) {
            let mut expected_error = 0.0;
            for (j, &delta_val) in delta_output.iter().enumerate().take(2) {
                expected_error += delta_val * nn.output_layer.weights[i][j];
            }
            let expected_delta_hidden = expected_error * sigmoid_derivative(hidden_output);
            assert_relative_eq!(delta_hidden[i], expected_delta_hidden, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_weight_update_dimensions_simple() {
        let mut layer = LinearLayer {
            input_size: 3,
            output_size: 2,
            weights: vec![vec![0.5, 0.6], vec![0.7, 0.8], vec![0.9, 1.0]],
            biases: vec![0.1, 0.2],
        };

        let inputs = vec![1.0, 2.0, 3.0];
        let deltas = vec![0.1, 0.2];
        let learning_rate = 0.01;

        let old_weights = layer.weights.clone();
        let old_biases = layer.biases.clone();

        update_weights_biases(&mut layer, &inputs, &deltas, learning_rate);

        assert_eq!(layer.weights.len(), 3);
        assert_eq!(layer.weights[0].len(), 2);
        assert_eq!(layer.biases.len(), 2);

        for (i, old_weight_row) in old_weights.iter().enumerate().take(3) {
            for (j, &old_weight_val) in old_weight_row.iter().enumerate().take(2) {
                assert_ne!(layer.weights[i][j], old_weight_val);
            }
        }

        for (i, &old_bias) in old_biases.iter().enumerate().take(2) {
            assert_ne!(layer.biases[i], old_bias);
        }
    }

    #[test]
    fn test_weight_update_zero_deltas() {
        let mut layer = LinearLayer {
            input_size: 2,
            output_size: 2,
            weights: vec![vec![0.5, 0.6], vec![0.7, 0.8]],
            biases: vec![0.1, 0.2],
        };

        let inputs = vec![1.0, 2.0];
        let deltas = vec![0.0, 0.0];
        let learning_rate = 0.01;

        let old_weights = layer.weights.clone();
        let old_biases = layer.biases.clone();

        update_weights_biases(&mut layer, &inputs, &deltas, learning_rate);

        for (i, old_weight_row) in old_weights.iter().enumerate().take(2) {
            for (j, &old_weight_val) in old_weight_row.iter().enumerate().take(2) {
                assert_relative_eq!(layer.weights[i][j], old_weight_val, epsilon = 1e-10);
            }
        }

        for (i, &old_bias) in old_biases.iter().enumerate().take(2) {
            assert_relative_eq!(layer.biases[i], old_bias, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_weight_update_calculation() {
        let mut layer = LinearLayer {
            input_size: 2,
            output_size: 1,
            weights: vec![vec![0.5], vec![0.6]],
            biases: vec![0.1],
        };

        let inputs = vec![2.0, 3.0];
        let deltas = vec![0.5];
        let learning_rate = 0.1;

        update_weights_biases(&mut layer, &inputs, &deltas, learning_rate);

        let expected_w0 = 0.5 + 0.1 * 0.5 * 2.0;
        let expected_w1 = 0.6 + 0.1 * 0.5 * 3.0;
        let expected_b = 0.1 + 0.1 * 0.5;

        assert_relative_eq!(layer.weights[0][0], expected_w0, epsilon = 1e-10);
        assert_relative_eq!(layer.weights[1][0], expected_w1, epsilon = 1e-10);
        assert_relative_eq!(layer.biases[0], expected_b, epsilon = 1e-10);
    }

    // ========================================================================
    // GEMM-based backward pass tests (f32, batch processing)
    // ========================================================================

    #[test]
    fn test_compute_delta_dimensions_batch() {
        let batch_size = 4;
        let num_classes = 10;

        let mut outputs = vec![0.1f32; batch_size * num_classes];
        softmax_rows(&mut outputs, batch_size, num_classes);

        let labels = vec![0, 3, 7, 9];
        let mut delta = vec![0.0f32; batch_size * num_classes];

        let loss = compute_delta_and_loss(&outputs, &labels, batch_size, num_classes, &mut delta);

        assert_eq!(delta.len(), batch_size * num_classes);
        assert!(loss > 0.0);
        assert!(!loss.is_nan() && !loss.is_infinite());
    }

    #[test]
    fn test_compute_delta_no_nan_inf() {
        let batch_size = 3;
        let num_classes = 5;

        let mut outputs = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ];
        softmax_rows(&mut outputs, batch_size, num_classes);

        let labels = vec![2, 1, 4];
        let mut delta = vec![0.0f32; batch_size * num_classes];

        let loss = compute_delta_and_loss(&outputs, &labels, batch_size, num_classes, &mut delta);

        assert!(!loss.is_nan() && !loss.is_infinite());

        for &d in &delta {
            assert!(!d.is_nan() && !d.is_infinite());
        }
    }

    #[test]
    fn test_compute_delta_correct_label() {
        let batch_size = 1;
        let num_classes = 3;

        let mut outputs = vec![0.2, 0.5, 0.3];
        softmax_rows(&mut outputs, batch_size, num_classes);

        let labels = vec![1];
        let mut delta = vec![0.0f32; batch_size * num_classes];

        compute_delta_and_loss(&outputs, &labels, batch_size, num_classes, &mut delta);

        assert_relative_eq!(delta[1], outputs[1] - 1.0, epsilon = 1e-6);

        assert_relative_eq!(delta[0], outputs[0], epsilon = 1e-6);
        assert_relative_eq!(delta[2], outputs[2], epsilon = 1e-6);
    }

    #[test]
    fn test_backward_gemm_gradient_dimensions() {
        let batch_size = 4;
        let input_size = 784;
        let hidden_size = 128;
        let output_size = 10;

        let batch_inputs = vec![0.1f32; batch_size * input_size];
        let mut a1 = vec![0.0f32; batch_size * hidden_size];
        let mut a2 = vec![0.0f32; batch_size * output_size];
        let mut dz2 = vec![0.0f32; batch_size * output_size];
        let mut dz1 = vec![0.0f32; batch_size * hidden_size];
        let mut grad_w1 = vec![0.0f32; input_size * hidden_size];
        let mut grad_w2 = vec![0.0f32; hidden_size * output_size];
        let mut grad_b1 = vec![0.0f32; hidden_size];
        let mut grad_b2 = vec![0.0f32; output_size];

        let w1 = vec![0.01f32; input_size * hidden_size];
        let b1 = vec![0.0f32; hidden_size];
        let w2 = vec![0.01f32; hidden_size * output_size];
        let b2 = vec![0.0f32; output_size];

        sgemm_wrapper(
            batch_size,
            hidden_size,
            input_size,
            &batch_inputs,
            input_size,
            &w1,
            hidden_size,
            &mut a1,
            hidden_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a1, batch_size, hidden_size, &b1);
        relu_inplace(&mut a1);

        sgemm_wrapper(
            batch_size,
            output_size,
            hidden_size,
            &a1,
            hidden_size,
            &w2,
            output_size,
            &mut a2,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a2, batch_size, output_size, &b2);
        softmax_rows(&mut a2, batch_size, output_size);

        let labels = vec![0, 3, 7, 9];
        compute_delta_and_loss(&a2, &labels, batch_size, output_size, &mut dz2);

        let scale = 1.0f32 / batch_size as f32;

        sgemm_wrapper(
            hidden_size,
            output_size,
            batch_size,
            &a1,
            hidden_size,
            &dz2,
            output_size,
            &mut grad_w2,
            output_size,
            true,
            false,
            scale,
            0.0,
        );
        sum_rows(&dz2, batch_size, output_size, &mut grad_b2);

        sgemm_wrapper(
            batch_size,
            hidden_size,
            output_size,
            &dz2,
            output_size,
            &w2,
            output_size,
            &mut dz1,
            hidden_size,
            false,
            true,
            1.0,
            0.0,
        );

        for i in 0..batch_size * hidden_size {
            if a1[i] <= 0.0 {
                dz1[i] = 0.0;
            }
        }

        sgemm_wrapper(
            input_size,
            hidden_size,
            batch_size,
            &batch_inputs,
            input_size,
            &dz1,
            hidden_size,
            &mut grad_w1,
            hidden_size,
            true,
            false,
            scale,
            0.0,
        );
        sum_rows(&dz1, batch_size, hidden_size, &mut grad_b1);

        assert_eq!(grad_w1.len(), input_size * hidden_size);
        assert_eq!(grad_b1.len(), hidden_size);
        assert_eq!(grad_w2.len(), hidden_size * output_size);
        assert_eq!(grad_b2.len(), output_size);

        for &g in &grad_w1 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_b1 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_w2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_b2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
    }

    #[test]
    fn test_backward_gemm_small_batch() {
        let batch_size = 2;
        let input_size = 4;
        let hidden_size = 3;
        let output_size = 2;

        let batch_inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut a1 = vec![0.0f32; batch_size * hidden_size];
        let mut a2 = vec![0.0f32; batch_size * output_size];
        let mut dz2 = vec![0.0f32; batch_size * output_size];
        let mut grad_w2 = vec![0.0f32; hidden_size * output_size];
        let mut grad_b2 = vec![0.0f32; output_size];

        let w1 = vec![0.1f32; input_size * hidden_size];
        let b1 = vec![0.1f32; hidden_size];
        let w2 = vec![0.1f32; hidden_size * output_size];
        let b2 = vec![0.1f32; output_size];

        sgemm_wrapper(
            batch_size,
            hidden_size,
            input_size,
            &batch_inputs,
            input_size,
            &w1,
            hidden_size,
            &mut a1,
            hidden_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a1, batch_size, hidden_size, &b1);
        relu_inplace(&mut a1);

        sgemm_wrapper(
            batch_size,
            output_size,
            hidden_size,
            &a1,
            hidden_size,
            &w2,
            output_size,
            &mut a2,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a2, batch_size, output_size, &b2);
        softmax_rows(&mut a2, batch_size, output_size);

        let labels = vec![0, 1];
        compute_delta_and_loss(&a2, &labels, batch_size, output_size, &mut dz2);

        let scale = 1.0f32 / batch_size as f32;

        sgemm_wrapper(
            hidden_size,
            output_size,
            batch_size,
            &a1,
            hidden_size,
            &dz2,
            output_size,
            &mut grad_w2,
            output_size,
            true,
            false,
            scale,
            0.0,
        );
        sum_rows(&dz2, batch_size, output_size, &mut grad_b2);

        assert_eq!(grad_w2.len(), hidden_size * output_size);
        assert_eq!(grad_b2.len(), output_size);

        for &g in &grad_w2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_b2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
    }

    #[test]
    fn test_backward_gemm_relu_gradient_masking() {
        let batch_size = 2;
        let hidden_size = 4;

        let mut a1 = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        relu_inplace(&mut a1);

        let dz1_before = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let mut dz1 = dz1_before.clone();

        for i in 0..batch_size * hidden_size {
            if a1[i] <= 0.0 {
                dz1[i] = 0.0;
            }
        }

        assert_relative_eq!(dz1[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(dz1[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(dz1[2], 0.7, epsilon = 1e-6);
        assert_relative_eq!(dz1[3], 0.0, epsilon = 1e-6);
        assert_relative_eq!(dz1[4], 0.9, epsilon = 1e-6);
        assert_relative_eq!(dz1[5], 0.0, epsilon = 1e-6);
        assert_relative_eq!(dz1[6], 1.1, epsilon = 1e-6);
        assert_relative_eq!(dz1[7], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sum_rows_gradient_accumulation() {
        let batch_size = 3;
        let cols = 4;

        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let mut out = vec![0.0f32; cols];
        sum_rows(&data, batch_size, cols, &mut out);

        assert_relative_eq!(out[0], 15.0, epsilon = 1e-5);
        assert_relative_eq!(out[1], 18.0, epsilon = 1e-5);
        assert_relative_eq!(out[2], 21.0, epsilon = 1e-5);
        assert_relative_eq!(out[3], 24.0, epsilon = 1e-5);
    }

    #[test]
    fn test_backward_gradient_batch_consistency() {
        let batch_size = 4;
        let input_size = 3;
        let output_size = 2;

        let batch_inputs = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let mut a1 = vec![0.0f32; batch_size * output_size];
        let mut delta = vec![0.0f32; batch_size * output_size];
        let mut grad_w = vec![0.0f32; input_size * output_size];

        let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = vec![0.1, 0.2];

        sgemm_wrapper(
            batch_size,
            output_size,
            input_size,
            &batch_inputs,
            input_size,
            &w,
            output_size,
            &mut a1,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a1, batch_size, output_size, &b);
        softmax_rows(&mut a1, batch_size, output_size);

        let labels = vec![0, 1, 0, 1];
        compute_delta_and_loss(&a1, &labels, batch_size, output_size, &mut delta);

        let scale = 1.0f32 / batch_size as f32;
        sgemm_wrapper(
            input_size,
            output_size,
            batch_size,
            &batch_inputs,
            input_size,
            &delta,
            output_size,
            &mut grad_w,
            output_size,
            true,
            false,
            scale,
            0.0,
        );

        for &g in &grad_w {
            assert!(!g.is_nan() && !g.is_infinite());
        }
    }

    #[test]
    fn test_backward_full_pass_two_layers() {
        let batch_size = 2;
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;

        let batch_inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut a1 = vec![0.0f32; batch_size * hidden_size];
        let mut a2 = vec![0.0f32; batch_size * output_size];
        let mut dz2 = vec![0.0f32; batch_size * output_size];
        let mut dz1 = vec![0.0f32; batch_size * hidden_size];
        let mut grad_w1 = vec![0.0f32; input_size * hidden_size];
        let mut grad_w2 = vec![0.0f32; hidden_size * output_size];
        let mut grad_b1 = vec![0.0f32; hidden_size];
        let mut grad_b2 = vec![0.0f32; output_size];

        let w1 = vec![0.1f32; input_size * hidden_size];
        let b1 = vec![0.1f32; hidden_size];
        let w2 = vec![0.1f32; hidden_size * output_size];
        let b2 = vec![0.1f32; output_size];

        sgemm_wrapper(
            batch_size,
            hidden_size,
            input_size,
            &batch_inputs,
            input_size,
            &w1,
            hidden_size,
            &mut a1,
            hidden_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a1, batch_size, hidden_size, &b1);
        relu_inplace(&mut a1);

        sgemm_wrapper(
            batch_size,
            output_size,
            hidden_size,
            &a1,
            hidden_size,
            &w2,
            output_size,
            &mut a2,
            output_size,
            false,
            false,
            1.0,
            0.0,
        );
        add_bias(&mut a2, batch_size, output_size, &b2);
        softmax_rows(&mut a2, batch_size, output_size);

        let labels = vec![0, 1];
        let loss = compute_delta_and_loss(&a2, &labels, batch_size, output_size, &mut dz2);

        assert!(loss > 0.0);
        assert!(!loss.is_nan() && !loss.is_infinite());

        let scale = 1.0f32 / batch_size as f32;

        sgemm_wrapper(
            hidden_size,
            output_size,
            batch_size,
            &a1,
            hidden_size,
            &dz2,
            output_size,
            &mut grad_w2,
            output_size,
            true,
            false,
            scale,
            0.0,
        );
        sum_rows(&dz2, batch_size, output_size, &mut grad_b2);

        sgemm_wrapper(
            batch_size,
            hidden_size,
            output_size,
            &dz2,
            output_size,
            &w2,
            output_size,
            &mut dz1,
            hidden_size,
            false,
            true,
            1.0,
            0.0,
        );

        for i in 0..batch_size * hidden_size {
            if a1[i] <= 0.0 {
                dz1[i] = 0.0;
            }
        }

        sgemm_wrapper(
            input_size,
            hidden_size,
            batch_size,
            &batch_inputs,
            input_size,
            &dz1,
            hidden_size,
            &mut grad_w1,
            hidden_size,
            true,
            false,
            scale,
            0.0,
        );
        sum_rows(&dz1, batch_size, hidden_size, &mut grad_b1);

        assert_eq!(grad_w1.len(), input_size * hidden_size);
        assert_eq!(grad_b1.len(), hidden_size);
        assert_eq!(grad_w2.len(), hidden_size * output_size);
        assert_eq!(grad_b2.len(), output_size);

        for &g in &grad_w1 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_b1 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_w2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
        for &g in &grad_b2 {
            assert!(!g.is_nan() && !g.is_infinite());
        }
    }
}
