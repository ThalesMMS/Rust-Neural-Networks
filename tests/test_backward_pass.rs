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
    for (i, d_out) in delta_output
        .iter_mut()
        .enumerate()
        .take(nn.output_layer.output_size)
    {
        // delta_out = error * activation derivative.
        *d_out = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    for (i, d_hid) in delta_hidden
        .iter_mut()
        .enumerate()
        .take(nn.hidden_layer.output_size)
    {
        // Error backpropagated from output to hidden layer.
        let mut error = 0.0;
        for (j, &d_out) in delta_output
            .iter()
            .enumerate()
            .take(nn.output_layer.output_size)
        {
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

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;
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
#[path = "test_backward_pass/tests.rs"]
mod tests;
