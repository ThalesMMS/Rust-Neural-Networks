// Tests for numerical gradient checking using finite differences.
// These tests verify that analytical gradients match numerical approximations.
// Following the pattern from mlp_simple.rs.

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
#[derive(Clone)]
struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

// Network with one hidden layer and one output layer.
#[derive(Clone)]
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
        *d_out = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    for (i, d_hid) in delta_hidden
        .iter_mut()
        .enumerate()
        .take(nn.hidden_layer.output_size)
    {
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

// Compute mean squared error for a single sample.
fn compute_loss(nn: &NeuralNetwork, inputs: &[f64], expected_outputs: &[f64]) -> f64 {
    let mut hidden_outputs = vec![0.0; nn.hidden_layer.output_size];
    let mut output_outputs = vec![0.0; nn.output_layer.output_size];

    forward_propagation(&nn.hidden_layer, inputs, &mut hidden_outputs);
    forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

    let mut loss = 0.0;
    for i in 0..nn.output_layer.output_size {
        let error = expected_outputs[i] - output_outputs[i];
        loss += error * error;
    }
    loss / 2.0
}

// Compute numerical gradient for a weight using finite differences.
fn numerical_gradient_weight(
    nn: &NeuralNetwork,
    inputs: &[f64],
    expected_outputs: &[f64],
    layer_idx: usize,
    i: usize,
    j: usize,
    epsilon: f64,
) -> f64 {
    let mut nn_plus = nn.clone();
    let mut nn_minus = nn.clone();

    if layer_idx == 0 {
        nn_plus.hidden_layer.weights[i][j] += epsilon;
        nn_minus.hidden_layer.weights[i][j] -= epsilon;
    } else {
        nn_plus.output_layer.weights[i][j] += epsilon;
        nn_minus.output_layer.weights[i][j] -= epsilon;
    }

    let loss_plus = compute_loss(&nn_plus, inputs, expected_outputs);
    let loss_minus = compute_loss(&nn_minus, inputs, expected_outputs);

    (loss_plus - loss_minus) / (2.0 * epsilon)
}

// Compute numerical gradient for a bias using finite differences.
fn numerical_gradient_bias(
    nn: &NeuralNetwork,
    inputs: &[f64],
    expected_outputs: &[f64],
    layer_idx: usize,
    i: usize,
    epsilon: f64,
) -> f64 {
    let mut nn_plus = nn.clone();
    let mut nn_minus = nn.clone();

    if layer_idx == 0 {
        nn_plus.hidden_layer.biases[i] += epsilon;
        nn_minus.hidden_layer.biases[i] -= epsilon;
    } else {
        nn_plus.output_layer.biases[i] += epsilon;
        nn_minus.output_layer.biases[i] -= epsilon;
    }

    let loss_plus = compute_loss(&nn_plus, inputs, expected_outputs);
    let loss_minus = compute_loss(&nn_minus, inputs, expected_outputs);

    (loss_plus - loss_minus) / (2.0 * epsilon)
}

// Compute analytical gradients using backpropagation.
#[allow(clippy::ptr_arg)]
fn compute_analytical_gradients(
    nn: &NeuralNetwork,
    inputs: &[f64],
    expected_outputs: &[f64],
    grad_w1: &mut Vec<Vec<f64>>,
    grad_b1: &mut Vec<f64>,
    grad_w2: &mut Vec<Vec<f64>>,
    grad_b2: &mut Vec<f64>,
) {
    let mut hidden_outputs = vec![0.0; nn.hidden_layer.output_size];
    let mut output_outputs = vec![0.0; nn.output_layer.output_size];

    forward_propagation(&nn.hidden_layer, inputs, &mut hidden_outputs);
    forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

    let mut errors = vec![0.0; nn.output_layer.output_size];
    for i in 0..nn.output_layer.output_size {
        errors[i] = expected_outputs[i] - output_outputs[i];
    }

    let mut delta_hidden = vec![0.0; nn.hidden_layer.output_size];
    let mut delta_output = vec![0.0; nn.output_layer.output_size];

    backward(
        nn,
        inputs,
        &hidden_outputs,
        &output_outputs,
        &errors,
        &mut delta_hidden,
        &mut delta_output,
    );

    for i in 0..nn.hidden_layer.input_size {
        for j in 0..nn.hidden_layer.output_size {
            grad_w1[i][j] = -delta_hidden[j] * inputs[i];
        }
    }

    for i in 0..nn.hidden_layer.output_size {
        grad_b1[i] = -delta_hidden[i];
    }

    for i in 0..nn.output_layer.input_size {
        for j in 0..nn.output_layer.output_size {
            grad_w2[i][j] = -delta_output[j] * hidden_outputs[i];
        }
    }

    for i in 0..nn.output_layer.output_size {
        grad_b2[i] = -delta_output[i];
    }
}

// Compute relative error between numerical and analytical gradients.
fn relative_error(numerical: f64, analytical: f64) -> f64 {
    let numerator = (numerical - analytical).abs();
    let denominator = (numerical.abs() + analytical.abs()).max(1e-8);
    numerator / denominator
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ========================================================================
    // Gradient checking tests
    // ========================================================================

    #[test]
    fn test_gradient_checking_small_network_output_weights() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.3], vec![0.2, 0.7]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 2.0];
        let expected_outputs = vec![0.8];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for i in 0..2 {
            for j in 0..1 {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 1, i, j, epsilon);
                let analytical_grad = grad_w2[i][j];
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Output weight gradient mismatch at [{},{}]: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    i, j, numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_small_network_output_biases() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.3], vec![0.2, 0.7]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 1.0];
        let expected_outputs = vec![0.8];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for (i, &analytical_grad) in grad_b2.iter().enumerate().take(1) {
            let numerical_grad =
                numerical_gradient_bias(&nn, &inputs, &expected_outputs, 1, i, epsilon);
            let rel_error = relative_error(numerical_grad, analytical_grad);
            assert!(
                rel_error < 1e-5,
                "Output bias gradient mismatch at [{}]: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                i, numerical_grad, analytical_grad, rel_error
            );
        }
    }

    #[test]
    fn test_gradient_checking_small_network_hidden_weights() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.3], vec![0.2, 0.7]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 2.0];
        let expected_outputs = vec![0.8];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for i in 0..2 {
            for j in 0..2 {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 0, i, j, epsilon);
                let analytical_grad = grad_w1[i][j];
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Hidden weight gradient mismatch at [{},{}]: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    i, j, numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_small_network_hidden_biases() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.3], vec![0.2, 0.7]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 2.0];
        let expected_outputs = vec![0.8];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for i in 0..2 {
            let numerical_grad =
                numerical_gradient_bias(&nn, &inputs, &expected_outputs, 0, i, epsilon);
            let analytical_grad = grad_b1[i];
            let rel_error = relative_error(numerical_grad, analytical_grad);
            assert!(
                rel_error < 1e-5,
                "Hidden bias gradient mismatch at [{}]: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                i, numerical_grad, analytical_grad, rel_error
            );
        }
    }

    #[test]
    fn test_gradient_checking_zero_inputs() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 3,
                output_size: 2,
                weights: vec![vec![0.5, 0.3], vec![0.2, 0.7], vec![0.4, 0.6]],
                biases: vec![0.1, 0.2],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![0.0, 0.0, 0.0];
        let expected_outputs = vec![0.5];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 3];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for i in 0..2 {
            for j in 0..1 {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 1, i, j, epsilon);
                let analytical_grad = grad_w2[i][j];
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Gradient mismatch with zero inputs: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_negative_inputs() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 3,
                weights: vec![vec![0.5, 0.3, 0.2], vec![0.7, 0.4, 0.6]],
                biases: vec![0.1, 0.2, 0.3],
            },
            output_layer: LinearLayer {
                input_size: 3,
                output_size: 1,
                weights: vec![vec![0.4], vec![0.6], vec![0.8]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![-1.0, -2.0];
        let expected_outputs = vec![0.3];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 3]; 2];
        let mut grad_b1 = vec![0.0; 3];
        let mut grad_w2 = vec![vec![0.0; 1]; 3];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for i in 0..2 {
            for j in 0..3 {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 0, i, j, epsilon);
                let analytical_grad = grad_w1[i][j];
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Gradient mismatch with negative inputs: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_large_error() {
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

        let inputs = vec![1.0, 1.0];
        let expected_outputs = vec![0.99];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for (i, grad_w2_row) in grad_w2.iter().enumerate().take(2) {
            for (j, &analytical_grad) in grad_w2_row.iter().enumerate().take(1) {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 1, i, j, epsilon);
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Gradient mismatch with large error: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_multiple_outputs() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 3,
                output_size: 4,
                weights: vec![
                    vec![0.1, 0.2, 0.3, 0.4],
                    vec![0.5, 0.6, 0.7, 0.8],
                    vec![0.9, 1.0, 1.1, 1.2],
                ],
                biases: vec![0.1, 0.2, 0.3, 0.4],
            },
            output_layer: LinearLayer {
                input_size: 4,
                output_size: 2,
                weights: vec![
                    vec![0.2, 0.3],
                    vec![0.4, 0.5],
                    vec![0.6, 0.7],
                    vec![0.8, 0.9],
                ],
                biases: vec![0.1, 0.2],
            },
        };

        let inputs = vec![0.5, 1.5, 2.5];
        let expected_outputs = vec![0.7, 0.3];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 4]; 3];
        let mut grad_b1 = vec![0.0; 4];
        let mut grad_w2 = vec![vec![0.0; 2]; 4];
        let mut grad_b2 = vec![0.0; 2];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for (i, grad_w2_row) in grad_w2.iter().enumerate().take(4) {
            for (j, &analytical_grad) in grad_w2_row.iter().enumerate().take(2) {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 1, i, j, epsilon);
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Gradient mismatch with multiple outputs: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_symmetry() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                biases: vec![0.1, 0.1],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.5], vec![0.5]],
                biases: vec![0.1],
            },
        };

        let inputs = vec![1.0, 1.0];
        let expected_outputs = vec![0.8];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for (i, grad_w1_row) in grad_w1.iter().enumerate().take(2) {
            for (j, &analytical_grad) in grad_w1_row.iter().enumerate().take(2) {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 0, i, j, epsilon);
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(
                    rel_error < 1e-5,
                    "Gradient mismatch with symmetric network: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                    numerical_grad, analytical_grad, rel_error
                );
            }
        }
    }

    #[test]
    fn test_gradient_checking_deep_network() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 4,
                output_size: 6,
                weights: vec![
                    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                ],
                biases: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            },
            output_layer: LinearLayer {
                input_size: 6,
                output_size: 3,
                weights: vec![
                    vec![0.1, 0.2, 0.3],
                    vec![0.2, 0.3, 0.4],
                    vec![0.3, 0.4, 0.5],
                    vec![0.4, 0.5, 0.6],
                    vec![0.5, 0.6, 0.7],
                    vec![0.6, 0.7, 0.8],
                ],
                biases: vec![0.1, 0.2, 0.3],
            },
        };

        let inputs = vec![0.2, 0.4, 0.6, 0.8];
        let expected_outputs = vec![0.3, 0.5, 0.7];
        let epsilon = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 6]; 4];
        let mut grad_b1 = vec![0.0; 6];
        let mut grad_w2 = vec![vec![0.0; 3]; 6];
        let mut grad_b2 = vec![0.0; 3];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        let sample_indices = [(0, 0), (2, 3), (3, 5)];
        for (i, j) in sample_indices.iter() {
            let numerical_grad =
                numerical_gradient_weight(&nn, &inputs, &expected_outputs, 0, *i, *j, epsilon);
            let analytical_grad = grad_w1[*i][*j];
            let rel_error = relative_error(numerical_grad, analytical_grad);
            assert!(
                rel_error < 1e-5,
                "Gradient mismatch in deep network: numerical={:.10}, analytical={:.10}, rel_error={:.10}",
                numerical_grad, analytical_grad, rel_error
            );
        }
    }

    #[test]
    fn test_gradient_checking_all_parameters_sample() {
        let nn = NeuralNetwork {
            hidden_layer: LinearLayer {
                input_size: 2,
                output_size: 2,
                weights: vec![vec![0.3, 0.5], vec![0.7, 0.9]],
                biases: vec![0.2, 0.4],
            },
            output_layer: LinearLayer {
                input_size: 2,
                output_size: 1,
                weights: vec![vec![0.6], vec![0.8]],
                biases: vec![0.3],
            },
        };

        let inputs = vec![0.5, 1.5];
        let expected_outputs = vec![0.75];
        let epsilon = 1e-5;
        let max_rel_error = 1e-5;

        let mut grad_w1 = vec![vec![0.0; 2]; 2];
        let mut grad_b1 = vec![0.0; 2];
        let mut grad_w2 = vec![vec![0.0; 1]; 2];
        let mut grad_b2 = vec![0.0; 1];

        compute_analytical_gradients(
            &nn,
            &inputs,
            &expected_outputs,
            &mut grad_w1,
            &mut grad_b1,
            &mut grad_w2,
            &mut grad_b2,
        );

        for (i, grad_w1_row) in grad_w1.iter().enumerate().take(2) {
            for (j, &analytical_grad) in grad_w1_row.iter().enumerate().take(2) {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 0, i, j, epsilon);
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(rel_error < max_rel_error);
            }
        }

        for (i, &analytical_grad) in grad_b1.iter().enumerate().take(2) {
            let numerical_grad =
                numerical_gradient_bias(&nn, &inputs, &expected_outputs, 0, i, epsilon);
            let rel_error = relative_error(numerical_grad, analytical_grad);
            assert!(rel_error < max_rel_error);
        }

        for (i, grad_w2_row) in grad_w2.iter().enumerate().take(2) {
            for (j, &analytical_grad) in grad_w2_row.iter().enumerate().take(1) {
                let numerical_grad =
                    numerical_gradient_weight(&nn, &inputs, &expected_outputs, 1, i, j, epsilon);
                let rel_error = relative_error(numerical_grad, analytical_grad);
                assert!(rel_error < max_rel_error);
            }
        }

        for (i, &analytical_grad) in grad_b2.iter().enumerate().take(1) {
            let numerical_grad =
                numerical_gradient_bias(&nn, &inputs, &expected_outputs, 1, i, epsilon);
            let rel_error = relative_error(numerical_grad, analytical_grad);
            assert!(rel_error < max_rel_error);
        }
    }
}
