use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::SimpleRng;

// Small MLP to learn XOR (educational example).
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN: usize = 4;
const NUM_OUTPUTS: usize = 1;
const NUM_SAMPLES: usize = 4;
// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 1_000_000;

// Sigmoid activation function (f32 version for XOR).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Sigmoid derivative assuming x = sigmoid(z).
fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: DenseLayer,
    output_layer: DenseLayer,
}

// Create the full network with fixed XOR sizes.
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
    let hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

// Forward pass through a layer with sigmoid activation.
fn forward_with_sigmoid(layer: &DenseLayer, inputs: &[f32], outputs: &mut [f32]) {
    // Use the Layer trait's forward method (which does linear transformation).
    layer.forward(inputs, outputs, 1);
    // Apply sigmoid activation.
    for output in outputs.iter_mut() {
        *output = sigmoid(*output);
    }
}

// Backward pass and update for a single layer with sigmoid activation.
fn backward_and_update(
    layer: &mut DenseLayer,
    inputs: &[f32],
    outputs: &[f32],
    deltas: &[f32],
    grad_input: &mut [f32],
) {
    // Compute gradient through sigmoid activation.
    let mut grad_output = vec![0.0f32; deltas.len()];
    for (i, (&delta, &output)) in deltas.iter().zip(outputs.iter()).enumerate() {
        grad_output[i] = delta * sigmoid_derivative(output);
    }

    // Backward pass through the layer.
    layer.backward(inputs, &grad_output, grad_input, 1);

    // Update parameters.
    layer.update_parameters(LEARNING_RATE);
}

// Training with mean squared error per sample.
fn train(
    nn: &mut NeuralNetwork,
    inputs: &[[f32; NUM_INPUTS]],
    expected_outputs: &[[f32; NUM_OUTPUTS]],
) {
    // Buffers reused to avoid per-sample allocations.
    let mut grad_hidden = vec![0.0f32; NUM_INPUTS];

    for epoch in 0..EPOCHS {
        let mut total_errors = 0.0f32;

        for sample in 0..NUM_SAMPLES {
            let mut hidden_outputs = vec![0.0f32; NUM_HIDDEN];
            let mut output_outputs = vec![0.0f32; NUM_OUTPUTS];

            // Forward pass.
            forward_with_sigmoid(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
            forward_with_sigmoid(&nn.output_layer, &hidden_outputs, &mut output_outputs);

            // Compute error (expected - predicted).
            let mut errors = vec![0.0f32; NUM_OUTPUTS];
            for i in 0..NUM_OUTPUTS {
                errors[i] = expected_outputs[sample][i] - output_outputs[i];
                total_errors += errors[i] * errors[i];
            }

            // Backward pass and update for output layer.
            backward_and_update(
                &mut nn.output_layer,
                &hidden_outputs,
                &output_outputs,
                &errors,
                &mut hidden_outputs.clone(), // Temporary buffer (not used for input gradient in this case)
            );

            // Backward pass and update for hidden layer.
            // First, compute the gradient to backpropagate to hidden layer.
            let mut grad_hidden_outputs = vec![0.0f32; NUM_HIDDEN];
            let mut grad_output_for_hidden = vec![0.0f32; NUM_OUTPUTS];
            for (i, (&error, &output)) in errors.iter().zip(output_outputs.iter()).enumerate() {
                grad_output_for_hidden[i] = error * sigmoid_derivative(output);
            }
            nn.output_layer.backward(&hidden_outputs, &grad_output_for_hidden, &mut grad_hidden_outputs, 1);

            backward_and_update(
                &mut nn.hidden_layer,
                &inputs[sample],
                &hidden_outputs,
                &grad_hidden_outputs,
                &mut grad_hidden,
            );
        }

        // Average loss per epoch, printed every 1000 epochs.
        let loss = total_errors / NUM_SAMPLES as f32;
        if (epoch + 1) % 1000 == 0 {
            println!("Epoch {}, Error: {:.6}", epoch + 1, loss);
        }
    }
}

// Simple evaluation on XOR samples.
fn test(nn: &NeuralNetwork, inputs: &[[f32; NUM_INPUTS]], expected_outputs: &[[f32; NUM_OUTPUTS]]) {
    println!("\nTesting the trained network:");
    for sample in 0..NUM_SAMPLES {
        let mut hidden_outputs = vec![0.0f32; NUM_HIDDEN];
        let mut output_outputs = vec![0.0f32; NUM_OUTPUTS];

        // Forward pass to get the prediction.
        forward_with_sigmoid(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
        forward_with_sigmoid(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        println!(
            "Input: {:.1}, {:.1}, Expected Output: {:.1}, Predicted Output: {:.3}",
            inputs[sample][0], inputs[sample][1], expected_outputs[sample][0], output_outputs[0]
        );
    }
}

fn main() {
    // Fixed initial seed for partial reproducibility.
    let mut rng = SimpleRng::new(42);

    // XOR dataset (binary inputs and expected outputs).
    let inputs: [[f32; NUM_INPUTS]; NUM_SAMPLES] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected_outputs: [[f32; NUM_OUTPUTS]; NUM_SAMPLES] = [[0.0], [1.0], [1.0], [0.0]];

    // Training and testing in the same process.
    let mut nn = initialize_network(&mut rng);
    train(&mut nn, &inputs, &expected_outputs);
    test(&nn, &inputs, &expected_outputs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.9);
        assert!(sigmoid(-10.0) < 0.1);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sig_half = sigmoid(0.0);
        let deriv = sigmoid_derivative(sig_half);
        assert!((deriv - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_initialize_network() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(&mut rng);

        assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
        assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
    }

    #[test]
    fn test_forward_with_sigmoid() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(2, 1, &mut rng);
        let inputs = [0.5, 0.3];
        let mut outputs = vec![0.0f32; 1];

        forward_with_sigmoid(&layer, &inputs, &mut outputs);

        // Sigmoid output should be between 0 and 1.
        assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);
    }
}
