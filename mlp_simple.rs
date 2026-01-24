use std::time::{SystemTime, UNIX_EPOCH};

// Small MLP to learn XOR (educational example).
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN: usize = 4;
const NUM_OUTPUTS: usize = 1;
const NUM_SAMPLES: usize = 4;
// Training hyperparameters.
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 1_000_000;

// Simple RNG to avoid external dependencies (not cryptographic).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    // Create RNG with explicit seed.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    // Reseed using the current time.
    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 {
            0x9e3779b97f4a7c15
        } else {
            nanos
        };
    }

    // Generate a pseudo-random u32 (xorshift).
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    // Convert u32 to [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    // Uniform sample in [low, high).
    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

// Sigmoid activation function.
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

// Initialize weights and biases with small random values.
fn initialize_layer(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> LinearLayer {
    let mut weights = vec![vec![0.0; output_size]; input_size];
    for i in 0..input_size {
        for j in 0..output_size {
            weights[i][j] = rng.gen_range_f64(-0.5, 0.5);
        }
    }

    let mut biases = vec![0.0; output_size];
    for bias in biases.iter_mut() {
        *bias = rng.gen_range_f64(-0.5, 0.5);
    }

    LinearLayer {
        input_size,
        output_size,
        weights,
        biases,
    }
}

// Create the full network with fixed XOR sizes.
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
    let hidden_layer = initialize_layer(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = initialize_layer(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

// Layer forward: z = W*x + b, followed by sigmoid.
fn forward_propagation(layer: &LinearLayer, inputs: &[f64], outputs: &mut [f64]) {
    for i in 0..layer.output_size {
        let mut activation = layer.biases[i];
        for j in 0..layer.input_size {
            activation += inputs[j] * layer.weights[j][i];
        }
        outputs[i] = sigmoid(activation);
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
    for i in 0..nn.output_layer.output_size {
        // delta_out = error * activation derivative.
        delta_output[i] = errors[i] * sigmoid_derivative(output_outputs[i]);
    }

    for i in 0..nn.hidden_layer.output_size {
        // Error backpropagated from output to hidden layer.
        let mut error = 0.0;
        for j in 0..nn.output_layer.output_size {
            error += delta_output[j] * nn.output_layer.weights[i][j];
        }
        delta_hidden[i] = error * sigmoid_derivative(hidden_outputs[i]);
    }
}

// Update weights and biases with gradient descent (SGD).
fn update_weights_biases(layer: &mut LinearLayer, inputs: &[f64], deltas: &[f64]) {
    for i in 0..layer.input_size {
        for j in 0..layer.output_size {
            layer.weights[i][j] += LEARNING_RATE * deltas[j] * inputs[i];
        }
    }

    for i in 0..layer.output_size {
        layer.biases[i] += LEARNING_RATE * deltas[i];
    }
}

// Training with mean squared error per sample.
fn train(
    nn: &mut NeuralNetwork,
    inputs: &[[f64; NUM_INPUTS]],
    expected_outputs: &[[f64; NUM_OUTPUTS]],
) {
    // Buffers reused to avoid per-sample allocations.
    let mut delta_hidden = vec![0.0; NUM_HIDDEN];
    let mut delta_output = vec![0.0; NUM_OUTPUTS];
    let mut errors = [0.0; NUM_OUTPUTS];

    for epoch in 0..EPOCHS {
        let mut total_errors = 0.0;
        for sample in 0..NUM_SAMPLES {
            let mut hidden_outputs = [0.0; NUM_HIDDEN];
            let mut output_outputs = [0.0; NUM_OUTPUTS];

            // Forward pass.
            forward_propagation(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
            forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

            // Per-output error and MSE accumulation.
            for i in 0..NUM_OUTPUTS {
                errors[i] = expected_outputs[sample][i] - output_outputs[i];
                total_errors += errors[i] * errors[i];
            }

            // Backprop and parameter updates.
            backward(
                nn,
                &inputs[sample],
                &hidden_outputs,
                &output_outputs,
                &errors,
                &mut delta_hidden,
                &mut delta_output,
            );

            update_weights_biases(&mut nn.output_layer, &hidden_outputs, &delta_output);
            update_weights_biases(&mut nn.hidden_layer, &inputs[sample], &delta_hidden);
        }

        // Average loss per epoch, printed every 1000 epochs.
        let loss = total_errors / NUM_SAMPLES as f64;
        if (epoch + 1) % 1000 == 0 {
            println!("Epoch {}, Error: {:.6}", epoch + 1, loss);
        }
    }
}

// Simple evaluation on XOR samples.
fn test(nn: &NeuralNetwork, inputs: &[[f64; NUM_INPUTS]], expected_outputs: &[[f64; NUM_OUTPUTS]]) {
    println!("\nTesting the trained network:");
    for sample in 0..NUM_SAMPLES {
        let mut hidden_outputs = [0.0; NUM_HIDDEN];
        let mut output_outputs = [0.0; NUM_OUTPUTS];

        // Forward pass to get the prediction.
        forward_propagation(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

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
    let inputs: [[f64; NUM_INPUTS]; NUM_SAMPLES] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected_outputs: [[f64; NUM_OUTPUTS]; NUM_SAMPLES] = [[0.0], [1.0], [1.0], [0.0]];

    // Training and testing in the same process.
    let mut nn = initialize_network(&mut rng);
    train(&mut nn, &inputs, &expected_outputs);
    test(&nn, &inputs, &expected_outputs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng_new() {
        let rng1 = SimpleRng::new(42);
        assert_eq!(rng1.state, 42);

        let rng2 = SimpleRng::new(0);
        assert_eq!(rng2.state, 0x9e3779b97f4a7c15);
    }

    #[test]
    fn test_simple_rng_reproducibility() {
        let mut rng1 = SimpleRng::new(123);
        let mut rng2 = SimpleRng::new(123);

        for _ in 0..10 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_simple_rng_next_f64() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let val = rng.next_f64();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_simple_rng_gen_range_f64() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let val = rng.gen_range_f64(-1.0, 1.0);
            assert!(val >= -1.0 && val < 1.0);
        }
    }

    #[test]
    fn test_simple_rng_reseed_from_time() {
        let mut rng = SimpleRng::new(42);
        let original_state = rng.state;
        rng.reseed_from_time();
        assert_ne!(rng.state, original_state);
    }

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
    fn test_initialize_layer() {
        let mut rng = SimpleRng::new(42);
        let layer = initialize_layer(3, 2, &mut rng);

        assert_eq!(layer.input_size, 3);
        assert_eq!(layer.output_size, 2);
        assert_eq!(layer.weights.len(), 3);
        assert_eq!(layer.weights[0].len(), 2);
        assert_eq!(layer.biases.len(), 2);
    }

    #[test]
    fn test_forward_propagation() {
        let mut rng = SimpleRng::new(42);
        let layer = initialize_layer(2, 1, &mut rng);
        let inputs = [0.5, 0.3];
        let mut outputs = [0.0];

        forward_propagation(&layer, &inputs, &mut outputs);

        assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);
    }

    #[test]
    fn test_backward() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(&mut rng);

        let inputs = [0.5, 0.3];
        let mut hidden_outputs = [0.0; NUM_HIDDEN];
        let mut output_outputs = [0.0; NUM_OUTPUTS];

        forward_propagation(&nn.hidden_layer, &inputs, &mut hidden_outputs);
        forward_propagation(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let errors = [0.5];
        let mut delta_hidden = [0.0; NUM_HIDDEN];
        let mut delta_output = [0.0; NUM_OUTPUTS];

        backward(
            &nn,
            &inputs,
            &hidden_outputs,
            &output_outputs,
            &errors,
            &mut delta_hidden,
            &mut delta_output,
        );

        assert_ne!(delta_output[0], 0.0);
    }

    #[test]
    fn test_update_weights_biases() {
        let mut rng = SimpleRng::new(42);
        let mut layer = initialize_layer(2, 1, &mut rng);

        let original_weight = layer.weights[0][0];
        let original_bias = layer.biases[0];

        let inputs = [0.5, 0.3];
        let deltas = [0.1];

        update_weights_biases(&mut layer, &inputs, &deltas);

        assert_ne!(layer.weights[0][0], original_weight);
        assert_ne!(layer.biases[0], original_bias);
    }

    #[test]
    fn test_initialize_network() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(&mut rng);

        assert_eq!(nn.hidden_layer.input_size, NUM_INPUTS);
        assert_eq!(nn.hidden_layer.output_size, NUM_HIDDEN);
        assert_eq!(nn.output_layer.input_size, NUM_HIDDEN);
        assert_eq!(nn.output_layer.output_size, NUM_OUTPUTS);
    }
}
