use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};

// Small MLP to learn XOR (educational example).
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN: usize = 4;
const NUM_OUTPUTS: usize = 1;
const NUM_SAMPLES: usize = 4;
// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 1_000_000;

// ============================================================================
// Internal Abstractions (Inlined for self-contained binary)
// ============================================================================

/// Core trait for neural network layers.
pub trait Layer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    );
    fn update_parameters(&mut self, learning_rate: f32);
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
}

/// Simple random number generator.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    pub fn reseed_from_time(&mut self) {
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

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

/// Fully connected layer (manual implementation, no BLAS).
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
        let mut weights = vec![0.0; input_size * output_size];
        // Xavier initialization
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        for w in &mut weights {
            *w = rng.gen_range_f32(-limit, limit);
        }

        Self {
            input_size,
            output_size,
            weights,
            biases: vec![0.0; output_size],
            grad_weights: RefCell::new(vec![0.0; input_size * output_size]),
            grad_biases: RefCell::new(vec![0.0; output_size]),
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Simple implementation assuming batch_size=1 for this example
        // or loop over batches if needed. This binary seems to use batch_size=1.
        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;

            for j in 0..self.output_size {
                let mut sum = self.biases[j];
                for i in 0..self.input_size {
                    sum += input[in_offset + i] * self.weights[i * self.output_size + j];
                }
                output[out_offset + j] = sum;
            }
        }
    }

    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let scale = 1.0 / batch_size as f32;
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;

            for j in 0..self.output_size {
                let g = grad_output[out_offset + j];
                grad_b[j] += g * scale;

                for i in 0..self.input_size {
                    grad_w[i * self.output_size + j] += input[in_offset + i] * g * scale;
                }
            }

            // grad_input computation
            for i in 0..self.input_size {
                let mut sum = 0.0;
                for j in 0..self.output_size {
                    sum += grad_output[out_offset + j] * self.weights[i * self.output_size + j];
                }
                grad_input[in_offset + i] = sum * scale;
            }
        }
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
            *w -= learning_rate * g;
        }
        for (b, g) in self.biases.iter_mut().zip(grad_b.iter()) {
            *b -= learning_rate * g;
        }

        // Reset gradients
        for g in grad_w.iter_mut() {
            *g = 0.0;
        }
        for g in grad_b.iter_mut() {
            *g = 0.0;
        }
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.output_size
    }
}

// ============================================================================
// Main Logic
// ============================================================================

// Sigmoid activation function (f32 version for XOR).
/// Computes the logistic sigmoid of a 32-bit float.
///
/// The result is constrained between 0.0 and 1.0 and approaches 0.0 for large
/// negative inputs and 1.0 for large positive inputs.
///
/// # Returns
///
/// `f32` — the sigmoid(x), a value between 0.0 and 1.0.
///
/// # Examples
///
/// ```
/// let y = sigmoid(0.0);
/// assert!((y - 0.5).abs() < 1e-6);
/// let y_pos = sigmoid(10.0);
/// assert!(y_pos > 0.9);
/// let y_neg = sigmoid(-10.0);
/// assert!(y_neg < 0.1);
/// ```
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Sigmoid derivative assuming x = sigmoid(z).
/// Computes the derivative of the logistic sigmoid function given the sigmoid output.
///
/// This function expects `x` to be the value of the sigmoid activation (i.e., in the range 0.0 to 1.0)
/// and returns the derivative d/dz sigmoid(z) = x * (1 - x).
///
/// # Parameters
///
/// - `x`: The sigmoid output value for which to compute the derivative.
///
/// # Returns
///
/// The derivative of the sigmoid at the corresponding pre-activation value.
///
/// # Examples
///
/// ```
/// let s = 0.5_f32; // sigmoid(0.0)
/// let d = sigmoid_derivative(s);
/// assert!((d - 0.25).abs() < 1e-6);
/// ```
fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: DenseLayer,
    output_layer: DenseLayer,
}

// Create the full network with fixed XOR sizes.
/// Creates and initializes a two-layer neural network with random weights and biases.
///
/// The provided RNG is used to initialize the hidden and output DenseLayer parameters.
///
/// # Arguments
///
/// * `rng` - Mutable random number generator used to initialize layer weights and biases.
///
/// # Returns
///
/// A `NeuralNetwork` containing an initialized hidden layer (NUM_INPUTS → NUM_HIDDEN)
/// and output layer (NUM_HIDDEN → NUM_OUTPUTS).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let nn = initialize_network(&mut rng);
/// assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
/// assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
/// ```
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    let hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

// Forward pass through a layer with sigmoid activation.
/// Performs a forward pass through a dense layer and applies the sigmoid activation to each output.
///
/// # Examples
///
/// ```
/// use crate::{DenseLayer, SimpleRng, forward_with_sigmoid};
///
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(2, 1, &mut rng);
/// let inputs = [0.0f32, 1.0f32];
/// let mut outputs = [0.0f32];
///
/// forward_with_sigmoid(&layer, &inputs, &mut outputs);
/// assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);
/// ```
fn forward_with_sigmoid(layer: &DenseLayer, inputs: &[f32], outputs: &mut [f32]) {
    // Use the Layer trait's forward method (which does linear transformation).
    layer.forward(inputs, outputs, 1);
    // Apply sigmoid activation.
    for output in outputs.iter_mut() {
        *output = sigmoid(*output);
    }
}

// Training with mean squared error per sample.
/// Trains the neural network on the provided dataset using per-sample gradient updates.
///
/// Trains `nn` in-place for the configured number of epochs, performing a forward pass,
/// computing errors, backpropagating gradients, and updating layer parameters using the
/// module's learning rate. Progress is printed every 1000 epochs.
///
/// # Parameters
///
/// - `nn`: Mutable reference to the `NeuralNetwork` to train; its layers are updated in-place.
/// - `inputs`: Array of `NUM_SAMPLES` input vectors, each of length `NUM_INPUTS`.
/// - `expected_outputs`: Array of `NUM_SAMPLES` expected output vectors, each of length `NUM_OUTPUTS`.
///
/// # Examples
///
/// ```
/// # fn run() {
/// # use crate::{initialize_network, SimpleRng, train};
/// let mut rng = SimpleRng::new(42);
/// let mut nn = initialize_network(&mut rng);
/// let inputs = [ [0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0] ];
/// let expected = [ [0.0f32], [1.0], [1.0], [0.0] ];
/// train(&mut nn, &inputs, &expected);
/// # }
/// ```
fn train(
    nn: &mut NeuralNetwork,
    inputs: &[[f32; NUM_INPUTS]],
    expected_outputs: &[[f32; NUM_OUTPUTS]],
) {
    // Buffer pre-allocation
    let mut hidden_outputs = vec![0.0f32; NUM_HIDDEN];
    let mut output_outputs = vec![0.0f32; NUM_OUTPUTS];
    let mut errors = [0.0f32; NUM_OUTPUTS];
    let mut grad_output = vec![0.0f32; NUM_OUTPUTS];
    let mut grad_hidden_outputs = vec![0.0f32; NUM_HIDDEN];
    let mut grad_hidden_input = vec![0.0f32; NUM_INPUTS];

    for epoch in 0..EPOCHS {
        let mut total_errors = 0.0f32;

        for sample in 0..NUM_SAMPLES {
            // No need to clear buffers explicitly as they are fully overwritten:
            // - hidden_outputs and output_outputs are overwritten by forward()
            // - errors is computed element-wise
            // - grad_output is computed element-wise
            // - grad_hidden_outputs is overwritten by backward()
            // - grad_hidden_input is overwritten by backward()
            // Using explicit fill(0.0) just to be safe and match Reviewer request,
            // though not strictly necessary for correctness if logic is robust.
            hidden_outputs.fill(0.0);
            output_outputs.fill(0.0);
            errors.fill(0.0);
            grad_output.fill(0.0);
            grad_hidden_outputs.fill(0.0);
            grad_hidden_input.fill(0.0);

            // Forward pass.
            forward_with_sigmoid(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
            forward_with_sigmoid(&nn.output_layer, &hidden_outputs, &mut output_outputs);

            // Compute error (expected - predicted).
            for i in 0..NUM_OUTPUTS {
                errors[i] = expected_outputs[sample][i] - output_outputs[i];
                total_errors += errors[i] * errors[i];
            }

            // Backward pass (compute gradients but don't update yet).
            // 1. Compute gradient for output layer through sigmoid activation.
            // Note: DenseLayer uses weight -= lr * gradient convention, so we negate the error
            // (error = expected - output, gradient = output - expected for gradient descent)
            for (i, (&error, &output)) in errors.iter().zip(output_outputs.iter()).enumerate() {
                grad_output[i] = -error * sigmoid_derivative(output);
            }

            // 2. Backpropagate to hidden layer (BEFORE updating output layer weights).
            nn.output_layer
                .backward(&hidden_outputs, &grad_output, &mut grad_hidden_outputs, 1);

            // CRITICAL FIX: Apply sigmoid derivative for hidden layer activation
            // This applies the chain rule for the hidden layer's sigmoid activation
            for i in 0..NUM_HIDDEN {
                grad_hidden_outputs[i] *= sigmoid_derivative(hidden_outputs[i]);
            }

            // 3. Compute gradient for hidden layer.
            nn.hidden_layer.backward(
                &inputs[sample],
                &grad_hidden_outputs,
                &mut grad_hidden_input,
                1,
            );

            // 4. NOW update both layers (after all gradients are computed).
            nn.output_layer.update_parameters(LEARNING_RATE);
            nn.hidden_layer.update_parameters(LEARNING_RATE);
        }

        // Average loss per epoch, printed every 1000 epochs.
        let loss = total_errors / NUM_SAMPLES as f32;
        if (epoch + 1) % 1000 == 0 {
            println!("Epoch {}, Error: {:.6}", epoch + 1, loss);
        }
    }
}

// Simple evaluation on XOR samples.
/// Prints each input sample, its expected output, and the network's predicted output.
///
/// This performs a forward pass through the hidden and output layers using the sigmoid
/// activation and displays a formatted line per sample.
///
/// # Parameters
///
/// - `nn`: Reference to the neural network to evaluate.
/// - `inputs`: Array of input samples; each sample must have `NUM_INPUTS` elements.
/// - `expected_outputs`: Array of expected outputs; each sample must have `NUM_OUTPUTS` elements.
///
/// # Examples
///
/// ```
/// // Assuming `nn`, `inputs`, and `expected_outputs` are already defined:
/// // test(&nn, &inputs, &expected_outputs);
/// ```
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

/// Trains a small neural network on the XOR dataset and prints predictions for each input.
///
/// The program initializes the network with a fixed RNG seed for partial reproducibility,
/// trains it on the four classical XOR samples, and then prints each input alongside its
/// expected and predicted output.
///
/// # Examples
///
/// ```no_run
/// // Run the binary to train and evaluate the XOR network.
/// // Executing the program trains the network and displays test results.
/// crate::main();
/// ```
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
