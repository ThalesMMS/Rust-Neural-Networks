//! Vanilla autoencoder with encoder-decoder architecture and MSE reconstruction loss
//!
//! This module provides a standard (deterministic) autoencoder that learns to
//! compress input data into a lower-dimensional latent representation and
//! reconstruct it with minimal loss.
//!
//! # Architecture
//!
//! The autoencoder consists of:
//! - **Encoder**: A stack of dense layers that compresses input to a latent representation.
//!   Hidden layers use ReLU activation; the final (latent) encoder layer is linear.
//! - **Decoder**: A stack of dense layers that reconstructs the input from the latent code.
//!   Hidden layers use ReLU activation; the final decoder layer uses Sigmoid to output
//!   values in [0, 1].
//!
//! # Training Objective
//!
//! The autoencoder minimizes the MSE reconstruction loss:
//! ```text
//! L = (1/N) * Σ (output_i - input_i)^2
//! ```
//! where N = batch_size × input_size.
//!
//! # Example
//!
//! ```ignore
//! use rust_neural_networks::autoencoder::vanilla::VanillaAutoencoder;
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! // Architecture: 784 -> 256 -> 64 -> 256 -> 784
//! let mut ae = VanillaAutoencoder::new(784, &[256], 64, &[256], &mut rng);
//!
//! let input = vec![0.5f32; 784];
//! let output = ae.forward(&input, 1);
//! let loss = ae.compute_loss(&output, &input);
//! ae.backward(&input, 1);
//! ae.update_parameters(0.001);
//! ```

use crate::layers::dense::DenseLayer;
use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::activations::relu_inplace;
use crate::utils::rng::SimpleRng;

/// Vanilla (deterministic) autoencoder with encoder-decoder architecture.
///
/// The encoder compresses input data into a compact latent representation using
/// a stack of dense layers with ReLU activations (linear for the latent layer).
/// The decoder reconstructs the input from the latent code, using ReLU for hidden
/// layers and Sigmoid for the output layer to produce values in [0, 1].
///
/// # Fields
///
/// * `encoder` - Stack of dense layers forming the encoder
/// * `decoder` - Stack of dense layers forming the decoder
/// * `input_size` - Dimensionality of the input (and output) data
/// * `latent_dim` - Dimensionality of the latent (bottleneck) representation
/// * `encoder_layer_sizes` - Full encoder architecture: [input_size, ..., latent_dim]
/// * `decoder_layer_sizes` - Full decoder architecture: [latent_dim, ..., input_size]
pub struct VanillaAutoencoder {
    encoder: Vec<DenseLayer>,
    decoder: Vec<DenseLayer>,
    input_size: usize,
    latent_dim: usize,
    encoder_layer_sizes: Vec<usize>,
    decoder_layer_sizes: Vec<usize>,
    // Cached inputs and post-activation outputs for each layer (used in backward pass)
    encoder_inputs: Vec<Vec<f32>>,
    encoder_post_acts: Vec<Vec<f32>>,
    decoder_inputs: Vec<Vec<f32>>,
    decoder_post_acts: Vec<Vec<f32>>,
}

impl VanillaAutoencoder {
    /// Creates a new vanilla autoencoder with the specified architecture.
    ///
    /// The encoder architecture is: `input_size -> encoder_sizes[0] -> ... -> latent_dim`
    /// The decoder architecture is: `latent_dim -> decoder_sizes[0] -> ... -> input_size`
    ///
    /// Encoder hidden layers use ReLU activation; the latent (last encoder) layer is linear.
    /// Decoder hidden layers use ReLU activation; the output (last decoder) layer uses Sigmoid.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Dimensionality of input and output data (e.g., 784 for MNIST)
    /// * `encoder_sizes` - Hidden layer sizes in the encoder (not including input or latent)
    /// * `latent_dim` - Dimensionality of the latent bottleneck representation
    /// * `decoder_sizes` - Hidden layer sizes in the decoder (not including latent or output)
    /// * `rng` - Random number generator for weight initialization (Xavier)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::autoencoder::vanilla::VanillaAutoencoder;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// // 784 -> 256 -> 64 -> 256 -> 784
    /// let ae = VanillaAutoencoder::new(784, &[256], 64, &[256], &mut rng);
    /// ```
    pub fn new(
        input_size: usize,
        encoder_sizes: &[usize],
        latent_dim: usize,
        decoder_sizes: &[usize],
        rng: &mut SimpleRng,
    ) -> Self {
        // Build encoder layers: input_size -> [encoder_sizes] -> latent_dim
        let mut encoder = Vec::new();
        let mut prev_size = input_size;
        for &size in encoder_sizes {
            encoder.push(DenseLayer::new(prev_size, size, rng));
            prev_size = size;
        }
        // Final encoder layer outputs the latent representation (linear, no activation)
        encoder.push(DenseLayer::new(prev_size, latent_dim, rng));

        // Build decoder layers: latent_dim -> [decoder_sizes] -> input_size
        let mut decoder = Vec::new();
        prev_size = latent_dim;
        for &size in decoder_sizes {
            decoder.push(DenseLayer::new(prev_size, size, rng));
            prev_size = size;
        }
        // Final decoder layer outputs reconstruction (Sigmoid activation applied externally)
        decoder.push(DenseLayer::new(prev_size, input_size, rng));

        let n_enc = encoder.len();
        let n_dec = decoder.len();

        // Store full layer size sequences for reference
        let mut encoder_layer_sizes = vec![input_size];
        for &s in encoder_sizes {
            encoder_layer_sizes.push(s);
        }
        encoder_layer_sizes.push(latent_dim);

        let mut decoder_layer_sizes = vec![latent_dim];
        for &s in decoder_sizes {
            decoder_layer_sizes.push(s);
        }
        decoder_layer_sizes.push(input_size);

        Self {
            encoder,
            decoder,
            input_size,
            latent_dim,
            encoder_layer_sizes,
            decoder_layer_sizes,
            encoder_inputs: vec![Vec::new(); n_enc],
            encoder_post_acts: vec![Vec::new(); n_enc],
            decoder_inputs: vec![Vec::new(); n_dec],
            decoder_post_acts: vec![Vec::new(); n_dec],
        }
    }

    /// Returns the input (and output) dimensionality.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the latent space dimensionality.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Returns the full encoder layer size sequence.
    ///
    /// The returned slice starts with `input_size` and ends with `latent_dim`,
    /// with any hidden layer sizes in between.
    pub fn encoder_layer_sizes(&self) -> &[usize] {
        &self.encoder_layer_sizes
    }

    /// Returns the full decoder layer size sequence.
    ///
    /// The returned slice starts with `latent_dim` and ends with `input_size`,
    /// with any hidden layer sizes in between.
    pub fn decoder_layer_sizes(&self) -> &[usize] {
        &self.decoder_layer_sizes
    }

    /// Returns the total number of trainable parameters in the autoencoder.
    ///
    /// Includes all weights and biases in both encoder and decoder layers.
    pub fn parameter_count(&self) -> usize {
        self.encoder
            .iter()
            .map(|l| l.parameter_count())
            .sum::<usize>()
            + self
                .decoder
                .iter()
                .map(|l| l.parameter_count())
                .sum::<usize>()
    }

    /// Returns a slice of the encoder's `DenseLayer` stack.
    ///
    /// Useful for serializing model weights or inspecting encoder structure.
    pub fn encoder_layers(&self) -> &[DenseLayer] {
        &self.encoder
    }

    /// Returns a slice of the decoder's `DenseLayer` stack.
    ///
    /// Useful for serializing model weights or inspecting decoder structure.
    pub fn decoder_layers(&self) -> &[DenseLayer] {
        &self.decoder
    }

    /// Encodes a batch of inputs to the latent representation.
    ///
    /// Passes the input through each encoder layer, applying ReLU to all hidden
    /// layers and leaving the final (latent) layer linear. Caches the input and
    /// post-activation output at each layer for use in the backward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened batch data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Latent representations, shape: batch_size × latent_dim
    pub fn encode(&mut self, input: &[f32], batch_size: usize) -> Vec<f32> {
        let n_enc = self.encoder.len();
        let mut current = input.to_vec();

        for i in 0..n_enc {
            // Cache the input to this encoder layer for backward pass
            self.encoder_inputs[i].clear();
            self.encoder_inputs[i].extend_from_slice(&current);

            // Linear transform
            let out_size = self.encoder[i].output_size() * batch_size;
            let mut output = vec![0.0f32; out_size];
            self.encoder[i].forward(&current, &mut output, batch_size);

            // Apply activation: ReLU for hidden layers, linear (no-op) for latent layer
            let is_last = i == n_enc - 1;
            if !is_last {
                relu_inplace(&mut output);
            }

            // Cache post-activation output for backward pass
            self.encoder_post_acts[i].clear();
            self.encoder_post_acts[i].extend_from_slice(&output);

            current = output;
        }

        current
    }

    /// Decodes a batch of latent codes to the reconstructed output.
    ///
    /// Passes the latent code through each decoder layer, applying ReLU to hidden
    /// layers and Sigmoid to the final output layer to produce values in (0, 1).
    /// Caches the input and post-activation output at each layer for the backward pass.
    ///
    /// # Arguments
    ///
    /// * `latent` - Latent representations, shape: batch_size × latent_dim
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Reconstructed data, shape: batch_size × input_size, values in (0, 1)
    pub fn decode(&mut self, latent: &[f32], batch_size: usize) -> Vec<f32> {
        let n_dec = self.decoder.len();
        let mut current = latent.to_vec();

        for i in 0..n_dec {
            // Cache the input to this decoder layer for backward pass
            self.decoder_inputs[i].clear();
            self.decoder_inputs[i].extend_from_slice(&current);

            // Linear transform
            let out_size = self.decoder[i].output_size() * batch_size;
            let mut output = vec![0.0f32; out_size];
            self.decoder[i].forward(&current, &mut output, batch_size);

            // Apply activation: ReLU for hidden layers, Sigmoid for output layer
            let is_last = i == n_dec - 1;
            if is_last {
                // Sigmoid activation: maps linear output to (0, 1) for pixel values
                for v in output.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            } else {
                relu_inplace(&mut output);
            }

            // Cache post-activation output for backward pass
            self.decoder_post_acts[i].clear();
            self.decoder_post_acts[i].extend_from_slice(&output);

            current = output;
        }

        current
    }

    /// Full forward pass through the autoencoder (encode then decode).
    ///
    /// Caches all intermediate activations required for the backward pass.
    /// The `target` for reconstruction is the original input (passed to `backward`).
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened batch data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Reconstructed data, shape: batch_size × input_size, values in (0, 1)
    pub fn forward(&mut self, input: &[f32], batch_size: usize) -> Vec<f32> {
        let latent = self.encode(input, batch_size);
        self.decode(&latent, batch_size)
    }

    /// Computes the mean squared error (MSE) reconstruction loss.
    ///
    /// ```text
    /// MSE = (1/N) * Σ (output_i - target_i)^2
    /// ```
    /// where N = output.len() = batch_size × input_size.
    ///
    /// # Arguments
    ///
    /// * `output` - Reconstructed data from forward pass
    /// * `target` - Original input data (ground truth)
    ///
    /// # Returns
    ///
    /// The mean squared error scalar.
    ///
    /// # Panics
    ///
    /// Panics if `output` and `target` have different lengths.
    pub fn compute_loss(&self, output: &[f32], target: &[f32]) -> f32 {
        assert_eq!(
            output.len(),
            target.len(),
            "VanillaAutoencoder compute_loss: output (len={}) and target (len={}) \
             must have the same length.",
            output.len(),
            target.len()
        );
        let n = output.len() as f32;
        output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| {
                let diff = o - t;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    /// Backward pass through the autoencoder computing MSE reconstruction gradients.
    ///
    /// Must be called after `forward()` with the same `batch_size`. The target for
    /// an autoencoder is the original input (since the goal is reconstruction).
    ///
    /// Gradient of MSE w.r.t. each output element:
    /// ```text
    /// d_loss/d_output_i = 2 * (output_i - target_i) / N
    /// ```
    /// where N = batch_size × input_size.
    ///
    /// This gradient is propagated back through the Sigmoid output, decoder layers,
    /// latent layer, and encoder layers, accumulating weight and bias gradients in
    /// each `DenseLayer` for subsequent parameter updates.
    ///
    /// # Arguments
    ///
    /// * `target` - Original input data (ground truth for MSE loss), shape: batch_size × input_size
    /// * `batch_size` - Number of samples (must match the `forward()` call)
    pub fn backward(&mut self, target: &[f32], batch_size: usize) {
        let n_enc = self.encoder.len();
        let n_dec = self.decoder.len();
        let n_elements = batch_size * self.input_size;

        // Compute MSE gradient w.r.t. decoder output (post-Sigmoid)
        // d_loss/d_output_i = 2 * (output_i - target_i) / N
        let scale = 2.0 / n_elements as f32;
        let last_post_act = self.decoder_post_acts[n_dec - 1].clone();
        let mut grad = vec![0.0f32; n_elements];
        for i in 0..n_elements {
            grad[i] = scale * (last_post_act[i] - target[i]);
        }

        // Backward through decoder layers
        // Take ownership of caches temporarily to avoid borrow checker conflicts
        // (we need to read decoder_inputs[i] and decoder_post_acts[i] while mutably
        // borrowing self.decoder[i] for the backward call).
        let dec_inputs = std::mem::take(&mut self.decoder_inputs);
        let dec_post_acts = std::mem::take(&mut self.decoder_post_acts);

        for i in (0..n_dec).rev() {
            let is_last = i == n_dec - 1;
            let prev_size = self.decoder[i].input_size() * batch_size;

            // Convert grad w.r.t. post-activation to grad w.r.t. linear pre-activation
            // by applying the activation's derivative element-wise.
            if is_last {
                // Sigmoid derivative: d_sigmoid/d_z = sigma * (1 - sigma)
                // where sigma is the cached post-activation value
                for j in 0..grad.len() {
                    let s = dec_post_acts[i][j];
                    grad[j] *= s * (1.0 - s);
                }
            } else {
                // ReLU derivative: 1 if post_act > 0, else 0
                for j in 0..grad.len() {
                    if dec_post_acts[i][j] <= 0.0 {
                        grad[j] = 0.0;
                    }
                }
            }

            // Backward through linear layer; accumulates weight/bias gradients internally
            let mut grad_prev = vec![0.0f32; prev_size];
            self.decoder[i].backward(&dec_inputs[i], &grad, &mut grad_prev, batch_size);

            grad = grad_prev;
        }

        // Restore decoder caches
        self.decoder_inputs = dec_inputs;
        self.decoder_post_acts = dec_post_acts;

        // grad is now the gradient w.r.t. the latent representation
        // Backward through encoder layers
        let enc_inputs = std::mem::take(&mut self.encoder_inputs);
        let enc_post_acts = std::mem::take(&mut self.encoder_post_acts);

        for i in (0..n_enc).rev() {
            let is_last = i == n_enc - 1;
            let prev_size = self.encoder[i].input_size() * batch_size;

            // Convert grad w.r.t. post-activation to grad w.r.t. linear pre-activation
            // Last encoder layer (latent) is linear: gradient passes through unchanged
            if !is_last {
                // ReLU derivative for hidden encoder layers
                for j in 0..grad.len() {
                    if enc_post_acts[i][j] <= 0.0 {
                        grad[j] = 0.0;
                    }
                }
            }

            // Backward through linear layer; accumulates weight/bias gradients internally
            let mut grad_prev = vec![0.0f32; prev_size];
            self.encoder[i].backward(&enc_inputs[i], &grad, &mut grad_prev, batch_size);

            // Propagate gradient to previous layer (stop at layer 0 input, not needed)
            if i > 0 {
                grad = grad_prev;
            }
        }

        // Restore encoder caches
        self.encoder_inputs = enc_inputs;
        self.encoder_post_acts = enc_post_acts;
    }

    /// Update all layer parameters using vanilla SGD.
    ///
    /// Applies accumulated gradients (from `backward()`) to encoder and decoder layers
    /// using the standard SGD rule: `param -= learning_rate * gradient`.
    /// Clears accumulated gradients after the update.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    pub fn update_parameters(&mut self, learning_rate: f32) {
        for layer in &mut self.encoder {
            layer.update_parameters(learning_rate);
        }
        for layer in &mut self.decoder {
            layer.update_parameters(learning_rate);
        }
    }

    /// Update all layer parameters using an optimizer (e.g., Adam, AdamW, RMSprop).
    ///
    /// Applies the optimizer's update rule to all encoder and decoder layers.
    /// Supports adaptive optimizers that maintain momentum and learning rate state.
    /// Should be called after `backward()`.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Mutable reference to an optimizer implementing the `Optimizer` trait
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::optimizers::Adam;
    ///
    /// let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// ae.forward(&input, batch_size);
    /// ae.backward(&input, batch_size);
    /// ae.update_with_optimizer(&mut optimizer);
    /// ```
    pub fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        for layer in &mut self.encoder {
            layer.update_with_optimizer(optimizer);
        }
        for layer in &mut self.decoder {
            layer.update_with_optimizer(optimizer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng::SimpleRng;

    #[test]
    fn test_vanilla_ae_construction() {
        let mut rng = SimpleRng::new(42);
        let ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        assert_eq!(ae.input_size(), 16);
        assert_eq!(ae.latent_dim(), 4);
        // Encoder: 16->8 + 8->4 = 2 layers
        assert_eq!(ae.encoder.len(), 2);
        // Decoder: 4->8 + 8->16 = 2 layers
        assert_eq!(ae.decoder.len(), 2);
    }

    #[test]
    fn test_vanilla_ae_no_hidden_layers() {
        let mut rng = SimpleRng::new(0);
        // Minimal AE: 10 -> 2 -> 10 (no hidden layers)
        let ae = VanillaAutoencoder::new(10, &[], 2, &[], &mut rng);
        assert_eq!(ae.encoder.len(), 1);
        assert_eq!(ae.decoder.len(), 1);
    }

    #[test]
    fn test_vanilla_ae_parameter_count() {
        let mut rng = SimpleRng::new(0);
        // 4 -> 3 -> 2 (encoder), 2 -> 3 -> 4 (decoder)
        let ae = VanillaAutoencoder::new(4, &[3], 2, &[3], &mut rng);
        // Encoder: layer0 = 4*3+3 = 15, layer1 = 3*2+2 = 8 → total 23
        // Decoder: layer0 = 2*3+3 = 9, layer1 = 3*4+4 = 16 → total 25
        let expected = (4 * 3 + 3) + (3 * 2 + 2) + (2 * 3 + 3) + (3 * 4 + 4);
        assert_eq!(ae.parameter_count(), expected);
    }

    #[test]
    fn test_vanilla_ae_forward_output_shape() {
        let mut rng = SimpleRng::new(1);
        let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let input = vec![0.5f32; 8 * 3]; // batch_size = 3
        let output = ae.forward(&input, 3);
        assert_eq!(output.len(), 8 * 3);
    }

    #[test]
    fn test_vanilla_ae_output_sigmoid_range() {
        let mut rng = SimpleRng::new(2);
        let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let input = vec![0.3f32; 8]; // batch_size = 1
        let output = ae.forward(&input, 1);
        for &v in &output {
            assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
        }
    }

    #[test]
    fn test_vanilla_ae_mse_loss_zero_on_perfect_reconstruction() {
        let output = vec![0.5f32; 4];
        let target = vec![0.5f32; 4];
        let mut rng = SimpleRng::new(0);
        let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
        let loss = ae.compute_loss(&output, &target);
        assert!(
            loss < 1e-6,
            "Loss should be ~0 for perfect reconstruction, got {}",
            loss
        );
    }

    #[test]
    fn test_vanilla_ae_mse_loss_nonzero() {
        let output = vec![0.0f32; 4];
        let target = vec![1.0f32; 4];
        let mut rng = SimpleRng::new(0);
        let ae = VanillaAutoencoder::new(4, &[], 2, &[], &mut rng);
        let loss = ae.compute_loss(&output, &target);
        // MSE = mean((0-1)^2) = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "Expected loss=1.0, got {}", loss);
    }

    #[test]
    fn test_vanilla_ae_backward_does_not_panic() {
        let mut rng = SimpleRng::new(42);
        let mut ae = VanillaAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let input = vec![0.5f32; 8];
        ae.forward(&input, 1);
        ae.backward(&input, 1); // Should not panic
    }

    #[test]
    fn test_vanilla_ae_encode_output_shape() {
        let mut rng = SimpleRng::new(10);
        let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        let input = vec![0.5f32; 16 * 2]; // batch_size = 2
        let latent = ae.encode(&input, 2);
        assert_eq!(latent.len(), 4 * 2);
    }

    #[test]
    fn test_vanilla_ae_decode_output_shape() {
        let mut rng = SimpleRng::new(11);
        let mut ae = VanillaAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        let latent = vec![0.1f32; 4 * 2]; // batch_size = 2
        let output = ae.decode(&latent, 2);
        assert_eq!(output.len(), 16 * 2);
    }
}
