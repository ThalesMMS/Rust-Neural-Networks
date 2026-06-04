//! Variational autoencoder with reparameterization trick and ELBO training objective
//!
//! This module provides a Variational Autoencoder (VAE) that learns a probabilistic
//! latent space by encoding inputs as distributions (mean and log variance) rather
//! than fixed point representations.
//!
//! # Architecture
//!
//! The VAE consists of:
//! - **Shared Encoder**: A stack of dense layers with ReLU activations that maps the
//!   input to a shared hidden representation.
//! - **Mu Head**: A linear layer mapping the shared encoder output to the mean vector `μ`
//!   of the latent Gaussian distribution.
//! - **Log-Variance Head**: A linear layer mapping the shared encoder output to the
//!   log-variance vector `log(σ²)` of the latent Gaussian distribution.
//! - **Decoder**: A stack of dense layers (ReLU for hidden, Sigmoid for output) that
//!   reconstructs the input from a sampled latent vector `z`.
//!
//! # Reparameterization Trick
//!
//! To allow gradients to flow through the sampling operation:
//! ```text
//! z = μ + ε · exp(0.5 · log(σ²))    where ε ~ N(0, I)
//! ```
//!
//! # Training Objective (ELBO)
//!
//! The VAE maximises the Evidence Lower BOund (ELBO):
//! ```text
//! ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z))
//! ```
//! which is equivalent to minimising:
//! ```text
//! L = reconstruction_loss + kl_weight · KL
//! ```
//! where:
//! - `reconstruction_loss` = MSE(reconstruction, input) averaged over all elements
//! - `KL = −0.5 · mean(1 + log_var − μ² − exp(log_var))` averaged over all (sample, latent) pairs
//! - `kl_weight` controls the trade-off (β-VAE style)
//!
//! # Example
//!
//! ```ignore
//! use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! // Architecture: 784 -> 256 -> (mu: 32, log_var: 32) -> z:32 -> 256 -> 784
//! let mut vae = VariationalAutoencoder::new(784, &[256], 32, &[256], &mut rng);
//!
//! let input = vec![0.5f32; 784];
//! let (reconstruction, mu, log_var) = vae.forward(&input, 1);
//! let loss = vae.compute_elbo_loss(&reconstruction, &input, &mu, &log_var, 1.0);
//! vae.backward(&input, 1, 1.0);
//! vae.update_parameters(0.001);
//! ```

use crate::layers::dense::DenseLayer;
use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::activations::relu_inplace;
use crate::utils::rng::SimpleRng;

/// Variational Autoencoder (VAE) with reparameterization trick and ELBO training.
///
/// The encoder maps inputs to distributions in latent space (μ and log σ²).
/// The decoder reconstructs inputs from samples drawn from these distributions.
/// During training, the reparameterization trick `z = μ + ε·σ` (ε ~ N(0,I)) makes
/// the sampling differentiable.
///
/// # Fields
///
/// * `encoder` - Shared encoder trunk (hidden layers, ReLU activation)
/// * `mu_layer` - Linear head producing the latent mean μ
/// * `log_var_layer` - Linear head producing the latent log-variance log(σ²)
/// * `decoder` - Decoder network (ReLU hidden layers, Sigmoid output)
/// * `input_size` - Dimensionality of the input/output data
/// * `latent_dim` - Dimensionality of the latent space
pub struct VariationalAutoencoder {
    // Shared encoder trunk (hidden layers only; mu and log_var are separate heads)
    encoder: Vec<DenseLayer>,
    // Two separate output heads from the trunk
    mu_layer: DenseLayer,
    log_var_layer: DenseLayer,
    // Decoder layers
    decoder: Vec<DenseLayer>,

    input_size: usize,
    latent_dim: usize,
    encoder_layer_sizes: Vec<usize>,
    decoder_layer_sizes: Vec<usize>,

    // Caches for backward pass (encoder trunk)
    encoder_inputs: Vec<Vec<f32>>,
    encoder_post_acts: Vec<Vec<f32>>,
    // Input to the mu and log_var heads (output of encoder trunk)
    trunk_output: Vec<f32>,

    // Cached latent variables
    cached_mu: Vec<f32>,
    cached_log_var: Vec<f32>,
    cached_eps: Vec<f32>, // Gaussian noise used in reparameterization
    cached_z: Vec<f32>,   // Sampled latent vector: z = mu + eps * exp(0.5 * log_var)

    // Caches for backward pass (decoder)
    decoder_inputs: Vec<Vec<f32>>,
    decoder_post_acts: Vec<Vec<f32>>,
}

impl VariationalAutoencoder {
    /// Creates a new Variational Autoencoder with the specified architecture.
    ///
    /// The shared encoder architecture is: `input_size -> encoder_sizes[0] -> ... -> hidden`
    /// The mu and log_var heads then map `hidden -> latent_dim`.
    /// The decoder architecture is: `latent_dim -> decoder_sizes[0] -> ... -> input_size`
    ///
    /// Encoder hidden layers use ReLU activation. The mu and log_var heads are linear.
    /// Decoder hidden layers use ReLU; the output layer uses Sigmoid to produce values in (0,1).
    ///
    /// # Arguments
    ///
    /// * `input_size` - Dimensionality of input and output data (e.g., 784 for MNIST)
    /// * `encoder_sizes` - Hidden layer sizes in the shared encoder trunk
    /// * `latent_dim` - Dimensionality of the latent space
    /// * `decoder_sizes` - Hidden layer sizes in the decoder (not including latent or output)
    /// * `rng` - Random number generator for weight initialization (Xavier)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// // 784 -> 256 -> (mu:32, log_var:32) -> z:32 -> 256 -> 784
    /// let vae = VariationalAutoencoder::new(784, &[256], 32, &[256], &mut rng);
    /// ```
    pub fn new(
        input_size: usize,
        encoder_sizes: &[usize],
        latent_dim: usize,
        decoder_sizes: &[usize],
        rng: &mut SimpleRng,
    ) -> Self {
        // Build shared encoder trunk: input_size -> [encoder_sizes]
        let mut encoder = Vec::new();
        let mut prev_size = input_size;
        for &size in encoder_sizes {
            encoder.push(DenseLayer::new(prev_size, size, rng));
            prev_size = size;
        }

        // Two separate linear heads for mu and log_var
        let trunk_out = prev_size;
        let mu_layer = DenseLayer::new(trunk_out, latent_dim, rng);
        let log_var_layer = DenseLayer::new(trunk_out, latent_dim, rng);

        // Build decoder: latent_dim -> [decoder_sizes] -> input_size
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
            mu_layer,
            log_var_layer,
            decoder,
            input_size,
            latent_dim,
            encoder_layer_sizes,
            decoder_layer_sizes,
            encoder_inputs: vec![Vec::new(); n_enc],
            encoder_post_acts: vec![Vec::new(); n_enc],
            trunk_output: Vec::new(),
            cached_mu: Vec::new(),
            cached_log_var: Vec::new(),
            cached_eps: Vec::new(),
            cached_z: Vec::new(),
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

    /// Returns the full encoder layer size sequence (including latent_dim as final entry).
    ///
    /// The returned slice starts with `input_size` and ends with `latent_dim`.
    pub fn encoder_layer_sizes(&self) -> &[usize] {
        &self.encoder_layer_sizes
    }

    /// Returns the full decoder layer size sequence.
    ///
    /// The returned slice starts with `latent_dim` and ends with `input_size`.
    pub fn decoder_layer_sizes(&self) -> &[usize] {
        &self.decoder_layer_sizes
    }

    /// Returns the total number of trainable parameters.
    ///
    /// Includes encoder trunk, mu head, log_var head, and decoder layers.
    pub fn parameter_count(&self) -> usize {
        let enc_params: usize = self.encoder.iter().map(|l| l.parameter_count()).sum();
        let dec_params: usize = self.decoder.iter().map(|l| l.parameter_count()).sum();
        enc_params
            + self.mu_layer.parameter_count()
            + self.log_var_layer.parameter_count()
            + dec_params
    }

    /// Encodes a batch of inputs through the shared encoder trunk and the two heads.
    ///
    /// Returns `(mu, log_var)` where both have shape `batch_size × latent_dim`.
    /// Caches intermediate activations needed for the backward pass.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened batch data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// `(mu, log_var)` both of shape batch_size × latent_dim
    pub fn encode(&mut self, input: &[f32], batch_size: usize) -> (Vec<f32>, Vec<f32>) {
        let n_enc = self.encoder.len();
        let mut current = input.to_vec();

        // Pass through shared encoder trunk with ReLU activations
        for i in 0..n_enc {
            self.encoder_inputs[i].clear();
            self.encoder_inputs[i].extend_from_slice(&current);

            let out_size = self.encoder[i].output_size() * batch_size;
            let mut output = vec![0.0f32; out_size];
            self.encoder[i].forward(&current, &mut output, batch_size);

            // All trunk layers use ReLU activation
            relu_inplace(&mut output);

            self.encoder_post_acts[i].clear();
            self.encoder_post_acts[i].extend_from_slice(&output);

            current = output;
        }

        // Cache trunk output (used as input to both mu and log_var heads)
        self.trunk_output = current.clone();

        // Compute mu (linear head - no activation)
        let latent_size = self.latent_dim * batch_size;
        let mut mu = vec![0.0f32; latent_size];
        self.mu_layer.forward(&current, &mut mu, batch_size);

        // Compute log_var (linear head - no activation)
        let mut log_var = vec![0.0f32; latent_size];
        self.log_var_layer
            .forward(&current, &mut log_var, batch_size);

        (mu, log_var)
    }

    /// Samples latent vectors using the reparameterization trick.
    ///
    /// Computes `z = μ + ε · exp(0.5 · log_var)` where `ε ~ N(0, I)`.
    /// Caches `ε` for use in the backward pass.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean vectors, shape: batch_size × latent_dim
    /// * `log_var` - Log-variance vectors, shape: batch_size × latent_dim
    /// * `rng` - Random number generator for sampling ε
    ///
    /// # Returns
    ///
    /// Sampled latent vectors `z`, shape: batch_size × latent_dim
    pub fn reparameterize(&mut self, mu: &[f32], log_var: &[f32], rng: &mut SimpleRng) -> Vec<f32> {
        let n = mu.len();
        let mut eps = vec![0.0f32; n];

        // Sample ε ~ N(0, I) using the Box-Muller transform.
        // Generates pairs (z0, z1) from uniform samples (u1, u2):
        //   z0 = sqrt(-2 * ln(u1)) * cos(2π * u2)
        //   z1 = sqrt(-2 * ln(u1)) * sin(2π * u2)
        let mut i = 0;
        while i < n {
            // Use a small offset to avoid ln(0)
            let u1 = rng.next_f32().max(1e-10);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            eps[i] = r * theta.cos();
            if i + 1 < n {
                eps[i + 1] = r * theta.sin();
            }
            i += 2;
        }

        let z: Vec<f32> = mu
            .iter()
            .zip(log_var.iter())
            .zip(eps.iter())
            .map(|((&m, &lv), &e)| m + e * (0.5 * lv).exp())
            .collect();

        self.cached_eps = eps;
        self.cached_z = z.clone();

        z
    }

    /// Decodes a batch of latent vectors to reconstructed outputs.
    ///
    /// Passes the latent code through each decoder layer, applying ReLU to hidden
    /// layers and Sigmoid to the final output layer. Caches intermediate activations
    /// for the backward pass.
    ///
    /// # Arguments
    ///
    /// * `z` - Sampled latent vectors, shape: batch_size × latent_dim
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Reconstructed data, shape: batch_size × input_size, values in (0, 1)
    pub fn decode(&mut self, z: &[f32], batch_size: usize) -> Vec<f32> {
        let n_dec = self.decoder.len();
        let mut current = z.to_vec();

        for i in 0..n_dec {
            self.decoder_inputs[i].clear();
            self.decoder_inputs[i].extend_from_slice(&current);

            let out_size = self.decoder[i].output_size() * batch_size;
            let mut output = vec![0.0f32; out_size];
            self.decoder[i].forward(&current, &mut output, batch_size);

            let is_last = i == n_dec - 1;
            if is_last {
                // Sigmoid activation: maps linear output to (0, 1) for pixel values
                for v in output.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            } else {
                relu_inplace(&mut output);
            }

            self.decoder_post_acts[i].clear();
            self.decoder_post_acts[i].extend_from_slice(&output);

            current = output;
        }

        current
    }

    /// Full forward pass: encode, reparameterize, decode.
    ///
    /// Caches all intermediate values needed for `backward()`.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened batch data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples in the batch
    /// * `rng` - Random number generator for sampling ε (reparameterization)
    ///
    /// # Returns
    ///
    /// `(reconstruction, mu, log_var)` where:
    /// - `reconstruction` has shape batch_size × input_size, values in (0, 1)
    /// - `mu` and `log_var` have shape batch_size × latent_dim
    pub fn forward(
        &mut self,
        input: &[f32],
        batch_size: usize,
        rng: &mut SimpleRng,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mu, log_var) = self.encode(input, batch_size);
        let z = self.reparameterize(&mu, &log_var, rng);

        // Cache mu and log_var for backward pass
        self.cached_mu = mu.clone();
        self.cached_log_var = log_var.clone();

        let reconstruction = self.decode(&z, batch_size);
        (reconstruction, mu, log_var)
    }

    /// Deterministic forward pass using μ directly (no sampling).
    ///
    /// Useful for inference: encodes the input and decodes using the mean latent
    /// vector (no reparameterization). Does **not** cache gradients.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened batch data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Reconstructed data, shape: batch_size × input_size, values in (0, 1)
    pub fn forward_mean(&mut self, input: &[f32], batch_size: usize) -> Vec<f32> {
        let (mu, _log_var) = self.encode(input, batch_size);
        self.decode(&mu, batch_size)
    }

    /// Computes the MSE reconstruction loss.
    ///
    /// ```text
    /// recon_loss = (1/N) * Σ (reconstruction_i - target_i)²
    /// ```
    /// where N = reconstruction.len() = batch_size × input_size.
    ///
    /// # Panics
    ///
    /// Panics if `reconstruction` and `target` have different lengths.
    pub fn compute_reconstruction_loss(&self, reconstruction: &[f32], target: &[f32]) -> f32 {
        assert_eq!(
            reconstruction.len(),
            target.len(),
            "VariationalAutoencoder compute_reconstruction_loss: reconstruction (len={}) and \
             target (len={}) must have the same length.",
            reconstruction.len(),
            target.len()
        );
        let n = reconstruction.len() as f32;
        reconstruction
            .iter()
            .zip(target.iter())
            .map(|(&r, &t)| {
                let diff = r - t;
                diff * diff
            })
            .sum::<f32>()
            / n
    }

    /// Computes the KL divergence between the approximate posterior q(z|x) and the prior p(z).
    ///
    /// Assumes diagonal Gaussian distributions:
    /// ```text
    /// KL = −0.5 · mean(1 + log_var − μ² − exp(log_var))
    /// ```
    /// averaged over all `(sample, latent_dim)` pairs.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean vectors, shape: batch_size × latent_dim
    /// * `log_var` - Log-variance vectors, shape: batch_size × latent_dim
    ///
    /// # Returns
    ///
    /// Non-negative scalar KL divergence value (per element, averaged).
    pub fn compute_kl_divergence(&self, mu: &[f32], log_var: &[f32]) -> f32 {
        assert_eq!(
            mu.len(),
            log_var.len(),
            "VariationalAutoencoder compute_kl_divergence: mu and log_var must have the same length."
        );
        let n = mu.len() as f32;
        mu.iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
            .sum::<f32>()
            / n
    }

    /// Computes the ELBO loss (reconstruction + KL term).
    ///
    /// ```text
    /// L = reconstruction_loss + kl_weight · KL
    /// ```
    ///
    /// # Arguments
    ///
    /// * `reconstruction` - Decoder output, shape: batch_size × input_size
    /// * `target` - Original input (ground truth), shape: batch_size × input_size
    /// * `mu` - Latent mean, shape: batch_size × latent_dim
    /// * `log_var` - Latent log-variance, shape: batch_size × latent_dim
    /// * `kl_weight` - Weight for the KL divergence term (β in β-VAE; typically 1.0)
    ///
    /// # Returns
    ///
    /// The scalar ELBO loss value.
    pub fn compute_elbo_loss(
        &self,
        reconstruction: &[f32],
        target: &[f32],
        mu: &[f32],
        log_var: &[f32],
        kl_weight: f32,
    ) -> f32 {
        let recon_loss = self.compute_reconstruction_loss(reconstruction, target);
        let kl = self.compute_kl_divergence(mu, log_var);
        recon_loss + kl_weight * kl
    }

    /// Backward pass through the full VAE computing ELBO gradients.
    ///
    /// Must be called after `forward()` with the same `batch_size` and `kl_weight`.
    /// Accumulates weight and bias gradients in all layer `DenseLayer`s for subsequent
    /// parameter updates.
    ///
    /// Gradient flow:
    /// 1. MSE gradient → backward through decoder (Sigmoid + dense layers) → grad_z
    /// 2. Reparameterization backward:
    ///    - `grad_mu = grad_z + kl_weight · d(KL)/d(μ)`
    ///    - `grad_log_var = grad_z ⊙ 0.5·(z−μ) + kl_weight · d(KL)/d(log_var)`
    /// 3. Backward through mu_layer and log_var_layer → accumulated into trunk gradient
    /// 4. Backward through shared encoder trunk
    ///
    /// # Arguments
    ///
    /// * `target` - Original input data, shape: batch_size × input_size
    /// * `batch_size` - Number of samples (must match the `forward()` call)
    /// * `kl_weight` - Weight for the KL divergence term (must match the loss used)
    pub fn backward(&mut self, target: &[f32], batch_size: usize, kl_weight: f32) {
        let n_enc = self.encoder.len();
        let n_dec = self.decoder.len();
        let n_recon = batch_size * self.input_size;
        let n_latent = batch_size * self.latent_dim;

        // --- Step 1: Gradient of MSE loss w.r.t. decoder output (post-Sigmoid) ---
        // d_loss/d_recon_i = 2 * (recon_i - target_i) / n_recon
        let recon_scale = 2.0 / n_recon as f32;
        let last_dec_post_act = self.decoder_post_acts[n_dec - 1].clone();
        let mut grad = vec![0.0f32; n_recon];
        for i in 0..n_recon {
            grad[i] = recon_scale * (last_dec_post_act[i] - target[i]);
        }

        // --- Step 2: Backward through decoder ---
        let dec_inputs = std::mem::take(&mut self.decoder_inputs);
        let dec_post_acts = std::mem::take(&mut self.decoder_post_acts);

        for i in (0..n_dec).rev() {
            let is_last = i == n_dec - 1;
            let prev_size = self.decoder[i].input_size() * batch_size;

            // Apply activation gradient
            if is_last {
                // Sigmoid derivative: sigma * (1 - sigma), where sigma = post_act
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

            let mut grad_prev = vec![0.0f32; prev_size];
            self.decoder[i].backward(&dec_inputs[i], &grad, &mut grad_prev, batch_size);
            grad = grad_prev;
        }

        self.decoder_inputs = dec_inputs;
        self.decoder_post_acts = dec_post_acts;

        // grad is now the gradient w.r.t. z (the sampled latent vector)
        let grad_z = grad;

        // --- Step 3: Reparameterization backward ---
        // z = mu + eps * exp(0.5 * log_var)
        // d(z_i)/d(mu_i) = 1  →  grad_mu_recon = grad_z
        // d(z_i)/d(log_var_i) = eps_i * 0.5 * exp(0.5 * log_var_i) = 0.5 * (z_i - mu_i)
        //
        // KL gradients (per element, divided by n_latent to match the normalizer):
        //   d(KL)/d(mu_i) = mu_i / n_latent
        //   d(KL)/d(log_var_i) = 0.5 * (exp(log_var_i) - 1.0) / n_latent

        let kl_scale = kl_weight / n_latent as f32;

        let mut grad_mu = vec![0.0f32; n_latent];
        let mut grad_log_var = vec![0.0f32; n_latent];

        for i in 0..n_latent {
            let m = self.cached_mu[i];
            let lv = self.cached_log_var[i];
            let z = self.cached_z[i];

            // Reconstruction gradient through reparameterization
            grad_mu[i] = grad_z[i];
            grad_log_var[i] = grad_z[i] * 0.5 * (z - m);

            // Add KL gradient
            grad_mu[i] += kl_scale * m;
            grad_log_var[i] += kl_scale * 0.5 * (lv.exp() - 1.0);
        }

        // --- Step 4: Backward through mu_layer and log_var_layer ---
        // Both heads receive gradient; their grad_trunk contributions are summed.
        let trunk_size = self.mu_layer.input_size() * batch_size;
        let trunk_output_cache = self.trunk_output.clone();

        let mut grad_trunk_from_mu = vec![0.0f32; trunk_size];
        self.mu_layer.backward(
            &trunk_output_cache,
            &grad_mu,
            &mut grad_trunk_from_mu,
            batch_size,
        );

        let mut grad_trunk_from_log_var = vec![0.0f32; trunk_size];
        self.log_var_layer.backward(
            &trunk_output_cache,
            &grad_log_var,
            &mut grad_trunk_from_log_var,
            batch_size,
        );

        // Sum the two trunk gradients
        let mut grad_trunk = vec![0.0f32; trunk_size];
        for i in 0..trunk_size {
            grad_trunk[i] = grad_trunk_from_mu[i] + grad_trunk_from_log_var[i];
        }

        // --- Step 5: Backward through shared encoder trunk ---
        if n_enc == 0 {
            // No shared encoder layers; nothing more to propagate
            return;
        }

        let enc_inputs = std::mem::take(&mut self.encoder_inputs);
        let enc_post_acts = std::mem::take(&mut self.encoder_post_acts);

        let mut grad = grad_trunk;

        for i in (0..n_enc).rev() {
            let prev_size = self.encoder[i].input_size() * batch_size;

            // All trunk layers use ReLU activation
            for j in 0..grad.len() {
                if enc_post_acts[i][j] <= 0.0 {
                    grad[j] = 0.0;
                }
            }

            let mut grad_prev = vec![0.0f32; prev_size];
            self.encoder[i].backward(&enc_inputs[i], &grad, &mut grad_prev, batch_size);

            if i > 0 {
                grad = grad_prev;
            }
        }

        self.encoder_inputs = enc_inputs;
        self.encoder_post_acts = enc_post_acts;
    }

    /// Returns L2 norms of accumulated weight and bias gradients for each trainable layer.
    ///
    /// Intended for training diagnostics after `backward()` and before an update clears
    /// the layer gradient accumulators.
    pub fn gradient_magnitudes(&self) -> Vec<(String, f32, f32)> {
        let mut gradients = Vec::with_capacity(self.encoder.len() + 2 + self.decoder.len());

        for (i, layer) in self.encoder.iter().enumerate() {
            let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
            gradients.push((format!("encoder_{}", i), weight_norm, bias_norm));
        }

        let (mu_weight_norm, mu_bias_norm) = self.mu_layer.get_gradient_magnitude();
        gradients.push(("mu".to_string(), mu_weight_norm, mu_bias_norm));

        let (log_var_weight_norm, log_var_bias_norm) = self.log_var_layer.get_gradient_magnitude();
        gradients.push((
            "log_var".to_string(),
            log_var_weight_norm,
            log_var_bias_norm,
        ));

        for (i, layer) in self.decoder.iter().enumerate() {
            let (weight_norm, bias_norm) = layer.get_gradient_magnitude();
            gradients.push((format!("decoder_{}", i), weight_norm, bias_norm));
        }

        gradients
    }

    /// Updates all layer parameters using vanilla SGD.
    ///
    /// Applies `param -= learning_rate * gradient` to the encoder trunk, mu head,
    /// log_var head, and decoder layers. Clears accumulated gradients after the update.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    pub fn update_parameters(&mut self, learning_rate: f32) {
        for layer in &mut self.encoder {
            layer.update_parameters(learning_rate);
        }
        self.mu_layer.update_parameters(learning_rate);
        self.log_var_layer.update_parameters(learning_rate);
        for layer in &mut self.decoder {
            layer.update_parameters(learning_rate);
        }
    }

    /// Updates all layer parameters using an optimizer (e.g., Adam, AdamW, RMSprop).
    ///
    /// Applies the optimizer's update rule to the encoder trunk, mu head, log_var head,
    /// and decoder layers. Should be called after `backward()`.
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
    /// let (recon, mu, log_var) = vae.forward(&input, batch_size, &mut rng);
    /// vae.backward(&input, batch_size, 1.0);
    /// vae.update_with_optimizer(&mut optimizer);
    /// ```
    pub fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        for layer in &mut self.encoder {
            layer.update_with_optimizer(optimizer);
        }
        self.mu_layer.update_with_optimizer(optimizer);
        self.log_var_layer.update_with_optimizer(optimizer);
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
    fn test_vae_construction() {
        let mut rng = SimpleRng::new(42);
        let vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        assert_eq!(vae.input_size(), 16);
        assert_eq!(vae.latent_dim(), 4);
        // Encoder trunk: 16->8 = 1 layer
        assert_eq!(vae.encoder.len(), 1);
        // Decoder: 4->8 + 8->16 = 2 layers
        assert_eq!(vae.decoder.len(), 2);
    }

    #[test]
    fn test_vae_no_hidden_layers() {
        let mut rng = SimpleRng::new(0);
        // Minimal VAE: 10 -> (mu:2, log_var:2) -> z:2 -> 10
        let vae = VariationalAutoencoder::new(10, &[], 2, &[], &mut rng);
        assert_eq!(vae.encoder.len(), 0);
        assert_eq!(vae.decoder.len(), 1);
    }

    #[test]
    fn test_vae_parameter_count() {
        let mut rng = SimpleRng::new(0);
        // 4 -> 3 (trunk), then (mu: 3->2, log_var: 3->2), then 2 -> 3 -> 4
        let vae = VariationalAutoencoder::new(4, &[3], 2, &[3], &mut rng);
        // Encoder trunk: 4*3+3 = 15
        // mu_layer: 3*2+2 = 8
        // log_var_layer: 3*2+2 = 8
        // Decoder layer0: 2*3+3 = 9
        // Decoder layer1: 3*4+4 = 16
        let expected = (4 * 3 + 3) + (3 * 2 + 2) + (3 * 2 + 2) + (2 * 3 + 3) + (3 * 4 + 4);
        assert_eq!(vae.parameter_count(), expected);
    }

    #[test]
    fn test_vae_forward_output_shape() {
        let mut rng = SimpleRng::new(1);
        let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let mut rng2 = SimpleRng::new(99);
        let input = vec![0.5f32; 8 * 3]; // batch_size = 3
        let (reconstruction, mu, log_var) = vae.forward(&input, 3, &mut rng2);
        assert_eq!(reconstruction.len(), 8 * 3);
        assert_eq!(mu.len(), 2 * 3);
        assert_eq!(log_var.len(), 2 * 3);
    }

    #[test]
    fn test_vae_output_sigmoid_range() {
        let mut rng = SimpleRng::new(2);
        let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let mut rng2 = SimpleRng::new(7);
        let input = vec![0.3f32; 8]; // batch_size = 1
        let (reconstruction, _, _) = vae.forward(&input, 1, &mut rng2);
        for &v in &reconstruction {
            assert!(v > 0.0 && v < 1.0, "Sigmoid output {} not in (0,1)", v);
        }
    }

    #[test]
    fn test_vae_kl_divergence_non_negative() {
        let mut rng = SimpleRng::new(0);
        let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
        let mu = vec![0.5f32, -0.3, 0.1, 0.0];
        let log_var = vec![-0.5f32, 0.2, -0.1, 0.0];
        let kl = vae.compute_kl_divergence(&mu, &log_var);
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {}", kl);
    }

    #[test]
    fn test_vae_kl_divergence_zero_at_prior() {
        let mut rng = SimpleRng::new(0);
        let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
        // KL(N(0,1) || N(0,1)) = 0
        let mu = vec![0.0f32; 4];
        let log_var = vec![0.0f32; 4]; // log_var=0 means var=1, std=1
        let kl = vae.compute_kl_divergence(&mu, &log_var);
        assert!(kl.abs() < 1e-5, "KL at prior should be ~0, got {}", kl);
    }

    #[test]
    fn test_vae_elbo_loss_components() {
        let mut rng = SimpleRng::new(0);
        let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
        let reconstruction = vec![0.5f32; 4];
        let target = vec![0.5f32; 4];
        let mu = vec![0.0f32; 2];
        let log_var = vec![0.0f32; 2];
        let elbo = vae.compute_elbo_loss(&reconstruction, &target, &mu, &log_var, 1.0);
        // recon_loss = 0 (perfect), kl = 0 (at prior) => ELBO loss = 0
        assert!(
            elbo.abs() < 1e-5,
            "ELBO at ideal point should be ~0, got {}",
            elbo
        );
    }

    #[test]
    fn test_vae_reconstruction_loss_matches_mse() {
        let mut rng = SimpleRng::new(0);
        let vae = VariationalAutoencoder::new(4, &[], 2, &[], &mut rng);
        let output = vec![0.0f32; 4];
        let target = vec![1.0f32; 4];
        let loss = vae.compute_reconstruction_loss(&output, &target);
        // MSE = mean((0-1)^2) = 1.0
        assert!((loss - 1.0).abs() < 1e-5, "Expected MSE=1.0, got {}", loss);
    }

    #[test]
    fn test_vae_backward_does_not_panic() {
        let mut rng = SimpleRng::new(42);
        let mut vae = VariationalAutoencoder::new(8, &[4], 2, &[4], &mut rng);
        let mut rng2 = SimpleRng::new(7);
        let input = vec![0.5f32; 8];
        vae.forward(&input, 1, &mut rng2);
        vae.backward(&input, 1, 1.0); // Should not panic
    }

    #[test]
    fn test_vae_encode_output_shapes() {
        let mut rng = SimpleRng::new(10);
        let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        let input = vec![0.5f32; 16 * 2]; // batch_size = 2
        let (mu, log_var) = vae.encode(&input, 2);
        assert_eq!(mu.len(), 4 * 2);
        assert_eq!(log_var.len(), 4 * 2);
    }

    #[test]
    fn test_vae_forward_mean_output_shape() {
        let mut rng = SimpleRng::new(11);
        let mut vae = VariationalAutoencoder::new(16, &[8], 4, &[8], &mut rng);
        let input = vec![0.5f32; 16 * 2]; // batch_size = 2
        let output = vae.forward_mean(&input, 2);
        assert_eq!(output.len(), 16 * 2);
    }

    #[test]
    fn test_vae_reparameterize_shape() {
        let mut rng = SimpleRng::new(5);
        let mut vae = VariationalAutoencoder::new(8, &[], 4, &[], &mut rng);
        let mut rng2 = SimpleRng::new(99);
        let mu = vec![0.0f32; 4 * 2]; // batch_size=2, latent_dim=4
        let log_var = vec![0.0f32; 4 * 2];
        let z = vae.reparameterize(&mu, &log_var, &mut rng2);
        assert_eq!(z.len(), 4 * 2);
    }
}
