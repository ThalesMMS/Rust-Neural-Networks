use super::*;

/// GAN Generator: transforms random noise into synthetic digit images.
///
/// # Forward pass
/// ```text
/// noise (NOISE_DIM) → Layer1 + LeakyReLU → a1 (G_HIDDEN1)
///                   → Layer2 + LeakyReLU → a2 (G_HIDDEN2)
///                   → Layer3 + Tanh      → output (IMG_SIZE)
/// ```
///
/// The output is in [−1, 1] (tanh range) to match the rescaled real images.
///
/// Each layer has its own Adam optimizer so that the generator and
/// discriminator can use different learning rates.
pub(crate) struct Generator {
    /// Dense layer: NOISE_DIM → G_HIDDEN1
    pub(crate) layer1: DenseLayer,
    /// Dense layer: G_HIDDEN1 → G_HIDDEN2
    pub(crate) layer2: DenseLayer,
    /// Dense layer: G_HIDDEN2 → IMG_SIZE
    pub(crate) layer3: DenseLayer,
    /// Adam optimizer for `layer1`
    pub(crate) optimizer1: Box<dyn Optimizer>,
    /// Adam optimizer for `layer2`
    pub(crate) optimizer2: Box<dyn Optimizer>,
    /// Adam optimizer for `layer3`
    pub(crate) optimizer3: Box<dyn Optimizer>,
    /// Internal RNG used by [`generate_noise`]
    pub(crate) rng: SimpleRng,
}

impl Generator {
    /// Create a Generator with Xavier-uniform weight initialization and per-layer Adam optimizers.
    ///
    /// The provided `rng` is used only for initializing layer weights and biases (Xavier uniform).
    /// The generator also constructs three independent Adam optimizers using `g_lr`, `beta1`, and
    /// `beta2` with epsilon = 1e-8. A separate fixed-seed RNG (seed 7919) is allocated internally
    /// for noise sampling so noise generation does not affect weight initialization.
    ///
    /// # Arguments
    ///
    /// * `rng` — RNG used for weight initialization (Xavier uniform).
    /// * `g_lr` — Adam learning rate for all generator layers.
    /// * `beta1` — Adam first-moment decay (commonly 0.5 in GAN training).
    /// * `beta2` — Adam second-moment decay (commonly 0.999).
    ///
    /// # Returns
    ///
    /// A configured `Generator` with initialized layers, per-layer Adam optimizers, and an internal
    /// noise RNG.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut rng = SimpleRng::new(42);
    /// let _gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    /// ```
    pub(crate) fn new(rng: &mut SimpleRng, g_lr: f32, beta1: f32, beta2: f32) -> Self {
        let layer1 = DenseLayer::new(NOISE_DIM, G_HIDDEN1, rng);
        let layer2 = DenseLayer::new(G_HIDDEN1, G_HIDDEN2, rng);
        let layer3 = DenseLayer::new(G_HIDDEN2, IMG_SIZE, rng);

        let optimizer1: Box<dyn Optimizer> = Box::new(Adam::new(g_lr, beta1, beta2, 1e-8));
        let optimizer2: Box<dyn Optimizer> = Box::new(Adam::new(g_lr, beta1, beta2, 1e-8));
        let optimizer3: Box<dyn Optimizer> = Box::new(Adam::new(g_lr, beta1, beta2, 1e-8));

        // Separate RNG for noise so that weight-init seed is independent
        let noise_rng = SimpleRng::new(7919);

        Generator {
            layer1,
            layer2,
            layer3,
            optimizer1,
            optimizer2,
            optimizer3,
            rng: noise_rng,
        }
    }

    /// Sample `batch_size × NOISE_DIM` noise values uniformly from -1.0 to 1.0.
    ///
    /// # Arguments
    ///
    /// * `batch_size` — number of noise vectors to generate
    ///
    /// # Returns
    ///
    /// Flattened `Vec<f32>` of length `batch_size * NOISE_DIM` with values in `[-1.0, 1.0]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // assuming `gen` is a `Generator`
    /// let mut noise = gen.generate_noise(4);
    /// assert_eq!(noise.len(), 4 * NOISE_DIM);
    /// assert!(noise.iter().all(|&v| v >= -1.0 && v <= 1.0));
    /// ```
    pub(crate) fn generate_noise(&mut self, batch_size: usize) -> Vec<f32> {
        sample_noise(&mut self.rng, batch_size)
    }

    /// Produces generator activations and synthetic images from input noise.
    ///
    /// `noise` must be a flattened slice of length `batch_size * NOISE_DIM`. The
    /// method runs the noise through three dense layers with LeakyReLU on the first
    /// two layers and Tanh on the final layer.
    ///
    /// # Arguments
    ///
    /// * `noise` — Noise input, flattened as `(batch_size × NOISE_DIM)`.
    /// * `batch_size` — Number of samples in the batch.
    ///
    /// # Returns
    ///
    /// A tuple `(a1, a2, output)`:
    /// - `a1`: post-LeakyReLU activations from layer1 with shape `(batch_size × G_HIDDEN1)`.
    /// - `a2`: post-LeakyReLU activations from layer2 with shape `(batch_size × G_HIDDEN2)`.
    /// - `output`: Tanh-activated generated images with shape `(batch_size × IMG_SIZE)`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Given a configured `gen` and a noise buffer of the correct length:
    /// let (a1, a2, output) = gen.forward(&noise, batch_size);
    /// assert_eq!(a1.len(), batch_size * G_HIDDEN1);
    /// assert_eq!(a2.len(), batch_size * G_HIDDEN2);
    /// assert_eq!(output.len(), batch_size * IMG_SIZE);
    /// ```
    pub(crate) fn forward(
        &self,
        noise: &[f32],
        batch_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Layer 1: NOISE_DIM → G_HIDDEN1 + LeakyReLU(0.2)
        let mut a1 = vec![0.0f32; batch_size * G_HIDDEN1];
        self.layer1.forward(noise, &mut a1, batch_size);
        leaky_relu_inplace(&mut a1, LEAKY_RELU_ALPHA);

        // Layer 2: G_HIDDEN1 → G_HIDDEN2 + LeakyReLU(0.2)
        let mut a2 = vec![0.0f32; batch_size * G_HIDDEN2];
        self.layer2.forward(&a1, &mut a2, batch_size);
        leaky_relu_inplace(&mut a2, LEAKY_RELU_ALPHA);

        // Layer 3: G_HIDDEN2 → IMG_SIZE + Tanh
        let mut output = vec![0.0f32; batch_size * IMG_SIZE];
        self.layer3.forward(&a2, &mut output, batch_size);
        tanh_inplace(&mut output);

        (a1, a2, output)
    }

    /// Backward pass: backpropagate gradient through G and update G's parameters.
    ///
    /// The caller provides `grad_output`, the gradient of the adversarial loss
    /// with respect to the generator's Tanh output.  This is obtained from
    /// [`Discriminator::propagate_gradient`] during the generator training step.
    ///
    /// # Gradient flow
    /// ```text
    /// grad_output (∂L/∂tanh_out)
    ///   → * (1 − tanh_out²)          ← tanh derivative
    ///   → layer3.backward → d_a2
    ///   → * LeakyReLU'(a2)
    ///   → layer2.backward → d_a1
    ///   → * LeakyReLU'(a1)
    ///   → layer1.backward → d_noise (discarded)
    /// ```
    ///
    /// Parameter updates are applied to all three layers after the backward pass.
    ///
    /// # Arguments
    /// * `noise`      – Original noise from the corresponding forward pass
    /// * `a1`         – Post-activation output of `layer1` from forward pass
    /// * `a2`         – Post-activation output of `layer2` from forward pass
    /// * `gen_output` – Tanh output of `layer3` from forward pass
    /// * `grad_output` – `∂L/∂gen_output` from discriminator gradient propagation
    /// * `batch_size` – Number of samples in the batch
    pub(crate) fn backward(
        &mut self,
        noise: &[f32],
        a1: &[f32],
        a2: &[f32],
        gen_output: &[f32],
        grad_output: &[f32],
        batch_size: usize,
    ) {
        // ── Tanh derivative: d(tanh(x))/dx = 1 − tanh(x)² ──
        // gen_output is the post-tanh value, so 1 − output² is correct.
        let mut d_logit3 = vec![0.0f32; batch_size * IMG_SIZE];
        for i in 0..(batch_size * IMG_SIZE) {
            d_logit3[i] = grad_output[i] * (1.0 - gen_output[i] * gen_output[i]);
        }

        // ── Layer3 backward: G_HIDDEN2 → IMG_SIZE ──
        let mut d_a2 = vec![0.0f32; batch_size * G_HIDDEN2];
        self.layer3.backward(a2, &d_logit3, &mut d_a2, batch_size);

        // ── LeakyReLU' at layer2 output ──
        // a2 is the post-activation value; sign of post-activation ≡ sign of pre-activation
        // because LeakyReLU with α=0.2 > 0 preserves sign.
        for i in 0..(batch_size * G_HIDDEN2) {
            if a2[i] <= 0.0 {
                d_a2[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer2 backward: G_HIDDEN1 → G_HIDDEN2 ──
        let mut d_a1 = vec![0.0f32; batch_size * G_HIDDEN1];
        self.layer2.backward(a1, &d_a2, &mut d_a1, batch_size);

        // ── LeakyReLU' at layer1 output ──
        for i in 0..(batch_size * G_HIDDEN1) {
            if a1[i] <= 0.0 {
                d_a1[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer1 backward: NOISE_DIM → G_HIDDEN1 ──
        // The gradient w.r.t. noise (d_noise) is not used.
        let mut d_noise = vec![0.0f32; batch_size * NOISE_DIM];
        self.layer1.backward(noise, &d_a1, &mut d_noise, batch_size);

        // ── Update generator parameters ──
        self.layer3.update_with_optimizer(self.optimizer3.as_mut());
        self.layer2.update_with_optimizer(self.optimizer2.as_mut());
        self.layer1.update_with_optimizer(self.optimizer1.as_mut());
    }

    /// Write the generator's weights and biases to `writer` in a compact binary format.
    ///
    /// The file format (little-endian) is:
    /// 1) Number of layers as `i32` (always `3`).
    /// 2) For each layer, in order: `input_size: i32`, `output_size: i32`, all weights as `f32` (row-major flattened), then all biases as `f32`.
    ///
    /// On any I/O failure this function prints an error message to stderr and terminates the process.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::BufWriter;
    ///
    /// // Assume `gen` is an initialized `Generator`.
    /// // let mut gen = Generator::new(...);
    ///
    /// let file = File::create("generator.bin").unwrap();
    /// let mut writer = BufWriter::new(file);
    /// // gen.save(&mut writer);
    /// ```
    pub(crate) fn save(&self, writer: &mut BufWriter<File>) {
        let write_i32 = |w: &mut BufWriter<File>, v: i32| {
            w.write_all(&v.to_le_bytes()).unwrap_or_else(|_| {
                eprintln!("Failed writing model data");
                process::exit(1);
            })
        };
        let write_f32 = |w: &mut BufWriter<File>, v: f32| {
            w.write_all(&v.to_le_bytes()).unwrap_or_else(|_| {
                eprintln!("Failed writing model data");
                process::exit(1);
            })
        };

        write_i32(writer, 3); // generator has 3 layers

        for layer in [&self.layer1, &self.layer2, &self.layer3] {
            write_i32(writer, layer.input_size() as i32);
            write_i32(writer, layer.output_size() as i32);
            for &w in layer.weights() {
                write_f32(writer, w);
            }
            for &b in layer.biases() {
                write_f32(writer, b);
            }
        }
    }
}

pub(crate) fn sample_noise(rng: &mut SimpleRng, batch_size: usize) -> Vec<f32> {
    let n = batch_size * NOISE_DIM;
    let mut noise = Vec::with_capacity(n);
    for _ in 0..n {
        noise.push(rng.next_f32() * 2.0 - 1.0);
    }
    noise
}

// ============================================================================
// Discriminator
// ============================================================================
