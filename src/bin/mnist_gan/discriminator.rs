use super::*;

/// GAN Discriminator: classifies images as real (≈1) or generated (≈0).
///
/// # Forward pass
/// ```text
/// image (IMG_SIZE) → Layer1 + LeakyReLU → a1 (D_HIDDEN1)
///                  → Layer2 + LeakyReLU → a2 (D_HIDDEN2)
///                  → Layer3 + Sigmoid   → output (scalar)
/// ```
///
/// Each layer has its own Adam optimizer.
pub(crate) struct Discriminator {
    /// Dense layer: IMG_SIZE → D_HIDDEN1
    pub(crate) layer1: DenseLayer,
    /// Dense layer: D_HIDDEN1 → D_HIDDEN2
    pub(crate) layer2: DenseLayer,
    /// Dense layer: D_HIDDEN2 → 1
    pub(crate) layer3: DenseLayer,
    /// Adam optimizer for `layer1`
    pub(crate) optimizer1: Box<dyn Optimizer>,
    /// Adam optimizer for `layer2`
    pub(crate) optimizer2: Box<dyn Optimizer>,
    /// Adam optimizer for `layer3`
    pub(crate) optimizer3: Box<dyn Optimizer>,
}

impl Discriminator {
    /// Creates a new Discriminator with Xavier-initialized weights and per-layer Adam optimizers.
    ///
    /// The discriminator contains three fully connected layers (IMG_SIZE → D_HIDDEN1 → D_HIDDEN2 → 1).
    /// Each layer receives an independent Adam optimizer constructed with the provided `d_lr`, `beta1`,
    /// `beta2`, and epsilon = 1e-8.
    ///
    /// # Arguments
    ///
    /// * `rng` — random number generator used for Xavier weight initialization.
    /// * `d_lr` — learning rate for the Adam optimizers.
    /// * `beta1` — Adam first-moment decay (GAN convention: 0.5).
    /// * `beta2` — Adam second-moment decay (GAN convention: 0.999).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let mut rng = SimpleRng::seed(42);
    /// let disc = Discriminator::new(&mut rng, 2e-4, 0.5, 0.999);
    /// ```
    pub(crate) fn new(rng: &mut SimpleRng, d_lr: f32, beta1: f32, beta2: f32) -> Self {
        let layer1 = DenseLayer::new(IMG_SIZE, D_HIDDEN1, rng);
        let layer2 = DenseLayer::new(D_HIDDEN1, D_HIDDEN2, rng);
        let layer3 = DenseLayer::new(D_HIDDEN2, 1, rng);

        let optimizer1: Box<dyn Optimizer> = Box::new(Adam::new(d_lr, beta1, beta2, 1e-8));
        let optimizer2: Box<dyn Optimizer> = Box::new(Adam::new(d_lr, beta1, beta2, 1e-8));
        let optimizer3: Box<dyn Optimizer> = Box::new(Adam::new(d_lr, beta1, beta2, 1e-8));

        Discriminator {
            layer1,
            layer2,
            layer3,
            optimizer1,
            optimizer2,
            optimizer3,
        }
    }

    /// Computes discriminator outputs from input images.
    ///
    /// Processes flattened images through three dense layers with LeakyReLU after the first two
    /// layers and a sigmoid on the final logit, returning the two intermediate activations and
    /// the per-sample probability outputs.
    ///
    /// # Arguments
    ///
    /// * `images` — Flattened input images with shape `(batch_size × IMG_SIZE)`, values in [-1, 1].
    /// * `batch_size` — Number of samples in the batch.
    ///
    /// # Returns
    ///
    /// A tuple `(a1, a2, output)`:
    /// - `a1`: post-LeakyReLU activations from `layer1`, shape `(batch_size × D_HIDDEN1)`.
    /// - `a2`: post-LeakyReLU activations from `layer2`, shape `(batch_size × D_HIDDEN2)`.
    /// - `output`: sigmoid-activated probabilities, one `f32` per sample (`batch_size`).
    ///
    /// # Examples
    ///
    /// ```
    /// // Given an initialized `discriminator`, flattened `images` and `batch_size`:
    /// // let discriminator = Discriminator::new(...);
    /// // let images = vec![0.0f32; batch_size * IMG_SIZE];
    /// let (a1, a2, output) = discriminator.forward(&images, batch_size);
    /// assert_eq!(output.len(), batch_size);
    /// assert_eq!(a1.len(), batch_size * D_HIDDEN1);
    /// assert_eq!(a2.len(), batch_size * D_HIDDEN2);
    /// ```
    pub(crate) fn forward(
        &self,
        images: &[f32],
        batch_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Layer 1: IMG_SIZE → D_HIDDEN1 + LeakyReLU(0.2)
        let mut a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        self.layer1.forward(images, &mut a1, batch_size);
        leaky_relu_inplace(&mut a1, LEAKY_RELU_ALPHA);

        // Layer 2: D_HIDDEN1 → D_HIDDEN2 + LeakyReLU(0.2)
        let mut a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        self.layer2.forward(&a1, &mut a2, batch_size);
        leaky_relu_inplace(&mut a2, LEAKY_RELU_ALPHA);

        // Layer 3: D_HIDDEN2 → 1 + Sigmoid
        let mut output = vec![0.0f32; batch_size];
        self.layer3.forward(&a2, &mut output, batch_size);
        sigmoid_inplace(&mut output);

        (a1, a2, output)
    }

    /// Backward pass: updates D's parameters and returns grad w.r.t. input images.
    ///
    /// The caller passes `grad_logit` = `pred − target`, the combined gradient of the
    /// BCE loss plus the sigmoid non-linearity.  This is the gradient w.r.t. the
    /// **pre-sigmoid logit** computed as:
    ///
    /// ```text
    /// ∂L/∂logit = σ(logit) − target = pred − target
    /// ```
    ///
    /// This avoids redundant multiplication by the sigmoid derivative inside backward.
    ///
    /// After computing all gradients, the method immediately updates D's parameters
    /// using the per-layer Adam optimizers.
    ///
    /// # Arguments
    /// * `images`     – Input images from the forward pass
    /// * `a1`         – Post-activation output of `layer1` from forward pass
    /// * `a2`         – Post-activation output of `layer2` from forward pass
    /// * `grad_logit` – `pred − target` per sample; shape `(batch_size)`
    /// * `batch_size` – Number of samples
    ///
    /// # Returns
    /// Gradient w.r.t. the discriminator's input (images), shape `(batch_size × IMG_SIZE)`.
    pub(crate) fn backward(
        &mut self,
        images: &[f32],
        a1: &[f32],
        a2: &[f32],
        grad_logit: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        // ── Layer3 backward: D_HIDDEN2 → 1 ──
        // grad_logit is already ∂L/∂logit (combined BCE+sigmoid gradient).
        let mut d_a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        self.layer3.backward(a2, grad_logit, &mut d_a2, batch_size);

        // ── LeakyReLU' at layer2 output ──
        for i in 0..(batch_size * D_HIDDEN2) {
            if a2[i] <= 0.0 {
                d_a2[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer2 backward: D_HIDDEN1 → D_HIDDEN2 ──
        let mut d_a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        self.layer2.backward(a1, &d_a2, &mut d_a1, batch_size);

        // ── LeakyReLU' at layer1 output ──
        for i in 0..(batch_size * D_HIDDEN1) {
            if a1[i] <= 0.0 {
                d_a1[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer1 backward: IMG_SIZE → D_HIDDEN1 ──
        let mut d_images = vec![0.0f32; batch_size * IMG_SIZE];
        self.layer1
            .backward(images, &d_a1, &mut d_images, batch_size);

        // ── Update discriminator parameters ──
        self.layer3.update_with_optimizer(self.optimizer3.as_mut());
        self.layer2.update_with_optimizer(self.optimizer2.as_mut());
        self.layer1.update_with_optimizer(self.optimizer1.as_mut());

        d_images
    }

    /// Propagate the adversarial gradient through the discriminator without updating its parameters.
    ///
    /// Computes gradients w.r.t. the discriminator inputs by multiplying the incoming
    /// logit gradients with each layer's weight transpose (grad_input = grad_output @ Wᵀ),
    /// applying the LeakyReLU derivative at each hidden activation. Intended for use
    /// during generator training so the adversarial signal flows back into the generator
    /// while leaving discriminator parameter accumulators unchanged.
    ///
    /// # Arguments
    ///
    /// * `a1` — post-activation outputs from `layer1` (shape: batch_size × D_HIDDEN1)
    /// * `a2` — post-activation outputs from `layer2` (shape: batch_size × D_HIDDEN2)
    /// * `grad_logit` — per-sample gradient w.r.t. the pre-sigmoid logit (`pred - target`), shape (batch_size)
    /// * `batch_size` — number of samples in the batch
    ///
    /// # Returns
    ///
    /// A vector containing the gradient w.r.t. the discriminator input (generated images),
    /// with shape (batch_size × IMG_SIZE).
    ///
    /// # Examples
    ///
    /// ```
    /// // let d = Discriminator::new(...);
    /// // let (a1, a2, output) = d.forward(&images, batch_size);
    /// // let d_images = d.propagate_gradient(&a1, &a2, &grad_logit, batch_size);
    /// // assert_eq!(d_images.len(), batch_size * IMG_SIZE);
    /// ```
    pub(crate) fn propagate_gradient(
        &self,
        a1: &[f32],
        a2: &[f32],
        grad_logit: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        // ── Layer3: grad_logit (batch×1) × W3ᵀ (1×D_HIDDEN2) → d_a2 (batch×D_HIDDEN2) ──
        // W3 is stored as (D_HIDDEN2 × 1); we compute grad_logit @ W3ᵀ.
        // sgemm: m=batch_size, n=D_HIDDEN2, k=1, trans_b=true, B=W3 (D_HIDDEN2×1), ldb=1
        let w3 = self.layer3.weights(); // D_HIDDEN2 × 1
        let mut d_a2 = vec![0.0f32; batch_size * D_HIDDEN2];
        sgemm_row(
            batch_size, // m
            D_HIDDEN2,  // n
            1,          // k  (output_size of layer3 = 1)
            1.0, grad_logit, // A: (batch_size × 1)
            w3,         // B: (D_HIDDEN2 × 1) → Bᵀ is (1 × D_HIDDEN2)
            0.0, &mut d_a2,
            true, // trans_b: B is stored as D_HIDDEN2×1, we want its transpose
        );

        // ── LeakyReLU' at layer2 output ──
        for i in 0..(batch_size * D_HIDDEN2) {
            if a2[i] <= 0.0 {
                d_a2[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer2: d_a2 (batch×D_HIDDEN2) × W2ᵀ → d_a1 (batch×D_HIDDEN1) ──
        // W2 is stored as (D_HIDDEN1 × D_HIDDEN2).
        let w2 = self.layer2.weights(); // D_HIDDEN1 × D_HIDDEN2
        let mut d_a1 = vec![0.0f32; batch_size * D_HIDDEN1];
        sgemm_row(
            batch_size, // m
            D_HIDDEN1,  // n
            D_HIDDEN2,  // k
            1.0, &d_a2, // A: (batch_size × D_HIDDEN2)
            w2,    // B: (D_HIDDEN1 × D_HIDDEN2) → Bᵀ is (D_HIDDEN2 × D_HIDDEN1)
            0.0, &mut d_a1, true, // trans_b
        );

        // ── LeakyReLU' at layer1 output ──
        for i in 0..(batch_size * D_HIDDEN1) {
            if a1[i] <= 0.0 {
                d_a1[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // ── Layer1: d_a1 (batch×D_HIDDEN1) × W1ᵀ → d_images (batch×IMG_SIZE) ──
        // W1 is stored as (IMG_SIZE × D_HIDDEN1).
        let w1 = self.layer1.weights(); // IMG_SIZE × D_HIDDEN1
        let mut d_images = vec![0.0f32; batch_size * IMG_SIZE];
        sgemm_row(
            batch_size, // m
            IMG_SIZE,   // n
            D_HIDDEN1,  // k
            1.0,
            &d_a1, // A: (batch_size × D_HIDDEN1)
            w1,    // B: (IMG_SIZE × D_HIDDEN1) → Bᵀ is (D_HIDDEN1 × IMG_SIZE)
            0.0,
            &mut d_images,
            true, // trans_b
        );

        d_images
    }

    /// Save discriminator weights and biases to a binary file.
    ///
    /// Format matches the generator (3 layers, same encoding).
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

        write_i32(writer, 3); // discriminator has 3 layers

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

// ============================================================================
// Mode Collapse Detection
// ============================================================================
