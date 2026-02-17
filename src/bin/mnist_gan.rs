//! MNIST GAN – Generative Adversarial Network for MNIST digit generation.
//!
//! # Architecture
//!
//! ## Generator (noise → image)
//! - Input:  NOISE_DIM (100) latent noise vector, values in [-1, 1]
//! - Layer1: 100 → 256, LeakyReLU(α=0.2)
//! - Layer2: 256 → 512, LeakyReLU(α=0.2)
//! - Layer3: 512 → 784, Tanh
//! - Output: 784-dimensional image in [-1, 1]
//!
//! ## Discriminator (image → real/fake probability)
//! - Input:  784-dimensional image in [-1, 1]
//! - Layer1: 784 → 512, LeakyReLU(α=0.2)
//! - Layer2: 512 → 256, LeakyReLU(α=0.2)
//! - Layer3: 256 → 1,   Sigmoid
//! - Output: scalar probability that the image is real
//!
//! # Training
//!
//! The GAN uses the standard minimax loss with one-sided label smoothing.
//! Real MNIST images are rescaled from [0, 1] to [-1, 1] to match the
//! generator's Tanh output range.  The discriminator is trained first on
//! real images (target = 1.0 - label_smoothing) then on generated fakes
//! (target = 0.0).  The generator is then trained to fool the discriminator
//! (target = 1.0 for the generator step).
//!
//! See `docs/gan_tutorial.md` for a detailed explanation of GAN theory.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::optimizers::{Adam, Optimizer};
use rust_neural_networks::training::parse_config_path;
use rust_neural_networks::utils::rng::SimpleRng;

// BLAS back-end (mirrors dense.rs setup)
#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;
use cblas::{sgemm, Layout, Transpose};

// ============================================================================
// Architecture Constants
// ============================================================================

/// Dimension of the latent noise vector fed to the generator.
const NOISE_DIM: usize = 100;

/// Number of units in the first generator hidden layer.
const G_HIDDEN1: usize = 256;

/// Number of units in the second generator hidden layer.
const G_HIDDEN2: usize = 512;

/// Number of pixels in a flattened 28×28 MNIST image.
const IMG_SIZE: usize = 784;

/// Number of units in the first discriminator hidden layer.
const D_HIDDEN1: usize = 512;

/// Number of units in the second discriminator hidden layer.
const D_HIDDEN2: usize = 256;

/// Negative-slope coefficient for Leaky ReLU in both networks.
const LEAKY_RELU_ALPHA: f32 = 0.2;

/// Default configuration file path for GAN training.
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_gan_default.json";

// ============================================================================
// Utility Functions
// ============================================================================

/// Sigmoid activation: maps any real number to (0, 1).
///
/// σ(x) = 1 / (1 + exp(−x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid activation in-place to every element of `v`.
#[inline]
fn sigmoid_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = sigmoid(*x);
    }
}

/// Apply tanh activation in-place to every element of `v`.
#[inline]
fn tanh_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

/// Apply Leaky ReLU in-place: f(x) = x if x ≥ 0, else alpha * x.
#[inline]
fn leaky_relu_inplace(v: &mut [f32], alpha: f32) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x *= alpha;
        }
    }
}

/// Binary cross-entropy loss for a single (pred, target) pair.
///
/// L = −[target · log(pred + ε) + (1 − target) · log(1 − pred + ε)]
///
/// A small epsilon (1e-7) guards against log(0).
fn bce_loss(pred: f32, target: f32) -> f32 {
    -(target * (pred + 1e-7).ln() + (1.0 - target) * (1.0 - pred + 1e-7).ln())
}

// ============================================================================
// BLAS Helper
// ============================================================================

/// Compute C ← alpha · A · op(B) + beta · C using BLAS sgemm (row-major).
///
/// All matrices are in row-major order.
///
/// # Arguments
/// * `m`      – rows of C
/// * `n`      – cols of C
/// * `k`      – inner dimension
/// * `a`      – left operand (m × k if not transposed)
/// * `b`      – right operand; stored as (n × k) when `trans_b = Ordinary`
/// * `c`      – output matrix (m × n); overwritten when beta = 0.0
/// * `trans_b` – whether B should be transposed before multiplication
#[allow(clippy::too_many_arguments)]
fn sgemm_row(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    trans_b: bool,
) {
    let (trans_b_flag, ldb) = if trans_b {
        (Transpose::Ordinary, k as i32) // B stored n×k → after transpose k×n
    } else {
        (Transpose::None, n as i32) // B stored k×n
    };
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            trans_b_flag,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32, // lda = cols of A (k)
            b,
            ldb,
            beta,
            c,
            n as i32, // ldc = cols of C (n)
        );
    }
}

// ============================================================================
// Generator
// ============================================================================

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
struct Generator {
    /// Dense layer: NOISE_DIM → G_HIDDEN1
    layer1: DenseLayer,
    /// Dense layer: G_HIDDEN1 → G_HIDDEN2
    layer2: DenseLayer,
    /// Dense layer: G_HIDDEN2 → IMG_SIZE
    layer3: DenseLayer,
    /// Adam optimizer for `layer1`
    optimizer1: Box<dyn Optimizer>,
    /// Adam optimizer for `layer2`
    optimizer2: Box<dyn Optimizer>,
    /// Adam optimizer for `layer3`
    optimizer3: Box<dyn Optimizer>,
    /// Internal RNG used by [`generate_noise`]
    rng: SimpleRng,
}

impl Generator {
    /// Construct a Generator with Xavier-initialized weights.
    ///
    /// # Arguments
    /// * `rng`   – RNG used for weight initialisation (Xavier uniform)
    /// * `g_lr`  – Generator Adam learning rate
    /// * `beta1` – Adam first-moment decay (GAN convention: 0.5)
    /// * `beta2` – Adam second-moment decay (GAN convention: 0.999)
    fn new(rng: &mut SimpleRng, g_lr: f32, beta1: f32, beta2: f32) -> Self {
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

    /// Sample `batch_size × NOISE_DIM` noise values uniformly from [−1, 1].
    ///
    /// # Arguments
    /// * `batch_size` – Number of noise vectors to generate
    ///
    /// # Returns
    /// Flattened `Vec<f32>` of shape `(batch_size × NOISE_DIM)`.
    fn generate_noise(&mut self, batch_size: usize) -> Vec<f32> {
        let n = batch_size * NOISE_DIM;
        let mut noise = Vec::with_capacity(n);
        for _ in 0..n {
            noise.push(self.rng.next_f32() * 2.0 - 1.0);
        }
        noise
    }

    /// Forward pass: noise → synthetic images.
    ///
    /// # Arguments
    /// * `noise`      – Noise input, shape `(batch_size × NOISE_DIM)`
    /// * `batch_size` – Number of samples
    ///
    /// # Returns
    /// `(a1, a2, output)` where
    /// - `a1`:     post-LeakyReLU activations of `layer1` `(batch_size × G_HIDDEN1)`
    /// - `a2`:     post-LeakyReLU activations of `layer2` `(batch_size × G_HIDDEN2)`
    /// - `output`: Tanh-activated output (generated images) `(batch_size × IMG_SIZE)`
    fn forward(&self, noise: &[f32], batch_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn backward(
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

    /// Save generator weights and biases to a binary file.
    ///
    /// Format (little-endian):
    /// 1. Number of layers as `i32` (always 3)
    /// 2. For each layer: `input_size i32`, `output_size i32`, weights `f32[]`, biases `f32[]`
    fn save(&self, writer: &mut BufWriter<File>) {
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

// ============================================================================
// Discriminator
// ============================================================================

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
struct Discriminator {
    /// Dense layer: IMG_SIZE → D_HIDDEN1
    layer1: DenseLayer,
    /// Dense layer: D_HIDDEN1 → D_HIDDEN2
    layer2: DenseLayer,
    /// Dense layer: D_HIDDEN2 → 1
    layer3: DenseLayer,
    /// Adam optimizer for `layer1`
    optimizer1: Box<dyn Optimizer>,
    /// Adam optimizer for `layer2`
    optimizer2: Box<dyn Optimizer>,
    /// Adam optimizer for `layer3`
    optimizer3: Box<dyn Optimizer>,
}

impl Discriminator {
    /// Construct a Discriminator with Xavier-initialized weights.
    ///
    /// # Arguments
    /// * `rng`   – RNG for weight initialisation
    /// * `d_lr`  – Discriminator Adam learning rate
    /// * `beta1` – Adam first-moment decay (GAN convention: 0.5)
    /// * `beta2` – Adam second-moment decay (GAN convention: 0.999)
    fn new(rng: &mut SimpleRng, d_lr: f32, beta1: f32, beta2: f32) -> Self {
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

    /// Forward pass: images → real/fake probabilities.
    ///
    /// # Arguments
    /// * `images`     – Input images, shape `(batch_size × IMG_SIZE)`, values in [−1, 1]
    /// * `batch_size` – Number of samples
    ///
    /// # Returns
    /// `(a1, a2, output)` where
    /// - `a1`:     post-LeakyReLU activations of `layer1` `(batch_size × D_HIDDEN1)`
    /// - `a2`:     post-LeakyReLU activations of `layer2` `(batch_size × D_HIDDEN2)`
    /// - `output`: Sigmoid-activated probabilities `(batch_size)` – one scalar per image
    fn forward(&self, images: &[f32], batch_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn backward(
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

    /// Propagate gradient through D **without** updating D's parameters.
    ///
    /// Used during the generator training step: the adversarial gradient must
    /// flow from D's output back to G's output, but D must **not** be updated.
    ///
    /// Uses BLAS `sgemm` to compute C = grad · Wᵀ efficiently for each layer,
    /// bypassing the [`DenseLayer::backward`] API so that D's gradient
    /// accumulators are never modified.
    ///
    /// # Mathematical basis
    /// For a dense layer with weight matrix W (input_size × output_size):
    /// ```text
    /// grad_input = grad_output @ Wᵀ   (batch_size × input_size)
    /// ```
    ///
    /// # Arguments
    /// * `a1`         – Post-activation output of `layer1` from forward pass
    /// * `a2`         – Post-activation output of `layer2` from forward pass
    /// * `grad_logit` – `pred − target` per sample; shape `(batch_size)`
    /// * `batch_size` – Number of samples
    ///
    /// # Returns
    /// Gradient w.r.t. the discriminator's input (generated images),
    /// shape `(batch_size × IMG_SIZE)`.  Pass this to [`Generator::backward`].
    fn propagate_gradient(
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
    fn save(&self, writer: &mut BufWriter<File>) {
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

/// Compute the diversity metric for a batch of generated images.
///
/// Generates `n_samples` images, computes the mean pixel value for each,
/// then returns the standard deviation across samples.  Low std-dev
/// (< 0.02) indicates mode collapse.
///
/// # Arguments
/// * `gen`       – The generator network (mutable for noise generation)
/// * `n_samples` – Number of images to generate for the diversity estimate
///
/// # Returns
/// Standard deviation of per-sample mean pixel values.
fn compute_diversity(gen: &mut Generator, n_samples: usize) -> f32 {
    let noise = gen.generate_noise(n_samples);
    let (_, _, samples) = gen.forward(&noise, n_samples);

    // Mean pixel value per sample
    let means: Vec<f32> = (0..n_samples)
        .map(|i| {
            let start = i * IMG_SIZE;
            let end = start + IMG_SIZE;
            samples[start..end].iter().sum::<f32>() / IMG_SIZE as f32
        })
        .collect();

    // Std-dev across samples
    let mean_of_means = means.iter().sum::<f32>() / n_samples as f32;
    let variance = means
        .iter()
        .map(|&m| (m - mean_of_means) * (m - mean_of_means))
        .sum::<f32>()
        / n_samples as f32;
    variance.sqrt()
}

// ============================================================================
// GAN Training Loop
// ============================================================================

/// Train the GAN for the specified number of epochs.
///
/// Each epoch iterates over shuffled mini-batches of real MNIST images and:
/// 1. Trains D on real images (target = `label_smoothing_real`)
/// 2. Trains D on generated fake images (target = 0.0)
/// 3. Trains G to fool D (target = 1.0 for G's loss)
///
/// Logs per-epoch metrics to `logs/mnist_gan_log.csv` and detects mode
/// collapse via the sample diversity metric.
///
/// # Arguments
/// * `gen`               – Generator network
/// * `disc`              – Discriminator network
/// * `train_images`      – Flattened MNIST training images (values in [0, 1])
/// * `num_train`         – Number of training samples
/// * `rng`               – RNG for Fisher-Yates shuffle
/// * `epochs`            – Total number of training epochs
/// * `batch_size`        – Mini-batch size
/// * `label_smoothing`   – One-sided label smoothing (real target = 1.0 − smoothing)
/// * `noise_dim_check`   – Expected noise dimension (sanity check against `NOISE_DIM`)
#[allow(clippy::too_many_arguments)]
fn train_gan(
    gen: &mut Generator,
    disc: &mut Discriminator,
    train_images: &[f32],
    num_train: usize,
    rng: &mut SimpleRng,
    epochs: usize,
    batch_size: usize,
    label_smoothing: f32,
    noise_dim_check: usize,
) {
    assert_eq!(
        noise_dim_check, NOISE_DIM,
        "Config noise_dim {} does not match compiled NOISE_DIM {}",
        noise_dim_check, NOISE_DIM
    );

    std::fs::create_dir_all("./logs").ok();

    // Open GAN training log
    let log_file = File::create("./logs/mnist_gan_log.csv").unwrap_or_else(|_| {
        eprintln!("Could not open logs/mnist_gan_log.csv for writing");
        process::exit(1);
    });
    let mut log_writer = BufWriter::new(log_file);
    writeln!(
        log_writer,
        "epoch,g_loss,d_loss,d_real,d_fake,sample_diversity"
    )
    .unwrap_or_else(|_| {
        eprintln!("Failed writing CSV header");
        process::exit(1);
    });

    let real_target = 1.0 - label_smoothing; // one-sided label smoothing
    let fake_target_d = 0.0f32; // discriminator sees fakes as 0
    let real_target_g = 1.0f32; // generator wants D to output 1

    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut best_g_loss = f32::INFINITY;
    let mut best_epoch = 0usize;
    let mut mode_collapse_detected = false;

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut d_loss_sum = 0.0f32;
        let mut g_loss_sum = 0.0f32;
        let mut d_real_sum = 0.0f32;
        let mut d_fake_sum = 0.0f32;
        let mut batch_count = 0usize;

        // Fisher-Yates shuffle
        for i in (1..num_train).rev() {
            let j = rng.gen_usize(i + 1);
            indices.swap(i, j);
        }

        for batch_start in (0..num_train).step_by(batch_size) {
            let bs = (num_train - batch_start).min(batch_size);

            // ── Gather real images and rescale [0, 1] → [−1, 1] ──
            let mut real_imgs = vec![0.0f32; bs * IMG_SIZE];
            for (slot_idx, &sample_idx) in indices[batch_start..batch_start + bs].iter().enumerate()
            {
                let src = sample_idx * IMG_SIZE;
                let dst = slot_idx * IMG_SIZE;
                for k in 0..IMG_SIZE {
                    real_imgs[dst + k] = train_images[src + k] * 2.0 - 1.0;
                }
            }

            // ──────────────────────────────────────────────────────────────
            // Step 1: Train Discriminator on real images
            // ──────────────────────────────────────────────────────────────
            let (d_a1_real, d_a2_real, d_pred_real) = disc.forward(&real_imgs, bs);
            let mut grad_logit_real = vec![0.0f32; bs];
            let mut d_real_acc = 0.0f32;
            for i in 0..bs {
                grad_logit_real[i] = d_pred_real[i] - real_target; // pred - target
                d_real_acc += d_pred_real[i];
                d_loss_sum += bce_loss(d_pred_real[i], real_target);
            }
            disc.backward(&real_imgs, &d_a1_real, &d_a2_real, &grad_logit_real, bs);

            // ──────────────────────────────────────────────────────────────
            // Step 2: Train Discriminator on fake images
            // ──────────────────────────────────────────────────────────────
            let noise = gen.generate_noise(bs);
            let (g_a1, g_a2, fake_imgs) = gen.forward(&noise, bs);

            let (d_a1_fake, d_a2_fake, d_pred_fake) = disc.forward(&fake_imgs, bs);
            let mut grad_logit_fake = vec![0.0f32; bs];
            let mut d_fake_acc = 0.0f32;
            for i in 0..bs {
                grad_logit_fake[i] = d_pred_fake[i] - fake_target_d; // pred - 0 = pred
                d_fake_acc += d_pred_fake[i];
                d_loss_sum += bce_loss(d_pred_fake[i], fake_target_d);
            }
            disc.backward(&fake_imgs, &d_a1_fake, &d_a2_fake, &grad_logit_fake, bs);

            // ──────────────────────────────────────────────────────────────
            // Step 3: Train Generator (fool the discriminator)
            // ──────────────────────────────────────────────────────────────
            // Re-run D forward on the same fake images to get fresh activations
            // (D's parameters may have changed in steps 1 and 2).
            let noise2 = gen.generate_noise(bs);
            let (g_a1_new, g_a2_new, fake_imgs_new) = gen.forward(&noise2, bs);
            let (d_a1_g, d_a2_g, d_pred_g) = disc.forward(&fake_imgs_new, bs);

            // G's adversarial gradient: target=1.0 (G wants D to output 1)
            let mut grad_logit_g = vec![0.0f32; bs];
            for i in 0..bs {
                grad_logit_g[i] = d_pred_g[i] - real_target_g; // pred - 1 (negative → pushes D toward 1)
                g_loss_sum += bce_loss(d_pred_g[i], real_target_g);
            }

            // Propagate gradient through D (without updating D's parameters)
            let d_grad_input = disc.propagate_gradient(&d_a1_g, &d_a2_g, &grad_logit_g, bs);

            // Backpropagate through G and update G
            gen.backward(
                &noise2,
                &g_a1_new,
                &g_a2_new,
                &fake_imgs_new,
                &d_grad_input,
                bs,
            );

            d_real_sum += d_real_acc / bs as f32;
            d_fake_sum += d_fake_acc / bs as f32;
            batch_count += 1;

            // Suppress unused variable warnings for first D pass activations
            let _ = (&g_a1, &g_a2, &fake_imgs);
        }

        let n_batches = batch_count as f32;
        let avg_d_loss = d_loss_sum / (n_batches * 2.0); // averaged over real+fake
        let avg_g_loss = g_loss_sum / n_batches;
        let avg_d_real = d_real_sum / n_batches;
        let avg_d_fake = d_fake_sum / n_batches;
        let epoch_time = epoch_start.elapsed().as_secs_f32();

        // Mode collapse detection
        let diversity = compute_diversity(gen, 100);
        if diversity < 0.02 {
            mode_collapse_detected = true;
            println!(
                "WARNING: Mode collapse detected at epoch {} (diversity: {:.4})",
                epoch + 1,
                diversity
            );
        }

        println!(
            "Epoch {}/{}, G loss: {:.4}, D loss: {:.4}, D(real): {:.4}, D(fake): {:.4}, \
             Diversity: {:.4}, Time: {:.1}s",
            epoch + 1,
            epochs,
            avg_g_loss,
            avg_d_loss,
            avg_d_real,
            avg_d_fake,
            diversity,
            epoch_time
        );

        writeln!(
            log_writer,
            "{},{},{},{},{},{}",
            epoch + 1,
            avg_g_loss,
            avg_d_loss,
            avg_d_real,
            avg_d_fake,
            diversity
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing epoch log");
            process::exit(1);
        });

        // Save checkpoint on G loss improvement
        if avg_g_loss < best_g_loss {
            best_g_loss = avg_g_loss;
            best_epoch = epoch + 1;
            save_model(gen, disc, "mnist_gan_best.bin");
        }
    }

    // Export 16 generated samples to CSV
    export_samples(gen, 16, "./logs/mnist_gan_samples.csv");

    // Save final model
    save_model(gen, disc, "mnist_gan_final.bin");

    println!(
        "\nTraining complete. Best G loss: {:.4} at epoch {}. Mode collapse detected: {}",
        best_g_loss,
        best_epoch,
        if mode_collapse_detected { "Yes" } else { "No" }
    );
}

// ============================================================================
// Model I/O
// ============================================================================

/// Save generator and discriminator weights to a single binary file.
fn save_model(gen: &Generator, disc: &Discriminator, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open {} for writing", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);
    gen.save(&mut writer);
    disc.save(&mut writer);
    println!("Model saved to {}", filename);
}

/// Export `n` generated samples to a CSV file (one row = 784 pixel values).
///
/// Pixel values are in [−1, 1] (Tanh output range).
fn export_samples(gen: &mut Generator, n: usize, path: &str) {
    let noise = gen.generate_noise(n);
    let (_, _, samples) = gen.forward(&noise, n);

    let file = File::create(path).unwrap_or_else(|_| {
        eprintln!("Could not open {} for writing", path);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    for i in 0..n {
        let start = i * IMG_SIZE;
        let row: Vec<String> = samples[start..start + IMG_SIZE]
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect();
        writeln!(writer, "{}", row.join(",")).unwrap_or_else(|_| {
            eprintln!("Failed writing sample row");
            process::exit(1);
        });
    }
    println!("Generated samples saved to {}", path);
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    let program_start = Instant::now();
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    println!("=== MNIST GAN Training ===");
    println!("Loading configuration from: {}", config_path);

    // Extract hyperparameters; fall back to GAN defaults if config is unavailable.
    let (g_lr, d_lr, beta1, beta2, noise_dim, label_smoothing, epochs, batch_size) =
        match load_config(&config_path) {
            Ok(cfg) => {
                let lr = cfg.learning_rate.unwrap_or(0.0002);
                (
                    cfg.g_lr.unwrap_or(lr),
                    cfg.d_lr.unwrap_or(lr),
                    cfg.adam_beta1.unwrap_or(0.5),
                    cfg.adam_beta2.unwrap_or(0.999),
                    cfg.noise_dim.unwrap_or(NOISE_DIM),
                    cfg.label_smoothing.unwrap_or(0.1),
                    cfg.epochs.unwrap_or(50),
                    cfg.batch_size.unwrap_or(64),
                )
            }
            Err(e) => {
                eprintln!(
                    "Warning: could not load config from '{}': {}",
                    config_path, e
                );
                eprintln!("Using built-in GAN defaults.");
                (0.0002, 0.0002, 0.5, 0.999, NOISE_DIM, 0.1, 50, 64)
            }
        };

    println!(
        "G lr: {}, D lr: {}, beta1: {}, beta2: {}, noise_dim: {}",
        g_lr, d_lr, beta1, beta2, noise_dim
    );
    println!(
        "label_smoothing: {}, epochs: {}, batch_size: {}",
        label_smoothing, epochs, batch_size
    );

    // Load MNIST training data
    println!("Loading MNIST training data...");
    let load_start = Instant::now();
    let train_images = read_mnist_images("./data/train-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    // Labels are not used for GAN training but we load them for completeness
    let _train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    let num_train = train_images.len() / IMG_SIZE;
    println!(
        "Loaded {} training images in {:.2}s",
        num_train,
        load_start.elapsed().as_secs_f64()
    );

    // Initialise networks
    let mut rng = SimpleRng::new(42);
    let mut generator = Generator::new(&mut rng, g_lr, beta1, beta2);
    let mut discriminator = Discriminator::new(&mut rng, d_lr, beta1, beta2);

    println!(
        "Generator:     {} → {} → {} → {} ({} params)",
        NOISE_DIM,
        G_HIDDEN1,
        G_HIDDEN2,
        IMG_SIZE,
        generator.layer1.parameter_count()
            + generator.layer2.parameter_count()
            + generator.layer3.parameter_count()
    );
    println!(
        "Discriminator: {} → {} → {} → 1  ({} params)",
        IMG_SIZE,
        D_HIDDEN1,
        D_HIDDEN2,
        discriminator.layer1.parameter_count()
            + discriminator.layer2.parameter_count()
            + discriminator.layer3.parameter_count()
    );

    println!("Starting GAN training...");
    let mut shuffle_rng = SimpleRng::new(1337);
    train_gan(
        &mut generator,
        &mut discriminator,
        &train_images,
        num_train,
        &mut shuffle_rng,
        epochs,
        batch_size,
        label_smoothing,
        noise_dim,
    );

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Total program time: {:.2} seconds", total_time);
    println!("===========================");
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(-100.0) < 0.001);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_midpoint() {
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn test_bce_loss_near_zero_when_correct() {
        assert!(bce_loss(0.99, 1.0) < 0.05);
    }

    #[test]
    fn test_bce_loss_high_when_wrong() {
        assert!(bce_loss(0.01, 1.0) > 4.0);
    }

    #[test]
    fn test_generator_create() {
        let mut rng = SimpleRng::new(42);
        let gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
        assert_eq!(gen.layer1.input_size(), NOISE_DIM);
        assert_eq!(gen.layer1.output_size(), G_HIDDEN1);
        assert_eq!(gen.layer2.input_size(), G_HIDDEN1);
        assert_eq!(gen.layer2.output_size(), G_HIDDEN2);
        assert_eq!(gen.layer3.input_size(), G_HIDDEN2);
        assert_eq!(gen.layer3.output_size(), IMG_SIZE);
    }

    #[test]
    fn test_discriminator_create() {
        let mut rng = SimpleRng::new(42);
        let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
        assert_eq!(disc.layer1.input_size(), IMG_SIZE);
        assert_eq!(disc.layer1.output_size(), D_HIDDEN1);
        assert_eq!(disc.layer2.input_size(), D_HIDDEN1);
        assert_eq!(disc.layer2.output_size(), D_HIDDEN2);
        assert_eq!(disc.layer3.input_size(), D_HIDDEN2);
        assert_eq!(disc.layer3.output_size(), 1);
    }

    #[test]
    fn test_generator_noise_range() {
        let mut rng = SimpleRng::new(42);
        let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
        let noise = gen.generate_noise(10);
        assert_eq!(noise.len(), 10 * NOISE_DIM);
        for &v in &noise {
            assert!(
                (-1.0..=1.0).contains(&v),
                "Noise value {} outside [-1, 1]",
                v
            );
        }
    }

    #[test]
    fn test_generator_forward_shape() {
        let mut rng = SimpleRng::new(42);
        let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
        let batch_size = 4;
        let noise = gen.generate_noise(batch_size);
        let (a1, a2, output) = gen.forward(&noise, batch_size);
        assert_eq!(a1.len(), batch_size * G_HIDDEN1);
        assert_eq!(a2.len(), batch_size * G_HIDDEN2);
        assert_eq!(output.len(), batch_size * IMG_SIZE);
    }

    #[test]
    fn test_generator_output_in_tanh_range() {
        let mut rng = SimpleRng::new(42);
        let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
        let batch_size = 4;
        let noise = gen.generate_noise(batch_size);
        let (_, _, output) = gen.forward(&noise, batch_size);
        for &v in &output {
            assert!(
                v > -1.0 && v < 1.0,
                "Generator output {} outside (-1, 1)",
                v
            );
        }
    }

    #[test]
    fn test_discriminator_forward_shape() {
        let mut rng = SimpleRng::new(42);
        let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
        let batch_size = 4;
        let images = vec![0.0f32; batch_size * IMG_SIZE];
        let (a1, a2, output) = disc.forward(&images, batch_size);
        assert_eq!(a1.len(), batch_size * D_HIDDEN1);
        assert_eq!(a2.len(), batch_size * D_HIDDEN2);
        assert_eq!(output.len(), batch_size);
    }

    #[test]
    fn test_discriminator_output_in_sigmoid_range() {
        let mut rng = SimpleRng::new(42);
        let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
        let batch_size = 4;
        let images = vec![0.0f32; batch_size * IMG_SIZE];
        let (_, _, output) = disc.forward(&images, batch_size);
        for &v in &output {
            assert!(
                v > 0.0 && v < 1.0,
                "Discriminator output {} outside (0, 1)",
                v
            );
        }
    }
}
