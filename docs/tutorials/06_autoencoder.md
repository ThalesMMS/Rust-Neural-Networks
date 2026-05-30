# Tutorial 06: Autoencoders — Vanilla and Variational

**Level:** Advanced
**Time:** 90-120 minutes
**Prerequisites:** Tutorial 02 (MNIST MLP), understanding of MSE loss, basic probability (Gaussian distributions)
**Implementation:** See `src/autoencoder/` for complete working code

**Navigation:**
← [Tutorial 05: Automatic Differentiation Engine](05_autograd_engine.md) | [Tutorial Index](README.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is an Autoencoder?](#what-is-an-autoencoder)
3. [Architecture: Encoder-Bottleneck-Decoder](#architecture-encoder-bottleneck-decoder)
4. [Reconstruction Loss: MSE vs. BCE](#reconstruction-loss-mse-vs-bce)
5. [Vanilla Autoencoder — Implementation Walkthrough](#vanilla-autoencoder--implementation-walkthrough)
6. [Forward Pass Walkthrough](#forward-pass-walkthrough)
7. [Backward Pass Walkthrough](#backward-pass-walkthrough)
8. [Running the Vanilla Autoencoder](#running-the-vanilla-autoencoder)
9. [Latent Space Visualization](#latent-space-visualization)
10. [Variational Autoencoders — Theory](#variational-autoencoders--theory)
11. [The ELBO: Evidence Lower Bound](#the-elbo-evidence-lower-bound)
12. [KL Divergence Explained](#kl-divergence-explained)
13. [The Reparameterization Trick](#the-reparameterization-trick)
14. [VAE Architecture: Two Encoder Heads](#vae-architecture-two-encoder-heads)
15. [VAE Implementation Walkthrough](#vae-implementation-walkthrough)
16. [Running the VAE](#running-the-vae)
17. [Vanilla AE vs. VAE: Key Differences](#vanilla-ae-vs-vae-key-differences)
18. [Verification Checkpoints](#verification-checkpoints)
19. [Exercises](#exercises)
20. [Next Steps](#next-steps)

---

## Introduction

All previous tutorials taught **supervised learning**: given a labelled example (image + digit label), update the network to predict that label. Autoencoders introduce something fundamentally different — **unsupervised representation learning**. The only training signal is the data itself. No labels required.

Why is this powerful? Because to reconstruct an image faithfully from a compressed bottleneck, the network *must* learn which features matter. A well-trained autoencoder discovers internal structure in the data — not because we told it to, but because structure is what compresses.

By the end of this tutorial you will understand:
- How the encoder-bottleneck-decoder architecture forces compression
- Why MSE is used as the reconstruction loss for normalised pixel values
- How gradients flow through the full encoder-decoder stack
- What the VAE's probabilistic latent space adds over a plain autoencoder
- The three-part ELBO objective: reconstruction + KL divergence
- Why the reparameterization trick is necessary for backpropagation
- How to run both models and visualise their latent spaces

**Implementation reference:**
- `src/autoencoder/vanilla.rs` — `VanillaAutoencoder`
- `src/autoencoder/vae.rs` — `VariationalAutoencoder`
- `src/bin/mnist_autoencoder.rs` — Vanilla AE training binary
- `src/bin/mnist_vae.rs` — VAE training binary

---

## What is an Autoencoder?

An autoencoder is a neural network trained to output a copy of its input:

```
Input x  →  [Encoder]  →  Latent code z  →  [Decoder]  →  Reconstruction x̂
```

The trick is that the latent code `z` has far fewer dimensions than the input `x`. For MNIST, we compress 784 pixels down to 64 numbers — a **12× compression ratio**.

Because the bottleneck is so tight, the network cannot simply memorise inputs. Instead, it must learn a compact encoding that retains the most important information. This forces the network to discover structure: that all 7s share similar stroke patterns, that 1s and 7s differ in a specific direction of latent space, etc.

### What Autoencoders Are Used For

| Application | How |
|-------------|-----|
| **Dimensionality reduction** | Use encoder output `z` as compact features |
| **Anomaly detection** | High reconstruction error signals unusual inputs |
| **Data denoising** | Train on noisy inputs, reconstruct clean targets |
| **Generative modelling** | Sample in latent space, decode new images (VAE) |
| **Pre-training** | Initialise layers with encoder weights before fine-tuning |

### Unsupervised = No Labels Needed

The training target is the input itself. The MNIST dataset has 60,000 labelled images, but for autoencoder training those labels are entirely ignored. The model learns purely from pixel patterns. Labels are only needed *later* if you want to evaluate what the latent space has learned (e.g., for a 2-D scatter plot where each point is coloured by digit class).

---

## Architecture: Encoder-Bottleneck-Decoder

### Vanilla Autoencoder Architecture for MNIST

```
Input Layer          Encoder            Latent         Decoder           Output Layer
  (784)               (256)              (64)           (256)               (784)

 x₁ ────┐                                                              x̂₁
 x₂ ────┤       ┌──[256 ReLU]──┐    ┌──[64 linear]──┐   ┌──[256 ReLU]──┐  x̂₂
  ...   ├──────►│               ├──►│                ├──►│               ├► ...
x₇₈₄ ──┘       └───────────────┘   └────────────────┘   └───────────────┘  x̂₇₈₄
                                            │                     │
                                     Bottleneck           Sigmoid output
                                    (compression)         (pixels in [0,1])
```

**Architecture in numbers:**
- **Input:** 784 (flattened 28×28 grayscale image)
- **Encoder hidden:** 256 neurons, ReLU activation
- **Latent layer:** 64 neurons, **linear** (no activation)
- **Decoder hidden:** 256 neurons, ReLU activation
- **Output layer:** 784 neurons, **Sigmoid** activation (outputs in (0, 1))

**Parameter count:**

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Encoder: 784 → 256 | 200,704 | 256 | 200,960 |
| Encoder: 256 → 64 | 16,384 | 64 | 16,448 |
| Decoder: 64 → 256 | 16,384 | 256 | 16,640 |
| Decoder: 256 → 784 | 200,704 | 784 | 201,488 |
| **Total** | | | **435,536** |

### Why Is the Latent Layer Linear?

Hidden layers use ReLU because non-linearity is needed to express complex functions. But the **final encoder layer** (which produces `z`) is intentionally **linear** — no activation function applied.

Why? The latent code `z` needs to be a free, real-valued representation. Applying ReLU would clip negative values to zero, destroying half the representational capacity. Applying Sigmoid would squash all values to (0, 1), making it harder to spread information across the latent dimensions. A linear layer lets the encoder learn whatever distribution of values minimises the reconstruction loss.

### Why Does the Decoder Output Use Sigmoid?

MNIST pixels are normalised to range [0.0, 1.0] for training. The decoder's final layer must produce values in the same range. The **Sigmoid function** maps any real number to the open interval (0, 1):

```
sigmoid(x) = 1 / (1 + exp(-x))
```

This is the natural output activation when targets are in [0, 1].

---

## Reconstruction Loss: MSE vs. BCE

The autoencoder's only training signal is the difference between the input and the reconstruction. Two loss functions are common:

### Mean Squared Error (MSE)

```
L_MSE = (1/N) · Σᵢ (x̂ᵢ - xᵢ)²
```

where N = batch\_size × input\_size, `x̂ᵢ` is the reconstruction, and `xᵢ` is the original pixel.

**Characteristics:**
- Penalises large errors more than small ones (quadratic)
- Sensitive to outlier pixels with extreme reconstruction error
- Gradient: `2 · (x̂ᵢ - xᵢ) / N` — simple and numerically stable
- Optimised for L2-style reconstruction fidelity

### Binary Cross-Entropy (BCE)

```
L_BCE = -(1/N) · Σᵢ [xᵢ · log(x̂ᵢ) + (1 - xᵢ) · log(1 - x̂ᵢ)]
```

**Characteristics:**
- Treats each pixel as a Bernoulli random variable
- Formally correct when input pixels are probabilities (perfectly in [0, 1])
- Gradient becomes unstable if `x̂ᵢ` is near 0 or 1 (log(0) → -∞)
- Often combined with logits (pre-Sigmoid values) for numerical stability

### Why We Use MSE

This implementation uses **MSE** because:
1. MNIST pixels, normalised to [0.0, 1.0], are not strict binary probabilities — many pixels have soft gray values
2. MSE is numerically robust: no log computations that blow up near boundaries
3. The gradient formula is clean and easy to verify analytically
4. MSE works well in practice for MNIST, achieving good reconstruction quality

> **Note:** For binary (black/white) images or when you need a probabilistic interpretation, BCE is theoretically preferable. For natural images with continuous pixel values, MSE is standard.

---

## Vanilla Autoencoder — Implementation Walkthrough

The complete implementation lives in `src/autoencoder/vanilla.rs`. Let's trace through the key parts.

### Constructing the Model

```rust
use rust_neural_networks::autoencoder::vanilla::VanillaAutoencoder;
use rust_neural_networks::utils::rng::SimpleRng;

let mut rng = SimpleRng::new(42);
// Architecture: 784 -> 256 -> 64 -> 256 -> 784
let mut ae = VanillaAutoencoder::new(784, &[256], 64, &[256], &mut rng);
```

The `new` function signature:
```rust
pub fn new(
    input_size: usize,     // 784: flattened 28×28 image
    encoder_sizes: &[usize], // [256]: one hidden encoder layer
    latent_dim: usize,     // 64: bottleneck size
    decoder_sizes: &[usize], // [256]: one hidden decoder layer
    rng: &mut SimpleRng,   // for Xavier weight initialisation
) -> Self
```

The constructor builds:
- **Encoder:** `input_size → encoder_sizes[0] → ... → latent_dim` (N+1 layers)
- **Decoder:** `latent_dim → decoder_sizes[0] → ... → input_size` (M+1 layers)

All layers are `DenseLayer` instances with Xavier-initialised weights.

### The VanillaAutoencoder Struct

```rust
pub struct VanillaAutoencoder {
    encoder: Vec<DenseLayer>,          // Encoder layer stack
    decoder: Vec<DenseLayer>,          // Decoder layer stack
    input_size: usize,                 // 784
    latent_dim: usize,                 // 64
    encoder_layer_sizes: Vec<usize>,   // [784, 256, 64]
    decoder_layer_sizes: Vec<usize>,   // [64, 256, 784]
    // Caches for backward pass:
    encoder_inputs: Vec<Vec<f32>>,     // Input to each encoder layer
    encoder_post_acts: Vec<Vec<f32>>,  // Post-activation output of each encoder layer
    decoder_inputs: Vec<Vec<f32>>,     // Input to each decoder layer
    decoder_post_acts: Vec<Vec<f32>>,  // Post-activation output of each decoder layer
}
```

The cache fields (`encoder_inputs`, `encoder_post_acts`, etc.) store intermediate values during the forward pass so that the backward pass can compute gradients without recomputing them.

---

## Forward Pass Walkthrough

### Encode

The encoder maps a batch of inputs to latent codes:

```rust
pub fn encode(&mut self, input: &[f32], batch_size: usize) -> Vec<f32>
```

Step by step for our 784→256→64 encoder:

```
Input:   [batch_size × 784] floats (pixel values in [0, 1])

Layer 0: DenseLayer(784 → 256)
  1. Cache encoder_inputs[0] = input
  2. Linear: hidden = W₀ · x + b₀    [batch × 256]
  3. ReLU:   hidden = max(0, hidden)   [batch × 256]
  4. Cache encoder_post_acts[0] = hidden

Layer 1: DenseLayer(256 → 64)          ← LAST ENCODER LAYER
  1. Cache encoder_inputs[1] = hidden
  2. Linear: latent = W₁ · hidden + b₁  [batch × 64]
  3. NO activation (linear latent layer)
  4. Cache encoder_post_acts[1] = latent

Output:  [batch_size × 64] latent codes z
```

**Key rule:** All hidden encoder layers use ReLU. The **final encoder layer is linear** (no activation).

### Decode

The decoder maps latent codes back to reconstructed inputs:

```rust
pub fn decode(&mut self, latent: &[f32], batch_size: usize) -> Vec<f32>
```

Step by step for our 64→256→784 decoder:

```
Input:   [batch_size × 64] latent codes z

Layer 0: DenseLayer(64 → 256)
  1. Cache decoder_inputs[0] = latent
  2. Linear: hidden = W₂ · z + b₂      [batch × 256]
  3. ReLU:   hidden = max(0, hidden)    [batch × 256]
  4. Cache decoder_post_acts[0] = hidden

Layer 1: DenseLayer(256 → 784)          ← LAST DECODER LAYER
  1. Cache decoder_inputs[1] = hidden
  2. Linear: output = W₃ · hidden + b₃  [batch × 784]
  3. Sigmoid: output = 1/(1+exp(-output))  [batch × 784, values in (0,1)]
  4. Cache decoder_post_acts[1] = output

Output:  [batch_size × 784] reconstruction x̂  (values in (0, 1))
```

**Key rule:** All hidden decoder layers use ReLU. The **final decoder layer uses Sigmoid** to produce values in (0, 1).

### Full Forward Pass

```rust
pub fn forward(&mut self, input: &[f32], batch_size: usize) -> Vec<f32>
```

Simply calls encode then decode:
```rust
let latent = self.encode(input, batch_size);
self.decode(&latent, batch_size)
```

### Computing the Loss

```rust
let reconstruction = ae.forward(&input_batch, batch_size);
let loss = ae.compute_loss(&reconstruction, &input_batch);
// MSE = mean of (x̂_i - x_i)²
```

The loss uses the **original input as both input and target** — this is the defining feature of autoencoders. The model is trying to reconstruct its own input.

---

## Backward Pass Walkthrough

After the forward pass, `backward()` computes gradients for all parameters:

```rust
ae.backward(&input_batch, batch_size);
```

### Gradient of MSE w.r.t. Decoder Output

The MSE loss is:
```
L = (1/N) · Σᵢ (x̂ᵢ - xᵢ)²
```

Gradient w.r.t. the reconstructed output `x̂ᵢ`:
```
∂L/∂x̂ᵢ = 2 · (x̂ᵢ - xᵢ) / N
```

where `N = batch_size × input_size`.

### Backward Through the Sigmoid Output Layer

The Sigmoid gradient chains with the MSE gradient:
```
∂L/∂zᵢ = ∂L/∂x̂ᵢ · ∂x̂ᵢ/∂zᵢ = ∂L/∂x̂ᵢ · σᵢ · (1 - σᵢ)
```

where `σᵢ = x̂ᵢ` is the cached post-Sigmoid value and `zᵢ` is the pre-Sigmoid linear output.

In code:
```rust
// Sigmoid derivative applied to the MSE gradient
for j in 0..grad.len() {
    let s = dec_post_acts[i][j];  // Cached sigmoid value
    grad[j] *= s * (1.0 - s);
}
```

### Backward Through Dense Layers

For each `DenseLayer`, the backward pass computes:
1. **Weight gradient:** `∂L/∂W = (1/batch) · input⊤ × grad_output`
2. **Bias gradient:** `∂L/∂b = (1/batch) · sum_over_batch(grad_output)`
3. **Input gradient:** `grad_input = grad_output × W⊤`  (to propagate further back)

ReLU gradient (applied before the dense layer backward):
```rust
// ReLU: gradient is 0 where post-activation was ≤ 0, else passes through
for j in 0..grad.len() {
    if dec_post_acts[i][j] <= 0.0 {
        grad[j] = 0.0;
    }
}
```

The backward pass proceeds through **decoder layers in reverse** (last to first), then **encoder layers in reverse** (last to first). The gradient flowing from the decoder into the encoder is the gradient w.r.t. the latent code `z`.

### Updating Parameters

After `backward()`, update weights with:
```rust
// SGD update (for demonstration):
ae.update_parameters(learning_rate);

// Or with Adam optimizer (recommended):
ae.update_with_optimizer(&mut optimizer);
```

---

## Running the Vanilla Autoencoder

### Prerequisites

Ensure the MNIST data is in the `data/` directory.

You can verify the dataset layout with:

```bash
cargo run --bin dataset-helper -- verify --mnist
```

(Download instructions: see the MNIST dataset notes in this repo's docs/tutorials.)

### Build and Run

```bash
# Build release binary (BLAS acceleration required for speed)
cargo build --release --bin mnist_autoencoder

# Run with default config (20 epochs, Adam lr=0.001, batch=64, latent=64)
cargo run --release --bin mnist_autoencoder

# Run with a custom config file
cargo run --release --bin mnist_autoencoder -- my_config.json
```

### Expected Training Output

```
mnist_autoencoder: MNIST Vanilla Autoencoder (784 -> 256 -> 64 -> 256 -> 784)
Loading MNIST data...
  Train: 54000 samples | Val: 6000 samples | Test: 10000 samples
Config: lr=0.001, epochs=20, batch=64, optimizer=adam

Epoch  1/20, Train Loss: 0.045213, Val Loss: 0.038741, Time: 3.2s
Epoch  2/20, Train Loss: 0.031806, Val Loss: 0.029954, Time: 3.1s
Epoch  3/20, Train Loss: 0.027342, Val Loss: 0.026118, Time: 3.1s
...
Epoch 10/20, Train Loss: 0.019451, Val Loss: 0.019227, Time: 3.2s
  ✓ New best model saved (val loss: 0.019227)
...
Training complete. Best val loss: 0.018431
Saved final model to mnist_ae_model_final.bin
Exported latent codes for 10000 test samples → logs/mnist_ae_latent.csv
```

### Output Files

| File | Contents |
|------|----------|
| `mnist_ae_model_best.bin` | Model weights at best validation loss |
| `mnist_ae_model_final.bin` | Model weights at end of training |
| `logs/mnist_autoencoder_train.csv` | Per-epoch training log |
| `logs/mnist_ae_latent.csv` | Test set latent codes for visualisation |

### Configuration Options

The default config (`config/training/mnist_autoencoder_default.json`):
```json
{
  "optimizer_type": "adam",
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "scheduler_type": "step_decay",
  "step_size": 5,
  "gamma": 0.5,
  "learning_rate": 0.001,
  "epochs": 20,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.0001
}
```

**Key parameters to experiment with:**
- `latent_dim` (hardcoded to 64): Reduce to 2 for direct 2D visualisation; increase for better reconstruction
- `learning_rate`: Adam typically works well at 0.001; try 0.0001 for more stable training
- `batch_size`: 64–128 is the sweet spot; larger batches = faster but coarser gradients
- `epochs`: With early stopping, training typically stops around epoch 10–15

---

## Latent Space Visualization

After training, `logs/mnist_ae_latent.csv` contains the latent encodings of all 10,000 MNIST test images. Each row has 64 latent dimensions plus the digit label:

```
z0,z1,z2,...,z63,label
-0.312,0.819,-0.022,...,0.441,7
 0.551,-0.183,0.774,...,-0.291,3
...
```

### Reducing to 2D for Plotting

64 dimensions are too many to plot directly. Use PCA or t-SNE to reduce to 2D:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load latent codes
df = pd.read_csv('logs/mnist_ae_latent.csv')
z = df.iloc[:, :-1].values   # [10000, 64]
labels = df['label'].values

# t-SNE reduction to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(z)  # [10000, 2]

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1],
                      c=labels, cmap='tab10', alpha=0.4, s=5)
plt.colorbar(scatter, label='Digit class')
plt.title('Vanilla Autoencoder Latent Space (t-SNE)')
plt.savefig('latent_space_ae.png', dpi=150, bbox_inches='tight')
```

### What a Good Latent Space Looks Like

**Well-separated clusters** indicate the autoencoder has learned meaningful representations. With a 64-dimensional latent space and a well-trained model:
- Each digit class tends to cluster together in latent space
- Similar digits (1 vs. 7, 4 vs. 9) may overlap partially
- Clusters are not perfectly separated — unlike a supervised classifier

**Why clusters form without labels:** The digit class is the single most predictive factor for reconstruction. All digit "3"s share similar stroke structure, so their encodings naturally cluster.

---

## Variational Autoencoders — Theory

The vanilla autoencoder has a critical limitation: the latent space is **unstructured**. There is no guarantee that nearby points in latent space correspond to similar images. If you sample a random point between two clusters, the decoder might produce nonsense — there was never any training signal for that region of space.

The **Variational Autoencoder** (VAE) solves this by making the latent space a proper probability distribution. Instead of encoding each input as a single point `z`, the encoder outputs a **distribution** over latent codes:

```
Vanilla AE:  input → z                (one point per input)
VAE:         input → (μ, σ²)         (a Gaussian per input)
             then:    z ~ N(μ, σ²)   (sample from that Gaussian)
```

This seemingly small change has profound consequences:
1. The latent space must be **continuous** — nearby points decode to similar images
2. The latent space must be **complete** — any sampled point should decode to something reasonable
3. We can **generate new images** by sampling z ~ N(0, I) and decoding

---

## The ELBO: Evidence Lower Bound

Training a VAE requires maximising the probability that the decoder could reproduce the original data from sampled latent codes. This is mathematically expressed as maximising the **log-likelihood** log p(x).

Log-likelihood is intractable to compute directly (requires integrating over all possible z), so instead we maximise a lower bound called the **ELBO** (Evidence Lower BOund):

```
ELBO = E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))
```

**In practice, we minimise the negative ELBO:**
```
L_VAE = reconstruction_loss + β · KL_loss
```

Where:
- **reconstruction\_loss** = MSE between input and reconstruction (= −E_q[log p(x|z)])
- **KL\_loss** = KL divergence between the encoder distribution q(z|x) and the prior p(z)
- **β** = weight on the KL term (β-VAE; typically 1.0)

### Intuition for Each Term

**Reconstruction term:** "Decode accurately from the sampled z."
- Minimising this encourages the decoder to produce outputs that match the input
- Without this term, the network would learn nothing about the data

**KL term:** "Keep the encoder distributions close to N(0, I)."
- Minimising this prevents the encoder from encoding each input into a tiny isolated region
- Forces the learned distributions to overlap, creating a structured, continuous latent space
- Without this term, the encoder would collapse to point estimates (= vanilla AE)

The tension between these two terms is productive: the encoder must learn representations that are both **informative** (to minimise reconstruction error) and **regular** (to minimise KL divergence).

---

## KL Divergence Explained

The KL divergence measures how different the encoder's distribution q(z|x) is from the prior p(z) = N(0, I):

```
KL(q ‖ p) = E_q[log q(z|x) − log p(z)]
```

For diagonal Gaussians (which our VAE uses), this has a closed-form solution:

```
KL = −½ · Σⱼ (1 + log σⱼ² − μⱼ² − σⱼ²)
```

Or equivalently in terms of `log_var = log σ²`:

```
KL = −½ · mean(1 + log_var − μ² − exp(log_var))
```

This is the formula used in `compute_kl_divergence()`.

### Verifying the Formula

Let's check three cases:

**Case 1: q = p (posterior equals prior)**
- μ = 0, log\_var = 0 (so σ² = 1)
- KL = −½ · (1 + 0 − 0 − 1) = 0 ✓

**Case 2: Posterior shifted away from prior**
- μ = 2, log\_var = 0 (so σ² = 1)
- KL = −½ · (1 + 0 − 4 − 1) = −½ · (−4) = 2.0 > 0 ✓

**Case 3: Posterior variance smaller than prior**
- μ = 0, log\_var = −2 (so σ² ≈ 0.135)
- KL = −½ · (1 + (−2) − 0 − 0.135) = −½ · (−1.135) ≈ 0.57 > 0 ✓

The KL divergence is always ≥ 0, with equality only when the posterior exactly matches the prior.

```rust
// In src/autoencoder/vae.rs:
pub fn compute_kl_divergence(&self, mu: &[f32], log_var: &[f32]) -> f32 {
    let n = mu.len() as f32;
    mu.iter()
        .zip(log_var.iter())
        .map(|(&m, &lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
        .sum::<f32>()
        / n
}
```

---

## The Reparameterization Trick

The VAE samples `z ~ q(z|x) = N(μ, σ²)` during the forward pass. But you cannot backpropagate through a **random sampling operation** — sampling is not differentiable.

The **reparameterization trick** separates the randomness from the parameters:

```
z = μ + ε · σ    where ε ~ N(0, I)
```

Or equivalently, using log\_var = log σ²:
```
z = μ + ε · exp(0.5 · log_var)
```

Now `z` is a **deterministic function** of μ and log\_var (plus the fixed noise ε). Gradients can flow through the multiplication and addition to reach μ and log\_var.

### Why This Works

```
Before reparameterization:
  q(z|x) --[sample]--> z       ← NOT differentiable w.r.t. μ, σ

After reparameterization:
  ε ~ N(0, I)                  ← randomness is isolated here (fixed noise)
  z = μ + ε · σ               ← differentiable function of μ and σ
```

The gradient of the loss w.r.t. μ and log\_var:
```
∂z/∂μ = 1
∂z/∂log_var = ε · 0.5 · exp(0.5 · log_var) = 0.5 · (z − μ)
```

```rust
// Reparameterization in src/autoencoder/vae.rs:
pub fn reparameterize(&mut self, mu: &[f32], log_var: &[f32], rng: &mut SimpleRng) -> Vec<f32> {
    // Sample ε ~ N(0, I) using Box-Muller transform
    let mut eps = vec![0.0f32; n];
    // ... Box-Muller sampling (pairs of Gaussian samples from uniform randoms) ...

    // z = μ + ε · exp(0.5 · log_var)
    let z: Vec<f32> = mu.iter()
        .zip(log_var.iter())
        .zip(eps.iter())
        .map(|((&m, &lv), &e)| m + e * (0.5 * lv).exp())
        .collect();

    self.cached_eps = eps;  // Cache for backward pass
    self.cached_z = z.clone();
    z
}
```

The cached `eps` values are needed in the backward pass to compute `∂z/∂log_var`.

---

## VAE Architecture: Two Encoder Heads

The VAE encoder has a different structure from the vanilla autoencoder:

```
                   ┌─────────────────┐
Input (784)        │  Shared Encoder  │      → μ head (linear)  → μ [latent_dim]
────────────►      │  Trunk (256)     ├──►
                   │  ReLU hidden     │      → log σ² head (linear) → log_var [latent_dim]
                   └─────────────────┘

                   Then: z = μ + ε · exp(0.5 · log_var)   where ε ~ N(0, I)

                   ┌─────────────────┐
                   │    Decoder       │
z [latent_dim] ──► │  (256 ReLU)     ├──► x̂ [784] (Sigmoid)
                   └─────────────────┘
```

The shared trunk computes a common hidden representation. Then **two separate linear layers** (heads) each read from the trunk:
- **μ head:** outputs the mean of the latent Gaussian
- **log\_var head:** outputs the log-variance of the latent Gaussian

Having two separate heads lets the network independently control the location (μ) and spread (σ²) of each latent dimension's distribution.

---

## VAE Implementation Walkthrough

### Constructing the VAE

```rust
use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
use rust_neural_networks::utils::rng::SimpleRng;

let mut rng = SimpleRng::new(42);
// Architecture: 784 -> 256 (trunk) -> (μ: 20, log_var: 20) -> z:20 -> 256 -> 784
let mut vae = VariationalAutoencoder::new(784, &[256], 20, &[256], &mut rng);
```

The VAE binary uses a 20-dimensional latent space (`latent_dim = 20`) — much smaller than the vanilla AE's 64. The lower dimensionality is intentional: the KL regularisation makes every latent dimension carry meaningful information, so you need fewer of them.

### Forward Pass (Training Mode)

```rust
let (reconstruction, mu, log_var) = vae.forward(&input_batch, batch_size, &mut rng);
```

Returns three tensors:
- `reconstruction`: decoded output, shape [batch × 784], values in (0, 1)
- `mu`: latent means, shape [batch × latent\_dim]
- `log_var`: latent log-variances, shape [batch × latent\_dim]

Internally, `forward()`:
1. Calls `encode()` → computes `(mu, log_var)` from encoder trunk + two heads
2. Calls `reparameterize(mu, log_var, rng)` → samples `z`
3. Calls `decode(z)` → produces reconstruction

### Forward Pass (Inference Mode)

During inference (no randomness needed), use the deterministic mean:
```rust
let reconstruction = vae.forward_mean(&input_batch, batch_size);
```

This calls `encode()` and then `decode(mu)` directly, skipping the sampling step.

### Computing the ELBO Loss

```rust
let kl_weight = 1.0_f32;  // β = 1 for standard VAE
let elbo = vae.compute_elbo_loss(
    &reconstruction,    // decoder output
    &input_batch,       // original input (target)
    &mu,                // latent means
    &log_var,           // latent log-variances
    kl_weight,
);
// elbo = reconstruction_loss + kl_weight * kl_divergence
```

The function also allows you to inspect components separately:
```rust
let recon_loss = vae.compute_reconstruction_loss(&reconstruction, &input_batch);
let kl_loss    = vae.compute_kl_divergence(&mu, &log_var);
let elbo       = recon_loss + kl_weight * kl_loss;
```

### Backward Pass

```rust
vae.backward(&input_batch, batch_size, kl_weight);
```

The VAE backward pass has more steps than the vanilla AE:

1. **MSE gradient** w.r.t. decoder output (same as vanilla AE)
2. **Decoder backward** through Sigmoid + dense layers → `grad_z`
3. **Reparameterization backward:**
   ```
   grad_mu     = grad_z + (kl_weight / n_latent) · μ
   grad_log_var = grad_z ⊙ 0.5·(z − μ) + (kl_weight / n_latent) · 0.5·(exp(log_var) − 1)
   ```
4. **Backward through μ head** using `grad_mu` → `grad_trunk_from_mu`
5. **Backward through log\_var head** using `grad_log_var` → `grad_trunk_from_logvar`
6. **Sum** both trunk gradients: `grad_trunk = grad_trunk_from_mu + grad_trunk_from_logvar`
7. **Backward through shared encoder trunk** using `grad_trunk`

### Updating Parameters

```rust
vae.update_with_optimizer(&mut optimizer);
// Updates: encoder trunk + mu_layer + log_var_layer + decoder
```

---

## Running the VAE

### Build and Run

```bash
# Build release binary
cargo build --release --bin mnist_vae

# Run with default config (20 epochs, Adam lr=0.001, latent_dim=20)
cargo run --release --bin mnist_vae

# Run with a custom config file
cargo run --release --bin mnist_vae -- config/training/mnist_vae_default.json
```

### Expected Training Output

```
mnist_vae: MNIST Variational Autoencoder (784 -> 256 -> (μ:20, σ²:20) -> 256 -> 784)
Loading MNIST data...
  Train: 54000 samples | Val: 6000 samples | Test: 10000 samples
Config: lr=0.001, epochs=20, batch=64, latent_dim=20, kl_weight=1.0

Epoch  1/20, Recon: 0.081234, KL: 0.003421, ELBO: 0.084655, Val ELBO: 0.076312, Time: 4.1s
Epoch  2/20, Recon: 0.058941, KL: 0.012847, ELBO: 0.071788, Val ELBO: 0.065234, Time: 4.0s
Epoch  3/20, Recon: 0.049312, KL: 0.021583, ELBO: 0.070895, Val ELBO: 0.059814, Time: 4.1s
...
Epoch 12/20, Recon: 0.033218, KL: 0.028451, ELBO: 0.061669, Val ELBO: 0.051122, Time: 4.0s
  ✓ New best model saved (val ELBO: 0.051122)
...
Training complete. Best val ELBO: 0.049873
Saved final model to mnist_vae_model_final.bin
Exported latent mu for 10000 test samples → logs/mnist_vae_latent.csv
```

### Interpreting the Training Curves

**What you should see:**
- **Recon loss** decreases quickly in early epochs — the decoder learns basic reconstruction
- **KL loss** increases from near-zero as the encoder learns to use the latent space
- **ELBO** first decreases fast (reconstruction improvement), then more slowly (KL-reconstruction tradeoff)

**Warning signs:**
- **KL collapse:** KL remains near 0 throughout — the encoder is ignoring the latent space, acting like a deterministic AE. Solution: reduce learning rate or try KL annealing (start β at 0, increase to 1)
- **ELBO not decreasing:** Both losses stuck — try a higher learning rate or reduce `latent_dim`
- **Very high reconstruction loss:** Latent space too small — increase `latent_dim`

### Output Files

| File | Contents |
|------|----------|
| `mnist_vae_model_best.bin` | Model weights at best validation ELBO |
| `mnist_vae_model_final.bin` | Model weights at end of training |
| `logs/mnist_vae_train.csv` | Per-epoch log with recon, KL, ELBO |
| `logs/mnist_vae_latent.csv` | Test set μ values for visualisation (columns: z0...z19, label) |

### VAE Latent Space: 2D Direct Visualisation

With `latent_dim = 20`, you still need dimensionality reduction for visualisation. But with `latent_dim = 2` (modify the binary constant), you can plot directly:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/mnist_vae_latent.csv')
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['z0'], df['z1'],
                      c=df['label'], cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='Digit class')
plt.title('VAE 2D Latent Space (using μ values)')
plt.savefig('vae_latent_2d.png', dpi=150, bbox_inches='tight')
```

### Generating New Images

With a trained VAE, you can generate new MNIST-like images by sampling from the prior:

```python
import numpy as np

# Load VAE encoder/decoder (implement a Python reader for the .bin format)
# Then: sample z ~ N(0, I) and decode
z_sample = np.random.randn(1, 20).astype(np.float32)  # latent_dim=20
# decode(z_sample) → new image
```

The VAE's structured latent space ensures that samples from N(0, I) decode to recognisable digits — this is what sets VAEs apart from vanilla autoencoders.

---

## Vanilla AE vs. VAE: Key Differences

| Aspect | Vanilla Autoencoder | Variational Autoencoder |
|--------|--------------------|-----------------------|
| **Latent representation** | Single point z ∈ ℝᵈ | Distribution N(μ, σ²) |
| **Encoder output** | z directly | (μ, log σ²) pair |
| **Sampling** | None (deterministic) | z = μ + ε·σ during training |
| **Loss function** | MSE only | MSE + β·KL divergence |
| **Latent space structure** | Unstructured (gaps allowed) | Continuous, well-covered |
| **Generative quality** | Poor (gaps decode to noise) | Good (smooth interpolation) |
| **Inference** | Deterministic | Deterministic (use μ) |
| **Training stability** | Simple | More delicate (KL balance) |
| **Architecture** | Single encoder head | Two encoder heads (μ, log\_var) |
| **Implementation** | `VanillaAutoencoder` | `VariationalAutoencoder` |
| **Binary** | `mnist_autoencoder` | `mnist_vae` |
| **Latent dim** | 64 | 20 |

### When to Use Each

**Use Vanilla Autoencoder when:**
- You want the best possible reconstruction quality
- You need fast training without hyperparameter sensitivity
- The downstream task only needs compressed features (not generation)
- You are learning the architecture concepts for the first time

**Use VAE when:**
- You want to generate new samples
- You need a structured, interpolatable latent space
- You are doing latent space arithmetic (e.g., "digit 3 style + digit 7 shape")
- Downstream tasks benefit from regularised, smooth representations

---

## Verification Checkpoints

Use these checkpoints to verify your understanding:

### Checkpoint 1: Architecture Sizes

For `VanillaAutoencoder::new(784, &[256], 64, &[256], &mut rng)`:
- How many encoder layers does the model have? → **2** (784→256, 256→64)
- What is the activation of the last encoder layer? → **Linear (none)**
- What is the activation of the last decoder layer? → **Sigmoid**
- How many total parameters? → **435,536**

### Checkpoint 2: Loss Values

For MSE loss:
- `compute_loss(&[0.5, 0.5], &[0.5, 0.5])` = ? → **0.0** (perfect reconstruction)
- `compute_loss(&[0.0, 0.0], &[1.0, 1.0])` = ? → **1.0** (mean of (0-1)² = 1)

For KL divergence:
- `compute_kl_divergence(&[0.0, 0.0], &[0.0, 0.0])` = ? → **0.0** (posterior equals prior)

### Checkpoint 3: Reparameterization

Given μ = 0.5, log\_var = 0.0, ε = 1.0:
```
σ = exp(0.5 · log_var) = exp(0) = 1.0
z = μ + ε · σ = 0.5 + 1.0 · 1.0 = 1.5
```

Given μ = 0.5, log\_var = −2.0, ε = 1.0:
```
σ = exp(0.5 · (−2)) = exp(−1) ≈ 0.368
z = 0.5 + 1.0 · 0.368 ≈ 0.868
```

### Checkpoint 4: Gradient Flow

Trace the gradient from the MSE loss backward through one decoder layer:

1. MSE gradient: `g = 2 · (x̂ − x) / N`
2. Sigmoid gradient: `g = g · σ(z) · (1 − σ(z))`
3. Dense layer backward: weight\_grad, bias\_grad, grad\_input

If any step produces `NaN` or `Inf`, check:
- Division by zero in loss gradient (ensure N > 0)
- Sigmoid overflow (check pre-sigmoid values aren't ±1000)
- Exploding gradients (monitor gradient norms, reduce learning rate)

---

## Exercises

### Beginner

1. **Modify the latent dimension:** Change `LATENT_DIM` from 64 to 2 in `mnist_autoencoder.rs`. Train and plot the 2D latent space directly (no t-SNE needed). What do you observe?

2. **Verify MSE gradient:** For a single pixel with x=0.8 and x̂=0.3, N=784, manually compute the MSE gradient and check it matches what the code produces after one forward+backward pass.

3. **Inspect reconstruction quality:** After training, load the model and reconstruct 5 test images. Display the original and reconstructed images side-by-side. What details are lost in the 64D bottleneck?

### Intermediate

4. **Add BCE loss:** Implement a `compute_bce_loss` method in `VanillaAutoencoder` that computes binary cross-entropy. Compare reconstruction quality with MSE for the same number of training epochs.

5. **KL annealing for VAE:** Implement a β schedule that starts at 0 and linearly increases to 1 over 10 epochs. This often helps avoid KL collapse. Does it improve the latent space structure?

6. **Denoising autoencoder:** Modify `mnist_autoencoder.rs` to add Gaussian noise (σ=0.1) to input images during training, but use clean images as the reconstruction target. Does training become more challenging? Does the reconstruction improve or degrade on clean test images?

### Advanced

7. **Convolutional autoencoder:** Replace the dense encoder/decoder with convolutional layers using the existing `Conv2DLayer`. What input shape changes are needed? Compare reconstruction quality with the MLP autoencoder.

8. **Latent space interpolation:** Given two test images of different digits, linearly interpolate 8 steps between their latent codes and decode each. Compare the interpolation quality between vanilla AE and VAE. What do gaps in the vanilla AE's latent space look like when decoded?

9. **β-VAE:** Implement a β-VAE by varying `kl_weight` across {0.1, 0.5, 1.0, 4.0}. How does increasing β affect reconstruction quality and latent space disentanglement? Plot the t-SNE visualisation for each β.

---

## Next Steps

You have now completed the full tutorial series from XOR to autoencoders! Here are directions for further exploration:

**Within this codebase:**
- Explore `src/autoencoder/vanilla.rs` and `src/autoencoder/vae.rs` for implementation details
- Read the integration tests in `tests/test_autoencoder.rs` and `tests/test_vae.rs`
- Try the exercises above to deepen your understanding

**Theory to study next:**
- **Flow-based models** (normalising flows): exact likelihood computation, no approximation needed
- **Generative Adversarial Networks (GANs)**: adversarial training as an alternative to ELBO
- **Vector Quantized VAE (VQ-VAE)**: discrete latent codes for sharper generation
- **Diffusion models**: state-of-the-art image generation via iterative denoising

**Practical skills:**
- Implement a full pipeline: train VAE, extract latent codes, train a simple classifier on the latent codes (semi-supervised learning)
- Visualise what each latent dimension encodes by traversing it while holding others fixed
- Study how reconstruction quality scales with latent dimension size

---

**Summary of what you learned:**

| Concept | Key Insight |
|---------|-------------|
| Autoencoder | Forced compression reveals structure in data |
| Bottleneck | Smaller latent dimension = more compression, harder reconstruction |
| MSE loss | Natural for continuous, normalised pixel values |
| Latent space | Compact feature representation learned without labels |
| VAE vs AE | VAE adds KL regularisation for a structured, generative latent space |
| ELBO | Reconstruction + KL: accuracy vs. regularity tradeoff |
| KL divergence | Measures how far the encoder distribution deviates from N(0, I) |
| Reparameterization | Separates randomness from parameters to enable backpropagation |

← [Tutorial 05: Automatic Differentiation Engine](05_autograd_engine.md) | [Tutorial Index](README.md)
