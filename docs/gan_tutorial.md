# GAN Tutorial: Generative Adversarial Networks for MNIST

This document explains GAN theory, the architecture implemented in this project, common training challenges, mode collapse detection, and a complete usage guide for the `mnist_gan` binary.

## Table of Contents

- [What is a GAN?](#what-is-a-gan)
- [The Minimax Game](#the-minimax-game)
- [Loss Functions](#loss-functions)
  - [Discriminator Loss](#discriminator-loss)
  - [Generator Loss](#generator-loss)
  - [Label Smoothing](#label-smoothing)
- [Architecture](#architecture)
  - [Generator (Noise → Image)](#generator-noise--image)
  - [Discriminator (Image → Real/Fake)](#discriminator-image--realfake)
  - [Activation Functions: Why Leaky ReLU?](#activation-functions-why-leaky-relu)
- [Training Algorithm](#training-algorithm)
  - [Step-by-Step Training Loop](#step-by-step-training-loop)
  - [Gradient Flow Diagram](#gradient-flow-diagram)
  - [Why Adam with β₁=0.5?](#why-adam-with-β₁05)
- [Training Challenges](#training-challenges)
  - [Mode Collapse](#mode-collapse)
  - [Vanishing Gradients](#vanishing-gradients)
  - [Training Instability](#training-instability)
  - [The Discriminator–Generator Balance](#the-discriminatorgenerator-balance)
- [Mode Collapse Detection](#mode-collapse-detection)
- [Interpreting Training Metrics](#interpreting-training-metrics)
  - [What Healthy Training Looks Like](#what-healthy-training-looks-like)
  - [Signs of Trouble](#signs-of-trouble)
- [Configuration Reference](#configuration-reference)
- [Usage Guide](#usage-guide)
  - [Quick Start](#quick-start)
  - [Custom Configuration](#custom-configuration)
  - [Visualising Generated Samples](#visualising-generated-samples)
  - [Output Files](#output-files)
- [Hyperparameter Tuning Tips](#hyperparameter-tuning-tips)

---

## What is a GAN?

A **Generative Adversarial Network** (Goodfellow et al., 2014) consists of two neural networks trained simultaneously in opposition:

- The **Generator (G)** learns to synthesise realistic-looking data from random noise.
- The **Discriminator (D)** learns to distinguish real data from generated fakes.

The two networks are trained against each other. G tries to fool D, and D tries not to be fooled. Over time, this adversarial pressure drives G to produce increasingly realistic outputs.

**Intuition**: Think of G as a forger learning to paint counterfeit artworks, and D as an art critic trying to spot fakes. As the critic gets better at detection, the forger is forced to improve their technique, and vice versa. In equilibrium, the forger produces work indistinguishable from genuine art.

In this project, G learns to generate handwritten digit images that look like they came from the MNIST dataset, starting from nothing but random noise.

---

## The Minimax Game

GANs are formally framed as a **two-player minimax game**. The value function is:

```
min_G  max_D  V(D, G) =
    𝔼[log D(x)]  (D maximises this — real data x should score high)
  + 𝔼[log(1 − D(G(z)))]  (D maximises this too; G minimises it)
```

Where:
- `x` ~ real data distribution `p_data`
- `z` ~ noise distribution (uniform over `[−1, 1]` here)
- `G(z)` is a generated (fake) sample
- `D(x)` is the discriminator's probability that `x` is real

**Discriminator's goal**: maximise `V` → output high probability for real images, low for fakes.

**Generator's goal**: minimise `V` → make `D(G(z))` close to 1 (fool D into thinking fakes are real).

**Nash equilibrium**: When neither player can improve unilaterally. At equilibrium `p_G = p_data` (the generator perfectly replicates the data distribution) and `D(x) = 0.5` everywhere (D cannot tell real from fake).

In practice, the full equilibrium is rarely reached, but the adversarial pressure still drives G to produce visually convincing samples.

---

## Loss Functions

### Discriminator Loss

The discriminator is trained with **binary cross-entropy (BCE)** loss to solve a binary classification problem: real (label = 1) vs. fake (label = 0).

```
L_D = −[y · log(D(x) + ε) + (1 − y) · log(1 − D(x) + ε)]
```

For a mini-batch of real images (y = 1 − label_smoothing ≈ 0.9) and fake images (y = 0):

```
L_D(real)  = −log(D(x_real) + ε)         → push D(real) toward 1
L_D(fake)  = −log(1 − D(G(z)) + ε)       → push D(fake) toward 0
L_D_total  = (L_D(real) + L_D(fake)) / 2
```

### Generator Loss

The generator is trained using the **non-saturating** version of the GAN loss. Instead of minimising `log(1 − D(G(z)))` (which saturates early), G maximises `log(D(G(z)))`:

```
L_G = −log(D(G(z)) + ε)   (equivalently: BCE(D(G(z)), target=1))
```

In implementation terms: G is trained to make D output `1.0` for its generated images.

```
grad_logit_g[i] = D(G(z))[i] − 1.0    (negative → gradient pushes D output toward 1)
```

This non-saturating form provides stronger gradients when D confidently rejects fakes (`D(G(z)) ≈ 0`), avoiding vanishing gradients early in training.

### Label Smoothing

**One-sided label smoothing** replaces the hard real target of `1.0` with a softer value:

```
real_target = 1.0 − label_smoothing     (default: 0.9 when label_smoothing = 0.1)
fake_target = 0.0                        (fake targets are not smoothed)
```

This prevents D from becoming too confident on real images, which:
- Reduces the risk of D providing poor gradient signals to G
- Acts as a mild regularizer for D
- Helps maintain training stability

Label smoothing is only applied to the real side (one-sided) because smoothing fake labels to a non-zero value would give G incorrect gradient signals.

---

## Architecture

### Generator (Noise → Image)

The generator maps a random **latent noise vector** `z ∈ ℝ^100` to a synthetic image `G(z) ∈ ℝ^784` (a flattened 28×28 pixel grid).

```
Input:   z ∈ ℝ^100           Uniform noise in [−1, 1]
Layer 1: 100 → 256           Linear + LeakyReLU(α=0.2)
Layer 2: 256 → 512           Linear + LeakyReLU(α=0.2)
Layer 3: 512 → 784           Linear + Tanh
Output:  G(z) ∈ ℝ^784        Pixel values in (−1, 1)
```

**Parameter count:**
| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Layer 1 | 100 × 256 = 25,600 | 256 | 25,856 |
| Layer 2 | 256 × 512 = 131,072 | 512 | 131,584 |
| Layer 3 | 512 × 784 = 401,408 | 784 | 402,192 |
| **Total** | | | **559,632** |

**Tanh output**: The final Tanh activation maps outputs to the open interval `(−1, 1)`. Real MNIST images (originally in `[0, 1]`) are rescaled to `[−1, 1]` before being fed to the discriminator, ensuring both real and fake images live in the same value range.

### Discriminator (Image → Real/Fake)

The discriminator maps a 784-dimensional image to a single scalar probability that the image is real.

```
Input:   x ∈ ℝ^784           Pixel values in [−1, 1]
Layer 1: 784 → 512           Linear + LeakyReLU(α=0.2)
Layer 2: 512 → 256           Linear + LeakyReLU(α=0.2)
Layer 3: 256 → 1             Linear + Sigmoid
Output:  D(x) ∈ (0, 1)       Probability that x is real
```

**Parameter count:**
| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Layer 1 | 784 × 512 = 401,408 | 512 | 401,920 |
| Layer 2 | 512 × 256 = 131,072 | 256 | 131,328 |
| Layer 3 | 256 × 1 = 256 | 1 | 257 |
| **Total** | | | **533,505** |

### Activation Functions: Why Leaky ReLU?

Both networks use **Leaky ReLU** (α=0.2) in hidden layers instead of standard ReLU. This choice is important for GAN stability:

**Standard ReLU** problem — the "dying ReLU":
```
ReLU(x) = max(0, x)
```
If a neuron receives consistently negative inputs, its gradient becomes exactly 0. The neuron "dies" and can never recover, reducing the network's effective capacity.

**Leaky ReLU** fix:
```
LeakyReLU(x, α) = x         if x ≥ 0
                = α · x      if x < 0    (α = 0.2 in this implementation)
```

With α=0.2, negative-input neurons still receive a gradient of 0.2 (rather than 0), allowing them to recover. This is especially important in GANs because:
- D may produce large negative logits for generated images early in training
- G must propagate gradients through D back to update its own weights
- If D's neurons die, G receives zero gradient and cannot improve

---

## Training Algorithm

### Step-by-Step Training Loop

Each mini-batch performs three gradient updates:

```
For each mini-batch of real images:

  ──── Step 1: Train D on real images ─────────────────────────────────
  1a. Forward: D(real_images) → d_pred_real
  1b. Compute gradient: grad_logit_real = d_pred_real − real_target
      (real_target = 1.0 − label_smoothing ≈ 0.9)
  1c. Backward: update D's parameters to output high scores for real images

  ──── Step 2: Train D on fake images ─────────────────────────────────
  2a. Sample noise z ~ Uniform(−1, 1)
  2b. Forward G: G(z) → fake_images
  2c. Forward D: D(fake_images) → d_pred_fake
  2d. Compute gradient: grad_logit_fake = d_pred_fake − 0.0  (fake target = 0)
  2e. Backward: update D's parameters to output low scores for fake images

  ──── Step 3: Train G (fool D) ────────────────────────────────────────
  3a. Sample fresh noise z' ~ Uniform(−1, 1)
  3b. Forward G: G(z') → fake_images_new
  3c. Forward D: D(fake_images_new) → d_pred_g
  3d. Compute gradient: grad_logit_g = d_pred_g − 1.0  (G wants D to output 1)
  3e. Propagate gradient through D **without** updating D's parameters
  3f. Backward through G: update G's parameters using the propagated gradient
```

**Key design decision in Step 3**: Fresh noise `z'` is sampled (not reusing `z` from Step 2). This ensures G is optimised with gradients from the current (already-updated) D, keeping the adversarial signal accurate.

**Why D is not updated in Step 3**: During G's update, we propagate the adversarial gradient through D to compute `∂L_G/∂G(z)`. But D's weights must not change — we only want to update G. This is implemented by `Discriminator::propagate_gradient()`, which computes the gradient flow using BLAS matrix multiplications without touching D's gradient accumulators.

### Gradient Flow Diagram

```
Generator training step (Step 3):

  z' ─→ [G: Layer1 → Layer2 → Layer3 → Tanh] ─→ G(z')
                                                     │
                                                     ▼
                              [D: Layer1 → Layer2 → Layer3 → Sigmoid] ─→ D(G(z'))
                                                                               │
                                                              L_G = −log(D(G(z')))
                                                                               │
                                        ∂L_G/∂logit_D = D(G(z')) − 1.0       │
                                                                               │
                      ◄──── propagate_gradient (D not updated) ◄──────────────┘
                      │
              ∂L_G/∂G(z')  (gradient w.r.t. G's output)
                      │
      ◄──── G.backward (G IS updated) ◄────────────────────────────────────────
```

### Why Adam with β₁=0.5?

The default GAN configuration uses **Adam with β₁=0.5** (instead of the typical 0.9). This is a well-established GAN training recommendation:

- **β₁=0.9** (standard Adam): High momentum accumulates past gradients over ~10 steps. In a GAN, the loss landscape shifts as both networks update simultaneously. High momentum can cause the optimiser to overshoot.
- **β₁=0.5** (GAN Adam): Lower momentum means the optimiser responds more quickly to the current gradient direction, adapting as the adversarial landscape evolves.

The second moment decay β₂=0.999 remains standard, providing adaptive learning rates per parameter.

---

## Training Challenges

### Mode Collapse

**Mode collapse** is the most common GAN failure mode. It occurs when the generator maps many different noise vectors to the same (or very similar) output — typically the one or two MNIST digit classes that are easiest to fool the discriminator with.

**Conceptual example:**
```
Before mode collapse:
  G(z₁) → "3"  G(z₂) → "7"  G(z₃) → "1"  G(z₄) → "9"  (diverse)

After mode collapse:
  G(z₁) → "1"  G(z₂) → "1"  G(z₃) → "1"  G(z₄) → "1"  (all the same)
```

**Why it happens**: If G discovers a particular output that consistently fools D, it can exploit this by mapping all noise vectors to that output. D then adapts to recognise this pattern, G shifts to a different output, and the cycle continues — but G never learns to cover the full data distribution.

**How to detect it**: The diversity metric measures the standard deviation of per-sample mean pixel values across 100 generated images:

```
diversity = std_dev(mean_pixels(G(z₁)), mean_pixels(G(z₂)), ..., mean_pixels(G(z₁₀₀)))
```

A low diversity score (< 0.02) indicates that generated images all have similar overall brightness, suggesting mode collapse. The training loop prints a warning:

```
WARNING: Mode collapse detected at epoch 12 (diversity: 0.0153)
```

**Mitigation strategies:**
- Reduce the learning rate
- Increase the batch size (more gradient diversity per update)
- Use one-sided label smoothing (default: 0.1)
- Monitor D(real) and D(fake): if D becomes too strong (D(real) → 0.9, D(fake) → 0.0) very quickly, G has no useful gradient to learn from

### Vanishing Gradients

If D becomes too accurate too quickly, the loss `log(1 − D(G(z)))` saturates near zero. G receives tiny gradients and cannot improve.

**Signs**: G loss increases or plateaus while D loss approaches 0.

**Why the non-saturating loss helps**: The non-saturating formulation `−log(D(G(z)))` provides gradient `−1/D(G(z))`, which is large when `D(G(z))` is near 0 (D confidently rejecting fakes). This gives G useful signal even when it is far behind D.

### Training Instability

Unlike supervised learning, GAN training loss does not monotonically decrease. Both losses oscillate because:
1. When G improves, D's task gets harder → D loss increases.
2. When D improves, G's gradient signal strengthens → G must adapt.

**Healthy oscillation** (the losses are coupled and both remain bounded) is normal. **Unbounded divergence** (one or both losses growing without bound) indicates a problem.

### The Discriminator–Generator Balance

The GAN training dynamics depend critically on the relative strength of D and G:

| Scenario | What happens | What to do |
|----------|-------------|------------|
| D too strong | D(fake) ≈ 0 early; G gets no gradient | Lower `d_lr`, raise `g_lr` |
| G too strong | D cannot distinguish real from fake | Raise `d_lr`, lower `g_lr` |
| Balanced | Both losses oscillate; diversity stays high | Keep current config |

The separate `g_lr` and `d_lr` configuration parameters let you tune this balance independently.

---

## Mode Collapse Detection

The `compute_diversity` function samples 100 images from the generator and computes a simple diversity metric:

```
1. Generate 100 images: G(z₁), G(z₂), ..., G(z₁₀₀)
2. For each image, compute mean pixel value: μᵢ = mean(G(zᵢ))
3. Compute std-dev across means: diversity = std_dev(μ₁, μ₂, ..., μ₁₀₀)
```

**Interpretation:**
```
diversity > 0.10  → Healthy: generator is producing diverse images
diversity 0.05–0.10 → Mild mode collapse: reduced variety
diversity 0.02–0.05 → Moderate mode collapse: only a few modes covered
diversity < 0.02  → WARNING: severe mode collapse, generator stuck
```

This metric is logged to `logs/mnist_gan_log.csv` as the `sample_diversity` column every epoch, enabling you to track diversity over the full training run.

---

## Interpreting Training Metrics

### What Healthy Training Looks Like

A well-functioning GAN shows these patterns in the training output:

```
Epoch 1/50,  G loss: 1.8432, D loss: 0.6921, D(real): 0.5812, D(fake): 0.4239, Diversity: 0.1823, Time: 8.2s
Epoch 5/50,  G loss: 1.4218, D loss: 0.6345, D(real): 0.7023, D(fake): 0.3041, Diversity: 0.1654, Time: 8.0s
Epoch 10/50, G loss: 1.3901, D loss: 0.6512, D(real): 0.6712, D(fake): 0.3287, Diversity: 0.1501, Time: 8.1s
Epoch 25/50, G loss: 1.2987, D loss: 0.6234, D(real): 0.6901, D(fake): 0.3099, Diversity: 0.1432, Time: 7.9s
```

**Key observations:**
- **D loss ≈ 0.60–0.70**: Near the theoretical optimum of `log(2) ≈ 0.693` for a balanced GAN.
- **D(real) ≈ 0.6–0.8**: D correctly identifies most real images.
- **D(fake) ≈ 0.2–0.4**: D correctly identifies most fakes (but not perfectly).
- **G loss decreasing slowly**: G is improving.
- **Diversity > 0.10**: No mode collapse.

### Signs of Trouble

```
# Discriminator winning (too strong):
Epoch 5/50, G loss: 4.2314, D loss: 0.1823, D(real): 0.9801, D(fake): 0.0089, Diversity: 0.0821
# → D(real) near 1.0 and D(fake) near 0.0 means G's gradient is vanishing

# Mode collapse occurring:
Epoch 15/50, G loss: 2.1234, D loss: 0.5234, D(real): 0.7123, D(fake): 0.2877, Diversity: 0.0143
WARNING: Mode collapse detected at epoch 15 (diversity: 0.0143)
# → Diversity has dropped below 0.02

# Generator winning (discriminator collapsed):
Epoch 20/50, G loss: 0.1234, D loss: 1.9823, D(real): 0.1023, D(fake): 0.8977, Diversity: 0.2341
# → D(real) near 0 and D(fake) near 1 means D cannot distinguish real from fake
# This looks like success but often means D's gradients have become useless
```

---

## Configuration Reference

Configuration is controlled via JSON files in `config/`. The full list of GAN-specific parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_type` | string | `"adam"` | Optimizer type (use `"adam"` for GANs) |
| `learning_rate` | float | `0.0002` | Fallback LR if `g_lr`/`d_lr` not set |
| `g_lr` | float | `0.0002` | Generator learning rate |
| `d_lr` | float | `0.0002` | Discriminator learning rate |
| `beta1` | float | `0.5` | Adam β₁ (use 0.5 for GANs, not 0.9) |
| `beta2` | float | `0.999` | Adam β₂ |
| `epsilon` | float | `1e-8` | Adam numerical stability constant |
| `epochs` | int | `50` | Number of training epochs |
| `batch_size` | int | `64` | Mini-batch size |
| `noise_dim` | int | `100` | Latent noise vector dimension |
| `label_smoothing` | float | `0.1` | One-sided label smoothing (real target = 1 − this) |
| `activation_function` | string | `"leaky_relu"` | Activation (keep `"leaky_relu"` for GANs) |
| `leaky_relu_alpha` | float | `0.2` | Leaky ReLU negative slope |
| `validation_split` | float | `0.0` | GAN training uses all data (no validation split) |
| `early_stopping_patience` | int | `10` | Epochs without G loss improvement before stopping |

**Default config** (`config/training/mnist_gan_default.json`):

```json
{
  "optimizer_type": "adam",
  "beta1": 0.5,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.2,
  "learning_rate": 0.0002,
  "g_lr": 0.0002,
  "d_lr": 0.0002,
  "epochs": 50,
  "batch_size": 64,
  "validation_split": 0.0,
  "early_stopping_patience": 10,
  "early_stopping_min_delta": 0.001,
  "noise_dim": 100,
  "label_smoothing": 0.1,
  "enable_augmentation": false
}
```

---

## Usage Guide

### Quick Start

Ensure MNIST data is present in `./data/` (see `docs/2b MNIST-Data-Setup.md`), then:

```bash
# Build in release mode (required for BLAS performance)
cargo build --release

# Run with default configuration (50 epochs, batch_size=64)
cargo run --release --bin mnist_gan
```

Expected output:
```
=== MNIST GAN Training ===
Loading configuration from: config/training/mnist_gan_default.json
G lr: 0.0002, D lr: 0.0002, beta1: 0.5, beta2: 0.999, noise_dim: 100
label_smoothing: 0.1, epochs: 50, batch_size: 64
Loading MNIST training data...
Loaded 60000 training images in 0.12s
Generator:     100 → 256 → 512 → 784 (559632 params)
Discriminator: 784 → 512 → 256 → 1  (533505 params)
Starting GAN training...
Epoch 1/50, G loss: 1.8234, D loss: 0.6924, D(real): 0.6012, D(fake): 0.4231, Diversity: 0.1834, Time: 8.3s
...
```

### Custom Configuration

Override any hyperparameter by passing a JSON config file as the first argument:

```bash
# Run with a custom config
cargo run --release --bin mnist_gan -- my_config.json
```

**Example: slower discriminator to prevent mode collapse**

```json
{
  "optimizer_type": "adam",
  "beta1": 0.5,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.2,
  "learning_rate": 0.0002,
  "g_lr": 0.0003,
  "d_lr": 0.0001,
  "epochs": 100,
  "batch_size": 128,
  "validation_split": 0.0,
  "early_stopping_patience": 10,
  "early_stopping_min_delta": 0.001,
  "noise_dim": 100,
  "label_smoothing": 0.15
}
```

**Example: faster training with less label smoothing**

```json
{
  "optimizer_type": "adam",
  "beta1": 0.5,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.2,
  "learning_rate": 0.0005,
  "g_lr": 0.0005,
  "d_lr": 0.0005,
  "epochs": 30,
  "batch_size": 64,
  "validation_split": 0.0,
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.001,
  "noise_dim": 100,
  "label_smoothing": 0.05
}
```

### Visualising Generated Samples

After training, 16 generated digit images are exported to `logs/mnist_gan_samples.csv`.
Each row in the file represents one 28×28 generated image as 784 comma-separated `f32` values in `[−1, 1]`.

To rescale to the `[0, 255]` range for display:

```python
import numpy as np
import matplotlib.pyplot as plt

samples = np.loadtxt("logs/mnist_gan_samples.csv", delimiter=",")
# Rescale from [-1, 1] to [0, 1]
samples = (samples + 1.0) / 2.0

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for idx, ax in enumerate(axes.flat):
    ax.imshow(samples[idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.tight_layout()
plt.savefig("gan_samples.png", dpi=150)
plt.show()
```

To visualise the training metrics (G loss, D loss, diversity over epochs):

```bash
python plot_comparison.py logs/mnist_gan_log.csv
```

### Output Files

| File | Description |
|------|-------------|
| `logs/mnist_gan_log.csv` | Per-epoch metrics: `epoch,g_loss,d_loss,d_real,d_fake,sample_diversity` |
| `logs/mnist_gan_samples.csv` | 16 generated images after training (784 pixels each) |
| `mnist_gan_best.bin` | Model checkpoint saved at the epoch with best G loss |
| `mnist_gan_final.bin` | Final model after all training epochs complete |

**CSV log column descriptions:**

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1-indexed) |
| `g_loss` | Generator loss averaged over all batches |
| `d_loss` | Discriminator loss averaged over all batches (real + fake) |
| `d_real` | Average D(x_real) — should stay in 0.6–0.8 for healthy training |
| `d_fake` | Average D(G(z)) — should stay in 0.2–0.4 for healthy training |
| `sample_diversity` | Std-dev of per-sample mean pixel values (< 0.02 = mode collapse warning) |

---

## Hyperparameter Tuning Tips

**Learning rate (`g_lr` / `d_lr`):**
- Default `0.0002` is a standard DCGAN starting point.
- If D loss collapses to near 0 quickly, reduce `d_lr` (e.g., `0.0001`) or increase `g_lr`.
- If G loss diverges and D(fake) stays high, try reducing `g_lr`.

**Batch size:**
- Larger batches (128–256) provide more stable gradient estimates and can reduce mode collapse.
- Smaller batches (32–64) introduce more stochasticity, which can help escape local optima but makes training noisier.

**Label smoothing:**
- Start at `0.1` (default). This replaces hard real targets of `1.0` with `0.9`.
- Increase to `0.15` or `0.2` if D becomes too confident too early.
- Decrease to `0.0` (no smoothing) only if G loss is consistently high and D is not converging.

**Beta₁ (Adam momentum):**
- Keep at `0.5` for GAN training. Increasing toward `0.9` can cause instability.

**Epochs:**
- 50 epochs is typically sufficient to see recognisable digit generation.
- Run for 100+ epochs only if diversity remains high and losses are still improving.
- Monitor mode collapse: if diversity drops below 0.05 before epoch 20, consider restarting with adjusted hyperparameters.

**Troubleshooting checklist:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| G loss > 3.0 and not decreasing | D too strong | Reduce `d_lr`, increase `label_smoothing` |
| Diversity < 0.02 by epoch 10 | Mode collapse early | Reduce `g_lr`, increase `batch_size` |
| D loss > 1.5 and increasing | G too strong | Reduce `g_lr`, increase `d_lr` |
| D(real) < 0.5 | D not learning | Check MNIST data loading; try smaller LR |
| Both losses diverging | LR too high | Reduce both `g_lr` and `d_lr` by 5× |
