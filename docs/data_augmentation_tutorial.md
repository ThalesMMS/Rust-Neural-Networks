# Data Augmentation Tutorial

This document explains data augmentation concepts, the augmentation types implemented in this project, how to enable them via configuration, and what accuracy improvements to expect.

## Table of Contents

- [What is Data Augmentation?](#what-is-data-augmentation)
- [Why Augmentation Improves Generalization](#why-augmentation-improves-generalization)
- [Augmentation Types](#augmentation-types)
  - [Horizontal Flip](#horizontal-flip)
  - [Random Crop](#random-crop)
  - [Brightness Jitter](#brightness-jitter)
  - [Contrast Jitter](#contrast-jitter)
  - [Saturation Jitter](#saturation-jitter)
- [Enabling Augmentation via Config](#enabling-augmentation-via-config)
  - [Before: Default Config (No Augmentation)](#before-default-config-no-augmentation)
  - [After: Augmented Config](#after-augmented-config)
- [Model-Specific Configs](#model-specific-configs)
  - [MNIST CNN with Augmentation](#mnist-cnn-with-augmentation)
  - [MNIST MLP with Augmentation](#mnist-mlp-with-augmentation)
  - [CIFAR-10 CNN with Augmentation](#cifar-10-cnn-with-augmentation)
- [Expected Accuracy Improvements](#expected-accuracy-improvements)
- [Implementation Details](#implementation-details)
- [Best Practices](#best-practices)

---

## What is Data Augmentation?

Data augmentation is a **regularization technique** that artificially increases the size and diversity of the training dataset by applying random transformations to training images. Each time a training image is used, a new randomly-transformed version is generated on-the-fly.

The key insight is that these transformations preserve the label: a horizontally-flipped image of the digit "3" is still a "3", and a slightly brighter photograph of a dog is still a dog. The model learns to be invariant to these types of variation, which improves generalization to unseen data.

**Without augmentation:**
```
Training set: 54,000 unique MNIST images
Each epoch:  Model sees the same 54,000 images
Problem:     Model memorizes exact pixel patterns → overfitting
```

**With augmentation:**
```
Training set: 54,000 images × random transforms per epoch
Each epoch:  Model sees slightly different versions of each image
Result:      Model learns robust, invariant representations → better generalization
```

---

## Why Augmentation Improves Generalization

Neural networks are powerful enough to memorize the training set. Augmentation prevents this by ensuring that exact memorization is never possible — the same image will look slightly different every epoch.

**Conceptual example** — without augmentation the model overfits:

```
Epoch 1:  Train loss 0.05, Val loss 0.18  ← gap growing
Epoch 2:  Train loss 0.03, Val loss 0.22  ← overfitting
Epoch 3:  Train loss 0.01, Val loss 0.27  ← worse generalization
```

**With augmentation**, the training-validation gap is smaller:

```
Epoch 1:  Train loss 0.12, Val loss 0.14  ← small gap
Epoch 2:  Train loss 0.09, Val loss 0.11  ← improving together
Epoch 3:  Train loss 0.07, Val loss 0.09  ← healthy convergence
```

Augmentation is especially important for CIFAR-10, where the model can easily memorize the training images but needs to generalize to unseen photographs.

---

## Augmentation Types

All augmentation functions are implemented in `src/data/augmentation.rs` and work with pixel-interleaved RGB format (`RGBRGBRGB...`), as used by the CIFAR-10 data loader. They operate on normalized pixel values in `[0.0, 1.0]`.

### Horizontal Flip

**What it does:** Randomly mirrors the image left-to-right.

```
Before:  [← cat facing right →]
After:   [← cat facing left  →]
```

**Mathematical operation:**
```
For each row r, pixel column c:
  output[r][c] = input[r][width - 1 - c]
```

**Configuration parameter:** `horizontal_flip_prob` (probability 0.0–1.0)

**Best for:** Images where left-right orientation is irrelevant (photographs of objects, animals). **Not ideal for MNIST digits** — a flipped "6" looks like a "9" — but we apply it with low probability for regularization.

**Rust function:**
```rust
use rust_neural_networks::data::augmentation::random_horizontal_flip;

let mut rng = SimpleRng::new(42);
// 50% chance of flipping
random_horizontal_flip(&mut image, width, height, channels, 0.5, &mut rng);
```

---

### Random Crop

**What it does:** Pads the image with zeros, then randomly crops a region of the original size. This introduces **translation invariance** — the model learns that objects can appear at slightly different positions.

```
Step 1 - Pad:   32x32 image → 40x40 with zero border (padding=4)
Step 2 - Crop:  Randomly select a 32x32 window from the 40x40 padded image
Result:         32x32 image, slightly shifted
```

**Configuration parameter:** `random_crop_padding` (padding pixels per side, e.g., 2 or 4)

**Best for:** All image tasks. Random crop is one of the most effective augmentations for both MNIST and CIFAR-10.

**Rust function:**
```rust
use rust_neural_networks::data::augmentation::random_crop;

let mut rng = SimpleRng::new(42);
// Pad by 4 pixels each side, then crop back to original 32x32
let augmented = random_crop(&image, 32, 32, 3, 4, 32, 32, &mut rng);
```

---

### Brightness Jitter

**What it does:** Randomly adds a constant offset `δ ∈ [-max_delta, max_delta]` to all pixel values. Pixel values are clamped to `[0.0, 1.0]` to prevent overflow.

```
Mathematical definition:
  δ ~ Uniform(-max_delta, max_delta)
  output[i] = clamp(input[i] + δ, 0.0, 1.0)
```

**Configuration parameter:** `brightness_jitter` (typical range: 0.1–0.3)

**Best for:** Photographs where lighting conditions vary (CIFAR-10). Less useful for MNIST where lighting is controlled.

**Rust function:**
```rust
use rust_neural_networks::data::augmentation::random_brightness;

let mut rng = SimpleRng::new(42);
// Randomly adjust brightness by up to ±0.2
random_brightness(&mut image, width, height, channels, 0.2, &mut rng);
```

---

### Contrast Jitter

**What it does:** Scales pixel values around their mean by a random factor `f ∈ [1 - max_delta, 1 + max_delta]`. A factor of `1.0` leaves the image unchanged; values below `1.0` reduce contrast; values above `1.0` increase contrast.

```
Mathematical definition:
  mean = average(image)
  f ~ Uniform(1 - max_delta, 1 + max_delta)
  output[i] = clamp(mean + f * (input[i] - mean), 0.0, 1.0)
```

**Configuration parameter:** `contrast_jitter` (typical range: 0.1–0.5)

**Note:** Contrast jitter approximately **preserves the image mean** because the scaling is centered on the mean pixel value.

**Rust function:**
```rust
use rust_neural_networks::data::augmentation::random_contrast;

let mut rng = SimpleRng::new(42);
// Contrast factor in [0.8, 1.2]
random_contrast(&mut image, width, height, channels, 0.2, &mut rng);
```

---

### Saturation Jitter

**What it does:** Interpolates between the grayscale version and the original color image by a random factor `f ∈ [1 - max_delta, 1 + max_delta]`. Uses standard luminance weights `L = 0.299R + 0.587G + 0.114B`.

```
Mathematical definition:
  gray = 0.299 * R + 0.587 * G + 0.114 * B
  f ~ Uniform(1 - max_delta, 1 + max_delta)
  output_channel = clamp(gray + f * (channel - gray), 0.0, 1.0)

  f = 0.0 → fully grayscale
  f = 1.0 → original colors unchanged
  f > 1.0 → enhanced, more vivid colors
```

**Configuration parameter:** `saturation_jitter` (typical range: 0.1–0.5)

**Constraint:** Only applies to RGB images (3-channel). Will panic on single-channel (grayscale) images.

**Best for:** CIFAR-10 photographs where color intensity may vary.

**Rust function:**
```rust
use rust_neural_networks::data::augmentation::random_saturation;

let mut rng = SimpleRng::new(42);
// Saturation factor in [0.8, 1.2]
random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
```

---

## Enabling Augmentation via Config

Augmentation is controlled entirely through JSON configuration files. No code changes are required.

### Before: Default Config (No Augmentation)

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

Training without augmentation:
```
Epoch 1/10, Loss: 0.3421, Val Loss: 0.1823, Val Acc: 94.8%, Time: 2.1s
Epoch 2/10, Loss: 0.1502, Val Loss: 0.1421, Val Acc: 95.9%, Time: 2.0s
Epoch 3/10, Loss: 0.0987, Val Loss: 0.1389, Val Acc: 96.1%, Time: 2.0s
...
Final test accuracy: ~94-97%
```

### After: Augmented Config

Add the augmentation fields to your config:

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 2,
  "brightness_jitter": 0.1,
  "contrast_jitter": 0.1
}
```

Training with augmentation (training loss is slightly higher because each batch sees augmented/harder examples):
```
Epoch 1/10, Loss: 0.4102, Val Loss: 0.1710, Val Acc: 95.2%, Time: 2.3s
Epoch 2/10, Loss: 0.2341, Val Loss: 0.1350, Val Acc: 96.4%, Time: 2.3s
Epoch 3/10, Loss: 0.1821, Val Loss: 0.1289, Val Acc: 96.8%, Time: 2.3s
...
Final test accuracy: ~95-97% (improved generalization)
```

**Key observation:** Training loss is higher with augmentation (harder task), but validation/test accuracy is better because the model generalizes more robustly.

---

## Model-Specific Configs

Pre-built augmentation configs are provided for all three MNIST/CIFAR-10 models.

### MNIST CNN with Augmentation

**File:** `config/training/mnist_cnn_augmented.json`

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 3,
  "batch_size": 32,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 2,
  "brightness_jitter": 0.1,
  "contrast_jitter": 0.1
}
```

**Run it:**
```bash
cargo run --release --bin mnist_cnn -- config/training/mnist_cnn_augmented.json
```

**Notes:**
- Horizontal flip is applied conservatively (50% probability). Flipped MNIST digits may be ambiguous (e.g., "6"↔"9"), but the small probability adds useful regularization.
- Crop padding of 2 pixels provides mild translation invariance for 28×28 grayscale digits.
- Brightness and contrast jitter at 0.1 are mild — appropriate for the controlled lighting in MNIST.

---

### MNIST MLP with Augmentation

**File:** `config/training/mnist_mlp_augmented.json`

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 2,
  "brightness_jitter": 0.1,
  "contrast_jitter": 0.1
}
```

**Run it:**
```bash
cargo run --release --bin mnist_mlp -- config/training/mnist_mlp_augmented.json
```

**Notes:**
- The MLP (784→512→10) lacks convolutions and operates on flattened pixel vectors, so spatial augmentations (flip, crop) are applied to the 2D image before flattening.
- Augmentation may provide smaller gains for MLPs compared to CNNs, since MLPs don't have the spatial inductive biases that convolutions provide.

---

### CIFAR-10 CNN with Augmentation

For CIFAR-10, more aggressive augmentation is beneficial because photographs have much more variation than handwritten digits.

```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 1e-5,
  "T_max": 10,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 32,
  "validation_split": 0.1,
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.001,
  "enable_augmentation": true,
  "horizontal_flip_prob": 0.5,
  "random_crop_padding": 4,
  "brightness_jitter": 0.2,
  "contrast_jitter": 0.2,
  "saturation_jitter": 0.2
}
```

**Run it:**
```bash
cargo run --release --bin cifar10_cnn -- --config config/training/cifar10_cnn_default.json
```

**Key differences from MNIST configs:**
- `random_crop_padding: 4` — more padding for 32×32 images
- `brightness_jitter: 0.2` and `contrast_jitter: 0.2` — stronger color augmentation
- `saturation_jitter: 0.2` — saturation augmentation (CIFAR-10 only, requires 3 channels)
- Early stopping patience is higher (5) because CIFAR-10 takes longer to converge

---

## Expected Accuracy Improvements

Augmentation primarily benefits generalization — the gap between training accuracy and validation/test accuracy shrinks.

| Model | Without Augmentation | With Augmentation | Improvement |
|-------|---------------------|-------------------|-------------|
| MNIST MLP | 94–97% | 95–97% | +0.5–1% |
| MNIST CNN | 95–97% | 96–98% | +0.5–1% |
| CIFAR-10 CNN | 50–60% | 53–63% | +2–4% |

**Why CIFAR-10 benefits more:**
- Photographs have much more inherent variation (lighting, angles, backgrounds)
- Augmentation mimics this real-world variation during training
- The model learns to be robust to color and position changes

**Why improvement is modest for simple architectures:**
- The baseline MNIST architectures are already regularized by their small size
- CIFAR-10's baseline CNN is capacity-limited; deeper architectures benefit more from augmentation
- For maximum benefit, pair augmentation with deeper networks and more training epochs

**Important note on training loss:** With augmentation enabled, the training loss will appear higher than without augmentation. This is expected — each batch sees harder, transformed examples. The true metric is validation/test accuracy, which should be equal or better with augmentation.

---

## Implementation Details

### Where Augmentation Happens

Augmentation is applied **per sample during training batch construction**, not when loading data. This means:

1. Each epoch, every training image is independently transformed with a random seed derived from the global RNG
2. Validation and test sets are **never augmented** — they use the original images
3. The RNG state ensures reproducible results when using the same seed

### Data Flow

```
Raw training data
       ↓
  Load image (32×32×3 or 28×28×1)
       ↓
  [if enable_augmentation]
  ├── random_horizontal_flip (if horizontal_flip_prob > 0)
  ├── random_crop (if random_crop_padding > 0)
  ├── random_brightness (if brightness_jitter > 0)
  ├── random_contrast (if contrast_jitter > 0)
  └── random_saturation (if saturation_jitter > 0, RGB only)
       ↓
  Flattened pixel vector → model input
```

### Checking if Augmentation is Active

The training output includes a line confirming augmentation status:

```
Data augmentation: enabled
  Horizontal flip:  50.0%
  Random crop:      padding=2
  Brightness jitter: ±0.10
  Contrast jitter:  ±0.10
```

If augmentation is disabled (or `enable_augmentation` is absent/false):
```
Data augmentation: disabled
```

---

## Best Practices

**Do use augmentation when:**
- Training with limited data (fewer samples per class)
- Validation accuracy diverges from training accuracy (overfitting)
- Training CIFAR-10 or other natural image datasets
- Using deeper networks with more capacity to overfit

**Be cautious with:**
- **Horizontal flip on MNIST digits**: Some digits (2, 5, 6, 9) look like other digits when flipped. Use low probability (≤0.3) or disable flip for MNIST.
- **Saturation jitter on MNIST**: MNIST is single-channel grayscale — saturation adjustment requires 3 channels and will cause a panic if enabled for grayscale models. Saturation is automatically skipped for single-channel images.
- **Too-aggressive augmentation**: Very high `brightness_jitter` (> 0.4) or `contrast_jitter` (> 0.5) can distort images beyond recognition, making training unstable.
- **Augmenting the validation set**: Always ensure `enable_augmentation: true` only affects training data. The validation split is computed before any augmentation is applied.

**Recommended starting configs:**

| Dataset | Flip | Crop Padding | Brightness | Contrast | Saturation |
|---------|------|-------------|-----------|---------|-----------|
| MNIST (mild) | 0.3 | 2 | 0.1 | 0.1 | — |
| MNIST (aggressive) | 0.5 | 3 | 0.2 | 0.2 | — |
| CIFAR-10 (mild) | 0.5 | 2 | 0.1 | 0.1 | 0.1 |
| CIFAR-10 (standard) | 0.5 | 4 | 0.2 | 0.2 | 0.2 |
| CIFAR-10 (aggressive) | 0.5 | 6 | 0.3 | 0.4 | 0.4 |

**Starting point for a new experiment:**
1. Start with the default config (no augmentation) to get a baseline
2. Enable augmentation with conservative values (flip=0.5, crop_padding=2, jitter=0.1)
3. Compare validation accuracy curves using `plot_comparison.py`
4. If accuracy improves, try more aggressive augmentation
5. If training becomes unstable (loss spikes), reduce jitter values

**Plot training curves to compare:**
```bash
python plot_comparison.py logs/mnist_cnn_training.csv logs/mnist_cnn_augmented_training.csv
```
