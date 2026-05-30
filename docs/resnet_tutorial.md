# ResNet & Residual Learning: Theory and Implementation

## At a glance

- **Level:** Advanced (optional deep-dive)
- **Estimated time:** 45–90 minutes
- **Prerequisites:** Comfort with backprop/vanishing gradients; recommended after `docs/tutorials/03_mnist_cnn.md`
- **How to run (smoke check):**
  ```bash
  cargo test -q
  ```
- **Expected output/artifacts:** Test suite completes successfully; no artifacts expected for the smoke check

> **Learning Objectives**
>
> After reading this tutorial you will understand:
> - Why very deep neural networks fail to train without skip connections
> - What *residual learning* is and how it addresses the vanishing-gradient problem
> - How the `ResidualBlock` in this project is implemented and why every design choice was made
> - How gradients flow through skip connections during back-propagation
> - When to use identity vs. projection shortcuts
> - Practical tips for training ResNets from scratch

---

## Table of Contents

1. [The Vanishing Gradient Problem](#1-the-vanishing-gradient-problem)
2. [Residual Learning: F(x) + x Instead of H(x)](#2-residual-learning-fx--x-instead-of-hx)
3. [Why Skip Connections Help](#3-why-skip-connections-help)
4. [ResNet Architecture Overview](#4-resnet-architecture-overview)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Gradient Flow Analysis](#6-gradient-flow-analysis)
7. [Identity vs. Projection Shortcuts](#7-identity-vs-projection-shortcuts)
8. [Gradient Magnitude Visualization](#8-gradient-magnitude-visualization)
9. [Practical Training Tips](#9-practical-training-tips)
10. [Further Reading](#10-further-reading)

---

## 1. The Vanishing Gradient Problem

### What happens when you stack many layers?

Intuitively, deeper networks should learn richer representations.  For image
classification, early layers detect edges, middle layers detect shapes, and late
layers detect semantic concepts.  Stacking more layers should, in theory, give
more expressive power.

In practice, networks deeper than ~10–20 layers trained with vanilla SGD exhibit
two failure modes:

1. **Vanishing gradients** – Gradients shrink exponentially as they are
   back-propagated through layers.  By the time the gradient reaches the first
   few layers, it is essentially zero and those layers stop learning.

2. **Exploding gradients** – The opposite: gradients grow exponentially and
   training diverges.

Both arise from repeated multiplication of the Jacobians of each layer.

### The chain rule in a deep network

For a network with *L* layers and a scalar loss ℒ, the gradient with respect to
layer *k*'s parameters is:

```
∂ℒ/∂W_k  =  (∂ℒ/∂a_L) · (∂a_L/∂a_{L-1}) · ... · (∂a_{k+1}/∂a_k) · (∂a_k/∂W_k)
```

Each intermediate Jacobian `∂a_{i+1}/∂a_i` is the matrix of partial derivatives
through layer *i*.  If the singular values of these matrices are mostly < 1, the
product shrinks to near-zero.

### Empirical evidence (pre-ResNet)

In the 2015 ImageNet competition, plain (non-residual) networks with 56 layers
performed **worse** than networks with only 20 layers — not because of
overfitting, but because of poor optimisation.  This was called the *degradation
problem*:

```
Training error (plain network, CIFAR-10):

  Layers   Training Error   Test Error
  ------   --------------   ----------
    20           0.75%         8.75%
    56           1.92%        10.96%   ← deeper is WORSE
```

Batch normalisation and careful initialisation help, but are not sufficient for
networks much deeper than ~30 layers.

---

## 2. Residual Learning: F(x) + x Instead of H(x)

### The key insight

He et al. (2016) observed that if the optimal transformation is close to the
identity function, it is easier to learn a *residual* (a small correction) than
to learn the full identity from scratch.

**Traditional layer:** learn `H(x)` directly.

**Residual block:** let the network learn `F(x)` where `H(x) = F(x) + x`.
The network only needs to learn `F(x) = H(x) − x`, i.e. *how much to change x*.

If the optimal transformation **is** the identity, then `F(x) = 0` — and it is
much easier to push a function to zero than to make it replicate an identity
mapping.

### Mathematical formulation

For a residual block with input **x** and residual function **F**:

```
output = ReLU( F(x, {W_i}) + shortcut(x) )
```

Where:
- `F(x, {W_i})` is the main branch (two 3×3 conv layers in the basic block)
- `shortcut(x)` is either the identity or a linear projection W_s · x
- The element-wise addition combines both branches

This reformulation has a profound effect on gradients (see Section 6).

---

## 3. Why Skip Connections Help

### Reason 1: Gradient highway

Skip connections create a *direct path* for gradients to flow from the loss all
the way back to the earliest layers, bypassing the weight matrices of the
residual function.

Think of it as an express lane on a motorway: gradients can skip multiple layers
and arrive at earlier layers much stronger than in a plain network.

### Reason 2: Identity mapping as a lower bound

Even if the learned residual function `F(x)` produces garbage early in training,
the skip connection guarantees the block outputs at least `x` (before the final
ReLU).  The network can never do *worse* than the identity — it always has a
sensible baseline to build on.

### Reason 3: Ensemble-like behaviour

Recent work (Veit et al., 2016) shows that ResNets behave like an implicit
ensemble of networks of exponentially many different depths.  The skip
connections allow gradients to travel paths of varying lengths, giving the
optimiser multiple gradient signals.

### Visualising the effect

```
Plain 56-layer network gradient flow:

  Loss --> Layer 56 --> Layer 55 --> ... --> Layer 2 --> Layer 1
            |               |                   |           |
           0.9x            0.9x       ...       ~0          ~0
           (each layer attenuates the gradient)

ResNet 56-layer gradient flow:

  Loss --> Block 28 +----> Block 27 +----> ... +----> Block 1
            |        |       |        |              |
            F_28(x)  x      F_27(x)  x              F_1(x)
                   ↑ skip        ↑ skip           ↑ skip
           (gradient also flows via skip, much stronger)
```

---

## 4. ResNet Architecture Overview

### Original ResNet-18

The original ResNet-18 (for ImageNet, 224×224 input) consists of:

```
Input (224×224×3)
  │
  ▼
Conv(7×7, 64 filters, stride=2)  →  112×112×64
  │
BatchNorm → ReLU
  │
MaxPool(3×3, stride=2)           →  56×56×64
  │
  ▼
Stage 1: 2 × ResBlock(64→64,   stride=1)  →  56×56×64
Stage 2: 2 × ResBlock(64→128,  stride=2,1) →  28×28×128
Stage 3: 2 × ResBlock(128→256, stride=2,1) →  14×14×256
Stage 4: 2 × ResBlock(256→512, stride=2,1) →   7×7×512
  │
GlobalAvgPool                              →  512
  │
Dense(512→1000)                            →  1000 (ImageNet classes)
```

### CIFAR-10 variant (this project)

For CIFAR-10 (32×32 input), the large 7×7 initial convolution and max-pool are
replaced with a single 3×3 conv (the *stem*), and channels are scaled down:

```
Input (32×32×3)
  │
  ▼  Stem
Conv(3×3, 16 filters, padding=1, stride=1)  →  32×32×16
BatchNorm → ReLU
  │
  ▼  Stage 1  (spatial: 32×32)
ResBlock(16→16, stride=1)
ResBlock(16→16, stride=1)
  │
  ▼  Stage 2  (spatial: 32→16)
ResBlock(16→32, stride=2)   ← projection shortcut, halves spatial dims
ResBlock(32→32, stride=1)
  │
  ▼  Stage 3  (spatial: 16→8)
ResBlock(32→64, stride=2)   ← projection shortcut
ResBlock(64→64, stride=1)
  │
  ▼  Stage 4  (spatial: 8→4)
ResBlock(64→128, stride=2)  ← projection shortcut
ResBlock(128→128, stride=1)
  │
  ▼  Head
GlobalAvgPool(4×4→1×1)      →  128
Dense(128→10)               →  10 (CIFAR-10 classes)
```

**Parameter counts (approximate):**
- Stem Conv:  16×3×3×3 = 432
- Stage 1 (2 identity blocks):  ~2 × 2×(16×16×9) = ~9,216
- Stage 2 (1 proj + 1 identity): ~2 × 2×(32×32×9) + shortcut = ~18,560
- Stage 3: ~2 × 2×(64×64×9) + shortcut = ~74,240
- Stage 4: ~2 × 2×(128×128×9) + shortcut = ~296,960
- Dense head: 128×10 = 1,280
- **Total: ~11 million parameters** (after including batch norm params)

---

## 5. Implementation Walkthrough

### ResidualBlock struct

```rust
pub struct ResidualBlock {
    // Dimension bookkeeping
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,

    // Main branch: two conv-BN pairs
    conv1: Conv2DLayer,   // 3×3, stride=stride
    bn1:   BatchNormLayer,
    conv2: Conv2DLayer,   // 3×3, stride=1
    bn2:   BatchNormLayer,

    // Optional projection shortcut (None = identity)
    shortcut_conv: Option<Conv2DLayer>,   // 1×1 conv
    shortcut_bn:   Option<BatchNormLayer>,

    // Cached activations for backward (interior mutability)
    cached_input:          RefCell<Vec<f32>>,
    cached_hidden:         RefCell<Vec<f32>>,  // after conv1→bn1→ReLU
    cached_after_conv2_bn2: RefCell<Vec<f32>>,
    cached_shortcut:       RefCell<Vec<f32>>,
    cached_batch_size:     RefCell<usize>,
}
```

**Why `RefCell`?**  The `Layer` trait requires `forward()` to take `&self`
(shared reference) so that multiple blocks can share the same trait object
machinery.  But caching activations requires mutation.  `RefCell` provides
*interior mutability* — it enforces Rust's borrow rules at *runtime* instead of
compile time, which is the standard pattern for stateful layer objects in this
codebase.  See `src/layers/batchnorm.rs` and `src/layers/transformer.rs` for
the same pattern.

### Constructor: selecting the shortcut type

```rust
let needs_projection = in_channels != out_channels || stride != 1;
let (shortcut_conv, shortcut_bn) = if needs_projection {
    // 1×1 conv to project input dimensions to match output
    let sc  = Conv2DLayer::new(in_channels, out_channels, 1, 0, stride, ...);
    let sbn = BatchNormLayer::new(out_size, 1e-5, 0.9);
    (Some(sc), Some(sbn))
} else {
    (None, None)   // Identity: no extra parameters
};
```

The rule is simple:
- **Identity shortcut**: `in_channels == out_channels && stride == 1`
  → the input can be added directly to the main branch output.
- **Projection shortcut**: channels differ or stride > 1
  → a 1×1 conv (+ BN) resizes the input to match.

### Forward pass

```
input
  │
  ├─── Main branch ──────────────────────────────────────────┐
  │    conv1(3×3, stride) → bn1 → ReLU → conv2(3×3) → bn2   │
  │                                                           │
  ├─── Shortcut branch ────────────────────────────────────── │
  │    Identity: pass through directly (if same dims)         │
  │    Projection: 1×1 conv → bn                             │
  │                                                           ▼
  └──────────────────────────────────────────────── (+) → ReLU → output
```

In Rust (simplified):
```rust
fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
    // 1. Cache input for backward
    self.cached_input.borrow_mut().copy_from_slice(input);

    // 2. Main branch
    let mut hidden = vec![0.0f32; batch_size * out_size];
    self.conv1.forward(input, &mut conv1_out, batch_size);
    self.bn1.forward(&conv1_out, &mut hidden, batch_size);
    relu_inplace(&mut hidden);                    // ← ReLU after bn1

    let mut main_out = vec![0.0f32; batch_size * out_size];
    self.conv2.forward(&hidden, &mut conv2_out, batch_size);
    self.bn2.forward(&conv2_out, &mut main_out, batch_size);

    // 3. Shortcut branch
    let shortcut_out = match (&self.shortcut_conv, &self.shortcut_bn) {
        (Some(sc), Some(sbn)) => { /* 1×1 conv → BN */ }
        _                      => input.to_vec(),   // identity
    };

    // 4. Add + final ReLU
    for i in 0..output.len() { output[i] = main_out[i] + shortcut_out[i]; }
    relu_inplace(output);
}
```

### Backward pass structure

The backward pass must correctly implement the chain rule through the branching
structure.  This is covered in depth in Section 6.

---

## 6. Gradient Flow Analysis

### The add junction is the key

At the heart of residual learning is the element-wise addition:

```
pre_relu2 = F(x) + shortcut(x)
```

The gradient of a sum with respect to both inputs is just the gradient itself:

```
∂ℒ/∂F(x)        = ∂ℒ/∂pre_relu2
∂ℒ/∂shortcut(x) = ∂ℒ/∂pre_relu2
```

Both branches receive **identical gradients**.  There is no attenuation at the
junction.

### Full backward pass diagram

```
                    ┌─────────────────────────────────────────┐
                    │              FORWARD PASS                │
                    │                                          │
    input ──────────┼──► conv1 ──► bn1 ──► ReLU ──► conv2 ──► bn2 ──┐
           │        │                                                  │
           │        │                                            add ──┼──► ReLU ──► output
           │        │                                                  │
           └────────┼──► [identity or 1×1 conv → BN] ────────────────┘
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │             BACKWARD PASS                │
                    │                                          │
  grad_input ◄──────┼──  conv1_bp ◄── bn1_bp ◄── ReLU_bp ◄── conv2_bp ◄── bn2_bp ◄──┐
           ▲        │                                                                   │
           │        │                                                      grad_add ───┤
           │        │                                                 (split, same val) │
           └────────┼── [identity passthrough or shortcut_bn_bp ◄── shortcut_conv_bp] ┘
                    └─────────────────────────────────────────┘
```

### Step-by-step gradient computation

**Step 1: ReLU2 backward**

The final ReLU passes the gradient only where the pre-activation was positive:

```rust
for i in 0..batch_size * out_size {
    let pre_relu2 = cached_after_conv2_bn2[i] + cached_shortcut[i];
    grad_pre_relu2[i] = if pre_relu2 > 0.0 { grad_output[i] } else { 0.0 };
}
```

**Step 2: Gradient split at the add junction**

Since `pre_relu2 = main_out + shortcut_out`, gradients are identical for both:

```rust
let grad_add = &grad_pre_relu2;   // same for both branches
```

**Step 3: Main branch (backward through bn2 → conv2 → ReLU1 → bn1 → conv1)**

```rust
self.bn2.backward(&cached_after_conv2_bn2, grad_add, &mut grad_conv2_out, batch_size);
self.conv2.backward(&cached_hidden, &grad_conv2_out, &mut grad_hidden, batch_size);

// ReLU1 backward: zero gradient where cached_hidden was zeroed
for i in 0..batch_size * out_size {
    if cached_hidden[i] <= 0.0 { grad_hidden[i] = 0.0; }
}

self.bn1.backward(&dummy, &grad_hidden, &mut grad_conv1_out, batch_size);
self.conv1.backward(&cached_input, &grad_conv1_out, &mut grad_input_main, batch_size);
```

**Step 4: Shortcut branch backward**

- *Identity*: gradient flows unchanged (no parameters, no attenuation).
- *Projection*: gradient flows through BN then 1×1 conv.

```rust
match (&self.shortcut_conv, &self.shortcut_bn) {
    (Some(sc), Some(sbn)) => {
        sbn.backward(&dummy, grad_add, &mut grad_sc_out, batch_size);
        sc.backward(&cached_input, &grad_sc_out, &mut grad_input_shortcut, batch_size);
    }
    _ => {
        // Identity: gradient passes straight through
        grad_input_shortcut.copy_from_slice(grad_add);
    }
}
```

**Step 5: Sum gradients**

Both branches contribute to the gradient at the block input:

```rust
for i in 0..batch_size * in_size {
    grad_input[i] = grad_input_main[i] + grad_input_shortcut[i];
}
```

### Why this prevents vanishing gradients

For the identity shortcut, the gradient flowing into the block is at minimum:

```
grad_input ≥ grad_input_shortcut = grad_add
```

The shortcut branch carries the full gradient without going through *any* weight
matrix.  No matter how many residual blocks are stacked, the gradient from the
loss can reach the first block relatively intact through these shortcut paths.

Formally, for a network with *L* residual blocks with identity shortcuts
(He et al. 2016 Eq. 5):

```
∂ℒ/∂x_k  =  ∂ℒ/∂x_L · (1 + ∑ ∂F_i/∂x_k)
              ^^^^^^^^
              This term flows through ALL shortcut paths
              without any multiplicative attenuation
```

The `1` is the key: even if all the residual gradients `∂F_i/∂x_k` vanish,
the gradient still reaches layer *k* intact.

---

## 7. Identity vs. Projection Shortcuts

### Decision rule

```
                             ┌─────────────────────────┐
                             │  in_channels == out_channels  │
                             │       AND stride == 1         │
                             └─────────────────────────┘
                                         │
                         ┌───────────────┴──────────────┐
                        YES                             NO
                         │                               │
                         ▼                               ▼
                  IDENTITY SHORTCUT              PROJECTION SHORTCUT
                  ─────────────────              ──────────────────
                  input ──────────►              input ──► 1×1 Conv ──► BN
                  (no parameters)                (adds parameters to match dims)
```

### When each is used in the CIFAR-10 ResNet

| Stage | Block | in_channels | out_channels | stride | Shortcut type     |
|-------|-------|-------------|--------------|--------|-------------------|
| 1     | 1     | 16          | 16           | 1      | Identity          |
| 1     | 2     | 16          | 16           | 1      | Identity          |
| 2     | 1     | 16          | 32           | 2      | **Projection**    |
| 2     | 2     | 32          | 32           | 1      | Identity          |
| 3     | 1     | 32          | 64           | 2      | **Projection**    |
| 3     | 2     | 64          | 64           | 1      | Identity          |
| 4     | 1     | 64          | 128          | 2      | **Projection**    |
| 4     | 2     | 128         | 128          | 1      | Identity          |

### Why 1×1 convolutions for projection?

A 1×1 convolution with `out_channels` filters and `stride=2` can:

1. **Change the number of channels**: `in_channels → out_channels`
2. **Downsample spatially**: stride=2 halves H and W

It is parameter-efficient (no spatial mixing, just channel mixing) and allows
the shortcut to exactly match the main branch output dimensions.

### Alternative: zero-padding

Some ResNet variants pad the shortcut with zeros instead of using a 1×1 conv.
This has zero extra parameters but the new channels receive no gradient signal
through the shortcut, which can slow convergence.  The projection shortcut is
recommended for best accuracy.

---

## 8. Gradient Magnitude Visualization

### Conceptual comparison

The following shows a *conceptual* gradient magnitude profile through layers in
a 56-layer network (not real numbers, but representative of typical behaviour):

```
Plain 56-layer network (no skip connections):

  Layer:   56  52  48  44  40  36  32  28  24  20  16  12   8   4   1
  │grad│:  1.0 0.6 0.4 0.2 0.1 0.05 0.02 0.01 ...  ~0  ~0  ~0  ~0  ~0  ~0

  ████████
  ██████
  █████
  ████
  ██
  █
  ░                            ← gradients have died out
  ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░

  Early layers receive no useful gradient → no learning

ResNet-56 (with skip connections):

  Layer:   56  52  48  44  40  36  32  28  24  20  16  12   8   4   1
  │grad│:  1.0 0.8 0.7 0.6 0.7 0.5 0.6 0.5 0.4 0.5 0.4 0.4 0.3 0.4 0.3

  ████████
  ██████████
  ██████████
  ██████████
  ████████
  ████████
  ████████
  ██████
  ██████
  ████████

  Gradients remain substantial all the way to the first layers
  (oscillations due to the combination of main-branch + shortcut paths)
```

### Monitoring gradient health in practice

To monitor gradient flow during training, you can log the L2 norm of gradients
at each block.  In Python:

```python
for epoch in range(num_epochs):
    # After backward pass, log gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
```

In this Rust implementation, the CSV logs capture training/validation loss which
indirectly reflects gradient health: if training loss stagnates from epoch 1,
gradients are likely vanishing.

### Signs of gradient problems

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Training loss doesn't decrease from epoch 1 | Vanishing gradients | Add skip connections, check LR |
| Loss jumps to NaN immediately | Exploding gradients | Reduce LR, add gradient clipping |
| Training loss decreases but validation plateaus | Overfitting | Add weight decay, dropout |
| Very slow initial convergence | Dying ReLU | Use LeakyReLU, check initialisation |

---

## 9. Practical Training Tips

### Learning rate

ResNets with AdamW typically train well with `lr = 0.001`.  With SGD + momentum,
start with `lr = 0.1` and use step-decay (divide by 10 at epochs 30 and 45 for
a 50-epoch run).

The default config for this project uses cosine annealing:

```json
{
    "optimizer_type": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler_type": "cosine_annealing",
    "T_max": 50,
    "min_lr": 0.00001,
    "epochs": 50,
    "batch_size": 64
}
```

### Batch normalisation mode

**Always** call `set_training(true)` before training and `set_training(false)`
before evaluation.  The `ResidualBlock` exposes `set_training()` which
propagates to all internal `BatchNormLayer` instances:

```rust
// Before training loop
for layer in &mut model.layers {
    if let Some(block) = layer.as_any_mut().downcast_mut::<ResidualBlock>() {
        block.set_training(true);
    }
}

// Before validation loop
for layer in &mut model.layers {
    if let Some(block) = layer.as_any_mut().downcast_mut::<ResidualBlock>() {
        block.set_training(false);
    }
}
```

Forgetting this is one of the most common bugs.  In training mode, batch
statistics are used for normalisation; in inference mode, the running
(exponential moving average) statistics are used.  Using batch statistics
during validation on small batches gives noisy, unreliable metrics.

### Weight initialisation

This implementation uses Xavier uniform initialisation (via `Conv2DLayer`).
For layers before ReLU, He (Kaiming) initialisation is theoretically better
(`std = sqrt(2/fan_in)`), but Xavier is a good approximation for moderate
network depths and still trains well in practice.

### Data augmentation

CIFAR-10 benefits significantly from augmentation.  Even without modifying
the data loader, training accuracy > test accuracy by 5–10% indicates
overfitting that augmentation can address.  Standard augmentations:

- Random horizontal flip (50% probability)
- Random crop: pad by 4 pixels, then crop 32×32
- Cutout / random erasing

### Early stopping

The default config uses `early_stopping_patience: 10`.  ResNets often take
longer than shallow networks to begin improving, so a patience of 5–10 epochs
is appropriate.  Watch validation loss rather than training loss.

### Expected accuracy on CIFAR-10

With this CIFAR-10 ResNet-18 variant and the default config (no augmentation):

| Epochs | Validation Accuracy |
|--------|---------------------|
| 10     | ~65–70%             |
| 25     | ~75–80%             |
| 50     | ~82–86%             |

With data augmentation, 90%+ is achievable.

---

## 10. Further Reading

### Papers

- **He et al. (2016)** — "Deep Residual Learning for Image Recognition"
  The original ResNet paper. Introduces identity mappings and demonstrates
  training of 152-layer networks.
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- **He et al. (2016b)** — "Identity Mappings in Deep Residual Networks"
  Explores alternative placement of BN and ReLU (pre-activation ResNet).
  Shows that full pre-activation improves results for very deep networks (>1000 layers).
  [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

- **Veit et al. (2016)** — "Residual Networks Behave Like Ensembles of
  Relatively Shallow Networks"
  Explains why ResNets work through an unravelled view showing exponentially
  many paths from input to output.
  [arXiv:1605.06431](https://arxiv.org/abs/1605.06431)

- **Huang et al. (2017)** — "Densely Connected Convolutional Networks" (DenseNet)
  Extends the skip-connection idea: every layer connects to every subsequent layer.
  [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)

### Related implementations in this project

| File | Description |
|------|-------------|
| `src/layers/residual.rs` | `ResidualBlock` implementation |
| `src/layers/pooling.rs` | `GlobalAvgPoolLayer` used at the head |
| `src/layers/batchnorm.rs` | `BatchNormLayer` used inside each block |
| `src/bin/resnet_cifar10.rs` | Full training script |
| `config/training/resnet_cifar10_default.json` | Default training config |
| `tests/test_residual_block.rs` | Integration tests for `ResidualBlock` |
| `tests/test_pooling.rs` | Integration tests for `GlobalAvgPoolLayer` |

---

*This tutorial accompanies the Rust Neural Networks educational project.
All code examples are from the actual implementation — see the source files for
the complete, production-quality versions.*
