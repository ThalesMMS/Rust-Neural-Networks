# Layer Normalization Layer Mathematics

This document provides a comprehensive explanation of the mathematics behind the Layer Normalization (LayerNorm) layer, covering both forward and backward propagation with full derivations, including the key distinction from batch normalization and numerical stability considerations.

## Table of Contents

- [Overview](#overview)
- [Layer Norm vs Batch Norm](#layer-norm-vs-batch-norm)
- [Forward Pass](#forward-pass)
  - [Mathematical Definition](#mathematical-definition)
  - [Dimension Analysis](#dimension-analysis)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass](#backward-pass)
  - [Gradient Flow Visualization](#gradient-flow-visualization)
  - [Gradient Computation](#gradient-computation)
  - [Chain Rule Application](#chain-rule-application)
  - [Gradient Formulas](#gradient-formulas)
  - [Implementation Notes](#implementation-notes)
- [No Training vs Inference Mode Difference](#no-training-vs-inference-mode-difference)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Numerical Considerations](#numerical-considerations)

## Overview

Layer Normalization (Ba, Kiros & Hinton, 2016) normalizes each sample across its feature dimension to have zero mean and unit variance, then applies learnable scale (γ) and shift (β) parameters. Unlike Batch Normalization, it operates per-sample and requires no batch statistics.

**Key characteristics:**
- Normalizes per-sample statistics across the feature dimension
- Two learnable parameters per feature: scale γ and shift β
- No running statistics — behavior is identical in training and inference
- Preserves input/output dimensionality (no shape change)
- Works correctly with any batch size, including batch_size=1

**Parameters:**
- `γ` (gamma): Learnable scale vector of shape (size,) — initialized to 1.0
- `β` (beta): Learnable shift vector of shape (size,) — initialized to 0.0
- `ε` (epsilon): Small constant for numerical stability (typical: 1e-5)

## Layer Norm vs Batch Norm

This is the most important conceptual distinction to understand.

| Aspect | Batch Normalization | Layer Normalization |
|--------|--------------------|--------------------|
| Normalizes across | Batch dimension (per feature) | Feature dimension (per sample) |
| Mean/variance shape | (D,) — one per feature | (B,) — one per sample |
| Batch-size sensitivity | Unstable with small batches | Works with any batch size |
| Training/inference difference | Yes (batch vs running stats) | **No** (always same computation) |
| Running statistics | Yes (mean and variance EMA) | **No** |
| Best for | CNNs, large batches | RNNs, Transformers, any sequence |
| Behavior with B=1 | Collapses (variance=0) | Works correctly |

**BatchNorm** computes statistics across the batch for each feature:
```
μ_j = (1/B) × Σᵢ x_ij    ← average over samples, shape: (D,)
σ²_j = (1/B) × Σᵢ (x_ij - μ_j)²
```

**LayerNorm** computes statistics across features for each sample:
```
μ_i = (1/D) × Σⱼ x_ij    ← average over features, shape: (B,)
σ²_i = (1/D) × Σⱼ (x_ij - μ_i)²
```

**Intuition:** In a batch of sentences being processed by a transformer, BatchNorm would mix statistics across different sentences (problematic for variable-length sequences). LayerNorm normalizes each sentence's representation independently, making it position- and batch-agnostic.

## Forward Pass

### Mathematical Definition

The forward pass of a layer normalization layer consists of five steps:

**Step 1: Compute per-sample mean across features**

```
μ_i = (1/D) × Σⱼ x_ij    for j = 1 to D
```

**Step 2: Compute per-sample variance across features**

```
σ²_i = (1/D) × Σⱼ (x_ij - μ_i)²    for j = 1 to D
```

**Step 3: Compute standard deviation (with numerical stability)**

```
std_i = sqrt(σ²_i + ε)
```

**Step 4: Normalize**

```
x̂_ij = (x_ij - μ_i) / std_i
```

**Step 5: Scale and shift (affine transformation)**

```
y_ij = γ_j × x̂_ij + β_j
```

Where:
- **x**: Input matrix of shape (batch_size, size)
- **y**: Output matrix of shape (batch_size, size)
- **D**: size (number of features per sample)
- **B**: batch_size (number of samples)
- **j**: Feature index, ranging from 0 to D-1
- **i**: Sample index, ranging from 0 to B-1
- **μ_i**: Mean of sample i across all features, shape (B,)
- **σ²_i**: Variance of sample i across all features, shape (B,)
- **std_i**: Standard deviation of sample i, shape (B,)
- **x̂_ij**: Normalized value (zero mean, unit variance per sample), shape (B, D)
- **γ_j**: Learnable scale parameter for feature j, shape (D,)
- **β_j**: Learnable shift parameter for feature j, shape (D,)
- **ε**: Small constant for numerical stability

**Combined forward pass formula:**

```
y_ij = γ_j × ((x_ij - μ_i) / sqrt(σ²_i + ε)) + β_j
```

**Intuition:** After normalization, x̂ has zero mean and variance 1 per sample across features. The γ and β parameters are shared across all samples but are feature-specific — they allow the network to represent any distribution per feature regardless of the input sample statistics.

### Dimension Analysis

**Single Sample (B=1, common in inference):**
- Input: (1, D) where D = size (number of features)
- Statistics: μ, σ², std are scalars (one per sample)
- Normalized: (1, D)
- Output: (1, D)
- Note: Layer norm works correctly here, unlike BatchNorm!

**Batched Computation (standard):**
- Input x: (B, D) where B = batch_size, D = size
- Sample mean μ: (B,) — one mean per sample (not per feature)
- Sample variance σ²: (B,) — one variance per sample
- Standard deviation std: (B,)
- Normalized x̂: (B, D)
- Gamma γ: (D,) — broadcasted across batch
- Beta β: (D,) — broadcasted across batch
- Output y: (B, D)

**Operation breakdown:**

```
Step 1: Mean computation (sum over features, not batch)
  μ_i = sum(x[i, :]) / D    → (B,)

Step 2: Variance computation (sum over features, not batch)
  σ²_i = sum((x[i, :] - μ_i)²) / D    → (B,)

Step 3: Normalize
  x̂ = (x - μ[:, None]) / std[:, None]    → (B, D)

Step 4: Scale and shift
  y = γ[None, :] × x̂ + β[None, :]    → (B, D)
```

**Example with concrete dimensions:**

For a layer norm with size=512 and batch_size=32:
- Input x: (32, 512)
- Sample mean μ: (32,) — sum along feature axis divided by 512
- Sample variance σ²: (32,)
- Standard deviation: (32,)
- Normalized x̂: (32, 512)
- Scale γ: (512,) — broadcasted to (32, 512)
- Shift β: (512,) — broadcasted to (32, 512)
- Output y: (32, 512)
- Parameters: 512 × 2 = 1,024 (gamma + beta)

**Compare to BatchNorm with same dimensions:**
- BatchNorm mean μ: (512,) — per feature
- LayerNorm mean μ: (32,) — per sample

### Rust Implementation (Forward Pass)

```rust
// Allocate storage for per-sample statistics
let mut sample_mean = vec![0.0f32; batch_size];
let mut sample_var = vec![0.0f32; batch_size];
let mut sample_std = vec![0.0f32; batch_size];
let mut normalized = vec![0.0f32; total_size];

// Step 1: Compute mean for each sample across features
// μ_i = (1/D) × Σⱼ x_ij
for i in 0..batch_size {
    let mut sum = 0.0f32;
    for j in 0..self.size {
        sum += input[i * self.size + j];
    }
    sample_mean[i] = sum / self.size as f32;
}

// Step 2: Compute variance for each sample across features
// σ²_i = (1/D) × Σⱼ (x_ij - μ_i)²
for i in 0..batch_size {
    let mut sum_sq = 0.0f32;
    for j in 0..self.size {
        let diff = input[i * self.size + j] - sample_mean[i];
        sum_sq += diff * diff;
    }
    sample_var[i] = sum_sq / self.size as f32;
    sample_std[i] = (sample_var[i] + self.epsilon).sqrt();
}

// Steps 4+5: Normalize, scale, and shift
// x̂_ij = (x_ij - μ_i) / std_i
// y_ij = γ_j × x̂_ij + β_j
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        normalized[idx] = (input[idx] - sample_mean[i]) / sample_std[i];
        output[idx] = self.gamma[j] * normalized[idx] + self.beta[j];
    }
}

// Cache values needed for the backward pass
*self.cached_input.borrow_mut() = input.to_vec();
*self.cached_mean.borrow_mut() = sample_mean;
*self.cached_var.borrow_mut() = sample_var;
*self.cached_std.borrow_mut() = sample_std;
*self.cached_normalized.borrow_mut() = normalized;
```

**Key caching:** The forward pass saves sample_mean, sample_var, normalized (x̂), and std. These are essential for the backward pass, which must differentiate through the normalization statistics. Unlike BatchNorm, these statistics depend only on a single sample, simplifying the backward pass.

### Computational Complexity

**Time Complexity:**

```
Step 1 (mean):     O(B × D)
Step 2 (variance): O(B × D)
Step 3 (std):      O(B)
Step 4+5 (norm):   O(B × D)

Total: O(B × D)
```

Both LayerNorm and BatchNorm have the same asymptotic complexity O(B × D). However, LayerNorm has no EMA update step, making it slightly faster in practice.

**Space Complexity:**

```
Learnable parameters: O(D)     — γ and β
Cached values:        O(B × D) — normalized, plus O(B) for mean/var/std
Running statistics:   None     — not needed!
```

**Example calculation:**

For layer norm with size=512 and batch_size=32:
- Operations: ~4 × 32 × 512 ≈ 65K FLOPs
- Parameters: 512 × 2 = 1,024 floats ≈ 4 KB
- Cache memory: 32 × 512 × 4 bytes ≈ 64 KB (normalized) + 32 × 3 × 4 bytes ≈ 0.4 KB (stats)
- No running statistics memory required!

## Backward Pass

### Gradient Flow Visualization

The following diagram illustrates how gradients flow backward through the layer normalization layer:

```
                      FORWARD PASS  |  BACKWARD PASS
                                    |
      Input x                       |                  ∂L/∂x
      (B, D)                        |                  (B, D)
         │                          |                     ▲
         │                          |                     │
         ▼                          |                     │
    ┌─────────┐                    |               ┌─────┴──────┐
    │ compute │──────── μ ─────────┼───────────────┤  ∂L/∂μ     │
    │  mean μ │         (B,)        |    ∂L/∂μ      │  contribution│
    └────┬────┘                     |    (B,)       └─────┬──────┘
         │                          |                     │
         ▼                          |                     │
    ┌─────────┐                    |               ┌─────┴──────┐
    │ compute │──────── σ² ────────┼───────────────┤  ∂L/∂σ²    │
    │  var σ² │         (B,)        |    ∂L/∂σ²     │  contribution│
    └────┬────┘                     |    (B,)       └─────┬──────┘
         │                          |                     │
         ▼                          |                     │
    ┌─────────┐                    |               ┌─────┴──────┐
    │normalize│◄─── x, μ, std ──── |               │  ∂L/∂x̂ /  │
    │ x̂=...  │                     |               │  std       │
    └────┬────┘                     |               └─────┬──────┘
         │                          |                     ▲
         │ x̂ (cached)              |                     │
         ▼                          |                ∂L/∂x̂ = ∂L/∂y ⊙ γ
    ┌─────────┐                    |               (B, D)
    │ y = γ⊙x̂│◄──────── γ ────────┼────────────────────────────────
    │    + β  │         (D,)        |    ∂L/∂γ = Σᵢ(∂L/∂y ⊙ x̂)/B
    └────┬────┘                     |    ∂L/∂β = Σᵢ(∂L/∂y)/B
         │                          |
         ▼                          |
      Output y                      |          ∂L/∂y (given)
      (B, D)                        |          (B, D)

Legend:
  ──►   Forward data flow
  ──▲   Backward gradient flow
  ⊙     Element-wise multiplication
  Σᵢ    Sum over batch dimension (axis 0)
  μ, σ² shape: (B,) — one per sample (contrast with BatchNorm's (D,))
```

**Key insight compared to BatchNorm:** In LayerNorm, each sample is normalized independently. Therefore, the gradient for a single input sample depends only on *that sample's* mean and variance — not on any other sample. This decouples the samples during backpropagation and simplifies the computation.

### Gradient Computation

During backpropagation, we receive the gradient of the loss with respect to the layer's output (∂L/∂y) and must compute:

1. **Gradient w.r.t. input (∂L/∂x)**: Needed to propagate error to previous layer
2. **Gradient w.r.t. γ (∂L/∂γ)**: Needed to update scale parameters
3. **Gradient w.r.t. β (∂L/∂β)**: Needed to update shift parameters

### Chain Rule Application

The forward pass computes y through a chain of operations:

```
x → μ, σ² → std → x̂ → y
```

For LayerNorm, this chain is applied per sample (index i), then the γ/β parameters aggregate across all samples.

**Level 1: Gradient through the affine transform (y = γ ⊙ x̂ + β)**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂y_ij/∂γ_j = x̂_ij
∂y_ij/∂β_j = 1
∂y_ij/∂x̂_ij = γ_j
```

**Level 2: Gradient through normalization (x̂ = (x - μ) / std)**

Since x̂_ij = (x_ij - μ_i) / std_i:

```
∂x̂_ij/∂x_ij = 1 / std_i
∂x̂_ij/∂μ_i  = -1 / std_i
∂x̂_ij/∂σ²_i = (x_ij - μ_i) × (-1/2) × (σ²_i + ε)^(-3/2)
             = -x̂_ij / (2 × std_i)
```

**Level 3: Gradient through variance (σ² = (1/D) × Σ(x - μ)²)**

Since μ_i is constant with respect to σ²_i:

```
∂σ²_i/∂x_ij = (2/D) × (x_ij - μ_i)
```

**Level 4: Gradient through mean (μ = (1/D) × Σ x)**

```
∂μ_i/∂x_ij = 1/D
```

Note: The mean also contributes to the variance, creating a secondary path. However, by definition, Σⱼ(x_ij - μ_i) = 0 for each sample, so this indirect contribution is zero in exact arithmetic (and near-zero in finite precision).

### Gradient Formulas

#### 1. Gradient w.r.t. Beta (∂L/∂β) — Simplest

**Formula:**

```
∂L/∂β_j = (1/B) × Σᵢ (∂L/∂y_ij)
```

**Derivation:**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂L/∂β_j = Σᵢ (∂L/∂y_ij × ∂y_ij/∂β_j)
         = Σᵢ (∂L/∂y_ij × 1)
         = Σᵢ ∂L/∂y_ij
```

With the (1/B) scale factor for gradient normalization:

```
∂L/∂β_j = (1/B) × Σᵢ ∂L/∂y_ij
```

**Dimensions:** (D,) = sum over batch of (B, D)

**Implementation:**

```rust
let scale = 1.0 / batch_size as f32;
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        grad_beta[j] += grad_output[idx] * scale;  // ∂L/∂β_j += (1/B) × ∂L/∂y_ij
    }
}
```

#### 2. Gradient w.r.t. Gamma (∂L/∂γ)

**Formula:**

```
∂L/∂γ_j = (1/B) × Σᵢ (∂L/∂y_ij × x̂_ij)
```

**Derivation:**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂L/∂γ_j = Σᵢ (∂L/∂y_ij × ∂y_ij/∂γ_j)
         = Σᵢ (∂L/∂y_ij × x̂_ij)
```

With the (1/B) scale factor:

```
∂L/∂γ_j = (1/B) × Σᵢ (∂L/∂y_ij × x̂_ij)
```

**Dimensions:** (D,) = element-wise sum over batch of (B, D) ⊙ (B, D)

**Implementation:**

```rust
let scale = 1.0 / batch_size as f32;
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        grad_gamma[j] += grad_output[idx] * normalized[idx] * scale;  // ∂L/∂y_ij × x̂_ij
    }
}
```

Note: `normalized[idx]` is the cached x̂_ij from the forward pass.

#### 3. Gradient w.r.t. Input (∂L/∂x) — Per-Sample Computation

Unlike BatchNorm where every sample interacts through shared batch statistics, LayerNorm allows computing ∂L/∂x_i independently for each sample i. For each sample, three paths contribute:

**Path A:** Direct path through normalization
**Path B:** Through variance σ²_i
**Path C:** Through mean μ_i

**Step 3a: Gradient w.r.t. normalized values (∂L/∂x̂_i)**

For sample i:

```
∂L/∂x̂_ij = ∂L/∂y_ij × γ_j
```

**Dimensions:** (D,) per sample

```rust
// ∂L/∂x̂_ij = ∂L/∂y_ij × γ_j  (for each sample i)
let mut grad_normalized = vec![0.0f32; self.size];
for j in 0..self.size {
    let idx = i * self.size + j;
    grad_normalized[j] = grad_output[idx] * self.gamma[j];
}
```

**Step 3b: Gradient w.r.t. variance (∂L/∂σ²_i)**

The variance σ²_i enters the computation through x̂_ij = (x_ij - μ_i) / sqrt(σ²_i + ε).

```
∂x̂_ij/∂σ²_i = (x_ij - μ_i) × (-1/2) × (σ²_i + ε)^(-3/2)
             = -x̂_ij / (2 × std_i)
```

Therefore:

```
∂L/∂σ²_i = Σⱼ (∂L/∂x̂_ij × ∂x̂_ij/∂σ²_i)
          = Σⱼ (∂L/∂x̂_ij × (x_ij - μ_i) × (-0.5) × (std_i)^(-3))
```

**Dimensions:** scalar per sample

```rust
// ∂L/∂σ²_i = Σⱼ (∂L/∂x̂_ij × (x_ij - μ_i) × (-0.5) × std_i^(-3))
let mut grad_var = 0.0f32;
for j in 0..self.size {
    let idx = i * self.size + j;
    let x_centered = input[idx] - mean[i];
    grad_var += grad_normalized[j] * x_centered * (-0.5) * (std[i].powi(3)).recip();
}
```

**Step 3c: Gradient w.r.t. mean (∂L/∂μ_i)**

The mean μ_i enters through two paths:
1. Directly through normalization: x̂_ij = (x_ij - μ_i) / std_i
2. Indirectly through variance: σ²_i = (1/D) × Σⱼ (x_ij - μ_i)²

**Path 1 (direct):**

```
∂x̂_ij/∂μ_i = -1 / std_i
```

So the direct gradient contribution is:

```
∂L/∂μ_i|_direct = Σⱼ (∂L/∂x̂_ij × (-1 / std_i))
```

**Path 2 (through variance):**

Since σ²_i depends on μ_i:

```
∂σ²_i/∂μ_i = (1/D) × Σⱼ 2(x_ij - μ_i) × (-1)
            = (-2/D) × Σⱼ (x_ij - μ_i)
```

But Σⱼ (x_ij - μ_i) = 0 by definition of the mean! So the indirect path via variance is zero in exact arithmetic. In the implementation, it is computed for numerical correctness:

```
∂L/∂μ_i|_indirect = ∂L/∂σ²_i × (-2/D) × Σⱼ (x_ij - μ_i)
```

The total gradient w.r.t. mean:

```
∂L/∂μ_i = Σⱼ (∂L/∂x̂_ij × (-1 / std_i)) + ∂L/∂σ²_i × (-2/D) × Σⱼ (x_ij - μ_i)
```

```rust
// ∂L/∂μ_i (direct path): Σⱼ (∂L/∂x̂_ij × (-1 / std_i))
let mut grad_mean = 0.0f32;
for j in 0..self.size {
    grad_mean += grad_normalized[j] * (-1.0 / std[i]);
}

// ∂L/∂μ_i (indirect path via variance): ∂L/∂σ²_i × (-2/D) × Σⱼ (x_ij - μ_i)
// Note: Σⱼ (x_ij - μ_i) ≈ 0 in exact arithmetic (included for numerical completeness)
let mut sum_centered = 0.0f32;
for j in 0..self.size {
    let idx = i * self.size + j;
    sum_centered += input[idx] - mean[i];
}
grad_mean += grad_var * (-2.0 * sum_centered / self.size as f32);
```

**Step 3d: Final gradient w.r.t. input (∂L/∂x_i)**

The input x_ij contributes to the loss through three paths:
1. Directly through x̂_ij: x̂_ij = (x_ij - μ_i) / std_i → ∂x̂_ij/∂x_ij = 1/std_i
2. Through variance σ²_i: σ²_i = (1/D)Σ(x_ij - μ_i)² → ∂σ²_i/∂x_ij = (2/D)(x_ij - μ_i)
3. Through mean μ_i: μ_i = (1/D)Σ x_ij → ∂μ_i/∂x_ij = 1/D

Combining all three paths:

```
∂L/∂x_ij = ∂L/∂x̂_ij × (1 / std_i)
          + ∂L/∂σ²_i × (2/D) × (x_ij - μ_i)
          + ∂L/∂μ_i × (1/D)
```

**Dimensions:** (B, D) — computed per-sample loop

```rust
// ∂L/∂x_ij = ∂L/∂x̂_ij / std_i
//           + ∂L/∂σ²_i × (2/D) × (x_ij - μ_i)
//           + ∂L/∂μ_i / D
for j in 0..self.size {
    let idx = i * self.size + j;
    let x_centered = input[idx] - mean[i];
    grad_input[idx] = grad_normalized[j] / std[i]
        + grad_var * 2.0 * x_centered / self.size as f32
        + grad_mean / self.size as f32;
}
```

### Implementation Notes

**Why is LayerNorm's backward simpler than BatchNorm's?**

In BatchNorm, every x_ij in the batch affects the shared batch mean μ_j and batch variance σ²_j, creating dense coupling between samples. In LayerNorm, each sample has its own μ_i and σ²_i, so the gradient for x_ij depends only on sample i — not on any other sample. This means the input gradient computation can be done in a straightforward per-sample loop.

**Structural comparison of backward passes:**

```
BatchNorm backward:  Loop over features j {
                       grad_var[j] = sum over all samples i {
                           ... x_ij from other samples affect x_ij via μ_j, σ²_j
                       }
                     }

LayerNorm backward:  Loop over samples i {
                       grad_var = scalar (from sample i only)
                       grad_mean = scalar (from sample i only)
                       Loop over features j {
                           grad_input[i, j] = ...
                       }
                     }
```

**The complete backward pass — combining everything:**

```rust
fn backward(&self, _input: &[f32], grad_output: &[f32], grad_input: &mut [f32], batch_size: usize) {
    let normalized = self.cached_normalized.borrow();
    let std = self.cached_std.borrow();
    let mean = self.cached_mean.borrow();
    let input = self.cached_input.borrow();

    let mut grad_gamma = self.grad_gamma.borrow_mut();
    let mut grad_beta = self.grad_beta.borrow_mut();

    // Step 1: Accumulate gamma and beta gradients across the batch
    let scale = 1.0 / batch_size as f32;
    for i in 0..batch_size {
        for j in 0..self.size {
            let idx = i * self.size + j;
            grad_gamma[j] += grad_output[idx] * normalized[idx] * scale;
            grad_beta[j] += grad_output[idx] * scale;
        }
    }

    // Step 2: Compute input gradients — independently per sample
    for i in 0..batch_size {
        // 2a: ∂L/∂x̂_ij = ∂L/∂y_ij × γ_j
        let mut grad_normalized = vec![0.0f32; self.size];
        for j in 0..self.size {
            let idx = i * self.size + j;
            grad_normalized[j] = grad_output[idx] * self.gamma[j];
        }

        // 2b: ∂L/∂σ²_i = Σⱼ (∂L/∂x̂_ij × (x_ij - μ_i) × (-0.5) / std_i³)
        let mut grad_var = 0.0f32;
        for j in 0..self.size {
            let idx = i * self.size + j;
            let x_centered = input[idx] - mean[i];
            grad_var += grad_normalized[j] * x_centered * (-0.5) * (std[i].powi(3)).recip();
        }

        // 2c: ∂L/∂μ_i = Σⱼ (∂L/∂x̂_ij × (-1 / std_i)) + ∂L/∂σ²_i × (-2/D) × Σⱼ (x_ij - μ_i)
        let mut grad_mean = 0.0f32;
        for j in 0..self.size {
            grad_mean += grad_normalized[j] * (-1.0 / std[i]);
        }
        let sum_centered: f32 = (0..self.size)
            .map(|j| input[i * self.size + j] - mean[i])
            .sum();
        grad_mean += grad_var * (-2.0 * sum_centered / self.size as f32);

        // 2d: ∂L/∂x_ij = ∂L/∂x̂_ij / std_i + ∂L/∂σ²_i × (2/D) × (x_ij - μ_i) + ∂L/∂μ_i / D
        for j in 0..self.size {
            let idx = i * self.size + j;
            let x_centered = input[idx] - mean[i];
            grad_input[idx] = grad_normalized[j] / std[i]
                + grad_var * 2.0 * x_centered / self.size as f32
                + grad_mean / self.size as f32;
        }
    }
}
```

**Memory access pattern:**

The per-sample loop structure is efficient: each sample i is processed in a single pass through the feature dimension, which is cache-friendly since the features are contiguous in memory (row-major layout: `input[i * size + j]`).

**RefCell interior mutability:**

```rust
let mut grad_gamma = self.grad_gamma.borrow_mut();
let mut grad_beta = self.grad_beta.borrow_mut();
```

The `borrow_mut()` pattern allows accumulating gradients during `backward(&self)`, which takes an immutable reference to self. This is the same pattern used by BatchNorm and is standard for the Layer trait interface.

## No Training vs Inference Mode Difference

**This is a defining characteristic of Layer Normalization.**

Because LayerNorm computes statistics from the current input at each call (not from accumulated batch statistics), there is no distinction between training and inference modes:

```
Training forward:   y_ij = γ_j × ((x_ij - μ_i) / sqrt(σ²_i + ε)) + β_j
Inference forward:  y_ij = γ_j × ((x_ij - μ_i) / sqrt(σ²_i + ε)) + β_j
```

The formula is identical. There are no running statistics, no mode flags, and no separate code paths.

**Contrast with BatchNorm:**

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Training forward | Use batch stats (μ_batch, σ²_batch) | Use per-sample stats |
| Inference forward | Use running stats (μ_run, σ²_run) | Use per-sample stats |
| Running stats maintained | Yes (EMA update per batch) | No |
| `set_training(true/false)` needed | Yes — critical! | Not applicable |
| Risk of forgetting mode switch | Yes — common bug | None |

**Benefits of mode invariance:**
- No risk of the common BatchNorm bug (forgetting `set_training(false)` before evaluation)
- Truly deterministic inference — same input always produces same output
- No warm-up period needed (BatchNorm needs batches to converge running stats)
- Consistent behavior for streaming/online inference with batch_size=1

**Implementation note — no mode flag in LayerNormLayer:**

The `LayerNormLayer` struct has no `training` field and no `set_training()` method, because none is needed:

```rust
pub struct LayerNormLayer {
    size: usize,
    epsilon: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    grad_gamma: RefCell<Vec<f32>>,
    grad_beta: RefCell<Vec<f32>>,
    // No running_mean, running_var — not needed!
    // No training: bool field — not applicable!
    cached_input: RefCell<Vec<f32>>,
    cached_mean: RefCell<Vec<f32>>,
    cached_var: RefCell<Vec<f32>>,
    cached_normalized: RefCell<Vec<f32>>,
    cached_std: RefCell<Vec<f32>>,
}
```

## Parameter Updates

After computing gradients, parameters are updated using gradient descent or an optimizer:

**Vanilla gradient descent:**

```
γ := γ - η × (∂L/∂γ)
β := β - η × (∂L/∂β)
```

Where η is the learning rate.

**Implementation:**

```rust
fn update_parameters(&mut self, learning_rate: f32) {
    let grad_gamma = self.grad_gamma.borrow();
    let grad_beta = self.grad_beta.borrow();

    // Update gamma: gamma = gamma - learning_rate * gradient
    for (param, &gradient) in self.gamma.iter_mut().zip(grad_gamma.iter()) {
        *param -= learning_rate * gradient;
    }

    // Update beta: beta = beta - learning_rate * gradient
    for (param, &gradient) in self.beta.iter_mut().zip(grad_beta.iter()) {
        *param -= learning_rate * gradient;
    }

    // Clear gradients for next iteration
    drop(grad_gamma);
    drop(grad_beta);
    self.grad_gamma.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
    self.grad_beta.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
}
```

**With optimizer (e.g., Adam):**

```rust
fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
    let grad_gamma = self.grad_gamma.borrow();
    let grad_beta = self.grad_beta.borrow();

    // Update gamma using optimizer
    optimizer.update(&mut self.gamma, &grad_gamma);

    // Update beta using optimizer
    optimizer.update(&mut self.beta, &grad_beta);

    // Clear gradients for next iteration
    drop(grad_gamma);
    drop(grad_beta);
    self.grad_gamma.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
    self.grad_beta.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
}
```

**Gradient zeroing:** After applying updates (SGD or optimizer), accumulated gradients are reset to zero, preventing their accumulation across training steps.

**Note:** Unlike BatchNorm, there are no running statistics to update here. The only parameters are γ (gamma) and β (beta), both updated by gradient descent.

## Initialization

### Gamma and Beta Initialization

Layer normalization parameters are initialized to represent the identity function:

```
γ = 1.0  (identity scale: no change to variance)
β = 0.0  (no shift: zero mean)
```

**Rationale:** With this initialization, at the start of training:

```
y_ij = 1.0 × x̂_ij + 0.0 = x̂_ij
```

The output is exactly the normalized value. This gives the network a well-conditioned starting point, with all samples having approximately zero mean and unit variance across features. As training progresses, γ and β adapt to the optimal scale and shift for the task.

**Rust implementation:**

```rust
Self {
    size,
    epsilon,

    // Initialize gamma to 1.0 (identity scaling), beta to 0.0 (no shift)
    gamma: vec![1.0f32; size],
    beta: vec![0.0f32; size],

    // Zero-initialize gradients
    grad_gamma: RefCell::new(vec![0.0f32; size]),
    grad_beta: RefCell::new(vec![0.0f32; size]),

    // Initialize caches (will be resized during forward pass)
    cached_input: RefCell::new(Vec::new()),
    cached_mean: RefCell::new(Vec::new()),
    cached_var: RefCell::new(Vec::new()),
    cached_normalized: RefCell::new(Vec::new()),
    cached_std: RefCell::new(Vec::new()),
    // No running_mean or running_var needed!
}
```

**Compare to BatchNorm initialization:**

BatchNorm also initializes γ=1 and β=0, but additionally initializes:
- `running_mean = 0.0` — needed for inference
- `running_var = 0.0` — needed for inference

LayerNorm requires no such running statistics, simplifying both the struct definition and initialization.

## Numerical Considerations

### Epsilon (ε) and Division by Zero

The most critical numerical issue in layer normalization is division by nearly-zero standard deviation.

**Without epsilon:**

```
std_i = sqrt(σ²_i)  ← Can be exactly 0!
x̂_ij = (x_ij - μ_i) / 0.0  ← NaN or Inf!
```

**With epsilon:**

```
std_i = sqrt(σ²_i + ε)  ← Always positive
x̂_ij = (x_ij - μ_i) / std_i  ← Finite and bounded
```

**When does σ²_i ≈ 0 happen?**
- All features are constant for sample i (e.g., all-zero embedding vector)
- Very small feature dimension D (e.g., D=1 always has σ²=0)
- After ReLU activation when many features are zero

**Typical epsilon values:**
- `1e-5` (0.00001): Most common, good balance of stability vs precision
- `1e-8`: More precise but may not prevent instability with very small variances
- `1e-3`: More robust but adds more bias to the normalized values

```rust
sample_std[i] = (sample_var[i] + self.epsilon).sqrt();  // epsilon prevents division by zero
```

### Backward Pass Numerical Stability

The backward pass computes `1 / std[i]` and `1 / std[i]³` (in the variance gradient). Since std_i > 0 (guaranteed by ε > 0), these are safe. However, if std_i is very small (close to √ε ≈ 0.003 for ε=1e-5), gradients can be large:

```
∂L/∂x̂_ij / std_i ≈ ∂L/∂x̂_ij / 0.003 = 333 × ∂L/∂x̂_ij
```

The variance gradient path uses `(std[i].powi(3)).recip()` which computes `1/std³`:

```rust
grad_var += grad_normalized[j] * x_centered * (-0.5) * (std[i].powi(3)).recip();
```

**Mitigation for gradient amplification:**
- Use gradient clipping at the optimizer level
- Use slightly larger ε (1e-4) for very small feature dimensions
- Ensure input features are not degenerate (all-zero vectors)

### Feature Dimension Size Considerations

Unlike BatchNorm which requires large batch sizes, LayerNorm requires a reasonably large feature dimension D for reliable statistics:

- **D=1:** Variance is always 0! LayerNorm collapses (every output is β). Don't use with D=1.
- **D<8:** High variance in statistics, limited effectiveness
- **D≥32:** Generally reliable; most transformer models use D≥64 or D≥128 where LayerNorm excels

**Rule of thumb:** Layer normalization is most effective when the feature dimension D is large (≥32). This is why it's commonly applied in transformers with D=64 to D=1024.

### Potential Numerical Issues Summary

**1. NaN from zero variance:**
- Cause: All features of a sample have identical values (σ²=0)
- Symptom: NaN in output and gradients
- Fix: Ensure ε > 0, check for degenerate input (all-zero vectors)

**2. Large gradients from small variance:**
- Cause: Very small feature variance amplifies the `1/std` term
- Symptom: Loss spikes, very large gradient norms
- Fix: Gradient clipping, larger ε, check input scale

**3. Incorrect behavior for D=1:**
- Cause: Single feature always has σ²=0 regardless of value
- Symptom: Output is always β (the shift), gradients are all zero or NaN
- Fix: Never apply LayerNorm to 1-dimensional features

**Gradient correctness verification:**

```
numerical_grad = (L(γ + ε·eⱼ) - L(γ - ε·eⱼ)) / (2ε)
```

Compare with analytical gradient ∂L/∂γ_j. For ε=1e-4, they should match within ~10⁻⁴ relative error.

## Summary

Layer normalization applies per-sample normalization across features, then rescales with learnable parameters. The key formula is:

**Forward pass (identical for training and inference):**

```
μ_i = (1/D) × Σⱼ x_ij                          → per-sample mean
σ²_i = (1/D) × Σⱼ (x_ij - μ_i)²                → per-sample variance
x̂_ij = (x_ij - μ_i) / sqrt(σ²_i + ε)           → normalize
y_ij = γ_j × x̂_ij + β_j                          → scale and shift
```

**Backward pass:**

```
∂L/∂β_j = (1/B) × Σᵢ ∂L/∂y_ij                                        → beta gradient
∂L/∂γ_j = (1/B) × Σᵢ (∂L/∂y_ij × x̂_ij)                               → gamma gradient
∂L/∂x̂_ij = ∂L/∂y_ij × γ_j                                              → grad through scale
∂L/∂σ²_i = Σⱼ (∂L/∂x̂_ij × (x_ij - μ_i) × (-0.5) / std_i³)            → grad through variance
∂L/∂μ_i = Σⱼ (∂L/∂x̂_ij × (-1/std_i)) + ∂L/∂σ²_i × (-2/D) × Σⱼ(x-μ)  → grad through mean
∂L/∂x_ij = ∂L/∂x̂_ij/std_i + ∂L/∂σ²_i × 2(x-μ)/D + ∂L/∂μ_i/D        → input gradient
```

**Key implementation details:**
- ε prevents division by zero when sample variance is near zero
- Statistics are per-sample (shape (B,)), not per-feature (shape (D,)) like BatchNorm
- No running statistics — the struct has no `running_mean` or `running_var` fields
- RefCell provides interior mutability for accumulating gradients during `backward(&self)`
- Cached x̂, μ, σ², std from forward pass are required for the backward pass
- Per-sample backward loop simplifies computation vs BatchNorm's cross-sample dependency
- Works correctly with batch_size=1, making it ideal for streaming inference
- Feature dimension D should be ≥32 for statistically reliable normalization

This forms a critical component in transformer architectures and other sequence models where batch normalization is inappropriate due to variable batch sizes and sequence-level statistics.

## Related Documentation

**Alternative Normalization:**
- [Batch Normalization](batchnorm_layer.md) - Normalizes across the batch dimension per feature, suitable for CNNs and large batch training

**Preceding/Following Layers:**
- [Dense Layer](dense_layer.md) - Often used before or after LayerNorm in transformer FFN blocks
- [Convolutional Layer](conv2d_layer.md) - Typically combined with BatchNorm rather than LayerNorm

**Core Architecture:**
- [Backpropagation Overview](README.md) - General backpropagation concepts and notation
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface implementation
- [Layer Norm Source](../../src/layers/layernorm.rs) - Full Rust implementation

**References:**
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv:1607.06450.
- Ioffe, S., & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). ICML.
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS. (Original transformer paper using LayerNorm)
