# Batch Normalization Layer Mathematics

This document provides a comprehensive explanation of the mathematics behind the Batch Normalization (BatchNorm) layer, covering both forward and backward propagation with full derivations, including the training/inference distinction and numerical stability considerations.

## Table of Contents

- [Overview](#overview)
- [Forward Pass](#forward-pass)
  - [Mathematical Definition](#mathematical-definition)
  - [Dimension Analysis](#dimension-analysis)
  - [Running Statistics Update](#running-statistics-update)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass](#backward-pass)
  - [Gradient Flow Visualization](#gradient-flow-visualization)
  - [Gradient Computation](#gradient-computation)
  - [Chain Rule Application](#chain-rule-application)
  - [Gradient Formulas](#gradient-formulas)
  - [Implementation Notes](#implementation-notes)
- [Training vs Inference Mode](#training-vs-inference-mode)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Numerical Considerations](#numerical-considerations)

## Overview

Batch Normalization (Ioffe & Szegedy, 2015) normalizes each feature dimension across a mini-batch to have zero mean and unit variance, then applies learnable scale (γ) and shift (β) parameters. This stabilizes training by reducing internal covariate shift.

**Key characteristics:**
- Normalizes per-feature statistics across the batch dimension
- Two learnable parameters per feature: scale γ and shift β
- Maintains running statistics (mean, variance) for deterministic inference
- Preserves input/output dimensionality (no shape change)
- Provides implicit regularization through batch-level noise

**Parameters:**
- `γ` (gamma): Learnable scale vector of shape (size,) — initialized to 1.0
- `β` (beta): Learnable shift vector of shape (size,) — initialized to 0.0
- `ε` (epsilon): Small constant for numerical stability (typical: 1e-5)
- `α` (momentum): EMA factor for running statistics (typical: 0.9 or 0.99)

## Forward Pass

### Mathematical Definition

The forward pass of a batch normalization layer consists of five steps:

**Step 1: Compute batch mean per feature**

```
μ_j = (1/m) × Σᵢ x_ij    for i = 1 to m
```

**Step 2: Compute batch variance per feature**

```
σ²_j = (1/m) × Σᵢ (x_ij - μ_j)²    for i = 1 to m
```

**Step 3: Compute standard deviation (with numerical stability)**

```
std_j = sqrt(σ²_j + ε)
```

**Step 4: Normalize**

```
x̂_ij = (x_ij - μ_j) / std_j
```

**Step 5: Scale and shift (affine transformation)**

```
y_ij = γ_j × x̂_ij + β_j
```

Where:
- **x**: Input matrix of shape (batch_size, size)
- **y**: Output matrix of shape (batch_size, size)
- **m**: batch_size (number of samples)
- **j**: Feature index, ranging from 0 to size-1
- **i**: Sample index, ranging from 0 to m-1
- **μ_j**: Mean of feature j across the batch, shape (size,)
- **σ²_j**: Variance of feature j across the batch, shape (size,)
- **std_j**: Standard deviation of feature j, shape (size,)
- **x̂_ij**: Normalized value (zero mean, unit variance), shape (batch_size, size)
- **γ_j**: Learnable scale parameter for feature j, shape (size,)
- **β_j**: Learnable shift parameter for feature j, shape (size,)
- **ε**: Small constant for numerical stability

**Combined forward pass formula:**

```
y_ij = γ_j × ((x_ij - μ_j) / sqrt(σ²_j + ε)) + β_j
```

**Intuition:** After normalization, x̂ has zero mean and variance 1 per feature. The γ and β parameters allow the network to undo this normalization if needed, giving the layer the capacity to represent any distribution.

### Dimension Analysis

**Single Sample (B=1, unusual for training):**
- Input: (1, D) where D = size (number of features)
- Statistics: (D,) — mean and variance
- Normalized: (1, D)
- Output: (1, D)

**Batched Computation (standard):**
- Input x: (B, D) where B = batch_size, D = size
- Batch mean μ: (D,) — one mean per feature
- Batch variance σ²: (D,) — one variance per feature
- Standard deviation std: (D,)
- Normalized x̂: (B, D)
- Gamma γ: (D,) — broadcasted across batch
- Beta β: (D,) — broadcasted across batch
- Output y: (B, D)

**Operation breakdown:**

```
Step 1: Mean computation
  μ_j = sum(x[:, j]) / B    → (D,)

Step 2: Variance computation
  σ²_j = sum((x[:, j] - μ_j)²) / B    → (D,)

Step 3: Normalize
  x̂ = (x - μ[None, :]) / std[None, :]    → (B, D)

Step 4: Scale and shift
  y = γ[None, :] × x̂ + β[None, :]    → (B, D)
```

**Example with concrete dimensions:**

For a batch norm layer with size=512 and batch_size=32:
- Input x: (32, 512)
- Batch mean μ: (512,) — sum along batch axis divided by 32
- Batch variance σ²: (512,)
- Standard deviation: (512,)
- Normalized x̂: (32, 512)
- Scale γ: (512,) — broadcasted to (32, 512)
- Shift β: (512,) — broadcasted to (32, 512)
- Output y: (32, 512)
- Parameters: 512 × 2 = 1,024 (gamma + beta)

### Rust Implementation (Forward Pass)

```rust
// Training mode: compute batch statistics and normalize
let mut batch_mean = vec![0.0f32; self.size];
let mut batch_var = vec![0.0f32; self.size];

// Step 1: Compute mean for each feature across the batch
// μ_j = (1/m) × Σᵢ x_ij
for i in 0..batch_size {
    for j in 0..self.size {
        batch_mean[j] += input[i * self.size + j];
    }
}
for mean in &mut batch_mean {
    *mean /= batch_size as f32;
}

// Step 2: Compute variance for each feature across the batch
// σ²_j = (1/m) × Σᵢ (x_ij - μ_j)²
for i in 0..batch_size {
    for j in 0..self.size {
        let diff = input[i * self.size + j] - batch_mean[j];
        batch_var[j] += diff * diff;
    }
}
for var in &mut batch_var {
    *var /= batch_size as f32;
}

// Step 3: Compute standard deviation: std_j = sqrt(σ²_j + ε)
let std: Vec<f32> = batch_var
    .iter()
    .map(|&v| (v + self.epsilon).sqrt())
    .collect();

// Steps 4+5: Normalize, scale, and shift
// x̂_ij = (x_ij - μ_j) / std_j
// y_ij = γ_j × x̂_ij + β_j
let mut normalized = vec![0.0f32; total_size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        normalized[idx] = (input[idx] - batch_mean[j]) / std[j];
        output[idx] = self.gamma[j] * normalized[idx] + self.beta[j];
    }
}

// Cache values needed for the backward pass
*self.cached_mean.borrow_mut() = batch_mean;
*self.cached_var.borrow_mut() = batch_var;
*self.cached_normalized.borrow_mut() = normalized;
*self.cached_std.borrow_mut() = std;
```

**Key caching:** The forward pass saves batch_mean, batch_var, normalized (x̂), and std. These are essential for computing the correct backward gradients, since the backward pass must differentiate through the normalization statistics that depend on the entire batch.

### Running Statistics Update

During training, running statistics are updated after each batch using exponential moving average (EMA):

```
running_μ_j = α × running_μ_j + (1 - α) × μ_j
running_σ²_j = α × running_σ²_j + (1 - α) × σ²_j
```

Where:
- **α**: Momentum parameter (typical: 0.9 or 0.99)
- **μ_j**: Current batch mean for feature j
- **σ²_j**: Current batch variance for feature j
- **running_μ_j**: Accumulated running mean (used during inference)
- **running_σ²_j**: Accumulated running variance (used during inference)

**Interpretation:**
- High α (e.g., 0.99) → slow adaptation, smoothly tracks long-term statistics
- Low α (e.g., 0.9) → faster adaptation, more responsive to recent batches

**Rust Implementation:**

```rust
// Update running statistics with exponential moving average
// running = momentum * running + (1 - momentum) * batch
let mut running_mean = self.running_mean.borrow_mut();
let mut running_var = self.running_var.borrow_mut();
for j in 0..self.size {
    running_mean[j] =
        self.momentum * running_mean[j] + (1.0 - self.momentum) * batch_mean[j];
    running_var[j] =
        self.momentum * running_var[j] + (1.0 - self.momentum) * batch_var[j];
}
```

**RefCell pattern:** Running statistics use `RefCell<Vec<f32>>` to allow interior mutability during the forward pass, which takes `&self` (immutable reference). This avoids making the entire forward pass `&mut self`.

### Computational Complexity

**Time Complexity:**

```
Step 1 (mean):     O(B × D)
Step 2 (variance): O(B × D)
Step 3 (std):      O(D)
Step 4+5 (norm):   O(B × D)
Running update:    O(D)

Total: O(B × D)
```

Compared to a dense layer with the same input size, batch norm has linear complexity in B (no matrix multiplication). The constant factor is small: a few passes over the input.

**Space Complexity:**

```
Learnable parameters: O(D)   — γ and β
Running statistics:   O(D)   — running_mean and running_var
Cached values:        O(B × D) — normalized, plus O(D) for mean/var/std
```

**Example calculation:**

For batch norm with size=512 and batch_size=32:
- Operations: ~4 × 32 × 512 ≈ 65K FLOPs
- Parameters: 512 × 2 = 1,024 floats ≈ 4 KB
- Cache memory: 32 × 512 × 4 bytes ≈ 64 KB (normalized) + small overheads

## Backward Pass

### Gradient Flow Visualization

The following diagram illustrates how gradients flow backward through the batch normalization layer:

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
    │  mean μ │         (D,)        |    ∂L/∂μ      │  contribution│
    └────┬────┘                     |    (D,)       └─────┬──────┘
         │                          |                     │
         ▼                          |                     │
    ┌─────────┐                    |               ┌─────┴──────┐
    │ compute │──────── σ² ────────┼───────────────┤  ∂L/∂σ²    │
    │  var σ² │         (D,)        |    ∂L/∂σ²     │  contribution│
    └────┬────┘                     |    (D,)       └─────┬──────┘
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
    │    + β  │         (D,)        |    ∂L/∂γ = Σ(∂L/∂y ⊙ x̂)/m
    └────┬────┘                     |    ∂L/∂β = Σ(∂L/∂y)/m
         │                          |
         ▼                          |
      Output y                      |          ∂L/∂y (given)
      (B, D)                        |          (B, D)

Legend:
  ──►   Forward data flow
  ──▲   Backward gradient flow
  ⊙     Element-wise multiplication
  Σ     Sum over batch dimension (axis 0)
```

**Key insight:** Batch normalization's backward pass is more complex than the dense layer because the mean and variance are computed from the entire batch. Therefore, the gradient for a single input sample depends on *all* other samples in the batch through the mean and variance.

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

To differentiate the loss through this chain, we apply the chain rule at each step, working backward:

**Level 1: Gradient through the affine transform (y = γ ⊙ x̂ + β)**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂y_ij/∂γ_j = x̂_ij
∂y_ij/∂β_j = 1
∂y_ij/∂x̂_ij = γ_j
```

**Level 2: Gradient through normalization (x̂ = (x - μ) / std)**

Since x̂_ij = (x_ij - μ_j) / std_j:

```
∂x̂_ij/∂x_ij = 1 / std_j
∂x̂_ij/∂μ_j  = -1 / std_j
∂x̂_ij/∂σ²_j = (x_ij - μ_j) × (-1/2) × (σ²_j + ε)^(-3/2)
             = -x̂_ij / (2 × std_j)
```

**Level 3: Gradient through variance (σ² = (1/m) × Σ(x - μ)²)**

Since μ_j is constant with respect to σ²_j:

```
∂σ²_j/∂x_ij = (2/m) × (x_ij - μ_j)
```

**Level 4: Gradient through mean (μ = (1/m) × Σ x)**

```
∂μ_j/∂x_ij = 1/m
```

Note: The mean also contributes to the variance, creating a secondary path.

### Gradient Formulas

#### 1. Gradient w.r.t. Beta (∂L/∂β) — Simplest

**Formula:**

```
∂L/∂β_j = (1/m) × Σᵢ (∂L/∂y_ij)
```

**Derivation:**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂L/∂β_j = Σᵢ (∂L/∂y_ij × ∂y_ij/∂β_j)
         = Σᵢ (∂L/∂y_ij × 1)
         = Σᵢ ∂L/∂y_ij
```

Divided by m (the scale factor applied in the Rust implementation for normalization):

```
∂L/∂β_j = (1/m) × Σᵢ ∂L/∂y_ij
```

**Dimensions:** (D,) = sum over batch of (B, D)

**Implementation:**

```rust
// Accumulate beta gradient across the batch
let mut db = vec![0.0f32; self.size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        db[j] += grad_output[idx];  // ∂L/∂β_j += ∂L/∂y_ij
    }
}
// Apply (1/m) scaling when accumulating to the gradient accumulator
self.grad_beta.accumulate_scaled(&db, scale);  // scale = 1/m
```

#### 2. Gradient w.r.t. Gamma (∂L/∂γ)

**Formula:**

```
∂L/∂γ_j = (1/m) × Σᵢ (∂L/∂y_ij × x̂_ij)
```

**Derivation:**

Since y_ij = γ_j × x̂_ij + β_j:

```
∂L/∂γ_j = Σᵢ (∂L/∂y_ij × ∂y_ij/∂γ_j)
         = Σᵢ (∂L/∂y_ij × x̂_ij)
```

With the (1/m) scale factor:

```
∂L/∂γ_j = (1/m) × Σᵢ (∂L/∂y_ij × x̂_ij)
```

**Dimensions:** (D,) = element-wise sum over batch of (B, D) ⊙ (B, D)

**Implementation:**

```rust
// Accumulate gamma gradient across the batch
let mut dg = vec![0.0f32; self.size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        dg[j] += grad_output[idx] * normalized[idx];  // ∂L/∂y_ij × x̂_ij
    }
}
// Apply (1/m) scaling when accumulating to the gradient accumulator
self.grad_gamma.accumulate_scaled(&dg, scale);  // scale = 1/m
```

Note: `normalized[idx]` is the cached x̂_ij from the forward pass.

#### 3. Gradient w.r.t. Input (∂L/∂x) — Most Complex

The input gradient requires combining three paths through the computation graph:

**Path A:** Direct path through normalization
**Path B:** Through variance σ²
**Path C:** Through mean μ

**Step 3a: Gradient w.r.t. normalized values (∂L/∂x̂)**

```
∂L/∂x̂_ij = ∂L/∂y_ij × γ_j
```

**Dimensions:** (B, D)

```rust
// ∂L/∂x̂_ij = ∂L/∂y_ij × γ_j
let mut grad_normalized = vec![0.0f32; total_size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        grad_normalized[idx] = grad_output[idx] * self.gamma[j];
    }
}
```

**Step 3b: Gradient w.r.t. variance (∂L/∂σ²)**

The variance σ²_j enters the computation through x̂_ij = (x_ij - μ_j) / sqrt(σ²_j + ε).

```
∂x̂_ij/∂σ²_j = (x_ij - μ_j) × (-1/2) × (σ²_j + ε)^(-3/2)
             = -x̂_ij / (2 × std_j)
```

Therefore:

```
∂L/∂σ²_j = Σᵢ (∂L/∂x̂_ij × ∂x̂_ij/∂σ²_j)
          = Σᵢ (∂L/∂x̂_ij × (-x̂_ij) / (2 × std_j))
          = Σᵢ (∂L/∂x̂_ij × x̂_ij × (-0.5) / std_j)
```

**Dimensions:** (D,) = sum over batch of per-element products

```rust
// ∂L/∂σ²_j = Σᵢ (∂L/∂x̂_ij × x̂_ij × (-0.5) / std_j)
let mut grad_var = vec![0.0f32; self.size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        // x̂_ij = normalized[idx], std_j = std[j]
        grad_var[j] += grad_normalized[idx] * normalized[idx] * (-0.5) / std[j];
    }
}
```

**Step 3c: Gradient w.r.t. mean (∂L/∂μ)**

The mean μ_j enters through two paths:
1. Directly through normalization: x̂_ij = (x_ij - μ_j) / std_j
2. Indirectly through variance: σ²_j = (1/m) × Σᵢ (x_ij - μ_j)²

**Path 1 (direct):**

```
∂x̂_ij/∂μ_j = -1 / std_j
```

So the direct gradient contribution is:

```
∂L/∂μ_j|_direct = Σᵢ (∂L/∂x̂_ij × (-1 / std_j))
```

**Path 2 (through variance):**

Since σ²_j depends on μ_j:

```
∂σ²_j/∂μ_j = (1/m) × Σᵢ 2(x_ij - μ_j) × (-1)
            = (-2/m) × Σᵢ (x_ij - μ_j)
```

But Σᵢ (x_ij - μ_j) = 0 by definition of the mean! So the indirect path via variance is zero in the exact case, but in practice it's computed for correctness:

```
∂L/∂μ_j|_indirect = ∂L/∂σ²_j × (-2/m) × Σᵢ (x_ij - μ_j)
```

The total gradient w.r.t. mean:

```
∂L/∂μ_j = Σᵢ (∂L/∂x̂_ij × (-1 / std_j)) + ∂L/∂σ²_j × (-2/m) × Σᵢ (x_ij - μ_j)
```

```rust
// ∂L/∂μ_j (direct path): Σᵢ (∂L/∂x̂_ij × (-1 / std_j))
let mut grad_mean = vec![0.0f32; self.size];
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        grad_mean[j] += grad_normalized[idx] * (-1.0 / std[j]);
    }
}

// ∂L/∂μ_j (indirect path via variance): ∂L/∂σ²_j × (-2/m) × Σᵢ (x_ij - μ_j)
// Note: x_ij - μ_j = x̂_ij × std_j = normalized[idx] × std[j]
for j in 0..self.size {
    let sum_diff = (0..batch_size)
        .map(|i| {
            let idx = i * self.size + j;
            normalized[idx] * std[j]  // x_ij - μ_j = x̂_ij × std_j
        })
        .sum::<f32>();
    grad_mean[j] += grad_var[j] * (-2.0 * sum_diff / batch_size as f32);
}
```

**Step 3d: Final gradient w.r.t. input (∂L/∂x)**

The input x_ij contributes to the loss through three paths:
1. Directly through x̂_ij: x̂_ij = (x_ij - μ_j) / std_j → ∂x̂_ij/∂x_ij = 1/std_j
2. Through variance σ²_j: σ²_j = (1/m)Σ(x_ij - μ_j)² → ∂σ²_j/∂x_ij = (2/m)(x_ij - μ_j)
3. Through mean μ_j: μ_j = (1/m)Σ x_ij → ∂μ_j/∂x_ij = 1/m

Combining all three paths:

```
∂L/∂x_ij = ∂L/∂x̂_ij × (1 / std_j)
          + ∂L/∂σ²_j × (2/m) × (x_ij - μ_j)
          + ∂L/∂μ_j × (1/m)
```

**Dimensions:** (B, D)

```rust
// ∂L/∂x_ij = ∂L/∂x̂_ij / std_j
//           + ∂L/∂σ²_j × (2/m) × (x_ij - μ_j)
//           + ∂L/∂μ_j / m
// Note: x_ij - μ_j = x̂_ij × std_j = normalized[idx] × std[j]
for i in 0..batch_size {
    for j in 0..self.size {
        let idx = i * self.size + j;
        let x_centered = normalized[idx] * std[j];  // x_ij - μ_j
        grad_input[idx] = grad_normalized[idx] / std[j]
            + grad_var[j] * 2.0 * x_centered / batch_size as f32
            + grad_mean[j] / batch_size as f32;
    }
}
```

### Implementation Notes

**Why is the backward pass complex?**

In a dense layer, each input x_i affects only the output y corresponding to its row in the weight matrix. In batch normalization, every x_ij in the batch affects the mean μ_j and variance σ²_j, which in turn affect the normalization of *every other sample in the batch*. This creates a dense coupling between all samples in the batch during backpropagation.

**Gradient flow from the variance path:**

Mathematically, Σᵢ (x_ij - μ_j) = 0, so the indirect mean gradient via variance should be zero. However, the implementation computes it anyway because:
1. In finite precision, rounding errors mean it's not exactly zero
2. It keeps the implementation mathematically general and correct

**Memory access pattern:**

The backward pass iterates over (batch, feature) pairs multiple times (one pass for each of: dg/db, grad_normalized, grad_var, grad_mean, grad_input). This is cache-friendly for features but requires multiple scans of the batch.

**Gradient accumulation:**

```rust
let scale = 1.0 / batch_size as f32;
// ...
self.grad_gamma.accumulate_scaled(&dg, scale);
self.grad_beta.accumulate_scaled(&db, scale);
```

The `accumulate_scaled` method multiplies each element by `scale` before adding to the accumulator. The (1/m) factor normalizes the parameter gradients by batch size. Note that the input gradients (grad_input) do NOT apply this (1/m) factor — they are naturally normalized by the Σ/m terms.

## Training vs Inference Mode

This is one of the most important aspects of batch normalization.

### Training Mode

During training:
- Batch statistics (μ, σ²) are computed fresh for each forward pass
- These statistics are specific to the current mini-batch
- Running statistics are updated via EMA after each batch
- The backward pass uses cached batch statistics from the forward pass

**Rust code:**

```rust
if self.training {
    // Compute fresh batch statistics: μ, σ², std
    // Normalize using batch statistics
    // Update running statistics via EMA
    // Cache batch statistics for backward pass
}
```

### Inference Mode

During inference:
- Use accumulated running statistics (not batch statistics)
- Normalization is deterministic regardless of other samples
- Running statistics should have converged to represent the training distribution

**Forward pass:**

```
x̂_ij = (x_ij - running_μ_j) / sqrt(running_σ²_j + ε)
y_ij = γ_j × x̂_ij + β_j
```

**Rust code:**

```rust
} else {
    // Inference mode: use running statistics
    let running_mean = self.running_mean.borrow();
    let running_var = self.running_var.borrow();
    for i in 0..batch_size {
        for j in 0..self.size {
            let idx = i * self.size + j;
            let normalized =
                (input[idx] - running_mean[j]) / (running_var[j] + self.epsilon).sqrt();
            output[idx] = self.gamma[j] * normalized + self.beta[j];
        }
    }
}
```

### Inference Backward Pass

In inference mode, the backward pass simplifies significantly:

```
∂L/∂x_ij = ∂L/∂y_ij × γ_j / sqrt(running_σ²_j + ε)
```

Since running statistics are constants (not computed from the current input), the gradients through μ and σ² do not need to be computed:

```rust
if !self.training {
    // Inference mode: simple gradient pass-through with gamma scaling
    let running_var = self.running_var.borrow();
    for i in 0..batch_size {
        for j in 0..self.size {
            let idx = i * self.size + j;
            grad_input[idx] =
                grad_output[idx] * self.gamma[j] / (running_var[j] + self.epsilon).sqrt();
        }
    }
    return;
}
```

Parameter gradients (∂L/∂γ, ∂L/∂β) are also not accumulated in inference mode.

### Mode Comparison

| Aspect | Training Mode | Inference Mode |
|--------|--------------|----------------|
| Normalization stats | Batch statistics (μ_batch, σ²_batch) | Running statistics (μ_run, σ²_run) |
| Deterministic? | No (depends on batch) | Yes |
| Running stats updated | Yes (via EMA) | No |
| Backward complexity | Full chain rule through μ and σ² | Simple scaling by γ/std |
| Parameter gradients | Accumulated | Not accumulated |
| When to use | `layer.set_training(true)` | `layer.set_training(false)` |

**Important:** Always call `set_training(false)` before inference/evaluation. Forgetting this is a common bug that causes evaluation results to vary between runs (depending on batch composition).

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
    self.grad_gamma
        .apply_sgd_update(&mut self.gamma, learning_rate);
    self.grad_beta
        .apply_sgd_update(&mut self.beta, learning_rate);
}
```

**With optimizer (e.g., Adam):**

```rust
fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
    self.grad_gamma
        .apply_optimizer_update(&mut self.gamma, optimizer);
    self.grad_beta
        .apply_optimizer_update(&mut self.beta, optimizer);
}
```

**Gradient zeroing:** The `GradientAccumulator::apply_sgd_update` and `apply_optimizer_update` methods zero the accumulated gradients after applying the update, preventing accumulation across training steps.

**Note on running statistics:** Running mean and running variance are **not** trainable parameters — they are updated via the EMA formula during training, not via gradient descent.

## Initialization

### Gamma and Beta Initialization

Batch normalization parameters are initialized to represent the identity function:

```
γ = 1.0  (identity scale: no change to variance)
β = 0.0  (no shift: zero mean)
```

**Rationale:** With this initialization, at the start of training:

```
y_ij = 1.0 × x̂_ij + 0.0 = x̂_ij
```

The output is exactly the normalized value. This gives the network a well-conditioned starting point, with all features having approximately zero mean and unit variance. As training progresses, γ and β adapt to the optimal scale and shift for the task.

**Running statistics initialization:**

```
running_mean = 0.0  (zero mean: reasonable prior)
running_var = 0.0   (zero variance: will grow as batches are processed)
```

Note: Starting running_var at 0.0 means the first inference (before any training) would divide by sqrt(0 + ε) ≈ √(1e-5) ≈ 0.003, producing very large normalized values. Always train for at least a few steps before running inference.

**Rust implementation:**

```rust
Self {
    // Initialize gamma to 1.0 (identity scaling), beta to 0.0 (no shift)
    gamma: vec![1.0f32; size],
    beta: vec![0.0f32; size],

    // Zero-initialize gradients
    grad_gamma: GradientAccumulator::new(size),
    grad_beta: GradientAccumulator::new(size),

    // Zero-initialize running statistics
    running_mean: RefCell::new(vec![0.0f32; size]),
    running_var: RefCell::new(vec![0.0f32; size]),

    // Initialize caches (will be resized during forward pass)
    cached_mean: RefCell::new(Vec::new()),
    cached_var: RefCell::new(Vec::new()),
    cached_normalized: RefCell::new(Vec::new()),
    cached_std: RefCell::new(Vec::new()),
}
```

## Numerical Considerations

### Epsilon (ε) and Division by Zero

The most critical numerical issue in batch normalization is division by nearly-zero standard deviation.

**Without epsilon:**

```
std_j = sqrt(σ²_j)  ← Can be exactly 0!
x̂_ij = (x_ij - μ_j) / 0.0  ← NaN or Inf!
```

**With epsilon:**

```
std_j = sqrt(σ²_j + ε)  ← Always positive
x̂_ij = (x_ij - μ_j) / std_j  ← Finite and bounded
```

**When does σ² ≈ 0 happen?**
- Feature is constant across the batch (e.g., all zeros from a dead ReLU)
- Small batch sizes with homogeneous data
- Early training when weights are initialized to produce similar outputs

**Typical epsilon values:**
- `1e-5` (0.00001): Most common, good balance of stability vs precision
- `1e-8`: More precise but may not prevent instability with very small variances
- `1e-3`: More robust but adds more bias to the normalized values

```rust
let std: Vec<f32> = batch_var
    .iter()
    .map(|&v| (v + self.epsilon).sqrt())  // epsilon prevents division by zero
    .collect();
```

### Small Batch Sizes

Batch normalization statistics become unreliable with very small batches:

- **B=1:** Variance is always 0! Mean = the single sample value. x̂ = 0 always. BatchNorm collapses to a trivial mapping.
- **B=2:** High variance in statistics, gradient noise is large
- **B<8:** Generally not recommended for BatchNorm; consider Layer Normalization instead

**Rule of thumb:** Use batch sizes ≥ 16, ideally ≥ 32, for stable BatchNorm statistics.

### Numerical Stability of the Backward Pass

The backward pass computes `1 / std[j]` multiple times. Since std_j > 0 (guaranteed by ε > 0), this is safe. However, if std_j is very small (close to ε^0.5 ≈ 0.003 for ε=1e-5), the gradients can be very large:

```
∂L/∂x̂_ij / std_j ≈ ∂L/∂x̂_ij / 0.003 = 333 × ∂L/∂x̂_ij
```

This amplification can cause exploding gradients when features are nearly constant within a batch.

**Mitigation:**
- Use gradient clipping at the optimizer level
- Ensure batch diversity (shuffle data before batching)
- Use slightly larger ε (1e-4 or 1e-3) for problematic layers

### Running Statistics Warmup

At the start of training, running statistics are zero-initialized. They converge over time:

```
After 1 batch:  running = 0.9 × 0 + 0.1 × batch = 0.1 × batch
After 2 batches: running ≈ 0.1 × batch₂ + 0.09 × batch₁
After k batches: running ≈ Σᵢ (0.9^(k-i) × (1-0.9) × batchᵢ)
```

With momentum = 0.9, the effective number of past batches in the running average is approximately:
```
1 / (1 - momentum) = 1 / 0.1 = 10 batches
```

After ~50 batches, the running statistics have converged (99.5% of weight from recent batches).

**Best practices:**
- Do not evaluate or run inference until running statistics have converged
- For fine-tuning pre-trained models, running statistics may already be well-calibrated
- With very high momentum (0.999), convergence takes ~1000 batches

### Potential Numerical Issues Summary

**1. NaN from zero variance:**
- Cause: All batch samples identical for a feature
- Symptom: NaN in output and gradients
- Fix: Ensure ε > 0, check for dead neurons/features

**2. Inf from very large normalized values:**
- Cause: Running variance near zero during inference
- Symptom: Inf output in inference mode
- Fix: Warm up running statistics before inference

**3. Gradient explosion through normalization:**
- Cause: Very small std_j amplifies gradients in backward pass
- Symptom: Loss diverges, very large gradient norms
- Fix: Gradient clipping, larger ε, check batch diversity

**4. Incorrect mode during evaluation:**
- Cause: Forgetting to call `set_training(false)` before evaluation
- Symptom: Evaluation accuracy varies between runs, worse than training accuracy
- Fix: Always switch to inference mode for evaluation

**Gradient correctness verification:**

```
numerical_grad = (L(γ + ε·eⱼ) - L(γ - ε·eⱼ)) / (2ε)
```

Compare with analytical gradient ∂L/∂γ_j. For ε=1e-4, they should match within ~10⁻⁴ relative error.

## Summary

Batch normalization applies per-feature normalization across the batch, then rescales with learnable parameters:

**Forward pass (training):**

```
μ_j = (1/m) × Σᵢ x_ij                          → batch mean per feature
σ²_j = (1/m) × Σᵢ (x_ij - μ_j)²                → batch variance per feature
x̂_ij = (x_ij - μ_j) / sqrt(σ²_j + ε)           → normalize
y_ij = γ_j × x̂_ij + β_j                          → scale and shift
running_μ = α × running_μ + (1-α) × μ            → update EMA
running_σ² = α × running_σ² + (1-α) × σ²         → update EMA
```

**Backward pass (training):**

```
∂L/∂β_j = (1/m) × Σᵢ ∂L/∂y_ij                                        → beta gradient
∂L/∂γ_j = (1/m) × Σᵢ (∂L/∂y_ij × x̂_ij)                               → gamma gradient
∂L/∂x̂_ij = ∂L/∂y_ij × γ_j                                              → grad through scale
∂L/∂σ²_j = Σᵢ (∂L/∂x̂_ij × x̂_ij × (-0.5) / std_j)                     → grad through variance
∂L/∂μ_j = Σᵢ (∂L/∂x̂_ij × (-1/std_j)) + ∂L/∂σ²_j × (-2/m) × Σᵢ(x-μ)  → grad through mean
∂L/∂x_ij = ∂L/∂x̂_ij/std_j + ∂L/∂σ²_j × 2(x-μ)/m + ∂L/∂μ_j/m        → input gradient
```

**Key implementation details:**
- ε prevents division by zero when variance is near zero
- Running statistics (EMA) enable deterministic inference
- RefCell provides interior mutability for updating statistics during `forward(&self)`
- Cached x̂, μ, σ², std from forward pass are required for the backward pass
- Batch size < 8 leads to unreliable statistics; consider LayerNorm as an alternative
- Always call `set_training(false)` before evaluation/inference

This forms a critical component for training deep networks efficiently. Batch normalization allows higher learning rates, reduces sensitivity to initialization, and provides mild regularization.

## Related Documentation

**Alternative Normalization:**
- [Layer Normalization](layernorm_layer.md) - Normalizes across features instead of batch, suitable for small batches and sequence models

**Preceding/Following Layers:**
- [Dense Layer](dense_layer.md) - Often used before or after BatchNorm in MLP architectures
- [Convolutional Layer](conv2d_layer.md) - BatchNorm commonly applied after Conv2D in CNNs

**Core Architecture:**
- [Backpropagation Overview](README.md) - General backpropagation concepts and notation
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface implementation
- [Batch Norm Source](../../src/layers/batchnorm.rs) - Full Rust implementation

**References:**
- Ioffe, S., & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). ICML.
- Ba, J., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv.
