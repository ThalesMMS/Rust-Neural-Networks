# Dense Layer Mathematics

This document provides a comprehensive explanation of the mathematics behind the Dense (Fully Connected) layer, covering both forward and backward propagation with detailed derivations.

## Table of Contents

- [Overview](#overview)
- [Forward Pass](#forward-pass)
  - [Mathematical Definition](#mathematical-definition)
  - [Dimension Analysis](#dimension-analysis)
  - [BLAS Implementation](#blas-implementation)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass](#backward-pass)
  - [Gradient Computation](#gradient-computation)
  - [Chain Rule Application](#chain-rule-application)
  - [Gradient Formulas](#gradient-formulas)
  - [Implementation Notes](#implementation-notes)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Numerical Considerations](#numerical-considerations)

## Overview

A Dense Layer (also called Fully Connected or Linear layer) is the fundamental building block of neural networks. It performs an affine transformation on the input, mapping it from one feature space to another through learned weights and biases.

**Key characteristics:**
- Every input feature connects to every output feature
- Parameters: weight matrix W and bias vector b
- Enables learning arbitrary linear transformations
- Forms the basis for multi-layer perceptrons (MLPs)

## Forward Pass

### Mathematical Definition

The forward pass of a dense layer computes:

```
y = xW + b
```

Where:
- **x**: Input matrix of shape (batch_size, input_size)
- **W**: Weight matrix of shape (input_size, output_size)
- **b**: Bias vector of shape (output_size,)
- **y**: Output matrix of shape (batch_size, output_size)

**Expanded form for a single sample:**

For input vector x = [x₁, x₂, ..., xₙ] and output vector y = [y₁, y₂, ..., yₘ]:

```
yⱼ = Σᵢ (xᵢ · Wᵢⱼ) + bⱼ    for j = 1, 2, ..., m
```

Or in matrix notation:

```
┌    ┐   ┌              ┐ ┌          ┐   ┌    ┐
│ y₁ │   │ x₁  x₂  ...xₙ│ │ W₁₁ W₁₂ │   │ b₁ │
│ y₂ │ = │              │ │ W₂₁ W₂₂ │ + │ b₂ │
│ .. │   │              │ │ ..  ..  │   │ .. │
│ yₘ │   │              │ │ Wₙ₁ Wₙₘ │   │ bₘ │
└    ┘   └              ┘ └          ┘   └    ┘
```

### Dimension Analysis

**Single Sample:**
- Input: (1, n) where n = input_size
- Weights: (n, m) where m = output_size
- Biases: (m,)
- Output: (1, m)

**Batched Computation:**
- Input: (B, n) where B = batch_size
- Weights: (n, m)
- Biases: (m,) - broadcasted across batch
- Output: (B, m)

**Operation breakdown:**

```
Step 1: Matrix multiplication
  x @ W → (B, n) @ (n, m) = (B, m)

Step 2: Bias addition (broadcasted)
  (B, m) + (m,) = (B, m)
```

**Example with concrete dimensions:**

For a layer mapping 784 inputs to 512 outputs with batch size 32:
- Input x: (32, 784)
- Weights W: (784, 512)
- Biases b: (512,)
- Output y: (32, 784) @ (784, 512) + (512,) = (32, 512)

### BLAS Implementation

The forward pass uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix multiplication via the `sgemm` (Single-precision GEneral Matrix Multiply) routine.

**BLAS sgemm signature:**

```
C := alpha * op(A) * op(B) + beta * C
```

**For dense layer forward pass:**

```rust
// Compute: output = input @ weights + biases
// Dimensions: (B, n) @ (n, m) = (B, m)
sgemm(
    Layout::RowMajor,
    Transpose::None,      // Don't transpose input
    Transpose::None,      // Don't transpose weights
    B,                    // M: number of rows in input (batch_size)
    m,                    // N: number of cols in weights (output_size)
    n,                    // K: number of cols in input / rows in weights
    1.0,                  // alpha
    input,                // A matrix
    n,                    // lda: leading dimension of A
    weights,              // B matrix
    m,                    // ldb: leading dimension of B
    0.0,                  // beta (overwrite output, don't accumulate)
    output,               // C matrix
    m                     // ldc: leading dimension of C
);
```

**Matrix layout in memory (row-major):**

For a 2×3 matrix:
```
┌       ┐
│ a b c │  →  [a, b, c, d, e, f]  (contiguous in memory)
│ d e f │
└       ┘
```

**Leading dimension:** The number of elements between consecutive rows. For row-major layout, `ld = number of columns`.

**Performance benefits:**
- BLAS libraries (Accelerate on macOS, OpenBLAS on Linux) are highly optimized
- Utilizes CPU vectorization (SIMD instructions)
- Cache-aware memory access patterns
- Typical speedup: 10-100× over naive loops
- Enables efficient multi-threading

### Computational Complexity

**Time Complexity:**

Matrix multiplication dominates the forward pass:
```
O(B × n × m)
```

Where:
- B = batch_size
- n = input_size
- m = output_size

**Space Complexity:**

```
Parameters: O(n × m + m) = O(n × m)
Activations: O(B × m)
```

**Example calculation:**

For layer 784→512 with batch size 32:
- Operations: 32 × 784 × 512 ≈ 12.8 million FLOPs
- Parameters: 784 × 512 + 512 ≈ 401K parameters
- Memory (forward): 32 × 512 × 4 bytes ≈ 64 KB

## Backward Pass

### Gradient Flow Visualization

The following diagram illustrates how gradients flow backward through the dense layer during backpropagation:

```
                        FORWARD PASS  |  BACKWARD PASS
                                     |
         Input x                     |                    ∂L/∂x
         (B, n)                      |                    (B, n)
            │                        |                       ▲
            │                        |                       │
            ▼                        |                       │
      ┌──────────┐                  |                 ┌─────┴─────┐
      │  x @ W   │◄─────── W ───────┼─────────────────┤  ∂L/∂y @  │
      │          │         (n, m)    |      ∂L/∂W      │    W^T    │
      └────┬─────┘                   |      (n, m)     └───────────┘
           │                         |         ▲              ▲
           │                         |         │              │
           ▼                         |         │              │
      ┌─────────┐                   |    ┌────┴─────┐        │
      │  + b    │◄─────── b ────────┼────┤  x^T @   │        │
      │         │         (m)        |    │  ∂L/∂y   │        │
      └────┬────┘                    |    └──────────┘        │
           │                         |         ▲              │
           │                         |         │              │
           ▼                         |         │              │
        Output y                     |     ∂L/∂b          ∂L/∂y
         (B, m)                      |      (m)           (B, m)
            │                        |      ▲              ▲
            ▼                        |      │              │
          Loss                       |      └──────┬───────┘
                                     |             │
                                     |        sum over
                                     |       batch dim

Legend:
  ──►  Forward data flow
  ◄──  Parameter (weights/biases)
  ──▲  Backward gradient flow
  @    Matrix multiplication
  ^T   Transpose
```

**Key insight:** Each parameter contributes to the output through a specific mathematical operation, and its gradient is computed by applying the chain rule to that operation.

**Gradient computation summary:**
- **∂L/∂x = ∂L/∂y @ W^T**: Gradient flows back through weights (transposed)
- **∂L/∂W = x^T @ ∂L/∂y**: Outer product of input and output gradient
- **∂L/∂b = sum(∂L/∂y, axis=0)**: Sum output gradients across batch dimension

### Gradient Computation

During backpropagation, we receive the gradient of the loss with respect to the layer's output (∂L/∂y) and must compute:

1. **Gradient w.r.t. input (∂L/∂x)**: Needed to propagate error to previous layer
2. **Gradient w.r.t. weights (∂L/∂W)**: Needed to update weights
3. **Gradient w.r.t. biases (∂L/∂b)**: Needed to update biases

### Chain Rule Application

The chain rule states:

```
∂L/∂x = ∂L/∂y · ∂y/∂x
```

We need to find the partial derivatives of the output y with respect to each input variable.

**Recall the forward pass:**
```
yⱼ = Σᵢ (xᵢ · Wᵢⱼ) + bⱼ
```

**Partial derivatives:**

1. **∂y/∂x**: How output changes with input
   ```
   ∂yⱼ/∂xᵢ = Wᵢⱼ
   ```

2. **∂y/∂W**: How output changes with weights
   ```
   ∂yⱼ/∂Wᵢⱼ = xᵢ
   ```

3. **∂y/∂b**: How output changes with biases
   ```
   ∂yⱼ/∂bⱼ = 1
   ```

### Gradient Formulas

#### 1. Gradient w.r.t. Input (∂L/∂x)

**Formula:**
```
∂L/∂x = (∂L/∂y) @ Wᵀ
```

**Derivation:**

For a single element:
```
∂L/∂xᵢ = Σⱼ (∂L/∂yⱼ · ∂yⱼ/∂xᵢ)
       = Σⱼ (∂L/∂yⱼ · Wᵢⱼ)
```

In matrix form:
```
∂L/∂x = (∂L/∂y) @ Wᵀ
```

**Dimensions:**
```
(B, n) = (B, m) @ (m, n)
```

**BLAS implementation:**
```rust
// grad_input = grad_output @ weights^T
sgemm(
    Layout::RowMajor,
    Transpose::None,      // Don't transpose grad_output
    Transpose::Ordinary,  // Transpose weights
    B,                    // M: batch_size
    n,                    // N: input_size
    m,                    // K: output_size
    1.0,                  // alpha
    grad_output,          // A: (B, m)
    m,                    // lda
    weights,              // B: (n, m), used as (m, n)^T
    m,                    // ldb
    0.0,                  // beta
    grad_input,           // C: (B, n)
    n                     // ldc
);
```

#### 2. Gradient w.r.t. Weights (∂L/∂W)

**Formula:**
```
∂L/∂W = xᵀ @ (∂L/∂y)
```

**Derivation:**

For a single weight element:
```
∂L/∂Wᵢⱼ = Σₖ (∂L/∂yₖⱼ · ∂yₖⱼ/∂Wᵢⱼ)
        = Σₖ (∂L/∂yₖⱼ · xₖᵢ)    [summing over batch]
```

In matrix form:
```
∂L/∂W = xᵀ @ (∂L/∂y)
```

**Dimensions:**
```
(n, m) = (n, B) @ (B, m)
```

**BLAS implementation:**
```rust
// grad_weights = input^T @ grad_output
sgemm(
    Layout::RowMajor,
    Transpose::Ordinary,  // Transpose input
    Transpose::None,      // Don't transpose grad_output
    n,                    // M: input_size
    m,                    // N: output_size
    B,                    // K: batch_size
    1.0,                  // alpha
    input,                // A: (B, n), used as (n, B)^T
    n,                    // lda
    grad_output,          // B: (B, m)
    m,                    // ldb
    1.0,                  // beta (accumulate gradients)
    grad_weights,         // C: (n, m)
    m                     // ldc
);
```

**Note:** The beta=1.0 parameter accumulates gradients across batches rather than overwriting.

#### 3. Gradient w.r.t. Biases (∂L/∂b)

**Formula:**
```
∂L/∂b = Σₖ (∂L/∂yₖ)    [sum over batch dimension]
```

**Derivation:**

For a single bias element:
```
∂L/∂bⱼ = Σₖ (∂L/∂yₖⱼ · ∂yₖⱼ/∂bⱼ)
       = Σₖ (∂L/∂yₖⱼ · 1)
       = Σₖ ∂L/∂yₖⱼ
```

**Dimensions:**
```
(m,) = sum over batch of (B, m) → (m,)
```

**Implementation:**
```rust
// Sum grad_output across batch dimension
for batch_idx in 0..B {
    for j in 0..m {
        grad_biases[j] += grad_output[batch_idx * m + j];
    }
}
```

### Implementation Notes

**Gradient accumulation:**

In mini-batch training, gradients are accumulated across batches before parameter updates. This is why `grad_weights` and `grad_biases` use `+=` operations.

**Memory efficiency:**

The backward pass reuses the cached input from the forward pass, avoiding redundant storage:
```rust
// Forward: cache input for backward pass
cached_input = input.to_vec();

// Backward: reuse cached input
grad_weights = cached_input^T @ grad_output;
```

**Numerical stability:**

- Gradients are computed in single precision (f32) for efficiency
- Gradient clipping may be applied at the optimizer level to prevent exploding gradients
- Batch normalization or layer normalization can stabilize gradient flow

## Parameter Updates

After computing gradients, parameters are updated using gradient descent (or variants like Adam, SGD with momentum):

**Vanilla gradient descent:**
```
W := W - η · (∂L/∂W)
b := b - η · (∂L/∂b)
```

Where η is the learning rate.

**Implementation:**
```rust
pub fn update_parameters(&mut self, learning_rate: f32) {
    let grad_weights = self.grad_weights.borrow();
    let grad_biases = self.grad_biases.borrow();

    // Update weights: W -= learning_rate * grad_W
    for i in 0..self.weights.len() {
        self.weights[i] -= learning_rate * grad_weights[i];
    }

    // Update biases: b -= learning_rate * grad_b
    for i in 0..self.biases.len() {
        self.biases[i] -= learning_rate * grad_biases[i];
    }
}
```

**Gradient zeroing:**

After each parameter update, gradients must be zeroed to prevent accumulation across training steps:
```rust
grad_weights.fill(0.0);
grad_biases.fill(0.0);
```

## Initialization

### Xavier (Glorot) Initialization

The layer uses Xavier initialization for weights to maintain variance across layers and prevent vanishing/exploding gradients.

**Formula:**
```
W ~ Uniform(-limit, limit)
where limit = sqrt(6 / (input_size + output_size))
```

**Rationale:**

For a layer with n inputs and m outputs:
- Forward pass variance: Var(y) ≈ n · Var(x) · Var(W)
- To keep Var(y) ≈ Var(x), we want Var(W) ≈ 1/n

Xavier initialization balances both forward and backward pass variance by using:
```
Var(W) = 2 / (n + m)
```

For uniform distribution on [-a, a]:
```
Var = a² / 3
a² / 3 = 2 / (n + m)
a = sqrt(6 / (n + m))
```

**Bias initialization:**

Biases are initialized to zero:
```
b = 0
```

This is standard practice as the weights provide sufficient initial randomness.

**Implementation:**
```rust
let limit = (6.0f32 / (input_size + output_size) as f32).sqrt();
for value in &mut weights {
    *value = rng.gen_range_f32(-limit, limit);
}
```

### Alternative Initialization Schemes

**He initialization (for ReLU):**
```
limit = sqrt(2 / input_size)
```

Designed specifically for ReLU activations to account for the fact that half the neurons are inactive.

**LeCun initialization (for tanh):**
```
limit = sqrt(1 / input_size)
```

Suitable for tanh activations with different variance properties.

## Numerical Considerations

### Potential Issues

**1. Vanishing gradients:**
- If weights are too small, gradients shrink exponentially through layers
- Mitigation: Proper initialization (Xavier/He), batch normalization, residual connections

**2. Exploding gradients:**
- If weights are too large, gradients grow exponentially
- Mitigation: Gradient clipping, proper initialization, lower learning rates

**3. Loss of precision:**
- Single precision (f32) can lose accuracy with very large/small numbers
- Typical dynamic range: ~10⁻³⁸ to 10³⁸
- Mitigation: Gradient scaling, batch normalization

**4. Dead neurons:**
- If gradients become zero, weights stop updating
- Common with ReLU when neurons always output zero
- Mitigation: Leaky ReLU, proper initialization, lower learning rates

### Best Practices

**During forward pass:**
- Use BLAS for efficiency
- Cache inputs for backward pass
- Apply activation functions after dense layer

**During backward pass:**
- Check for NaN/Inf values in gradients
- Implement gradient clipping if needed
- Use beta=1.0 in sgemm for gradient accumulation

**During training:**
- Monitor gradient norms
- Use appropriate learning rates
- Apply regularization (L1/L2) if needed
- Consider learning rate schedules

### Debugging Tips

**Check gradient correctness:**

Numerical gradient checking:
```
numerical_grad = (loss(W + ε) - loss(W - ε)) / (2ε)
```

Compare with analytical gradient from backprop. Should match within ~10⁻⁵ for ε=10⁻⁴.

**Monitor gradient statistics:**
- Mean absolute gradient value
- Gradient norm: ||∇W||₂
- Ratio of update to parameter magnitude

**Verify dimensions:**
```
assert_eq!(grad_input.shape, input.shape);
assert_eq!(grad_weights.shape, weights.shape);
assert_eq!(grad_biases.shape, biases.shape);
```

## Summary

The dense layer performs the affine transformation `y = xW + b`:

**Forward pass:**
- Matrix multiplication: O(B × n × m) operations
- BLAS-accelerated for efficiency
- Output shape: (batch_size, output_size)

**Backward pass:**
- Three gradients computed using chain rule
- ∂L/∂x = (∂L/∂y) @ Wᵀ - propagates to previous layer
- ∂L/∂W = xᵀ @ (∂L/∂y) - updates weights
- ∂L/∂b = sum(∂L/∂y) - updates biases

**Key implementation details:**
- Xavier initialization prevents gradient issues
- BLAS library (Accelerate/OpenBLAS) provides optimal performance
- Gradient accumulation supports mini-batch training
- RefCell allows interior mutability for gradient storage

This forms the foundation for understanding more complex layers like convolutional and attention-based architectures.

## Related Documentation

**Activation Functions:**
- [Activation Functions](../activation_functions.md) - Detailed mathematical documentation for ReLU, softmax, sigmoid, and other activations used with dense layers

**Alternative Layer Types:**
- [Convolutional Layer](conv2d_layer.md) - Spatially-aware layers for image processing with parameter sharing
- [Attention Mechanism](attention_mechanism.md) - Self-attention layers for sequence modeling with dynamic weighting

**Core Architecture:**
- [Backpropagation Overview](README.md) - General backpropagation concepts and notation
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface implementation
