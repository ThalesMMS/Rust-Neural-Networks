# Patch Embedding Layer Mathematics

This document provides the mathematical derivation of the patch embedding layer's forward and backward passes as used in the Vision Transformer (ViT) architecture.

## Table of Contents

- [Overview](#overview)
- [Forward Pass](#forward-pass)
  - [Mathematical Definition](#mathematical-definition)
  - [Dimension Analysis](#dimension-analysis)
- [Backward Pass](#backward-pass)
  - [Gradient w.r.t. Input](#gradient-wrt-input)
  - [Gradient w.r.t. Weights](#gradient-wrt-weights)
  - [Gradient w.r.t. Biases](#gradient-wrt-biases)
- [Implementation Notes](#implementation-notes)
- [Connection to ViT Architecture](#connection-to-vit-architecture)

## Overview

The Patch Embedding layer is the first learnable component in the Vision Transformer. It takes flattened image patches and projects them to the model dimension via a linear transformation. In our implementation, `PatchEmbeddingLayer` wraps a `DenseLayer`, so the mathematical operations are identical to a standard dense layer applied to patch tokens.

**Key characteristics:**
- Thin wrapper around `DenseLayer` (composition, not reimplementation)
- Projects each flattened patch independently to the model dimension
- Uses Xavier initialization and BLAS-accelerated matrix multiplication
- Applied to all patches in a batch simultaneously

## Forward Pass

### Mathematical Definition

The forward pass computes:

```
y = xW + b
```

Where:
- **x**: Input patches, shape (N, patch_dim) where N = batch_size * num_patches
- **W**: Weight matrix, shape (patch_dim, d_model)
- **b**: Bias vector, shape (d_model,)
- **y**: Output embeddings, shape (N, d_model)

For CIFAR-10 with 4x4 patches:
- patch_dim = 4 * 4 * 3 = 48 (height * width * channels)
- d_model = 128
- num_patches = 8 * 8 = 64 (for 32x32 images)

### Dimension Analysis

For a single image:
```
Input:  [64 patches, 48 features]
Weight: [48, 128]
Bias:   [128]
Output: [64 patches, 128 features]
```

For a batch of B images:
```
Input:  [B * 64, 48]
Weight: [48, 128]
Bias:   [128]
Output: [B * 64, 128]
```

The key insight is that each patch is projected independently — the same weight matrix W and bias b are shared across all 64 patches and all images in the batch.

**Expanded form for a single patch vector x = [x_1, x_2, ..., x_48]:**

```
y_j = b_j + sum_{i=1}^{48} x_i * W_{ij}    for j = 1, ..., 128
```

## Backward Pass

Given the loss gradient w.r.t. the output dL/dy of shape (N, d_model), we need three gradients:

### Gradient w.r.t. Input

To propagate gradients to preceding layers (though patch embedding is typically the first layer):

```
dL/dx = dL/dy * W^T
```

**Dimensions:**
```
dL/dy: (N, d_model)     = (N, 128)
W^T:   (d_model, patch_dim) = (128, 48)
dL/dx: (N, patch_dim)    = (N, 48)
```

**Derivation using the chain rule:**

Since y_j = sum_i x_i * W_ij + b_j, we have:

```
dy_j/dx_i = W_ij
```

Therefore:

```
dL/dx_i = sum_{j=1}^{d_model} (dL/dy_j) * (dy_j/dx_i)
        = sum_{j=1}^{d_model} (dL/dy_j) * W_ij
```

In matrix form: `dL/dx = dL/dy * W^T`

### Gradient w.r.t. Weights

```
dL/dW = x^T * dL/dy / N
```

**Dimensions:**
```
x^T:   (patch_dim, N)    = (48, N)
dL/dy: (N, d_model)      = (N, 128)
dL/dW: (patch_dim, d_model) = (48, 128)
```

The division by N (batch_size * num_patches) averages the gradient over all tokens in the batch, consistent with the DenseLayer implementation.

**Derivation:**

```
dL/dW_ij = sum_{n=1}^{N} x_n_i * (dL/dy_n_j) / N
```

In matrix form: `dL/dW = x^T * dL/dy / N`

### Gradient w.r.t. Biases

```
dL/db = sum_{n=1}^{N} dL/dy_n / N
```

**Dimensions:**
```
dL/dy: (N, d_model) = (N, 128)
dL/db: (d_model,)   = (128,)
```

This is the column-wise sum of the gradient matrix, averaged over the batch.

**Derivation:**

Since y_j = ... + b_j (bias appears linearly), we have:

```
dL/db_j = sum_{n=1}^{N} (dL/dy_n_j) / N
```

## Implementation Notes

### BLAS Acceleration

All three gradient computations delegate to the wrapped `DenseLayer`, which uses BLAS `sgemm` for matrix multiplication:

1. **Weight gradient**: `sgemm(x^T, dL/dy)` with alpha=1/N, beta=1.0 (accumulate)
2. **Input gradient**: `sgemm(dL/dy, W^T)` with alpha=1.0, beta=0.0
3. **Bias gradient**: Column-wise sum + scale by 1/N

### Gradient Accumulation

The `DenseLayer` uses `GradientAccumulator` (from `src/layers/gradient.rs`) which:
- Accumulates gradients across multiple backward calls
- Clears gradients after each parameter update
- Supports both SGD and optimizer-based updates

### Xavier Initialization

Weights are initialized using Xavier uniform initialization:

```
limit = sqrt(6 / (patch_dim + d_model))
W_ij ~ Uniform(-limit, limit)
```

For patch_dim=48, d_model=128:
```
limit = sqrt(6 / 176) = 0.1846
```

## Connection to ViT Architecture

In the full ViT pipeline, the patch embedding gradient receives its signal from the transformer encoder via the chain rule:

```
Input Image
  |
  v
[Patch Extraction] -- no learnable params, no gradient needed
  |
  v
[Patch Embedding] -- dL/dW computed here
  |
  v
[ReLU] -- gradient masked where pre-activation <= 0
  |
  v
[+ Positional Encoding] -- gradient passes through unchanged
  |
  v
[Transformer Encoder] -- sends dL/dy back to patch embedding
  |
  v
[Mean Pool -> Classifier -> Softmax -> Loss]
```

The gradient flow from loss to patch embedding passes through:
1. Cross-entropy + softmax: `dL/d_logits = probs - one_hot(label)`
2. Classifier (DenseLayer): `dL/d_pooled = dL/d_logits * W_cls^T`
3. Mean pool: `dL/d_token_i = dL/d_pooled / num_patches` (for all i)
4. Transformer encoder: complex multi-layer backprop
5. Positional encoding: identity (additive, gradient passes through)
6. ReLU: `dL/d_pre_relu = dL/d_post_relu * (pre_relu > 0 ? 1 : 0)`
7. **Patch embedding**: `dL/dW = patches^T * dL/d_embeddings / N`
