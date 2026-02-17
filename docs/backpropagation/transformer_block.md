# Transformer Block Mathematics

This document provides a comprehensive explanation of the mathematics behind the Transformer Block, covering both forward and backward propagation with detailed derivations. This implementation follows the Pre-LN (Pre-Layer Normalization) architecture from "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020), building on the original "Attention Is All You Need" (Vaswani et al., 2017).

## Table of Contents

- [Overview](#overview)
- [Pre-LN Architecture Diagram](#pre-ln-architecture-diagram)
- [Sub-Component Summary](#sub-component-summary)
- [Forward Pass](#forward-pass)
  - [1. First Layer Normalization (LN1)](#1-first-layer-normalization-ln1)
  - [2. Multi-Head Self-Attention](#2-multi-head-self-attention)
  - [3. First Residual Connection](#3-first-residual-connection)
  - [4. Second Layer Normalization (LN2)](#4-second-layer-normalization-ln2)
  - [5. Feed-Forward Network (FFN)](#5-feed-forward-network-ffn)
  - [6. Second Residual Connection](#6-second-residual-connection)
  - [Complete Forward Pass Summary](#complete-forward-pass-summary)
- [Backward Pass](#backward-pass)
  - [Gradient Flow Overview](#gradient-flow-overview)
  - [Step 1: Gradient Through Second Residual Connection](#step-1-gradient-through-second-residual-connection)
  - [Step 2: Gradient Through FFN Layer 2](#step-2-gradient-through-ffn-layer-2)
  - [Step 3: Gradient Through ReLU](#step-3-gradient-through-relu)
  - [Step 4: Gradient Through FFN Layer 1](#step-4-gradient-through-ffn-layer-1)
  - [Step 5: Gradient Through Second Layer Norm](#step-5-gradient-through-second-layer-norm)
  - [Step 6: Gradient Accumulation at Intermediate State](#step-6-gradient-accumulation-at-intermediate-state)
  - [Step 7: Gradient Through First Residual Connection](#step-7-gradient-through-first-residual-connection)
  - [Step 8: Gradient Through Multi-Head Attention](#step-8-gradient-through-multi-head-attention)
  - [Step 9: Gradient Through First Layer Norm](#step-9-gradient-through-first-layer-norm)
  - [Step 10: Gradient Accumulation at Input](#step-10-gradient-accumulation-at-input)
  - [Complete Backward Pass Summary](#complete-backward-pass-summary)
- [Pre-LN vs Post-LN Comparison](#pre-ln-vs-post-ln-comparison)
  - [Architectural Differences](#architectural-differences)
  - [Mathematical Implications](#mathematical-implications)
  - [Training Stability Analysis](#training-stability-analysis)
- [Parameter Count Analysis](#parameter-count-analysis)
- [Stacked Transformer Encoder](#stacked-transformer-encoder)
- [Numerical Considerations](#numerical-considerations)
- [Implementation References](#implementation-references)

## Overview

The Transformer Block is the core building block of Transformer architectures. It composes multi-head self-attention with a position-wise feed-forward network, layer normalization, and residual connections.

**Key characteristics:**
- Pre-LN: layer normalization applied *before* each sub-layer for better gradient flow
- Residual connections: allow gradients to bypass sub-layers, preventing vanishing gradients
- Position-wise FFN: applies the same two-layer MLP independently to each token
- Permutation-equivariant: attention treats each token equally regardless of position

**TransformerBlock Configuration (typical example):**
- **d_model:** 64 — token embedding dimension (input/output feature size)
- **num_heads:** 4 — number of attention heads (d_model must be divisible by num_heads)
- **d_ff:** 256 — hidden dimension of feed-forward network (typically 4 × d_model)
- **ε (epsilon):** 1e-5 — layer normalization numerical stability constant

**Input/Output shapes (flattened 3D tensors):**
- Input: `(B × S × d_model)` flattened to `(B*S*d_model,)` — batch of token sequences
- Output: `(B × S × d_model)` flattened to `(B*S*d_model,)` — same shape as input

Where:
- **B** = batch_size (number of sequences)
- **S** = seq_len (number of tokens per sequence)
- **d_model** = token embedding dimension

## Pre-LN Architecture Diagram

```
                    input  x  (B, S, d_model)
                      │
          ┌───────────┤ (residual shortcut)
          │           │
          │        LayerNorm (LN1)
          │           │
          │    Multi-Head Attention
          │           │
          │           ▼
          └──────► Add (+) ──────► residual1  (B, S, d_model)
                      │
          ┌───────────┤ (residual shortcut)
          │           │
          │        LayerNorm (LN2)
          │           │
          │      Dense (d_model → d_ff)
          │           │
          │         ReLU
          │           │
          │      Dense (d_ff → d_model)
          │           │
          │           ▼
          └──────► Add (+) ──────► output  (B, S, d_model)
```

**Data flow with cached intermediate values:**

```
x ──► [cached_input]
  │
  └─► LN1 ──► [cached_ln1_out]
                  │
                  └─► MHA ──► [cached_attn_out]
                                 │
x ──────────────────────────► Add ──► [cached_residual1]
                                         │
                                         └─► LN2 ──► [cached_ln2_out]
                                                        │
                                                        └─► FFN1 ──► [cached_ffn1_out (after ReLU)]
                                                                        │
                                                                        └─► FFN2 ──► [cached_ffn2_out]
                                                                                        │
                                        cached_residual1 ───────────────────────────► Add ──► output
```

## Sub-Component Summary

The TransformerBlock composes five learnable sub-layers:

| Sub-Layer | Type | Parameters | Input Shape | Output Shape |
|-----------|------|-----------|-------------|-------------|
| `ln1` | LayerNormLayer | 2 × d_model | (B×S, d_model) | (B×S, d_model) |
| `attention` | MultiHeadAttentionLayer | 4 × (d_model² + d_model) | (B×S, d_model) | (B×S, d_model) |
| `ln2` | LayerNormLayer | 2 × d_model | (B×S, d_model) | (B×S, d_model) |
| `ffn1` | DenseLayer | d_model × d_ff + d_ff | (B×S, d_model) | (B×S, d_ff) |
| `ffn2` | DenseLayer | d_ff × d_model + d_model | (B×S, d_ff) | (B×S, d_model) |

**Note on batch treatment:** Layer norms and FFN layers treat each token independently: batch_size is passed as `batch_size * seq_len` so each token vector is normalized/projected separately. The attention layer receives `batch_size` to process full sequences.

## Forward Pass

The forward pass processes a batch of token sequences through six sequential operations.

### 1. First Layer Normalization (LN1)

**Purpose:** Normalize each token's embedding before attention, stabilizing the attention's input distribution.

**Mathematical Formula:**

For each token embedding `x[b,s]` (a vector of length d_model):

```
μ[b,s]    = (1/d_model) × Σⱼ x[b,s,j]              (per-token mean)
σ²[b,s]   = (1/d_model) × Σⱼ (x[b,s,j] - μ[b,s])²  (per-token variance)
std[b,s]  = sqrt(σ²[b,s] + ε)                       (per-token std dev)
x̂[b,s,j]  = (x[b,s,j] - μ[b,s]) / std[b,s]          (normalized)
ln1_out[b,s,j] = γ₁ⱼ × x̂[b,s,j] + β₁ⱼ              (scale and shift)
```

**Dimensions:**
- `x` input: `(B×S, d_model)` — treated as (batch_size×seq_len) independent samples
- `μ`, `σ²`, `std`: `(B×S,)` — one statistic per token embedding
- `γ₁`, `β₁`: `(d_model,)` — learnable scale and shift (shared across all tokens)
- `ln1_out`: `(B×S, d_model)` — normalized token embeddings

**Code Reference:** `src/layers/transformer.rs` lines 232-234:
```rust
self.ln1.forward(input, &mut cached_ln1_out, batch_size * seq_len);
```

> **See:** [layernorm_layer.md](./layernorm_layer.md) for full LayerNorm forward/backward derivations.

### 2. Multi-Head Self-Attention

**Purpose:** Allow each token to attend to all other tokens in the sequence, computing weighted aggregations of value vectors.

**Mathematical Formula (scaled dot-product attention for one head):**

```
Q = ln1_out × W_Q + b_Q    (B×S, d_model) × (d_model, d_head) → (B×S, d_head)
K = ln1_out × W_K + b_K    (B×S, d_model) × (d_model, d_head) → (B×S, d_head)
V = ln1_out × W_V + b_V    (B×S, d_model) × (d_model, d_head) → (B×S, d_head)

scores = Q × K^T / sqrt(d_head)   (B, S, S) — attention score matrix
attn_weights = softmax(scores)     (B, S, S) — normalized attention weights
head_out = attn_weights × V        (B, S, d_head) — attended value aggregation
```

For `num_heads` heads, the outputs are concatenated:
```
attn_out = concat(head₀_out, ..., head_{H-1}_out)   (B×S, d_model)
```

Where `d_head = d_model / num_heads`.

**Dimensions:**
- `ln1_out` input: `(B×S, d_model)` — normalized token embeddings
- `W_Q, W_K, W_V`: `(d_model, d_head)` each — per-head projection weights
- `Q, K, V`: `(B, S, d_head)` each — per-head query, key, value
- `scores`: `(B, S, S)` — attention score matrix
- `attn_weights`: `(B, S, S)` — softmax-normalized weights
- `attn_out`: `(B×S, d_model)` — multi-head attention output

**Code Reference:** `src/layers/transformer.rs` lines 236-238:
```rust
self.attention.forward(&cached_ln1_out, &mut cached_attn_out, batch_size);
```

> **See:** [attention_mechanism.md](./attention_mechanism.md) for full attention forward/backward derivations.

### 3. First Residual Connection

**Purpose:** Add the original input to the attention output, preserving the input signal and enabling gradient bypass during backpropagation.

**Mathematical Formula:**

```
residual1[i] = input[i] + attn_out[i]    for i = 0 to B*S*d_model - 1
```

Or in tensor notation:
```
residual1 = x + attn_out    (B×S, d_model)
```

**Dimensions:**
- `x` (input): `(B×S, d_model)` — original block input
- `attn_out`: `(B×S, d_model)` — attention output
- `residual1`: `(B×S, d_model)` — first intermediate state

**Why residual connections?**
- **Gradient highway:** gradients can flow directly through the addition, bypassing sub-layers
- **Identity preservation:** the network can learn near-identity mappings easily (output = input + small perturbation)
- **Depth scalability:** enables training of very deep networks without vanishing gradients

**Code Reference:** `src/layers/transformer.rs` lines 240-243:
```rust
for i in 0..total_size {
    cached_residual1[i] = input[i] + cached_attn_out[i];
}
```

### 4. Second Layer Normalization (LN2)

**Purpose:** Normalize each token's embedding before the feed-forward network, keeping FFN inputs well-conditioned.

**Mathematical Formula:**

For each token embedding `residual1[b,s]`:

```
μ₂[b,s]   = (1/d_model) × Σⱼ residual1[b,s,j]
σ²₂[b,s]  = (1/d_model) × Σⱼ (residual1[b,s,j] - μ₂[b,s])²
std₂[b,s] = sqrt(σ²₂[b,s] + ε)
x̂₂[b,s,j] = (residual1[b,s,j] - μ₂[b,s]) / std₂[b,s]
ln2_out[b,s,j] = γ₂ⱼ × x̂₂[b,s,j] + β₂ⱼ
```

**Dimensions:**
- `residual1` input: `(B×S, d_model)`
- `γ₂`, `β₂`: `(d_model,)` — separate learnable parameters from LN1
- `ln2_out`: `(B×S, d_model)` — normalized intermediate state

**Code Reference:** `src/layers/transformer.rs` lines 245-247:
```rust
self.ln2.forward(&cached_residual1, &mut cached_ln2_out, batch_size * seq_len);
```

### 5. Feed-Forward Network (FFN)

**Purpose:** Apply a position-wise (per-token) two-layer MLP with non-linear activation. The FFN provides the model's "memory" and allows it to transform each token's representation non-linearly.

**Mathematical Formula (three sub-steps):**

**FFN Layer 1 (expansion):**
```
ffn1_pre[b,s,k] = b_ff1[k] + Σⱼ (ln2_out[b,s,j] × W_ff1[j,k])
                   for j = 0..d_model-1, k = 0..d_ff-1

ffn1_out[b,s,k] = max(0, ffn1_pre[b,s,k])    (ReLU activation)
```

**FFN Layer 2 (projection back):**
```
ffn2_out[b,s,j] = b_ff2[j] + Σₖ (ffn1_out[b,s,k] × W_ff2[k,j])
                   for k = 0..d_ff-1, j = 0..d_model-1
```

**Combined FFN formula:**
```
FFN(ln2_out) = ReLU(ln2_out × W_ff1 + b_ff1) × W_ff2 + b_ff2
```

**Dimensions:**
- `ln2_out` input: `(B×S, d_model)` — LN2 output per token
- `W_ff1`: `(d_model, d_ff)` — expansion weight matrix
- `b_ff1`: `(d_ff,)` — expansion bias
- `ffn1_pre`: `(B×S, d_ff)` — pre-activation FFN1 output
- `ffn1_out` (after ReLU): `(B×S, d_ff)` — expanded hidden representation
- `W_ff2`: `(d_ff, d_model)` — projection weight matrix
- `b_ff2`: `(d_model,)` — projection bias
- `ffn2_out`: `(B×S, d_model)` — FFN output, back to d_model dimension

**Important:** The ReLU activation is applied **after** FFN1 and **before** FFN2. The forward cache stores `ffn1_out` **after** ReLU, which is used in the backward pass to reconstruct the ReLU derivative.

**Code Reference:** `src/layers/transformer.rs` lines 249-259:
```rust
// FFN layer 1: d_model -> d_ff
self.ffn1.forward(&cached_ln2_out, &mut cached_ffn1_out, batch_size * seq_len);
// ReLU activation
Self::relu_inplace(&mut cached_ffn1_out);
// FFN layer 2: d_ff -> d_model
self.ffn2.forward(&cached_ffn1_out, &mut cached_ffn2_out, batch_size * seq_len);
```

### 6. Second Residual Connection

**Purpose:** Add the intermediate state (residual1) to the FFN output to produce the final block output.

**Mathematical Formula:**

```
output[i] = residual1[i] + ffn2_out[i]    for i = 0 to B*S*d_model - 1
```

Or in tensor notation:
```
output = residual1 + FFN(LN2(residual1))    (B×S, d_model)
```

**Dimensions:**
- `residual1`: `(B×S, d_model)` — first intermediate state
- `ffn2_out`: `(B×S, d_model)` — FFN output
- `output`: `(B×S, d_model)` — final block output

**Code Reference:** `src/layers/transformer.rs` lines 261-263:
```rust
for i in 0..total_size {
    output[i] = cached_residual1[i] + cached_ffn2_out[i];
}
```

### Complete Forward Pass Summary

Combining all six steps, the full Pre-LN Transformer Block forward pass is:

```
ln1_out   = LN1(x)                         Step 1
attn_out  = MHA(ln1_out)                    Step 2
residual1 = x + attn_out                    Step 3  ← first residual
ln2_out   = LN2(residual1)                  Step 4
ffn1_out  = ReLU(ln2_out × W_ff1 + b_ff1)  Step 5a (FFN expansion + ReLU)
ffn2_out  = ffn1_out × W_ff2 + b_ff2       Step 5b (FFN projection)
output    = residual1 + ffn2_out            Step 6  ← second residual
```

**Compact notation:**
```
output = x + MHA(LN1(x)) + FFN(LN2(x + MHA(LN1(x))))
```

Where `FFN(z) = ReLU(z × W_ff1 + b_ff1) × W_ff2 + b_ff2`.

**Dimension summary:**

| Tensor | Shape |
|--------|-------|
| `x` (input) | (B×S, d_model) |
| `ln1_out` | (B×S, d_model) |
| `attn_out` | (B×S, d_model) |
| `residual1` | (B×S, d_model) |
| `ln2_out` | (B×S, d_model) |
| `ffn1_out` | (B×S, d_ff) |
| `ffn2_out` | (B×S, d_model) |
| `output` | (B×S, d_model) |

## Backward Pass

The backward pass propagates gradients through the block in reverse order, using cached intermediate activations from the forward pass. The key challenge is correctly handling the two branching paths created by residual connections.

### Gradient Flow Overview

```
∂L/∂output
      │
      ▼                          ← Step 1: Second residual connection splits gradient
      ├──────────────────────────────────────────────────────────┐
      │                                                          │
∂L/∂ffn2_out                                             ∂L/∂residual1 (path A)
      │                                                          │
      ▼                                                          │
   FFN Layer 2 backward         ← Step 2                        │
      │                                                          │
∂L/∂ffn1_out (after ReLU)                                        │
      │                                                          │
   ReLU backward                ← Step 3                        │
      │                                                          │
∂L/∂ffn1_out (before ReLU)                                       │
      │                                                          │
   FFN Layer 1 backward         ← Step 4                        │
      │                                                          │
∂L/∂ln2_out                                                      │
      │                                                          │
   LN2 backward                 ← Step 5                        │
      │                                                          │
∂L/∂residual1 (path B)                                           │
      │                                                          │
      └──────────────────────────── + ────────────────────────────┘
                                    │                ← Step 6: Accumulate at residual1
                            ∂L/∂residual1 (total)
                                    │
                                    ▼               ← Step 7: First residual splits gradient
                      ┌─────────────┤
                      │             │
              ∂L/∂attn_out  ∂L/∂x (path A from residual1)
                      │
               MHA backward         ← Step 8
                      │
              ∂L/∂ln1_out
                      │
               LN1 backward         ← Step 9
                      │
              ∂L/∂x (path B)
                      │
                      └──────────── + ──────────── (path A)
                                    │              ← Step 10: Accumulate at input
                              ∂L/∂x (total)
```

### Step 1: Gradient Through Second Residual Connection

The second residual `output = residual1 + ffn2_out` is an element-wise addition. Gradients distribute equally to both branches:

```
∂L/∂residual1 (path A) = ∂L/∂output    (gradient passes through directly)
∂L/∂ffn2_out           = ∂L/∂output    (gradient passes through directly)
```

**Intuition:** For any `z = a + b`, `∂z/∂a = 1` and `∂z/∂b = 1`, so by chain rule, `∂L/∂a = ∂L/∂z` and `∂L/∂b = ∂L/∂z`.

**Code Reference:** `src/layers/transformer.rs` lines 308-311:
```rust
// Backward through second residual connection: output = residual1 + ffn2_out
grad_residual1.copy_from_slice(grad_output);
grad_ffn2_out.copy_from_slice(grad_output);
```

### Step 2: Gradient Through FFN Layer 2

FFN2 is a dense linear layer `ffn2_out = ffn1_out × W_ff2 + b_ff2`.

```
∂L/∂W_ff2 = ffn1_out^T × ∂L/∂ffn2_out           (d_ff × d_model)
∂L/∂b_ff2 = Σ_tokens ∂L/∂ffn2_out               (d_model,)
∂L/∂ffn1_out = ∂L/∂ffn2_out × W_ff2^T           (B×S, d_ff)
```

**Dimension checks:**
- `ffn1_out^T`: (d_ff, B×S), `∂L/∂ffn2_out`: (B×S, d_model) → `∂L/∂W_ff2`: (d_ff, d_model) ✓
- `∂L/∂ffn2_out`: (B×S, d_model), `W_ff2^T`: (d_model, d_ff) → `∂L/∂ffn1_out`: (B×S, d_ff) ✓

**Code Reference:** `src/layers/transformer.rs` lines 313-319:
```rust
self.ffn2.backward(
    &cached_ffn1_out,
    &grad_ffn2_out,
    &mut grad_ffn1_out,
    batch_size * seq_len,
);
```

### Step 3: Gradient Through ReLU

The ReLU activation `ffn1_out = max(0, ffn1_pre)` has derivative:

```
∂ReLU/∂ffn1_pre[i] = 1  if ffn1_pre[i] > 0
                      0  otherwise
```

Since we cached `ffn1_out` **after** ReLU, we use the sign of the cached value as the indicator:

```
∂L/∂ffn1_pre[i] = ∂L/∂ffn1_out[i]  if cached_ffn1_out[i] > 0
                   0                  if cached_ffn1_out[i] ≤ 0
```

**Why cached_ffn1_out works as indicator:** After applying ReLU, values that were negative become exactly 0. So checking `cached_ffn1_out[i] <= 0` is equivalent to checking `ffn1_pre[i] <= 0`.

**Code Reference:** `src/layers/transformer.rs` lines 320-331:
```rust
// Backward through ReLU
for i in 0..grad_ffn1_out.len() {
    if cached_ffn1_out[i] <= 0.0 {
        grad_ffn1_out[i] = 0.0;
    }
}
```

### Step 4: Gradient Through FFN Layer 1

FFN1 is a dense linear layer `ffn1_pre = ln2_out × W_ff1 + b_ff1`.

```
∂L/∂W_ff1 = ln2_out^T × ∂L/∂ffn1_pre           (d_model × d_ff)
∂L/∂b_ff1 = Σ_tokens ∂L/∂ffn1_pre               (d_ff,)
∂L/∂ln2_out = ∂L/∂ffn1_pre × W_ff1^T           (B×S, d_model)
```

**Dimension checks:**
- `ln2_out^T`: (d_model, B×S), `∂L/∂ffn1_pre`: (B×S, d_ff) → `∂L/∂W_ff1`: (d_model, d_ff) ✓
- `∂L/∂ffn1_pre`: (B×S, d_ff), `W_ff1^T`: (d_ff, d_model) → `∂L/∂ln2_out`: (B×S, d_model) ✓

**Code Reference:** `src/layers/transformer.rs` lines 333-339:
```rust
self.ffn1.backward(
    &cached_ln2_out,
    &grad_ffn1_out,
    &mut grad_ln2_out,
    batch_size * seq_len,
);
```

### Step 5: Gradient Through Second Layer Norm

LN2 backward computes `∂L/∂residual1` given `∂L/∂ln2_out` (and the LN2 internal gradients `∂L/∂γ₂`, `∂L/∂β₂` for parameter updates).

**LayerNorm backward formula** (for one token `i`):

```
∂L/∂x̂ᵢⱼ = ∂L/∂yᵢⱼ × γⱼ                            (affine backward)

∂L/∂xᵢⱼ = (1/D) × (1/stdᵢ) × [D × ∂L/∂x̂ᵢⱼ
                                  - Σₖ ∂L/∂x̂ᵢₖ
                                  - x̂ᵢⱼ × Σₖ (∂L/∂x̂ᵢₖ × x̂ᵢₖ)]
```

Where D = d_model. The gradient `∂L/∂residual1` is the gradient flowing back through LN2.

**Code Reference:** `src/layers/transformer.rs` lines 341-348:
```rust
let mut grad_residual1_from_ln2 = vec![0.0f32; total_size];
self.ln2.backward(
    &cached_residual1,
    &grad_ln2_out,
    &mut grad_residual1_from_ln2,
    batch_size * seq_len,
);
```

> **See:** [layernorm_layer.md](./layernorm_layer.md) for the full LayerNorm backward derivation.

### Step 6: Gradient Accumulation at Intermediate State

The gradient at `residual1` receives contributions from **two paths**:
- **Path A** (direct): gradient from second residual connection (Step 1)
- **Path B** (through LN2+FFN): gradient backpropagated through LN2 (Step 5)

```
∂L/∂residual1 (total) = ∂L/∂residual1 (path A) + ∂L/∂residual1 (path B)
                       = ∂L/∂output + LN2_backward(∂L/∂ln2_out)
```

This is a fundamental property of backprop through addition nodes: **all incoming gradient paths are summed**.

**Code Reference:** `src/layers/transformer.rs` lines 350-353:
```rust
// Add gradients from both paths into residual1
for i in 0..total_size {
    grad_residual1[i] += grad_residual1_from_ln2[i];
}
```

### Step 7: Gradient Through First Residual Connection

The first residual `residual1 = x + attn_out` is another element-wise addition. Gradients distribute equally:

```
∂L/∂x (path A from residual1)    = ∂L/∂residual1 (total)
∂L/∂attn_out                     = ∂L/∂residual1 (total)
```

**Code Reference:** `src/layers/transformer.rs` lines 355-357:
```rust
// Backward through first residual connection: residual1 = input + attn_out
grad_attn_out.copy_from_slice(&grad_residual1);
```

### Step 8: Gradient Through Multi-Head Attention

The attention layer backward computes `∂L/∂ln1_out` and updates the Q, K, V projection weights.

The MHA backward propagates through:
1. Value aggregation: `∂L/∂attn_weights` and `∂L/∂V`
2. Softmax: `∂L/∂scores`
3. Scaled dot-product: `∂L/∂Q` and `∂L/∂K`
4. Q, K, V projections: `∂L/∂ln1_out` and parameter gradients

**Code Reference:** `src/layers/transformer.rs` lines 359-365:
```rust
self.attention.backward(
    &cached_ln1_out,
    &grad_attn_out,
    &mut grad_ln1_out,
    batch_size,
);
```

> **See:** [attention_mechanism.md](./attention_mechanism.md) for the full multi-head attention backward derivation.

### Step 9: Gradient Through First Layer Norm

LN1 backward computes `∂L/∂x` (gradient w.r.t. the original block input) given `∂L/∂ln1_out`, and updates LN1's learnable parameters `γ₁`, `β₁`.

```
∂L/∂x (path B, through attention) = LN1_backward(∂L/∂ln1_out)
```

**Code Reference:** `src/layers/transformer.rs` lines 367-374:
```rust
self.ln1.backward(
    &cached_input,
    &grad_ln1_out,
    grad_input,
    batch_size * seq_len,
);
```

### Step 10: Gradient Accumulation at Input

Like `residual1`, the block input `x` also receives gradients from **two paths**:
- **Path A** (direct): gradient from first residual shortcut (Step 7)
- **Path B** (through attention): gradient backpropagated through LN1 → MHA → LN1_backward (Step 9)

```
∂L/∂x (total) = ∂L/∂x (path A from residual1) + ∂L/∂x (path B from attention)
              = ∂L/∂residual1 (total) + LN1_backward(MHA_backward(∂L/∂attn_out))
```

**Code Reference:** `src/layers/transformer.rs` lines 375-378:
```rust
// Add gradient from first residual connection
for i in 0..total_size {
    grad_input[i] += grad_residual1[i];
}
```

### Complete Backward Pass Summary

The full backward pass, in reverse order of the forward pass:

```
Given: ∂L/∂output

Step 1:  ∂L/∂residual1 (A)  = ∂L/∂output          (second residual: direct path)
         ∂L/∂ffn2_out        = ∂L/∂output          (second residual: FFN path)

Step 2:  ∂L/∂W_ff2, ∂L/∂b_ff2, ∂L/∂ffn1_out_relu  ← FFN2.backward(∂L/∂ffn2_out)

Step 3:  ∂L/∂ffn1_out_pre[i] = ∂L/∂ffn1_out_relu[i] if ffn1_out_cached[i] > 0
                                 0                     otherwise

Step 4:  ∂L/∂W_ff1, ∂L/∂b_ff1, ∂L/∂ln2_out  ← FFN1.backward(∂L/∂ffn1_out_pre)

Step 5:  ∂L/∂γ₂, ∂L/∂β₂, ∂L/∂residual1 (B)  ← LN2.backward(∂L/∂ln2_out)

Step 6:  ∂L/∂residual1 (total) = ∂L/∂residual1 (A) + ∂L/∂residual1 (B)

Step 7:  ∂L/∂attn_out  = ∂L/∂residual1 (total)   (first residual: attention path)
         ∂L/∂x (A)     = ∂L/∂residual1 (total)   (first residual: direct path)

Step 8:  ∂L/∂W_Q, ∂L/∂W_K, ∂L/∂W_V, ∂L/∂ln1_out  ← MHA.backward(∂L/∂attn_out)

Step 9:  ∂L/∂γ₁, ∂L/∂β₁, ∂L/∂x (B)  ← LN1.backward(∂L/∂ln1_out)

Step 10: ∂L/∂x (total) = ∂L/∂x (A) + ∂L/∂x (B)
```

**Key insight: Gradient splitting at residual connections**

At each residual `output = x + f(x)`:
- The skip connection creates two gradient paths to `x`
- Both paths contribute to `∂L/∂x` via summation
- This is exactly why residual networks avoid vanishing gradients: `∂L/∂x` always includes the direct term `∂L/∂output` regardless of how small the gradients through `f(x)` become

## Pre-LN vs Post-LN Comparison

### Architectural Differences

**Post-LN (original Vaswani et al., 2017):**

```
x ──────────────────────────────────┐
                                    │
MHA(x) ──────────────────────────► Add ──► LN1 ──► residual1
                                                       │
                               ┌───────────────────────┤
                               │                       │
                            FFN(residual1) ──────────► Add ──► LN2 ──► output
```

```
residual1 = LN1(x + MHA(x))        (norm after residual)
output    = LN2(residual1 + FFN(residual1))
```

**Pre-LN (Xiong et al., 2020) — this implementation:**

```
x ──────────────────────────────────────────┐
                                            │
MHA(LN1(x)) ────────────────────────────► Add ──► residual1
                                                       │
                          ┌────────────────────────────┤
                          │                            │
                       FFN(LN2(residual1)) ──────────► Add ──► output
```

```
residual1 = x + MHA(LN1(x))        (norm before sub-layer)
output    = residual1 + FFN(LN2(residual1))
```

### Mathematical Implications

**Post-LN gradient at the input of block `l`:**

```
∂L/∂x_l = ∂L/∂x_{l+1} × ∂x_{l+1}/∂x_l
```

Because LN is applied **after** the residual in Post-LN, the gradient must pass through the LN Jacobian:

```
∂L/∂x_l = ∂L/∂x_{l+1} × (I + J_MHA × J_LN1) × (I + J_FFN × J_LN2)
```

Where `J_MHA`, `J_FFN`, `J_LN1`, `J_LN2` are Jacobian matrices. The multiplication of many such terms can lead to gradient explosion or vanishing.

**Pre-LN gradient at the input of block `l`:**

In Pre-LN, the residuals carry raw (unnormalized) inputs, so the gradient identity path is cleaner:

```
∂L/∂x_l = ∂L/∂residual1 + ∂L/∂residual1 × J_MHA × J_LN1
         = ∂L/∂x_{l+1} × I + [correction terms]
         = ∂L/∂x_{l+1} + [correction terms]
```

The key term `∂L/∂x_{l+1} × I` is an **identity mapping**: gradient flows directly from later layers to earlier layers without being scaled by the LN Jacobian at the residual merge point. This ensures gradient norms remain stable across depth.

### Training Stability Analysis

| Property | Post-LN | Pre-LN |
|----------|---------|--------|
| Gradient magnitude at initialization | Can explode/vanish | More stable (close to 1) |
| Requires learning rate warmup | Yes (critical) | Often optional |
| Final model performance | Potentially better (if trained correctly) | Slightly lower but more robust |
| Gradient norm across layers | Non-monotonic | More uniform |
| Training with large depth | Difficult without warmup | More tractable |
| Appropriate initialization | Needs special care | Standard Xavier works |

**Practical guidance:**
- Use **Pre-LN** for easier training, especially without warmup schedulers or careful LR tuning
- Use **Post-LN** only if you can afford LR warmup and want potentially higher peak accuracy
- The performance gap between Pre-LN and Post-LN narrows with proper tuning; Pre-LN is generally recommended for production systems

**Code Reference:** `src/layers/transformer.rs` module docstring lines 24-28:
```
This implementation uses Pre-LN where layer normalization is applied *before*
each sub-layer (attention and FFN), which provides better training stability
and gradient flow compared to the original Post-LN architecture.
```

## Parameter Count Analysis

For a TransformerBlock with `d_model`, `num_heads`, and `d_ff`:

| Sub-Layer | Parameters | Formula |
|-----------|-----------|---------|
| LN1 (γ₁, β₁) | 2 × d_model | scale + shift per feature |
| Multi-Head Attention | 4 × (d_model² + d_model) | W_Q, W_K, W_V, W_O each (d_model, d_model) + bias |
| LN2 (γ₂, β₂) | 2 × d_model | scale + shift per feature |
| FFN1 (W_ff1, b_ff1) | d_model × d_ff + d_ff | expansion layer |
| FFN2 (W_ff2, b_ff2) | d_ff × d_model + d_model | projection layer |
| **Total** | **4d_model² + 2d_model×d_ff + 8d_model + d_ff** | |

**Example (d_model=64, num_heads=4, d_ff=256):**

```
LN1:       2 × 64              =     128
Attention: 4 × (64² + 64)     = 4 × 4,160  = 16,640
LN2:       2 × 64              =     128
FFN1:      64 × 256 + 256     = 16,384 + 256 = 16,640
FFN2:      256 × 64 + 64      = 16,384 + 64  = 16,448
──────────────────────────────────────────────────────
Total:                                        = 49,984
```

**Code Reference:** `src/layers/transformer.rs` lines 438-444:
```rust
fn parameter_count(&self) -> usize {
    self.ln1.parameter_count()
        + self.attention.parameter_count()
        + self.ln2.parameter_count()
        + self.ffn1.parameter_count()
        + self.ffn2.parameter_count()
}
```

## Stacked Transformer Encoder

Multiple `TransformerBlock` layers can be stacked in a `TransformerEncoder` where the output of block `l` becomes the input to block `l+1`:

```
x ──► Block_0 ──► Block_1 ──► ... ──► Block_{L-1} ──► output
```

**Forward pass (sequential):**
```
h_0 = x
h_{l+1} = TransformerBlock_l(h_l)    for l = 0, ..., L-1
output = h_L
```

**Backward pass (reverse sequential):**

Gradients flow from `∂L/∂output` back through each block in reverse:

```
∂L/∂h_{L-1} = TransformerBlock_{L-1}.backward(∂L/∂h_L)
∂L/∂h_{l-1} = TransformerBlock_l.backward(∂L/∂h_l)
...
∂L/∂x = TransformerBlock_0.backward(∂L/∂h_1)
```

**Implementation note:** The `TransformerEncoder` backward pass runs a second forward pass to re-cache activations before backpropagating. In a production system, all intermediate activations would be cached during the initial forward pass to avoid redundant computation.

**Code Reference:** `src/layers/transformer.rs` lines 598-629 (TransformerEncoder backward).

**Total parameters for L stacked blocks:**
```
total_params = L × (4d_model² + 2d_model×d_ff + 8d_model + d_ff)
```

For L=6, d_model=512, num_heads=8, d_ff=2048 (standard BERT-base configuration):
```
Per block:  4 × 512² + 2 × 512 × 2048 + 8 × 512 + 2048
          = 1,048,576 + 2,097,152 + 4,096 + 2,048
          = 3,151,872
6 blocks:  6 × 3,151,872 = 18,911,232 ≈ 18.9M parameters
```

## Numerical Considerations

**1. Layer Normalization epsilon:**
- The epsilon (ε = 1e-5) prevents division by zero when variance is near zero
- This occurs most commonly at initialization when all weights produce nearly identical outputs

**2. ReLU dead neuron problem:**
- ReLU units with `ffn1_pre[i] ≤ 0` have zero gradient and never update
- Xavier initialization (`sqrt(2/(fan_in + fan_out))`) reduces the probability of dead neurons at initialization
- Residual connections also help: even if FFN1 neurons die, gradients still flow through the skip connection

**3. Attention softmax stability:**
- The scale factor `1/sqrt(d_head)` prevents extreme attention scores that cause softmax saturation
- Without scaling, large `d_head` causes dot products to grow proportionally, pushing softmax toward one-hot distributions with near-zero gradients

**4. Gradient accumulation at residual nodes:**
- At each residual merge, gradients from two paths are summed
- This ensures gradient magnitude roughly doubles at each residual merge, but the direct path always contributes a term proportional to the upstream gradient — preventing vanishing

**5. RefCell interior mutability:**
- The Rust implementation uses `RefCell<Vec<f32>>` for cached activations to allow mutation inside the `&self` forward method (required by the `Layer` trait)
- This is safe in single-threaded contexts; multi-threaded use would require `Arc<Mutex<Vec<f32>>>` or `Arc<RwLock<Vec<f32>>>`

## Implementation References

| File | Lines | Description |
|------|-------|-------------|
| `src/layers/transformer.rs` | 1-33 | Module documentation, architecture overview |
| `src/layers/transformer.rs` | 69-92 | `TransformerBlock` struct with cached activations |
| `src/layers/transformer.rs` | 117-176 | `TransformerBlock::new()` constructor |
| `src/layers/transformer.rs` | 198-265 | `forward()` — 6-step forward pass |
| `src/layers/transformer.rs` | 267-379 | `backward()` — 10-step backward pass |
| `src/layers/transformer.rs` | 381-412 | `update_parameters()` and `update_with_optimizer()` |
| `src/layers/transformer.rs` | 459-537 | `TransformerEncoder` struct and constructor |
| `src/layers/transformer.rs` | 539-629 | `TransformerEncoder` forward and backward |
| `src/layers/layernorm.rs` | — | `LayerNormLayer` implementation |
| `src/layers/attention.rs` | — | `MultiHeadAttentionLayer` implementation |
| `src/layers/dense.rs` | — | `DenseLayer` implementation |

**Related documentation:**
- [attention_mechanism.md](./attention_mechanism.md) — Multi-head self-attention forward and backward derivations
- [layernorm_layer.md](./layernorm_layer.md) — Layer normalization forward and backward derivations
- [dense_layer.md](./dense_layer.md) — Dense layer (FFN sub-component) mathematics
- [README.md](./README.md) — Overview of all backpropagation documentation

**References:**
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
