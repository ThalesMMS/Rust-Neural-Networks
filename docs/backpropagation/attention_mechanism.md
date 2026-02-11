# Attention Mechanism Mathematics

This document provides a comprehensive explanation of the mathematics behind the Transformer-style self-attention mechanism, covering both forward and backward propagation with detailed derivations. This implementation follows the scaled dot-product attention from "Attention Is All You Need" (Vaswani et al., 2017).

## Table of Contents

- [Overview](#overview)
- [Architecture Components](#architecture-components)
- [Forward Pass](#forward-pass)
  - [1. Patch Extraction and Tokenization](#1-patch-extraction-and-tokenization)
  - [2. Token Embedding Projection](#2-token-embedding-projection)
  - [3. Positional Encoding](#3-positional-encoding)
  - [4. Query, Key, Value Projections](#4-query-key-value-projections)
  - [5. Scaled Dot-Product Attention](#5-scaled-dot-product-attention)
  - [6. Attention Score Computation](#6-attention-score-computation)
  - [7. Softmax Normalization](#7-softmax-normalization)
  - [8. Weighted Value Aggregation](#8-weighted-value-aggregation)
  - [9. Feed-Forward Network](#9-feed-forward-network)
  - [10. Mean Pooling](#10-mean-pooling)
  - [11. Classification Head](#11-classification-head)
- [Backward Pass](#backward-pass)
- [Computational Complexity](#computational-complexity)
- [Implementation References](#implementation-references)
- [Numerical Considerations](#numerical-considerations)

## Overview

The attention mechanism enables the model to dynamically focus on different parts of the input sequence when processing each element. For vision tasks like MNIST, the input image is divided into patches (tokens), and attention allows the model to learn which patches are most relevant for classification.

**Key characteristics:**
- Self-attention: each token attends to all other tokens in the sequence
- Scaled dot-product: prevents softmax saturation with large embedding dimensions
- Position-aware: sinusoidal positional encodings provide spatial information
- Permutation-equivariant: attention weights depend on content, not position alone

**MNIST Attention Model Configuration:**
- **Image size:** 28×28 pixels
- **Patch size:** 4×4 pixels
- **Grid:** 7×7 patches
- **Sequence length:** SEQ_LEN = 49 tokens
- **Patch dimension:** PATCH_DIM = 16 features per patch
- **Embedding dimension:** D_MODEL = 64
- **Feed-forward dimension:** FF_DIM = 128
- **Output classes:** NUM_CLASSES = 10

## Architecture Components

The attention model consists of the following learnable parameters:

**1. Patch Projection:**
- `w_patch`: Weight matrix (16, 64) - projects 16-dim patches to 64-dim embeddings
- `b_patch`: Bias vector (64,)

**2. Positional Embeddings:**
- `pos`: Positional encoding matrix (49, 64) - one embedding per token position

**3. Attention Projections:**
- `w_q, b_q`: Query projection (64, 64) and bias (64,)
- `w_k, b_k`: Key projection (64, 64) and bias (64,)
- `w_v, b_v`: Value projection (64, 64) and bias (64,)

**4. Feed-Forward Network:**
- `w_ff1, b_ff1`: First layer (64, 128) and bias (128,)
- `w_ff2, b_ff2`: Second layer (128, 64) and bias (64,)

**5. Classification Head:**
- `w_cls, b_cls`: Classifier weights (64, 10) and bias (10,)

## Forward Pass

The forward pass transforms a batch of images through patch extraction, attention, and classification.

### 1. Patch Extraction and Tokenization

**Purpose:** Convert each 28×28 image into a sequence of 49 tokens (patches).

**Mathematical Operation:**

For each image in the batch:
```
For py = 0 to 6:                    (patch row)
  For px = 0 to 6:                  (patch column)
    t = py × 7 + px                 (token index: 0 to 48)

    For dy = 0 to 3:                (pixel row within patch)
      For dx = 0 to 3:              (pixel column within patch)
        iy = py × 4 + dy            (image row: 0 to 27)
        ix = px × 4 + dx            (image column: 0 to 27)

        patch[t][dy × 4 + dx] = image[iy][ix]
```

**Dimensions:**
- Input: `batch_inputs` shape (B, 784) - flattened images
- Output: `patches` shape (B, 49, 16) - 49 patches, each 16 pixels

**Visual Example:**

```
28×28 Image Grid:
┌─────────────────────────────┐
│ 0  1  2  3 │ 4  5  6  7 │..│  Each cell = 1 pixel
│ 28 29 30 31│ 32 33 34 35│..│
│ 56 57 58 59│ 60 61 62 63│..│
│112113114115│116117118119│..│
├────────────┼────────────┤
│784......   │            │  │
└─────────────────────────────┘

Patch Grid (7×7 = 49 tokens):
Token  0: pixels [0,1,2,3, 28,29,30,31, 56,57,58,59, 84,85,86,87]
Token  1: pixels [4,5,6,7, 32,33,34,35, 60,61,62,63, 88,89,90,91]
...
Token 48: bottom-right 4×4 patch
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 728-748: `extract_patches()` function

### 2. Token Embedding Projection

**Purpose:** Project each 16-dimensional patch to a 64-dimensional token embedding.

**Mathematical Formula:**

```
tok_pre[b][t] = patch[b][t] · W_patch + b_patch
tok[b][t] = ReLU(tok_pre[b][t] + pos[t])
```

**Expanded form:**

For batch index `b`, token index `t`, and embedding dimension `d`:

```
tok_pre[b][t][d] = b_patch[d] + Σⱼ (patch[b][t][j] · W_patch[j][d])
                                  j=0 to 15

tok[b][t][d] = max(0, tok_pre[b][t][d] + pos[t][d])
```

**Dimensions:**
- `patch[b][t]`: (16,) - single patch vector
- `W_patch`: (16, 64) - projection weights
- `b_patch`: (64,) - projection bias
- `pos[t]`: (64,) - positional encoding for token t
- `tok[b][t]`: (64,) - output token embedding

**Step-by-step computation:**

```
1. Linear projection:    patch (16) @ W_patch (16, 64) → (64)
2. Add bias:             (64) + b_patch (64) → (64)
3. Add position:         (64) + pos[t] (64) → (64)
4. ReLU activation:      max(0, ·) → (64)
```

**Why ReLU after position?**
- Introduces non-linearity early in the network
- Allows learning of sparse, selective features
- Empirically improves gradient flow in early layers

**Code Reference:**
- `mnist_attention_pool.rs` lines 1195-1213: Token projection loop

### 3. Positional Encoding

**Purpose:** Provide spatial information about where each patch is located in the 7×7 grid.

**CRITICAL IMPORTANCE:** Positional encoding was identified as the PRIMARY factor affecting model accuracy (+38.56 percentage points improvement from random to sinusoidal encoding).

**Sinusoidal Positional Encoding Formula:**

For position `pos` (0 to 48) and dimension `i` (0 to 63):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))     (even dimensions)
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))     (odd dimensions)
```

Where `d_model = 64`.

**Expanded notation:**

```
angle[pos][i] = pos / 10000^(2·⌊i/2⌋ / 64)

If i is even (0, 2, 4, ...):
  PE[pos][i] = sin(angle[pos][i])

If i is odd (1, 3, 5, ...):
  PE[pos][i] = cos(angle[pos][i])
```

**Properties:**

1. **Unique encodings:** Each position gets a distinct embedding
2. **Smooth transitions:** Adjacent positions have similar encodings
3. **Periodic patterns:** Different frequencies across dimensions
   - Low frequencies (early dimensions): coarse position information
   - High frequencies (late dimensions): fine-grained position information
4. **Deterministic:** Same position always gets same encoding (not learned from data initially, but gradients can adjust)

**Example values for D_MODEL=64:**

```
Position 0 (top-left patch):
  PE[0][0] = sin(0 / 1) = 0.0
  PE[0][1] = cos(0 / 1) = 1.0
  PE[0][2] = sin(0 / 10000^(2/64)) = 0.0
  ...

Position 24 (center patch):
  PE[24][0] = sin(24 / 1) = 0.9055...
  PE[24][1] = cos(24 / 1) = -0.4242...
  ...

Position 48 (bottom-right patch):
  PE[48][0] = sin(48 / 1) = -0.7682...
  PE[48][1] = cos(48 / 1) = 0.6402...
  ...
```

**Why sinusoidal encoding works:**

- **Spatial structure:** Encodes the 7×7 grid layout naturally
- **Relative position learning:** Attention can learn "nearby patches" patterns
- **Strong prior:** Provides structured information from epoch 1
- **Smooth gradients:** Continuous functions enable better backpropagation

**Alternative strategies tested (performance comparison):**

| Strategy | Accuracy | Notes |
|----------|----------|-------|
| Sinusoidal | 83.45% | ✓ BEST - Transformer-style encoding |
| Larger Random [-0.5, 0.5] | 71.86% | Better than small random, but lacks structure |
| Xavier Random | 45.63% | Similar to small random |
| Small Random [-0.1, 0.1] | 44.89% | Original baseline - insufficient signal |
| Zero (learn from scratch) | 35.65% | Worst - no positional prior |

**Code Reference:**
- `mnist_attention_pool.rs` lines 559-593: Sinusoidal encoding implementation
- `mnist_attention_pool.rs` lines 476-508: Positional encoding strategy enum

### 4. Query, Key, Value Projections

**Purpose:** Transform token embeddings into query, key, and value representations for attention computation.

**Mathematical Formulas:**

```
Q[b][t] = tok[b][t] · W_q + b_q    (Query)
K[b][t] = tok[b][t] · W_k + b_k    (Key)
V[b][t] = tok[b][t] · W_v + b_v    (Value)
```

**Expanded form for dimension d_out:**

```
Q[b][t][d_out] = b_q[d_out] + Σ (tok[b][t][d_in] · W_q[d_in][d_out])
                               d_in=0 to 63

K[b][t][d_out] = b_k[d_out] + Σ (tok[b][t][d_in] · W_k[d_in][d_out])
                               d_in=0 to 63

V[b][t][d_out] = b_v[d_out] + Σ (tok[b][t][d_in] · W_v[d_in][d_out])
                               d_in=0 to 63
```

**Dimensions:**
- `tok[b][t]`: (64,) - input token embedding
- `W_q, W_k, W_v`: (64, 64) - projection matrices
- `b_q, b_k, b_v`: (64,) - bias vectors
- `Q[b][t], K[b][t], V[b][t]`: (64,) - output projections

**Batch dimensions:**
- Input: `tok` shape (B, 49, 64)
- Output: `Q`, `K`, `V` each shape (B, 49, 64)

**Intuition:**

- **Query (Q):** "What am I looking for?" - represents the information needs of token i
- **Key (K):** "What do I contain?" - represents the content available in token j
- **Value (V):** "What information do I provide?" - the actual features to aggregate

**Why separate projections?**

1. **Different semantic spaces:** Q and K live in a "similarity space", V in "feature space"
2. **Expressiveness:** Learned projections allow model to extract relevant aspects
3. **Flexibility:** Q·K measures relevance; V provides content (decoupled operations)

**Code Reference:**
- `mnist_attention_pool.rs` lines 1221-1239: Q/K/V projection loops

### 5. Scaled Dot-Product Attention

**Purpose:** Compute attention scores that determine how much each token should attend to every other token.

**Complete Attention Formula:**

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- `Q·K^T`: Attention scores (similarity between queries and keys)
- `√d_k`: Scaling factor (d_k = D_MODEL = 64)
- `softmax`: Normalization to get attention weights (probabilities)
- Result · V: Weighted sum of values

This operation is broken down into steps 6-8 below.

### 6. Attention Score Computation

**Purpose:** Calculate raw similarity scores between each query and all keys.

**Mathematical Formula:**

```
scores[b][i][j] = (Q[b][i] · K[b][j]) / √d_k

Where:
  · denotes dot product
  d_k = D_MODEL = 64
  √d_k ≈ 8.0
```

**Expanded form:**

```
score_raw = Σ (Q[b][i][d] · K[b][j][d])
            d=0 to 63

scores[b][i][j] = score_raw / √64 = score_raw / 8.0
```

**Dimensions:**
- For each batch b:
  - `Q[b]`: (49, 64) - queries for all tokens
  - `K[b]`: (49, 64) - keys for all tokens
  - `scores[b]`: (49, 49) - attention score matrix

**Score matrix interpretation:**

```
scores[b] =
       ↓ Key tokens (j)
       0     1     2  ...  48
Q   0 [s00   s01   s02 ... s0,48]  ← How much token 0 attends to each token
u   1 [s10   s11   s12 ... s1,48]
e   2 [s20   s21   s22 ... s2,48]
r   .  ...   ...   ...     ...
y   48[s48,0 s48,1 ...    s48,48]

Each row i: token i's attention scores to all tokens
Each column j: how much all tokens attend to token j
Diagonal: self-attention (token attending to itself)
```

**Why scale by √d_k?**

Without scaling, dot products grow with dimension:
- For d_k = 64 and normalized vectors, dot products can be ~8× larger
- Large magnitudes → extreme softmax values → vanishing gradients
- Scaling maintains variance and prevents saturation

**Derivation of scaling factor:**

If Q and K have zero mean and unit variance:
```
Var(Q·K) = Σ Var(Qᵢ·Kᵢ) = d_k · Var(Qᵢ) · Var(Kᵢ) = d_k · 1 · 1 = d_k

To normalize: divide by √d_k
Var((Q·K) / √d_k) = d_k / d_k = 1  ✓
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 1241-1262: Score computation with scaling
- Line 1242: `inv_sqrt_d = 1.0 / √64`

### 7. Softmax Normalization

**Purpose:** Convert raw attention scores into probability distribution (attention weights) for each query.

**Mathematical Formula:**

For each query token i in batch b:

```
α[b][i][j] = exp(scores[b][i][j]) / Σₖ exp(scores[b][i][k])
                                      k=0 to 48
```

**Properties of attention weights α:**

```
1. Non-negative: α[b][i][j] ≥ 0  for all i, j
2. Normalized: Σⱼ α[b][i][j] = 1  for each query i
3. Differentiable: smooth gradient flow through softmax
```

**Numerically stable softmax implementation:**

```
For row i:
  max_val = max(scores[b][i][j]) for all j

  exp_scores[j] = exp(scores[b][i][j] - max_val)  (subtract max for stability)

  sum = Σⱼ exp_scores[j]

  α[b][i][j] = exp_scores[j] / sum
```

**Interpretation:**

- **High α[i][j]:** Query token i strongly attends to key token j (relevant)
- **Low α[i][j]:** Query token i weakly attends to key token j (less relevant)
- **Uniform α[i][·]:** No clear attention pattern (all tokens equally weighted)

**Example attention pattern:**

```
For digit "7", token at top-right might attend strongly to:
  - Itself (top-right patch)
  - Vertical stroke tokens (middle-right patches)
  - Horizontal stroke tokens (top patches)

Attention weights might look like:
α[24][·] = [0.02, 0.05, 0.08, 0.15, 0.18, 0.12, 0.05, 0.03, ...]
            ↑                     ↑     ↑     ↑
         low weight          high weights for relevant patches
```

**Why softmax?**

1. **Probabilistic interpretation:** Weights sum to 1 (weighted average)
2. **Competition:** Amplifies differences between high/low scores
3. **Differentiable:** Enables gradient-based learning
4. **Sparse attention:** exp() creates exponential gaps (high scores dominate)

**Code Reference:**
- `mnist_attention_pool.rs` line 1264: `softmax_inplace()` applied per query row
- `mnist_attention_pool.rs` lines 290-308: Softmax implementation

### 8. Weighted Value Aggregation

**Purpose:** Compute the attention output by taking a weighted average of value vectors.

**Mathematical Formula:**

```
attn_out[b][i] = Σ (α[b][i][j] · V[b][j])
                 j=0 to 48
```

**Expanded form for dimension d:**

```
attn_out[b][i][d] = Σ (α[b][i][j] · V[b][j][d])
                    j=0 to 48
```

**Dimensions:**
- `α[b][i]`: (49,) - attention weights for query i
- `V[b][j]`: (64,) - value vectors for all keys j
- `attn_out[b][i]`: (64,) - weighted aggregation of values

**Full batch dimensions:**
- Input: `α` shape (B, 49, 49), `V` shape (B, 49, 64)
- Output: `attn_out` shape (B, 49, 64)

**Intuition:**

Each output token is a **context-aware representation** combining information from all input tokens, weighted by relevance (attention scores).

**Concrete example:**

```
For query token i=24 (center patch):

α[b][24][·] = [0.01, 0.02, 0.05, ..., 0.08, ..., 0.03]
               ↑                        ↑
            weak attention         strong attention to token 30

attn_out[b][24] = 0.01 · V[b][0] + 0.02 · V[b][1] + ... + 0.08 · V[b][30] + ...

Result: Output embedding emphasizes features from highly-attended tokens (e.g., token 30),
        while incorporating context from all tokens in proportion to attention weights.
```

**Matrix interpretation:**

```
attn_out[b] = α[b] @ V[b]

Where:
  α[b]: (49, 49) attention weight matrix
  V[b]: (49, 64) value matrix
  Result: (49, 64) attended representations

Each row of attn_out[b] is a weighted sum of all value vectors.
```

**Why this works:**

- **Information routing:** High attention weights → more influence on output
- **Context integration:** Each token receives information from relevant tokens
- **Learned relevance:** Model learns *what* to attend to via Q/K projections
- **Permutation equivariance:** Reordering tokens changes attention, but consistently

**Code Reference:**
- `mnist_attention_pool.rs` lines 1266-1274: Weighted value aggregation loop

### 9. Feed-Forward Network

**Purpose:** Apply position-wise (per-token) non-linear transformation to attended representations.

**Mathematical Formulas:**

```
ffn1[b][t] = ReLU(attn_out[b][t] · W_ff1 + b_ff1)
ffn2[b][t] = ffn1[b][t] · W_ff2 + b_ff2
```

**Expanded form:**

```
Layer 1 (expansion):
  ffn1[b][t][h] = max(0, b_ff1[h] + Σ (attn_out[b][t][d] · W_ff1[d][h]))
                                     d=0 to 63
  where h = 0 to 127

Layer 2 (projection):
  ffn2[b][t][d] = b_ff2[d] + Σ (ffn1[b][t][h] · W_ff2[h][d])
                              h=0 to 127
  where d = 0 to 63
```

**Dimensions:**
- Input: `attn_out[b][t]` shape (64,)
- Layer 1: `W_ff1` (64, 128), `b_ff1` (128,) → `ffn1[b][t]` (128,)
- Layer 2: `W_ff2` (128, 64), `b_ff2` (64,) → `ffn2[b][t]` (64,)
- Output: `ffn2[b][t]` shape (64,) - same as input

**Architecture:**

```
Input (64) → Linear (64 → 128) → ReLU → Linear (128 → 64) → Output (64)
             ↑ Expansion               ↓ Projection
```

**Why feed-forward network?**

1. **Non-linearity:** ReLU enables learning complex transformations
2. **Capacity:** Expansion to 128 dims provides computational "room"
3. **Position-wise:** Each token processed independently (no cross-token interaction)
4. **Feature refinement:** Transforms attended representations into task-specific features

**Standard Transformer practice:**
- FF dimension typically 4× embedding dimension (here: 128 = 2× 64)
- Two-layer MLP with activation in between
- Same network applied to every token (parameter sharing)

**Code Reference:**
- `mnist_attention_pool.rs` lines 1285-1306: Feed-forward network loop
- Lines 1291-1296: Layer 1 with ReLU
- Lines 1299-1304: Layer 2 (linear projection)

### 10. Mean Pooling

**Purpose:** Aggregate sequence of token representations into a single image-level embedding.

**Mathematical Formula:**

```
pooled[b][d] = (1 / SEQ_LEN) · Σ (ffn2[b][t][d])
                                t=0 to 48

             = (1 / 49) · Σ (ffn2[b][t][d])
                          t=0 to 48
```

**Dimensions:**
- Input: `ffn2` shape (B, 49, 64) - per-token representations
- Output: `pooled` shape (B, 64) - per-image representations

**Alternative pooling strategies:**

| Strategy | Formula | Characteristics |
|----------|---------|-----------------|
| **Mean pooling** (used) | `mean(tokens)` | ✓ Treats all tokens equally; smooth gradients |
| Max pooling | `max(tokens)` | Selects strongest signal; sparse gradients |
| Weighted pooling | `Σ wᵢ·tokenᵢ` | Learned weights; adds parameters |
| CLS token | Use `token[0]` | Requires special [CLS] token; Transformer default |

**Why mean pooling?**

- **Simplicity:** No additional parameters
- **Stability:** All tokens contribute (robust to outliers)
- **Even gradient flow:** All tokens receive gradients equally during backprop
- **Effective for vision:** Spatial information already captured by attention

**Code Reference:**
- `mnist_attention_pool.rs` lines 1313-1322: Mean pooling loop
- Line 1313: `inv_seq = 1.0 / 49.0` (averaging factor)

### 11. Classification Head

**Purpose:** Map the pooled image embedding to class logits and probabilities.

**Mathematical Formulas:**

```
logits[b][c] = pooled[b] · W_cls + b_cls
             = Σ (pooled[b][d] · W_cls[d][c]) + b_cls[c]
               d=0 to 63

probs[b][c] = exp(logits[b][c]) / Σₖ exp(logits[b][k])
                                    k=0 to 9
```

**Dimensions:**
- `pooled[b]`: (64,) - image embedding
- `W_cls`: (64, 10) - classifier weights
- `b_cls`: (10,) - classifier bias
- `logits[b]`: (10,) - raw class scores
- `probs[b]`: (10,) - predicted probabilities (sums to 1.0)

**Interpretation:**

- `logits[b][c]`: Raw score for class c (higher = more confident)
- `probs[b][c]`: P(class = c | image[b]) - probability prediction
- `argmax(probs[b])`: Predicted class label

**Loss computation (cross-entropy):**

```
For true label y:
  loss = -log(probs[b][y])
```

**Gradient w.r.t. logits (for backprop):**

```
dlogits[b][c] = (probs[b][c] - 1.0) / B   if c == y (true class)
              = probs[b][c] / B            otherwise

Where B = batch_size (average over batch)
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 1330-1344: Classification logits and softmax
- `mnist_attention_pool.rs` lines 792-812: Loss and gradient computation

## Backward Pass

The backward pass computes gradients for all parameters using the chain rule, flowing from the loss back through each component. This enables gradient descent optimization.

### Complete Gradient Flow Visualization

The following diagram shows the complete forward and backward pass through the attention model:

```
                        FORWARD PASS                    |                    BACKWARD PASS
                                                        |
    Input Image (28×28)                                 |
         (B, 784)                                       |
            │                                           |
            ▼                                           |
    ┌──────────────────┐                               |
    │ Patch Extraction │                               |                    (not backpropped)
    │  7×7 grid of     │                               |
    │  4×4 patches     │                               |
    └────────┬─────────┘                               |
             │                                          |
             ▼                                          |
        Patches                                         |
      (B, 49, 16)                                       |                           ∂L/∂tok
            │                                           |                          (B, 49, 64)
            ▼                                           |                               ▲
    ┌──────────────────┐                               |                               │
    │ Token Projection │◄──── W_patch, b_patch ────────┼──────────────────────┬────────┤
    │ Linear + ReLU    │      (16, 64), (64)           |           ∂L/∂W_patch│∂L/∂b   │
    │      + pos       │◄──── pos (49, 64) ────────────┼────┐      (16, 64)   │ (64)   │
    └────────┬─────────┘                               |    │ ∂L/∂pos (49, 64)│        │
             │                                          |    │            ┌────┴────────┤
             ▼                                          |    │            │ReLU gradient│
          Tokens                                        |    │            │   masking   │
        (B, 49, 64)                                     |    │            └─────────────┘
             │                                          |    │
             ├────────────────────┬───────────────┐     |    │
             ▼                    ▼               ▼     |    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Query Projection│  │  Key Projection │  │ Value Projection│
    │  Q = tok·W_q+b_q│  │  K = tok·W_k+b_k│  │  V = tok·W_v+b_v│
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                     │
             ▼                    ▼                     ▼
          Q (B,49,64)          K (B,49,64)          V (B,49,64)
             │                    │                     │
             │                    │                     │            ∂L/∂Q      ∂L/∂K      ∂L/∂V
             └────────┬───────────┘                     │           (B,49,64) (B,49,64) (B,49,64)
                      │                                 │                ▲         ▲         ▲
                      ▼                                 │                │         │         │
              ┌──────────────────┐                      │                │         │         │
              │ Attention Scores │                      │         ┌──────┴─────────┴───┐     │
              │  Q·K^T / √d_k    │                      │         │  ∂scores / √d_k    │     │
              │   (B, 49, 49)    │                      │         │  (scaled dot-prod  │     │
              └────────┬─────────┘                      │         │    backward)       │     │
                       │                                │         └────────────────────┘     │
                       ▼                                │                  ▲                  │
              ┌──────────────────┐                      │                  │                  │
              │     Softmax      │                      │         ┌────────┴────────┐         │
              │   per row (i)    │                      │         │Softmax Jacobian:│         │
              │   α (B,49,49)    │                      │         │ α·(dα - α^T·dα) │         │
              └────────┬─────────┘                      │         └─────────────────┘         │
                       │                                │                  ▲                  │
                       └────────┬───────────────────────┤                  │                  │
                                │                       │            ∂L/∂α (B,49,49)          │
                                ▼                       │                  ▲                  │
                    ┌───────────────────────┐           |                  │                  │
                    │ Weighted Aggregation  │           |         ┌────────┴─────────┐        │
                    │   attn_out = α @ V    │           |         │ ∂L/∂α from       │        │
                    │     (B, 49, 64)       │           |         │ dattn @ V^T      │◄───────┤
                    └───────────┬───────────┘           |         └──────────────────┘        │
                                │                       |                  ▲                  │
                                ▼                       |                  │           ┌──────┴──────┐
                      Attention Output                  |            ∂L/∂attn_out     │  ∂L/∂V from │
                        (B, 49, 64)                     |             (B, 49, 64)     │  α^T @ dattn│
                                │                       |                  ▲           └─────────────┘
                                ▼                       |                  │
                    ┌─────────────────────┐             |                  │
                    │  Feed-Forward Net   │◄─ W_ff1, b_ff1, W_ff2, b_ff2  │
                    │                     │  (64,128), (128), (128,64),(64)|
                    │  FFN1: Linear+ReLU  │             |       ∂L/∂W_ff1, ∂L/∂b_ff1
                    │  FFN2: Linear       │             |       ∂L/∂W_ff2, ∂L/∂b_ff2
                    └──────────┬──────────┘             |                  │
                               │                        |         ┌────────┴────────┐
                               ▼                        |         │ Linear backward │
                          FFN Output                    |         │ ReLU gradient   │
                          (B, 49, 64)                   |         │    masking      │
                               │                        |         └─────────────────┘
                               ▼                        |                  ▲
                    ┌─────────────────────┐             |                  │
                    │    Mean Pooling     │             |            ∂L/∂ffn2
                    │ pool = mean(ffn, 1) │             |            (B, 49, 64)
                    │      over 49 tokens │             |                  ▲
                    └──────────┬──────────┘             |                  │
                               │                        |         ┌────────┴────────┐
                               ▼                        |         │  Distribute     │
                          Pooled                        |         │  dpooled / 49   │
                          (B, 64)                       |         │  to all tokens  │
                               │                        |         └─────────────────┘
                               ▼                        |                  ▲
                    ┌─────────────────────┐             |                  │
                    │ Classification Head │◄── W_cls, b_cls                │
                    │   logits = pool·W   │    (64, 10), (10)              │
                    │   + cls + b_cls     │             |       ∂L/∂W_cls, ∂L/∂b_cls
                    └──────────┬──────────┘             |                  │
                               │                        |         ┌────────┴────────┐
                               ▼                        |         │   dpooled =     │
                            Logits                      |         │ dlogits @ W_cls^T│
                            (B, 10)                     |         └─────────────────┘
                               │                        |                  ▲
                               ▼                        |                  │
                    ┌─────────────────────┐             |              ∂L/∂pooled
                    │      Softmax        │             |               (B, 64)
                    │  probs = exp / sum  │             |                  ▲
                    └──────────┬──────────┘             |                  │
                               │                        |            ∂L/∂logits
                               ▼                        |              (B, 10)
                         Probabilities                  |                  ▲
                            (B, 10)                     |                  │
                               │                        |         ┌────────┴────────┐
                               ▼                        |         │ Softmax +       │
                    ┌─────────────────────┐             |         │ Cross-Entropy:  │
                    │   Cross-Entropy     │             |         │ probs - onehot  │
                    │   Loss = -log(p_y)  │             |         └─────────────────┘
                    └──────────┬──────────┘             |
                               │                        |
                               ▼                        |
                             Loss ─────────────────────►└─── Backpropagation starts here


Parameter Gradient Summary:
═══════════════════════════

1. ∂L/∂W_cls, ∂L/∂b_cls  ← From classifier backward (pooled^T @ dlogits)
2. ∂L/∂W_ff2, ∂L/∂b_ff2  ← From FFN layer 2 backward (ffn1^T @ dffn2)
3. ∂L/∂W_ff1, ∂L/∂b_ff1  ← From FFN layer 1 backward (attn_out^T @ dffn1, with ReLU mask)
4. ∂L/∂W_v, ∂L/∂b_v      ← From Value projection backward (tok^T @ dV)
5. ∂L/∂W_k, ∂L/∂b_k      ← From Key projection backward (tok^T @ dK)
6. ∂L/∂W_q, ∂L/∂b_q      ← From Query projection backward (tok^T @ dQ)
7. ∂L/∂W_patch, ∂L/∂b    ← From token projection backward (patch^T @ dtok, with ReLU mask)
8. ∂L/∂pos               ← From positional encoding (sum dtok over batch)


Critical Gradient Computations:
════════════════════════════════

Attention Mechanism Backward (most complex):
  1. ∂L/∂V: Computed from α^T @ dattn_out
  2. ∂L/∂α: Computed from dattn_out @ V^T
  3. ∂L/∂scores: Apply softmax Jacobian to ∂L/∂α
  4. ∂L/∂Q, ∂L/∂K: Compute from ∂L/∂scores using scaled dot-product backward

Legend:
  ──►  Forward data flow
  ◄──  Parameter (weights/biases)
  ──▲  Backward gradient flow
  @    Matrix multiplication
  ^T   Transpose
```

**Gradient flow sequence:**

```
Loss → dlogits → dpooled → dffn2 → dffn1 → dattn → dalpha → dscores → dQ/dK/dV → dtok → dpatches
  ↓                ↓          ↓        ↓        ↓       ↓        ↓          ↓          ↓
dW_cls         dW_ff2    dW_ff1   dW_v/k/q  softmax  attention  Q·K^T   projections  W_patch
db_cls         db_ff2    db_ff1   db_v/k/q  Jacobian  scaling   scaling   ReLU      b_patch
                                                                                      pos
```

**Key gradient computations:**

### Backward Through Classifier

**Inputs:**
- `dlogits[b][c]`: Gradient of loss w.r.t. logits (from softmax + cross-entropy)

**Gradient formulas:**

```
∂L/∂b_cls[c] = Σ dlogits[b][c]
               b=0 to B-1

∂L/∂W_cls[d][c] = Σ (pooled[b][d] · dlogits[b][c])
                  b=0 to B-1

∂L/∂pooled[b][d] = Σ (dlogits[b][c] · W_cls[d][c])
                   c=0 to 9
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 851-870: Classifier backward pass

### Backward Through Mean Pooling

**Purpose:** Distribute pooled gradients evenly to all tokens.

**Gradient formula:**

```
∂L/∂ffn2[b][t][d] = (1 / SEQ_LEN) · ∂L/∂pooled[b][d]
                  = (1 / 49) · dpooled[b][d]
```

Each token receives an equal share of the pooled gradient.

**Code Reference:**
- `mnist_attention_pool.rs` lines 873-882: Mean pooling gradient distribution

### Backward Through Feed-Forward Network

**Layer 2 gradients (linear):**

```
∂L/∂b_ff2[d] = Σ Σ dffn2[b][t][d]
               b t

∂L/∂W_ff2[h][d] = Σ Σ (ffn1[b][t][h] · dffn2[b][t][d])
                  b t

∂L/∂ffn1[b][t][h] = Σ (dffn2[b][t][d] · W_ff2[h][d])
                    d=0 to 63
```

**Layer 1 gradients (ReLU + linear):**

ReLU backward (zero out gradients where activation was negative):
```
If ffn1[b][t][h] <= 0:
  dffn1[b][t][h] = 0
```

Linear layer:
```
∂L/∂b_ff1[h] = Σ Σ dffn1[b][t][h]
               b t

∂L/∂W_ff1[d][h] = Σ Σ (attn_out[b][t][d] · dffn1[b][t][h])
                  b t

∂L/∂attn_out[b][t][d] = Σ (dffn1[b][t][h] · W_ff1[d][h])
                        h=0 to 127
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 884-941: Feed-forward backward pass
- Lines 913-918: ReLU gradient masking

### Backward Through Attention Mechanism

**This is the most complex part of the backward pass.**

**Step 1: Gradient w.r.t. Values (dV)**

From `attn_out[b][i] = Σⱼ α[b][i][j] · V[b][j]`:

```
∂L/∂V[b][j][d] = Σ (α[b][i][j] · dattn[b][i][d])
                 i=0 to 48
```

**Step 2: Gradient w.r.t. Attention Weights (dalpha)**

```
∂L/∂α[b][i][j] = Σ (dattn[b][i][d] · V[b][j][d])
                 d=0 to 63
```

**Step 3: Gradient Through Softmax (dscores)**

Softmax Jacobian for row i:

```
For α = softmax(scores):
  ∂α[i][j] / ∂scores[i][k] = α[i][j] · (δ[j,k] - α[i][k])

Where δ[j,k] = 1 if j==k, else 0

Applying chain rule:
  dscores[b][i][j] = α[b][i][j] · (dalpha[b][i][j] - Σₖ (dalpha[b][i][k] · α[b][i][k]))
```

**Step 4: Gradient Through Scaling (back to Q and K)**

From `scores[b][i][j] = (Q[b][i] · K[b][j]) / √d_k`:

```
∂L/∂Q[b][i][d] = Σ (dscores[b][i][j] · K[b][j][d] / √d_k)
                 j=0 to 48

∂L/∂K[b][j][d] = Σ (dscores[b][i][j] · Q[b][i][d] / √d_k)
                 i=0 to 48
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 943-993: Attention mechanism backward pass
- Lines 944-976: dV, dalpha, and softmax gradient
- Lines 978-993: dQ and dK through scaled dot-product

### Backward Through Q/K/V Projections

For each projection (Q, K, V), apply linear layer backward:

```
∂L/∂b_q[d] = Σ Σ dQ[b][t][d]
             b t

∂L/∂W_q[d_in][d_out] = Σ Σ (tok[b][t][d_in] · dQ[b][t][d_out])
                       b t

∂L/∂tok[b][t][d_in] = Σ (dQ[b][t][d_out] · W_q[d_in][d_out])
                      d_out
                    + Σ (dK[b][t][d_out] · W_k[d_in][d_out])
                      d_out
                    + Σ (dV[b][t][d_out] · W_v[d_in][d_out])
                      d_out
```

(Similar for K and V projections)

**Code Reference:**
- `mnist_attention_pool.rs` lines 995-1024: Q/K/V projection backward pass

### Backward Through Token Embeddings

**ReLU gradient:**

```
If tok[b][t][d] <= 0:
  dtok[b][t][d] = 0
```

**Patch projection gradients:**

```
∂L/∂pos[t][d] = Σ dtok[b][t][d]
                b

∂L/∂b_patch[d] = Σ Σ dtok[b][t][d]
                 b t

∂L/∂W_patch[j][d] = Σ Σ (patch[b][t][j] · dtok[b][t][d])
                    b t
```

**Code Reference:**
- `mnist_attention_pool.rs` lines 1026-1054: Token embedding backward pass
- Lines 1026-1031: ReLU gradient masking
- Lines 1033-1054: Patch projection gradients

### Parameter Updates (SGD)

After computing all gradients, update parameters using stochastic gradient descent:

```
W ← W - lr · ∂L/∂W
b ← b - lr · ∂L/∂b
```

Where `lr = 0.01` (learning rate).

**Code Reference:**
- `mnist_attention_pool.rs` lines 1059-1106: `apply_sgd()` function

## Computational Complexity

**Forward Pass:**

| Operation | Complexity | Note |
|-----------|------------|------|
| Patch extraction | O(B · 784) | Copy pixels to patches |
| Token projection | O(B · 49 · 16 · 64) = O(B · 50K) | Patch → embedding |
| Q/K/V projections | 3 × O(B · 49 · 64²) = O(B · 600K) | Three linear layers |
| Attention scores | O(B · 49² · 64) = O(B · 153K) | Q @ K^T |
| Softmax | O(B · 49²) = O(B · 2.4K) | Per-row normalization |
| Value aggregation | O(B · 49² · 64) = O(B · 153K) | α @ V |
| Feed-forward | O(B · 49 · (64·128 + 128·64)) = O(B · 800K) | Two linear layers |
| Pooling | O(B · 49 · 64) = O(B · 3K) | Mean over sequence |
| Classifier | O(B · 64 · 10) = O(B · 640) | Final linear layer |

**Total forward:** O(B · 1.76M) operations per batch

**Backward Pass:**

Approximately 2-3× forward pass complexity (gradients for weights, activations, and chain rule).

**Total backward:** O(B · 4-5M) operations per batch

**Memory Complexity:**

| Buffer | Size | Note |
|--------|------|------|
| Parameters | ~230K floats | All weights and biases |
| Forward activations | ~140K floats per batch | Intermediate results |
| Backward gradients | ~370K floats | Gradients + parameter grads |

**Total memory:** ~740K floats ≈ 3 MB per batch (negligible for modern systems)

## Implementation References

**Key code locations in `mnist_attention_pool.rs`:**

| Component | Lines | Function |
|-----------|-------|----------|
| Model structure | 318-339 | `AttnModel` struct |
| Batch buffers | 400-474 | `BatchBuffers` struct |
| Positional encoding | 559-593 | Sinusoidal initialization |
| Patch extraction | 728-748 | `extract_patches()` |
| Forward pass | 1175-1345 | `forward_inference()` |
| Token projection | 1195-1213 | Patch → embedding + ReLU |
| Q/K/V projections | 1221-1239 | Linear projections |
| Attention scores | 1250-1262 | Scaled dot-product |
| Softmax | 1264 | Normalization per query |
| Value aggregation | 1266-1274 | Weighted sum |
| Feed-forward | 1285-1306 | Two-layer MLP |
| Mean pooling | 1313-1322 | Sequence → image embedding |
| Classifier | 1330-1344 | Logits + softmax |
| Loss computation | 792-812 | Cross-entropy + gradients |
| Backward pass | 816-1057 | `backward_batch()` |
| Classifier backward | 851-870 | ∂L/∂W_cls, ∂L/∂pooled |
| Pooling backward | 873-882 | Gradient distribution |
| FFN backward | 884-941 | Feed-forward gradients |
| Attention backward | 943-993 | ∂V, ∂α, softmax Jacobian, ∂Q/∂K |
| Projection backward | 995-1024 | Q/K/V projection gradients |
| Token backward | 1026-1054 | ReLU, patch projection, positional |
| SGD update | 1059-1106 | `apply_sgd()` |

## Numerical Considerations

**1. Attention Score Scaling**

- **Problem:** Without scaling, dot products grow with dimension → softmax saturation
- **Solution:** Divide by √d_k = 8.0 to maintain unit variance
- **Impact:** Prevents vanishing gradients in softmax

**2. Softmax Numerical Stability**

- **Problem:** `exp(large_number)` overflows; `exp(small_number)` underflows
- **Solution:** Subtract max before exp: `exp(x - max(x))`
- **Impact:** Mathematically equivalent, numerically stable

**3. Positional Encoding Initialization**

- **Critical finding:** Sinusoidal > Random by 38.56 percentage points
- **Reason:** Provides structured spatial prior from epoch 1
- **Impact:** Enables attention to learn relative position relationships

**4. Gradient Clipping (not implemented, but recommended for production)**

- **Potential issue:** Attention gradients can occasionally spike
- **Solution:** Clip gradients by norm or value
- **Impact:** More stable training, especially with larger learning rates

**5. Learning Rate**

- **Optimal:** 0.01 (validated through experiments)
- **Impact of LR=0.001:** Only 18.98% accuracy (too slow)
- **Impact of LR=0.01:** 91.08% accuracy (fast convergence, stable)

**6. Weight Initialization**

- **Strategy:** Xavier uniform initialization
- **Formula:** `U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))`
- **Impact:** Proper variance scaling prevents gradient vanishing/explosion

**7. Epsilon for Numerical Stability**

- **Loss computation:** `loss = -log(p + 1e-9)` prevents log(0)
- **Impact:** Avoids NaN/Inf in loss computation

---

This completes the documentation of the attention mechanism forward pass. The backward pass section provides the mathematical framework for gradient computation, enabling the model to learn through backpropagation.

## Related Documentation

**Activation Functions:**
- [Activation Functions](../activation_functions.md) - Detailed mathematical documentation for ReLU, softmax, and other activations used throughout the attention mechanism

**Component Layers:**
- [Dense Layer](dense_layer.md) - Fully connected layers used in Q/K/V projections, feed-forward network, and classification head
- [Convolutional Layer](conv2d_layer.md) - Alternative architecture for image processing using local connectivity

**Core Architecture:**
- [Backpropagation Overview](README.md) - General backpropagation concepts and notation
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface implementation
