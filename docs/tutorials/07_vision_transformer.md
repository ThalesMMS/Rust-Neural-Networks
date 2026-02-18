# Tutorial 07: Vision Transformer (ViT) for CIFAR-10

**Level:** Expert
**Time:** 120-150 minutes
**Prerequisites:** Tutorial 03 (MNIST CNN), understanding of self-attention (Tutorial on Transformers or NLP background), familiarity with CIFAR-10 RGB format
**Implementation:** See `src/bin/cifar10_vit.rs` and `src/layers/patch_embedding.rs`

**Navigation:**
[Tutorial 06: Autoencoders](06_autoencoder.md) | [Tutorial Index](README.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [From NLP Transformers to Vision](#from-nlp-transformers-to-vision)
3. [Architecture Overview](#architecture-overview)
4. [Patch Tokenization](#patch-tokenization)
5. [Patch Embedding Layer](#patch-embedding-layer)
6. [Positional Encoding for 2D Images](#positional-encoding-for-2d-images)
7. [Self-Attention on Image Patches](#self-attention-on-image-patches)
8. [Mean Pooling vs CLS Token](#mean-pooling-vs-cls-token)
9. [Full Forward Pass Walkthrough](#full-forward-pass-walkthrough)
10. [Backward Pass and Gradient Flow](#backward-pass-and-gradient-flow)
11. [Training the Vision Transformer](#training-the-vision-transformer)
12. [Running the Model](#running-the-model)
13. [Attention Visualization](#attention-visualization)
14. [Verification Checkpoints](#verification-checkpoints)
15. [Exercises](#exercises)
16. [Next Steps](#next-steps)

---

## Introduction

Every previous tutorial in this series processed images with architectures designed specifically for spatial data: CNNs exploit local connectivity with sliding filters, and MLPs flatten pixels into one long vector. Both approaches encode strong assumptions about how image features should be extracted.

The **Vision Transformer (ViT)** takes a radically different approach. It borrows the Transformer architecture -- originally designed for natural language processing -- and applies it directly to images. Instead of convolving filters across pixels, ViT slices the image into small patches, treats each patch as a "token" (analogous to a word in a sentence), and lets self-attention learn which patches relate to which.

This tutorial builds a complete ViT for CIFAR-10 classification. By the end, you will understand:

- How images are converted into sequences of patch tokens
- Why positional encoding is essential when spatial structure is discarded
- How self-attention discovers relationships between image regions
- Why mean pooling provides a robust image-level representation
- How to train and evaluate a ViT on 32x32 RGB images
- How to visualize attention maps to interpret what the model has learned

**Implementation reference:**
- `src/layers/patch_embedding.rs` -- `PatchEmbeddingLayer`
- `src/bin/cifar10_vit.rs` -- Training binary
- `config/training/cifar10_vit_default.json` -- Default training config
- `config/cifar10_vit_small.json` -- Quick experiment config
- `visualize_vit_attention.py` -- Attention map visualization

---

## From NLP Transformers to Vision

The original Transformer (Vaswani et al., 2017) was designed for machine translation. It processes a sequence of word tokens, and self-attention lets every token attend to every other token -- capturing long-range dependencies that RNNs struggle with.

The key insight of ViT (Dosovitskiy et al., 2020) is that images can be reframed as sequences too:

| NLP Transformer | Vision Transformer |
|-----------------|-------------------|
| Sentence of words | Grid of image patches |
| Word embedding (lookup table) | Patch embedding (linear projection) |
| Positional encoding for word order | Positional encoding for spatial position |
| Self-attention between words | Self-attention between patches |
| [CLS] token or pooling for classification | Mean pooling or [CLS] token for classification |

**Why does this work?** Self-attention is permutation-invariant: it does not care about the order of its inputs. By adding positional encodings, we inject spatial structure back in. The model then learns, through training, which spatial relationships matter for classification -- without the rigid locality assumption of convolution filters.

**Trade-offs compared to CNNs:**
- ViTs need more data or stronger regularization (no built-in translation equivariance)
- ViTs can capture long-range dependencies from the first layer (CNNs need deep stacks)
- ViTs are more computationally expensive for small images (attention is O(n^2) in sequence length)
- ViTs scale better to very large datasets and model sizes

For our CIFAR-10 task (32x32 images, 50K training samples), the ViT is somewhat data-starved compared to industrial-scale settings. But it still learns meaningful attention patterns and achieves reasonable accuracy, making it an excellent educational example.

---

## Architecture Overview

Here is the complete ViT pipeline for CIFAR-10:

```
CIFAR-10 Image (32x32x3 RGB, pixel-interleaved)
  |
  v
[Patch Extraction] -- Split into 4x4 patches
  |                    8x8 grid = 64 patches
  |                    Each patch: 4x4x3 = 48 values
  v
[64 patches, each 48-dim]
  |
  v
[PatchEmbeddingLayer] -- Linear projection: 48 -> 128
  |                       Wraps DenseLayer (BLAS-accelerated)
  v
[64 tokens, each 128-dim]
  |
  v
[ReLU Activation] -- Element-wise non-linearity
  |
  v
[+ Sinusoidal Positional Encoding] -- 64 positions, 128 dims
  |
  v
[TransformerEncoder] -- 4 stacked TransformerBlocks
  |                      Each block:
  |                        LayerNorm -> Multi-Head Attention (4 heads, d_k=32)
  |                        + Residual connection
  |                        LayerNorm -> FFN (128 -> 256 -> 128)
  |                        + Residual connection
  v
[64 tokens, each 128-dim]
  |
  v
[Mean Pooling] -- Average over 64 tokens -> single 128-dim vector
  |
  v
[DenseLayer Classifier] -- Linear: 128 -> 10
  |
  v
[Softmax] -- 10-class probability distribution
  |
  v
Predicted class (airplane, automobile, bird, cat, deer,
                  dog, frog, horse, ship, truck)
```

**Architecture in numbers:**

| Component | Dimensions | Parameters |
|-----------|-----------|------------|
| Patch Embedding: 48 -> 128 | 48 x 128 weights + 128 biases | 6,272 |
| Transformer Encoder (4 blocks) | See below | ~400K |
| Classifier: 128 -> 10 | 128 x 10 weights + 10 biases | 1,290 |

Each TransformerBlock contains:
- LayerNorm (128 dims): 256 params (scale + shift)
- Multi-Head Attention (4 heads): Q, K, V projections (128->128 each) + output (128->128) = 65,536 + 512 params
- LayerNorm (128 dims): 256 params
- FFN: Dense(128->256) + Dense(256->128) = 65,920 params

---

## Patch Tokenization

### The Core Idea: Images as Sequences

A CNN sees an image as a 2D grid of pixels and slides filters across it. A ViT sees an image as a flat sequence of patch tokens. The conversion is straightforward: divide the image into a regular grid of non-overlapping patches, and flatten each patch into a 1D vector.

For CIFAR-10:
- Image size: 32 x 32 pixels x 3 channels (RGB)
- Patch size: 4 x 4 pixels
- Grid: 32/4 = 8 patches per row, 8 per column
- Number of patches: 8 x 8 = **64 tokens**
- Patch dimension: 4 x 4 x 3 = **48 values per token**

```
32x32 RGB image                  64 patch tokens
 ___________________________________
|    |    |    |    |    |    |    |    |     Token 0:  patch(0,0) -> [48 floats]
|  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |     Token 1:  patch(0,1) -> [48 floats]
|____|____|____|____|____|____|____|____|     Token 2:  patch(0,2) -> [48 floats]
|    |    |    |    |    |    |    |    |         ...
|  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |     Token 63: patch(7,7) -> [48 floats]
|____|____|____|____|____|____|____|____|
|    |    |    |    |    |    |    |    |
| 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |     Each 4x4 patch contains:
|____|____|____|____|____|____|____|____|       R R R R
|    |    |    |    |    |    |    |    |       R R R R
| 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |       R R R R   (x3 channels)
|____|____|____|____|____|____|____|____|       R R R R
|    |    |    |    |    |    |    |    |
| 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |     Flattened in row-major order:
|____|____|____|____|____|____|____|____|       [R,G,B, R,G,B, R,G,B, ... ]
|    |    |    |    |    |    |    |    |          pixel(0,0)  pixel(0,1) ...
| 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |       16 pixels x 3 channels = 48
|____|____|____|____|____|____|____|____|
|    |    |    |    |    |    |    |    |
| 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |
|____|____|____|____|____|____|____|____|
|    |    |    |    |    |    |    |    |
| 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
|____|____|____|____|____|____|____|____|
```

### Analogy to Word Tokenization

In NLP, a sentence like "The cat sat on the mat" is split into tokens: ["The", "cat", "sat", "on", "the", "mat"]. Each token is a discrete symbol that gets mapped to a continuous embedding vector via a lookup table.

In ViT, the image is split into patches. Each patch is already a continuous vector (48 floats of pixel values). Instead of a lookup table, we use a **linear projection** (the patch embedding layer) to map each 48-dim patch to a higher-dimensional embedding space (128 dims).

The analogy:

| NLP | ViT |
|-----|-----|
| Word | Image patch |
| Vocabulary size | Infinite (continuous pixel values) |
| Embedding lookup | Linear projection (DenseLayer) |
| Token embedding dim | d_model = 128 |
| Sequence length | 64 (8x8 grid of patches) |

### RGB Pixel-Interleaved Format

CIFAR-10 images in this codebase use pixel-interleaved format: for each pixel, the R, G, B values are stored consecutively. This means a 4x4 patch is flattened as:

```
pixel(0,0).R, pixel(0,0).G, pixel(0,0).B,
pixel(0,1).R, pixel(0,1).G, pixel(0,1).B,
...
pixel(3,3).R, pixel(3,3).G, pixel(3,3).B
```

The `extract_patches_rgb` function in `cifar10_vit.rs` handles this extraction:

```rust
fn extract_patches_rgb(images: &[f32], batch_size: usize, patches: &mut [f32]) {
    for b in 0..batch_size {
        for py in 0..GRID {         // patch row (0..8)
            for px in 0..GRID {     // patch col (0..8)
                let token_idx = py * GRID + px;
                for dy in 0..PATCH_SIZE {       // pixel row within patch (0..4)
                    for dx in 0..PATCH_SIZE {   // pixel col within patch (0..4)
                        let img_y = py * PATCH_SIZE + dy;
                        let img_x = px * PATCH_SIZE + dx;
                        for c in 0..IMG_CHANNELS {  // R, G, B
                            patches[...] = images[...];
                        }
                    }
                }
            }
        }
    }
}
```

The six nested loops traverse: batch -> patch grid row -> patch grid column -> pixel row -> pixel column -> channel. This produces a flat buffer of shape `[batch_size * 64 * 48]`.

---

## Patch Embedding Layer

### Purpose

Raw pixel patches (48 floats) are too low-dimensional and too "raw" for self-attention to work effectively. The patch embedding layer projects each 48-dim patch into a richer 128-dim representation space where the Transformer can operate.

This is conceptually identical to word embeddings in NLP: converting a raw input (word index or pixel patch) into a learned, continuous representation.

### Implementation

The `PatchEmbeddingLayer` in `src/layers/patch_embedding.rs` is a thin wrapper around `DenseLayer`:

```rust
pub struct PatchEmbeddingLayer {
    patch_dim: usize,   // 48
    d_model: usize,     // 128
    dense: DenseLayer,  // Linear projection: 48 -> 128
}

impl PatchEmbeddingLayer {
    pub fn new(patch_dim: usize, d_model: usize, rng: &mut SimpleRng) -> Self {
        Self {
            patch_dim,
            d_model,
            dense: DenseLayer::new(patch_dim, d_model, rng),
        }
    }
}
```

The `DenseLayer` inside performs the linear transformation:

```
embedding = patch * W + b
```

where W is a [48 x 128] weight matrix and b is a [128] bias vector, both initialized with Xavier initialization. The BLAS-accelerated `sgemm` handles the batch matrix multiplication efficiently.

### Why a Wrapper?

The `PatchEmbeddingLayer` implements the `Layer` trait by delegating to its inner `DenseLayer`. This gives it a semantically meaningful name and type within the architecture, even though the computation is identical to a standard dense layer. It makes the code self-documenting: when you see `PatchEmbeddingLayer::new(48, 128, &mut rng)`, you immediately know this is the patch-to-token projection.

### Activation After Embedding

After the linear projection, the implementation applies **ReLU activation** element-wise:

```rust
// Forward pass: Patch embedding
patch_embedding.forward(&patches, &mut patch_embeds, current_batch_size * NUM_PATCHES);

// Apply ReLU activation
relu_inplace(&mut patch_embeds);
```

This introduces a non-linearity between the raw projection and the positional encoding addition. Without it, the projection would be purely linear, and the positional encoding would be added to a linear function of the input -- limiting the model's representational capacity at the very first stage.

### Parameter Count

```
Weights: 48 x 128 = 6,144
Biases:  128
Total:   6,272 parameters
```

This is a small fraction of the total model, but it is the critical first transformation that converts raw pixels into the token space where attention operates.

---

## Positional Encoding for 2D Images

### Why Position Matters

Self-attention is **permutation-invariant**: if you shuffle the order of the 64 patch tokens, the attention output (before pooling) would be identical, just in a different order. This means, without positional encoding, the model would have no way to know that token 0 is the top-left patch and token 63 is the bottom-right patch.

For image classification, spatial position is critical. A red patch in the top-left might be part of a car's taillight, while the same red patch in the center might be a fire truck. The model needs positional information to disambiguate these cases.

### Sinusoidal Positional Encoding

This implementation uses the sinusoidal positional encoding from "Attention is All You Need":

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

where:
- `pos` = patch index (0 to 63)
- `i` = dimension index (0 to 127)
- Even dimensions use sine, odd dimensions use cosine

From `src/utils/positional_encoding.rs`:

```rust
pub fn sinusoidal_positional_encoding(seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut pos_encoding = vec![0.0f32; seq_len * d_model];

    for t in 0..seq_len {
        for d in 0..d_model {
            let angle = (t as f32) / 10000.0f32.powf((2 * (d / 2)) as f32 / d_model as f32);
            if d % 2 == 0 {
                pos_encoding[t * d_model + d] = angle.sin();
            } else {
                pos_encoding[t * d_model + d] = angle.cos();
            }
        }
    }
    pos_encoding
}
```

**Key properties:**
- Each position gets a unique encoding vector (64 distinct 128-dim vectors)
- Low-frequency dimensions capture coarse position, high-frequency dimensions capture fine position
- The encoding is deterministic and computed once before training
- All values lie in [-1, 1]

### 1D Encoding for 2D Patches

Our 8x8 patch grid is linearized into a 1D sequence: patch(0,0) is token 0, patch(0,1) is token 1, ..., patch(7,7) is token 63. The positional encoding is applied to this 1D index.

This means the model must learn from the 1D positional signal that token 8 is directly below token 0 (not next to it). In practice, self-attention learns these 2D spatial relationships during training -- nearby patches in the image tend to have correlated features, and the positional encoding provides enough signal for the attention mechanism to discover the 2D layout.

An alternative would be to use 2D positional encodings (separate encodings for row and column), but the 1D sinusoidal approach works well for small grids like 8x8.

### How Positional Encoding Is Added

The encoding is precomputed once and added to the patch embeddings after ReLU:

```rust
// Generate positional encoding (once, before training)
let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

// During forward pass (per batch):
for b in 0..current_batch_size {
    for t in 0..NUM_PATCHES {
        let offset = (b * NUM_PATCHES + t) * D_MODEL;
        let pos_offset = t * D_MODEL;
        for d in 0..D_MODEL {
            patch_embeds[offset + d] += pos_encoding[pos_offset + d];
        }
    }
}
```

The same positional encoding is added to every image in the batch. This is a simple element-wise addition -- no learnable parameters are involved.

---

## Self-Attention on Image Patches

### What Self-Attention Does

In a CNN, each filter sees a fixed-size local neighborhood (e.g., 3x3 pixels). To connect distant regions, you need many stacked layers. In a Transformer, self-attention lets every patch attend to every other patch in a single operation. This means even in the first layer, the top-left patch can directly interact with the bottom-right patch.

For image classification, this is powerful: the model can learn to relate a wheel patch with a windshield patch (both part of a car) regardless of their distance in the image.

### Multi-Head Attention

Our ViT uses 4 attention heads, each with dimension d_k = d_model / num_heads = 128 / 4 = 32:

```
For each head h (0..3):
    Q_h = X * W_Q_h    [64 x 32]    (Queries)
    K_h = X * W_K_h    [64 x 32]    (Keys)
    V_h = X * W_V_h    [64 x 32]    (Values)

    Attention_h = softmax(Q_h * K_h^T / sqrt(32)) * V_h    [64 x 32]

Concatenate heads: [64 x 128]
Output projection: concat * W_O    [64 x 128]
```

Each head learns to attend to different types of relationships:
- One head might focus on spatially adjacent patches (local texture)
- Another might focus on patches with similar color statistics (global color)
- A third might attend to edges and contours across the image
- A fourth might learn semantic groupings (all patches belonging to the same object)

### Attention Within the TransformerBlock

Each of the 4 TransformerBlocks follows the Pre-LN architecture:

```
input
  |
  +---> LayerNorm --> Multi-Head Attention --> Add(residual) --> intermediate
                                                 |
                                                 +---> LayerNorm --> FFN --> Add(residual) --> output
```

Where the FFN (Feed-Forward Network) is:
```
Dense(128 -> 256) --> ReLU --> Dense(256 -> 128)
```

The residual connections ensure that information can flow directly through the network, and LayerNorm stabilizes training by normalizing activations before each sub-layer.

### Stacking 4 Blocks

With 4 stacked TransformerBlocks, each block refines the token representations:
- **Block 1:** Learns low-level patch interactions (color similarity, edge continuity)
- **Block 2:** Combines low-level features into mid-level patterns
- **Block 3:** Builds higher-level semantic groupings
- **Block 4:** Produces final representations tuned for classification

Each block reads the output of the previous block, progressively building richer representations. The self-attention mechanism at each level can access all previous computations through the residual stream.

---

## Mean Pooling vs CLS Token

After the Transformer encoder processes the 64 tokens, we need a single vector to feed into the classifier. There are two standard approaches:

### CLS Token (Original ViT)

The original ViT paper prepends a special learnable [CLS] token to the sequence, making it 65 tokens. After the Transformer, only the [CLS] token's output is used for classification. The idea is that self-attention allows information from all patches to flow into the [CLS] token.

**Advantages:**
- Clean separation: one dedicated token aggregates classification-relevant information
- Standard in NLP (BERT, GPT)

**Disadvantages:**
- Adds a learnable parameter (the [CLS] embedding)
- The [CLS] token must "compete" with real patches for attention
- Can be harder to train on small datasets

### Mean Pooling (Our Choice)

Our implementation averages all 64 token outputs:

```rust
// Mean pooling over sequence dimension
for b in 0..current_batch_size {
    for d in 0..D_MODEL {
        let mut sum = 0.0;
        for t in 0..NUM_PATCHES {
            sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
        }
        pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
    }
}
```

Mathematically:

```
pooled[d] = (1/64) * SUM(t=0..63) transformer_out[t][d]
```

**Why mean pooling for CIFAR-10:**
1. **Simpler:** No extra learnable token, fewer hyperparameters
2. **More robust on small data:** Every token contributes equally, so the model does not depend on one special token learning the right aggregation
3. **Better gradient flow:** Gradients from the classifier propagate to all 64 tokens equally (each gets grad/64), rather than concentrating in one token
4. **Empirically effective:** For small images and datasets, mean pooling often matches or outperforms [CLS] token

The pooling reduces the shape from [batch_size x 64 x 128] to [batch_size x 128], giving one feature vector per image for the final classification layer.

---

## Full Forward Pass Walkthrough

Let us trace a single CIFAR-10 image through the entire pipeline, tracking shapes at each step.

### Step 1: Input Image

```
Input: [1 x 3072]   (32 x 32 x 3 = 3072 pixel values in [0, 1])
```

### Step 2: Patch Extraction

```
extract_patches_rgb:
  Input:  [1 x 3072]
  Output: [64 x 48]    (64 patches, each 4x4x3 = 48 values)
```

### Step 3: Patch Embedding (Linear Projection)

```
PatchEmbeddingLayer.forward:
  Input:  [64 x 48]    (treated as batch_size=64 for the DenseLayer)
  Weight: [48 x 128]
  Bias:   [128]
  Output: [64 x 128]   (each patch projected to d_model dimensions)
```

### Step 4: ReLU Activation

```
relu_inplace:
  Input/Output: [64 x 128]   (negative values zeroed out)
```

### Step 5: Add Positional Encoding

```
pos_encoding: [64 x 128]   (precomputed, constant)
patch_embeds += pos_encoding   (element-wise addition)
Output: [64 x 128]
```

### Step 6: Transformer Encoder (4 blocks)

Each block preserves the shape [64 x 128]. Inside each block:

```
Block input: [64 x 128]

  LayerNorm:   [64 x 128] -> [64 x 128]
  Attention:
    Q, K, V:   [64 x 128] -> [4 heads x 64 x 32]  (split across heads)
    Scores:    [4 x 64 x 64]  (attention weights per head)
    Softmax:   [4 x 64 x 64]
    Weighted:  [4 x 64 x 32]  (weighted sum of values)
    Concat:    [64 x 128]     (concatenate heads)
    Project:   [64 x 128]
  Residual:    [64 x 128] + attention_output

  LayerNorm:   [64 x 128] -> [64 x 128]
  FFN:
    Dense1:    [64 x 128] -> [64 x 256]  (expand)
    ReLU:      [64 x 256]
    Dense2:    [64 x 256] -> [64 x 128]  (contract)
  Residual:    [64 x 128] + ffn_output

Block output: [64 x 128]
```

After 4 blocks:
```
Transformer output: [64 x 128]
```

### Step 7: Mean Pooling

```
Mean over 64 tokens:
  Input:  [64 x 128]
  Output: [1 x 128]    (average of all token representations)
```

### Step 8: Classifier

```
DenseLayer.forward:
  Input:  [1 x 128]
  Weight: [128 x 10]
  Bias:   [10]
  Output: [1 x 10]     (raw logits for 10 classes)
```

### Step 9: Softmax

```
softmax_rows:
  Input:  [1 x 10]     (logits)
  Output: [1 x 10]     (probabilities summing to 1.0)

  Example: [0.02, 0.01, 0.03, 0.01, 0.72, 0.05, 0.01, 0.10, 0.03, 0.02]
                                          ^^^^
                                          deer (class 4) = highest probability
```

### Complete Shape Summary

```
[batch x 3072] -- extract_patches --> [batch x 64 x 48]
                -- patch_embed ------> [batch x 64 x 128]
                -- ReLU -------------> [batch x 64 x 128]
                -- + pos_encoding ---> [batch x 64 x 128]
                -- transformer ------> [batch x 64 x 128]
                -- mean_pool --------> [batch x 128]
                -- classifier -------> [batch x 10]
                -- softmax ----------> [batch x 10]
```

---

## Backward Pass and Gradient Flow

The backward pass reverses the forward pass, propagating gradients from the cross-entropy loss back through every layer.

### Loss Gradient

Cross-entropy loss with softmax has the clean combined gradient:

```
grad_logits[i][c] = probs[i][c] - one_hot(label[i])[c]
```

For a batch of size B:
```
grad_logits /= B
```

### Classifier Backward

Standard DenseLayer backward through the 128->10 classifier:
```
grad_pooled = grad_logits * W_classifier^T    [batch x 128]
```

### Mean Pooling Backward

The gradient distributes equally to all 64 tokens:
```
grad_transformer[b][t][d] = grad_pooled[b][d] / 64    for all t in 0..64
```

This is the derivative of the mean operation: each element's contribution to the mean has derivative 1/N.

### Transformer Encoder Backward

The gradient propagates backward through all 4 blocks (last to first). Each block's backward pass involves:

1. **FFN residual:** Split gradient between skip connection and FFN path
2. **FFN backward:** Dense(256->128) backward, ReLU gradient, Dense(128->256) backward
3. **LayerNorm backward:** Normalize gradient computation
4. **Attention residual:** Split gradient between skip connection and attention path
5. **Attention backward:** Output projection backward, per-head backward through softmax and Q/K/V projections
6. **LayerNorm backward:** Normalize gradient computation

The residual connections ensure gradient can flow directly through the block without attenuation.

### Patch Embedding Backward

The gradient from the Transformer flows back through the patch embedding's DenseLayer:
```
grad_patches = grad_patch_embeds * W_embedding^T    [batch x 64 x 48]
```

In practice, `grad_patches` is computed but not used further (there are no learnable parameters before the patch extraction). However, the weight and bias gradients of the embedding layer are accumulated for the optimizer update.

### Parameter Updates

All three components are updated with separate Adam optimizers:

```rust
patch_embedding.update_with_optimizer(&mut patch_emb_optimizer);
transformer_encoder.update_with_optimizer(&mut transformer_optimizer);
classifier.update_with_optimizer(&mut classifier_optimizer);
```

Each Adam optimizer maintains its own first-moment (m) and second-moment (v) estimates for the parameters it manages, providing adaptive per-parameter learning rates.

---

## Training the Vision Transformer

### Configuration System

All hyperparameters are externalized to JSON config files, consistent with the rest of the project. The binary loads config at startup:

```rust
use rust_neural_networks::config::load_config;

let config_path = args().nth(1)
    .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_string());
let config = load_config(&config_path)?;
```

The default config (`config/training/cifar10_vit_default.json`):

```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.00001,
  "T_max": 20,
  "activation_function": "relu",
  "optimizer_type": "adam",
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "learning_rate": 0.001,
  "epochs": 20,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.001
}
```

A quick-experiment config (`config/cifar10_vit_small.json`) trains for only 5 epochs:

```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001,
  "T_max": 5,
  "activation_function": "relu",
  "optimizer_type": "adam",
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8,
  "learning_rate": 0.001,
  "epochs": 5,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

### Optimizer Choice: Why Adam

Adam is the default for ViT training because:

1. **Adaptive learning rates** handle the diverse parameter scales across patch embeddings, attention weights, and the classifier
2. **Momentum** helps navigate the complex loss landscape of self-attention
3. **lr=0.001** is the standard starting point for Adam on classification tasks

AdamW (Adam with decoupled weight decay) would be even better for larger models and longer training, as it provides implicit regularization. You can switch by changing `"optimizer_type": "adamw"` and adding `"weight_decay": 0.01` in the config.

### Learning Rate Schedule: Cosine Annealing

The default config uses cosine annealing, which smoothly reduces the learning rate from 0.001 to 0.00001 over 20 epochs:

```
lr(epoch) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / T_max))
```

This is particularly well-suited for Transformers: the initial high learning rate helps escape poor initializations, and the gradual decay allows fine-grained convergence.

### Data Splitting and Validation

Following the project standard:
- **Training:** 45,000 images (50K minus 10% validation)
- **Validation:** 5,000 images (10% of training set)
- **Test:** 10,000 images (held-out, only used for final evaluation)

### Early Stopping

Patience of 5 epochs with minimum delta of 0.001. If validation loss does not improve by at least 0.001 for 5 consecutive epochs, training stops early. This prevents overfitting, which ViTs are particularly susceptible to on small datasets like CIFAR-10.

---

## Running the Model

### Prerequisites

Ensure the CIFAR-10 data is in `data/cifar-10-batches-bin/`. See `docs/cifar10_dataset.md` for download instructions. You need:
- `data_batch_1.bin` through `data_batch_5.bin` (training data)
- `test_batch.bin` (test data)

### Build and Run

```bash
# Build release binary (BLAS acceleration required for speed)
cargo build --release --bin cifar10_vit

# Run with default config (20 epochs, Adam lr=0.001, cosine annealing)
cargo run --release --bin cifar10_vit

# Run with explicit default config
cargo run --release --bin cifar10_vit -- config/training/cifar10_vit_default.json

# Quick experiment (5 epochs, for testing the pipeline)
cargo run --release --bin cifar10_vit -- config/cifar10_vit_small.json

# Optimized execution (macOS)
RUSTFLAGS="-C target-cpu=native" VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin cifar10_vit
```

### Expected Training Output

```
=== Vision Transformer (ViT) -- CIFAR-10 Classifier ===

Configuration:
  Learning rate: 0.001
  Epochs: 20
  Batch size: 64
  Validation split: 10.0%
  Model: d_model=128, heads=4, d_ff=256, blocks=4, patch=4x4
  Patches: 8x8 = 64 tokens, patch_dim=48

Loading CIFAR-10 data...
Training samples: 45000
Validation samples: 5000
Test samples: 10000

Initializing model layers...
Model architecture:
  Patch embedding: 48 -> 128 (6272 params)
  Transformer encoder: 4 blocks (... params)
  Classifier: 128 -> 10 (1290 params)
  Total parameters: ...

Starting training...

Epoch 1/20, Loss: 1.8432, Val Loss: 1.6218, Val Acc: 40.12%, Time: 45.2s
Epoch 2/20, Loss: 1.5634, Val Loss: 1.4871, Val Acc: 45.80%, Time: 44.8s
Epoch 3/20, Loss: 1.4210, Val Loss: 1.3842, Val Acc: 49.56%, Time: 44.9s
...
Epoch 10/20, Loss: 1.1023, Val Loss: 1.2134, Val Acc: 56.20%, Time: 45.1s
...

=== Final Test Evaluation ===
Test Loss: ...
Test Accuracy: ...%

Saving attention maps for 16 test images...
Attention maps saved to logs/vit_attention_maps.csv
Training logs saved to logs/cifar10_vit_log.csv
```

**Note:** Exact numbers depend on hardware and random initialization. CIFAR-10 is a harder task than MNIST, and a small ViT without data augmentation or pre-training will not reach the 95%+ accuracy of large-scale ViTs. The educational value is in understanding the architecture, not in achieving state-of-the-art results.

### Output Files

| File | Contents |
|------|----------|
| `logs/cifar10_vit_log.csv` | Per-epoch training log (epoch, train_loss, train_time, val_loss, val_accuracy) |
| `logs/vit_attention_maps.csv` | Attention weights from the first Transformer block for 16 test images |

### Training Performance Tips

- **Release mode is essential:** Debug mode will be 10-50x slower due to BLAS and iterator optimizations
- **Batch size 64** balances memory and gradient quality for this model size
- **macOS Accelerate:** Set `VECLIB_MAXIMUM_THREADS=8` to parallelize BLAS operations
- **Training time:** Expect 30-60 seconds per epoch depending on hardware (the O(n^2) attention over 64 tokens is more expensive than CNN convolutions)

---

## Attention Visualization

One of the most compelling aspects of the Vision Transformer is the interpretability of its attention maps. After training, we can inspect which patches attend to which, revealing what spatial relationships the model has learned.

### Saving Attention Maps

The training binary automatically saves attention weights from the first Transformer block for 16 test images:

```rust
fn save_attention_maps(
    images: &[f32],
    num_images: usize,
    patch_embedding: &PatchEmbeddingLayer,
    transformer_encoder: &TransformerEncoder,
    pos_encoding: &[f32],
) {
    // Run forward pass one image at a time
    // Extract attention weights from first block
    let attn_weights = blocks[0].attention_layer().get_attention_weights();
    // Write to logs/vit_attention_maps.csv
}
```

The CSV format stores flattened attention matrices: each line contains `num_heads * seq_len * seq_len` = 4 * 64 * 64 = 16,384 comma-separated values for one image.

### Running the Visualizer

After training, use the Python visualization script:

```bash
# Default paths (reads logs/vit_attention_maps.csv, writes to plots/)
python visualize_vit_attention.py

# Custom paths
python visualize_vit_attention.py --input logs/vit_attention_maps.csv --output plots/vit_attention_maps.png
```

The script produces two types of visualizations:

**1. Attention Heatmaps:** A grid of [images x heads], where each cell is a 64x64 matrix showing how strongly each token attends to every other token.

```
                Head 0          Head 1          Head 2          Head 3
Image 0     [64x64 heatmap] [64x64 heatmap] [64x64 heatmap] [64x64 heatmap]
Image 1     [64x64 heatmap] [64x64 heatmap] [64x64 heatmap] [64x64 heatmap]
Image 2     ...
```

**2. Spatial Attention Maps:** For a chosen query token, the attention weights are reshaped to the 8x8 patch grid, showing which spatial regions the query patch attends to. The query patch is marked with a cyan star.

### Interpreting Attention Patterns

**What to look for:**

- **Diagonal dominance:** Tokens attending mostly to themselves. Common in early training or when the model is uncertain.
- **Local clusters:** Attention concentrated on spatially nearby patches. Indicates the model learned local feature relationships (similar to convolution).
- **Global patterns:** Attention distributed across distant patches. Indicates the model is relating semantically similar regions (e.g., all "sky" patches attending to each other).
- **Head specialization:** Different heads showing different patterns. One head might be local, another global -- this diversity is a sign of a well-trained model.

**Per-head statistics** printed by the visualizer:

```
Image 0:
  Head 0: min=0.0023, max=0.0412, mean=0.0156, diag_mean=0.0234, entropy=3.8912
  Head 1: min=0.0015, max=0.0587, mean=0.0156, diag_mean=0.0189, entropy=3.7234
  ...
```

- **diag_mean:** Average self-attention weight. Higher means the token focuses more on itself.
- **entropy:** Measures how spread out the attention is. Higher entropy means more uniform attention; lower entropy means the head is more selective.

---

## Verification Checkpoints

Use these to verify your understanding of the ViT architecture.

### Checkpoint 1: Patch Dimensions

For a 32x32x3 CIFAR-10 image with 4x4 patches:
- How many patches? 8 x 8 = **64**
- Dimension of each patch? 4 x 4 x 3 = **48**
- Total values in patch buffer (one image)? 64 x 48 = **3,072** (same as the original image, no information lost)

### Checkpoint 2: Embedding Sizes

For PatchEmbeddingLayer(48, 128):
- Weight matrix shape? **[48 x 128]**
- Bias vector shape? **[128]**
- Parameter count? 48 x 128 + 128 = **6,272**
- Output shape for one image? **[64 x 128]**

### Checkpoint 3: Attention Dimensions

For Multi-Head Attention with d_model=128, num_heads=4:
- Per-head dimension d_k? 128 / 4 = **32**
- Attention score matrix shape (per head)? **[64 x 64]**
- Total attention weights (all heads, one image)? 4 x 64 x 64 = **16,384**

### Checkpoint 4: Mean Pooling

For Transformer output [64 x 128]:
- Pooled output shape? **[128]** (one vector per image)
- If all tokens are identical with value v in dimension d, what is pooled[d]? **v** (mean of identical values is the value itself)
- Gradient of mean pooling: if grad_pooled[d] = g, what is grad_transformer[t][d] for each token t? **g / 64**

### Checkpoint 5: Positional Encoding

For sinusoidal encoding at position 0:
- PE(0, 0) = sin(0) = **0.0**
- PE(0, 1) = cos(0) = **1.0**
- PE(0, 2) = sin(0) = **0.0**
- PE(0, 3) = cos(0) = **1.0**

For position 1, dimension 0:
- PE(1, 0) = sin(1 / 10000^0) = sin(1) = **0.8415**

All positional encoding values lie in **[-1, 1]** (range of sin and cos).

---

## Exercises

### Beginner

1. **Verify patch extraction:** Write a test that creates a synthetic 32x32x3 image where each patch has a unique constant color. Extract patches and verify that each patch token contains the expected 48 values. This confirms the `extract_patches_rgb` function handles the pixel-interleaved format correctly.

2. **Count parameters manually:** Using the formulas from the Architecture Overview section, compute the total parameter count for the full ViT model. Verify your calculation matches the output printed by the binary. Break it down by component: patch embedding, each Transformer block (attention + FFN + layer norms), and classifier.

3. **Inspect positional encodings:** Generate the 64x128 positional encoding matrix and plot the first 8 dimensions as a function of position (0 to 63). What patterns do you see? Do nearby positions have similar encodings?

### Intermediate

4. **Vary patch size:** Modify `PATCH_SIZE` from 4 to 8. This gives 4x4 = 16 patches of dimension 192. How does this affect: (a) the number of tokens (and thus attention computational cost), (b) classification accuracy, (c) training speed? What about patch size 2 (256 patches of dimension 12)?

5. **Replace mean pooling with max pooling:** Instead of averaging all 64 token outputs, take the element-wise maximum. Does this change classification accuracy? What about the gradient flow -- how does the backward pass differ?

6. **Experiment with model depth:** Try 2 blocks instead of 4 and 8 blocks instead of 4. How does depth affect: (a) total parameter count, (b) training time per epoch, (c) final accuracy, (d) tendency to overfit?

### Advanced

7. **Add a learnable CLS token:** Prepend a learnable 128-dim vector as token 0, making the sequence 65 tokens long. Use only the CLS token output (instead of mean pooling) for classification. Compare accuracy with the mean pooling approach. Does the CLS token's attention pattern differ from other tokens?

8. **Implement 2D positional encoding:** Replace the 1D sinusoidal encoding with a 2D variant: encode the row index and column index separately, then concatenate (64 dims for row + 64 dims for column = 128 dims). Does this improve accuracy? Visualize the attention maps -- do spatial patterns change?

9. **Data augmentation:** Implement random horizontal flip and random crop (pad by 4 pixels, then crop back to 32x32) during training. These are standard augmentations for CIFAR-10. Measure the accuracy improvement. ViTs benefit more from data augmentation than CNNs because they lack the built-in translation equivariance of convolution.

10. **Attention rollout:** Implement attention rollout by multiplying attention matrices across all 4 blocks: `A_rollout = A_4 * A_3 * A_2 * A_1`. This gives an approximation of how information flows from the input patches to the final representation. Visualize the rollout and compare it with the per-block attention maps.

---

## Next Steps

You have now built a Vision Transformer from scratch -- the same fundamental architecture behind modern vision models like DeiT, Swin Transformer, and the vision encoders in multimodal models.

**Within this codebase:**
- Compare the ViT with the CIFAR-10 CNN (`cifar10_cnn.rs`) -- same dataset, fundamentally different architectures
- Examine the Transformer block implementation in `src/layers/transformer.rs`
- Explore the MNIST attention model (`mnist_attention_pool.rs`) which applies a similar attention mechanism to grayscale digit classification
- Study the `PatchEmbeddingLayer` tests in `src/layers/patch_embedding.rs`

**Theory to study next:**
- **Data-efficient Image Transformers (DeiT):** Training ViTs effectively on smaller datasets with knowledge distillation
- **Swin Transformer:** Hierarchical vision Transformer with shifted windows for O(n) complexity
- **Hybrid CNN-Transformer models:** Use CNN features as patch embeddings instead of raw pixels
- **Pre-training strategies:** Masked image modeling (MAE, BEiT) for self-supervised ViT pre-training
- **Attention variants:** Linear attention, flash attention, and other efficient attention mechanisms

**Practical extensions:**
- Add dropout to the attention weights and FFN layers for regularization
- Implement gradient clipping to stabilize training
- Try AdamW with weight decay for better generalization
- Experiment with different d_model values (64, 256) and observe the accuracy-speed tradeoff
- Export attention maps for all 4 blocks (not just the first) to see how attention evolves through depth

---

**Summary of what you learned:**

| Concept | Key Insight |
|---------|-------------|
| Patch tokenization | Images become sequences by splitting into non-overlapping patches |
| Patch embedding | Linear projection maps raw patches to a learned representation space |
| Positional encoding | Sinusoidal encodings inject spatial information into permutation-invariant attention |
| Self-attention on patches | Every patch can attend to every other patch in a single layer |
| Multi-head attention | Different heads specialize in different spatial relationships |
| Mean pooling | Averaging all token outputs gives a robust image-level representation |
| Pre-LN Transformer | LayerNorm before each sub-layer improves training stability |
| Attention visualization | Inspecting attention weights reveals what the model has learned |

[Tutorial 06: Autoencoders](06_autoencoder.md) | [Tutorial Index](README.md)
