# Tutorial 04: LSTM Character-Level Language Model

**Level:** Expert
**Time:** 120-180 minutes
**Prerequisites:** Tutorial 03 (MNIST CNN), calculus (chain rule through time), matrix calculus
**Implementation:** See `rnn_char_level.rs` for complete working code

**Navigation:**
← [Previous Tutorial: MNIST CNN](03_mnist_cnn.md) | [Tutorial Index](README.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Recurrent Networks?](#why-recurrent-networks)
3. [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
4. [LSTM Architecture Overview](#lstm-architecture-overview)
5. [Gate 1: The Forget Gate](#gate-1-the-forget-gate)
6. [Gate 2: The Input Gate](#gate-2-the-input-gate)
7. [Gate 3: The Cell Candidate](#gate-3-the-cell-candidate)
8. [Cell State Update](#cell-state-update)
9. [Gate 4: The Output Gate](#gate-4-the-output-gate)
10. [Hidden State and Output Projection](#hidden-state-and-output-projection)
11. [Complete Forward Pass Walkthrough](#complete-forward-pass-walkthrough)
12. [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
13. [Gradient Clipping](#gradient-clipping)
14. [Character-Level Language Modeling](#character-level-language-modeling)
15. [Training the Model](#training-the-model)
16. [Text Generation](#text-generation)
17. [Verification and Expected Outputs](#verification-and-expected-outputs)
18. [Exercises](#exercises)
19. [Next Steps](#next-steps)

---

## Introduction

Welcome to the most advanced tutorial in this series! You'll implement and understand **Long Short-Term Memory (LSTM)** networks, the architecture that powered the first wave of practical sequence modeling: language translation, speech recognition, and text generation.

**What makes this tutorial different from the others:**

Unlike the MLP (fixed-size input → fixed-size output) or CNN (spatial patterns in 2D grids), LSTMs process **sequences of arbitrary length**. Each time step receives new input and passes information to the next step through two persistent state vectors.

**What you'll learn:**
- Why vanilla RNNs fail on long sequences (vanishing/exploding gradients)
- How LSTM gates selectively store, forget, and expose information
- The mathematics of Backpropagation Through Time (BPTT)
- Why gradient clipping is essential for stable LSTM training
- How to build a character-level language model that generates text

**The character-level task:**
Given the sequence `"Hell"`, predict the next character `"o"`. Given `"o Wor"`, predict `"l"`. By training on thousands of such predictions, the model learns spelling, grammar, and even writing style.

**Implementation reference:** All code shown here is from `rnn_char_level.rs` and `src/layers/lstm/`.

---

## Why Recurrent Networks?

### The Problem with Feedforward Networks for Sequences

Every architecture we've seen so far is **stateless** — it processes one fixed-size input and produces one output, with no memory of previous inputs:

```
MLP:   x ──→ Dense(512) ──→ Dense(10) ──→ y
CNN:   image ──→ Conv ──→ Pool ──→ Dense ──→ class
```

**What if your input is a sequence?**

Consider predicting the next word in:
- `"The cat sat on the ___"` → `"mat"`
- `"The capital of France is ___"` → `"Paris"`

The correct answer depends on **all previous words**, not just the last one. A feedforward network with a fixed input window can't handle:
- Variable-length sequences
- Long-range dependencies (the subject at position 0 determines the verb at position 15)
- Sequential patterns where order matters

### The Vanilla RNN Solution (and Its Problems)

A **Recurrent Neural Network (RNN)** adds a hidden state `h_t` that persists between time steps:

```
Vanilla RNN:
  h_{t-1} ──┐
             ├──→ h_t = tanh(x_t × W_x + h_{t-1} × W_h + b)
  x_t ───────┘
                   ↓
                   y_t = h_t × W_hy + b_y
```

**Key insight:** The hidden state `h_t` carries information from all previous inputs. At time step 5, `h_5` has "seen" inputs `x_0, x_1, x_2, x_3, x_4`.

**But vanilla RNNs have a fatal flaw** — they suffer from vanishing gradients during training, which we'll explore next.

---

## The Vanishing Gradient Problem

### The Core Issue

During backpropagation, gradients must flow **backward through time**. For a sequence of length `T`, the gradient of the loss at time `T` with respect to the weights at time `1` passes through `T-1` matrix multiplications:

```
∂L/∂W ∝ ∏_{t=1}^{T} ∂h_t/∂h_{t-1}

For vanilla RNN:
  ∂h_t/∂h_{t-1} = diag(tanh'(z_t)) × W_h
```

### Why This Causes Problems

Each term `∂h_t/∂h_{t-1}` contains the weight matrix `W_h`. When we multiply `T` such matrices together:

**If the largest singular value of `W_h` < 1:**
```
||∂L/∂h_1|| ≈ λ^T × ||∂L/∂h_T||
```
Where `λ < 1`. For T = 100, even λ = 0.9 gives: 0.9^100 ≈ 0.000027

**The gradient essentially vanishes** — weights early in the sequence receive near-zero updates and can't learn long-range dependencies.

**If the largest singular value of `W_h` > 1:**
Gradients **explode** instead, causing training to diverge with NaN values.

### Numerical Demonstration

```
Sequence: 50 time steps
tanh' range: (0, 1)
W_h singular values: all ≈ 0.95

Gradient magnification at each step: 0.95 × tanh'(z)
Average multiplier per step: ~0.85

Gradient at t=50 vs t=1 ratio: 0.85^49 ≈ 0.00034
```

A gradient 3000× smaller means the first character of a sentence effectively doesn't influence the loss. This is why vanilla RNNs struggle with sequences longer than ~10-15 steps.

### The LSTM Solution

LSTMs solve this through the **cell state** — a "highway" for gradients that allows them to flow backward with minimal attenuation:

```
LSTM cell state gradient path:
  ∂c_t/∂c_{t-1} = f_t  (element-wise, values in (0,1))

  ∂L/∂c_1 = (∏_{t=2}^{T} f_t) × ∂L/∂c_T
```

When forget gate values `f_t ≈ 1`, gradients flow unchanged. The gate learns to **open** when long-term memory is needed and **close** when it should be reset.

---

## LSTM Architecture Overview

### State Diagram

```
Time t-1                Time t                  Time t+1
─────────────────────────────────────────────────────────
           c_{t-1}                c_t
           ────────────────────────────────────────────→
                       ↑
                  ┌────┴────┐
                  │  Cell   │
  h_{t-1} ───────→  State   │─────────────────────→ h_t ───→
                  │  Update │
                  └────┬────┘
                       │ tanh(c_t)
                  ┌────▼────┐
     x_t ────────→  4 Gates │─────────────────────→ y_t
                  └─────────┘
```

### The Four Gates

| Gate | Symbol | Formula | Role |
|------|--------|---------|------|
| Forget | f_t | σ(x_t W_xf + h_{t-1} W_hf + b_f) | What to erase from memory |
| Input | i_t | σ(x_t W_xi + h_{t-1} W_hi + b_i) | What new info to write |
| Cell Candidate | c̃_t | tanh(x_t W_xc + h_{t-1} W_hc + b_c) | Candidate new information |
| Output | o_t | σ(x_t W_xo + h_{t-1} W_ho + b_o) | What to expose as output |

### Parameter Count for Input-size D, Hidden-size H, Output-size V

```
For each gate (×4): D×H + H×H + H = (D+H+1)×H input weights
Output projection:  H×V + V

Total ≈ 4(D+H+1)H + (H+1)V

Example: D=65 (ASCII chars), H=128, V=65
  Gates:    4 × (65 + 128 + 1) × 128 = 99,328
  Output:   128 × 65 + 65     = 8,385
  Total:    ≈ 107,713 parameters
```

This is **4× more parameters** than a vanilla RNN of the same hidden size — the price of gated memory.

---

## Gate 1: The Forget Gate

### Purpose

The forget gate decides which information in the **cell state** to discard. A value of `f_t[j] = 1.0` means "keep this memory unit completely." A value of `f_t[j] = 0.0` means "erase this memory unit completely."

**When would the forget gate activate?**
- Starting a new sentence: forget context from the previous sentence
- Counting items: reset the counter when a new list begins
- Pronoun resolution: when a new subject is introduced, forget the previous one

### Mathematical Definition

```
f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)
```

**Dimensions:**
- `x_t`: (1 × D) — input at time t (one-hot encoded character, D = vocab size)
- `W_xf`: (D × H) — input-to-forget weights
- `h_{t-1}`: (1 × H) — previous hidden state
- `W_hf`: (H × H) — hidden-to-forget weights
- `b_f`: (H,) — forget gate bias
- `f_t`: (1 × H) — forget gate values in (0, 1)

### Sigmoid Activation

The **sigmoid function** σ squashes all values to (0, 1):

```
σ(z) = 1 / (1 + e^{-z})

σ(0.0)  = 0.500  (neutral — partial forgetting)
σ(2.0)  = 0.880  (strong keep)
σ(5.0)  = 0.993  (almost complete keep)
σ(-2.0) = 0.119  (strong forget)
σ(-5.0) = 0.007  (almost complete forget)
```

This is ideal for gating: soft decisions between 0 (forget) and 1 (keep).

### Worked Example (Scalar, for intuition)

Let's trace through a **single forget gate unit** with scalar values:

```
Suppose H=1 (single hidden unit), D=2 (input size)

Input:    x_t = [0.8, 0.2]
Previous: h_{t-1} = [0.6]
Weights:  W_xf = [[0.5], [0.3]]   (2×1)
          W_hf = [[0.4]]          (1×1)
Bias:     b_f = [-0.1]

Step 1: x_t × W_xf
  = [0.8, 0.2] × [[0.5], [0.3]]
  = 0.8 × 0.5 + 0.2 × 0.3
  = 0.40 + 0.06 = 0.46

Step 2: h_{t-1} × W_hf
  = [0.6] × [[0.4]]
  = 0.6 × 0.4 = 0.24

Step 3: Pre-activation = 0.46 + 0.24 + (-0.1) = 0.60

Step 4: f_t = σ(0.60) = 1 / (1 + e^{-0.60})
            = 1 / (1 + 0.5488)
            = 1 / 1.5488
            = 0.645

Result: f_t = 0.645 → Keep 64.5% of cell state, discard 35.5%
```

### Rust Implementation Reference

```rust
// From src/layers/lstm/forward.rs

// x_t × W_xf  (BLAS sgemm)
unsafe {
    sgemm(Layout::RowMajor, Transpose::None, Transpose::None,
          batch_size as i32,    // M: rows of result
          hidden_size as i32,   // N: cols of result
          input_size as i32,    // K: shared dimension
          1.0,                  // alpha
          input, input_size as i32,    // A (input matrix)
          &self.w_xf, hidden_size as i32,  // B (weight matrix)
          0.0,                  // beta
          &mut forget_gate, hidden_size as i32);  // C (result)
}

// h_{t-1} × W_hf and add to forget_gate, then apply sigmoid
for i in 0..forget_gate.len() {
    let pre_activation = forget_gate[i] + hf_contrib[i] + self.b_f[i % hidden_size];
    forget_gate[i] = 1.0 / (1.0 + (-pre_activation).exp()); // sigmoid
}
```

---

## Gate 2: The Input Gate

### Purpose

The input gate controls **how much new information** to write into the cell state. It works in tandem with the cell candidate (next section):

- **Input gate** (i_t): "How much to write?" (a 0→1 scaling factor)
- **Cell candidate** (c̃_t): "What to write?" (proposed new values in (-1, 1))

### Mathematical Definition

```
i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)
```

Same structure as the forget gate, just different weights.

### What the Input Gate Learns

**High input gate (i_t ≈ 1):** "This is important new information, write it to memory"
- Example: At the start of a sentence, write the subject to memory
- Example: When encountering a number, store it for later arithmetic

**Low input gate (i_t ≈ 0):** "Ignore this input, don't update memory"
- Example: Common function words (the, a, an) rarely need to update long-term memory
- Example: Padding tokens in a padded batch

### Key Insight: Two Gates, Not One

Why have both a forget gate and an input gate? Why not use `(1 - forget_gate)` as the input gate?

**The decoupled design allows:**
- **Partial update:** `f_t ≈ 0.5, i_t ≈ 0.5` — blend old and new information
- **Clear and write:** `f_t ≈ 0, i_t ≈ 1` — replace old with new
- **Keep and write:** `f_t ≈ 1, i_t ≈ 1` — accumulate information (e.g., a running count)
- **Neither:** `f_t ≈ 1, i_t ≈ 0` — pass through unchanged

---

## Gate 3: The Cell Candidate

### Purpose

The cell candidate `c̃_t` (sometimes called the "cell gate" or "proposed update") generates the **content** that could be added to the cell state. Unlike the input/forget/output gates which use sigmoid (outputting 0→1 scalings), the cell candidate uses **tanh** (outputting -1→1 values):

```
c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)
```

### Why tanh, Not Sigmoid?

**Gate values** (forget, input, output) use sigmoid because they're **multiplicative scalings** — they scale other values between 0 and 1.

**Cell candidate** uses tanh because it represents **actual information content** that can be positive (store this value) or negative (store the opposite). The range (-1, 1) prevents the cell state from growing unboundedly.

| Function | Range | Use Case |
|----------|-------|----------|
| Sigmoid σ | (0, 1) | Scaling/gating — "how much?" |
| Tanh | (-1, 1) | Content — "what value?" |

### Numerical Example

```
Suppose the input encodes the character 'H':
  x_t = one-hot encoding of 'H' (mostly zeros, one 1.0)

The cell candidate might compute:
  c̃_t[0] = tanh(0.8)  = 0.664   (stores information about 'H')
  c̃_t[1] = tanh(-0.3) = -0.291  (stores negative signal)
  c̃_t[2] = tanh(0.1)  = 0.100   (small positive)
  ...
```

The actual meaning of each dimension is learned during training — these are not hand-engineered features.

---

## Cell State Update

### The Memory Highway

The **cell state** `c_t` is the LSTM's long-term memory. Unlike the hidden state, it flows through the network with only **element-wise operations** (no matrix multiplications), which is why gradients can flow more easily:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

Where `⊙` is element-wise (Hadamard) multiplication.

### Step-by-Step Interpretation

```
f_t ⊙ c_{t-1}:   Selectively forget old memory
                  f_t = 0 → erase this dimension
                  f_t = 1 → keep this dimension exactly

i_t ⊙ c̃_t:       Selectively write new information
                  i_t = 0 → don't add anything
                  i_t = 1 → add the full cell candidate

Sum:              Blend old memory and new information
```

### Why This Enables Long-Range Dependencies

Consider a simple counting task: count how many times the character 'a' has appeared.

```
Time 1: See 'a'  → f=1.0, i=1.0, c̃=+0.1
  c_1 = 1.0 × 0.0 + 1.0 × 0.1 = 0.1

Time 2: See 'b'  → f=1.0, i=0.0, c̃=...
  c_2 = 1.0 × 0.1 + 0.0 × ... = 0.1  (unchanged!)

Time 3: See 'a'  → f=1.0, i=1.0, c̃=+0.1
  c_3 = 1.0 × 0.1 + 1.0 × 0.1 = 0.2

Time 10: After 5 'a's seen: c ≈ 0.5
```

The count is preserved because `f_t = 1` for 'b' characters — they don't affect the counter. This simple mechanism, learned from data, enables LSTMs to track context across hundreds of time steps.

### Worked Numerical Example (Full Cell Update)

```
Hidden size H = 2 (for clarity)
Previous cell state: c_{t-1} = [0.5, -0.3]

Forget gate:      f_t = [0.8, 0.2]    (keep first unit, mostly forget second)
Input gate:       i_t = [0.1, 0.9]    (small input to first, large to second)
Cell candidate:   c̃_t = [0.4, 0.7]   (positive candidates)

Cell update:
  c_t[0] = f_t[0] × c_{t-1}[0] + i_t[0] × c̃_t[0]
          = 0.8 × 0.5 + 0.1 × 0.4
          = 0.40 + 0.04
          = 0.44   (preserved old memory + small addition)

  c_t[1] = f_t[1] × c_{t-1}[1] + i_t[1] × c̃_t[1]
          = 0.2 × (-0.3) + 0.9 × 0.7
          = -0.06 + 0.63
          = 0.57   (mostly forgot old, wrote new)
```

**Result:** `c_t = [0.44, 0.57]`

The first unit preserved old information (high forget, low input). The second unit was mostly reset and rewritten (low forget, high input). This asymmetric behavior is the key to LSTM's flexibility.

---

## Gate 4: The Output Gate

### Purpose

The output gate `o_t` controls **which parts of the cell state** to expose as the hidden state `h_t`. Even though the cell state contains all the accumulated memory, we might not want to reveal everything at every time step:

```
o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)
h_t = o_t ⊙ tanh(c_t)
```

### Why This Separation Matters

**The cell state** `c_t` is the raw long-term memory — it accumulates information over time.

**The hidden state** `h_t` is the output — it's what gets passed to the next time step AND used to generate predictions.

The output gate lets the LSTM maintain "private" memory in `c_t` without exposing it in every output. For example:
- A grammar-tracking unit might store subject-verb agreement context in `c_t`
- But only expose this information when generating a verb (via output gate)
- While freely accepting new information through the input gate at other times

### Cell State Tanh

Before applying the output gate, the cell state is passed through `tanh` to squeeze values back to (-1, 1):

```
tanh(c_t): maps cell state from arbitrary range to (-1, 1)
```

This prevents the hidden state from growing unboundedly even if the cell state accumulates large values over many time steps.

### Dimensions Recap

```
c_{t-1}: (H,)   — previous cell state
x_t:     (D,)   — current input (D = vocabulary size)
h_{t-1}: (H,)   — previous hidden state

Gates (all H-dimensional):
  f_t = σ(x_t W_xf + h_{t-1} W_hf + b_f)  ∈ (0,1)^H
  i_t = σ(x_t W_xi + h_{t-1} W_hi + b_i)  ∈ (0,1)^H
  c̃_t = tanh(x_t W_xc + h_{t-1} W_hc + b_c)  ∈ (-1,1)^H
  o_t = σ(x_t W_xo + h_{t-1} W_ho + b_o)  ∈ (0,1)^H

States:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t      ∈ ℝ^H
  h_t = o_t ⊙ tanh(c_t)                 ∈ (-1,1)^H

Output:
  y_t = h_t W_hy + b_y                  ∈ ℝ^V
```

---

## Hidden State and Output Projection

### Hidden State

The hidden state `h_t` serves two purposes:
1. **Passed to next time step:** Acts as "working memory" — the recent context
2. **Used for predictions:** Via the output projection

```
h_t = o_t ⊙ tanh(c_t)
```

Both `o_t` (sigmoid output) and `tanh(c_t)` are in (-1, 1), so `h_t ∈ (-1, 1)^H`.

### Output Projection

The output projection converts the hidden state to a vocabulary-size prediction:

```
y_t = h_t × W_hy + b_y
```

**Dimensions:**
- `h_t`: (H,)
- `W_hy`: (H × V) — hidden-to-vocabulary weights
- `b_y`: (V,)
- `y_t`: (V,) — raw scores (logits) for each character

### Softmax for Character Probabilities

The raw output `y_t` is converted to probabilities via softmax:

```
p_t[k] = exp(y_t[k]) / Σ_j exp(y_t[j])
```

For character-level modeling, `p_t` is a probability distribution over all characters in the vocabulary. We want `p_t[c_true]` to be as high as possible, where `c_true` is the actual next character.

### Cross-Entropy Loss

For a sequence of length T:

```
L = -1/T × Σ_{t=1}^{T} log(p_t[c_t])
```

Where `c_t` is the true character at time step t. This is the same cross-entropy loss used in Tutorial 02 (MNIST), but now summed over all time steps.

---

## Complete Forward Pass Walkthrough

### Example Sequence: "Hello"

Let's trace the LSTM through the sequence "Hell" predicting "ello" (character by character).

**Setup:**
- Vocabulary: a-z + A-Z + space + punctuation (≈ 65 characters)
- Encoding: one-hot vectors, D = 65
- Hidden size: H = 4 (tiny, for illustration)
- Initial state: h_0 = c_0 = [0, 0, 0, 0]

#### Time Step 1: Input 'H', Predict 'e'

```
x_1 = one-hot('H') = [0,...,0,1,0,...,0]  (1 at position 7, D=65)

With random initial weights (before training), suppose:
  Pre-activations for gates (scalar approximations):
    forget:        z_f = x_1 W_xf + h_0 W_hf + b_f ≈ [-0.3, 0.2, -0.1, 0.4]
    input:         z_i = x_1 W_xi + h_0 W_hi + b_i ≈ [0.5, -0.2, 0.8, 0.1]
    cell cand.:    z_c = x_1 W_xc + h_0 W_hc + b_c ≈ [0.7, -0.5, 0.3, -0.6]
    output:        z_o = x_1 W_xo + h_0 W_ho + b_o ≈ [0.2, 0.6, -0.4, 0.1]

After activations:
  f_1 = σ([-0.3, 0.2, -0.1, 0.4]) = [0.43, 0.55, 0.48, 0.60]
  i_1 = σ([0.5, -0.2, 0.8, 0.1])  = [0.62, 0.45, 0.69, 0.52]
  c̃_1 = tanh([0.7,-0.5, 0.3,-0.6]) = [0.60,-0.46, 0.29,-0.54]
  o_1 = σ([0.2, 0.6, -0.4, 0.1])  = [0.55, 0.65, 0.40, 0.52]

Cell state (c_0 = [0,0,0,0]):
  c_1 = f_1 ⊙ c_0 + i_1 ⊙ c̃_1
      = [0.43,0.55,0.48,0.60] ⊙ [0,0,0,0] + [0.62,0.45,0.69,0.52] ⊙ [0.60,-0.46,0.29,-0.54]
      = [0,0,0,0] + [0.372, -0.207, 0.200, -0.281]
      = [0.372, -0.207, 0.200, -0.281]

Hidden state:
  tanh(c_1) = tanh([0.372, -0.207, 0.200, -0.281])
            = [0.355, -0.205, 0.197, -0.273]
  h_1 = o_1 ⊙ tanh(c_1)
      = [0.55, 0.65, 0.40, 0.52] ⊙ [0.355, -0.205, 0.197, -0.273]
      = [0.195, -0.133, 0.079, -0.142]

Output logits (simplified, W_hy maps H=4 → V=65):
  y_1 = h_1 × W_hy + b_y  →  65-dimensional vector

Prediction: p_1 = softmax(y_1)
  (before training, roughly uniform ≈ 1/65 ≈ 1.5% per character)
  Target: 'e' should have highest probability
```

#### Time Step 2: Input 'e', Predict 'l' (states from step 1)

```
x_2 = one-hot('e'), h_1 = [0.195, -0.133, 0.079, -0.142], c_1 = [0.372, -0.207, 0.200, -0.281]

Now h_{t-1} ≠ 0! The previous state influences all gates.
This is what enables context: 'e' after 'H' is processed differently than 'e' alone.
```

After training on enough text, the LSTM learns that:
- 'H' at the start → likely followed by 'e', 'a', 'i', 'o', 'u'
- 'He' → likely followed by 'l', 'r', 'n'
- 'Hel' → very likely followed by 'l'
- 'Hell' → very likely followed by 'o', '!', 'o'

---

## Backpropagation Through Time (BPTT)

### The Challenge

Standard backpropagation works by applying the chain rule layer by layer. But in an RNN/LSTM, the same weights are **reused at every time step**. This means:

- The same `W_xf`, `W_hf`, etc. are used at time 1, 2, 3, ..., T
- The gradient for each weight is the **sum** of gradients across all time steps

### BPTT Algorithm

**Forward pass:** Store all inputs, gates, and states for each time step.

**Backward pass:** Process time steps in **reverse** order (T → 1), propagating gradients backward through both the output connections AND the recurrent connections:

```
Algorithm BPTT:

1. Run full forward pass, storing:
   - All inputs x_1, ..., x_T
   - All gate values: f_t, i_t, c̃_t, o_t for t=1..T
   - All states: h_t, c_t for t=0..T

2. Initialize state gradients:
   dh_T = 0  (no gradient from future hidden state)
   dc_T = 0  (no gradient from future cell state)

3. For t = T, T-1, ..., 1:
   a. Compute output gradient at this step
   b. Add incoming hidden-state gradient from step t+1: dh_t += dh_{t+1}_from_recurrence
   c. Add incoming cell-state gradient: dc_t += dc_{t+1}_from_recurrence
   d. Backprop through h_t = o_t ⊙ tanh(c_t)
   e. Backprop through c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
   f. Backprop through each gate's sigmoid/tanh
   g. Accumulate weight gradients: dW_xf += ..., etc.
   h. Compute state gradients for next (earlier) time step:
      dh_{t-1} = ...  (gradient to pass to step t-1)
      dc_{t-1} = ...  (gradient to pass to step t-1)

4. Update all weights with accumulated gradients
```

### Gradient Derivation for One Time Step

Starting from the loss gradient `dy_t = ∂L/∂y_t` at time step t, plus incoming state gradients `dh_next` and `dc_next`:

```
Step 1: Output projection backward
  dh_t = dy_t × W_hy^T + dh_next     (gradient w.r.t. h_t, adding recurrent gradient)
  dW_hy += h_t^T × dy_t               (weight gradient)
  db_y  += dy_t                        (bias gradient)

Step 2: Hidden state h_t = o_t ⊙ tanh(c_t)
  d_cell_tanh = dh_t ⊙ o_t            (gradient through tanh)
  do_t = dh_t ⊙ tanh(c_t)             (gradient to output gate)
  dc_t = d_cell_tanh ⊙ (1 - tanh²(c_t)) + dc_next  (gradient w.r.t. cell state)
       [where (1 - tanh²(c_t)) is tanh derivative, dc_next from next step]

Step 3: Cell state c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
  df_t  = dc_t ⊙ c_{t-1}              (gradient to forget gate)
  di_t  = dc_t ⊙ c̃_t                  (gradient to input gate)
  dc̃_t  = dc_t ⊙ i_t                   (gradient to cell candidate)
  dc_{t-1} = dc_t ⊙ f_t               (gradient to previous cell state!)

Step 4: Gate activations (sigmoid/tanh derivatives)
  Sigmoid derivative: σ'(z) = σ(z) × (1 - σ(z)) = gate × (1 - gate)
  Tanh derivative:    tanh'(z) = 1 - tanh²(z)

  δ_f = df_t ⊙ f_t ⊙ (1 - f_t)       (pre-activation gradient, forget gate)
  δ_i = di_t ⊙ i_t ⊙ (1 - i_t)       (pre-activation gradient, input gate)
  δ_c = dc̃_t ⊙ (1 - c̃_t²)             (pre-activation gradient, cell candidate)
  δ_o = do_t ⊙ o_t ⊙ (1 - o_t)       (pre-activation gradient, output gate)

Step 5: Weight gradients (accumulated over time steps)
  dW_xf += x_t^T × δ_f    dW_hf += h_{t-1}^T × δ_f    db_f += δ_f
  dW_xi += x_t^T × δ_i    dW_hi += h_{t-1}^T × δ_i    db_i += δ_i
  dW_xc += x_t^T × δ_c    dW_hc += h_{t-1}^T × δ_c    db_c += δ_c
  dW_xo += x_t^T × δ_o    dW_ho += h_{t-1}^T × δ_o    db_o += δ_o

Step 6: State gradients for previous time step
  dh_{t-1} = δ_f × W_hf^T + δ_i × W_hi^T + δ_c × W_hc^T + δ_o × W_ho^T
  dc_{t-1} = dc_t ⊙ f_t    (computed in step 3)
```

### Why the Cell State Gradient Flows Cleanly

Look at step 3: `dc_{t-1} = dc_t ⊙ f_t`

This is **element-wise multiplication by the forget gate**, not a matrix multiplication! The gradient flows backward through the cell state with only this multiplicative scaling. When `f_t ≈ 1` (network learns "keep this"), the gradient passes through unchanged.

Compare to vanilla RNN:
```
Vanilla: dh_{t-1} = dh_t × W_h^T × diag(tanh'(z))
LSTM:    dc_{t-1} = dc_t ⊙ f_t
```

The LSTM cell gradient avoids the recurrent matrix multiplication `W_h^T`, which is the primary source of vanishing/exploding gradients.

### BPTT in Rust

```rust
// From rnn_char_level.rs (simplified)

// Forward pass: collect all inputs and outputs
let mut all_inputs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
let mut all_outputs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

lstm.reset_state();
for t in 0..seq_len {
    let mut output = vec![0.0; vocab_size];
    lstm.forward(&inputs[t], &mut output, 1);
    all_inputs.push(inputs[t].clone());
    all_outputs.push(output);
}

// Backward pass in reverse order
let mut dh = vec![0.0f32; hidden_size];  // dh from t+1 step
let mut dc = vec![0.0f32; hidden_size];  // dc from t+1 step

for t in (0..seq_len).rev() {
    // Compute output gradient at this time step
    let grad_output = compute_softmax_cross_entropy_grad(&all_outputs[t], &targets[t]);

    let mut grad_input = vec![0.0f32; vocab_size];

    // backward_bptt returns (dh_{t-1}, dc_{t-1})
    (dh, dc) = lstm.backward_bptt(
        &all_inputs[t],
        &grad_output,
        &mut grad_input,
        &dh,   // incoming gradient from t+1
        &dc,   // incoming gradient from t+1
        1,
    );
}

// After full BPTT, apply gradient clipping and update
clip_gradients(&mut lstm, max_grad_norm);
lstm.update_parameters(learning_rate);
```

---

## Gradient Clipping

### The Exploding Gradient Problem

Even with LSTMs, gradients can occasionally explode. Consider a sequence where the forget gate becomes saturated near 1.0 for all time steps:

```
Each backward step: dh_{t-1} grows by a factor related to W_h magnitude
Over T=100 steps:   gradient can grow by factor 2^100 ≈ 10^30
Result: NaN values, training collapse
```

This is less common than in vanilla RNNs, but still possible with:
- Long sequences
- Poorly initialized weights
- Learning rates that are too high
- Certain unlucky random seeds

### Gradient Norm Clipping

The standard solution is **gradient norm clipping**: if the global gradient norm exceeds a threshold `max_norm`, scale all gradients proportionally:

```
Algorithm:
  g = all parameters' gradients as a flat vector
  norm = ||g||₂ = sqrt(Σᵢ gᵢ²)

  if norm > max_norm:
      g = g × (max_norm / norm)

  Apply g to update parameters
```

**Key properties:**
- **Direction preserved:** We only scale the magnitude, not the direction
- **All gradients scaled together:** Maintains relative ratios between parameter gradients
- **Bounded updates:** Each parameter update step has controlled magnitude
- **Threshold selection:** Typically max_norm = 1.0 to 5.0 (empirical choice)

### Worked Example

```
Suppose gradients before clipping:
  dW_hf = [[0.5, 3.2], [-1.4, 0.8]]  (some large values)
  db_f  = [0.1, -0.3]

  Flattened: [0.5, 3.2, -1.4, 0.8, 0.1, -0.3]

  Norm = sqrt(0.5² + 3.2² + 1.4² + 0.8² + 0.1² + 0.3²)
       = sqrt(0.25 + 10.24 + 1.96 + 0.64 + 0.01 + 0.09)
       = sqrt(13.19) ≈ 3.63

Clipping threshold: max_norm = 1.0
  Scale factor = 1.0 / 3.63 = 0.275

After clipping:
  dW_hf = [[0.138, 0.881], [-0.386, 0.220]]
  db_f  = [0.028, -0.083]

  New norm = 3.63 × 0.275 = 1.0 ✓
```

### Why Clipping Works Better Than Gradient Clamping

**Clamping** (element-wise): `g_i = clamp(g_i, -threshold, threshold)`
- **Problem:** Distorts the gradient direction
- Each dimension is independently clipped, so the angle of the gradient vector changes
- Leads to unstable oscillations

**Norm clipping** (global): Scale all gradients together
- **Preserves gradient direction** — only magnitude changes
- Mathematically equivalent to taking a step in the "true" gradient direction but with bounded step size
- Robust even when only a few gradients are large

### Rust Implementation

```rust
// Gradient clipping from rnn_char_level.rs

fn clip_gradients_by_norm(layer: &mut LstmLayer, max_norm: f32) {
    // Compute global gradient norm
    let mut total_norm_sq = 0.0f32;
    for grad in layer.get_all_gradients() {
        for g in grad {
            total_norm_sq += g * g;
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // Scale if exceeds threshold
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        layer.scale_gradients(scale);
    }
}
```

### When to Use Gradient Clipping

**Always use it for:**
- LSTMs (variable-length sequences)
- Vanilla RNNs (even more susceptible to exploding gradients)
- Any architecture with recurrent connections

**Typical hyperparameter ranges:**
- `max_norm = 1.0`: Conservative, prevents large updates
- `max_norm = 5.0`: More permissive, faster initial learning
- `max_norm = 0.5`: Very conservative, for unstable architectures

**Diagnostic:** If you see NaN loss values, gradient clipping threshold is likely too high (or not enabled).

---

## Character-Level Language Modeling

### Problem Formulation

**Input:** A text corpus (e.g., Shakespeare plays, Python code, Wikipedia articles)

**Task:** Learn a probability distribution over the next character given all previous characters:

```
P(c_t | c_1, c_2, ..., c_{t-1})
```

**Training procedure:**
1. Convert text to integer indices: 'H' → 7, 'e' → 4, 'l' → 11, ...
2. Create input-target pairs: for each position t, input = `c_t`, target = `c_{t+1}`
3. Encode inputs as one-hot vectors
4. Train LSTM to predict each next character

### Vocabulary and Encoding

```rust
// From rnn_char_level.rs

// Build vocabulary from training text
let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
    .into_iter()
    .collect();
chars.sort();

let vocab_size = chars.len();  // typically 65-100 for English text

// Character to index mapping
let char_to_idx: HashMap<char, usize> = chars.iter().enumerate()
    .map(|(i, &c)| (c, i))
    .collect();

// Index to character mapping (for generation)
let idx_to_char: HashMap<usize, char> = chars.iter().enumerate()
    .map(|(i, &c)| (i, c))
    .collect();
```

### One-Hot Encoding

Each character is represented as a one-hot vector of size `vocab_size`:

```
'H' (index 7 in alphabet) → [0,0,0,0,0,0,0,1,0,0,...,0]
                               0 1 2 3 4 5 6 7 8 9...

'e' (index 4) → [0,0,0,0,1,0,...,0]
```

**Why one-hot?** It treats all characters as equally dissimilar — no numeric ordering is imposed. Unlike integer encoding (`'a'=0, 'b'=1, ...`) which would imply 'b' is "closer" to 'a' than to 'z'.

### Sequence Batching

For training efficiency, we process **mini-sequences** of fixed length (e.g., `seq_len = 25`):

```
Full text: "To be or not to be, that is the question..."
         → Chunks of 25 characters (with target = shift by 1):

Input:   "To be or not to be, that"
Target:  "o be or not to be, that "

Input:   "is the question: Whether"
Target:  "s the question: Whether "
```

Between sequences, we **reset the LSTM state** (`h_0 = c_0 = 0`). This prevents gradients from flowing across sequences (which could theoretically help but makes memory management much simpler).

### Training Loop

```rust
// Simplified training loop from rnn_char_level.rs

for epoch in 0..num_epochs {
    let mut total_loss = 0.0;

    // Process text in overlapping/non-overlapping windows of seq_len
    for chunk_start in (0..text_len - seq_len).step_by(seq_len) {

        // Build input sequence (one-hot encoded)
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<usize> = Vec::new();

        for t in 0..seq_len {
            let input_char = text_chars[chunk_start + t];
            let target_char = text_chars[chunk_start + t + 1];

            let mut one_hot = vec![0.0f32; vocab_size];
            one_hot[char_to_idx[&input_char]] = 1.0;
            inputs.push(one_hot);
            targets.push(char_to_idx[&target_char]);
        }

        // Forward pass
        lstm.reset_state();
        let mut all_logits: Vec<Vec<f32>> = Vec::new();
        for t in 0..seq_len {
            let mut logits = vec![0.0f32; vocab_size];
            lstm.forward(&inputs[t], &mut logits, 1);
            all_logits.push(logits);
        }

        // Compute cross-entropy loss and gradient
        let mut sequence_loss = 0.0;
        let mut grad_outputs: Vec<Vec<f32>> = Vec::new();
        for t in 0..seq_len {
            let probs = softmax(&all_logits[t]);
            sequence_loss -= probs[targets[t]].ln();

            // Gradient of softmax cross-entropy
            let mut grad = probs.clone();
            grad[targets[t]] -= 1.0;  // dL/d(logit_t) = p_t - 1{t=true}
            grad_outputs.push(grad);
        }

        total_loss += sequence_loss / seq_len as f32;

        // BPTT: backward in reverse order
        let mut dh = vec![0.0f32; hidden_size];
        let mut dc = vec![0.0f32; hidden_size];
        for t in (0..seq_len).rev() {
            let mut grad_input = vec![0.0f32; vocab_size];
            (dh, dc) = lstm.backward_bptt(
                &inputs[t], &grad_outputs[t], &mut grad_input, &dh, &dc, 1
            );
        }

        // Clip gradients and update
        clip_gradients(&mut lstm, max_grad_norm);
        lstm.update_parameters(learning_rate);
    }

    println!("Epoch {}: Loss = {:.4}", epoch, total_loss / num_chunks as f32);
}
```

---

## Text Generation

### Sampling Strategy

Once trained, the LSTM generates text by **sampling from its own predictions**:

```
Algorithm:
  1. Set h_0 = c_0 = 0 (or use end-state from training text)
  2. Start with a seed character (or random choice)
  3. For each generation step:
     a. One-hot encode current character
     b. Forward pass → logits y_t
     c. Apply softmax → probability distribution p_t
     d. Sample: next_char ~ Categorical(p_t)
     e. Feed sampled character back as input for next step
  4. Repeat until desired length
```

### Temperature Sampling

The **temperature** parameter `τ` controls the randomness of generation:

```
Modified softmax with temperature:
  p_t[k] = exp(y_t[k] / τ) / Σ_j exp(y_t[j] / τ)

τ = 1.0: Normal distribution (as trained)
τ < 1.0: Sharper distribution (more conservative, repetitive)
τ > 1.0: Flatter distribution (more random, creative)
```

**Example:**

```
Logits: y = [2.0, 1.0, 0.5, 0.0, -1.0]  (for 5 characters)

τ = 1.0 (normal):
  p = softmax([2.0, 1.0, 0.5, 0.0, -1.0])
    = [0.537, 0.197, 0.122, 0.074, 0.027]

  Most likely: char 0 (53.7%)

τ = 0.5 (sharp):
  p = softmax([4.0, 2.0, 1.0, 0.0, -2.0])
    = [0.827, 0.112, 0.041, 0.015, 0.002]

  Much more deterministic: char 0 (82.7%)

τ = 2.0 (flat):
  p = softmax([1.0, 0.5, 0.25, 0.0, -0.5])
    = [0.336, 0.256, 0.200, 0.154, 0.094]

  Much more random: char 0 only 33.6%
```

### Generation Code

```rust
// From rnn_char_level.rs

fn generate_text(lstm: &LstmLayer, seed: char, length: usize,
                 temperature: f32, char_to_idx: &HashMap<char, usize>,
                 idx_to_char: &HashMap<usize, char>, rng: &mut SimpleRng) -> String {
    lstm.reset_state();
    let vocab_size = char_to_idx.len();

    let mut result = String::new();
    let mut current_char = seed;

    for _ in 0..length {
        // Encode current character
        let mut input = vec![0.0f32; vocab_size];
        input[char_to_idx[&current_char]] = 1.0;

        // Forward pass
        let mut logits = vec![0.0f32; vocab_size];
        lstm.forward(&input, &mut logits, 1);

        // Temperature-scaled softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scaled: Vec<f32> = logits.iter()
            .map(|&l| ((l - max_logit) / temperature).exp())
            .collect();
        let sum: f32 = exp_scaled.iter().sum();
        let probs: Vec<f32> = exp_scaled.iter().map(|&e| e / sum).collect();

        // Sample from distribution
        let sample = rng.gen_f32();
        let mut cumsum = 0.0;
        let mut next_idx = 0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                next_idx = i;
                break;
            }
        }

        current_char = idx_to_char[&next_idx];
        result.push(current_char);
    }

    result
}
```

### Expected Generation Quality

**After 0 epochs (random):**
```
Generated: "xqK!mzP3#Lv0aTcpRn2bYeOs"
```
(Random characters with no structure)

**After 1 epoch (some patterns):**
```
Generated: "the the the the the and "
```
(Has learned common words but repetitive)

**After 10 epochs (better structure):**
```
Generated: "To be or not to be, that is the question whether"
```
(Word and sentence structure beginning to emerge)

**After 50 epochs (reasonable text):**
```
Generated: "What light through yonder window breaks, it is the east
and Juliet is the sun. Arise fair sun and kill the envious moon"
```
(Plausible Shakespeare-style sentences)

---

## Training the Model

### Model Configuration

```rust
// Architecture constants from rnn_char_level.rs
const HIDDEN_SIZE: usize = 128;   // LSTM hidden units
const SEQ_LEN: usize = 25;        // Training sequence length
const BATCH_SIZE: usize = 1;      // One sequence at a time (online learning)

// Training hyperparameters
const LEARNING_RATE: f32 = 0.01;  // Initial learning rate
const MAX_GRAD_NORM: f32 = 5.0;   // Gradient clipping threshold
const NUM_EPOCHS: usize = 50;     // Training epochs
```

### Parameter Count

For a typical character-level model:

```
Vocabulary size D = 65  (printable ASCII)
Hidden size H = 128

Gates (4 × each):
  Input weights (D×H):   65 × 128 = 8,320
  Hidden weights (H×H): 128 × 128 = 16,384
  Bias (H):              128
  Per gate: 24,832
  Total gates: 4 × 24,832 = 99,328

Output projection:
  W_hy (H×D): 128 × 65 = 8,320
  b_y (D):    65
  Total: 8,385

Grand total: ≈ 107,713 parameters
```

This is a small model by modern standards, but sufficient to learn character patterns in modest text corpora (~100K characters).

### Expected Training Curve

```
Epoch  1: Loss ≈ 4.17  (random: -log(1/65) ≈ 4.17)
Epoch  5: Loss ≈ 2.80  (learning common character n-grams)
Epoch 10: Loss ≈ 2.40  (learning word boundaries)
Epoch 25: Loss ≈ 2.10  (learning word structure)
Epoch 50: Loss ≈ 1.85  (learning phrase structure)
```

**Expected runtime (CPU):** ~5-15 minutes for 50 epochs on 100K character corpus.

---

## Verification and Expected Outputs

### Checkpoint 1: LSTM Forward Pass Shapes ✓

Verify dimension correctness before training:

```rust
let mut rng = SimpleRng::new(42);
let lstm = LstmLayer::new(65, 128, 65, &mut rng);

let input = vec![0.0f32; 65];   // One-hot vector (all zeros for now)
let mut output = vec![0.0f32; 65];

lstm.reset_state();
lstm.forward(&input, &mut output, 1);

// Verifications:
assert_eq!(output.len(), 65);              // Output size matches vocab
assert_eq!(lstm.get_hidden_state().len(), 128);  // Hidden size correct
assert_eq!(lstm.get_cell_state().len(), 128);    // Cell size correct

// Output should be non-zero (bias term alone makes it nonzero)
assert!(output.iter().any(|&x| x != 0.0));
```

### Checkpoint 2: Loss Starts Near -log(1/vocab_size) ✓

For random weights, the model has no preference — cross-entropy loss should start near `-log(1/65) ≈ 4.17`:

```
Initial loss: 4.0 - 4.4 is expected
If loss starts near 0: something is wrong (overfitting on first batch?)
If loss starts near 10+: gradient explosion (check initialization)
```

### Checkpoint 3: Loss Decreases After Each Epoch ✓

A properly training model shows monotonically decreasing loss (with small fluctuations):

```
Expected loss progression:
  Epoch  1: 4.1-4.2
  Epoch  2: 3.5-3.8
  Epoch  5: 2.8-3.2
  Epoch 10: 2.3-2.7

If loss plateaus immediately (< 2 epochs): learning rate too low
If loss oscillates wildly: learning rate too high or gradient explosion
If loss increases: gradient explosion (lower max_grad_norm or learning_rate)
```

### Checkpoint 4: Generated Text Shows Learned Patterns ✓

After 10 epochs, generated text should show:
- **Words** separated by spaces (not random characters)
- **Capitalization** at sentence starts
- **Punctuation** used somewhat correctly
- **Common words** ('the', 'of', 'to') appearing frequently

**Red flags:**
- All same character: mode collapse
- Random characters: not enough training
- Identical output regardless of seed: model is stuck

### Checkpoint 5: Gradient Clipping Engages Occasionally ✓

You can add a diagnostic print:

```rust
let pre_clip_norm = compute_gradient_norm(&lstm);
clip_gradients(&mut lstm, MAX_GRAD_NORM);
if pre_clip_norm > MAX_GRAD_NORM {
    println!("  Clipped: {:.2} → {:.2}", pre_clip_norm, MAX_GRAD_NORM);
}
```

**Expected behavior:**
- Clipping should engage on ~5-20% of batches early in training
- Frequency should decrease as training stabilizes
- If clipping engages on >80% of batches: learning rate is too high

### Checkpoint 6: BPTT Gradients Are Non-Zero for Early Time Steps ✓

Verify that gradients actually flow to early time steps:

```rust
// After a backward pass through 25 time steps:
let grad_h_prev = lstm.get_grad_h_prev();  // gradient to step 0 from step 1
assert!(grad_h_prev.iter().any(|&g| g.abs() > 1e-10),
        "Gradient flow blocked — BPTT not working");
```

If all gradients at early time steps are ~0, BPTT is broken (check that `backward_bptt` is called in reverse order with correct state gradient passing).

---

## Exercises

### Beginner: Understanding Gates

**Exercise 1.1 — Trace the forget gate manually:**

Using the scalar example from the "Gate 1" section, change `b_f = [0.5]` (positive bias). Compute the new forget gate value. What does a positive bias in the forget gate mean for learning? (Hint: it initializes the gate toward "keep" rather than "forget.")

**Exercise 1.2 — Gate interaction:**

For H=1, suppose after one training iteration:
- `f_t = 0.1` (mostly forgetting)
- `i_t = 0.9` (mostly inputting)
- `c_{t-1} = 2.0` (strong existing memory)
- `c̃_t = -0.5` (negative candidate)

Compute `c_t`. What is the LSTM "deciding" to do here?

**Expected:** `c_t = 0.1 × 2.0 + 0.9 × (-0.5) = 0.20 - 0.45 = -0.25`
The LSTM is replacing strong positive memory with weak negative information — a major memory reset.

### Intermediate: Architecture Experiments

**Exercise 2.1 — Hidden size scaling:**

Modify `HIDDEN_SIZE` and compare:
- `HIDDEN_SIZE = 32`: Faster training, lower capacity
- `HIDDEN_SIZE = 128`: Baseline
- `HIDDEN_SIZE = 512`: Slower, higher capacity

For each, record: (1) parameters, (2) time per epoch, (3) final loss after 20 epochs.

**Prediction before running:** Loss should decrease with hidden size, but with diminishing returns. Training time should scale roughly as `O(H²)`.

**Exercise 2.2 — Sequence length impact:**

Try `SEQ_LEN = 10` vs `SEQ_LEN = 50`. Which converges faster (per epoch)? Which generates better long-range text patterns? Why?

**Hint:** Longer sequences require more BPTT steps, giving the model more context but also more gradient propagation distance.

### Advanced: BPTT Deep Dive

**Exercise 3.1 — Truncated BPTT:**

Implement **truncated BPTT**: instead of backpropagating through the full `SEQ_LEN`, truncate after `k` steps. Compare `k = 5`, `k = 10`, `k = 25` (full).

```rust
// Truncated: only backprop through last k steps
let trunc_k = 10;
for t in (seq_len - trunc_k..seq_len).rev() {
    (dh, dc) = lstm.backward_bptt(&inputs[t], &grad_outputs[t],
                                   &mut grad_input, &dh, &dc, 1);
}
```

**Expected:** `k < SEQ_LEN` trains faster but the model may not learn patterns longer than `k` characters. This is a common practical trade-off.

**Exercise 3.2 — Gradient norm monitoring:**

Add gradient norm logging before and after clipping:

```rust
let norms: Vec<f32> = (0..seq_len).rev().map(|t| {
    // record grad norm after each BPTT step
    compute_step_gradient_norm(&lstm)
}).collect();
```

Plot the gradient norm vs. time-step-from-end. Do early time steps have larger or smaller gradients than late time steps? Why?

**Expected:** Early time steps (far from loss) should have smaller gradients due to gradient attenuation, but larger than in vanilla RNN thanks to the cell state highway.

### Expert: Numerical Gradient Checking

**Exercise 4.1 — Verify BPTT is correct:**

Implement numerical gradient checking for the LSTM:

```rust
// Numerical gradient for parameter θ_i:
let epsilon = 1e-4;
theta_i += epsilon;
let loss_plus = compute_loss(&lstm, &inputs, &targets);
theta_i -= 2.0 * epsilon;
let loss_minus = compute_loss(&lstm, &inputs, &targets);
theta_i += epsilon;  // restore

let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
let analytical_grad = get_gradient_for_param(&lstm, i);

let relative_error = (numerical_grad - analytical_grad).abs()
    / (numerical_grad.abs() + analytical_grad.abs() + 1e-8);
assert!(relative_error < 1e-4, "Gradient check failed: {}", relative_error);
```

This test confirms that your BPTT implementation is mathematically correct. Relative errors below `1e-3` indicate correct gradients.

---

## Next Steps

### What You've Learned

Congratulations on completing the most mathematically intensive tutorial in this series! You now understand:

- **Why LSTMs outperform vanilla RNNs** — the cell state highway mitigates vanishing gradients
- **How all four gates work** — forget (erase), input (write), cell candidate (content), output (read)
- **The complete LSTM mathematics** — all 7 forward pass equations and their backward pass counterparts
- **Backpropagation Through Time** — how gradients flow backward through sequences with state gradient propagation
- **Gradient clipping** — why it's essential and how norm clipping preserves gradient direction
- **Character-level modeling** — one-hot encoding, sequence batching, temperature sampling

### Related Architectures to Explore

**Bidirectional LSTM:**
```
Forward LSTM:  h→  h→  h→  h→
Backward LSTM: ←h  ←h  ←h  ←h
Output: concatenate forward and backward hidden states
```
Processes sequences in both directions for tasks where full context is available (e.g., named entity recognition).

**Stacked LSTMs:**
```
Layer 2: LSTM with input = h_t from Layer 1
Layer 1: LSTM with input = x_t
```
Each layer learns increasingly abstract temporal patterns. Modern language models use 4-12 stacked layers.

**GRU (Gated Recurrent Unit):**
A simplified LSTM with only 2 gates (update + reset) and no separate cell state. Fewer parameters, comparable performance on many tasks.

**Transformer Attention:**
The architecture that replaced LSTMs for most NLP tasks in 2017+. Instead of processing sequences step-by-step, attention mechanisms allow each position to "attend" to all other positions simultaneously. For a transformer-based follow-up with self-attention, see [Tutorial 07: Vision Transformer](07_vision_transformer.md).

### From Character Level to Word Level

The same LSTM architecture works for word-level language models:
- Input: one-hot (or learned embedding) over vocabulary of 10K-100K words
- Output: probability distribution over all words
- Challenge: much larger vocabulary makes output layer expensive

**Word embeddings:** Instead of one-hot encoding words, learn a dense embedding vector for each word (e.g., 300-dimensional). Similar words cluster in embedding space, enabling generalization.

---

## Related Documentation

**Mathematical foundations:**
- [Backpropagation Guide](../backpropagation/README.md) — Gradient computation theory
- [Dense Layer Backpropagation](../backpropagation/dense_layer.md) — Fully connected layer gradients
- [Activation Functions](../activation_functions.md) — Sigmoid, tanh, ReLU and their derivatives
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) — Notation conventions

**Implementation details:**
- `rnn_char_level.rs` — Full character-level LSTM training (this tutorial's code)
- `src/layers/lstm/mod.rs` — LstmLayer struct and public interface
- `src/layers/lstm/forward.rs` — Gate computations (forward pass implementation)
- `src/layers/lstm/backward.rs` — BPTT gradient computations (backward pass)
- `src/layers/trait.rs` — Layer trait interface

**Training infrastructure:**
- `config/training/rnn_char_level_default.json` — Default LSTM training configuration
- `src/config.rs` — Configuration loading and validation

**Tests (gradient validation):**
- `tests/test_lstm_bptt.rs` — BPTT correctness tests
- `tests/test_gradient_checking.rs` — Numerical gradient validation

**Related tutorials:**
- [Tutorial 01: XOR MLP](01_xor_mlp.md) — Backpropagation fundamentals
- [Tutorial 02: MNIST MLP](02_mnist_mlp.md) — Dense layers and cross-entropy
- [Tutorial 03: MNIST CNN](03_mnist_cnn.md) — Spatial structure and convolutions
- [Tutorial 05: Automatic Differentiation Engine](05_autograd_engine.md) — Current next tutorial in the sequence
- [Tutorial 07: Vision Transformer](07_vision_transformer.md) — Transformer self-attention applied to images

---

*Tutorial 04 complete — you've mastered one of the most important architectures in deep learning history.*
