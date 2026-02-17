# LSTM Layer Mathematics: Forward Pass and Backpropagation Through Time

This document provides a comprehensive explanation of the mathematics behind the Long Short-Term Memory (LSTM) layer, with detailed derivations of the forward pass equations, Backpropagation Through Time (BPTT), and the critical cell state gradient highway that gives LSTMs their ability to learn long-range dependencies.

## Table of Contents

- [Overview](#overview)
- [Architecture and Key Characteristics](#architecture-and-key-characteristics)
- [Forward Pass Through Time](#forward-pass-through-time)
  - [Equation 1: Forget Gate](#equation-1-forget-gate)
  - [Equation 2: Input Gate](#equation-2-input-gate)
  - [Equation 3: Cell Candidate](#equation-3-cell-candidate)
  - [Equation 4: Cell State Update](#equation-4-cell-state-update)
  - [Equation 5: Output Gate](#equation-5-output-gate)
  - [Equation 6: Hidden State Update](#equation-6-hidden-state-update)
  - [Equation 7: Output Projection](#equation-7-output-projection)
  - [Dimension Analysis](#dimension-analysis)
  - [Computational Graph](#computational-graph)
  - [Implementation Details](#implementation-details)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass - BPTT](#backward-pass---bptt)
  - [BPTT Overview](#bptt-overview)
  - [Step 1: Output Projection Gradients](#step-1-output-projection-gradients)
  - [Step 2: Hidden State Gradient](#step-2-hidden-state-gradient)
  - [Step 3: Cell State Gradient](#step-3-cell-state-gradient)
  - [Step 4: Output Gate Gradients](#step-4-output-gate-gradients)
  - [Step 5: Cell State Decomposition](#step-5-cell-state-decomposition)
  - [Step 6: Forget Gate Gradients](#step-6-forget-gate-gradients)
  - [Step 7: Input Gate Gradients](#step-7-input-gate-gradients)
  - [Step 8: Cell Candidate Gradients](#step-8-cell-candidate-gradients)
  - [Step 9: Input Gradient](#step-9-input-gradient)
  - [Step 10: Previous State Gradients](#step-10-previous-state-gradients)
- [Cell State Gradient Highway](#cell-state-gradient-highway)
  - [Mathematical Analysis](#mathematical-analysis)
  - [Why This Solves Vanishing Gradients](#why-this-solves-vanishing-gradients)
  - [Comparison with RNN Gradient Flow](#comparison-with-rnn-gradient-flow)
- [LSTM vs Vanilla RNN](#lstm-vs-vanilla-rnn)
  - [Architecture Differences](#architecture-differences)
  - [Gradient Flow Comparison](#gradient-flow-comparison)
  - [Parameter Count](#parameter-count)
  - [When to Use Each](#when-to-use-each)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Numerical Considerations](#numerical-considerations)

## Overview

Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber (1997), solve the fundamental problem of vanilla RNNs: the **vanishing gradient problem**. LSTMs use a **gated architecture** with a separate **cell state** that acts as a gradient highway, allowing information and gradients to flow relatively unchanged across many time steps.

**Key characteristics:**
- **Cell state**: Separate long-term memory vector that flows with minimal modification
- **Gating mechanism**: Three learned gates control information flow
- **Gradient highway**: Cell state enables gradient propagation over hundreds of time steps
- **Selective memory**: Gates learn what to remember, update, and output
- **Parameter sharing**: Same gates and weights applied at every time step

**The fundamental insight:**

The LSTM uses two separate state vectors:
1. **Cell state c_t**: Long-term memory (analogous to conveyor belt)
2. **Hidden state h_t**: Short-term working memory (used for output)

The cell state flows with only multiplicative and additive interactions, making gradients much easier to propagate than through the tanh-squashing recurrence of vanilla RNNs.

**Core reference:**
- Implementation: `src/layers/lstm/mod.rs`, `src/layers/lstm/forward.rs`, `src/layers/lstm/backward.rs`

## Architecture and Key Characteristics

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| W_xf | (input_size × hidden_size) | Input-to-forget gate weights |
| W_hf | (hidden_size × hidden_size) | Hidden-to-forget gate weights |
| b_f | (hidden_size,) | Forget gate bias |
| W_xi | (input_size × hidden_size) | Input-to-input gate weights |
| W_hi | (hidden_size × hidden_size) | Hidden-to-input gate weights |
| b_i | (hidden_size,) | Input gate bias |
| W_xc | (input_size × hidden_size) | Input-to-cell candidate weights |
| W_hc | (hidden_size × hidden_size) | Hidden-to-cell candidate weights |
| b_c | (hidden_size,) | Cell candidate bias |
| W_xo | (input_size × hidden_size) | Input-to-output gate weights |
| W_ho | (hidden_size × hidden_size) | Hidden-to-output gate weights |
| b_o | (hidden_size,) | Output gate bias |
| W_hy | (hidden_size × output_size) | Hidden-to-output projection weights |
| b_y | (output_size,) | Output bias |

**State vectors:**

| State | Shape | Description |
|-------|-------|-------------|
| h_t | (hidden_size,) | Hidden state (short-term memory) |
| c_t | (hidden_size,) | Cell state (long-term memory) |

**Use cases:**
- Language modeling and text generation
- Sequence classification with long-range context
- Time series forecasting with complex temporal patterns
- Machine translation
- Any task requiring memory of events >10-20 time steps in the past

## Forward Pass Through Time

The LSTM forward pass computes **seven sequential operations** at each time step t, given input x_t ∈ ℝ^input_size and previous states h_{t-1} ∈ ℝ^hidden_size, c_{t-1} ∈ ℝ^hidden_size.

**Initial conditions:**
```
h⁽⁰⁾ = 0   (or learned initialization)
c⁽⁰⁾ = 0   (or learned initialization)
```

### Equation 1: Forget Gate

**Purpose:** Controls what fraction of the previous cell state to retain. Values near 0 → forget, values near 1 → keep.

```
f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)
```

**Element-wise expanded form** (for hidden dimension j):

```
z_f,j = Σᵢ x_t,i · W_xf[i,j]  +  Σₖ h_{t-1,k} · W_hf[k,j]  +  b_f[j]

f_t,j = σ(z_f,j) = 1 / (1 + exp(-z_f,j))
```

**Dimension check:**
```
x_t × W_xf: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
h_{t-1} × W_hf: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
b_f broadcast: (hidden_size,) → (batch_size × hidden_size)
f_t: (batch_size × hidden_size), values in (0, 1)
```

**Interpretation:** A forget gate value of 0.0 completely erases the corresponding cell state dimension; a value of 1.0 preserves it perfectly.

### Equation 2: Input Gate

**Purpose:** Controls what fraction of the new cell candidate to write to the cell state.

```
i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)
```

**Element-wise expanded form:**

```
z_i,j = Σᵢ x_t,i · W_xi[i,j]  +  Σₖ h_{t-1,k} · W_hi[k,j]  +  b_i[j]

i_t,j = σ(z_i,j) = 1 / (1 + exp(-z_i,j))
```

**Dimension check:**
```
i_t: (batch_size × hidden_size), values in (0, 1)
```

**Interpretation:** The input gate decides how much of the new candidate information to write. Working together with the forget gate, the LSTM can learn to keep old information OR write new information OR do both.

### Equation 3: Cell Candidate

**Purpose:** Computes the new candidate values that could be added to the cell state. Uses tanh to bound values in (-1, 1).

```
c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)
```

**Element-wise expanded form:**

```
z_c,j = Σᵢ x_t,i · W_xc[i,j]  +  Σₖ h_{t-1,k} · W_hc[k,j]  +  b_c[j]

c̃_t,j = tanh(z_c,j) = (exp(z_c,j) - exp(-z_c,j)) / (exp(z_c,j) + exp(-z_c,j))
```

**Dimension check:**
```
c̃_t: (batch_size × hidden_size), values in (-1, 1)
```

**Interpretation:** The candidate represents "what new information we might want to store." The tanh bounds these values and provides zero-centered representations, which is better for learning than one-sided values.

### Equation 4: Cell State Update

**Purpose:** The **critical equation** — updates the cell state by selectively forgetting old information and adding new information. This is the gradient highway.

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

Where ⊙ denotes element-wise multiplication (Hadamard product).

**Element-wise expanded form:**

```
c_t,j = f_t,j · c_{t-1,j}  +  i_t,j · c̃_t,j
```

**Dimension check:**
```
f_t ⊙ c_{t-1}: (batch_size × hidden_size) ⊙ (batch_size × hidden_size) → (batch_size × hidden_size)
i_t ⊙ c̃_t:   (batch_size × hidden_size) ⊙ (batch_size × hidden_size) → (batch_size × hidden_size)
c_t:           (batch_size × hidden_size)
```

**Interpretation:**
- `f_t ⊙ c_{t-1}`: Selectively retain relevant parts of old cell state
- `i_t ⊙ c̃_t`: Selectively write relevant new information
- The additive nature (`+`) means gradients can flow back through time without repeated multiplication by recurrent weights!

### Equation 5: Output Gate

**Purpose:** Controls what portion of the cell state to expose as the hidden state output.

```
o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)
```

**Element-wise expanded form:**

```
z_o,j = Σᵢ x_t,i · W_xo[i,j]  +  Σₖ h_{t-1,k} · W_ho[k,j]  +  b_o[j]

o_t,j = σ(z_o,j) = 1 / (1 + exp(-z_o,j))
```

**Dimension check:**
```
o_t: (batch_size × hidden_size), values in (0, 1)
```

**Interpretation:** The output gate decides which parts of the cell state are relevant to expose as output. The cell state may contain many encoded memories; the output gate selects which are relevant for the current prediction.

### Equation 6: Hidden State Update

**Purpose:** Computes the new hidden state by filtering the cell state through the output gate. The hidden state is used for all downstream computations.

```
h_t = o_t ⊙ tanh(c_t)
```

**Element-wise expanded form:**

```
h_t,j = o_t,j · tanh(c_t,j)
```

**Dimension check:**
```
tanh(c_t): (batch_size × hidden_size), values in (-1, 1)
h_t:       (batch_size × hidden_size), values in (-1, 1)
```

**Interpretation:**
- `tanh(c_t)`: Squashes cell state to bounded representation
- `o_t ⊙ tanh(c_t)`: Selectively expose relevant cell state dimensions
- The hidden state acts as the "short-term working memory" used for predictions

### Equation 7: Output Projection

**Purpose:** Maps the hidden state to the output space (e.g., vocabulary logits, class scores, regression values).

```
y_t = h_t × W_hy + b_y
```

**Element-wise expanded form** (for output dimension k):

```
y_t,k = Σⱼ h_t,j · W_hy[j,k]  +  b_y[k]
```

**Dimension check:**
```
h_t × W_hy: (batch_size × hidden_size) × (hidden_size × output_size) → (batch_size × output_size)
y_t:        (batch_size × output_size)
```

**Interpretation:** A standard linear projection maps the hidden representation to the task-specific output space. This is separate from the recurrent machinery and can be followed by softmax (for classification) or other activations.

### Dimension Analysis

**Complete parameter dimension table:**

| Parameter | Shape | Total Elements (input=128, hidden=256, output=10) |
|-----------|-------|--------------------------------------------------|
| W_xf | (input_size × hidden_size) | 128 × 256 = 32,768 |
| W_hf | (hidden_size × hidden_size) | 256 × 256 = 65,536 |
| b_f | (hidden_size,) | 256 |
| W_xi | (input_size × hidden_size) | 32,768 |
| W_hi | (hidden_size × hidden_size) | 65,536 |
| b_i | (hidden_size,) | 256 |
| W_xc | (input_size × hidden_size) | 32,768 |
| W_hc | (hidden_size × hidden_size) | 65,536 |
| b_c | (hidden_size,) | 256 |
| W_xo | (input_size × hidden_size) | 32,768 |
| W_ho | (hidden_size × hidden_size) | 65,536 |
| b_o | (hidden_size,) | 256 |
| W_hy | (hidden_size × output_size) | 256 × 10 = 2,560 |
| b_y | (output_size,) | 10 |
| **Total** | | **396,066** |

**State dimensions:**

| Variable | Shape | Description |
|----------|-------|-------------|
| x_t | (batch_size, input_size) | Input at time t |
| h_t | (batch_size, hidden_size) | Hidden state at time t |
| c_t | (batch_size, hidden_size) | Cell state at time t |
| f_t | (batch_size, hidden_size) | Forget gate values |
| i_t | (batch_size, hidden_size) | Input gate values |
| c̃_t | (batch_size, hidden_size) | Cell candidate values |
| o_t | (batch_size, hidden_size) | Output gate values |
| y_t | (batch_size, output_size) | Output at time t |

### Computational Graph

The computational graph for a single LSTM time step:

```
                    LSTM TIME STEP t

    x_t ─────────────────────────────────────────────┐
         │                │                │          │
         ▼                ▼                ▼          ▼
    [×W_xf]          [×W_xi]          [×W_xc]   [×W_xo]
         │                │                │          │
         +h_{t-1}×W_hf    +h_{t-1}×W_hi   +h_{t-1}×W_hc  +h_{t-1}×W_ho
         +b_f             +b_i             +b_c       +b_o
         │                │                │          │
         ▼                ▼                ▼          ▼
        [σ]              [σ]            [tanh]       [σ]
         │                │                │          │
         ▼                ▼                ▼          │
        f_t              i_t              c̃_t         │
         │                │                │          │
         │                └────────⊙───────┘          │
         │                         │                  │
         │          c_{t-1}        │                  │
         │              │          │                  │
         └──────⊙───────┘          │                  │
                │                  │                  │
                c_t  ←── + ────────┘                  │
                │                                     │
                ├──────────────[tanh]                 │
                │                   │                 │
                │                   └────────⊙────────┘
                │                            │
                │                           h_t
                │                            │
                │                         [×W_hy + b_y]
                │                            │
                │                           y_t
                ▼
           (to next step)
```

**Cell state "conveyor belt" visualization:**

```
c_{t-1} ──────────────────────────────────────────── c_t ──► (next step)
              ×f_t (forget)     + i_t ⊙ c̃_t (write)
```

### Implementation Details

The implementation in `src/layers/lstm/forward.rs` uses BLAS SGEMM for all matrix multiplications:

**Code snippet - Forget gate computation:**

```rust
// From src/layers/lstm/forward.rs

// Compute Forget Gate: f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)
let mut forget_gate = vec![0.0f32; batch_size * self.hidden_size];

// x_t × W_xf  (BLAS SGEMM)
unsafe {
    sgemm(
        Layout::RowMajor,
        Transpose::None, Transpose::None,
        batch_size as i32,      // M: number of rows in output
        self.hidden_size as i32, // N: number of cols in output
        self.input_size as i32,  // K: inner dimension
        1.0,
        input, self.input_size as i32,  // A matrix (input)
        &self.w_xf, self.hidden_size as i32, // B matrix (weights)
        0.0,
        &mut forget_gate, self.hidden_size as i32, // C matrix (output)
    );
}

// h_{t-1} × W_hf  (second SGEMM, accumulated into hf_contrib)
// ... (similar SGEMM call with hidden_batch and w_hf)

// Add bias and apply sigmoid element-wise
for i in 0..forget_gate.len() {
    let bias_idx = i % self.hidden_size;
    let pre_activation = forget_gate[i] + hf_contrib[i] + self.b_f[bias_idx];
    forget_gate[i] = 1.0 / (1.0 + (-pre_activation).exp()); // sigmoid
}
```

**Code snippet - Cell state update:**

```rust
// From src/layers/lstm/forward.rs

// Cell State Update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
let mut new_cell_state = vec![0.0f32; batch_size * self.hidden_size];
for b in 0..batch_size {
    for h in 0..self.hidden_size {
        let idx = b * self.hidden_size + h;
        new_cell_state[idx] =
            forget_gate[idx] * cell[h] + input_gate[idx] * cell_candidate[idx];
    }
}
```

**Code snippet - Hidden state and output:**

```rust
// From src/layers/lstm/forward.rs

// Hidden State: h_t = o_t ⊙ tanh(c_t)
let mut cell_tanh = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..new_cell_state.len() {
    cell_tanh[i] = new_cell_state[i].tanh();
}
let mut new_hidden_state = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..new_hidden_state.len() {
    new_hidden_state[i] = output_gate[i] * cell_tanh[i];
}

// Output Projection: y_t = h_t × W_hy + b_y
unsafe {
    sgemm(
        Layout::RowMajor,
        Transpose::None, Transpose::None,
        batch_size as i32, self.output_size as i32, self.hidden_size as i32,
        1.0,
        &new_hidden_state, self.hidden_size as i32,
        &self.w_hy, self.output_size as i32,
        0.0,
        output, self.output_size as i32,
    );
}
```

**Forward pass caching:**

The forward pass caches all gate values and intermediate states for use during backpropagation:

```rust
// From src/layers/lstm/mod.rs - cached fields
cached_h_prev: RefCell<Vec<f32>>,      // h_{t-1} before forward pass
cached_c_prev: RefCell<Vec<f32>>,      // c_{t-1} before forward pass
cached_forget_gate: RefCell<Vec<f32>>, // f_t after sigmoid
cached_input_gate: RefCell<Vec<f32>>,  // i_t after sigmoid
cached_cell_candidate: RefCell<Vec<f32>>, // c̃_t after tanh
cached_output_gate: RefCell<Vec<f32>>, // o_t after sigmoid
cached_cell_state: RefCell<Vec<f32>>,  // c_t after update
cached_cell_tanh: RefCell<Vec<f32>>,   // tanh(c_t)
```

### Computational Complexity

**Per time step:**

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| 4 × (x_t × W_x*) | 4 × 2·input_size·hidden_size | Input projections (BLAS) |
| 4 × (h_{t-1} × W_h*) | 4 × 2·hidden_size² | Recurrent projections (BLAS) |
| Gate activations | 4 × hidden_size | σ() and tanh() |
| Cell state update | 2 × hidden_size | Element-wise ops |
| Hidden state update | 2 × hidden_size | tanh + hadamard |
| Output projection | 2·hidden_size·output_size | BLAS |
| **Total** | **~4 × 2 × (input_size + hidden_size) × hidden_size** | Dominated by GEMMs |

**Comparison to vanilla RNN (same hidden_size):**
- LSTM: ~4× more FLOPs (due to 4 gates vs 1 recurrence)
- But LSTM learns what vanilla RNN cannot (long-range dependencies)

## Backward Pass - BPTT

BPTT for the LSTM applies the chain rule backwards through time. For a sequence of length T, gradients are computed from t=T back to t=1. The LSTM's BPTT is more complex than the vanilla RNN's but has **much better gradient properties** due to the cell state highway.

### BPTT Overview

**Total loss:**
```
L = Σₜ L⁽ᵗ⁾
```

**BPTT unfolding:**

```
        LSTM BPTT (Backward Direction)

Time:     T         T-1         T-2         ...    1

          ∂L/∂y^T   ∂L/∂y^{T-1}  ∂L/∂y^{T-2}
              │           │           │
              ▼           ▼           ▼
          [backward]  [backward]  [backward]
              │           │           │
  ◄── dh_T ──┤           │           │
  ◄── dc_T ──┤           │           │
              │ ◄── dh_{T-1} ─────────┤
              │ ◄── dc_{T-1} ─────────┤
                          │ ◄── dh_{T-2} ──...
                          │ ◄── dc_{T-2} ──...
```

**State gradient flow:**

At each time step t, the backward pass receives:
- `dh_next`: ∂L/∂h_t from the next time step's backward pass (hidden state gradient)
- `dc_next`: ∂L/∂c_t from the next time step's backward pass (cell state gradient)

And computes:
- `dh_prev`: ∂L/∂h_{t-1} to pass to the previous time step
- `dc_prev`: ∂L/∂c_{t-1} to pass to the previous time step

### Step 1: Output Projection Gradients

**Forward equation:** `y_t = h_t × W_hy + b_y`

**Given:** `∂L/∂y_t` (shape: batch_size × output_size)

**Weight gradient:**
```
∂L/∂W_hy = h_t^T × ∂L/∂y_t  / batch_size
           (hidden_size × batch_size) × (batch_size × output_size) → (hidden_size × output_size)
```

**Bias gradient:**
```
∂L/∂b_y = Σ_batch ∂L/∂y_t  / batch_size
          → (output_size,)
```

**Hidden state gradient from output:**
```
∂L/∂h_t|output = ∂L/∂y_t × W_hy^T
                 (batch_size × output_size) × (output_size × hidden_size) → (batch_size × hidden_size)
```

**Total hidden state gradient (including BPTT contribution):**
```
∂L/∂h_t = ∂L/∂h_t|output + dh_next    (add incoming gradient from next time step)
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// Gradient w.r.t. W_hy
let mut grad_w_hy = self.grad_w_hy.borrow_mut();
unsafe {
    sgemm(
        Layout::RowMajor,
        Transpose::Ordinary, Transpose::None,
        self.hidden_size as i32, self.output_size as i32, batch_size as i32,
        scale,                    // 1/batch_size for averaging
        &h_current_batch, self.hidden_size as i32,
        grad_output, self.output_size as i32,
        1.0,                      // accumulate into existing gradient
        &mut grad_w_hy, self.output_size as i32,
    );
}

// Add incoming BPTT hidden state gradient
for b in 0..batch_size {
    for h in 0..self.hidden_size {
        grad_h[b * self.hidden_size + h] += dh_next[h]; // broadcast over batch
    }
}
```

### Step 2: Hidden State Gradient

**Forward equation:** `h_t = o_t ⊙ tanh(c_t)`

**Given:** `∂L/∂h_t` (total hidden gradient including BPTT contribution)

**Output gate gradient:**
```
∂L/∂o_t = ∂L/∂h_t ⊙ tanh(c_t)
```

This uses ∂h_t/∂o_t = tanh(c_t) element-wise.

**Gradient through tanh(c_t):**
```
∂L/∂tanh(c_t) = ∂L/∂h_t ⊙ o_t
```

This uses ∂h_t/∂tanh(c_t) = o_t element-wise.

**Dimension check:** Both gradients are (batch_size × hidden_size)

### Step 3: Cell State Gradient

**Through tanh:**

The gradient of the loss with respect to c_t arrives via two paths:
1. Through `tanh(c_t)` in the hidden state computation
2. Directly as `dc_next` from the next time step

**Via tanh derivative:**
```
tanh'(c_t) = 1 - tanh²(c_t)
∂L/∂c_t|tanh = ∂L/∂tanh(c_t) ⊙ (1 - tanh²(c_t))
```

**Total cell state gradient (the gradient highway):**
```
∂L/∂c_t = ∂L/∂c_t|tanh + dc_next    (add incoming gradient from next time step)
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// Backprop through tanh: tanh'(c_t) = 1 - tanh(c_t)^2
let mut grad_cell_state = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..grad_cell_state.len() {
    let tanh_val = cell_tanh_batch[i];
    grad_cell_state[i] = grad_cell_tanh[i] * (1.0 - tanh_val * tanh_val);
}

// Add incoming cell state gradient from next time step (BPTT contribution)
for b in 0..batch_size {
    for h in 0..self.hidden_size {
        grad_cell_state[b * self.hidden_size + h] += dc_next[h];
    }
}
```

### Step 4: Output Gate Gradients

**Forward equation:** `o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)`

**Given:** `∂L/∂o_t` from Step 2

**Sigmoid derivative:**
```
σ'(z) = σ(z) · (1 - σ(z)) = o_t · (1 - o_t)
```

**Pre-activation gradient (gradient before sigmoid):**
```
∂L/∂z_o = ∂L/∂o_t ⊙ o_t ⊙ (1 - o_t)       (shape: batch_size × hidden_size)
```

**Weight gradients:**
```
∂L/∂W_xo = x_t^T × ∂L/∂z_o  / batch_size
            (input_size × batch_size) × (batch_size × hidden_size) → (input_size × hidden_size)

∂L/∂W_ho = h_{t-1}^T × ∂L/∂z_o  / batch_size
            (hidden_size × batch_size) × (batch_size × hidden_size) → (hidden_size × hidden_size)

∂L/∂b_o = Σ_batch ∂L/∂z_o  / batch_size    → (hidden_size,)
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// Backprop through sigmoid: σ'(z) = σ(z) * (1 - σ(z))
let mut grad_output_gate_pre = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..grad_output_gate_pre.len() {
    let o_val = output_gate_batch[i];
    grad_output_gate_pre[i] = grad_output_gate[i] * o_val * (1.0 - o_val);
}

// Gradient w.r.t. W_xo: x_t^T × ∂L/∂z_o
{
    let mut grad_w_xo = self.grad_w_xo.borrow_mut();
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::Ordinary, Transpose::None, // x_t transposed
            self.input_size as i32, self.hidden_size as i32, batch_size as i32,
            scale, input, self.input_size as i32,
            &grad_output_gate_pre, self.hidden_size as i32,
            1.0, &mut grad_w_xo, self.hidden_size as i32,
        );
    }
}
```

### Step 5: Cell State Decomposition

**Forward equation:** `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`

**Given:** `∂L/∂c_t` (total cell gradient from Step 3)

**Gradient with respect to each term:**

```
∂c_t/∂f_t = c_{t-1}        →   ∂L/∂f_t = ∂L/∂c_t ⊙ c_{t-1}
∂c_t/∂i_t = c̃_t            →   ∂L/∂i_t = ∂L/∂c_t ⊙ c̃_t
∂c_t/∂c̃_t = i_t            →   ∂L/∂c̃_t = ∂L/∂c_t ⊙ i_t
∂c_t/∂c_{t-1} = f_t        →   ∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t  ← (cell state highway!)
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// Gradient w.r.t. f_t: grad_f = grad_c_t ⊙ c_{t-1}
let mut grad_forget_gate = vec![0.0f32; batch_size * self.hidden_size];
for b in 0..batch_size {
    for h in 0..self.hidden_size {
        let idx = b * self.hidden_size + h;
        grad_forget_gate[idx] = grad_cell_state[idx] * cached_c_prev[h];
    }
}

// Gradient w.r.t. i_t: grad_i = grad_c_t ⊙ c̃_t
let mut grad_input_gate = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..grad_input_gate.len() {
    grad_input_gate[i] = grad_cell_state[i] * cell_candidate_batch[i];
}

// Gradient w.r.t. c̃_t: grad_c_tilde = grad_c_t ⊙ i_t
let mut grad_cell_candidate = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..grad_cell_candidate.len() {
    grad_cell_candidate[i] = grad_cell_state[i] * input_gate_batch[i];
}
```

### Step 6: Forget Gate Gradients

**Forward equation:** `f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)`

**Given:** `∂L/∂f_t` from Step 5

**Pre-activation gradient:**
```
∂L/∂z_f = ∂L/∂f_t ⊙ f_t ⊙ (1 - f_t)      (sigmoid derivative)
```

**Weight gradients:**
```
∂L/∂W_xf = x_t^T × ∂L/∂z_f  / batch_size   → (input_size × hidden_size)
∂L/∂W_hf = h_{t-1}^T × ∂L/∂z_f / batch_size → (hidden_size × hidden_size)
∂L/∂b_f = Σ_batch ∂L/∂z_f / batch_size       → (hidden_size,)
```

### Step 7: Input Gate Gradients

**Forward equation:** `i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)`

**Given:** `∂L/∂i_t` from Step 5

**Pre-activation gradient:**
```
∂L/∂z_i = ∂L/∂i_t ⊙ i_t ⊙ (1 - i_t)      (sigmoid derivative)
```

**Weight gradients:**
```
∂L/∂W_xi = x_t^T × ∂L/∂z_i  / batch_size   → (input_size × hidden_size)
∂L/∂W_hi = h_{t-1}^T × ∂L/∂z_i / batch_size → (hidden_size × hidden_size)
∂L/∂b_i = Σ_batch ∂L/∂z_i / batch_size       → (hidden_size,)
```

### Step 8: Cell Candidate Gradients

**Forward equation:** `c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)`

**Given:** `∂L/∂c̃_t` from Step 5

**Tanh derivative:**
```
tanh'(z) = 1 - tanh²(z) = 1 - c̃_t²
```

**Pre-activation gradient:**
```
∂L/∂z_c = ∂L/∂c̃_t ⊙ (1 - c̃_t²)           (tanh derivative)
```

**Weight gradients:**
```
∂L/∂W_xc = x_t^T × ∂L/∂z_c  / batch_size   → (input_size × hidden_size)
∂L/∂W_hc = h_{t-1}^T × ∂L/∂z_c / batch_size → (hidden_size × hidden_size)
∂L/∂b_c = Σ_batch ∂L/∂z_c / batch_size       → (hidden_size,)
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// Backprop through tanh: tanh'(z) = 1 - tanh²(z) = 1 - c̃_t²
let mut grad_cell_candidate_pre = vec![0.0f32; batch_size * self.hidden_size];
for i in 0..grad_cell_candidate_pre.len() {
    let c_tilde = cell_candidate_batch[i]; // cached c̃_t value (already tanh-activated)
    grad_cell_candidate_pre[i] = grad_cell_candidate[i] * (1.0 - c_tilde * c_tilde);
}
```

### Step 9: Input Gradient

The gradient with respect to x_t accumulates contributions from all four gates (since x_t feeds into all gate pre-activations):

```
∂L/∂x_t = ∂L/∂z_f × W_xf^T     (from forget gate)
         + ∂L/∂z_i × W_xi^T     (from input gate)
         + ∂L/∂z_c × W_xc^T     (from cell candidate)
         + ∂L/∂z_o × W_xo^T     (from output gate)
```

**Dimension check:**
```
Each term: (batch_size × hidden_size) × (hidden_size × input_size) → (batch_size × input_size)
∂L/∂x_t: (batch_size × input_size) — sum of 4 contributions
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs - gradient w.r.t. input accumulation

grad_input.fill(0.0);

// Forget gate contribution: grad_input += grad_forget_gate_pre × W_xf^T
unsafe { sgemm(/* grad_forget_gate_pre × W_xf^T, beta=1.0 to accumulate */); }

// Input gate contribution: grad_input += grad_input_gate_pre × W_xi^T
unsafe { sgemm(/* grad_input_gate_pre × W_xi^T, beta=1.0 to accumulate */); }

// Cell candidate contribution: grad_input += grad_cell_candidate_pre × W_xc^T
unsafe { sgemm(/* grad_cell_candidate_pre × W_xc^T, beta=1.0 to accumulate */); }

// Output gate contribution: grad_input += grad_output_gate_pre × W_xo^T
unsafe { sgemm(/* grad_output_gate_pre × W_xo^T, beta=1.0 to accumulate */); }
```

### Step 10: Previous State Gradients

The BPTT algorithm requires gradients with respect to the **previous time step's states** (h_{t-1}, c_{t-1}) to continue propagating backwards:

**Cell state gradient (simple — the highway!):**
```
∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t              (averaged over batch)
```

This is just elementwise multiplication by the forget gate — no matrix multiply, no squashing!

**Hidden state gradient (more complex — through all four gates):**
```
∂L/∂h_{t-1} = ∂L/∂z_f × W_hf^T
             + ∂L/∂z_i × W_hi^T
             + ∂L/∂z_c × W_hc^T
             + ∂L/∂z_o × W_ho^T
```

**Code snippet:**

```rust
// From src/layers/lstm/backward.rs

// grad_c_prev = ∂L/∂c_{t-1} = grad_cell_state ⊙ f_t (averaged over batch)
{
    let mut grad_c_prev_out = self.grad_c_prev.borrow_mut();
    grad_c_prev_out.fill(0.0);
    for b in 0..batch_size {
        for h in 0..self.hidden_size {
            let idx = b * self.hidden_size + h;
            grad_c_prev_out[h] += grad_cell_state[idx] * forget_gate_batch[idx];
        }
    }
    for h in 0..self.hidden_size { grad_c_prev_out[h] *= scale; }
}

// grad_h_prev = contributions from all four gate pre-activations via W_h matrices
// Accumulated using 4 SGEMM calls with beta=1.0 to add contributions:
//   += grad_forget_gate_pre × W_hf^T
//   += grad_input_gate_pre  × W_hi^T
//   += grad_cell_candidate_pre × W_hc^T
//   += grad_output_gate_pre × W_ho^T
```

**Usage pattern for full BPTT over a sequence:**

```rust
// From src/layers/lstm/mod.rs - backward_bptt usage pattern

// Forward pass through all time steps first
layer.reset_state();
for t in 0..seq_len {
    layer.forward(&inputs[t], &mut outputs[t], batch_size);
}

// Backward pass in reverse order with state gradient propagation
let mut dh = vec![0.0f32; hidden_size]; // zero at last time step
let mut dc = vec![0.0f32; hidden_size]; // zero at last time step
for t in (0..seq_len).rev() {
    let mut grad_in = vec![0.0; input_size];
    (dh, dc) = layer.backward_bptt(
        &inputs[t], &grad_outputs[t], &mut grad_in, &dh, &dc, batch_size
    );
}
layer.update_parameters(learning_rate);
```

## Cell State Gradient Highway

### Mathematical Analysis

The cell state gradient highway is the fundamental reason LSTMs can learn long-range dependencies. To understand it, compare the gradient flow through time.

**Vanilla RNN gradient flow from time t to t-k:**

```
∂h_t / ∂h_{t-k} = Π_{i=t-k+1}^{t} ∂h_i/∂h_{i-1}
                 = Π_{i=t-k+1}^{t} diag(tanh'(z_i)) × W_h
```

This is a product of k matrices, each the Jacobian of the recurrent step. The spectral radius of these Jacobians determines whether gradients vanish or explode:
- If max eigenvalue < 1: gradients vanish exponentially with k → network cannot learn dependencies > ~10 steps
- If max eigenvalue > 1: gradients explode → training unstable

**LSTM cell state gradient flow from time t to t-k:**

```
∂c_t / ∂c_{t-k} = Π_{i=t-k+1}^{t} ∂c_i/∂c_{i-1}
                 = Π_{i=t-k+1}^{t} diag(f_i)
                 = diag(f_t ⊙ f_{t-1} ⊙ ... ⊙ f_{t-k+1})
```

This is a product of k **diagonal matrices** (just the forget gate values)! The gradient is simply element-wise multiplication by the product of forget gates across time steps.

**Key properties:**
1. **Element-wise**: No matrix-matrix product — no cross-dimensional interference
2. **Learnable**: The forget gates learn to set values close to 1.0 when gradients should flow
3. **Selective**: Different cell state dimensions can have different gradient flows
4. **Bounded**: Forget gate values are in (0, 1), so gradients can be preserved but not explode

### Why This Solves Vanishing Gradients

**Intuition:**

Consider a sequence where the model needs to remember an event from 100 time steps ago. In a vanilla RNN, the gradient must flow back through 100 matrix multiplications, each involving tanh derivatives (which saturate near ±0). This gradient is essentially zero.

In an LSTM, the gradient flows back through 100 **forget gate multiplications**. The LSTM can learn to set these forget gates close to 1.0 (remembering), allowing the gradient to flow essentially unimpeded.

**Mathematical condition for gradient preservation:**

```
For cell gradient to survive k steps:
  |Π_{i} f_i,j| ≈ 1.0   for relevant dimensions j
  ⟺ f_i,j ≈ 1.0 for all i   (network learns to "keep the gate open")
```

When the network needs to carry information for many steps, it learns forget gates close to 1.0 for the relevant dimensions, creating a near-lossless gradient highway.

**Visualization:**

```
                GRADIENT HIGHWAY COMPARISON

Vanilla RNN (k=5 steps):
∂L/∂h_{t-5} ∝ W_h^T × diag(d_5) × W_h^T × diag(d_4) × ... × diag(d_1)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              Product of 5 weight matrices — high probability of vanishing!

LSTM Cell State (k=5 steps):
∂L/∂c_{t-5} ∝ f_t ⊙ f_{t-1} ⊙ f_{t-2} ⊙ f_{t-3} ⊙ f_{t-4} ⊙ ∂L/∂c_t
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              Elementwise product of forget gates — no matrix issues!
              If f_i ≈ 1.0: gradient preserved perfectly!
```

### Comparison with RNN Gradient Flow

| Property | Vanilla RNN | LSTM |
|----------|-------------|------|
| Gradient path type | Matrix products | Element-wise products |
| Gradient magnitude control | Uncontrolled | Learned via forget gates |
| Saturation risk | tanh'(z) → 0 at extremes | σ'(z) → 0 but only affects hidden state |
| Memory span (practical) | ~5-15 steps | 100s-1000s of steps |
| Cell state highway | None | Yes — through f_t ⊙ c_{t-1} |
| Parameters per "unit" | 3 matrices | 8 matrices (4× input, 4× hidden) |

## LSTM vs Vanilla RNN

### Architecture Differences

**Vanilla RNN (single recurrence):**
```
h_t = tanh(x_t × W_x + h_{t-1} × W_h + b_h)
y_t = h_t × W_y + b_y
```

**LSTM (gated architecture with cell state):**
```
f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)   ← forget gate
i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)   ← input gate
c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c) ← cell candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t               ← cell state (highway)
o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)   ← output gate
h_t = o_t ⊙ tanh(c_t)                          ← hidden state
y_t = h_t × W_hy + b_y                         ← output
```

### Gradient Flow Comparison

**Vanilla RNN — single gradient path:**
```
∂L/∂h_{t-1} = ∂L/∂h_t × ∂h_t/∂h_{t-1}
             = ∂L/∂h_t × diag(1 - h_t²) × W_h^T    (one squashed matrix path)
```

**LSTM — two gradient paths:**

1. **Hidden state path** (through all gates, similar to RNN but 4× paths):
```
∂L/∂h_{t-1} = ∂L/∂z_f × W_hf^T + ∂L/∂z_i × W_hi^T
             + ∂L/∂z_c × W_hc^T + ∂L/∂z_o × W_ho^T
```

2. **Cell state path** (the highway — minimal squashing):
```
∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t    (pure elementwise — the gradient highway!)
```

### Parameter Count

For the same `input_size` (n) and `hidden_size` (h) and `output_size` (m):

**Vanilla RNN:**
```
W_x: n × h
W_h: h × h
W_y: h × m
b_h: h
b_y: m
Total: n·h + h² + h·m + h + m
```

**LSTM (this implementation):**
```
4 gates × (W_x: n×h + W_h: h×h + b: h) + W_hy: h×m + b_y: m
= 4(n·h + h² + h) + h·m + m
Total: 4n·h + 4h² + 4h + h·m + m
```

**LSTM is approximately 4× more parameters** for the recurrent connections. This is the cost of the gating mechanism, but it enables learning that vanilla RNNs simply cannot do.

**Example (input=128, hidden=256, output=10):**
- Vanilla RNN: 128×256 + 256×256 + 256×10 + 256 + 10 = 101,130
- LSTM: 4×(128×256 + 256×256 + 256) + 256×10 + 10 = 396,066

### When to Use Each

**Use vanilla RNN when:**
- Sequences are short (< 15 time steps)
- Computational resources are very limited
- Task is simple (e.g., single-step prediction)
- Educational/baseline purposes

**Use LSTM when:**
- Sequences are long (> 15 time steps)
- Task requires memory of distant events (e.g., language modeling)
- Quality matters more than parameter efficiency
- Training on GPU where extra computation is cheap

**Consider GRU (Gated Recurrent Unit) as a middle ground:**
- Similar long-range learning to LSTM
- ~1.5× parameters of vanilla RNN (vs 4× for LSTM)
- Combines cell and hidden state into one vector

## Parameter Updates

After accumulating gradients through forward and backward passes, parameters are updated using gradient descent:

```
W_xf ← W_xf - α · ∂L/∂W_xf
W_hf ← W_hf - α · ∂L/∂W_hf
b_f  ← b_f  - α · ∂L/∂b_f
... (repeated for all 14 parameter groups)
```

**Implementation:**

```rust
// From src/layers/lstm/backward.rs - update_parameters_impl

fn update_parameters_impl(&mut self, learning_rate: f32) {
    // Forget gate weights
    self.grad_w_xf.apply_sgd_update(&mut self.w_xf, learning_rate);
    self.grad_w_hf.apply_sgd_update(&mut self.w_hf, learning_rate);
    self.grad_b_f.apply_sgd_update(&mut self.b_f, learning_rate);

    // Input gate weights
    self.grad_w_xi.apply_sgd_update(&mut self.w_xi, learning_rate);
    self.grad_w_hi.apply_sgd_update(&mut self.w_hi, learning_rate);
    self.grad_b_i.apply_sgd_update(&mut self.b_i, learning_rate);

    // Cell candidate weights
    self.grad_w_xc.apply_sgd_update(&mut self.w_xc, learning_rate);
    self.grad_w_hc.apply_sgd_update(&mut self.w_hc, learning_rate);
    self.grad_b_c.apply_sgd_update(&mut self.b_c, learning_rate);

    // Output gate weights
    self.grad_w_xo.apply_sgd_update(&mut self.w_xo, learning_rate);
    self.grad_w_ho.apply_sgd_update(&mut self.w_ho, learning_rate);
    self.grad_b_o.apply_sgd_update(&mut self.b_o, learning_rate);

    // Output projection
    self.grad_w_hy.apply_sgd_update(&mut self.w_hy, learning_rate);
    self.grad_b_y.apply_sgd_update(&mut self.b_y, learning_rate);
}
```

The `GradientAccumulator::apply_sgd_update` applies `param -= lr * grad` and clears the accumulator.

## Initialization

This implementation uses **Xavier (Glorot) uniform initialization** for all weight matrices:

```
limit_gate = sqrt(6.0 / (input_size + hidden_size))     for input weights
limit_rec  = sqrt(6.0 / (hidden_size + hidden_size))    for recurrent weights
limit_out  = sqrt(6.0 / (hidden_size + output_size))    for output weights

W ~ Uniform(-limit, limit)
```

**Code snippet:**

```rust
// From src/layers/lstm/mod.rs - new()

let init_weights = |size, fan_in, fan_out, rng: &mut SimpleRng| -> Vec<f32> {
    let mut weights = vec![0.0f32; size];
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt(); // Xavier limit
    for value in &mut weights {
        *value = rng.gen_range_f32(-limit, limit);
    }
    weights
};

// Forget gate: Xavier with fan_in=input_size, fan_out=hidden_size
let w_xf = init_weights(input_size * hidden_size, input_size, hidden_size, rng);
// Recurrent: Xavier with fan_in=fan_out=hidden_size
let w_hf = init_weights(hidden_size * hidden_size, hidden_size, hidden_size, rng);
```

**Biases are initialized to zero.**

**Why Xavier initialization?**

Xavier initialization ensures that the variance of activations and gradients is roughly preserved across layers at initialization. For sigmoid-gated networks like LSTMs:
- Too-large initial weights → gates saturate at 0 or 1 → vanishing gradients
- Too-small initial weights → activations near 0 → slow learning
- Xavier targets variance = 2/(fan_in + fan_out), which keeps sigmoid inputs in the linear region

**Special consideration — forget gate bias initialization:**

Research (Jozefowicz et al., 2015) suggests initializing the forget gate **bias to 1.0** (not 0.0) to encourage the LSTM to remember by default at initialization. This can help with long-range dependencies in early training. This implementation initializes all biases to 0.0 (standard default), but the architecture supports changing this.

## Numerical Considerations

### Sigmoid Saturation

**Problem:** When `|z| >> 0`, sigmoid saturates:
- σ(z) → 0 if z → -∞, gradient → 0
- σ(z) → 1 if z → +∞, gradient → 0

**Effect on training:** Gates can get "stuck" in fully-open or fully-closed positions, making learning slow or impossible for those dimensions.

**Mitigation:**
- Xavier initialization keeps initial gate inputs in (-3, 3) range where sigmoid is more linear
- Gradient clipping prevents large weights from developing

### Tanh Saturation

**Problem:** When `|z| >> 0`, tanh saturates similarly to sigmoid (but symmetric):
- tanh(z) → ±1 at extremes, derivative → 0
- Cell candidate c̃_t and the tanh(c_t) in hidden state computation can both saturate

**Effect:** If c_t grows very large in magnitude, tanh(c_t) → ±1 and its gradient → 0, breaking the connection between the cell state and the hidden state.

**Mitigation:**
- The forget gate naturally limits cell state growth (it multiplies by values in (0,1))
- Proper learning rate prevents weights from growing too large

### Gradient Clipping

**Problem:** Even with the cell state highway, the **hidden state gradient path** (through W_h matrices) can still explode for deep networks or very long sequences with large weights.

**Recommended approach:**

```rust
// Clip gradient norm before parameter update
let grad_norm = compute_gradient_norm(&all_gradients);
let clip_value = 1.0; // typical value
if grad_norm > clip_value {
    let scale = clip_value / grad_norm;
    scale_all_gradients(scale);
}
```

This implementation does not include gradient clipping in the LSTM backward pass itself; it should be applied at the training loop level.

### Numerical Stability of Sigmoid

The naive sigmoid implementation `1 / (1 + exp(-z))` can overflow for very negative z values. This implementation uses Rust's `f32::exp()`, which returns `f32::INFINITY` for large values, making sigmoid return approximately 0 (which is correct). For extreme negative values, exp(-z) → infinity, but the division keeps the result bounded in [0, 1].

**Code:**

```rust
// Numerically safe (Rust handles f32 overflow correctly for sigmoid)
forget_gate[i] = 1.0 / (1.0 + (-pre_activation).exp());
```

### Precision Considerations

This implementation uses `f32` (single-precision) throughout. For very deep recurrence (T > 1000), the accumulated rounding errors in f32 arithmetic could affect training stability. Consider:
- Using `f64` for gradient accumulation if training very long sequences
- Reducing sequence length or using truncated BPTT for very long sequences

### Truncated BPTT

For very long sequences, computing gradients back to the beginning of the sequence (full BPTT) is computationally expensive and may not be necessary. **Truncated BPTT** processes the sequence in segments of length k, stopping gradient propagation at segment boundaries:

```
For segments of length k (e.g., k=50):
  For each segment [t_start, t_start + k]:
    - Forward pass through segment, keeping states
    - Backward pass through segment only
    - Zero out state gradients at segment boundary
    - Continue with next segment (states are NOT zeroed)
```

**Trade-off:** Gradients cannot flow past segment boundaries, limiting learning of dependencies > k steps. But computation is O(k) per segment instead of O(T).

The `backward_bptt` method in this implementation supports truncated BPTT by accepting `dh_next` and `dc_next` — simply pass zero vectors to truncate gradient flow at any point.

---

## Summary

The LSTM's seven equations implement a sophisticated memory management system:

| Equation | Formula | Role |
|----------|---------|------|
| 1. Forget gate | f_t = σ(x_t W_xf + h_{t-1} W_hf + b_f) | What to discard from memory |
| 2. Input gate | i_t = σ(x_t W_xi + h_{t-1} W_hi + b_i) | How much to write to memory |
| 3. Cell candidate | c̃_t = tanh(x_t W_xc + h_{t-1} W_hc + b_c) | What new information to write |
| 4. Cell state | c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t | **The gradient highway** |
| 5. Output gate | o_t = σ(x_t W_xo + h_{t-1} W_ho + b_o) | What to read from memory |
| 6. Hidden state | h_t = o_t ⊙ tanh(c_t) | Working memory for output |
| 7. Output | y_t = h_t W_hy + b_y | Task-specific prediction |

The BPTT algorithm propagates gradients backwards through all seven operations, with the key insight that gradients flow through the cell state with **only elementwise multiplication** by the forget gates — avoiding the repeated matrix squashing that causes vanishing gradients in vanilla RNNs.

**Source files:**
- `src/layers/lstm/mod.rs` — Layer structure, initialization, state management
- `src/layers/lstm/forward.rs` — Forward pass (7 equations, BLAS-accelerated)
- `src/layers/lstm/backward.rs` — Backward pass / BPTT (10-step gradient derivation)
