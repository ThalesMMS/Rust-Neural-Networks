# RNN Backpropagation Through Time (BPTT)

This document provides a comprehensive explanation of the mathematics behind Recurrent Neural Networks (RNNs), with detailed derivations of Backpropagation Through Time (BPTT), the algorithm used to train RNNs on sequential data.

## Table of Contents

- [Overview](#overview)
- [Forward Pass Through Time](#forward-pass-through-time)
  - [Mathematical Definition](#mathematical-definition)
  - [Unfolding Through Time](#unfolding-through-time)
  - [Dimension Analysis](#dimension-analysis)
  - [Computational Graph](#computational-graph)
  - [Implementation Details](#implementation-details)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass - BPTT](#backward-pass---bptt)
  - [BPTT Overview](#bptt-overview)
  - [Chain Rule Through Time](#chain-rule-through-time)
  - [Gradient Derivations](#gradient-derivations)
  - [Gradient Flow Equations](#gradient-flow-equations)
  - [Implementation Strategy](#implementation-strategy)
- [The Vanishing/Exploding Gradient Problem](#the-vanishingexploding-gradient-problem)
  - [Mathematical Analysis](#mathematical-analysis)
  - [Gradient Clipping](#gradient-clipping)
  - [Other Mitigation Strategies](#other-mitigation-strategies)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Truncated BPTT](#truncated-bptt)
- [Numerical Considerations](#numerical-considerations)

## Overview

A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data by maintaining a hidden state that is updated at each time step. Unlike feedforward networks, RNNs have recurrent connections that allow information to persist across time steps.

**Key characteristics:**
- **Temporal processing**: Processes sequences one element at a time
- **Hidden state**: Maintains memory of previous inputs
- **Parameter sharing**: Same weights applied at every time step
- **Variable-length sequences**: Can handle inputs of different lengths

**Parameters:**
- **Wₓ (input weights)**: Maps input to hidden state
- **Wₕ (recurrent weights)**: Maps previous hidden state to current hidden state
- **Wᵧ (output weights)**: Maps hidden state to output
- **bₕ (hidden bias)**: Bias for hidden state computation
- **bᵧ (output bias)**: Bias for output computation

**Use cases:**
- Language modeling (character or word level)
- Time series prediction
- Speech recognition
- Machine translation
- Any task involving sequential or temporal dependencies

## Forward Pass Through Time

### Mathematical Definition

The forward pass of a vanilla RNN processes a sequence of inputs {x⁽¹⁾, x⁽²⁾, ..., x⁽ᵀ⁾} and produces:
- Hidden states: {h⁽¹⁾, h⁽²⁾, ..., h⁽ᵀ⁾}
- Outputs: {y⁽¹⁾, y⁽²⁾, ..., y⁽ᵀ⁾}

**Recurrence relation:**

At each time step t:

```
h⁽ᵗ⁾ = tanh(Wₓ · x⁽ᵗ⁾ + Wₕ · h⁽ᵗ⁻¹⁾ + bₕ)
y⁽ᵗ⁾ = Wᵧ · h⁽ᵗ⁾ + bᵧ
```

Where:
- **x⁽ᵗ⁾**: Input at time step t, shape (input_size,)
- **h⁽ᵗ⁾**: Hidden state at time step t, shape (hidden_size,)
- **h⁽ᵗ⁻¹⁾**: Previous hidden state, shape (hidden_size,)
- **y⁽ᵗ⁾**: Output at time step t, shape (output_size,)
- **tanh**: Hyperbolic tangent activation function

**Initial condition:**
```
h⁽⁰⁾ = 0  (or learned initialization)
```

**Expanded notation for hidden state:**

For hidden dimension j at time step t:

```
h_j⁽ᵗ⁾ = tanh(Σᵢ Wₓ[j,i] · x_i⁽ᵗ⁾ + Σₖ Wₕ[j,k] · h_k⁽ᵗ⁻¹⁾ + bₕ[j])
```

**Expanded notation for output:**

For output dimension k at time step t:

```
y_k⁽ᵗ⁾ = Σⱼ Wᵧ[k,j] · h_j⁽ᵗ⁾ + bᵧ[k]
```

### Unfolding Through Time

The key insight of RNNs is that the same computation is applied at each time step. To understand BPTT, we "unfold" the RNN through time into a feedforward network:

```
                    UNFOLDED RNN THROUGH TIME

Time:        t=0          t=1          t=2          t=3

Input:                  x⁽¹⁾         x⁽²⁾         x⁽³⁾
                         │            │            │
                         │            │            │
                         ▼            ▼            ▼
             h⁽⁰⁾──────►[RNN]───────►[RNN]───────►[RNN]
             (init)      │ │          │ │          │ │
                         │ │          │ │          │ │
                         │ h⁽¹⁾       │ h⁽²⁾       │ h⁽³⁾
                         │            │            │
                         ▼            ▼            ▼
                        y⁽¹⁾         y⁽²⁾         y⁽³⁾
                         │            │            │
                         ▼            ▼            ▼
                        L⁽¹⁾         L⁽²⁾         L⁽³⁾
                        (loss)      (loss)      (loss)
```

**Key observations:**
1. The same parameters (Wₓ, Wₕ, Wᵧ, bₕ, bᵧ) are used at every time step
2. Hidden state h⁽ᵗ⁾ depends on all previous inputs {x⁽¹⁾, ..., x⁽ᵗ⁾}
3. Each time step can have its own loss L⁽ᵗ⁾
4. Total loss is typically the sum: L = Σₜ L⁽ᵗ⁾

**Dependency chain:**

The hidden state at time t depends on all previous inputs:

```
h⁽ᵗ⁾ = f(x⁽ᵗ⁾, h⁽ᵗ⁻¹⁾)
     = f(x⁽ᵗ⁾, f(x⁽ᵗ⁻¹⁾, h⁽ᵗ⁻²⁾))
     = f(x⁽ᵗ⁾, f(x⁽ᵗ⁻¹⁾, f(x⁽ᵗ⁻²⁾, ..., f(x⁽¹⁾, h⁽⁰⁾)...)))
```

This recursive dependency is why backpropagation must flow backward through time.

### Dimension Analysis

**Parameter dimensions:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| Wₓ | (hidden_size, input_size) | Input-to-hidden weights |
| Wₕ | (hidden_size, hidden_size) | Hidden-to-hidden (recurrent) weights |
| Wᵧ | (output_size, hidden_size) | Hidden-to-output weights |
| bₕ | (hidden_size,) | Hidden bias |
| bᵧ | (output_size,) | Output bias |

**State dimensions at time step t:**

| Variable | Shape | Description |
|----------|-------|-------------|
| x⁽ᵗ⁾ | (input_size,) | Input vector |
| h⁽ᵗ⁾ | (hidden_size,) | Hidden state vector |
| y⁽ᵗ⁾ | (output_size,) | Output vector |

**Batched computation:**

For a batch of B sequences, each of length T:

| Variable | Shape | Description |
|----------|-------|-------------|
| X | (B, T, input_size) | Batch of input sequences |
| H | (B, T, hidden_size) | Batch of hidden state sequences |
| Y | (B, T, output_size) | Batch of output sequences |

**Operation breakdown for single time step:**

```
Step 1: Compute input contribution
  Wₓ @ x⁽ᵗ⁾ → (hidden_size, input_size) @ (input_size,) = (hidden_size,)

Step 2: Compute recurrent contribution
  Wₕ @ h⁽ᵗ⁻¹⁾ → (hidden_size, hidden_size) @ (hidden_size,) = (hidden_size,)

Step 3: Add bias and apply activation
  z⁽ᵗ⁾ = Wₓ @ x⁽ᵗ⁾ + Wₕ @ h⁽ᵗ⁻¹⁾ + bₕ  → (hidden_size,)
  h⁽ᵗ⁾ = tanh(z⁽ᵗ⁾) → (hidden_size,)

Step 4: Compute output
  y⁽ᵗ⁾ = Wᵧ @ h⁽ᵗ⁾ + bᵧ → (output_size,)
```

### Computational Graph

The computational graph for a single time step shows the dependencies:

```
        x⁽ᵗ⁾                h⁽ᵗ⁻¹⁾
         │                    │
         │                    │
    ┌────▼────┐         ┌────▼────┐
    │ Wₓ @ x  │         │ Wₕ @ h  │
    └────┬────┘         └────┬────┘
         │                    │
         └──────┬─────────────┘
                │
           ┌────▼────┐
           │  + bₕ   │
           └────┬────┘
                │
         ┌──────▼─────┐
         │ z⁽ᵗ⁾       │ (pre-activation)
         └──────┬─────┘
                │
           ┌────▼────┐
           │  tanh   │
           └────┬────┘
                │
         ┌──────▼─────┐
         │ h⁽ᵗ⁾       │ (hidden state)
         └──────┬─────┘
                │
           ┌────▼────┐
           │ Wᵧ @ h  │
           └────┬────┘
                │
           ┌────▼────┐
           │  + bᵧ   │
           └────┬────┘
                │
         ┌──────▼─────┐
         │ y⁽ᵗ⁾       │ (output)
         └──────┬─────┘
                │
           ┌────▼────┐
           │  Loss   │
           └─────────┘
```

### Implementation Details

**Forward pass pseudocode:**

```python
def forward(inputs, h_prev):
    """
    inputs: (T, input_size) - sequence of inputs
    h_prev: (hidden_size,) - initial hidden state
    """
    T = len(inputs)
    hidden_states = []
    outputs = []

    h = h_prev
    for t in range(T):
        # Compute pre-activation
        z = Wx @ inputs[t] + Wh @ h + bh

        # Apply activation
        h = tanh(z)
        hidden_states.append(h)

        # Compute output
        y = Wy @ h + by
        outputs.append(y)

    return outputs, hidden_states
```

**Key implementation notes:**

1. **State storage**: All hidden states must be stored for backward pass
2. **Pre-activation storage**: Store z⁽ᵗ⁾ = Wₓx⁽ᵗ⁾ + Wₕh⁽ᵗ⁻¹⁾ + bₕ for gradient computation
3. **Sequential dependency**: Cannot parallelize across time (but can across batch)
4. **Memory**: O(T × hidden_size) memory required for hidden states

### Computational Complexity

**Time Complexity per time step:**

```
Forward pass: O(hidden_size × input_size + hidden_size² + hidden_size × output_size)
            ≈ O(hidden_size²) when hidden_size dominates
```

**Total for sequence of length T:**

```
O(T × hidden_size²)
```

**Space Complexity:**

```
Parameters: O(hidden_size × input_size + hidden_size² + hidden_size × output_size)
Activations: O(T × hidden_size)  [must store all hidden states]
```

**Example calculation:**

For a character-level language model with:
- input_size = 128 (character vocabulary)
- hidden_size = 512
- output_size = 128
- sequence_length = 100

```
Operations per time step:
  Wₓ @ x: 512 × 128 = 65,536 ops
  Wₕ @ h: 512 × 512 = 262,144 ops
  Wᵧ @ h: 128 × 512 = 65,536 ops
  Total: ≈ 393K ops per time step
  Total for sequence: ≈ 39.3M ops

Parameters:
  Wₓ: 512 × 128 = 65,536
  Wₕ: 512 × 512 = 262,144
  Wᵧ: 128 × 512 = 65,536
  Biases: 512 + 128 = 640
  Total: ≈ 394K parameters

Memory for hidden states:
  100 × 512 × 4 bytes = 200 KB
```

## Backward Pass - BPTT

### BPTT Overview

**Backpropagation Through Time (BPTT)** is the training algorithm for RNNs. It is simply the application of backpropagation to the unfolded computational graph of an RNN.

**Key differences from standard backpropagation:**

1. **Temporal dependencies**: Gradients flow backward through time
2. **Parameter sharing**: Gradients for shared parameters are accumulated across all time steps
3. **Long-range dependencies**: Gradients can vanish or explode over many time steps
4. **Memory requirements**: Must store all intermediate states

**High-level algorithm:**

```
1. Forward pass: Compute all h⁽ᵗ⁾ and y⁽ᵗ⁾ for t = 1, ..., T
2. Compute loss at each time step: L⁽ᵗ⁾
3. Backward pass: For t = T, T-1, ..., 1:
   a. Compute output gradient: ∂L/∂y⁽ᵗ⁾
   b. Backprop through output layer: ∂L/∂h⁽ᵗ⁾
   c. Backprop through time: add gradient from future time step
   d. Backprop through activation: ∂L/∂z⁽ᵗ⁾
   e. Accumulate parameter gradients
4. Update parameters using accumulated gradients
```

### Chain Rule Through Time

The complexity of BPTT arises from the fact that the loss at time step t depends on the hidden state at time step t, which in turn depends on all previous hidden states.

**Loss function:**

For sequence-to-sequence tasks, the total loss is typically:

```
L = Σₜ L⁽ᵗ⁾
```

Where L⁽ᵗ⁾ is the loss at time step t (e.g., cross-entropy for classification).

**Gradient of total loss with respect to parameters:**

```
∂L/∂Wₓ = Σₜ ∂L⁽ᵗ⁾/∂Wₓ
∂L/∂Wₕ = Σₜ ∂L⁽ᵗ⁾/∂Wₕ
∂L/∂Wᵧ = Σₜ ∂L⁽ᵗ⁾/∂Wᵧ
```

**Hidden state gradient:**

The gradient of the loss with respect to hidden state h⁽ᵗ⁾ has two components:

1. **Direct contribution**: Gradient from the output at time t
2. **Indirect contribution**: Gradient from the next time step h⁽ᵗ⁺¹⁾

```
∂L/∂h⁽ᵗ⁾ = ∂L⁽ᵗ⁾/∂h⁽ᵗ⁾ + ∂L/∂h⁽ᵗ⁺¹⁾ · ∂h⁽ᵗ⁺¹⁾/∂h⁽ᵗ⁾
          └─────────────┘   └──────────────────────────┘
          direct from y⁽ᵗ⁾   indirect from future
```

This is the **key recursive relation** in BPTT.

**Gradient flow visualization:**

```
BACKWARD PASS (right to left)

Time:     t=1          t=2          t=3

        y⁽¹⁾         y⁽²⁾         y⁽³⁾
         ▲            ▲            ▲
         │            │            │
         │∂L/∂y       │∂L/∂y       │∂L/∂y
         │            │            │
        [RNN]◄───────[RNN]◄───────[RNN]
         ▲  ▲         ▲  ▲         ▲
         │  │         │  │         │
         │  └─────────┼──┘         │
         │   ∂L/∂h⁽²⁾ │   ∂L/∂h⁽³⁾ │
         │            │            │
         │            │            └────► ∂L/∂h⁽ᵀ⁾ = 0 (initial)
         │            │
    ∂L/∂x⁽¹⁾     ∂L/∂x⁽²⁾     ∂L/∂x⁽³⁾

Gradients flow:
1. From output loss ∂L/∂y⁽ᵗ⁾ down to h⁽ᵗ⁾
2. From future hidden state h⁽ᵗ⁺¹⁾ back to h⁽ᵗ⁾
3. From h⁽ᵗ⁾ to parameters (accumulated)
4. From h⁽ᵗ⁾ to input x⁽ᵗ⁾ (if needed)
```

### Gradient Derivations

Let's derive the gradients step by step, starting from the output and working backward.

#### Step 1: Gradient with respect to output

Given loss L⁽ᵗ⁾ (e.g., cross-entropy), we compute:

```
∂L⁽ᵗ⁾/∂y⁽ᵗ⁾
```

For cross-entropy with softmax:
```
∂L⁽ᵗ⁾/∂y⁽ᵗ⁾ = ŷ⁽ᵗ⁾ - target⁽ᵗ⁾
```

Where ŷ⁽ᵗ⁾ = softmax(y⁽ᵗ⁾) are the predicted probabilities.

#### Step 2: Gradient with respect to output layer weights

The output is computed as y⁽ᵗ⁾ = Wᵧ · h⁽ᵗ⁾ + bᵧ

**Gradient for Wᵧ:**

```
∂L⁽ᵗ⁾/∂Wᵧ = ∂L⁽ᵗ⁾/∂y⁽ᵗ⁾ · ∂y⁽ᵗ⁾/∂Wᵧ
          = (∂L⁽ᵗ⁾/∂y⁽ᵗ⁾) ⊗ h⁽ᵗ⁾
```

Where ⊗ denotes outer product:
```
∂L⁽ᵗ⁾/∂Wᵧ[i,j] = ∂L⁽ᵗ⁾/∂y_i⁽ᵗ⁾ · h_j⁽ᵗ⁾
```

In matrix form (for batch dimension):
```
∂L⁽ᵗ⁾/∂Wᵧ = (∂L⁽ᵗ⁾/∂y⁽ᵗ⁾)ᵀ @ h⁽ᵗ⁾
```

**Gradient for bᵧ:**

```
∂L⁽ᵗ⁾/∂bᵧ = ∂L⁽ᵗ⁾/∂y⁽ᵗ⁾
```

**Total gradient (summed over all time steps):**

```
∂L/∂Wᵧ = Σₜ (∂L⁽ᵗ⁾/∂y⁽ᵗ⁾)ᵀ @ h⁽ᵗ⁾
∂L/∂bᵧ = Σₜ ∂L⁽ᵗ⁾/∂y⁽ᵗ⁾
```

#### Step 3: Gradient with respect to hidden state (output contribution)

From the output layer:

```
∂L⁽ᵗ⁾/∂h⁽ᵗ⁾|ₒᵤₜₚᵤₜ = Wᵧᵀ · ∂L⁽ᵗ⁾/∂y⁽ᵗ⁾
```

This is the direct contribution from the output at time t.

#### Step 4: Gradient with respect to hidden state (recurrent contribution)

From the next time step h⁽ᵗ⁺¹⁾:

```
∂L/∂h⁽ᵗ⁾|ᵣₑcᵤᵣᵣₑₙₜ = ∂L/∂h⁽ᵗ⁺¹⁾ · ∂h⁽ᵗ⁺¹⁾/∂h⁽ᵗ⁾
```

To compute ∂h⁽ᵗ⁺¹⁾/∂h⁽ᵗ⁾, recall:

```
h⁽ᵗ⁺¹⁾ = tanh(Wₓ · x⁽ᵗ⁺¹⁾ + Wₕ · h⁽ᵗ⁾ + bₕ)
```

Let z⁽ᵗ⁺¹⁾ = Wₓ · x⁽ᵗ⁺¹⁾ + Wₕ · h⁽ᵗ⁾ + bₕ (pre-activation)

Then:
```
h⁽ᵗ⁺¹⁾ = tanh(z⁽ᵗ⁺¹⁾)
```

By chain rule:
```
∂h⁽ᵗ⁺¹⁾/∂h⁽ᵗ⁾ = ∂h⁽ᵗ⁺¹⁾/∂z⁽ᵗ⁺¹⁾ · ∂z⁽ᵗ⁺¹⁾/∂h⁽ᵗ⁾
                = tanh'(z⁽ᵗ⁺¹⁾) · Wₕ
                = (1 - h⁽ᵗ⁺¹⁾²) ⊙ (Wₕᵀ · δ⁽ᵗ⁺¹⁾)
```

Where:
- tanh'(z) = 1 - tanh²(z) = 1 - h²
- ⊙ denotes element-wise multiplication
- δ⁽ᵗ⁺¹⁾ = ∂L/∂z⁽ᵗ⁺¹⁾ is the gradient at the pre-activation

Therefore:
```
∂L/∂h⁽ᵗ⁾|ᵣₑcᵤᵣᵣₑₙₜ = Wₕᵀ · δ⁽ᵗ⁺¹⁾
```

#### Step 5: Total gradient with respect to hidden state

Combining both contributions:

```
∂L/∂h⁽ᵗ⁾ = Wᵧᵀ · (∂L⁽ᵗ⁾/∂y⁽ᵗ⁾) + Wₕᵀ · δ⁽ᵗ⁺¹⁾
          └────────────────────┘   └────────────┘
          direct from output      from future time
```

For the last time step T:
```
∂L/∂h⁽ᵀ⁾ = Wᵧᵀ · (∂L⁽ᵀ⁾/∂y⁽ᵀ⁾)
```
(No future time step contribution)

#### Step 6: Gradient with respect to pre-activation

```
δ⁽ᵗ⁾ = ∂L/∂z⁽ᵗ⁾ = ∂L/∂h⁽ᵗ⁾ ⊙ tanh'(z⁽ᵗ⁾)
                  = ∂L/∂h⁽ᵗ⁾ ⊙ (1 - h⁽ᵗ⁾²)
```

Where ⊙ is element-wise multiplication.

#### Step 7: Gradient with respect to input weights Wₓ

```
∂L⁽ᵗ⁾/∂Wₓ = δ⁽ᵗ⁾ ⊗ x⁽ᵗ⁾
```

In matrix form:
```
∂L⁽ᵗ⁾/∂Wₓ = δ⁽ᵗ⁾ @ x⁽ᵗ⁾ᵀ
```

**Total gradient (summed over all time steps):**

```
∂L/∂Wₓ = Σₜ δ⁽ᵗ⁾ @ x⁽ᵗ⁾ᵀ
```

#### Step 8: Gradient with respect to recurrent weights Wₕ

```
∂L⁽ᵗ⁾/∂Wₕ = δ⁽ᵗ⁾ ⊗ h⁽ᵗ⁻¹⁾
```

In matrix form:
```
∂L⁽ᵗ⁾/∂Wₕ = δ⁽ᵗ⁾ @ h⁽ᵗ⁻¹⁾ᵀ
```

**Total gradient (summed over all time steps):**

```
∂L/∂Wₕ = Σₜ δ⁽ᵗ⁾ @ h⁽ᵗ⁻¹⁾ᵀ
```

#### Step 9: Gradient with respect to hidden bias bₕ

```
∂L⁽ᵗ⁾/∂bₕ = δ⁽ᵗ⁾
```

**Total gradient (summed over all time steps):**

```
∂L/∂bₕ = Σₜ δ⁽ᵗ⁾
```

### Gradient Flow Equations

**Summary of BPTT equations (backward pass from t = T to 1):**

```
1. Output gradient:
   dLdy⁽ᵗ⁾ = ∂L⁽ᵗ⁾/∂y⁽ᵗ⁾  (from loss function)

2. Hidden gradient (direct):
   dLdh⁽ᵗ⁾_out = Wᵧᵀ @ dLdy⁽ᵗ⁾

3. Hidden gradient (recurrent):
   dLdh⁽ᵗ⁾_rec = Wₕᵀ @ δ⁽ᵗ⁺¹⁾  (if t < T, else 0)

4. Total hidden gradient:
   dLdh⁽ᵗ⁾ = dLdh⁽ᵗ⁾_out + dLdh⁽ᵗ⁾_rec

5. Pre-activation gradient:
   δ⁽ᵗ⁾ = dLdh⁽ᵗ⁾ ⊙ (1 - h⁽ᵗ⁾²)

6. Parameter gradients (accumulate):
   ∂L/∂Wᵧ += dLdy⁽ᵗ⁾ @ h⁽ᵗ⁾ᵀ
   ∂L/∂bᵧ += dLdy⁽ᵗ⁾
   ∂L/∂Wₓ += δ⁽ᵗ⁾ @ x⁽ᵗ⁾ᵀ
   ∂L/∂Wₕ += δ⁽ᵗ⁾ @ h⁽ᵗ⁻¹⁾ᵀ
   ∂L/∂bₕ += δ⁽ᵗ⁾

7. Input gradient (if needed):
   dLdx⁽ᵗ⁾ = Wₓᵀ @ δ⁽ᵗ⁾
```

### Implementation Strategy

**Backward pass pseudocode:**

```python
def backward(inputs, hidden_states, outputs, targets):
    """
    inputs: (T, input_size)
    hidden_states: (T+1, hidden_size) - includes h⁽⁰⁾
    outputs: (T, output_size)
    targets: (T, output_size)
    """
    T = len(inputs)

    # Initialize gradient accumulation
    dWx = zeros_like(Wx)
    dWh = zeros_like(Wh)
    dWy = zeros_like(Wy)
    dbh = zeros_like(bh)
    dby = zeros_like(by)

    # Initialize hidden gradient for next time step
    dh_next = zeros(hidden_size)

    # Backward pass through time
    for t in reversed(range(T)):
        # 1. Compute output gradient
        dy = outputs[t] - targets[t]  # e.g., for cross-entropy + softmax

        # 2. Accumulate output layer gradients
        dWy += outer(dy, hidden_states[t+1])  # Note: h⁽ᵗ⁾ is stored at index t+1
        dby += dy

        # 3. Gradient to hidden (from output)
        dh_from_output = Wy.T @ dy

        # 4. Total hidden gradient (from output + from future)
        dh = dh_from_output + dh_next

        # 5. Gradient through tanh activation
        h = hidden_states[t+1]
        dtanh = 1 - h**2  # tanh derivative
        dz = dh * dtanh   # Element-wise multiplication

        # 6. Accumulate recurrent layer gradients
        dWx += outer(dz, inputs[t])
        dWh += outer(dz, hidden_states[t])  # h⁽ᵗ⁻¹⁾
        dbh += dz

        # 7. Gradient to previous hidden state (for next iteration)
        dh_next = Wh.T @ dz

    return dWx, dWh, dWy, dbh, dby
```

**Key implementation notes:**

1. **Reverse iteration**: Process time steps from T down to 1
2. **Gradient accumulation**: Add gradients from each time step to parameter gradients
3. **State indexing**: Careful with indexing - h⁽⁰⁾, h⁽¹⁾, ..., h⁽ᵀ⁾ requires T+1 states
4. **Memory efficiency**: Only need to store dh_next, not all hidden gradients
5. **Numerical stability**: Apply gradient clipping before or during accumulation

## The Vanishing/Exploding Gradient Problem

### Mathematical Analysis

The fundamental challenge in training RNNs is the **vanishing and exploding gradient problem**. This occurs because gradients must be backpropagated through many time steps.

**Gradient through multiple time steps:**

To understand the problem, consider how the gradient flows from time step T to time step 1. The gradient ∂L/∂h⁽¹⁾ involves a product of many Jacobian matrices:

```
∂L/∂h⁽¹⁾ = ∂L/∂h⁽ᵀ⁾ · ∂h⁽ᵀ⁾/∂h⁽ᵀ⁻¹⁾ · ∂h⁽ᵀ⁻¹⁾/∂h⁽ᵀ⁻²⁾ · ... · ∂h⁽²⁾/∂h⁽¹⁾
```

**Jacobian of hidden state transition:**

Recall that h⁽ᵗ⁾ = tanh(Wₓx⁽ᵗ⁾ + Wₕh⁽ᵗ⁻¹⁾ + bₕ)

The Jacobian ∂h⁽ᵗ⁾/∂h⁽ᵗ⁻¹⁾ is:

```
∂h⁽ᵗ⁾/∂h⁽ᵗ⁻¹⁾ = diag(1 - h⁽ᵗ⁾²) · Wₕ
```

Where diag(1 - h⁽ᵗ⁾²) is a diagonal matrix with entries (1 - h_i⁽ᵗ⁾²).

**Gradient through k time steps:**

```
∂h⁽ᵗ⁺ᵏ⁾/∂h⁽ᵗ⁾ = ∏ᵢ₌₁ᵏ [diag(1 - h⁽ᵗ⁺ⁱ⁾²) · Wₕ]
```

**Spectral analysis:**

The norm of this product can grow or shrink exponentially:

```
‖∂h⁽ᵗ⁺ᵏ⁾/∂h⁽ᵗ⁾‖ ≤ ∏ᵢ₌₁ᵏ ‖diag(1 - h⁽ᵗ⁺ⁱ⁾²)‖ · ‖Wₕ‖
                   ≤ (‖Wₕ‖)ᵏ  (since |1 - h²| ≤ 1 for tanh)
```

**Two scenarios:**

1. **Vanishing gradients**: If ‖Wₕ‖ < 1, then gradients shrink exponentially:
   ```
   ‖∂h⁽ᵗ⁺ᵏ⁾/∂h⁽ᵗ⁾‖ ≈ (λₘₐₓ)ᵏ → 0 as k → ∞
   ```
   Where λₘₐₓ is the largest eigenvalue of Wₕ.

   **Consequence**: Network cannot learn long-term dependencies (gradients become too small).

2. **Exploding gradients**: If ‖Wₕ‖ > 1, then gradients grow exponentially:
   ```
   ‖∂h⁽ᵗ⁺ᵏ⁾/∂h⁽ᵗ⁾‖ ≈ (λₘₐₓ)ᵏ → ∞ as k → ∞
   ```

   **Consequence**: Numerical instability, NaN values, training divergence.

**Practical implications:**

- Vanilla RNNs typically struggle with dependencies longer than 10-20 time steps
- tanh activation saturates (derivative → 0), amplifying vanishing gradients
- This motivates architectures like LSTM and GRU that mitigate these issues

### Gradient Clipping

**Gradient clipping** is the primary technique to prevent exploding gradients. It caps the magnitude of gradients during training.

**Global norm clipping:**

The most common approach clips gradients based on their global norm:

```
g̃ = threshold · g / max(‖g‖, threshold)
```

Where:
- g is the gradient vector (concatenation of all parameter gradients)
- ‖g‖ is the L2 norm: ‖g‖ = √(Σᵢ gᵢ²)
- threshold is a hyperparameter (typically 1.0 to 5.0)

**Effect:**
- If ‖g‖ ≤ threshold: g̃ = g (no clipping)
- If ‖g‖ > threshold: g̃ is rescaled to have norm = threshold

**Implementation:**

```python
def clip_gradients_global_norm(gradients, threshold=5.0):
    """
    gradients: list of gradient arrays [dWx, dWh, dWy, dbh, dby]
    threshold: maximum allowed gradient norm
    """
    # Compute global norm
    total_norm = sqrt(sum(sum(g**2) for g in gradients))

    # Compute clipping ratio
    clip_ratio = threshold / max(total_norm, threshold)

    # Scale gradients if needed
    if clip_ratio < 1.0:
        gradients = [g * clip_ratio for g in gradients]

    return gradients
```

**Per-parameter clipping:**

Alternatively, clip each parameter gradient independently:

```python
def clip_gradients_per_param(gradients, threshold=5.0):
    """
    Clip each gradient array independently
    """
    clipped = []
    for g in gradients:
        norm = sqrt(sum(g**2))
        clip_ratio = threshold / max(norm, threshold)
        clipped.append(g * clip_ratio)
    return clipped
```

**When to apply:**
- After computing all gradients in backward pass
- Before applying optimizer updates
- Can also clip during accumulation for very long sequences

**Choosing threshold:**

- **Too small (< 1.0)**: May slow learning, underfitting
- **Too large (> 10.0)**: May not prevent exploding gradients
- **Typical values**: 1.0 to 5.0
- Monitor gradient norms during training to tune

**Diagnostic logging:**

```python
# During training:
grad_norm = compute_gradient_norm(gradients)
if grad_norm > threshold:
    print(f"Gradient clipped: {grad_norm:.2f} → {threshold}")
```

### Other Mitigation Strategies

**For vanishing gradients:**

1. **Use LSTM or GRU**: These architectures use gating mechanisms to preserve gradients
   - LSTM maintains a cell state with additive updates (not multiplicative)
   - GRU uses reset and update gates to control information flow

2. **Careful initialization**: Initialize Wₕ to preserve gradient flow
   - Identity or orthogonal initialization for Wₕ
   - Scale to have eigenvalues near 1

3. **Alternative activations**: ReLU instead of tanh (but less stable for RNNs)

4. **Residual connections**: Add skip connections: h⁽ᵗ⁾ = h⁽ᵗ⁻¹⁾ + f(h⁽ᵗ⁻¹⁾, x⁽ᵗ⁾)

**For exploding gradients:**

1. **Gradient clipping** (primary solution)

2. **Careful initialization**: Smaller initial weights

3. **Lower learning rate**: Reduces step size, making training more stable

4. **Regularization**: L2 penalty on Wₕ to keep norms small

## Parameter Updates

After computing gradients via BPTT, parameters are updated using an optimizer.

**Vanilla SGD:**

```
Wₓ ← Wₓ - η · ∂L/∂Wₓ
Wₕ ← Wₕ - η · ∂L/∂Wₕ
Wᵧ ← Wᵧ - η · ∂L/∂Wᵧ
bₕ ← bₕ - η · ∂L/∂bₕ
bᵧ ← bᵧ - η · ∂L/∂bᵧ
```

Where η is the learning rate (typically 0.001 to 0.01 for RNNs).

**With gradient clipping:**

```python
# 1. Compute gradients via BPTT
dWx, dWh, dWy, dbh, dby = backward_pass(...)

# 2. Clip gradients
gradients = [dWx, dWh, dWy, dbh, dby]
gradients = clip_gradients_global_norm(gradients, threshold=5.0)
dWx, dWh, dWy, dbh, dby = gradients

# 3. Apply updates
Wx -= learning_rate * dWx
Wh -= learning_rate * dWh
Wy -= learning_rate * dWy
bh -= learning_rate * dbh
by -= learning_rate * dby
```

**Adam optimizer (recommended):**

Adam adapts learning rates per parameter and includes momentum:

```python
# Initialize first and second moment estimates
m = {Wx: 0, Wh: 0, Wy: 0, bh: 0, by: 0}  # First moment
v = {Wx: 0, Wh: 0, Wy: 0, bh: 0, by: 0}  # Second moment
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8

# Update step
for param, grad in [(Wx, dWx), (Wh, dWh), ...]:
    m[param] = beta1 * m[param] + (1 - beta1) * grad
    v[param] = beta2 * v[param] + (1 - beta2) * grad**2

    m_hat = m[param] / (1 - beta1**t)  # Bias correction
    v_hat = v[param] / (1 - beta2**t)

    param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Batch updates:**

For minibatch training:
1. Accumulate gradients over batch
2. Average: ∂L/∂W = (1/B) Σᵦ ∂Lᵦ/∂W
3. Clip averaged gradients
4. Apply optimizer update

## Initialization

Proper initialization is crucial for RNN training due to the vanishing/exploding gradient problem.

**Input weights Wₓ:**

Xavier/Glorot initialization:
```
Wₓ ~ U(-√(6/(input_size + hidden_size)), √(6/(input_size + hidden_size)))
```

Or He initialization (for ReLU):
```
Wₓ ~ N(0, √(2/input_size))
```

**Recurrent weights Wₕ:**

**Identity initialization** (recommended for vanilla RNN):
```
Wₕ = α · I
```
Where I is the identity matrix and α ∈ [0.5, 1.0].

Benefits:
- Preserves gradient magnitude (eigenvalues = α)
- Helps mitigate vanishing gradients
- Allows learning long-term dependencies

**Orthogonal initialization** (alternative):
```
Wₕ = orthogonal_matrix() · α
```
Where orthogonal_matrix has singular values = 1.

Benefits:
- Preserves norm during forward and backward pass
- Eigenvalues have magnitude 1
- Widely used in practice

**Implementation:**

```python
import numpy as np

def initialize_rnn_weights(input_size, hidden_size, output_size):
    # Input weights: Xavier
    limit_x = np.sqrt(6.0 / (input_size + hidden_size))
    Wx = np.random.uniform(-limit_x, limit_x, (hidden_size, input_size))

    # Recurrent weights: Orthogonal
    Wh = orthogonal_matrix(hidden_size) * 0.9

    # Output weights: Xavier
    limit_y = np.sqrt(6.0 / (hidden_size + output_size))
    Wy = np.random.uniform(-limit_y, limit_y, (output_size, hidden_size))

    # Biases: Zero or small random
    bh = np.zeros(hidden_size)
    by = np.zeros(output_size)

    return Wx, Wh, Wy, bh, by

def orthogonal_matrix(size):
    """Generate orthogonal matrix via QR decomposition"""
    random_matrix = np.random.randn(size, size)
    q, r = np.linalg.qr(random_matrix)
    return q
```

**Output weights Wᵧ and biases:**
- Standard Xavier/Glorot initialization for Wᵧ
- Zero initialization for biases is typical

## Truncated BPTT

For very long sequences (thousands of time steps), standard BPTT becomes computationally expensive and memory-intensive. **Truncated BPTT** addresses this by limiting gradient backpropagation to a fixed number of steps.

**Algorithm:**

1. **Forward pass**: Process entire sequence, maintaining hidden state
2. **Backward pass**: Only backpropagate through last k₁ time steps
3. **Update**: Apply gradients computed over the k₁ steps
4. **Continue**: Maintain hidden state, advance k₂ steps, repeat

Where:
- k₁ = truncation length for backward pass (e.g., 20-50)
- k₂ = step size for moving forward (typically k₂ = k₁)

**Visualization:**

```
Sequence: x⁽¹⁾ x⁽²⁾ x⁽³⁾ ... x⁽²⁰⁾ x⁽²¹⁾ ... x⁽⁴⁰⁾ x⁽⁴¹⁾ ...
          └─────────┬─────────┘ └──────┬──────┘ └──────┬──────┘
               Chunk 1             Chunk 2         Chunk 3

          Forward: Process all
          Backward: Only within each chunk

Chunk 1: Forward x⁽¹⁾ → x⁽²⁰⁾, backward x⁽²⁰⁾ → x⁽¹⁾
         Save h⁽²⁰⁾ for next chunk

Chunk 2: Forward x⁽²¹⁾ → x⁽⁴⁰⁾ (starting from h⁽²⁰⁾)
         Backward x⁽⁴⁰⁾ → x⁽²¹⁾ (don't backprop further)
         Save h⁽⁴⁰⁾ for next chunk
```

**Implementation:**

```python
def truncated_bptt(inputs, targets, k1=20, k2=20):
    """
    inputs: (total_T, input_size)
    k1: truncation length (backprop steps)
    k2: forward step size
    """
    total_T = len(inputs)
    h = zeros(hidden_size)  # Initial hidden state

    for start_t in range(0, total_T, k2):
        end_t = min(start_t + k2, total_T)
        chunk_inputs = inputs[start_t:end_t]
        chunk_targets = targets[start_t:end_t]

        # Forward pass for this chunk
        outputs, hidden_states = forward_pass(chunk_inputs, h)

        # Backward pass (only within chunk)
        gradients = backward_pass(chunk_inputs, hidden_states, outputs, chunk_targets)

        # Update parameters
        update_parameters(gradients)

        # Continue with final hidden state (no gradient flow)
        h = hidden_states[-1].detach()  # Detach from computation graph
```

**Advantages:**
- **Memory**: O(k₁) instead of O(T) for activations
- **Computation**: Backward pass is O(k₁) per update
- **Scalability**: Can process arbitrarily long sequences

**Disadvantages:**
- **Gradient approximation**: Gradients beyond k₁ steps are ignored
- **Long dependencies**: Cannot learn dependencies longer than k₁
- **Tuning**: Requires choosing k₁ and k₂ hyperparameters

**Typical values:**
- k₁ = 20 to 100 (longer for LSTM/GRU)
- k₂ = k₁ (non-overlapping chunks)
- Or k₂ < k₁ for overlapping backprop

## Numerical Considerations

**1. Gradient clipping (critical):**

Always apply gradient clipping to prevent exploding gradients:

```python
threshold = 5.0  # Typical range: 1.0 to 5.0
gradients = clip_gradients_global_norm(gradients, threshold)
```

**2. Activation saturation:**

tanh outputs are in [-1, 1], with derivatives in [0, 1]:
- When |h| → 1, tanh'(z) → 0 (vanishing gradients)
- Monitor hidden state statistics: mean, std, saturation percentage

**3. Loss scaling:**

For sequences with many time steps, loss may be summed over time:
```
L = Σₜ L⁽ᵗ⁾
```

Consider normalizing by sequence length:
```
L = (1/T) Σₜ L⁽ᵗ⁾
```

This makes learning rates comparable across different sequence lengths.

**4. Hidden state initialization:**

For minibatch training with variable-length sequences:
- Reset h⁽⁰⁾ = 0 for each new sequence
- Or learn initial hidden state as a parameter

**5. Sequence padding:**

When batching sequences of different lengths:
- Pad shorter sequences to max length
- Use masking to ignore padded positions in loss computation

**6. Gradient checking:**

Verify BPTT implementation with numerical gradient checking:

```python
def check_gradients(inputs, targets, epsilon=1e-5):
    """Compare analytical gradients to numerical gradients"""

    # Compute analytical gradients
    analytical_grads = backward_pass(inputs, targets)

    # Compute numerical gradients
    for param_name, param in [("Wx", Wx), ("Wh", Wh), ...]:
        numerical_grad = zeros_like(param)

        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                # Perturb parameter
                param[i,j] += epsilon
                loss_plus = forward_and_loss(inputs, targets)

                param[i,j] -= 2*epsilon
                loss_minus = forward_and_loss(inputs, targets)

                param[i,j] += epsilon  # Restore

                # Compute numerical gradient
                numerical_grad[i,j] = (loss_plus - loss_minus) / (2*epsilon)

        # Compare
        diff = norm(analytical_grads[param_name] - numerical_grad)
        print(f"{param_name}: gradient difference = {diff}")
```

**7. Mixed precision training:**

For large models, consider mixed precision (FP16/FP32):
- Forward and backward in FP16 for speed
- Parameter updates in FP32 for precision
- Requires loss scaling to prevent underflow

**8. Monitoring:**

Track these metrics during training:
- Loss per epoch
- Gradient norms (before and after clipping)
- Hidden state statistics (mean, std, saturation)
- Percentage of clipped updates

**9. Debugging tips:**

If training fails:
- Check for NaN/Inf values in hidden states and gradients
- Reduce learning rate
- Increase gradient clipping threshold
- Verify input data normalization
- Check weight initialization
- Try simpler sequences first (shorter, less complex)

---

## Summary

Backpropagation Through Time (BPTT) is the core algorithm for training RNNs:

1. **Unfolding**: RNN is unfolded into a deep feedforward network through time
2. **Forward pass**: Compute hidden states and outputs sequentially
3. **Backward pass**: Backpropagate gradients from future to past
4. **Accumulation**: Sum gradients across all time steps (parameter sharing)
5. **Clipping**: Apply gradient clipping to prevent exploding gradients
6. **Update**: Use accumulated gradients to update parameters

**Key challenges:**
- Vanishing gradients (long-term dependencies)
- Exploding gradients (numerical instability)
- Memory requirements (storing all hidden states)

**Solutions:**
- Gradient clipping (exploding gradients)
- Careful initialization (vanishing gradients)
- Advanced architectures (LSTM/GRU for vanishing gradients)
- Truncated BPTT (memory and computation)

**Implementation checklist:**
- ✓ Store all hidden states and pre-activations during forward pass
- ✓ Implement backward pass iterating from t=T to t=1
- ✓ Accumulate gradients across time steps
- ✓ Apply gradient clipping before parameter updates
- ✓ Use appropriate initialization (orthogonal for Wₕ)
- ✓ Monitor gradient norms during training
- ✓ Verify with gradient checking on small examples

Understanding BPTT is essential for implementing and debugging RNN-based models effectively.
