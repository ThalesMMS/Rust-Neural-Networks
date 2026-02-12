# Activation Functions

This document provides a comprehensive comparison of activation functions implemented in this project. Each section includes mathematical definitions, use cases, advantages, disadvantages, and gradient formulas.

## Table of Contents

- [Overview](#overview)
- [Activation Functions](#activation-functions)
  - [Sigmoid](#sigmoid)
  - [Tanh (Hyperbolic Tangent)](#tanh-hyperbolic-tangent)
  - [ReLU (Rectified Linear Unit)](#relu-rectified-linear-unit)
  - [Leaky ReLU](#leaky-relu)
  - [ELU (Exponential Linear Unit)](#elu-exponential-linear-unit)
  - [GELU (Gaussian Error Linear Unit)](#gelu-gaussian-error-linear-unit)
  - [Swish (SiLU)](#swish-silu)
  - [Softmax](#softmax)
- [Comparison Table](#comparison-table)
- [Selection Guide](#selection-guide)

## Overview

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. The choice of activation function can significantly impact training dynamics, convergence speed, and final model performance.

**Key considerations:**
- **Gradient flow**: Avoid vanishing/exploding gradients
- **Computational efficiency**: Inference and training speed
- **Output range**: Bounded vs unbounded outputs
- **Zero-centered**: Helps with gradient descent optimization
- **Sparsity**: Some activations produce exactly zero outputs

## Activation Functions

### Sigmoid

**Mathematical Definition:**

```
σ(x) = 1 / (1 + e^(-x))
```

**Output Range:** (0, 1)

**Gradient Formula:**

```
σ'(x) = σ(x) * (1 - σ(x))
```

**When to Use:**
- Binary classification output layer
- Probability estimation (output represents probability)
- Gates in LSTM/GRU cells
- Legacy shallow networks

**Advantages:**
- Smooth gradient
- Output interpretable as probability
- Well-understood and stable
- Bounded output prevents extreme activations

**Disadvantages:**
- **Vanishing gradient problem**: Gradients near 0 for |x| > 4
- **Not zero-centered**: Outputs always positive, causing zig-zagging gradient updates
- **Computationally expensive**: Requires exponential calculation
- Saturates easily, slowing learning

**Implementation Note:**

Uses the standard formula `1 / (1 + exp(-x))`. For numerical stability with extreme values, consider clamping inputs.

---

### Tanh (Hyperbolic Tangent)

**Mathematical Definition:**

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Output Range:** (-1, 1)

**Gradient Formula:**

```
tanh'(x) = 1 - tanh²(x)
```

**When to Use:**
- Hidden layers in RNNs
- When zero-centered outputs are desired
- Shallow networks
- When output range needs to be symmetric around zero

**Advantages:**
- **Zero-centered**: Unlike sigmoid, helps optimization
- Stronger gradients than sigmoid (derivative range: 0 to 1)
- Smooth and differentiable everywhere
- Bounded output prevents extreme activations

**Disadvantages:**
- **Vanishing gradient problem**: Similar to sigmoid for |x| > 2
- **Computationally expensive**: Requires exponential calculations
- Still saturates for large inputs
- Rarely used in modern deep networks

**Implementation Note:**

Uses Rust's built-in `tanh()` function for efficiency and numerical stability.

---

### ReLU (Rectified Linear Unit)

**Mathematical Definition:**

```
ReLU(x) = max(0, x)
```

**Output Range:** [0, ∞)

**Gradient Formula:**

```
ReLU'(x) = { 1  if x > 0
           { 0  if x ≤ 0
```

**When to Use:**
- **Default choice for hidden layers** in most deep networks
- Convolutional neural networks
- Deep feedforward networks
- When training speed is critical

**Advantages:**
- **Computationally efficient**: Simple max operation
- **Mitigates vanishing gradient**: Gradient is 1 for positive inputs
- **Sparse activation**: Approximately 50% of neurons are zero
- Accelerates convergence (6x faster than sigmoid/tanh in practice)
- Unbounded output allows for greater expressiveness

**Disadvantages:**
- **Dying ReLU problem**: Neurons can permanently output zero if weights push all inputs negative
- **Not zero-centered**: All outputs are non-negative
- **Non-differentiable at zero**: Technically undefined, but typically treated as 0 or 1
- Unbounded outputs can lead to exploding activations without proper initialization

**Implementation Note:**

Implemented in-place for memory efficiency: `relu_inplace(data: &mut [f32])`.

---

### Leaky ReLU

**Mathematical Definition:**

```
LeakyReLU(x) = { x         if x > 0
               { α * x     if x ≤ 0
```

Typical α = 0.01

**Output Range:** (-∞, ∞)

**Gradient Formula:**

```
LeakyReLU'(x) = { 1   if x > 0
                { α   if x ≤ 0
```

**When to Use:**
- When experiencing dying ReLU problems
- Deep networks where gradient flow is critical
- As a drop-in ReLU replacement for experimentation
- When negative outputs are acceptable

**Advantages:**
- **Solves dying ReLU**: Small negative slope prevents dead neurons
- Computationally efficient
- All the benefits of ReLU with better gradient flow
- Can learn from negative inputs

**Disadvantages:**
- **Hyperparameter α**: Requires tuning (though 0.01 works well in practice)
- Inconsistent performance across tasks
- Still not zero-centered
- Unbounded outputs

**Implementation Note:**

Takes `alpha` as a parameter: `leaky_relu(x: f32, alpha: f32)`. Default alpha of 0.01 is typical.

---

### ELU (Exponential Linear Unit)

**Mathematical Definition:**

```
ELU(x) = { x                    if x > 0
         { α * (e^x - 1)        if x ≤ 0
```

Typical α = 1.0

**Output Range:** (-α, ∞)

**Gradient Formula:**

```
ELU'(x) = { 1           if x > 0
          { α * e^x     if x ≤ 0
```

**When to Use:**
- Deep networks requiring robust gradient flow
- When zero-centered activations improve performance
- Image classification tasks
- When willing to accept higher computational cost for better performance

**Advantages:**
- **Closer to zero-centered**: Negative outputs push mean activation toward zero
- **Smooth everywhere**: Unlike ReLU variants, differentiable at x=0
- **Reduces bias shift**: Self-normalizing properties
- No dying neuron problem
- Better noise robustness than ReLU

**Disadvantages:**
- **Computationally expensive**: Requires exponential calculation for x < 0
- **Hyperparameter α**: Requires tuning
- Slower than ReLU in practice
- Exploding gradient possible for very negative inputs

**Implementation Note:**

Takes `alpha` as parameter: `elu(x: f32, alpha: f32)`. Alpha = 1.0 is standard.

---

### GELU (Gaussian Error Linear Unit)

**Mathematical Definition:**

Exact form:
```
GELU(x) = x * Φ(x)
```
where Φ(x) is the cumulative distribution function of standard normal distribution.

Tanh approximation (used in implementation):
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Output Range:** (-∞, ∞)

**Gradient Formula:**

Using the tanh approximation:
```
GELU'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * √(2/π) * (1 + 3 * 0.044715 * x²)

where inner = √(2/π) * (x + 0.044715 * x³)
```

**When to Use:**
- **Transformer models** (BERT, GPT, etc.)
- Natural language processing tasks
- Vision transformers
- State-of-the-art architectures requiring smooth activation
- When probabilistic interpretation is beneficial

**Advantages:**
- **Smooth and differentiable**: Unlike ReLU, smooth everywhere
- **Non-monotonic**: Can model more complex functions
- **Stochastic regularization interpretation**: Acts as adaptive dropout
- **State-of-the-art performance**: Used in most modern transformers
- Better gradient flow than ReLU in deep networks

**Disadvantages:**
- **Computationally expensive**: Requires tanh and polynomial calculations
- More complex to implement
- Less interpretable than ReLU
- Not as well-studied as ReLU

**Implementation Note:**

Uses tanh approximation with constants `SQRT_2_OVER_PI = 0.7978845608` and `COEFF = 0.044715` for efficiency.

---

### Swish (SiLU)

**Mathematical Definition:**

```
Swish(x) = x * σ(x) = x / (1 + e^(-x))
```

Also known as SiLU (Sigmoid Linear Unit).

**Output Range:** (-∞, ∞)

**Gradient Formula:**

```
Swish'(x) = σ(x) * (1 + x * (1 - σ(x)))
          = Swish(x) + σ(x) * (1 - Swish(x))
```

where σ(x) is the sigmoid function.

**When to Use:**
- Deep convolutional networks
- Mobile and efficient architectures (EfficientNet, MobileNetV3)
- When smooth activation benefits training
- Computer vision tasks

**Advantages:**
- **Smooth and non-monotonic**: Allows for richer representations
- **Self-gated**: x modulated by its own sigmoid
- **Better than ReLU in deep networks**: Empirically shown in many tasks
- Unbounded above, bounded below
- Reduces vanishing gradient problem

**Disadvantages:**
- **Computationally expensive**: Requires sigmoid calculation
- Not as efficient as ReLU
- Can be slower to converge in early training
- Less interpretable than ReLU

**Implementation Note:**

Implemented as `x / (1 + exp(-x))` which is mathematically equivalent to `x * sigmoid(x)` but saves one function call.

---

### Softmax

**Mathematical Definition:**

For a vector **z** with elements z₁, z₂, ..., zₙ:

```
softmax(zᵢ) = e^(zᵢ - max(z)) / Σⱼ e^(zⱼ - max(z))
```

The max subtraction is for numerical stability.

**Output Range:** (0, 1) with Σᵢ softmax(zᵢ) = 1

**Gradient Formula:**

For output yᵢ = softmax(zᵢ):

```
∂yᵢ/∂zⱼ = { yᵢ * (1 - yᵢ)     if i = j
          { -yᵢ * yⱼ          if i ≠ j
```

**When to Use:**
- **Multi-class classification output layer**
- When outputs should be interpreted as probabilities
- Attention mechanisms (scaled dot-product attention)
- Any scenario requiring normalized probability distribution

**Advantages:**
- **Probability distribution**: Outputs sum to 1 and are all positive
- **Differentiable**: Smooth gradients for all inputs
- **Interpretable**: Each output is the probability of that class
- Emphasizes largest values while suppressing others

**Disadvantages:**
- **Only for output layer**: Not suitable for hidden layers
- **Computationally expensive**: Requires exponentials and normalization
- **Sensitive to outliers**: Very large logits can cause numerical issues
- Can lead to overconfident predictions

**Implementation Note:**

`softmax_rows(outputs: &mut [f32], rows: usize, cols: usize)` processes row-major matrices in place. Uses max-subtraction trick: `exp(x - max(x))` to prevent overflow from large exponentials.

---

## Comparison Table

| Function | Range | Zero-Centered | Monotonic | Computational Cost | Vanishing Gradient | Common Use Cases |
|----------|-------|---------------|-----------|-------------------|-------------------|------------------|
| Sigmoid | (0, 1) | ❌ No | ✅ Yes | High | ⚠️ Yes | Binary classification, LSTM gates |
| Tanh | (-1, 1) | ✅ Yes | ✅ Yes | High | ⚠️ Yes | RNN hidden layers |
| ReLU | [0, ∞) | ❌ No | ✅ Yes | Low | ✅ No (x>0) | **Default for hidden layers** |
| Leaky ReLU | (-∞, ∞) | ❌ No | ✅ Yes | Low | ✅ No | Deep networks, dying ReLU fix |
| ELU | (-α, ∞) | ⚠️ Closer | ✅ Yes | Medium | ✅ No | Deep networks, noise robustness |
| GELU | (-∞, ∞) | ✅ Yes | ❌ No | High | ✅ No | **Transformers, NLP** |
| Swish | (-∞, ∞) | ⚠️ Closer | ❌ No | Medium | ✅ No | Deep CNNs, efficient architectures |
| Softmax | (0, 1)* | N/A | N/A | High | ⚠️ Can saturate | **Multi-class output layer** |

*Softmax outputs sum to 1 across the vector.

**Cost breakdown:**
- **Low**: Simple arithmetic (ReLU, Leaky ReLU)
- **Medium**: One exponential or conditional (ELU, Swish)
- **High**: Multiple exponentials or transcendental functions (Sigmoid, Tanh, GELU, Softmax)

---

## Selection Guide

### For Hidden Layers

**General purpose / Default choice:**
- **ReLU**: Start here. Fast, effective, well-understood.

**Deep networks (>20 layers):**
- **Leaky ReLU** or **ELU**: Better gradient flow, prevents dying neurons.

**Transformer models / NLP:**
- **GELU**: State-of-the-art for attention-based architectures.

**Computer vision / Efficient networks:**
- **Swish**: Better than ReLU in deep CNNs, used in EfficientNet.

**RNNs / Sequence models:**
- **Tanh**: Traditional choice, zero-centered outputs help training.

### For Output Layers

**Binary classification:**
- **Sigmoid**: Outputs interpretable as class probability.

**Multi-class classification:**
- **Softmax**: Produces probability distribution over classes.

**Regression:**
- **None (linear)** or **ReLU**: Depending on output range constraints.

### Debugging Tips

**Dying neurons (all outputs zero):**
- Switch from ReLU → Leaky ReLU or ELU
- Check weight initialization (He initialization for ReLU)
- Lower learning rate

**Vanishing gradients (training stalls):**
- Switch from Sigmoid/Tanh → ReLU variants
- Add skip connections (ResNet-style)
- Use batch normalization

**Exploding gradients:**
- Check weight initialization
- Lower learning rate
- Use gradient clipping
- Consider ELU over unbounded activations

**Slow convergence:**
- Ensure activation is appropriate for architecture (GELU for transformers)
- Try Swish if using ReLU in deep networks
- Verify zero-centered properties match architecture needs

### Performance Considerations

**Inference speed priority:**
1. ReLU (fastest)
2. Leaky ReLU
3. ELU
4. Swish
5. GELU
6. Tanh / Sigmoid (slowest)

**Training convergence priority:**
1. GELU (transformers)
2. Swish (CNNs)
3. ReLU / Leaky ReLU (general)
4. ELU (deep networks)
5. Tanh (RNNs)
6. Sigmoid (avoid for hidden layers)

### Architecture-Specific Recommendations

**MLP (Fully-connected):**
- Hidden: ReLU or Leaky ReLU
- Output: Softmax (classification) or Linear (regression)

**CNN (Convolutional):**
- Hidden: ReLU, Swish (if deep), or GELU (vision transformers)
- Output: Softmax (classification)

**RNN/LSTM:**
- Hidden: Tanh
- Gates: Sigmoid
- Output: Softmax or Linear

**Transformer:**
- Attention: Softmax (for attention weights)
- FFN: GELU
- Output: Softmax (classification) or Linear (regression/generation)

---

## References

- Nair & Hinton (2010): Rectified Linear Units (ReLU)
- Maas et al. (2013): Leaky ReLU
- Clevert et al. (2015): Exponential Linear Units (ELU)
- Hendrycks & Gimpel (2016): Gaussian Error Linear Units (GELU)
- Ramachandran et al. (2017): Swish activation function
- Goodfellow et al. (2016): Deep Learning textbook (comprehensive coverage)

**Implementation location:** `src/utils/activations.rs`

**Related tests:** `tests/test_activations.rs`, `tests/test_gradient_checking.rs`
