# Backpropagation

This directory contains comprehensive mathematical documentation for backpropagation algorithms implemented in this project. Each document includes mathematical derivations, gradient formulas, and implementation details for neural network layers.

## Overview

Backpropagation is the fundamental algorithm for training neural networks through gradient descent. It efficiently computes gradients of the loss function with respect to network parameters by applying the chain rule of calculus backwards through the network layers.

**Key concepts:**
- **Forward pass**: Compute outputs layer by layer
- **Backward pass**: Compute gradients layer by layer in reverse
- **Chain rule**: Combine gradients across layers
- **Parameter updates**: Use gradients to update weights and biases

## Documentation Structure

This directory organizes backpropagation documentation by layer type and operation:

### Core Layers
- **[Dense (Fully Connected) Layer](dense_layer.md)** - Matrix multiplication, bias addition, and their gradients
- **[Convolutional Layer](conv2d_layer.md)** - Convolution operations and gradient computation
- **[Attention Mechanism](attention_mechanism.md)** - Multi-head attention, softmax attention, and gradient computation
- **Pooling Layer** - Max pooling and average pooling backpropagation

### Operations
- **Activation Functions** - Gradients for ReLU, sigmoid, tanh, etc. (see `docs/activation_functions.md`)
- **Loss Functions** - Cross-entropy, MSE, and their derivatives
- **Normalization** - Batch normalization and layer normalization gradients

## Mathematical Notation

Throughout these documents, we use the following notation conventions:

**Dimensions:**
- `B` = Batch size
- `D_in` = Input dimension
- `D_out` = Output dimension
- `H, W` = Height, Width (for images)
- `C` = Number of channels

**Variables:**
- `X` = Input activations (shape: `B × D_in`)
- `W` = Weight matrix (shape: `D_in × D_out`)
- `b` = Bias vector (shape: `D_out`)
- `Y` = Output activations (shape: `B × D_out`)
- `L` = Loss (scalar)

**Gradients:**
- `∂L/∂Y` = Gradient of loss with respect to output (upstream gradient)
- `∂L/∂X` = Gradient of loss with respect to input (downstream gradient)
- `∂L/∂W` = Gradient of loss with respect to weights
- `∂L/∂b` = Gradient of loss with respect to biases

**Operations:**
- `⊙` = Element-wise (Hadamard) product
- `⊗` = Matrix multiplication
- `⊤` = Transpose

## Implementation References

**Core trait:**
- `src/layers/trait.rs` - `Layer` trait defining `forward()`, `backward()`, `update_parameters()`

**Layer implementations:**
- `src/layers/dense.rs` - Dense layer with BLAS acceleration
- `src/layers/conv2d.rs` - Convolutional layer

**Utilities:**
- `src/utils/activations.rs` - Activation functions and gradients
- `src/utils/rng.rs` - Weight initialization

**Tests:**
- `tests/test_backward_pass.rs` - Backpropagation correctness tests
- `tests/test_gradient_checking.rs` - Numerical gradient validation

## General Backpropagation Algorithm

For a layer with forward pass `Y = f(X, θ)` where `θ` represents parameters:

**Forward Pass:**
```
1. Receive input X
2. Compute output Y = f(X, θ)
3. Cache any values needed for backward pass
```

**Backward Pass:**
```
1. Receive upstream gradient ∂L/∂Y
2. Compute parameter gradients: ∂L/∂θ = (∂L/∂Y) × (∂Y/∂θ)
3. Compute input gradient: ∂L/∂X = (∂L/∂Y) × (∂Y/∂X)
4. Return ∂L/∂X to previous layer
```

**Parameter Update:**
```
θ_new = θ_old - learning_rate × ∂L/∂θ
```

## Gradient Checking

All gradient implementations should be validated using numerical gradient checking:

```
numerical_gradient = (f(θ + ε) - f(θ - ε)) / (2ε)
```

where `ε ≈ 1e-4`. The analytical gradient should match the numerical gradient within tolerance `~1e-6`.

See `tests/test_gradient_checking.rs` for implementation.

## Common Issues and Solutions

**Vanishing gradients:**
- Gradients become very small in early layers of deep networks
- Solutions: ReLU activation, residual connections, careful initialization

**Exploding gradients:**
- Gradients become very large, causing unstable training
- Solutions: Gradient clipping, lower learning rate, normalization layers

**Dying ReLU:**
- ReLU units output zero for all inputs
- Solutions: Leaky ReLU, careful initialization, lower learning rate

**Numerical instability:**
- Large exponentials cause overflow/underflow
- Solutions: Log-space computation, max-subtraction trick (softmax)

## Further Reading

- Rumelhart et al. (1986): Learning representations by back-propagating errors
- Goodfellow et al. (2016): Deep Learning, Chapter 6 (Deep Feedforward Networks)
- Nielsen (2015): Neural Networks and Deep Learning, Chapter 2
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition

## Contributing

When adding new backpropagation documentation:

1. **Include mathematical derivations** - Show all steps in gradient computation
2. **Provide dimensional analysis** - Track matrix shapes through all operations
3. **Reference implementation** - Link to relevant Rust code
4. **Add examples** - Include concrete numerical examples where helpful
5. **Validate gradients** - Include gradient checking tests
6. **Follow notation** - Use consistent mathematical notation defined above

---

**Related documentation:**
- [Activation Functions](../activation_functions.md) - Activation function gradients
- [Dense Layer Backpropagation](dense_layer.md) - Fully connected layer gradients
- [Conv2D Layer Backpropagation](conv2d_layer.md) - Convolutional layer gradients
- [Attention Mechanism Backpropagation](attention_mechanism.md) - Attention layer gradients
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface
