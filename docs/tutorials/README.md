# Step-by-Step Learning Tutorials

This directory contains comprehensive step-by-step tutorials for building neural network architectures from scratch. Each tutorial progressively introduces concepts, layers, and operations with detailed explanations, worked examples, and expected outputs for verification.

## Overview

These tutorials provide a guided learning path through neural network implementation, starting from the simplest XOR classifier and progressing to convolutional networks for image recognition. Each tutorial is designed to be self-contained while building on concepts from previous tutorials.

**Key learning outcomes:**
- **Incremental understanding**: Build networks layer by layer with explanations at each step
- **Verification**: Check your understanding with expected intermediate outputs
- **Practical implementation**: See how mathematical concepts translate to Rust code
- **Progressive complexity**: Start simple and gradually tackle more complex architectures

## Tutorial Structure

This directory organizes tutorials by increasing complexity, from basic feedforward networks to advanced convolutional architectures:

### Beginner Level
- **[Tutorial 01: XOR MLP](01_xor_mlp.md)** - Build a simple 2→4→1 network from scratch
  - Introduction to layers, activation functions, and backpropagation
  - Step-by-step construction of a tiny network
  - Understanding gradient flow through manual calculations
  - Expected output: Perfect XOR classification after 1M epochs
  - **Time:** 30-45 minutes | **Prerequisites:** Basic calculus and matrix operations

### Intermediate Level
- **[Tutorial 02: MNIST MLP](02_mnist_mlp.md)** - Build a 784→512→10 digit classifier
  - Layer sizing decisions and why 512 hidden units
  - ReLU vs sigmoid activation choices
  - Softmax output for multi-class classification
  - BLAS acceleration for matrix operations
  - Expected output: ~97% test accuracy after 10 epochs
  - **Time:** 60-90 minutes | **Prerequisites:** Tutorial 01, softmax and cross-entropy

### Advanced Level
- **[Tutorial 03: MNIST CNN](03_mnist_cnn.md)** - Build a convolutional network
  - Understanding convolution operations and filter design
  - Pooling layers for spatial downsampling
  - Combining convolutional and dense layers
  - Channel and spatial dimension tracking
  - Expected output: ~98% test accuracy with fewer parameters
  - **Time:** 90-120 minutes | **Prerequisites:** Tutorial 02, convolution operation

### Expert Level
- **[Tutorial 04: RNN/LSTM Character-Level Model](04_rnn_lstm_char_level.md)** - Build a recurrent character-level language model
  - LSTM gates: forget, input, cell, and output gate mechanics
  - Backpropagation Through Time (BPTT) derivation and implementation
  - Gradient clipping to prevent exploding gradients
  - Character vocabulary encoding and text generation
  - Expected output: Model generates coherent-looking text sequences after training
  - **Time:** 120-150 minutes | **Prerequisites:** Tutorial 02, sequence modeling concepts

- **Tutorial 05: CIFAR-10 CNN** (Coming Soon) - Build an RGB image classifier
  - Multi-channel input handling (RGB vs grayscale)
  - Deeper networks and filter progression
  - Color feature extraction
  - Expected output: ~70% test accuracy on 10-class color images
  - **Time:** 120+ minutes | **Prerequisites:** Tutorial 03

## How to Use These Tutorials

### Prerequisites

Before starting these tutorials, you should:

1. **Install Rust** - Tutorials use this codebase's Rust implementations
2. **Understand basic calculus** - Derivatives and chain rule for backpropagation
3. **Know matrix operations** - Matrix multiplication, transposes, element-wise operations
4. **Have Python (optional)** - For visualization tools (`plot_comparison.py`)

**Recommended background:**
- Linear algebra (vectors, matrices, dot products)
- Basic machine learning concepts (training, validation, loss functions)
- Programming experience (any language, Rust knowledge not required)

### Tutorial Format

Each tutorial follows a consistent structure:

**1. Architecture Overview**
- Network diagram showing layer connections
- Input/output dimensions at each stage
- Parameter counts and memory requirements

**2. Step-by-Step Construction**
- Build one layer at a time
- Explain design choices (activation functions, layer sizes)
- Show dimension transformations through the network

**3. Forward Pass Walkthrough**
- Trace data through each layer
- Show intermediate activations with sample inputs
- Verify shapes match expected dimensions

**4. Backward Pass Explanation**
- Gradient computation for each layer
- Chain rule application
- Parameter update mechanics

**5. Training Procedure**
- Loss function selection and rationale
- Optimizer configuration (SGD, Adam, AdamW, RMSprop)
- Learning rate scheduling strategies
- Validation and early stopping

**6. Verification Checkpoints**
- Expected outputs at each stage
- Common debugging tips
- Performance benchmarks

**7. Exercises**
- Modifications to try (different architectures, hyperparameters)
- Expected outcomes from experiments
- Challenges for deeper understanding

### Learning Path

**Recommended progression:**

1. **Start with XOR** - Understand the basics with a tiny network you can trace by hand
2. **Move to MNIST MLP** - Scale up to real data while staying in familiar feedforward territory
3. **Tackle CNN** - Add spatial structure and convolutional operations
4. **Explore variations** - Experiment with attention mechanisms, different optimizers, etc.

**Each tutorial builds on previous concepts:**
- XOR → MNIST: Scaling to real data, multi-class classification
- MNIST MLP → MNIST CNN: Adding spatial structure, convolutions
- MNIST CNN → RNN/LSTM: Shifting from spatial to sequential data
- MNIST → CIFAR-10: Color images, deeper networks, harder problem

## Mathematical Notation

These tutorials follow the project's standard mathematical notation (see [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md)):

**Dimensions:**
- `B` = Batch size
- `D_in` = Input dimension
- `D_out` = Output dimension
- `H, W` = Height, Width (for images)
- `C` = Channels (1 for grayscale, 3 for RGB)

**Variables:**
- `X` = Input activations
- `W` = Weight matrix
- `b` = Bias vector
- `Y` = Output activations
- `L` = Loss (scalar)

**Gradients:**
- `∂L/∂Y` = Upstream gradient (from next layer)
- `∂L/∂X` = Downstream gradient (to previous layer)
- `∂L/∂W` = Weight gradients
- `∂L/∂b` = Bias gradients

**Operations:**
- `×` = Matrix multiplication
- `⊙` = Element-wise (Hadamard) product
- `⊤` = Transpose

## Implementation References

**Binary targets** (complete examples):
- `mlp_simple.rs` - XOR network (2→4→1)
- `mnist_mlp.rs` - MNIST feedforward (784→512→10)
- `mnist_cnn.rs` - MNIST convolutional (Conv→Pool→Dense)
- `cifar10_cnn.rs` - CIFAR-10 CNN (RGB 3×32×32)
- `rnn_char_level.rs` - Character-level language model (LSTM + BPTT)

**Shared library** (`rust_neural_networks` crate):
- `src/layers/trait.rs` - `Layer` trait interface
- `src/layers/dense.rs` - Dense layer with BLAS
- `src/layers/conv2d.rs` - Convolutional layer
- `src/utils/activations.rs` - Activation functions
- `src/config.rs` - Configuration system
- `src/data/` - Dataset loaders (MNIST, CIFAR-10)

**Tests** (validation):
- `tests/test_backward_pass.rs` - Gradient correctness
- `tests/test_gradient_checking.rs` - Numerical validation
- `tests/test_matrix_ops.rs` - BLAS operations

**Configuration examples**:
- `config/training/` - Default training configs
- `config/mnist_mlp_adam.json` - Adam optimizer example
- `config/mnist_mlp_cosine.json` - Cosine annealing scheduler

## Common Learning Challenges

**Mathematical concepts:**
- **Chain rule confusion**: Start with XOR tutorial's manual calculations
- **Dimension mismatches**: Each tutorial includes dimension tracking
- **Gradient vanishing**: Learn about activation choices and initialization

**Implementation issues:**
- **BLAS errors**: Tutorials explain dimension requirements for matrix operations
- **Memory layout**: Understanding row-major vs column-major storage
- **Numerical stability**: Softmax tricks and gradient clipping

**Training difficulties:**
- **Poor convergence**: Tutorials cover learning rate selection and scheduling
- **Overfitting**: Validation splits and early stopping explained
- **Debugging**: Verification checkpoints help isolate issues

## Worked Example Structure

Each tutorial includes concrete numerical examples. Here's what to expect:

**Example: XOR Forward Pass (from Tutorial 01)**
```
Input: x = [1.0, 0.0]
Hidden layer: h = ReLU(x × W1 + b1)
  - Matrix mult: [1.0, 0.0] × [[w11, w12, w13, w14], [w21, w22, w23, w24]]
  - Add bias: result + [b1, b2, b3, b4]
  - Apply ReLU: max(0, z)
  - Output shape: (4,)
Output layer: y = sigmoid(h × W2 + b2)
  - Matrix mult: h(4,) × W2(4×1)
  - Result: scalar prediction
  - Expected: y ≈ 1.0 for XOR(1,0)
```

**Why worked examples matter:**
- Verify your understanding at each step
- Debug dimension mismatches early
- Build intuition for expected value ranges
- Catch implementation errors before they compound

## Related Documentation

**Mathematical foundations:**
- [Activation Functions](../activation_functions.md) - Detailed function derivatives
- [Backpropagation Guide](../backpropagation/README.md) - Gradient computation theory
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) - Notation conventions

**Architecture design:**
- [Hyperparameters Guide](../hyperparameters.md) - Learning rate, batch size, optimizer selection
- [CIFAR-10 Architecture Design](../cifar10_architecture_design.md) - CNN design principles
- [Architecture Config](../architecture_config.md) - Configuration system

**Layer-specific details:**
- [Dense Layer Backprop](../backpropagation/dense_layer.md) - Fully connected layer gradients
- [Conv2D Layer Backprop](../backpropagation/conv2d_layer.md) - Convolutional layer gradients

**Data handling:**
- [CIFAR-10 Dataset Guide](../cifar10_dataset.md) - Binary format, RGB channel handling
- MNIST Data Setup - IDX format and download instructions

## Contributing

When creating new tutorials:

1. **Follow the tutorial format** - Use the structure defined in "Tutorial Format" section
2. **Include worked examples** - Show concrete numerical calculations
3. **Add verification checkpoints** - Provide expected outputs at each step
4. **Test all code** - Ensure examples run and produce expected results
5. **Link to implementations** - Reference actual code in `src/` and binary targets
6. **Progressive difficulty** - Build on previous tutorial concepts
7. **Clear explanations** - Assume minimal prior knowledge, explain jargon
8. **Visual aids** - Use ASCII diagrams for network architecture

**Tutorial quality checklist:**
- [ ] Architecture diagram included
- [ ] All dimensions tracked through network
- [ ] At least 3 verification checkpoints with expected outputs
- [ ] Common errors and solutions documented
- [ ] Exercises with difficulty levels (beginner/intermediate/advanced)
- [ ] Links to related documentation
- [ ] Code examples tested and working
- [ ] Mathematical notation follows project conventions

---

## Quick Start

**Ready to start learning?**

→ **Begin with [Tutorial 01: XOR MLP](01_xor_mlp.md)** - Your first neural network!

**Learning path:**
1. [Tutorial 01: XOR MLP](01_xor_mlp.md) → Understanding the basics
2. [Tutorial 02: MNIST MLP](02_mnist_mlp.md) → Scaling to real data
3. [Tutorial 03: MNIST CNN](03_mnist_cnn.md) → Adding spatial structure
4. [Tutorial 04: RNN/LSTM Character-Level Model](04_rnn_lstm_char_level.md) → Sequences and recurrent networks
5. Tutorial 05: CIFAR-10 CNN (Coming Soon) → Color images and deeper networks

**Need help?**
- Review [backpropagation documentation](../backpropagation/README.md) for mathematical details
- Check [hyperparameters guide](../hyperparameters.md) for training tips
- Examine working code in binary targets (`mlp_simple.rs`, `mnist_mlp.rs`, `mnist_cnn.rs`)
- Explore [configuration examples](../../config/training/) for different training strategies

---

**Related documentation:**
- [Backpropagation Guide](../backpropagation/README.md) - Mathematical foundations for gradient computation
- [Activation Functions](../activation_functions.md) - ReLU, sigmoid, tanh, and modern alternatives
- [Hyperparameters Guide](../hyperparameters.md) - Learning rate, batch size, optimizer selection
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) - Notation conventions
- [CIFAR-10 Architecture Design](../cifar10_architecture_design.md) - CNN design principles
- [Configuration System](../architecture_config.md) - JSON-based hyperparameter configs
