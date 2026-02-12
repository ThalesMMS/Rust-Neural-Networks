# Mathematical Documentation Guide

This guide establishes conventions for documenting mathematical operations, formulas, and gradient computations in the Rust Neural Networks project.

## Table of Contents

1. [LaTeX-Style Formula Format](#latex-style-formula-format)
2. [Chain Rule Explanation Pattern](#chain-rule-explanation-pattern)
3. [Matrix Dimension Notation](#matrix-dimension-notation)
4. [Layer Documentation Structure](#layer-documentation-structure)
5. [Complete Examples](#complete-examples)

---

## LaTeX-Style Formula Format

### Basic Conventions

Use ASCII-friendly notation that resembles LaTeX but is readable in code comments:

- **Subscripts**: Use `_` for subscripts (e.g., `x_t`, `h_{t-1}`)
- **Superscripts**: Use `^` for exponents (e.g., `x^2`)
- **Greek letters**: Write out names (e.g., `sigma`, `alpha`, `epsilon`)
- **Multiplication**: Use `×` for matrix/vector multiplication, `⊙` for element-wise
- **Functions**: Use standard notation (e.g., `tanh()`, `σ()`, `exp()`)

### Mathematical Symbols

| Operation | Symbol | Example |
|-----------|--------|---------|
| Matrix multiplication | `×` | `A × B` |
| Element-wise multiplication | `⊙` | `a ⊙ b` |
| Dot product | `·` | `a · b` |
| Sigmoid function | `σ()` | `σ(x)` |
| Hyperbolic tangent | `tanh()` | `tanh(x)` |
| ReLU | `ReLU()` | `ReLU(x)` |
| Summation | `Σ` | `Σ x_i` |
| Partial derivative | `∂` | `∂L/∂W` |

### Formula Layout

**Single-line formulas** (for simple operations):
```rust
/// Forward pass: y = xW + b
```

**Multi-step formulas** (for complex operations):
```rust
/// 1. **Forget gate**: Controls what information to discard from cell state
///    - `f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)`
/// 2. **Input gate**: Controls what new information to add to cell state
///    - `i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)`
/// 3. **Cell state update**: Combines forget and input gates
///    - `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
```

### Variable Definitions

Always define variables immediately after introducing formulas:

```rust
/// h_t = tanh(x_t × W_xh + h_{t-1} × W_hh + b_h)
///
/// where:
/// - `x_t` is the input at time step t
/// - `h_t` is the hidden state at time step t
/// - `W_xh` is the input-to-hidden weight matrix
/// - `W_hh` is the hidden-to-hidden weight matrix
/// - `b_h` is the hidden bias vector
```

---

## Chain Rule Explanation Pattern

### Gradient Flow Structure

Document backward pass gradients using the chain rule pattern. Show the flow from loss to parameters:

```rust
/// # Backward Pass (Dense Layer Example)
///
/// Given gradient w.r.t. output: ∂L/∂y (batch_size × output_size)
///
/// **Step 1: Weight gradients**
/// - ∂L/∂W = x^T × ∂L/∂y
/// - Dimension check: (input_size × batch_size) × (batch_size × output_size) → (input_size × output_size)
///
/// **Step 2: Bias gradients**
/// - ∂L/∂b = Σ(∂L/∂y) along batch dimension
/// - Dimension check: sum over (batch_size × output_size) → (output_size)
///
/// **Step 3: Input gradients (for backprop to previous layer)**
/// - ∂L/∂x = ∂L/∂y × W^T
/// - Dimension check: (batch_size × output_size) × (output_size × input_size) → (batch_size × input_size)
```

### Activation Function Gradients

Document activation derivatives with their mathematical form:

```rust
/// # Activation Gradients
///
/// **Sigmoid**: σ(x) = 1 / (1 + exp(-x))
/// - Derivative: σ'(x) = σ(x) ⊙ (1 - σ(x))
///
/// **Tanh**: tanh(x)
/// - Derivative: tanh'(x) = 1 - tanh²(x)
///
/// **ReLU**: ReLU(x) = max(0, x)
/// - Derivative: ReLU'(x) = 1 if x > 0, else 0
```

### Chain Rule Application

Show explicit chain rule steps for complex layers:

```rust
/// # LSTM Backward Pass (Simplified Cell State Example)
///
/// Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
///
/// **Chain rule application:**
///
/// 1. Gradient w.r.t. cell candidate (c̃_t):
///    - ∂L/∂c̃_t = ∂L/∂c_t ⊙ i_t ⊙ (1 - c̃_t²)
///    - Uses: ∂c_t/∂c̃_t = i_t and ∂c̃_t/∂(pre-activation) = 1 - tanh²
///
/// 2. Gradient w.r.t. input gate (i_t):
///    - ∂L/∂i_t = ∂L/∂c_t ⊙ c̃_t ⊙ i_t ⊙ (1 - i_t)
///    - Uses: ∂c_t/∂i_t = c̃_t and ∂i_t/∂(pre-activation) = σ'
///
/// 3. Gradient w.r.t. previous cell state (c_{t-1}):
///    - ∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t
///    - Uses: ∂c_t/∂c_{t-1} = f_t
```

---

## Matrix Dimension Notation

### Dimension Format

Use the `×` symbol to indicate matrix/tensor dimensions, written as comments:

```rust
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,     // (input_size × output_size)
    biases: Vec<f32>,      // (output_size)
    grad_weights: RefCell<Vec<f32>>,  // (input_size × output_size)
    grad_biases: RefCell<Vec<f32>>,   // (output_size)
}
```

### Dimension Checking in Operations

Always document dimension transformations in matrix operations:

```rust
/// Forward pass GEMM operation:
/// - Input x: (batch_size × input_size)
/// - Weights W: (input_size × output_size)
/// - Output y: (batch_size × output_size)
/// - Operation: y = x × W + b
```

### Multi-Dimensional Tensors

For convolutional and higher-dimensional operations, use clear dimension ordering:

```rust
pub struct Conv2DLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    // Weights: (out_channels × in_channels × kernel_size × kernel_size)
    weights: Vec<f32>,
    // Biases: (out_channels)
    biases: Vec<f32>,
}
```

**Dimension order conventions:**
- **Dense layers**: Batch dimension first, then features
- **Convolution filters**: `(out_channels × in_channels × height × width)`
- **Feature maps**: `(batch_size × height × width × channels)` (channels-last)
- **Recurrent layers**: `(sequence_length × batch_size × features)` or `(batch_size × features)` per time step

---

## Layer Documentation Structure

### Module-Level Documentation

Every layer module should have comprehensive module-level documentation (`//!`):

```rust
//! Layer Name Implementation
//!
//! Brief description of what the layer does and its purpose.
//!
//! # Architecture
//!
//! Detailed explanation of the mathematical operations performed:
//! - Step-by-step formulas
//! - Variable definitions
//! - Dimension transformations
//!
//! # Usage Example
//!
//! ```ignore
//! use rust_neural_networks::layers::{LayerName, Layer};
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! let layer = LayerName::new(...);
//! // Usage pattern
//! ```
//!
//! # Important Notes
//!
//! - Key considerations
//! - Gotchas or best practices
//! - Performance characteristics
```

### Struct Documentation

Provide detailed struct documentation with field descriptions:

```rust
/// Brief description of the layer.
///
/// Mathematical formula: y = f(x; θ)
///
/// where x is input, θ are parameters, and f is the transformation.
///
/// # Fields
///
/// ## Input/Output Dimensions
/// * `input_size` - Number of input features
/// * `output_size` - Number of output features
///
/// ## Learnable Parameters
/// * `weights` - Weight matrix (input_size × output_size)
/// * `biases` - Bias vector (output_size)
///
/// ## Gradient Accumulators
/// * `grad_weights` - Weight gradients (input_size × output_size)
/// * `grad_biases` - Bias gradients (output_size)
///
/// # Example
///
/// ```ignore
/// let layer = LayerName::new(128, 64, &mut rng);
/// ```
pub struct LayerName {
    // Fields...
}
```

### Method Documentation

Document initialization methods with mathematical details:

```rust
/// Creates a layer with Xavier-initialized weights and zero biases.
///
/// Weights are sampled uniformly from [-limit, limit], where
/// `limit = sqrt(6.0 / (input_size + output_size))`. This initialization
/// maintains variance across layers and helps prevent vanishing/exploding gradients.
///
/// # Arguments
///
/// * `input_size` - Number of input features
/// * `output_size` - Number of output features
/// * `rng` - Random number generator for weight initialization
///
/// # Examples
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(784, 512, &mut rng);
/// ```
pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
    // Implementation...
}
```

---

## Complete Examples

### Example 1: Dense Layer Documentation

```rust
//! Dense (fully connected) layer implementation
//!
//! This module provides a DenseLayer (also known as Linear or Fully Connected layer)
//! that performs the transformation: output = input × weights + biases
//!
//! # Architecture
//!
//! The dense layer performs a linear transformation:
//! - Forward pass: `y = xW + b`
//!
//! where:
//! - `x` is the input (batch_size × input_size)
//! - `W` is the weight matrix (input_size × output_size)
//! - `b` is the bias vector (output_size)
//! - `y` is the output (batch_size × output_size)
//!
//! # Backward Pass
//!
//! Gradients computed via chain rule:
//!
//! 1. **Weight gradients**: ∂L/∂W = x^T × ∂L/∂y
//!    - Dimension: (input_size × batch_size) × (batch_size × output_size) → (input_size × output_size)
//!
//! 2. **Bias gradients**: ∂L/∂b = Σ(∂L/∂y) along batch dimension
//!    - Dimension: sum over (batch_size × output_size) → (output_size)
//!
//! 3. **Input gradients**: ∂L/∂x = ∂L/∂y × W^T
//!    - Dimension: (batch_size × output_size) × (output_size × input_size) → (batch_size × input_size)
//!
//! # Usage Example
//!
//! ```ignore
//! use rust_neural_networks::layers::{DenseLayer, Layer};
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! let layer = DenseLayer::new(784, 512, &mut rng);
//!
//! let input = vec![0.5; 784];
//! let mut output = vec![0.0; 512];
//! layer.forward(&input, &mut output, 1);
//! ```

/// Dense (fully connected) layer with weights and biases.
///
/// Performs the linear transformation: y = xW + b
/// where x is the input (batch_size × input_size),
/// W is the weight matrix (input_size × output_size),
/// and b is the bias vector (output_size).
///
/// # Fields
///
/// * `input_size` - Number of input features
/// * `output_size` - Number of output features
/// * `weights` - Weight matrix stored in row-major format (input_size × output_size)
/// * `biases` - Bias vector (output_size)
///
/// # Example
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(784, 512, &mut rng);
/// ```
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,     // (input_size × output_size)
    biases: Vec<f32>,      // (output_size)
    grad_weights: RefCell<Vec<f32>>,  // (input_size × output_size)
    grad_biases: RefCell<Vec<f32>>,   // (output_size)
}
```

### Example 2: Conv2D Layer Documentation

```rust
//! 2D Convolutional layer implementation
//!
//! This module provides a Conv2DLayer that performs 2D convolution operations,
//! commonly used in computer vision tasks like image classification.
//!
//! # Architecture
//!
//! The Conv2D layer slides learnable filters over the input to produce feature maps:
//!
//! - **Convolution operation**: For each output position (i, j):
//!   ```
//!   y[i,j,c_out] = Σ(x[i*s + di, j*s + dj, c_in] × W[di, dj, c_in, c_out]) + b[c_out]
//!   ```
//!   where the sum is over all kernel positions (di, dj) and input channels (c_in).
//!
//! - **Output dimensions**:
//!   ```
//!   output_height = floor((input_height + 2*padding - kernel_size) / stride) + 1
//!   output_width = floor((input_width + 2*padding - kernel_size) / stride) + 1
//!   ```
//!
//! # Backward Pass
//!
//! 1. **Filter gradients**: ∂L/∂W accumulated by convolving input with gradient
//!    - For each filter position, compute: ∂L/∂W[di,dj,c_in,c_out] += x[...] × ∂L/∂y[...]
//!
//! 2. **Bias gradients**: ∂L/∂b = Σ(∂L/∂y) over spatial and batch dimensions
//!    - Sum gradients across all output positions for each channel
//!
//! 3. **Input gradients**: ∂L/∂x computed by convolving gradient with flipped filters
//!    - Implements transposed convolution (also called deconvolution)
//!
//! # Usage Example
//!
//! ```ignore
//! use rust_neural_networks::layers::{Conv2DLayer, Layer};
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! // 1 input channel (grayscale), 8 output channels, 3×3 kernel
//! let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
//!
//! let input = vec![0.5; 28 * 28];  // Grayscale 28×28 image
//! let mut output = vec![0.0; 28 * 28 * 8];  // 8 feature maps
//! layer.forward(&input, &mut output, 1);
//! ```
//!
//! # Important Notes
//!
//! - Input expected in channels-last format: (H × W × C)
//! - Filters stored as: (out_channels × in_channels × kernel_size × kernel_size)
//! - Zero-padding is applied symmetrically on all sides

/// 2D Convolutional layer with learnable filters.
///
/// Performs 2D convolution: slides filters over input to produce feature maps.
/// Supports zero-padding and configurable stride.
///
/// # Fields
///
/// * `in_channels` - Number of input channels (e.g., 1 for grayscale, 3 for RGB)
/// * `out_channels` - Number of output feature maps (number of filters)
/// * `kernel_size` - Size of the convolutional kernel (assumed square: kernel_size × kernel_size)
/// * `padding` - Zero-padding applied to input (symmetric on all sides)
/// * `stride` - Stride for the convolution operation
/// * `input_height` - Height of input feature map
/// * `input_width` - Width of input feature map
/// * `weights` - Convolutional filters (out_channels × in_channels × kernel_size × kernel_size)
/// * `biases` - Bias for each output channel (out_channels)
///
/// # Example
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
/// ```
pub struct Conv2DLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: isize,
    stride: usize,
    input_height: usize,
    input_width: usize,
    weights: Vec<f32>,  // (out_channels × in_channels × kernel_size × kernel_size)
    biases: Vec<f32>,   // (out_channels)
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}
```

### Example 3: RNN Layer with Chain Rule

```rust
//! Recurrent Neural Network (RNN) layer implementation
//!
//! # Architecture
//!
//! The RNN layer implements the following recurrent transformation:
//! - Hidden state update: `h_t = tanh(x_t × W_xh + h_{t-1} × W_hh + b_h)`
//! - Output computation: `y_t = h_t × W_hy + b_y`
//!
//! where:
//! - `x_t` is the input at time step t (input_size)
//! - `h_t` is the hidden state at time step t (hidden_size)
//! - `y_t` is the output at time step t (output_size)
//!
//! # Backward Pass Through Time (BPTT)
//!
//! Gradients flow backward through the recurrent connections:
//!
//! 1. **Output gradient**: ∂L/∂y_t
//!
//! 2. **Hidden-to-output weight gradients**:
//!    - ∂L/∂W_hy = h_t^T × ∂L/∂y_t
//!    - Dimension: (hidden_size × 1) × (1 × output_size) → (hidden_size × output_size)
//!
//! 3. **Hidden state gradient before activation**:
//!    - ∂L/∂h_t = ∂L/∂y_t × W_hy^T
//!    - Apply tanh derivative: ∂L/∂(pre_h_t) = ∂L/∂h_t ⊙ (1 - h_t²)
//!
//! 4. **Input-to-hidden weight gradients**:
//!    - ∂L/∂W_xh = x_t^T × ∂L/∂(pre_h_t)
//!    - Dimension: (input_size × 1) × (1 × hidden_size) → (input_size × hidden_size)
//!
//! 5. **Hidden-to-hidden weight gradients**:
//!    - ∂L/∂W_hh = h_{t-1}^T × ∂L/∂(pre_h_t)
//!    - Dimension: (hidden_size × 1) × (1 × hidden_size) → (hidden_size × hidden_size)
//!
//! 6. **Gradient to previous time step**:
//!    - ∂L/∂h_{t-1} = ∂L/∂(pre_h_t) × W_hh^T
//!    - This gradient flows backward to time step t-1
```

---

## Summary Checklist

When documenting a new layer or mathematical operation, ensure you include:

- [ ] **Module-level documentation** with architecture overview
- [ ] **LaTeX-style formulas** using standard notation (`×`, `⊙`, `σ`, etc.)
- [ ] **Variable definitions** for all symbols used in formulas
- [ ] **Matrix dimensions** as inline comments (e.g., `// (M × N)`)
- [ ] **Dimension checks** showing transformations through operations
- [ ] **Chain rule explanation** for backward pass with step-by-step derivations
- [ ] **Activation derivatives** with mathematical formulas
- [ ] **Usage examples** demonstrating typical use cases
- [ ] **Important notes** about gotchas, performance, or best practices
- [ ] **Struct field documentation** with dimensions and purposes

---

## References

- **Pattern files**: `src/layers/lstm.rs`, `src/layers/rnn.rs`
- **Example implementations**: `src/layers/dense.rs`, `src/layers/conv2d.rs`
- **Layer trait**: `src/layers/trait.rs`
