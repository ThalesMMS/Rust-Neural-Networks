# Tutorial 01: Building an XOR Neural Network from Scratch

**Level:** Beginner
**Time:** 30-45 minutes
**Prerequisites:** Basic understanding of functions, matrix multiplication, and derivatives
**Implementation:** See `mlp_simple.rs` for complete working code

**Navigation:**
← [Tutorial Index](README.md) | [Next Tutorial: MNIST MLP →](02_mnist_mlp.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [The XOR Problem](#the-xor-problem)
3. [Network Architecture](#network-architecture)
4. [Step 1: Input Layer](#step-1-input-layer)
5. [Step 2: Hidden Layer](#step-2-hidden-layer)
6. [Step 3: Output Layer](#step-3-output-layer)
7. [Forward Pass Walkthrough](#forward-pass-walkthrough)
8. [Backward Pass Walkthrough](#backward-pass-walkthrough)
9. [Training the Network](#training-the-network)
10. [Verification and Expected Outputs](#verification-and-expected-outputs)
11. [Exercises](#exercises)
12. [Next Steps](#next-steps)

---

## Introduction

Welcome to your first neural network! In this tutorial, you'll build a simple multi-layer perceptron (MLP) that learns the XOR function. This is the perfect starting point because:

- **It's small**: Only 2 inputs, 4 hidden neurons, and 1 output
- **You can trace it by hand**: Small enough to follow every calculation
- **It's non-trivial**: XOR is not linearly separable, so we need a hidden layer
- **It demonstrates core concepts**: Forward pass, backward pass, gradient descent

By the end of this tutorial, you'll understand:
- How data flows through a neural network (forward propagation)
- How gradients flow backward (backpropagation)
- How weights are updated to minimize error (gradient descent)
- Why activation functions are critical for learning non-linear patterns

**Implementation reference:** All code shown here is from `mlp_simple.rs`, a fully working implementation you can run and experiment with.

---

## The XOR Problem

### Problem Definition

The XOR (Exclusive OR) function returns `1` when inputs differ, and `0` when they're the same:

| Input A | Input B | XOR Output |
|---------|---------|------------|
| 0       | 0       | **0**      |
| 0       | 1       | **1**      |
| 1       | 0       | **1**      |
| 1       | 1       | **0**      |

### Why XOR Requires a Hidden Layer

XOR is **not linearly separable**. You cannot draw a single straight line to separate the outputs:

```
     B
     1  ┌───┬───┐
        │ 1 │ 0 │  ← Notice: 1s are on opposite corners
        ├───┼───┤
     0  │ 0 │ 1 │  ← No single line can separate them!
        └───┴───┘
     0     1     A
```

**Key insight:** A single-layer perceptron (just input → output) can only learn linear decision boundaries. To learn XOR, we need a **hidden layer** that transforms the input space, making it linearly separable in the hidden representation.

This is why XOR is the "Hello World" of neural networks — it's the simplest problem that demonstrates why we need depth (multiple layers).

---

## Network Architecture

### Architecture Diagram

```
Input Layer    Hidden Layer    Output Layer
   (2)            (4)              (1)

   x₁ ─────┐
           ├──→ h₁ ─┐
   x₂ ─────┤         │
           ├──→ h₂ ─┤
           │         ├──→ y
           ├──→ h₃ ─┤
           │         │
           └──→ h₄ ─┘

  Weights: W₁(2×4)   Weights: W₂(4×1)
  Biases:  b₁(4)     Biases:  b₂(1)
```

### Architecture Specifications

**Layer structure:**
- **Input layer**: 2 neurons (features x₁, x₂)
- **Hidden layer**: 4 neurons with sigmoid activation
- **Output layer**: 1 neuron with sigmoid activation

**Parameter counts:**
- Hidden layer: 2×4 + 4 = **12 parameters** (8 weights + 4 biases)
- Output layer: 4×1 + 1 = **5 parameters** (4 weights + 1 bias)
- **Total: 17 parameters**

**Implementation reference (from `mlp_simple.rs`):**
```rust
const NUM_INPUTS: usize = 2;
const NUM_HIDDEN: usize = 4;
const NUM_OUTPUTS: usize = 1;
```

### Why 4 Hidden Neurons?

The minimum for XOR is actually 2 hidden neurons, but we use 4 because:
- **Redundancy helps learning**: More neurons → more paths for gradients to flow
- **Faster convergence**: Multiple neurons can specialize in different input patterns
- **Still small enough to understand**: We can trace all activations by hand

**Experiment:** Try modifying `NUM_HIDDEN` to 2 or 8 and observe training speed differences!

---

## Step 1: Input Layer

### Input Representation

Each XOR sample is a 2D vector:
```
x = [x₁, x₂]
```

**The four training samples:**
```rust
let inputs: [[f32; 2]; 4] = [
    [0.0, 0.0],  // XOR(0,0) = 0
    [0.0, 1.0],  // XOR(0,1) = 1
    [1.0, 0.0],  // XOR(1,0) = 1
    [1.0, 1.0],  // XOR(1,1) = 0
];
```

**Expected outputs:**
```rust
let expected_outputs: [[f32; 1]; 4] = [
    [0.0],  // Expected for [0,0]
    [1.0],  // Expected for [0,1]
    [1.0],  // Expected for [1,0]
    [0.0],  // Expected for [1,1]
];
```

### Dimension Check ✓

- Input shape: `(2,)` — two-element vector
- Batch size: 1 (we process one sample at a time)
- No transformation needed — inputs flow directly to hidden layer

**Key concept:** The input layer doesn't have learnable parameters; it's just the data representation. Learning happens in the hidden and output layers.

---

## Step 2: Hidden Layer

### Layer Purpose

The hidden layer transforms the 2D input space into a 4D representation where XOR becomes linearly separable. Each hidden neuron learns to detect a specific pattern in the inputs.

### Mathematical Definition

**Forward pass:**
```
h = σ(x × W₁ + b₁)
```

Where:
- `x`: Input vector of shape (2,)
- `W₁`: Weight matrix of shape (2 × 4)
- `b₁`: Bias vector of shape (4,)
- `σ`: Sigmoid activation function
- `h`: Hidden activations of shape (4,)

**Expanded form:**
```
For each hidden neuron j (j = 1,2,3,4):
  z_j = x₁ · W₁[1,j] + x₂ · W₁[2,j] + b₁[j]
  h_j = σ(z_j) = 1 / (1 + exp(-z_j))
```

### Sigmoid Activation Function

**Function:**
```
σ(z) = 1 / (1 + exp(-z))
```

**Properties:**
- **Range:** (0, 1) — always outputs values between 0 and 1
- **Smooth:** Differentiable everywhere
- **Squashing:** Large positive values → ~1, large negative values → ~0
- **Centered:** σ(0) = 0.5

**Derivative (needed for backpropagation):**
```
σ'(z) = σ(z) × (1 - σ(z))
```

**Implementation (from `mlp_simple.rs`):**
```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)  // Where x = sigmoid(z)
}
```

**Why sigmoid for XOR?**
- Historically popular for binary classification
- Smooth gradients prevent "dead neurons"
- Output range (0,1) matches our XOR labels

**Modern alternative:** ReLU (used in MNIST tutorial) often trains faster, but sigmoid works well for tiny networks.

### Weight Initialization

**Xavier (Glorot) initialization:**
```
W₁ ~ Uniform(-limit, +limit)
where limit = sqrt(6 / (input_size + output_size))
```

For our hidden layer:
```
limit = sqrt(6 / (2 + 4)) = sqrt(1) = 1.0
W₁ ~ Uniform(-1.0, +1.0)
```

**Why Xavier?**
- Prevents gradients from vanishing or exploding
- Maintains variance of activations across layers
- Helps training start smoothly

**Implementation (from `mlp_simple.rs`):**
```rust
let limit = (6.0 / (input_size + output_size) as f32).sqrt();
for w in &mut weights {
    *w = rng.gen_range_f32(-limit, limit);
}
```

Biases are initialized to zero:
```rust
biases: vec![0.0; output_size],
```

### Implementation Details

**Dense layer structure:**
```rust
pub struct DenseLayer {
    input_size: usize,      // 2 for hidden layer
    output_size: usize,     // 4 for hidden layer
    weights: Vec<f32>,      // 2×4 = 8 parameters
    biases: Vec<f32>,       // 4 parameters
    grad_weights: RefCell<Vec<f32>>,  // Stores ∂L/∂W
    grad_biases: RefCell<Vec<f32>>,   // Stores ∂L/∂b
}
```

**Forward pass implementation:**
```rust
fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
    for b in 0..batch_size {
        for j in 0..self.output_size {
            let mut sum = self.biases[j];
            for i in 0..self.input_size {
                sum += input[i] * self.weights[i * self.output_size + j];
            }
            output[j] = sum;
        }
    }
}

fn forward_with_sigmoid(layer: &DenseLayer, inputs: &[f32], outputs: &mut [f32]) {
    layer.forward(inputs, outputs, 1);  // Linear transformation
    for output in outputs.iter_mut() {
        *output = sigmoid(*output);      // Apply activation
    }
}
```

### Dimension Tracking

**Step-by-step dimensions:**
1. Input: `x` of shape `(2,)`
2. Matrix multiplication: `x × W₁` → `(2,) × (2×4) = (4,)`
3. Bias addition: `(4,) + (4,) = (4,)`
4. Sigmoid activation: `σ((4,)) = (4,)`
5. Hidden output: `h` of shape `(4,)`

**Memory layout:**
- Weights stored in row-major order: `[W₁[0,0], W₁[0,1], W₁[0,2], W₁[0,3], W₁[1,0], ...]`
- Index calculation: `weights[i * output_size + j]` for W₁[i,j]

---

## Step 3: Output Layer

### Layer Purpose

The output layer combines the 4 hidden activations into a single prediction for XOR. It learns which combinations of hidden features predict 0 vs 1.

### Mathematical Definition

**Forward pass:**
```
y = σ(h × W₂ + b₂)
```

Where:
- `h`: Hidden activations of shape (4,)
- `W₂`: Weight matrix of shape (4 × 1)
- `b₂`: Bias scalar (shape (1,))
- `σ`: Sigmoid activation
- `y`: Output prediction of shape (1,)

**Expanded form:**
```
z = h₁·W₂[1] + h₂·W₂[2] + h₃·W₂[3] + h₄·W₂[4] + b₂
y = σ(z)
```

### Implementation

**Output layer creation:**
```rust
let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);
// Creates: 4×1 weight matrix + 1 bias = 5 parameters
```

**Forward pass (same as hidden layer, but with 4 inputs → 1 output):**
```rust
forward_with_sigmoid(&output_layer, &hidden_outputs, &mut output_outputs);
```

### Dimension Tracking

**Step-by-step dimensions:**
1. Hidden input: `h` of shape `(4,)`
2. Matrix multiplication: `h × W₂` → `(4,) × (4×1) = (1,)`
3. Bias addition: `(1,) + (1,) = (1,)`
4. Sigmoid activation: `σ((1,)) = (1,)`
5. Final output: `y` of shape `(1,)` — single prediction between 0 and 1

### Output Interpretation

The output `y` is interpreted as:
- `y ≈ 0.0`: Network predicts XOR = 0
- `y ≈ 1.0`: Network predicts XOR = 1
- `y ≈ 0.5`: Network is uncertain (happens early in training)

**Decision threshold (commonly 0.5):**
```
predicted_class = (y > 0.5) ? 1 : 0
```

---

## Forward Pass Walkthrough

Let's trace a complete forward pass with actual numbers for input `[1.0, 0.0]` (which should output ~1.0).

### Example: Forward Pass for XOR(1,0)

**Initial state (after random initialization):**
Suppose our network initialized with these weights (example values):

```
W₁ (2×4):          b₁ (4):
┌                ┐  ┌      ┐
│  0.5  -0.3  0.2  0.7│  │  0.0 │
│ -0.4   0.6 -0.1  0.3│  │  0.0 │
└                ┘  │  0.0 │
                    │  0.0 │
                    └      ┘

W₂ (4×1):    b₂ (1):
┌      ┐    ┌     ┐
│  0.8 │    │ 0.0 │
│ -0.5 │    └     ┘
│  0.3 │
│  0.4 │
└      ┘
```

### Step 1: Input
```
x = [1.0, 0.0]
```

### Step 2: Hidden Layer Computation

**Linear transformation (z = x × W₁ + b₁):**
```
For h₁: z₁ = 1.0 × 0.5 + 0.0 × (-0.4) + 0.0 = 0.5
For h₂: z₂ = 1.0 × (-0.3) + 0.0 × 0.6 + 0.0 = -0.3
For h₃: z₃ = 1.0 × 0.2 + 0.0 × (-0.1) + 0.0 = 0.2
For h₄: z₄ = 1.0 × 0.7 + 0.0 × 0.3 + 0.0 = 0.7

z = [0.5, -0.3, 0.2, 0.7]
```

**Apply sigmoid activation:**
```
h₁ = σ(0.5) = 1/(1+e^(-0.5)) ≈ 0.622
h₂ = σ(-0.3) = 1/(1+e^(0.3)) ≈ 0.426
h₃ = σ(0.2) = 1/(1+e^(-0.2)) ≈ 0.550
h₄ = σ(0.7) = 1/(1+e^(-0.7)) ≈ 0.668

h = [0.622, 0.426, 0.550, 0.668]
```

### Step 3: Output Layer Computation

**Linear transformation (z_out = h × W₂ + b₂):**
```
z_out = 0.622 × 0.8 + 0.426 × (-0.5) + 0.550 × 0.3 + 0.668 × 0.4 + 0.0
      = 0.498 - 0.213 + 0.165 + 0.267
      = 0.717
```

**Apply sigmoid activation:**
```
y = σ(0.717) = 1/(1+e^(-0.717)) ≈ 0.672
```

### Step 4: Compare to Expected Output

```
Expected:  1.0
Predicted: 0.672
Error:     1.0 - 0.672 = 0.328
```

**Interpretation:** The network's first prediction is off by 0.328. Training will adjust weights to reduce this error!

### Verification Checkpoint ✓

**Forward pass dimensions:**
- Input: `(2,)` ✓
- Hidden: `(2,) × (2×4) + (4,) → (4,)` ✓
- Output: `(4,) × (4×1) + (1,) → (1,)` ✓

**Expected behavior:**
- All hidden activations between 0 and 1 (sigmoid range) ✓
- Final output between 0 and 1 ✓
- Initial predictions are random (not accurate yet) ✓

---

## Backward Pass Walkthrough

Now let's trace how gradients flow backward to update weights. This is the **backpropagation** algorithm.

### Loss Function

We use **Mean Squared Error (MSE)** for this example:

```
L = (1/2) × (expected - predicted)²
```

For our example:
```
L = (1/2) × (1.0 - 0.672)² = (1/2) × 0.108 ≈ 0.054
```

**Why 1/2?** The factor of 1/2 cancels when we take the derivative, making the math cleaner.

### Gradient Flow Overview

```
Loss L
  ↓
∂L/∂y        (gradient w.r.t. output)
  ↓
∂L/∂W₂, ∂L/∂b₂   (output layer gradients)
  ↓
∂L/∂h        (gradient w.r.t. hidden)
  ↓
∂L/∂W₁, ∂L/∂b₁   (hidden layer gradients)
```

### Step 1: Output Layer Gradient

**Gradient of loss w.r.t. output prediction:**

The gradient depends on our loss function. For MSE:
```
∂L/∂y = ∂/∂y [(1/2)(expected - y)²]
      = -(expected - y)
      = y - expected
```

For our example:
```
∂L/∂y = 0.672 - 1.0 = -0.328
```

**But wait!** We need to account for the sigmoid activation. Using the chain rule:

```
∂L/∂z_out = ∂L/∂y × ∂y/∂z_out
          = ∂L/∂y × σ'(z_out)
          = ∂L/∂y × y × (1 - y)
```

For our example:
```
∂L/∂z_out = -0.328 × 0.672 × (1 - 0.672)
          = -0.328 × 0.672 × 0.328
          = -0.072
```

**Implementation (from `mlp_simple.rs`):**
```rust
// Compute gradient for output layer through sigmoid activation
for (i, (&error, &output)) in errors.iter().zip(output_outputs.iter()).enumerate() {
    grad_output[i] = -error * sigmoid_derivative(output);
}
// Note: error = expected - predicted, so we negate it for gradient descent
```

### Step 2: Output Layer Weight Gradients

**Weight gradients:**
```
∂L/∂W₂ = h^T × ∂L/∂z_out
```

For our example (∂L/∂z_out = -0.072, h = [0.622, 0.426, 0.550, 0.668]):
```
∂L/∂W₂[0] = 0.622 × (-0.072) = -0.045
∂L/∂W₂[1] = 0.426 × (-0.072) = -0.031
∂L/∂W₂[2] = 0.550 × (-0.072) = -0.040
∂L/∂W₂[3] = 0.668 × (-0.072) = -0.048
```

**Bias gradient:**
```
∂L/∂b₂ = ∂L/∂z_out = -0.072
```

**Implementation:**
```rust
// Backward pass through output layer
output_layer.backward(&hidden_outputs, &grad_output, &mut grad_hidden_outputs, 1);
```

Inside `backward()`:
```rust
for j in 0..self.output_size {
    let g = grad_output[j];
    grad_b[j] += g;  // Bias gradient

    for i in 0..self.input_size {
        grad_w[i * self.output_size + j] += input[i] * g;  // Weight gradient
    }
}
```

### Step 3: Gradient w.r.t. Hidden Activations

To backpropagate to the hidden layer, we need:
```
∂L/∂h = ∂L/∂z_out × W₂^T
```

For our example:
```
∂L/∂h₁ = -0.072 × 0.8 = -0.058
∂L/∂h₂ = -0.072 × (-0.5) = 0.036
∂L/∂h₃ = -0.072 × 0.3 = -0.022
∂L/∂h₄ = -0.072 × 0.4 = -0.029
```

**Critical step:** Apply sigmoid derivative for hidden layer activation!
```
∂L/∂z_h = ∂L/∂h ⊙ σ'(h)
        = ∂L/∂h ⊙ h ⊙ (1 - h)
```

For our example:
```
∂L/∂z_h₁ = -0.058 × 0.622 × 0.378 = -0.014
∂L/∂z_h₂ = 0.036 × 0.426 × 0.574 = 0.009
∂L/∂z_h₃ = -0.022 × 0.550 × 0.450 = -0.005
∂L/∂z_h₄ = -0.029 × 0.668 × 0.332 = -0.006
```

**Implementation:**
```rust
// Backpropagate through output layer
output_layer.backward(&hidden_outputs, &grad_output, &mut grad_hidden_outputs, 1);

// CRITICAL: Apply sigmoid derivative for hidden layer activation
for i in 0..NUM_HIDDEN {
    grad_hidden_outputs[i] *= sigmoid_derivative(hidden_outputs[i]);
}
```

### Step 4: Hidden Layer Weight Gradients

**Weight gradients:**
```
∂L/∂W₁ = x^T × ∂L/∂z_h
```

For our example (x = [1.0, 0.0], ∂L/∂z_h = [-0.014, 0.009, -0.005, -0.006]):
```
∂L/∂W₁[0,0] = 1.0 × (-0.014) = -0.014
∂L/∂W₁[0,1] = 1.0 × 0.009 = 0.009
∂L/∂W₁[0,2] = 1.0 × (-0.005) = -0.005
∂L/∂W₁[0,3] = 1.0 × (-0.006) = -0.006

∂L/∂W₁[1,0] = 0.0 × (-0.014) = 0.0
∂L/∂W₁[1,1] = 0.0 × 0.009 = 0.0
∂L/∂W₁[1,2] = 0.0 × (-0.005) = 0.0
∂L/∂W₁[1,3] = 0.0 × (-0.006) = 0.0
```

**Bias gradients:**
```
∂L/∂b₁ = ∂L/∂z_h = [-0.014, 0.009, -0.005, -0.006]
```

**Implementation:**
```rust
// Backpropagate through hidden layer
hidden_layer.backward(&inputs[sample], &grad_hidden_outputs, &mut grad_hidden_input, 1);
```

### Step 5: Parameter Updates

**Gradient descent update rule:**
```
W_new = W_old - learning_rate × ∂L/∂W
b_new = b_old - learning_rate × ∂L/∂b
```

With learning_rate = 0.01:
```
W₂[0]_new = 0.8 - 0.01 × (-0.045) = 0.8 + 0.00045 = 0.80045
W₂[1]_new = -0.5 - 0.01 × (-0.031) = -0.5 + 0.00031 = -0.49969
...
```

**Implementation:**
```rust
output_layer.update_parameters(learning_rate);
hidden_layer.update_parameters(learning_rate);
```

Inside `update_parameters()`:
```rust
for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
    *w -= learning_rate * g;
}
for (b, g) in self.biases.iter_mut().zip(grad_b.iter()) {
    *b -= learning_rate * g;
}

// Reset gradients to zero for next iteration
grad_w.fill(0.0);
grad_b.fill(0.0);
```

### Chain Rule Summary

The key to backpropagation is the **chain rule**:

```
∂L/∂W = (∂L/∂output) × (∂output/∂activation) × (∂activation/∂W)
```

**For our network:**
1. **Output layer:** ∂L/∂W₂ = ∂L/∂y × σ'(z_out) × h
2. **Hidden layer:** ∂L/∂W₁ = ∂L/∂y × σ'(z_out) × W₂ × σ'(z_h) × x

Each layer passes gradients backward, multiplying by its local derivatives.

### Verification Checkpoint ✓

**Gradient dimensions:**
- ∂L/∂W₂: `(4×1)` matches W₂ ✓
- ∂L/∂b₂: `(1,)` matches b₂ ✓
- ∂L/∂W₁: `(2×4)` matches W₁ ✓
- ∂L/∂b₁: `(4,)` matches b₁ ✓

**Gradient flow:**
- Gradients propagate from output to input ✓
- Each layer applies activation derivative ✓
- Parameters updated in direction that reduces loss ✓

---

## Training the Network

### Training Loop Structure

**High-level algorithm:**
```
1. Initialize network with random weights
2. For each epoch (1 to 1,000,000):
   a. For each training sample:
      - Forward pass: compute prediction
      - Compute error: expected - predicted
      - Backward pass: compute gradients
      - Update parameters: W -= lr × ∂L/∂W
   b. Every 1000 epochs: print average loss
3. Test final network on all samples
```

**Implementation (from `mlp_simple.rs`):**
```rust
fn train(
    nn: &mut NeuralNetwork,
    inputs: &[[f32; NUM_INPUTS]],
    expected_outputs: &[[f32; NUM_OUTPUTS]],
    scheduler: &mut dyn LRScheduler,
    epochs: usize,
) {
    // Pre-allocate buffers
    let mut hidden_outputs = vec![0.0f32; NUM_HIDDEN];
    let mut output_outputs = vec![0.0f32; NUM_OUTPUTS];
    let mut errors = [0.0f32; NUM_OUTPUTS];
    let mut grad_output = vec![0.0f32; NUM_OUTPUTS];
    let mut grad_hidden_outputs = vec![0.0f32; NUM_HIDDEN];
    let mut grad_hidden_input = vec![0.0f32; NUM_INPUTS];

    for epoch in 0..epochs {
        let mut total_errors = 0.0f32;
        let current_lr = scheduler.get_lr();

        for sample in 0..NUM_SAMPLES {
            // Clear buffers
            hidden_outputs.fill(0.0);
            output_outputs.fill(0.0);

            // Forward pass
            forward_with_sigmoid(&nn.hidden_layer, &inputs[sample], &mut hidden_outputs);
            forward_with_sigmoid(&nn.output_layer, &hidden_outputs, &mut output_outputs);

            // Compute error
            for i in 0..NUM_OUTPUTS {
                errors[i] = expected_outputs[sample][i] - output_outputs[i];
                total_errors += errors[i] * errors[i];
            }

            // Backward pass (gradients computed in detail above)
            for (i, (&error, &output)) in errors.iter().zip(output_outputs.iter()).enumerate() {
                grad_output[i] = -error * sigmoid_derivative(output);
            }

            nn.output_layer.backward(&hidden_outputs, &grad_output, &mut grad_hidden_outputs, 1);

            for i in 0..NUM_HIDDEN {
                grad_hidden_outputs[i] *= sigmoid_derivative(hidden_outputs[i]);
            }

            nn.hidden_layer.backward(&inputs[sample], &grad_hidden_outputs, &mut grad_hidden_input, 1);

            // Update parameters
            nn.output_layer.update_parameters(current_lr);
            nn.hidden_layer.update_parameters(current_lr);
        }

        // Print progress
        let loss = total_errors / NUM_SAMPLES as f32;
        if (epoch + 1) % 1000 == 0 {
            println!("Epoch {}, Error: {:.6}", epoch + 1, loss);
        }

        // Step learning rate scheduler
        scheduler.step();
    }
}
```

### Hyperparameters

**Default configuration (from `config/training/mlp_simple_default.json`):**
```json
{
  "learning_rate": 0.01,
  "epochs": 1000000,
  "batch_size": 1,
  "scheduler_type": "constant",
  "activation_function": "sigmoid"
}
```

**Learning rate:** 0.01
- **Too high (>0.1):** Training becomes unstable, loss oscillates
- **Too low (<0.001):** Training is very slow, may not converge in 1M epochs
- **Just right (0.01):** Steady convergence, reaches <0.001 error

**Epochs:** 1,000,000
- **Why so many?** XOR is small but non-trivial; with batch_size=1, we need many iterations
- **Modern alternative:** Use batch training and Adam optimizer (see Exercises)

**Batch size:** 1
- We update weights after **each sample** (online learning)
- Alternative: Update after all 4 samples (batch learning, faster convergence)

### Learning Rate Scheduling

The project supports several learning rate schedulers:

**Constant (default):**
```
lr(epoch) = 0.01  (never changes)
```

**Step decay:**
```
lr(epoch) = initial_lr × gamma^(epoch / step_size)
```

**Exponential decay:**
```
lr(epoch) = initial_lr × decay_rate^epoch
```

**Cosine annealing:**
```
lr(epoch) = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × epoch / T_max))
```

**To experiment with schedulers:**
```bash
cargo run --release --bin mlp_simple -- config/mnist_mlp_cosine.json
```

---

## Verification and Expected Outputs

### Expected Training Progress

**Early training (Epoch 1-1000):**
```
Epoch 1000, Error: 0.249832
```
- Network is learning, error decreasing from ~0.5 (random)
- Predictions still noisy: [0.3, 0.6, 0.5, 0.4]

**Mid training (Epoch 100,000):**
```
Epoch 100000, Error: 0.023156
```
- Clear XOR pattern emerging
- Predictions closer: [0.15, 0.87, 0.89, 0.12]

**Late training (Epoch 500,000+):**
```
Epoch 500000, Error: 0.001234
Epoch 1000000, Error: 0.000456
```
- Very accurate predictions
- Error asymptotically approaching 0

### Final Testing Output

**Expected test results (from `mlp_simple.rs`):**
```
Testing the trained network:
Input: 0.0, 0.0, Expected Output: 0.0, Predicted Output: 0.012
Input: 0.0, 1.0, Expected Output: 1.0, Predicted Output: 0.989
Input: 1.0, 0.0, Expected Output: 1.0, Predicted Output: 0.991
Input: 1.0, 1.0, Expected Output: 0.0, Predicted Output: 0.009
```

**Interpretation:**
- **[0,0] → 0.012:** Network correctly outputs ~0 (threshold: <0.5)
- **[0,1] → 0.989:** Network correctly outputs ~1 (threshold: >0.5)
- **[1,0] → 0.991:** Network correctly outputs ~1
- **[1,1] → 0.009:** Network correctly outputs ~0

**100% accuracy!** All four XOR cases classified correctly.

### Running the Implementation

**Basic usage:**
```bash
cargo run --release --bin mlp_simple
```

**With custom configuration:**
```bash
cargo run --release --bin mlp_simple -- config/mnist_mlp_adam.json
```

**Expected runtime:** ~5-15 seconds for 1M epochs (depends on CPU)

### Debugging Checklist

**If training doesn't converge:**

1. ✓ **Check sigmoid implementation:** Test `sigmoid(0) ≈ 0.5`
2. ✓ **Verify gradient computation:** Print gradients, ensure they're not zero or NaN
3. ✓ **Confirm weight updates:** Weights should change after each iteration
4. ✓ **Check learning rate:** Try 0.01 (default), not 0.1 or 0.0001
5. ✓ **Verify activation derivatives:** Hidden layer needs sigmoid_derivative applied!

**Common mistakes:**
- **Forgetting activation derivative in backward pass** → Gradients wrong, no learning
- **Wrong matrix dimensions** → Crashes or incorrect computations
- **Learning rate too high** → Loss oscillates or explodes to NaN
- **Not resetting gradients** → Gradients accumulate incorrectly across samples

### Verification Checkpoint ✓

**Training completed successfully if:**
- Final error < 0.01 ✓
- All four XOR cases have |predicted - expected| < 0.05 ✓
- Training log shows steady error decrease ✓
- No NaN or Inf values in outputs ✓

---

## Exercises

### Beginner Level

**Exercise 1: Modify number of hidden neurons**
- Change `NUM_HIDDEN` from 4 to 2
- Run training and observe convergence speed
- **Expected:** Slower convergence but still works (minimum is 2 for XOR)

**Exercise 2: Experiment with learning rates**
- Try learning_rate = 0.001, 0.01, 0.1, 1.0
- Observe training curves
- **Expected:** 0.01 is optimal; 0.001 too slow; 0.1+ unstable

**Exercise 3: Change epochs**
- Run with epochs = 10,000 (too few)
- Run with epochs = 10,000,000 (overkill)
- **Expected:** 10K → underfitting; 10M → perfect but wasteful

### Intermediate Level

**Exercise 4: Implement batch training**
- Modify train loop to accumulate gradients over all 4 samples
- Update weights once per epoch (not once per sample)
- **Expected:** 4× fewer parameter updates, faster convergence

**Exercise 5: Add momentum**
- Implement SGD with momentum: `v = beta × v + lr × grad; W -= v`
- Try beta = 0.9
- **Expected:** Smoother convergence, fewer oscillations

**Exercise 6: Visualize decision boundary**
- Create 100×100 grid of [x₁, x₂] points in [0,1]×[0,1]
- Run forward pass on each point
- Plot output as heatmap
- **Expected:** Four quadrants with clear XOR pattern

### Advanced Level

**Exercise 7: Change activation functions**
- Replace sigmoid with ReLU in hidden layer
- Adjust learning rate (ReLU often needs lower LR)
- **Expected:** Faster training but potential "dead neurons"

**Exercise 8: Numerical gradient checking**
- Compute ∂L/∂W numerically: (L(W+ε) - L(W-ε)) / (2ε)
- Compare to backprop gradients
- **Expected:** Should match to ~1e-4 precision

**Exercise 9: Implement Adam optimizer**
- Add adaptive learning rates per parameter
- Use beta1=0.9, beta2=0.999, epsilon=1e-8
- **Expected:** Much faster convergence (10K epochs instead of 1M)

**Exercise 10: Extend to 3-bit parity**
- Input: 3 bits, Output: 1 if odd number of 1s
- 8 training samples: [0,0,0]→0, [0,0,1]→1, ..., [1,1,1]→1
- **Expected:** Need more hidden neurons (8-16), takes longer to converge

---

## Next Steps

### What You've Learned

✓ **Forward propagation:** Data flows through layers via matrix operations and activations
✓ **Backpropagation:** Gradients flow backward via the chain rule
✓ **Gradient descent:** Parameters update in the direction that reduces loss
✓ **Non-linear activations:** Sigmoid enables learning non-linear patterns like XOR
✓ **Training loop:** Iterate forward/backward passes until convergence

### Ready for More?

**Next tutorial: [Tutorial 02: MNIST MLP](02_mnist_mlp.md)**

In the next tutorial, you'll scale to real-world data and learn:
- Scaling to 60,000 training images (vs 4 XOR samples!)
- Why ReLU replaces sigmoid for deeper networks
- Softmax for multi-class classification (10 digit classes)
- BLAS acceleration for fast matrix operations
- Minibatch training and validation splits
- Early stopping to prevent overfitting

**Key differences from XOR:**
- Input: 784 features (28×28 pixels) vs 2
- Hidden layer: 512 neurons vs 4
- Output: 10 classes (digits 0-9) vs 1 binary output
- Training: 60K samples vs 4
- Accuracy metric: Classification accuracy on 10K test set

---

## Related Documentation

**Mathematical foundations:**
- [Backpropagation Guide](../backpropagation/README.md) - Mathematical theory for gradient computation
- [Dense Layer Backpropagation](../backpropagation/dense_layer.md) - Detailed gradient derivations
- [Activation Functions](../activation_functions.md) - ReLU, sigmoid, tanh, and modern alternatives
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) - Notation reference

**Implementation details:**
- `mlp_simple.rs` - Full XOR implementation (this tutorial's code)
- `src/layers/trait.rs` - Layer trait interface
- `src/layers/dense.rs` - Dense layer with BLAS (used in MNIST)
- `src/utils/activations.rs` - Activation function implementations
- `tests/test_backward_pass.rs` - Gradient correctness tests
- `tests/test_gradient_checking.rs` - Numerical gradient validation

**Training infrastructure:**
- [Hyperparameters Guide](../hyperparameters.md) - Learning rate, batch size, optimizer selection
- [Configuration System](../architecture_config.md) - JSON-based hyperparameter configs
- `config/training/mlp_simple_default.json` - Default configuration for XOR

**Related tutorials:**
- [Tutorial 02: MNIST MLP](02_mnist_mlp.md) - Scale to real data with 784→512→10 classifier
- [Tutorial 03: MNIST CNN](03_mnist_cnn.md) - Add spatial structure with convolutional layers

---

## Congratulations!

You've built your first neural network from scratch and understand the core principles of deep learning. Everything else (CNNs, RNNs, Transformers) builds on these same fundamentals:

1. **Forward pass:** Compute predictions
2. **Loss function:** Measure prediction error
3. **Backward pass:** Compute gradients via chain rule
4. **Parameter update:** Adjust weights to reduce loss
5. **Iterate:** Repeat until convergence

**Keep experimenting!** Try the exercises, modify the code, and when you're ready, continue to [Tutorial 02: MNIST MLP](02_mnist_mlp.md) to tackle real-world handwritten digit classification. Happy learning!

---

**Navigation:**
← [Tutorial Index](README.md) | [Next Tutorial: MNIST MLP →](02_mnist_mlp.md)
