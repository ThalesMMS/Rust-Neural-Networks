# Tutorial 05: Automatic Differentiation Engine

**Level:** Expert
**Time:** 90-120 minutes
**Prerequisites:** Tutorial 01 (XOR MLP), chain rule, basic graph theory
**Implementation:** See `src/autograd/` for complete working code

**Navigation:**
← [Tutorial 04: RNN/LSTM Character-Level Model](04_rnn_lstm_char_level.md) | [Tutorial Index](README.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is Autograd and Why Does It Matter?](#what-is-autograd-and-why-does-it-matter)
3. [Computational Graphs: Building a DAG of Tensors](#computational-graphs-building-a-dag-of-tensors)
4. [Reverse-Mode Automatic Differentiation](#reverse-mode-automatic-differentiation)
5. [Architecture Overview](#architecture-overview)
6. [Forward Pass: Building the Graph](#forward-pass-building-the-graph)
7. [Backward Pass: Gradient Propagation](#backward-pass-gradient-propagation)
8. [Worked Example: Tracing a·b + c](#worked-example-tracing-ab--c)
9. [MLP Example with the Autograd Engine](#mlp-example-with-the-autograd-engine)
10. [Comparison: Autograd vs Manual Gradients](#comparison-autograd-vs-manual-gradients)
11. [Verification Checkpoints](#verification-checkpoints)
12. [Exercises](#exercises)
13. [Next Steps](#next-steps)

---

## Introduction

Welcome to the most foundational tutorial in this series. So far, every tutorial has shown you how to write gradients by hand — deriving each layer's backward pass on paper, then translating those formulas into Rust code. This works, but it is:

- **Error-prone**: One wrong sign or missing factor breaks training silently
- **Tedious**: Every new operation requires re-deriving all gradient rules
- **Inflexible**: Experimenting with architectures requires re-deriving everything

**Automatic differentiation** (autograd) solves all three problems. Instead of you deriving gradients, the engine does it for you — automatically, correctly, for any computation you express.

By the end of this tutorial you will understand:
- How a computational graph records every operation your code performs
- How reverse-mode AD traverses that graph to compute exact gradients
- How our implementation in `src/autograd/` achieves this in pure Rust
- How to trace through a complete example step by step
- How to use the autograd engine for a simple MLP

**Implementation reference:** All code described here lives in `src/autograd/`:
- `src/autograd/tensor.rs` — The `Tensor` type and `backward()`
- `src/autograd/tape.rs` — `Op` enum, `GradNode`, topological sort
- `src/autograd/ops.rs` — Differentiable forward operations

---

## What is Autograd and Why Does It Matter?

### The Problem with Manual Gradients

In Tutorial 01 you derived the XOR backprop by hand:

```
∂L/∂W₂ = h⊤ × (output_error)
∂L/∂W₁ = x⊤ × (hidden_error)
```

This is correct — but only for that exact architecture. Change the activation function and you must re-derive. Add a residual connection and you must re-derive. Use a different loss function and you must re-derive.

**The insight:** Every neural network is just a composition of differentiable functions. If we record which functions were applied in which order, we can automatically apply the chain rule to compute gradients — no manual derivation needed.

### What Autograd Does

Autograd tracks every mathematical operation your code performs on tensors. When you call `backward()` on a loss value, it:

1. **Traverses** the recorded operation history in reverse order
2. **Applies** the chain rule at each step
3. **Accumulates** gradients into every input tensor that participated

The result: every tensor that contributed to the loss gets the exact gradient of the loss with respect to its value.

### Two Modes of Automatic Differentiation

There are two main approaches:

| Mode | Also Called | Best For | How It Works |
|------|-------------|----------|--------------|
| **Forward-mode** | Tangent mode | Many outputs, few inputs | Propagates derivatives forward alongside values |
| **Reverse-mode** | Adjoint mode | Few outputs (scalar loss), many inputs | Records forward pass, propagates backward |

Neural networks have **millions of parameters (inputs)** and **one scalar loss (output)**.
Reverse-mode AD computes all parameter gradients in a single backward pass — this is why it is universally used in deep learning. Our implementation uses reverse-mode AD exclusively.

---

## Computational Graphs: Building a DAG of Tensors

### What is a Computational Graph?

A **computational graph** is a directed acyclic graph (DAG) where:
- **Nodes** are tensors (intermediate values and leaf parameters)
- **Edges** point from inputs to outputs of each operation
- **Leaf nodes** are tensors you created directly (weights, inputs, biases)
- **Non-leaf nodes** are tensors produced by operations (results of add, matmul, relu, etc.)

### Example Graph: a·b + c

Consider the computation `out = a·b + c` with:
- `a = 3.0`, `b = 4.0`, `c = 2.0`

```
Computational Graph (Forward Direction):

   a ──┐
       ├──[Mul]──→ d ──┐
   b ──┘               ├──[Add]──→ out
                   c ──┘

Leaves:  a, b, c    (no creating operation, require_grad = true)
d:       a * b      (recorded Op::Mul with inputs [a, b])
out:     d + c      (recorded Op::Add with inputs [d, c])
```

Every arrow records: "this input contributed to this output". The backward pass follows arrows in **reverse** to propagate gradients.

### How Nodes Are Recorded

In our implementation, each non-leaf tensor stores a `GradNode` that records:
1. **Which operation** produced it (`Op::Add`, `Op::Mul`, `Op::ReLU`, etc.)
2. **Which input tensors** were consumed by that operation

```rust
pub struct GradNode {
    pub op: Op,         // The operation that created this tensor
    pub inputs: Vec<Tensor>,  // The input tensors (cheap Rc clones)
}
```

Leaf tensors (created by `Tensor::from_vec(...)`) have `grad_node = None`.

### The Op Enum

Our engine supports these differentiable operations (defined in `src/autograd/tape.rs`):

```rust
pub enum Op {
    Add,              // Element-wise: out[i] = a[i] + b[i]
    Sub,              // Element-wise: out[i] = a[i] - b[i]
    Mul,              // Element-wise: out[i] = a[i] * b[i]
    MatMul { m, k, n }, // Matrix mult: A(m,k) @ B(k,n) = out(m,n)
    ReLU,             // max(0, a[i])
    Sigmoid,          // 1 / (1 + exp(-a[i]))
    Tanh,             // tanh(a[i])
    SoftmaxCE,        // Fused softmax + cross-entropy loss
    MSE,              // Mean squared error loss
    Sum,              // Reduce all elements to scalar
    Mean,             // Average all elements to scalar
}
```

---

## Reverse-Mode Automatic Differentiation

### The Chain Rule in Graph Form

For a scalar loss `L` and a tensor `a` somewhere in the computation, the chain rule says:

```
∂L/∂a = Σ (∂L/∂out) · (∂out/∂a)   for all outputs `out` that depend on `a`
```

Reverse-mode AD computes this by:
1. Starting with the seed: `∂L/∂L = 1`
2. For each operation in **reverse** topological order, computing how the gradient flows backward through it
3. Accumulating gradients at each input tensor

### Topological Ordering

Before we can traverse the graph backward, we need the correct order. A **topological ordering** of a DAG lists nodes so that every node appears after all its dependencies.

For backward propagation, we need **reverse** topological order: the loss first, leaves last.

```
Forward order (dependency order):  a, b, c → d → out
Reverse topological order:         out → d → a, b, c
                                   (loss first, leaves last)
```

Our `GradNode::build_topo()` implements this with a DFS post-order traversal followed by a reverse:

```
DFS post-order from `out`:
  1. Visit a (leaf)     → append a
  2. Visit b (leaf)     → append b
  3. Visit d (a*b done) → append d
  4. Visit c (leaf)     → append c
  5. Visit out (done)   → append out

Post-order result: [a, b, d, c, out]
After reverse():   [out, c, d, b, a]  ← correct backward order!
```

### Gradient Accumulation at Shared Nodes

A key feature of reverse-mode AD: when a tensor is used by **multiple operations**, gradients from all uses must be **summed**. For example, if a weight `w` appears in two layers:

```
     w
    / \
  op1  op2
    \  /
    loss
```

Both `op1` and `op2` contribute gradients to `w`, so we **accumulate** (add) them:
```
∂L/∂w = (∂L/∂op1_output)(∂op1_output/∂w) + (∂L/∂op2_output)(∂op2_output/∂w)
```

Our `accumulate_grad()` method handles this by adding incoming gradients to the existing buffer rather than overwriting it.

---

## Architecture Overview

### Module Structure

```
src/autograd/
├── mod.rs        — Public re-exports
├── tensor.rs     — Tensor type (data + grad storage + backward())
├── tape.rs       — Op enum + GradNode + topological sort
└── ops.rs        — Differentiable forward ops + backward_op dispatcher
```

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          FORWARD PASS                           │
│                                                                 │
│  Tensor::from_vec(data, shape, requires_grad=true)              │
│           │                                                     │
│           ▼                                                     │
│  tensor_matmul(&x, &w)    ← Records Op::MatMul in out.grad_node│
│           │                                                     │
│           ▼                                                     │
│  tensor_relu(&z)           ← Records Op::ReLU in out.grad_node │
│           │                                                     │
│           ▼                                                     │
│  tensor_mse_loss(&pred, &tgt) ← Records Op::MSE in loss.grad_node│
│           │                                                     │
│           ▼                                                     │
│  loss: Tensor (shape 1×1, scalar)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          BACKWARD PASS                          │
│                                                                 │
│  loss.backward()                                                │
│           │                                                     │
│           ▼                                                     │
│  GradNode::build_topo(&loss) → [loss, pred, relu_out, z, w, x] │
│           │                                                     │
│           ▼                                                     │
│  For each node in topo order:                                   │
│    upstream_grad ← node.grad()                                  │
│    backward_op(&node.op, &node.inputs, &upstream_grad)          │
│           │                                                     │
│           ▼                                                     │
│  w.grad() now holds ∂L/∂w  (use for parameter update)          │
└─────────────────────────────────────────────────────────────────┘
```

### Tensor Inner Structure

Each `Tensor` is a thin wrapper over `Rc<RefCell<TensorInner>>`:

```
TensorInner {
    data:         Vec<f32>,          // Flat row-major values
    grad:         Option<Vec<f32>>,  // Accumulated gradient (None until backward)
    shape:        (usize, usize),    // (rows, cols)
    requires_grad: bool,             // Track gradients for this tensor?
    grad_node:    Option<Box<GradNode>>, // None for leaves, Some for op results
}
```

The `Rc<RefCell<...>>` wrapper serves two purposes:
- **`Rc`**: Multiple tensors in the graph can share ownership of the same input (cheap pointer copy, not data copy)
- **`RefCell`**: Mutable interior access for accumulating gradients during backward

---

## Forward Pass: Building the Graph

### Step-by-Step Graph Construction

During the forward pass, each operation:
1. Reads the input data
2. Computes the output values
3. If any input has `requires_grad = true`, creates a `GradNode` and attaches it to the output

**Example: `tensor_add(a, b)`**

```rust
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    // Step 1: Compute forward values
    let out_data: Vec<f32> = a.data().iter().zip(b.data().iter())
        .map(|(x, y)| x + y)
        .collect();

    // Step 2: Propagate requires_grad flag
    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, shape_a, requires_grad);

    // Step 3: Record the operation for backward
    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(
            GradNode::new(Op::Add, vec![a.clone(), b.clone()])
            //                            ↑ Cheap Rc pointer copy, not data copy
        ));
    }
    out
}
```

**Key insight:** `a.clone()` and `b.clone()` inside `GradNode::new(...)` are cheap `Rc` pointer increments — they share the same underlying data, they do not copy the arrays.

### When Is a GradNode Not Recorded?

A `GradNode` is **only** recorded when at least one input has `requires_grad = true`. This is an important optimization:

```rust
// Pure data tensor (no grad tracking):
let x = Tensor::from_vec(input_data, shape, false);

// Weight tensor (gradient tracking enabled):
let w = Tensor::from_vec(weight_data, shape, true);

// Forward operation:
let out = tensor_matmul(&x, &w);
// out.requires_grad() == true (because w requires grad)
// out.grad_node == Some(Op::MatMul {...}, inputs=[x, w])
```

If BOTH inputs had `requires_grad = false`, the output would have no `grad_node` and would be skipped entirely during backward.

---

## Backward Pass: Gradient Propagation

### The `backward()` Method

Once you have computed a scalar loss tensor, call `loss.backward()`:

```rust
pub fn backward(&self) {
    // Step 1: Seed the backward pass with ∂L/∂L = 1
    let n = self.numel();
    self.0.borrow_mut().grad = Some(vec![1.0f32; n]);

    // Step 2: Topological sort to get backward traversal order
    let topo = GradNode::build_topo(self);

    // Step 3: Propagate gradients through each node
    for node in &topo {
        let upstream_grad = match node.grad() {
            Some(g) => g,
            None => continue,  // No gradient reached this node yet
        };

        let (op, inputs) = {
            let inner = node.0.borrow();
            match inner.grad_node.as_ref() {
                Some(gn) => (gn.op.clone(), gn.inputs.clone()),
                None => continue,  // Leaf tensor: no backward operation
            }
        };

        // Apply the chain rule for this specific operation
        backward_op(&op, &inputs, &upstream_grad);
    }
}
```

### The `backward_op` Dispatcher

`backward_op` applies the per-operation gradient rule (defined in `src/autograd/ops.rs`):

```
Op::Add:
    grad_a = upstream            (addition passes gradient through unchanged)
    grad_b = upstream            (same for both inputs)

Op::Mul:
    grad_a = upstream * b        (product rule: ∂(a·b)/∂a = b)
    grad_b = upstream * a        (product rule: ∂(a·b)/∂b = a)

Op::MatMul { m, k, n }:
    grad_A = upstream @ B.T      (shape: m×k)
    grad_B = A.T @ upstream      (shape: k×n)

Op::ReLU:
    grad_a = upstream * (a > 0)  (mask: 1 if active, 0 if clamped)

Op::Sigmoid:
    grad_a = upstream * s * (1-s)  where s = sigmoid(a)  (cached during forward)

Op::MSE:
    grad_pred = upstream * 2 * (pred - target) / n
```

### Gradient Accumulation

After `backward_op` computes the gradient for an input, it calls `accumulate_grad`:

```rust
pub fn accumulate_grad(&self, grad: &[f32]) {
    let mut inner = self.0.borrow_mut();
    // Allocate buffer on first use
    if inner.grad.is_none() {
        inner.grad = Some(vec![0.0f32; inner.shape.0 * inner.shape.1]);
    }
    // Add (not overwrite) the incoming gradient
    let g = inner.grad.as_mut().unwrap();
    for (acc, &dg) in g.iter_mut().zip(grad.iter()) {
        *acc += dg;
    }
}
```

The **add** operation (`*acc += dg`) is critical: it correctly handles the case where a tensor is used by multiple downstream operations — their gradients are summed, as required by the chain rule.

---

## Worked Example: Tracing a·b + c

Let us trace every computation step for `out = a·b + c` with:
- `a = 3.0`
- `b = 4.0`
- `c = 2.0`

All three are scalar tensors of shape `(1, 1)` with `requires_grad = true`.

### Step 1: Create Leaf Tensors

```rust
let a = Tensor::from_vec(vec![3.0], (1, 1), true);
let b = Tensor::from_vec(vec![4.0], (1, 1), true);
let c = Tensor::from_vec(vec![2.0], (1, 1), true);
```

**State:**
```
a: data=[3.0], grad=None, grad_node=None  (leaf)
b: data=[4.0], grad=None, grad_node=None  (leaf)
c: data=[2.0], grad=None, grad_node=None  (leaf)
```

### Step 2: Compute d = a · b (element-wise multiply)

```rust
let d = tensor_add(&a, &b);  // Wait — for element-wise Mul, we use:
// (Our engine provides tensor_mul_scalar; for element-wise tensor*tensor
//  the backward uses Op::Mul directly via GradNode)
```

Let us trace element-wise multiplication. The forward computation:
```
d.data = [a[0] * b[0]] = [3.0 * 4.0] = [12.0]
d.requires_grad = true (because a and b both require grad)
d.grad_node = GradNode { op: Op::Mul, inputs: [a, b] }
```

**State:**
```
d: data=[12.0], grad=None, grad_node=Op::Mul(inputs=[a,b])
```

**Computational graph so far:**
```
a(3.0) ──┐
          ├──[Mul]──→ d(12.0)
b(4.0) ──┘
```

### Step 3: Compute out = d + c

```rust
let out = tensor_add(&d, &c);
```

Forward computation:
```
out.data = [d[0] + c[0]] = [12.0 + 2.0] = [14.0]
out.requires_grad = true
out.grad_node = GradNode { op: Op::Add, inputs: [d, c] }
```

**Complete computational graph:**
```
a(3.0) ──┐
          ├──[Mul]──→ d(12.0) ──┐
b(4.0) ──┘                      ├──[Add]──→ out(14.0)
                    c(2.0)  ────┘
```

### Step 4: Call out.backward()

**Topological sort** — DFS post-order from `out`:
```
Visit a (leaf)  → topo = [a]
Visit b (leaf)  → topo = [a, b]
Visit d (Mul)   → topo = [a, b, d]
Visit c (leaf)  → topo = [a, b, d, c]
Visit out (Add) → topo = [a, b, d, c, out]
After reverse() → topo = [out, c, d, b, a]   ← backward order
```

**Seed:**
```
out.grad = [1.0]   (∂out/∂out = 1)
```

**Node 1: out — Op::Add with inputs=[d, c]**

Rule: `grad_a = upstream`, `grad_b = upstream`
```
upstream = out.grad = [1.0]

grad_d += upstream → d.grad = [1.0]
grad_c += upstream → c.grad = [1.0]
```

**Node 2: c — leaf (no grad_node)**
```
(Leaf tensor — skip, no backward operation)
```

**Node 3: d — Op::Mul with inputs=[a, b]**

Rule: `grad_a = upstream * b.data`, `grad_b = upstream * a.data`
```
upstream = d.grad = [1.0]
b.data   = [4.0]
a.data   = [3.0]

grad_a += upstream * b.data = 1.0 * 4.0 = 4.0 → a.grad = [4.0]
grad_b += upstream * a.data = 1.0 * 3.0 = 3.0 → b.grad = [3.0]
```

**Node 4: b — leaf (no grad_node)**
```
(Leaf tensor — skip, no backward operation)
```

**Node 5: a — leaf (no grad_node)**
```
(Leaf tensor — skip, no backward operation)
```

### Final Gradient Values

```
a.grad = [4.0]    ∂out/∂a = b = 4.0  ✓
b.grad = [3.0]    ∂out/∂b = a = 3.0  ✓
c.grad = [1.0]    ∂out/∂c = 1.0      ✓
d.grad = [1.0]    ∂out/∂d = 1.0      ✓ (intermediate — d = a*b)
```

**Verification by hand:**
```
out = a*b + c

∂out/∂a = ∂(a*b)/∂a + ∂c/∂a = b + 0 = 4.0  ✓
∂out/∂b = ∂(a*b)/∂b + ∂c/∂b = a + 0 = 3.0  ✓
∂out/∂c = ∂(a*b)/∂c + ∂c/∂c = 0 + 1 = 1.0  ✓
```

The autograd engine produces the exact same result as manual derivation.

---

## MLP Example with the Autograd Engine

### Architecture

We will build a simple 2→4→1 MLP (same as Tutorial 01's XOR network) using the autograd engine instead of manual gradients:

```
Input (2) → Linear (2→4) → ReLU → Linear (4→1) → MSE Loss
```

### Complete Autograd MLP Implementation

```rust
use rust_neural_networks::autograd::tensor::Tensor;
use rust_neural_networks::autograd::ops::{
    tensor_matmul, tensor_add_bias, tensor_relu, tensor_mse_loss,
};

// ─── Parameter initialization ─────────────────────────────────────────────
// Weights: requires_grad = true (we want to train them)
// Using small random values for initialization
let w1 = Tensor::from_vec(
    vec![0.5, -0.3, 0.2, 0.1, -0.4, 0.3, 0.2, -0.1],  // shape (2, 4)
    (2, 4),
    true,  // requires_grad = true
);
let b1 = Tensor::from_vec(vec![0.0; 4], (1, 4), true);

let w2 = Tensor::from_vec(
    vec![0.3, -0.2, 0.4, -0.3],  // shape (4, 1)
    (4, 1),
    true,
);
let b2 = Tensor::from_vec(vec![0.0], (1, 1), true);

// ─── Input data (XOR: [0,1] → 1.0) ─────────────────────────────────────
// Input does NOT require grad
let x      = Tensor::from_vec(vec![0.0, 1.0], (1, 2), false);
let target = Tensor::from_vec(vec![1.0],       (1, 1), false);

// ─── Forward pass ──────────────────────────────────────────────────────
// Hidden layer: z1 = x @ w1 + b1,  h1 = relu(z1)
let z1 = tensor_matmul(&x, &w1);      // (1,2) @ (2,4) = (1,4)
let z1_b = tensor_add_bias(&z1, &b1); // (1,4) + (1,4) = (1,4)
let h1 = tensor_relu(&z1_b);          // (1,4) → (1,4)

// Output layer: z2 = h1 @ w2 + b2,  pred = z2 (linear output)
let z2 = tensor_matmul(&h1, &w2);     // (1,4) @ (4,1) = (1,1)
let pred = tensor_add_bias(&z2, &b2); // (1,1) + (1,1) = (1,1)

// Loss: MSE between prediction and target
let loss = tensor_mse_loss(&pred, &target);

println!("Forward: pred = {:.4}, loss = {:.4}",
    pred.data()[0], loss.data()[0]);

// ─── Backward pass ──────────────────────────────────────────────────────
loss.backward();
// After this call, every tensor with requires_grad=true has .grad() set

// ─── Inspect gradients ──────────────────────────────────────────────────
let grad_w1 = w1.grad().unwrap();
let grad_b1 = b1.grad().unwrap();
let grad_w2 = w2.grad().unwrap();
let grad_b2 = b2.grad().unwrap();

println!("∂L/∂w2 = {:?}", grad_w2);
println!("∂L/∂b2 = {:?}", grad_b2);

// ─── SGD parameter update ───────────────────────────────────────────────
let lr = 0.01f32;

// Update w1
{
    let mut inner = w1.0.borrow_mut();
    for (p, g) in inner.data.iter_mut().zip(grad_w1.iter()) {
        *p -= lr * g;
    }
}
// (repeat for b1, w2, b2)

// ─── Zero gradients for next iteration ─────────────────────────────────
w1.zero_grad();
b1.zero_grad();
w2.zero_grad();
b2.zero_grad();
```

### Autograd Graph for the MLP

```
x (no grad) ──┐
               ├──[MatMul(1,2,4)]──→ z1 ──┐
w1 (grad) ────┘                            ├──[Add bias]──→ z1_b ──[ReLU]──→ h1
                             b1 (grad) ────┘
                                                                               │
                                                          ┌────────────────────┘
                                                          │
                                               h1 ───────┤
                                                          ├──[MatMul(1,4,1)]──→ z2 ──┐
                                              w2 (grad) ──┘                           ├──[Add bias]──→ pred
                                                                         b2 (grad) ────┘
                                                                                        │
                                                                                        ├──[MSE]──→ loss
                                                                        target ─────────┘
```

### Graph Traversal During backward()

```
Topological order (backward):
  loss → pred → z2 → h1 → z1_b → z1 → w1 (leaf), x (leaf)
              →  b2 (leaf)
         → b1 (leaf)
    → target (leaf)
    → w2 (leaf)

At each non-leaf node, backward_op computes and accumulates gradients into inputs.
```

---

## Comparison: Autograd vs Manual Gradients

### Manual Approach (Tutorial 01 Style)

In Tutorial 01, the XOR backward pass was written explicitly:

```rust
// Output layer backward
let output_error: Vec<f32> = output.iter().zip(target.iter())
    .map(|(o, t)| o - t)
    .collect();
let output_delta: Vec<f32> = output_error.iter().zip(output.iter())
    .map(|(e, o)| e * o * (1.0 - o))  // MSE * sigmoid_deriv — hand-derived!
    .collect();

// Hidden layer backward
let hidden_error: Vec<f32> = /* matmul(output_delta, W2.T) */;
let hidden_delta: Vec<f32> = hidden_error.iter().zip(hidden.iter())
    .map(|(e, h)| e * h * (1.0 - h))  // sigmoid_deriv — hand-derived!
    .collect();

// Gradient updates
for i in 0..NUM_HIDDEN {
    for j in 0..NUM_OUTPUTS {
        w2[i][j] -= lr * hidden[i] * output_delta[j];  // manually derived!
    }
}
```

### Autograd Approach

```rust
// Forward pass — just describe what you want to compute
let h = tensor_sigmoid(&tensor_add_bias(&tensor_matmul(&x, &w1), &b1));
let out = tensor_sigmoid(&tensor_add_bias(&tensor_matmul(&h, &w2), &b2));
let loss = tensor_mse_loss(&out, &target);

// Backward pass — one line, engine handles everything
loss.backward();

// Use w1.grad(), w2.grad(), b1.grad(), b2.grad() for updates
```

### Comparison Table

| Aspect | Manual Gradients | Autograd Engine |
|--------|-----------------|-----------------|
| **Gradient derivation** | Must derive by hand for each layer | Automatic for any computation |
| **New activation function** | Re-derive all layers | Add one `backward_op` case |
| **New architecture** | Re-derive everything | Just write the forward pass |
| **Bug risk** | High (sign errors, missing factors) | Low (centralized, tested rules) |
| **Performance** | Can optimize specific ops | General purpose |
| **Transparency** | Gradients are explicit | Implicit (requires understanding engine) |
| **Educational value** | Very clear what gradient is | Abstracts away derivation |

### When Manual Gradients Are Still Useful

Despite autograd, understanding manual gradients remains valuable:
1. **Debugging**: Comparing autograd output against manual calculation catches bugs
2. **Custom operations**: New ops require writing their gradient rule
3. **Optimization**: Sometimes a fused operation with hand-crafted gradient is faster
4. **Understanding**: Deep intuition about gradients comes from deriving them by hand

---

## Verification Checkpoints

### Checkpoint 1: Simple Scalar Computation ✓

**Test:** `out = a * b + c` with `a=3, b=4, c=2`

```rust
use rust_neural_networks::autograd::tensor::Tensor;
use rust_neural_networks::autograd::ops::{tensor_add, tensor_mul_scalar};

// We use element-wise: create scalars
let a = Tensor::from_vec(vec![3.0], (1, 1), true);
let b_data = Tensor::from_vec(vec![4.0], (1, 1), false);  // constant
let c = Tensor::from_vec(vec![2.0], (1, 1), true);

// d = a * 4.0 (tensor_mul_scalar)
let d = tensor_mul_scalar(&a, 4.0);
// out = d + c
let out = tensor_add(&d, &c);

out.backward();

let grad_a = a.grad().unwrap();
let grad_c = c.grad().unwrap();

assert!((grad_a[0] - 4.0).abs() < 1e-5, "∂out/∂a = 4.0, got {}", grad_a[0]);
assert!((grad_c[0] - 1.0).abs() < 1e-5, "∂out/∂c = 1.0, got {}", grad_c[0]);
```

**Expected values:**
```
∂out/∂a = 4.0   (because out = a*4 + c, so ∂out/∂a = 4)
∂out/∂c = 1.0   (because ∂(a*4 + c)/∂c = 1)
```

### Checkpoint 2: MSE Loss Gradient ✓

**Test:** `loss = MSE(pred, target)` with `pred=2.0, target=1.0`

```rust
use rust_neural_networks::autograd::tensor::Tensor;
use rust_neural_networks::autograd::ops::tensor_mse_loss;

let pred   = Tensor::from_vec(vec![2.0], (1, 1), true);
let target = Tensor::from_vec(vec![1.0], (1, 1), false);

let loss = tensor_mse_loss(&pred, &target);

// MSE = (pred - target)^2 / n = (2-1)^2 / 1 = 1.0
assert!((loss.data()[0] - 1.0).abs() < 1e-5,
    "MSE loss = 1.0, got {}", loss.data()[0]);

loss.backward();

let grad_pred = pred.grad().unwrap();

// ∂MSE/∂pred = 2*(pred-target)/n = 2*(2-1)/1 = 2.0
assert!((grad_pred[0] - 2.0).abs() < 1e-5,
    "∂loss/∂pred = 2.0, got {}", grad_pred[0]);
```

**Expected values:**
```
loss       = 1.0
∂loss/∂pred = 2.0   (from MSE gradient: 2*(pred-target)/n)
```

### Checkpoint 3: ReLU Gradient Mask ✓

**Test:** ReLU backward correctly zeros out gradients for negative pre-activations

```rust
use rust_neural_networks::autograd::tensor::Tensor;
use rust_neural_networks::autograd::ops::{tensor_relu, tensor_sum};

// Input with positive and negative elements
let a = Tensor::from_vec(vec![-2.0, 0.5, -0.1, 3.0], (1, 4), true);
let h = tensor_relu(&a);

// h = [0.0, 0.5, 0.0, 3.0]
assert_eq!(h.data(), vec![0.0f32, 0.5, 0.0, 3.0]);

// Sum to get a scalar loss, then backward
let loss = tensor_sum(&h);
loss.backward();

// ReLU gradient: 1 where input > 0, 0 where input ≤ 0
let grad_a = a.grad().unwrap();
assert!((grad_a[0] - 0.0).abs() < 1e-5, "Negative input → grad = 0");
assert!((grad_a[1] - 1.0).abs() < 1e-5, "Positive input → grad = 1");
assert!((grad_a[2] - 0.0).abs() < 1e-5, "Negative input → grad = 0");
assert!((grad_a[3] - 1.0).abs() < 1e-5, "Positive input → grad = 1");
```

**Expected values:**
```
h.data()     = [0.0, 0.5, 0.0, 3.0]   (ReLU clamps negatives to 0)
a.grad()     = [0.0, 1.0, 0.0, 1.0]   (mask: 0 for inputs ≤ 0, 1 for inputs > 0)
```

### Checkpoint 4: Gradient Accumulation at Shared Tensor ✓

**Test:** When a tensor is used twice, gradients from both uses are summed

```rust
use rust_neural_networks::autograd::tensor::Tensor;
use rust_neural_networks::autograd::ops::{tensor_add, tensor_sum};

// w is used in two separate additions
let w = Tensor::from_vec(vec![1.0], (1, 1), true);
let x = Tensor::from_vec(vec![2.0], (1, 1), false);
let y = Tensor::from_vec(vec![3.0], (1, 1), false);

// out = (w + x) + (w + y)
// ∂out/∂w = 1 + 1 = 2  (w appears in both branches)
let branch1 = tensor_add(&w, &x);
let branch2 = tensor_add(&w, &y);
let out = tensor_add(&branch1, &branch2);

let loss = tensor_sum(&out);
loss.backward();

let grad_w = w.grad().unwrap();
assert!((grad_w[0] - 2.0).abs() < 1e-5,
    "∂out/∂w = 2.0 (accumulates from both uses), got {}", grad_w[0]);
```

**Expected value:**
```
∂out/∂w = 2.0   (w contributes to two branches; gradients accumulate)
```

---

## Exercises

### Beginner

**Exercise 1: Trace Through Sigmoid Gradient**

Using the values `a = 0.0`:
1. Compute `s = sigmoid(a)` by hand: what is `s`?
2. The gradient rule is `∂s/∂a = s * (1 - s)`. Compute it by hand.
3. Create a `Tensor` with `a = 0.0` and `requires_grad = true`, call `tensor_sigmoid`, then `backward()`.
4. Verify `a.grad()[0]` matches your hand calculation.

*Expected:* `sigmoid(0) = 0.5`, `∂sigmoid/∂a at 0 = 0.5 * 0.5 = 0.25`

**Exercise 2: Longer Computation Chain**

Compute `out = relu(a * 3.0 + 2.0)` with `a = -1.0`. What is:
- The forward value `out`?
- The gradient `∂out/∂a`?

Hint: ReLU clamps negatives to zero. Think about what happens to the gradient when the pre-activation is negative.

*Expected:* `out = 0.0` (clamped), `∂out/∂a = 0.0` (ReLU mask blocks gradient)

### Intermediate

**Exercise 3: Manual vs Autograd Comparison**

For `loss = MSE(matmul(x, w), target)` with:
- `x = [[1.0, 2.0]]` (shape 1×2)
- `w = [[0.5], [0.5]]` (shape 2×1)
- `target = [[1.0]]` (shape 1×1)

1. Compute the forward pass by hand to get `pred` and `loss`.
2. Derive `∂loss/∂w` by hand using the chain rule.
3. Use the autograd engine to compute the same gradient.
4. Verify they match.

**Exercise 4: Diamond Graph**

Create a "diamond" computation:
```
     w
    / \
  out1  out2   (w is used in two different operations)
    \  /
    loss
```

Use `tensor_mul_scalar(&w, 2.0)` and `tensor_mul_scalar(&w, 3.0)` as the two branches, then add and sum them.

- What is `∂loss/∂w`?
- Why does `accumulate_grad` use `+=` instead of `=`?

*Expected:* `∂loss/∂w = 2.0 + 3.0 = 5.0`

### Advanced

**Exercise 5: Implement a New Differentiable Op**

The square operation `out[i] = a[i]^2` is not built-in. Implement it using existing ops:
- `tensor_mul_scalar` — not quite (doesn't multiply element-wise by `a` itself)
- Alternatively, use `tensor_add` and `tensor_mul_scalar` creatively

Then verify that `∂(a^2)/∂a = 2a` by calling `backward()` and checking the gradient.

*Hint:* You can write `a^2 = a * a` using element-wise multiplication. Look at how `backward_op` handles `Op::Mul`.

**Exercise 6: Extend the Engine**

Study `src/autograd/ops.rs` and `src/autograd/tape.rs`. Add a new operation `tensor_square(a)` that:
1. Computes `out[i] = a[i]^2` in the forward pass
2. Records a new `Op::Square` variant in the tape
3. Implements the backward rule: `grad_a[i] = upstream[i] * 2 * a[i]`

Add a test verifying `∂(a^2)/∂a = 2a`.

---

## Next Steps

**Related reading:**
- `src/autograd/tensor.rs` — Complete `Tensor` implementation with all accessor methods
- `src/autograd/tape.rs` — `Op` enum, `GradNode`, and `build_topo` DFS implementation
- `src/autograd/ops.rs` — All differentiable operations and their backward rules

**Related tutorials:**
- [Tutorial 01: XOR MLP](01_xor_mlp.md) — Manual gradient derivation to compare with
- [Tutorial 02: MNIST MLP](02_mnist_mlp.md) — Large-scale training where autograd saves effort

**Tests that validate the engine:**
- `tests/test_autograd_tensor.rs` — Tensor construction, accumulation, zero_grad
- `tests/test_autograd_tape.rs` — Topological sort correctness on various graph shapes
- `tests/test_autograd_ops.rs` — Forward/backward correctness for all operations

**Further exploration:**
- **Higher-order gradients**: What happens if you call `backward()` on a gradient?
- **Gradient checkpointing**: Trade compute for memory by recomputing forward activations during backward
- **Forward-mode AD**: Implement dual numbers for forward-mode differentiation and compare with reverse-mode
- **JIT compilation**: How frameworks like TorchScript trace and optimize computation graphs

---

**Well done!** You now understand the core engine that powers all modern deep learning frameworks. The three key ideas are:
1. **Record** every operation in a DAG during the forward pass
2. **Traverse** the DAG in reverse topological order during backward
3. **Accumulate** gradients at each node using per-operation chain rule formulas

Everything else — complex architectures, fancy optimizers, distributed training — builds on these three steps.
