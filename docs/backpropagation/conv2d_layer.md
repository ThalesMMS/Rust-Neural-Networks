# Convolutional Layer Mathematics

This document provides a comprehensive explanation of the mathematics behind 2D Convolutional layers, covering forward propagation, backward propagation, and gradient derivations with detailed code mappings.

## Table of Contents

- [Overview](#overview)
- [Forward Pass](#forward-pass)
  - [Mathematical Definition](#mathematical-definition)
  - [Convolution Operation](#convolution-operation)
  - [Padding Mechanics](#padding-mechanics)
  - [Stride Mechanics](#stride-mechanics)
  - [Output Dimensions](#output-dimensions)
  - [Sliding Window Visualization](#sliding-window-visualization)
  - [Implementation Details](#implementation-details)
  - [Computational Complexity](#computational-complexity)
- [Backward Pass](#backward-pass)
  - [Gradient Computation Overview](#gradient-computation-overview)
  - [Chain Rule Application](#chain-rule-application)
  - [Gradient Formulas](#gradient-formulas)
  - [Implementation Notes](#implementation-notes)
- [Parameter Updates](#parameter-updates)
- [Initialization](#initialization)
- [Numerical Considerations](#numerical-considerations)

## Overview

A 2D Convolutional layer (Conv2D) is the fundamental building block of Convolutional Neural Networks (CNNs), particularly for computer vision tasks. Unlike dense layers that connect every input to every output, convolutional layers apply learned filters (kernels) that slide across the input, detecting local spatial patterns.

**Key characteristics:**
- **Local connectivity**: Each output neuron connects to a small spatial region of the input
- **Parameter sharing**: The same filter weights are reused across all spatial positions
- **Translation equivariance**: Detects features regardless of their position in the input
- **Sparse interactions**: Dramatically fewer parameters than equivalent dense layers

**Parameters:**
- **Filters/Kernels (W)**: Learned weight tensors that detect features
- **Biases (b)**: One bias per output channel

**Use cases:**
- Image classification (MNIST, ImageNet)
- Object detection
- Semantic segmentation
- Any task with spatial or local structure

## Forward Pass

### Mathematical Definition

The forward pass of a 2D convolutional layer computes:

```
y[b, oc, oy, ox] = Σ Σ Σ x[b, ic, iy, ix] · W[oc, ic, ky, kx] + b[oc]
                   ic ky kx
```

Where:
- **x**: Input tensor of shape (batch, in_channels, input_height, input_width)
- **W**: Weight tensor (filters) of shape (out_channels, in_channels, kernel_size, kernel_size)
- **b**: Bias vector of shape (out_channels,)
- **y**: Output tensor of shape (batch, out_channels, output_height, output_width)

And the summation is over:
- **ic**: Input channels (depth dimension)
- **ky, kx**: Kernel spatial dimensions

The position (iy, ix) in the input is determined by the stride and padding:
```
iy = oy × stride + ky - padding
ix = ox × stride + kx - padding
```

### Convolution Operation

**Discrete 2D Convolution:**

For a single input channel and single output channel, the convolution at output position (oy, ox) is:

```
y[oy, ox] = Σ  Σ  x[iy, ix] · W[ky, kx]
           ky kx
```

This is a **correlation operation** (not true mathematical convolution, which would flip the kernel). In deep learning, we call this convolution by convention.

**Multi-channel convolution:**

With multiple input channels (e.g., RGB image has 3 channels), the convolution accumulates over all input channels:

```
y[oc, oy, ox] = Σ ( Σ  Σ  x[ic, iy, ix] · W[oc, ic, ky, kx] ) + b[oc]
               ic  ky kx
```

Each output channel has its own set of filters (one per input channel) that are convolved with the input and summed together.

**Batched computation:**

In practice, we process multiple images in a batch:

```
y[b, oc, oy, ox] = Σ ( Σ  Σ  x[b, ic, iy, ix] · W[oc, ic, ky, kx] ) + b[oc]
                  ic  ky kx
```

The batch dimension is independent - each sample is processed identically.

### Padding Mechanics

**Purpose:**
- Control output spatial dimensions
- Preserve input resolution (with appropriate padding and stride)
- Allow the filter to process border pixels

**Zero-padding:**

Adds zeros around the input borders. For padding p:

```
Original input (4×4):          Padded input (p=1, becomes 6×6):
┌─────────────┐                ┌─────────────────────┐
│ a  b  c  d │                │ 0  0  0  0  0  0  │
│ e  f  g  h │                │ 0  a  b  c  d  0  │
│ i  j  k  l │                │ 0  e  f  g  h  0  │
│ m  n  o  p │                │ 0  i  j  k  l  0  │
└─────────────┘                │ 0  m  n  o  p  0  │
                               │ 0  0  0  0  0  0  │
                               └─────────────────────┘
```

**Implementation:**

Padding is applied implicitly during the convolution loop by checking boundary conditions:

```rust
// Compute input position accounting for padding
let iy = oy as isize * stride as isize + ky as isize - padding;
let ix = ox as isize * stride as isize + kx as isize - padding;

// Only accumulate if within bounds (outside is treated as zero)
if iy >= 0 && iy < input_height as isize &&
   ix >= 0 && ix < input_width as isize {
    // Accumulate contribution
    sum += input[...] * weights[...];
}
// Implicit: if outside bounds, contribution is 0 (zero-padding)
```

**Common padding strategies:**

1. **Valid padding (p=0)**: No padding, output shrinks
   - Output size: `input_size - kernel_size + 1`

2. **Same padding**: Padding chosen to maintain input size (with stride=1)
   - For kernel_size k: `p = (k - 1) / 2` (when k is odd)
   - Example: 3×3 kernel → p=1, 5×5 kernel → p=2

3. **Full padding**: Maximum padding (p = kernel_size - 1)
   - Output size: `input_size + kernel_size - 1`

### Stride Mechanics

**Purpose:**
- Control output spatial dimensions
- Reduce computational cost
- Downsample feature maps

**Stride s:** The number of pixels the filter moves between consecutive applications.

**Visual example (stride=1 vs stride=2):**

```
Stride = 1:                    Stride = 2:
Filter slides one pixel        Filter slides two pixels

Input (4×4):                   Input (4×4):
┌───┬───┬───┬───┐             ┌───┬───┬───┬───┐
│ a │ b │ c │ d │             │ a │ b │ c │ d │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤
│ e │ f │ g │ h │             │ e │ f │ g │ h │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤
│ i │ j │ k │ l │             │ i │ j │ k │ l │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤
│ m │ n │ o │ p │             │ m │ n │ o │ p │
└───┴───┴───┴───┘             └───┴───┴───┴───┘

3×3 filter positions:          3×3 filter positions:

Position 0: (0,0)              Position 0: (0,0)
┌───┬───┬───┐                 ┌───┬───┬───┐
│ a │ b │ c │                 │ a │ b │ c │
├───┼───┼───┤                 ├───┼───┼───┤
│ e │ f │ g │                 │ e │ f │ g │
├───┼───┼───┤                 ├───┼───┼───┤
│ i │ j │ k │                 │ i │ j │ k │
└───┴───┴───┘                 └───┴───┴───┘

Position 1: (0,1)              Position 1: (0,2)  [skip column 1]
    ┌───┬───┬───┐                     ┌───┬───┬───┐
    │ b │ c │ d │                     │ c │ d │   │
    ├───┼───┼───┤                     ├───┼───┼───┤
    │ f │ g │ h │                     │ g │ h │   │
    ├───┼───┼───┤                     ├───┼───┼───┤
    │ j │ k │ l │                     │ k │ l │   │
    └───┴───┴───┘                     └───┴───┴───┘

Position 2: (0,2)              Output: 2×2
        ┌───┬───┬───┐         (vs 2×4 for stride=1)
        │ c │ d │   │
        ├───┼───┼───┤
        │ g │ h │   │
        ├───┼───┼───┤
        │ k │ l │   │
        └───┴───┴───┘

Output: 2×4
```

**Effect on output size:** Larger stride → smaller output spatial dimensions

### Output Dimensions

**Formula:**

For input size I, kernel size K, padding P, and stride S:

```
output_size = floor((I + 2P - K) / S) + 1
```

**Height and width computed independently:**

```
output_height = floor((input_height + 2×padding - kernel_size) / stride) + 1
output_width  = floor((input_width + 2×padding - kernel_size) / stride) + 1
```

**Implementation:**

```rust
pub fn output_height(&self) -> usize {
    ((self.input_height as isize + 2 * self.padding - self.kernel_size as isize)
        / self.stride as isize + 1) as usize
}

pub fn output_width(&self) -> usize {
    ((self.input_width as isize + 2 * self.padding - self.kernel_size as isize)
        / self.stride as isize + 1) as usize
}
```

**Examples:**

| Input | Kernel | Padding | Stride | Output |
|-------|--------|---------|--------|--------|
| 28×28 | 3×3 | 0 | 1 | 26×26 |
| 28×28 | 3×3 | 1 | 1 | 28×28 |
| 28×28 | 3×3 | 1 | 2 | 14×14 |
| 32×32 | 5×5 | 2 | 1 | 32×32 |
| 224×224 | 7×7 | 3 | 2 | 112×112 |

**Special case - "same" convolution:**

To preserve input spatial dimensions with stride=1:
```
padding = (kernel_size - 1) / 2
```

Example: 28×28 input, 3×3 kernel, padding=1, stride=1 → 28×28 output

### Sliding Window Visualization

**How convolution works - step by step:**

Consider a 5×5 input (single channel), 3×3 kernel, padding=0, stride=1:

```
Input (5×5):                  Kernel W (3×3):
┌─────────────────────┐      ┌─────────────┐
│  1   2   3   4   5  │      │ w₁  w₂  w₃ │
│  6   7   8   9  10  │      │ w₄  w₅  w₆ │
│ 11  12  13  14  15  │      │ w₇  w₈  w₉ │
│ 16  17  18  19  20  │      └─────────────┘
│ 21  22  23  24  25  │
└─────────────────────┘

Output (3×3):
┌─────────────────────┐
│ y₁₁  y₁₂  y₁₃ │
│ y₂₁  y₂₂  y₂₃ │
│ y₃₁  y₃₂  y₃₃ │
└─────────────────────┘
```

**Computing output position (0,0) → y₁₁:**

```
Step 1: Position kernel at top-left
┌─────────────────┐
│ 1*w₁  2*w₂  3*w₃│
│ 6*w₄  7*w₅  8*w₆│
│11*w₇ 12*w₈ 13*w₉│
└─────────────────┘

y₁₁ = 1·w₁ + 2·w₂ + 3·w₃ +
      6·w₄ + 7·w₅ + 8·w₆ +
     11·w₇ + 12·w₈ + 13·w₉ + b
```

**Computing output position (0,1) → y₁₂:**

```
Step 2: Slide kernel right by stride (1)
    ┌─────────────────┐
    │ 2*w₁  3*w₂  4*w₃│
    │ 7*w₄  8*w₅  9*w₆│
    │12*w₇ 13*w₈ 14*w₉│
    └─────────────────┘

y₁₂ = 2·w₁ + 3·w₂ + 4·w₃ +
      7·w₄ + 8·w₅ + 9·w₆ +
     12·w₇ + 13·w₈ + 14·w₉ + b
```

**Pattern continues:**
- Slide right until hitting the right edge
- Drop down one row (stride)
- Continue sliding right
- Repeat until bottom-right corner

**Total operations per output pixel:**
- Kernel size: K×K multiplications and additions
- Input channels: multiply by in_channels
- For each output: `in_channels × K × K` multiply-adds

### Implementation Details

**Code mapping to `src/layers/conv2d.rs` forward method (lines 281-335):**

```rust
fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
    let out_h = self.output_height();
    let out_w = self.output_width();
    let out_spatial = out_h * out_w;
    let in_spatial = self.input_height * self.input_width;

    // Loop over batch
    for b in 0..batch_size {
        let in_base = b * (self.in_channels * in_spatial);
        let out_base_b = b * (self.out_channels * out_spatial);

        // Loop over output channels (filters)
        for oc in 0..self.out_channels {
            let bias = self.biases[oc];
            let out_base = out_base_b + oc * out_spatial;

            // Loop over output spatial positions
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = bias;  // Start with bias

                    // Accumulate over input channels
                    for ic in 0..self.in_channels {
                        let w_base = (oc * self.in_channels + ic)
                                   * self.kernel_size * self.kernel_size;
                        let in_base_c = in_base + ic * in_spatial;

                        // Convolve kernel over input
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                // Map output position to input position
                                let iy = oy as isize * self.stride as isize
                                       + ky as isize - self.padding;
                                let ix = ox as isize * self.stride as isize
                                       + kx as isize - self.padding;

                                // Check bounds (implicit zero-padding)
                                if iy >= 0 && iy < self.input_height as isize &&
                                   ix >= 0 && ix < self.input_width as isize {
                                    let iyy = iy as usize;
                                    let ixx = ix as usize;

                                    // Index into flattened arrays
                                    let in_idx = in_base_c
                                               + iyy * self.input_width + ixx;
                                    let w_idx = w_base
                                              + ky * self.kernel_size + kx;

                                    // Accumulate: output += input * weight
                                    sum += input[in_idx] * self.weights[w_idx];
                                }
                            }
                        }
                    }

                    // Write output
                    let out_idx = out_base + oy * out_w + ox;
                    output[out_idx] = sum;
                }
            }
        }
    }
}
```

**Loop structure breakdown:**

1. **Batch loop** (`b`): Process each sample independently
2. **Output channel loop** (`oc`): Each filter produces one output channel
3. **Output spatial loops** (`oy`, `ox`): Each position in output feature map
4. **Input channel loop** (`ic`): Accumulate across all input channels
5. **Kernel loops** (`ky`, `kx`): Apply filter weights at current position

**Memory layout:**

All tensors are stored in **row-major, contiguous layout**:

```
Input:  [batch, in_channels, height, width]
        Flattened index: b*(C_in*H*W) + ic*(H*W) + y*W + x

Weights: [out_channels, in_channels, kernel_h, kernel_w]
         Flattened index: oc*(C_in*K*K) + ic*(K*K) + ky*K + kx

Output: [batch, out_channels, out_height, out_width]
        Flattened index: b*(C_out*H_out*W_out) + oc*(H_out*W_out) + y*W_out + x
```

**Boundary handling:**

The check `if iy >= 0 && iy < input_height && ix >= 0 && ix < input_width` implements implicit zero-padding:
- Inside bounds: accumulate `input[iy, ix] * weight[ky, kx]`
- Outside bounds: treat as zero, contribute nothing to sum

### Computational Complexity

**Time Complexity:**

For a single output pixel:
```
O(in_channels × kernel_height × kernel_width)
```

For entire forward pass:
```
O(batch × out_channels × out_height × out_width × in_channels × kernel_size²)
```

**Space Complexity:**

```
Parameters: O(out_channels × in_channels × kernel_size²)
Activations: O(batch × out_channels × out_height × out_width)
```

**Example calculation:**

For MNIST CNN layer: 1 input channel → 8 output channels, 3×3 kernel, 28×28 input:
- Parameters: 8 × 1 × 3 × 3 + 8 = 80 parameters
- Output: 8 × 28 × 28 = 6,272 activations per sample
- FLOPs (batch=1): 8 × 28 × 28 × 1 × 9 ≈ 56K operations

Compare to equivalent dense layer (784 → 6,272):
- Parameters: 784 × 6,272 + 6,272 ≈ 4.9M parameters (61,000× more!)
- FLOPs: 784 × 6,272 ≈ 4.9M operations (88× more!)

**Key efficiency insight:**
- **Parameter sharing**: Same kernel weights used at all spatial positions
- **Sparse connectivity**: Each output only depends on local input region
- Result: Dramatically fewer parameters and computations than dense layers

## Backward Pass

### Gradient Flow Visualization

The following diagram illustrates how gradients flow backward through the convolutional layer:

```
                    FORWARD PASS                    |                  BACKWARD PASS
                                                    |
    Input x                                         |                                ∂L/∂x
    (B, C_in, H, W)                                 |                         (B, C_in, H, W)
         │                                          |                                ▲
         │  ┌─────────────────┐                    |                                │
         │  │  Sliding Window │                    |          ┌─────────────────────┴─────────┐
         │  │  with Stride &  │                    |          │ Full Convolution (Transposed) │
         │  │     Padding     │                    |          │    ∂L/∂y ⊛ W (flipped)        │
         │  └─────────────────┘                    |          └───────────────────────────────┘
         ▼                                          |                   ▲              ▲
    ┌────────────────┐                             |                   │              │
    │                │                              |                   │              │
    │  Convolution:  │◄───────── Filters W ────────┼───────────────────┤              │
    │  x ⊗ W         │           (C_out, C_in,     |        ∂L/∂W      │              │
    │                │            K_h, K_w)         |   (C_out, C_in,   │              │
    │  (element-wise │                              |    K_h, K_w)      │              │
    │   multiply &   │                              |         ▲         │              │
    │     sum)       │                              |         │         │              │
    └────────┬───────┘                              |   ┌─────┴────┐    │              │
             │                                      |   │  Input x │    │              │
             │                                      |   │    ⊗     │    │              │
             ▼                                      |   │  ∂L/∂y   │    │              │
    ┌────────────────┐                             |   │ (conv)   │    │              │
    │     + b        │◄───────── Biases b ─────────┼───│          │    │              │
    │  (broadcast)   │           (C_out)           |   └──────────┘    │              │
    └────────┬───────┘                             |         ▲         │              │
             │                                      |         │         │              │
             ▼                                      |      ∂L/∂b        │              │
    Output y                                        |     (C_out)   ∂L/∂y         ∂L/∂y
    (B, C_out, H', W')                              |         ▲    (B, C_out,     (B, C_out,
             │                                      |         │     H', W')        H', W')
             ▼                                      |         │
           Loss                                     |    ┌────┴──────┐
                                                    |    │ Sum over  │
                                                    |    │  spatial  │
                                                    |    │ & batch   │
                                                    |    └───────────┘

Spatial Gradient Flow Detail:
────────────────────────────────────────────────────────────────────────────────────────

Forward: Each output pixel receives from a local receptive field

         Input slice:                Filter:              Output pixel:
         ┌─────────────┐            ┌─────────┐
         │ x₁  x₂  x₃ │            │ w₁  w₂  w₃ │              y = x₁w₁ + x₂w₂ + ...
         │ x₄  x₅  x₆ │      ⊗     │ w₄  w₅  w₆ │              ... + x₉w₉ + b
         │ x₇  x₈  x₉ │            │ w₇  w₈  w₉ │
         └─────────────┘            └─────────┘

Backward: Each input pixel receives from all outputs it contributed to

         ∂L/∂x₅ = Σ  (∂L/∂y[i,j] · w₅)   ← sum over all positions where x₅ was used
                 i,j
                      └──────────────────┘
                      All output positions that used x₅


Legend:
  ──►  Forward data flow
  ◄──  Parameter (weights/biases)
  ──▲  Backward gradient flow
  ⊗    Convolution operation
  ⊛    Full/transposed convolution
```

**Key insights:**
- **∂L/∂x computation**: Equivalent to a full (transposed) convolution of ∂L/∂y with flipped kernels
- **∂L/∂W computation**: Convolution of input x with ∂L/∂y (roles reversed from forward pass)
- **∂L/∂b computation**: Sum ∂L/∂y over all spatial positions and batch dimension
- **Spatial overlap**: Each input pixel's gradient accumulates contributions from all output pixels in its receptive field

### Gradient Computation Overview

During backpropagation, we receive the gradient of the loss with respect to the layer's output (∂L/∂y) and must compute:

1. **Gradient w.r.t. input (∂L/∂x)**: Propagate error to previous layer
2. **Gradient w.r.t. weights (∂L/∂W)**: Update filter weights
3. **Gradient w.r.t. biases (∂L/∂b)**: Update biases

Each gradient computation involves applying the chain rule to the convolution operation.

### Chain Rule Application

**Forward pass:**
```
y[oc, oy, ox] = Σ Σ Σ x[ic, iy, ix] · W[oc, ic, ky, kx] + b[oc]
               ic ky kx
```

**Chain rule:**
```
∂L/∂x = ∂L/∂y · ∂y/∂x
∂L/∂W = ∂L/∂y · ∂y/∂W
∂L/∂b = ∂L/∂y · ∂y/∂b
```

**Partial derivatives:**

1. **∂y/∂x**: How output changes with input
   ```
   ∂y[oc,oy,ox]/∂x[ic,iy,ix] = W[oc, ic, ky, kx]
   where: iy = oy*stride + ky - padding
          ix = ox*stride + kx - padding
   ```

2. **∂y/∂W**: How output changes with weights
   ```
   ∂y[oc,oy,ox]/∂W[oc,ic,ky,kx] = x[ic, iy, ix]
   ```

3. **∂y/∂b**: How output changes with biases
   ```
   ∂y[oc,oy,ox]/∂b[oc] = 1
   ```

### Gradient Formulas

#### 1. Gradient w.r.t. Input (∂L/∂x)

**Intuition:** The gradient flows back through the convolution operation. Each input pixel contributed to multiple output pixels (via the sliding kernel), so its gradient is the sum of all those contributions.

**Formula (for each input position):**
```
∂L/∂x[ic, iy, ix] = Σ  Σ  Σ  ∂L/∂y[oc, oy, ox] · W[oc, ic, ky, kx]
                   oc oy ox
```

Where the summation is over all output positions (oy, ox) and output channels (oc) where the kernel window at position (ky, kx) overlaps input position (iy, ix).

**Determining which outputs used this input:**

For input position (iy, ix) to contribute to output position (oy, ox) through kernel position (ky, kx):
```
iy = oy × stride + ky - padding
ix = ox × stride + kx - padding
```

Solving for kernel position:
```
ky = iy - oy × stride + padding
kx = ix - ox × stride + padding
```

Valid when: 0 ≤ ky < kernel_size and 0 ≤ kx < kernel_size

**Implementation (src/layers/conv2d.rs lines 448-492):**

```rust
// Zero out grad_input
for v in grad_input.iter_mut() {
    *v = 0.0;
}

for b in 0..batch_size {
    let in_base = b * (self.in_channels * in_spatial);
    let g_base_b = b * (self.out_channels * out_spatial);

    // For each input channel
    for ic in 0..self.in_channels {
        let in_base_c = in_base + ic * in_spatial;

        // For each output channel
        for oc in 0..self.out_channels {
            let g_base = g_base_b + oc * out_spatial;
            let w_base = (oc * self.in_channels + ic)
                       * self.kernel_size * self.kernel_size;

            // For each output position
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let g = grad_output[g_base + oy * out_w + ox];

                    // For each kernel position
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            // Map to input position
                            let iy = oy as isize * self.stride as isize
                                   + ky as isize - self.padding;
                            let ix = ox as isize * self.stride as isize
                                   + kx as isize - self.padding;

                            // Check bounds
                            if iy >= 0 && iy < self.input_height as isize &&
                               ix >= 0 && ix < self.input_width as isize {
                                let iyy = iy as usize;
                                let ixx = ix as usize;
                                let in_idx = in_base_c
                                           + iyy * self.input_width + ixx;
                                let w_idx = w_base
                                          + ky * self.kernel_size + kx;

                                // Accumulate gradient
                                grad_input[in_idx] += g * self.weights[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**Key insight:** This is equivalent to a **full convolution** (or transposed convolution) of grad_output with the flipped kernel.

#### 2. Gradient w.r.t. Weights (∂L/∂W)

**Intuition:** Each weight element affects all outputs where it was applied. The gradient for a weight is the sum of all input×gradient products at positions where that weight was used.

**Formula:**
```
∂L/∂W[oc, ic, ky, kx] = Σ  Σ  Σ  ∂L/∂y[b, oc, oy, ox] · x[b, ic, iy, ix]
                        b  oy ox
```

Where:
```
iy = oy × stride + ky - padding
ix = ox × stride + kx - padding
```

**Interpretation:** This is a convolution of the input with the gradient of the output.

**Implementation (src/layers/conv2d.rs lines 389-438):**

```rust
// Zero out gradient accumulators
let mut grad_w = self.grad_weights.borrow_mut();
let mut grad_b = self.grad_biases.borrow_mut();
for g in grad_w.iter_mut() { *g = 0.0; }
for g in grad_b.iter_mut() { *g = 0.0; }

// Accumulate gradients across batch
for b in 0..batch_size {
    let in_base = b * (self.in_channels * in_spatial);
    let g_base_b = b * (self.out_channels * out_spatial);

    for oc in 0..self.out_channels {
        let g_base = g_base_b + oc * out_spatial;

        // Accumulate weight gradients
        for ic in 0..self.in_channels {
            let w_base = (oc * self.in_channels + ic)
                       * self.kernel_size * self.kernel_size;
            let in_base_c = in_base + ic * in_spatial;

            for oy in 0..out_h {
                for ox in 0..out_w {
                    let g = grad_output[g_base + oy * out_w + ox];

                    // For each kernel position
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            // Map to input position
                            let iy = oy as isize * self.stride as isize
                                   + ky as isize - self.padding;
                            let ix = ox as isize * self.stride as isize
                                   + kx as isize - self.padding;

                            // Check bounds
                            if iy >= 0 && iy < self.input_height as isize &&
                               ix >= 0 && ix < self.input_width as isize {
                                let iyy = iy as usize;
                                let ixx = ix as usize;
                                let in_idx = in_base_c
                                           + iyy * self.input_width + ixx;
                                let w_idx = w_base
                                          + ky * self.kernel_size + kx;

                                // Accumulate: grad_W += grad_output * input
                                grad_w[w_idx] += g * input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Average gradients over batch
let scale = 1.0 / batch_size as f32;
for g in grad_w.iter_mut() { *g *= scale; }
for g in grad_b.iter_mut() { *g *= scale; }
```

#### 3. Gradient w.r.t. Biases (∂L/∂b)

**Intuition:** Each bias is added to all spatial positions of its output channel. Therefore, its gradient is the sum of all gradients in that channel.

**Formula:**
```
∂L/∂b[oc] = Σ  Σ  Σ  ∂L/∂y[b, oc, oy, ox]
            b  oy ox
```

Sum over batch and spatial dimensions for each output channel.

**Implementation (src/layers/conv2d.rs lines 397-403):**

```rust
// Accumulate bias gradient (sum over spatial dimensions)
for oc in 0..self.out_channels {
    let g_base = g_base_b + oc * out_spatial;

    for oy in 0..out_h {
        for ox in 0..out_w {
            let g = grad_output[g_base + oy * out_w + ox];
            grad_b[oc] += g;  // Sum all gradients for this channel
        }
    }
}
```

### Implementation Notes

**Gradient accumulation:**

Gradients are accumulated across the entire batch before being averaged:
```rust
// Accumulate across batch
for b in 0..batch_size {
    // ... accumulate gradients ...
}

// Average
let scale = 1.0 / batch_size as f32;
for g in grad_weights.iter_mut() { *g *= scale; }
for g in grad_biases.iter_mut() { *g *= scale; }
```

**Memory efficiency:**

- Input is cached during forward pass for use in backward pass
- Gradients are computed in-place where possible
- No need to store intermediate activations beyond layer outputs

**Numerical stability:**

- Gradients are computed in f32 (single precision)
- Boundary checks prevent out-of-bounds access
- Batch averaging prevents gradient explosion with large batches

**Computational cost:**

The backward pass has similar computational complexity to the forward pass:
- Computing ∂L/∂x: Same loop structure as forward pass
- Computing ∂L/∂W: Similar loop structure, iterating over all filter positions
- Computing ∂L/∂b: Simple summation over spatial dimensions

## Parameter Updates

After computing gradients, parameters are updated using gradient descent:

**Vanilla gradient descent:**
```
W := W - η · (∂L/∂W)
b := b - η · (∂L/∂b)
```

**Implementation (src/layers/conv2d.rs lines 523-548):**

```rust
fn update_parameters(&mut self, learning_rate: f32) {
    let grad_w = self.grad_weights.borrow();
    let grad_b = self.grad_biases.borrow();

    // Update weights: W -= learning_rate * grad_W
    for (weight, &gradient) in self.weights.iter_mut().zip(grad_w.iter()) {
        *weight -= learning_rate * gradient;
    }

    // Update biases: b -= learning_rate * grad_b
    for (bias, &gradient) in self.biases.iter_mut().zip(grad_b.iter()) {
        *bias -= learning_rate * gradient;
    }

    // Clear gradients for next iteration
    drop(grad_w);
    drop(grad_b);
    self.grad_weights.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
    self.grad_biases.borrow_mut().iter_mut().for_each(|g| *g = 0.0);
}
```

## Initialization

### Xavier (Glorot) Initialization

The layer uses Xavier initialization adapted for convolutional layers:

**Formula:**
```
W ~ Uniform(-limit, limit)
where limit = sqrt(6 / (fan_in + fan_out))
```

For convolutional layers:
```
fan_in = in_channels × kernel_height × kernel_width
fan_out = out_channels × kernel_height × kernel_width
```

**Rationale:**

Xavier initialization maintains variance across layers:
- Prevents vanishing gradients (weights too small)
- Prevents exploding gradients (weights too large)
- Accounts for the number of connections each filter has

**Implementation (src/layers/conv2d.rs lines 105-117):**

```rust
// Xavier initialization for convolutional layers
let fan_in = (in_channels * kernel_size * kernel_size) as f32;
let fan_out = (out_channels * kernel_size * kernel_size) as f32;
let limit = (6.0f32 / (fan_in + fan_out)).sqrt();

let weight_count = out_channels * in_channels * kernel_size * kernel_size;
let mut weights = vec![0.0f32; weight_count];

for value in &mut weights {
    *value = rng.gen_range_f32(-limit, limit);
}

// Biases initialized to zero
let biases = vec![0.0f32; out_channels];
```

**Example:**

For 1 input channel → 8 output channels, 3×3 kernel:
- fan_in = 1 × 3 × 3 = 9
- fan_out = 8 × 3 × 3 = 72
- limit = sqrt(6 / (9 + 72)) = sqrt(6/81) ≈ 0.272

Weights are initialized uniformly in [-0.272, 0.272].

## Numerical Considerations

### Potential Issues

**1. Vanishing gradients:**
- Deep CNNs can suffer from gradient decay
- Mitigation: Batch normalization, residual connections (ResNet), proper initialization

**2. Exploding gradients:**
- Gradients can grow unbounded in poorly initialized networks
- Mitigation: Gradient clipping, batch normalization, proper learning rates

**3. Border effects:**
- Padding introduces artificial zeros that can affect learning
- Corners and edges are processed less frequently than center pixels
- Mitigation: Use appropriate padding, consider using "valid" convolutions

**4. Computational cost:**
- Naive implementation has 7 nested loops (batch, out_channel, out_y, out_x, in_channel, kernel_y, kernel_x)
- Mitigation: Use optimized libraries (cuDNN, NNPACK), im2col+GEMM transformations

### Best Practices

**During forward pass:**
- Cache inputs for backward pass
- Use batch normalization after convolution
- Apply activation functions (ReLU) after batch norm

**During backward pass:**
- Verify gradient shapes match parameter shapes
- Check for NaN/Inf values in gradients
- Use gradient clipping if training is unstable

**During training:**
- Monitor gradient norms per layer
- Use data augmentation for computer vision tasks
- Consider learning rate warmup for first few epochs

### Debugging Tips

**Gradient checking:**

Numerical gradient approximation:
```
numerical_grad[i] = (loss(W[i] + ε) - loss(W[i] - ε)) / (2ε)
```

Compare with analytical gradient from backprop. Should match within ~10⁻⁴ for ε=10⁻⁵.

**Verify dimensions:**
```
assert_eq!(grad_input.shape, input.shape);
assert_eq!(grad_weights.shape, weights.shape);
assert_eq!(grad_biases.shape, biases.shape);
```

**Test with known patterns:**

1. **Identity test:** Single 1×1 kernel, weight=1, should act as identity
2. **Bias test:** Zero weights, non-zero bias, output should equal bias
3. **Symmetry test:** Symmetric input + symmetric kernel → symmetric output

## Summary

The 2D convolutional layer applies learned filters to detect spatial patterns:

**Forward pass:**
- Sliding kernel across input spatial dimensions
- O(batch × out_channels × out_height × out_width × in_channels × kernel²) operations
- Parameter sharing: Same kernel weights used at all positions
- Sparse connectivity: Each output depends only on local input region

**Backward pass:**
- Three gradients computed using chain rule
- ∂L/∂x: Full convolution with grad_output and weights (transposed convolution)
- ∂L/∂W: Convolution of input with grad_output
- ∂L/∂b: Sum of grad_output over spatial dimensions

**Key advantages over dense layers:**
- **Dramatically fewer parameters**: 80 vs 4.9M for MNIST example
- **Translation equivariance**: Detects features regardless of position
- **Local connectivity**: Respects spatial structure of images
- **Hierarchical features**: Early layers detect edges, later layers detect objects

**Implementation highlights:**
- Xavier initialization prevents gradient issues
- Implicit zero-padding via boundary checks
- Stride and padding control output dimensions
- Nested loops provide flexibility but can be slow (optimized implementations use im2col+GEMM)

This forms the foundation for understanding modern CNNs like VGG, ResNet, and EfficientNet.

## Related Documentation

**Activation Functions:**
- [Activation Functions](../activation_functions.md) - Detailed mathematical documentation for ReLU and other activations used with convolutional layers

**Alternative Layer Types:**
- [Dense Layer](dense_layer.md) - Fully connected layers often used after convolutional layers for classification
- [Attention Mechanism](attention_mechanism.md) - Alternative architecture for capturing spatial relationships without convolution

**Core Architecture:**
- [Backpropagation Overview](README.md) - General backpropagation concepts and notation
- [Layer Trait](../../src/layers/trait.rs) - Core layer interface implementation
