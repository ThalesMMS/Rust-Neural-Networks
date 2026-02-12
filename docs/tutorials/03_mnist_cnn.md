# Tutorial 03: Understanding Convolutional Neural Networks (CNN)

**Level:** Advanced
**Time:** 90-120 minutes
**Prerequisites:** Tutorial 02 (MNIST MLP), understanding of convolution operation
**Implementation:** See `mnist_cnn.rs` for complete working code

**Navigation:**
← [Previous Tutorial: MNIST MLP](02_mnist_mlp.md) | [Tutorial Index](README.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why CNNs for Images?](#why-cnns-for-images)
3. [Network Architecture](#network-architecture)
4. [Understanding Convolution Operations](#understanding-convolution-operations)
5. [Filters and Kernels](#filters-and-kernels)
6. [Feature Maps](#feature-maps)
7. [Pooling Operations](#pooling-operations)
8. [Dimension Calculations](#dimension-calculations)
9. [Forward Pass Walkthrough](#forward-pass-walkthrough)
10. [Backward Pass and Gradients](#backward-pass-and-gradients)
11. [Parameter Efficiency](#parameter-efficiency)
12. [Verification and Expected Outputs](#verification-and-expected-outputs)
13. [Exercises](#exercises)
14. [Next Steps](#next-steps)

---

## Introduction

Welcome to Convolutional Neural Networks (CNNs)! This tutorial introduces the fundamental architecture that revolutionized computer vision. While the MLP from Tutorial 02 achieved ~97% accuracy on MNIST, CNNs can reach ~98%+ with **fewer parameters** by exploiting spatial structure.

**What you'll learn:**
- How convolution operations detect local patterns (edges, curves, shapes)
- Why parameter sharing makes CNNs more efficient than MLPs
- How pooling layers downsample while preserving important features
- How to track dimensions through convolutional and pooling layers
- What intermediate feature maps actually look like

**Key insight:** CNNs don't treat images as flat vectors—they preserve and exploit the 2D spatial structure.

**Implementation reference:** All code shown here is from `mnist_cnn.rs`, achieving ~98% test accuracy in 3 epochs with only ~13K parameters (vs ~407K for the MLP!).

---

## Why CNNs for Images?

### The Problem with MLPs for Images

**MLP approach (from Tutorial 02):**
```
28×28 image → flatten to 784 → Dense(512) → Dense(10)
```

**What's wrong with this?**

1. **Lost spatial structure**: Flattening destroys relationships between neighboring pixels
   - Pixel at (i, j) is adjacent to (i+1, j), but this isn't represented
   - A vertical edge spans multiple rows, but MLP sees scattered input features

2. **Inefficient parameter usage**: Every pixel connects to every hidden neuron
   - 784 × 512 = 401,408 weights just in the first layer
   - Most weights learn redundant pattern detectors for different image regions

3. **Not translation invariant**: If a digit shifts by 1 pixel, the network treats it as entirely different input
   - Pattern at position (5, 5) is detected by different weights than same pattern at (6, 6)

### How CNNs Solve These Problems

**CNN approach:**
```
28×28 image → Conv(8 filters, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(10)
```

**Key advantages:**

1. **Preserves spatial structure**: Operates directly on 2D images
   - Each 3×3 filter "sees" a small patch of the image
   - Neighboring pixels are processed together

2. **Parameter sharing**: Same filter applied to entire image
   - 8 filters × 1 channel × 3×3 = 72 weights (vs 401,408!)
   - Each filter learns one pattern (e.g., vertical edge) and detects it everywhere

3. **Translation equivariance**: If input shifts, output shifts by same amount
   - Vertical edge detector works regardless of where the edge appears
   - Same pattern at different positions uses same weights

4. **Hierarchical features**: Early layers detect simple patterns (edges), later layers combine them into complex shapes (loops, curves)

**Analogy:** Instead of memorizing the entire multiplication table (MLP), learn the multiplication algorithm and apply it to any pair of numbers (CNN).

---

## Network Architecture

### Architecture Diagram

```
Input                Conv Layer            ReLU        MaxPool         Flatten      Output
1×28×28              8×28×28                          8×14×14         1568         10

┌─────────┐         ┌─────────┐                      ┌──────┐
│ •••••••│         │ •••••••│                      │ ••••│
│ •••5•••│         │ •••••••│   Apply    2×2       │ ••••│
│ •••••••│  →  8×  │ •••••••│  → ReLU  → MaxPool → │ ••••│  → Dense → [0-9]
│ •••••••│  filters│ •••••••│                      │ ••••│   (10)
│ •••••••│         │ •••••••│                      └──────┘
└─────────┘         └─────────┘
28×28 grayscale     8 feature maps                  Flattened
(1 channel)         (3×3 filters)                   vector

Parameters:
Conv weights: 8 filters × 1 channel × 3×3 = 72
Conv biases: 8
Dense weights: 1568 × 10 = 15,680
Dense biases: 10
Total: 15,770 parameters (vs 407,050 for MLP!)
```

### Layer-by-Layer Transformation

**Dimension tracking:**
```
Input:       [batch, 1, 28, 28]     # Grayscale images
             ↓
Conv2D:      [batch, 8, 28, 28]     # 8 feature maps, same size (padding=1)
             ↓
ReLU:        [batch, 8, 28, 28]     # Activation (no dimension change)
             ↓
MaxPool:     [batch, 8, 14, 14]     # Spatial downsampling (2×2 pool)
             ↓
Flatten:     [batch, 1568]          # 8 × 14 × 14 = 1568
             ↓
Dense:       [batch, 10]            # Class scores
             ↓
Softmax:     [batch, 10]            # Probabilities
```

### Architecture Specifications

**From `mnist_cnn.rs`:**
```rust
// CNN topology: 1x28x28 -> conv -> ReLU -> 2x2 maxpool -> FC(10)
const CONV_OUT: usize = 8;      // Number of convolutional filters
const KERNEL: usize = 3;        // 3×3 filter size
const PAD: isize = 1;           // Zero-padding to maintain size
const POOL: usize = 2;          // 2×2 max pooling

const POOL_H: usize = IMG_H / POOL; // 28 / 2 = 14
const POOL_W: usize = IMG_W / POOL; // 28 / 2 = 14
const FC_IN: usize = CONV_OUT * POOL_H * POOL_W; // 8*14*14 = 1568
```

---

## Understanding Convolution Operations

### What is Convolution?

**Convolution** is a mathematical operation that slides a small **filter** (also called **kernel**) over an input, computing the dot product at each position.

**1D Example (for intuition):**
```
Input:  [1, 2, 3, 4, 5]
Filter: [1, 0, -1]  (edge detector)

Slide filter across input:
Position 0: [1, 2, 3] • [1, 0, -1] = 1×1 + 2×0 + 3×(-1) = -2
Position 1: [2, 3, 4] • [1, 0, -1] = 2×1 + 3×0 + 4×(-1) = -2
Position 2: [3, 4, 5] • [1, 0, -1] = 3×1 + 4×0 + 5×(-1) = -2

Output: [-2, -2, -2]
```

### 2D Convolution for Images

**For images, we use 2D filters:**

**Input (5×5 image patch):**
```
┌──────────────────┐
│ 0  0  0  0  0  │
│ 0  0  1  0  0  │
│ 0  0  1  0  0  │  (Vertical line)
│ 0  0  1  0  0  │
│ 0  0  0  0  0  │
└──────────────────┘
```

**Filter (3×3 vertical edge detector):**
```
┌─────────┐
│ -1  0  1│
│ -1  0  1│
│ -1  0  1│
└─────────┘
```

**Convolution process (at one position):**

Place 3×3 filter over top-left of image (with padding):
```
Filter:          Input patch:      Element-wise multiply:
┌─────────┐     ┌─────────┐       ┌─────────┐
│ -1  0  1│  ×  │ 0  0  0│   =    │ 0  0  0│
│ -1  0  1│     │ 0  0  1│        │ 0  0  1│
│ -1  0  1│     │ 0  0  1│        │ 0  0  1│
└─────────┘     └─────────┘       └─────────┘

Sum all elements: 0 + 0 + 0 + 0 + 0 + 1 + 0 + 0 + 1 = 2
Output value at this position: 2
```

The filter slides across the entire image, computing one output value per position.

### Sliding Window Visualization

**Convolution slides the filter across all positions:**

```
Input (7×7):                     Stride = 1 (move 1 pixel at a time)
┌─────────────────────────┐
│ 0  0  0  0  0  0  0  │     Filter positions:
│ 0  1  2  3  4  5  0  │     ┌─────┐
│ 0  1  2  3  4  5  0  │  →  │ F F F│ ← Position (0,0)
│ 0  1  2  3  4  5  0  │     │ F F F│
│ 0  1  2  3  4  5  0  │     │ F F F│
│ 0  1  2  3  4  5  0  │     └─────┘
│ 0  0  0  0  0  0  0  │       ↓ Move right (stride=1)
└─────────────────────────┘     ┌─────┐
                                  │ F F F│ ← Position (0,1)
                                  │ F F F│
                                  │ F F F│
                                  └─────┘
                                    ... continue across all positions
```

**With padding=1, stride=1, 3×3 filter:**
- Input: 28×28 → Output: 28×28 (same size)
- Formula: `output_size = (input_size + 2×padding - kernel_size) / stride + 1`
- Example: `(28 + 2×1 - 3) / 1 + 1 = 28`

### Implementation (from `src/layers/conv2d.rs`)

```rust
// Simplified forward pass logic
for oy in 0..output_height {
    for ox in 0..output_width {
        let mut sum = bias;  // Start with bias

        // Slide 3×3 filter over input
        for ky in 0..3 {  // Kernel row
            for kx in 0..3 {  // Kernel column
                // Map output position to input position
                let iy = oy * stride + ky - padding;
                let ix = ox * stride + kx - padding;

                // Check bounds (zero-padding outside)
                if iy >= 0 && iy < input_height && ix >= 0 && ix < input_width {
                    sum += input[iy, ix] * weights[ky, kx];
                }
            }
        }

        output[oy, ox] = sum;
    }
}
```

**Key implementation details:**

1. **Zero-padding:** Implicit—out-of-bounds positions contribute 0
2. **Stride:** Controls how far filter moves (stride=1 means move 1 pixel)
3. **Bias:** One bias value per output channel (filter), added to every output position
4. **Multi-channel:** For multiple input channels (RGB), sum across all channels

---

## Filters and Kernels

### What Do Filters Detect?

**A filter is a pattern detector.** Each 3×3 filter learns to detect a specific feature.

**Common learned filters in early layers:**

**Vertical Edge Detector:**
```
┌─────────┐
│ -1  0  1│    Responds strongly to:    │ │ (vertical edges)
│ -1  0  1│    Responds weakly to:      ─── (horizontal edges)
│ -1  0  1│
└─────────┘
```

**Horizontal Edge Detector:**
```
┌─────────┐
│ -1 -1 -1│    Responds strongly to:    ─── (horizontal edges)
│  0  0  0│    Responds weakly to:      │ │ (vertical edges)
│  1  1  1│
└─────────┘
```

**Diagonal Edge Detector:**
```
┌─────────┐
│  0 -1 -1│    Responds strongly to:    ╱ (diagonal edges)
│  1  0 -1│    Detects:                ╲ (opposite diagonal)
│  1  1  0│
└─────────┘
```

**Blob Detector (Center-surround):**
```
┌─────────┐
│ -1 -1 -1│    Responds strongly to:    ● (bright center)
│ -1  8 -1│    Responds to corners,
│ -1 -1 -1│    dots, and small features
└─────────┘
```

### Filter Learning Process

**Initial weights (random):**
```
Filter 1:         Filter 2:         Filter 3:
┌─────────┐      ┌─────────┐      ┌─────────┐
│ 0.1 -0.2 0.3│  │-0.1  0.4 -0.2│  │ 0.2  0.1 -0.3│
│-0.1  0.4 0.1│  │ 0.2 -0.3  0.1│  │-0.4  0.2  0.1│
│ 0.2 -0.1 0.2│  │ 0.3  0.1 -0.2│  │ 0.1 -0.2  0.4│
└─────────┘      └─────────┘      └─────────┘
(Random noise)   (Random noise)   (Random noise)
```

**After training (learned patterns):**
```
Filter 1:         Filter 2:         Filter 3:
┌─────────┐      ┌─────────┐      ┌─────────┐
│-0.9  0.1 0.9│  │-0.3 -0.8 -0.3│  │ 0.8  0.8  0.8│
│-0.9  0.0 0.9│  │ 0.2  0.1  0.2│  │-0.1 -0.1 -0.1│
│-0.9  0.1 0.9│  │ 0.7  0.9  0.7│  │-0.8 -0.8 -0.8│
└─────────┘      └─────────┘      └─────────┘
(Vertical edge)  (Horizontal)     (Top edge)
```

**Why does this happen?**
- Backpropagation adjusts weights to minimize loss
- Weights converge to patterns that are most useful for classification
- Different filters specialize in different features

### Multiple Filters = Multiple Feature Maps

**Our MNIST CNN uses 8 filters**, producing 8 feature maps:

```
Input (1×28×28):
┌─────────┐
│    5    │  (One grayscale image)
└─────────┘
      ↓ Apply 8 different 3×3 filters
      ↓
Output (8×28×28):
┌─────────┐  ┌─────────┐  ┌─────────┐        ┌─────────┐
│ Filter 1│  │ Filter 2│  │ Filter 3│  ...   │ Filter 8│
│  edges  │  │  curves │  │  corners│        │  blobs  │
└─────────┘  └─────────┘  └─────────┘        └─────────┘
```

Each filter detects a different pattern, and all 8 feature maps are passed to the next layer.

---

## Feature Maps

### What is a Feature Map?

A **feature map** is the output of applying one filter to the input. It's a 2D grid showing where that pattern appears in the image.

**Example: Vertical edge detection on digit "1"**

**Input image (digit "1"):**
```
┌──────────────────────────┐
│ 0  0  0  0  0  0  0  0 │
│ 0  0  0  0  1  1  0  0 │
│ 0  0  0  1  1  1  0  0 │
│ 0  0  0  0  1  1  0  0 │
│ 0  0  0  0  1  1  0  0 │
│ 0  0  0  0  1  1  0  0 │
│ 0  0  0  0  1  1  0  0 │
│ 0  0  0  0  0  0  0  0 │
└──────────────────────────┘
```

**Feature map (after vertical edge filter + ReLU):**
```
┌──────────────────────────┐
│ 0  0  0  0  0  0  0  0 │
│ 0  0  0  2.1 0  0  0  0│  ← Strong response at left edge
│ 0  0  1.8 0  0  0  0  0│  ← Strong response at left edge
│ 0  0  0  2.3 0  0  0  0│  ← Strong response at left edge
│ 0  0  0  2.1 0  0  0  0│  ← Strong response at left edge
│ 0  0  0  2.0 0  0  0  0│  ← Strong response at left edge
│ 0  0  0  1.9 0  0  0  0│  ← Strong response at left edge
│ 0  0  0  0  0  0  0  0 │
└──────────────────────────┘
```

**Interpretation:**
- High values indicate the filter detected its pattern
- Zeros indicate no match
- The **position** of high values tells you **where** the pattern appears

### Visualizing All 8 Feature Maps

**For a digit "5":**

```
Input:           Filter 1:        Filter 2:        Filter 3:        Filter 4:
1×28×28          (Vertical edges) (Horizontal)     (Diagonals)      (Curves)
┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐
│  5  │    →    │ │ │ │          │─────│          │ ╱ ╲ │          │ ◠ ◡ │
└─────┘          └─────┘          └─────┘          └─────┘          └─────┘

Filter 5:        Filter 6:        Filter 7:        Filter 8:
(Corners)        (Blobs)          (Texture)        (Loops)
┌─────┐          ┌─────┐          ┌─────┐          ┌─────┐
│ └ ┘ │          │ ● ● │          │·····│          │ ○ ○ │
└─────┘          └─────┘          └─────┘          └─────┘
```

**Why multiple feature maps matter:**
- Each digit has unique combinations of features
- "5" has top horizontal edge, bottom curve, vertical left edge
- "8" has two loops, multiple curves
- The **pattern of activations** across all 8 maps encodes the digit's identity

### Feature Map Dimensions

**Dimension formula:**
```
output_height = (input_height + 2×padding - kernel_size) / stride + 1
output_width  = (input_width  + 2×padding - kernel_size) / stride + 1
```

**For MNIST CNN:**
```
Input:    28×28, padding=1, kernel=3, stride=1
Output:   (28 + 2×1 - 3) / 1 + 1 = 28
Result:   28×28 (same size as input!)
```

**Why padding=1?**
- Without padding: 28 → 26 (shrinks by 2 pixels)
- With padding=1: 28 → 28 (preserves size)
- Preserving size is useful for early layers to maintain spatial resolution

---

## Pooling Operations

### What is Pooling?

**Pooling** (also called **downsampling** or **subsampling**) reduces the spatial dimensions of feature maps while preserving important information.

**Max pooling** is the most common type: takes the **maximum value** in each region.

### 2×2 Max Pooling Example

**Input feature map (4×4):**
```
┌─────────────────┐
│ 1.2  0.5 │ 2.1  0.8 │
│ 0.3  1.8 │ 0.4  1.2 │
├─────────┼──────────┤
│ 0.7  0.2 │ 3.5  1.1 │
│ 1.1  0.9 │ 0.6  2.8 │
└─────────────────┘
  Pool 1    Pool 2
  Pool 3    Pool 4
```

**Pooling operation:**
```
Pool 1: max(1.2, 0.5, 0.3, 1.8) = 1.8
Pool 2: max(2.1, 0.8, 0.4, 1.2) = 2.1
Pool 3: max(0.7, 0.2, 1.1, 0.9) = 1.1
Pool 4: max(3.5, 1.1, 0.6, 2.8) = 3.5
```

**Output (2×2):**
```
┌──────────┐
│ 1.8  2.1 │
│ 1.1  3.5 │
└──────────┘
```

**Result:** Input size reduced by 2× in each dimension (4×4 → 2×2).

### Why Pooling Works

**1. Dimensionality reduction:**
```
Before pooling: 8 × 28 × 28 = 6,272 values
After pooling:  8 × 14 × 14 = 1,568 values
Reduction: 4× fewer values to process
```

**2. Translation invariance:**
```
If feature shifts by 1 pixel within pool region, output doesn't change:

Original:           Shifted 1 pixel:    Pooled output:
┌─────────┐        ┌─────────┐        ┌─────┐
│ 0  0  5 │        │ 0  5  0 │        │  5  │  (Same!)
│ 0  0  0 │  vs    │ 0  0  0 │   →   │     │
└─────────┘        └─────────┘        └─────┘

Max is still 5, position doesn't matter within pool region.
```

**3. Focuses on "whether feature is present" rather than "exact position":**
- Max pooling keeps strongest activation
- Discards weak activations (likely noise)
- More robust to small shifts and distortions

**4. Reduces overfitting:**
- Fewer parameters in subsequent layers
- Forces network to learn more general features

### Implementation (from `mnist_cnn.rs`)

```rust
fn maxpool_forward(
    batch: usize,
    conv_act: &[f32],    // [batch, channels, 28, 28]
    pool_out: &mut [f32], // [batch, channels, 14, 14]
    pool_idx: &mut [u8],  // Stores argmax for backprop
) {
    for b in 0..batch {
        for c in 0..CONV_OUT {
            for py in 0..POOL_H {  // py = 0..14
                for px in 0..POOL_W {  // px = 0..14
                    // Top-left corner of 2×2 pool region
                    let iy0 = py * POOL;  // iy0 = 0, 2, 4, ..., 26
                    let ix0 = px * POOL;  // ix0 = 0, 2, 4, ..., 26

                    // Find maximum in 2×2 region
                    let mut best = -f32::INFINITY;
                    let mut best_idx = 0u8;

                    for dy in 0..POOL {  // dy = 0, 1
                        for dx in 0..POOL {  // dx = 0, 1
                            let iy = iy0 + dy;
                            let ix = ix0 + dx;
                            let v = conv_act[c, iy, ix];

                            if v > best {
                                best = v;
                                best_idx = (dy * POOL + dx) as u8; // 0,1,2,3
                            }
                        }
                    }

                    pool_out[c, py, px] = best;
                    pool_idx[c, py, px] = best_idx; // For backprop routing
                }
            }
        }
    }
}
```

**Key implementation details:**

1. **Argmax storage:** `pool_idx` stores which of the 4 positions had the max
   - During backprop, gradient flows only to the max position
   - Other 3 positions get zero gradient

2. **Non-overlapping pools:** Stride = pool size (2×2 pools with stride 2)
   - Each input pixel used exactly once
   - Output size = input size / pool size

3. **Per-channel pooling:** Each feature map pooled independently
   - 8 channels before → 8 channels after
   - Only spatial dimensions reduced

### Pooling Dimension Calculation

**Formula (for non-overlapping pools):**
```
output_height = input_height / pool_size
output_width  = input_width  / pool_size
```

**For MNIST CNN:**
```
Input:  8 × 28 × 28
Pool:   2×2
Output: 8 × 14 × 14
```

---

## Dimension Calculations

### Complete Network Dimensions

**Tracking shapes through the entire network:**

| Layer | Operation | Input Shape | Output Shape | Parameters |
|-------|-----------|-------------|--------------|------------|
| Input | - | [B, 1, 28, 28] | [B, 1, 28, 28] | 0 |
| Conv2D | 8 filters, 3×3, pad=1, stride=1 | [B, 1, 28, 28] | [B, 8, 28, 28] | 72 + 8 = 80 |
| ReLU | Element-wise max(0, x) | [B, 8, 28, 28] | [B, 8, 28, 28] | 0 |
| MaxPool | 2×2, stride=2 | [B, 8, 28, 28] | [B, 8, 14, 14] | 0 |
| Flatten | Reshape | [B, 8, 14, 14] | [B, 1568] | 0 |
| Dense | 1568 → 10 | [B, 1568] | [B, 10] | 15680 + 10 = 15690 |
| Softmax | Normalize | [B, 10] | [B, 10] | 0 |

**Total parameters:** 80 + 15,690 = **15,770 parameters**

**Compare to MLP:** 407,050 parameters → **96% reduction!**

### Dimension Formulas Reference

**Convolution output size:**
```rust
output_h = (input_h + 2 * padding - kernel_size) / stride + 1
output_w = (input_w + 2 * padding - kernel_size) / stride + 1
```

**Examples:**
```
Input: 28×28, kernel=3, padding=1, stride=1:
  Output = (28 + 2*1 - 3) / 1 + 1 = 28×28

Input: 28×28, kernel=3, padding=0, stride=1:
  Output = (28 + 2*0 - 3) / 1 + 1 = 26×26

Input: 28×28, kernel=5, padding=2, stride=1:
  Output = (28 + 2*2 - 5) / 1 + 1 = 28×28

Input: 32×32, kernel=3, padding=1, stride=2:
  Output = (32 + 2*1 - 3) / 2 + 1 = 16×16
```

**Max pooling output size (non-overlapping):**
```rust
output_h = input_h / pool_size
output_w = input_w / pool_size
```

**Example:**
```
Input: 28×28, pool=2:
  Output = 28/2 × 28/2 = 14×14
```

### Memory Requirements

**Forward pass memory (for batch_size=32):**

| Array | Shape | Elements | Memory (f32) |
|-------|-------|----------|--------------|
| Input | [32, 1, 28, 28] | 25,088 | 100 KB |
| Conv output | [32, 8, 28, 28] | 200,704 | 803 KB |
| Pool output | [32, 8, 14, 14] | 50,176 | 201 KB |
| Dense output | [32, 10] | 320 | 1.3 KB |
| **Total** | - | **~276K** | **~1.1 MB** |

**Compare to MLP (batch_size=32):**
- Hidden layer: [32, 512] = 16,384 elements
- Total activations: ~41K elements vs CNN's ~276K
- **CNN uses more memory** due to preserving spatial dimensions

**Trade-off:** More activation memory, but 96% fewer parameters!

---

## Forward Pass Walkthrough

### Example: Processing Digit "3"

Let's trace a single image through the network with concrete values.

**Step 1: Input (1×28×28)**

```
Raw image (8×8 excerpt, normalized to 0-1):
┌────────────────────────────────┐
│ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 │
│ 0.0  0.0  0.9  0.9  0.9  0.8  0.0  0.0 │
│ 0.0  0.0  0.0  0.0  0.0  0.9  0.0  0.0 │
│ 0.0  0.0  0.7  0.9  0.9  0.8  0.0  0.0 │
│ 0.0  0.0  0.0  0.0  0.0  0.9  0.0  0.0 │
│ 0.0  0.0  0.8  0.9  0.9  0.7  0.0  0.0 │
│ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 │
└────────────────────────────────┘
Shape: [1, 28, 28]
```

**Step 2: Conv2D → (8×28×28)**

Apply 8 different 3×3 filters. Here are 3 example filters:

```
Filter 0 (Vertical edges):     Filter 1 (Horizontal edges):   Filter 2 (Diagonal):
┌──────────┐                   ┌──────────┐                   ┌──────────┐
│ -1  0  1 │                   │ -1 -1 -1 │                   │  1 -1 -1 │
│ -1  0  1 │                   │  0  0  0 │                   │ -1  1 -1 │
│ -1  0  1 │                   │  1  1  1 │                   │ -1 -1  1 │
└──────────┘                   └──────────┘                   └──────────┘
```

**Output feature maps (8×28×28):**

```
Filter 0 output:              Filter 1 output:              Filter 2 output:
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ 0.0  0.0  0.0  0.0│         │ 0.0  0.0  0.0  0.0│         │ 0.0  0.0  0.0  0.0│
│ 0.0  1.8  0.1  0.9│         │ 0.0  2.1  1.8  1.2│         │ 0.0  0.3  1.1  0.2│
│ 0.0  0.2  0.0  1.7│         │ 0.0  0.0  0.0  0.1│         │ 0.0  0.8  0.4  0.9│
│ 0.0  1.5  0.2  0.8│         │ 0.0  1.9  1.7  1.5│         │ 0.0  0.1  0.7  0.3│
└──────────────────┘         └──────────────────┘         └──────────────────┘
  (Detects vertical edges)     (Detects horizontal edges)    (Detects diagonals)

... 5 more feature maps for filters 3-7 ...
Shape: [8, 28, 28]
```

**Step 3: ReLU → (8×28×28)**

```
Before ReLU:                  After ReLU:
┌──────────────────┐         ┌──────────────────┐
│ -0.3  1.8  0.1  0.9│         │  0.0  1.8  0.1  0.9│
│  0.2 -0.1  0.0  1.7│    →   │  0.2  0.0  0.0  1.7│
│ -0.5  1.5  0.2 -0.8│         │  0.0  1.5  0.2  0.0│
└──────────────────┘         └──────────────────┘
(Negative values zeroed out)
Shape: [8, 28, 28] (unchanged)
```

**Step 4: MaxPool (2×2) → (8×14×14)**

```
Before pooling (4×4 excerpt):   After pooling (2×2):
┌─────────────────┐            ┌──────────┐
│ 1.8  0.1 │ 0.9  1.2│           │ 1.8  1.2 │
│ 0.2  0.0 │ 0.0  1.7│    →     │ 1.5  2.1 │
├─────────┼──────────┤           └──────────┘
│ 0.0  1.5 │ 0.2  0.0│
│ 0.8  0.3 │ 2.1  0.9│
└─────────────────┘
Shape: [8, 14, 14]  (Spatial dimensions halved)
```

**Step 5: Flatten → (1568,)**

```
Feature maps (8×14×14):         Flattened vector (1568,):
┌─────┐ ┌─────┐ ┌─────┐        [1.8, 1.2, 1.5, 2.1, ..., 0.3, 0.9, 0.7, ...]
│ map0│ │ map1│ │ map2│         │                                        │
└─────┘ └─────┘ └─────┘         └─────── 8 × 14 × 14 = 1568 values ─────┘
   ...                          Shape: [1568]
```

**Step 6: Dense (1568 → 10) → (10,)**

```
Matrix multiplication: [1568] × [1568×10] = [10]

Output (logits):
[
  -2.1,  # Score for digit 0
  -1.3,  # Score for digit 1
   0.5,  # Score for digit 2
   4.2,  # Score for digit 3 ← Highest!
  -0.8,  # Score for digit 4
   1.1,  # Score for digit 5
  -1.9,  # Score for digit 6
   0.2,  # Score for digit 7
  -0.5,  # Score for digit 8
   0.9   # Score for digit 9
]
Shape: [10]
```

**Step 7: Softmax → (10,)**

```
Softmax converts logits to probabilities (sum = 1.0):

probabilities = exp(logits) / sum(exp(logits))

Output (probabilities):
[
  0.01,  # P(digit 0) = 1%
  0.02,  # P(digit 1) = 2%
  0.12,  # P(digit 2) = 12%
  0.73,  # P(digit 3) = 73% ← Prediction!
  0.03,  # P(digit 4) = 3%
  0.04,  # P(digit 5) = 4%
  0.01,  # P(digit 6) = 1%
  0.02,  # P(digit 7) = 2%
  0.01,  # P(digit 8) = 1%
  0.01   # P(digit 9) = 1%
]
Shape: [10]
Sum: 1.00 (100%)

Prediction: argmax(probabilities) = 3
Confidence: 73%
```

### Summary of Transformations

```
Input image:        1 × 28 × 28 = 784 values
  ↓ Conv2D (8 filters)
Feature maps:       8 × 28 × 28 = 6,272 values (spatial structure preserved)
  ↓ ReLU
Activated maps:     8 × 28 × 28 = 6,272 values (negatives removed)
  ↓ MaxPool (2×2)
Pooled maps:        8 × 14 × 14 = 1,568 values (downsampled 4×)
  ↓ Flatten
Vector:             1,568 values (ready for dense layer)
  ↓ Dense (1568→10)
Logits:             10 values (class scores)
  ↓ Softmax
Probabilities:      10 values (normalized predictions)
```

---

## Backward Pass and Gradients

### Gradient Flow Overview

**Backward pass computes gradients from output to input:**

```
Forward:  Input → Conv → ReLU → Pool → Dense → Softmax → Loss
Backward: Input ← Conv ← ReLU ← Pool ← Dense ← Softmax ← Loss
             ↑      ↑      ↑      ↑       ↑        ↑
          ∂L/∂W  ∂L/∂x  ∂L/∂x  ∂L/∂W   ∂L/∂W    ∂L/∂y
```

**Key challenges for CNNs:**

1. **MaxPool backward**: Gradients only flow through max positions
2. **Conv backward**: Must compute gradients w.r.t. both weights and inputs
3. **Spatial structure**: Must preserve dimensions correctly

### MaxPool Backward Pass

**Problem:** During forward pass, max pooling selects one value from each 2×2 region. During backward pass, gradient flows **only to that selected position**.

**Example:**

```
Forward:
Input (4×4):                      Output (2×2):
┌─────────────────┐              ┌──────────┐
│ 1.2  0.5 │ 2.1  0.8 │  →        │ 1.8  2.1 │
│ 0.3  1.8*│ 0.4  1.2 │  →        │ 1.1  3.5 │
├─────────┼──────────┤            └──────────┘
│ 0.7  0.2 │ 3.5* 1.1 │
│ 1.1* 0.9 │ 0.6  2.8 │
└─────────────────┘
(* marks max positions)

Backward:
Gradient from next layer (2×2):  Gradient to previous layer (4×4):
┌──────────┐                    ┌─────────────────┐
│ 0.5  0.3 │                    │ 0.0  0.0 │ 0.3  0.0 │
│ 0.2  0.8 │  ←                 │ 0.0  0.5 │ 0.0  0.0 │
└──────────┘                    ├─────────┼──────────┤
                                │ 0.0  0.0 │ 0.8  0.0 │
                                │ 0.2  0.0 │ 0.0  0.0 │
                                └─────────────────┘
                                (Gradient routed to max positions only)
```

**Implementation (from `mnist_cnn.rs`):**

```rust
fn maxpool_backward_relu(
    batch: usize,
    conv_act: &[f32],       // Forward activations (to check ReLU mask)
    pool_grad: &[f32],      // Gradient from next layer [batch, 8, 14, 14]
    pool_idx: &[u8],        // Stored argmax indices (0-3 for 2×2)
    conv_grad: &mut [f32],  // Output gradient [batch, 8, 28, 28]
) {
    // Zero output gradient
    for v in conv_grad.iter_mut() {
        *v = 0.0;
    }

    for b in 0..batch {
        for c in 0..CONV_OUT {
            for py in 0..POOL_H {
                for px in 0..POOL_W {
                    let g = pool_grad[b, c, py, px];  // Gradient at pool output
                    let argmax = pool_idx[b, c, py, px] as usize;  // Which position had max? (0-3)

                    // Decode argmax to (dy, dx) offset
                    let dy = argmax / POOL;  // 0 or 1
                    let dx = argmax % POOL;  // 0 or 1

                    // Map to input position
                    let iy = py * POOL + dy;
                    let ix = px * POOL + dx;

                    // Route gradient to max position
                    // Also apply ReLU mask (zero gradient if forward activation was < 0)
                    if conv_act[b, c, iy, ix] > 0.0 {
                        conv_grad[b, c, iy, ix] = g;
                    }
                }
            }
        }
    }
}
```

**Key insight:** The `pool_idx` array stored during forward pass tells us exactly where to route gradients.

### Conv2D Backward Pass

**Convolution backward computes two gradients:**

1. **Gradient w.r.t. weights (∂L/∂W)**: How much to adjust each filter
2. **Gradient w.r.t. inputs (∂L/∂x)**: Pass gradient to previous layer

**High-level idea:**

```
Forward:  output[oy, ox] = Σ input[iy, ix] × weight[ky, kx] + bias
                           ky,kx

Backward (weight gradient):
  ∂L/∂weight[ky, kx] = Σ input[iy, ix] × grad_output[oy, ox]
                       oy,ox

Backward (input gradient):
  ∂L/∂input[iy, ix] = Σ weight[ky, kx] × grad_output[oy, ox]
                      ky,kx
```

**Implementation (simplified from `src/layers/conv2d.rs`):**

```rust
fn backward(&self, input: &[f32], grad_output: &[f32], grad_input: &mut [f32], batch_size: usize) {
    // Zero gradients
    for v in grad_input.iter_mut() {
        *v = 0.0;
    }

    for b in 0..batch_size {
        for oc in 0..out_channels {
            // Compute bias gradient (sum all spatial positions)
            for oy in 0..output_h {
                for ox in 0..output_w {
                    grad_biases[oc] += grad_output[b, oc, oy, ox];
                }
            }

            // Compute weight and input gradients
            for ic in 0..in_channels {
                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let g = grad_output[b, oc, oy, ox];

                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let iy = oy * stride + ky - padding;
                                let ix = ox * stride + kx - padding;

                                if iy >= 0 && iy < input_h && ix >= 0 && ix < input_w {
                                    // Weight gradient
                                    grad_weights[oc, ic, ky, kx] += input[b, ic, iy, ix] * g;

                                    // Input gradient
                                    grad_input[b, ic, iy, ix] += weights[oc, ic, ky, kx] * g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**Complexity:** Backward pass has same computational cost as forward pass (same nested loops).

### ReLU Backward Pass

**ReLU forward:** `y = max(0, x)`

**ReLU backward:** `∂L/∂x = ∂L/∂y if x > 0, else 0`

```rust
// ReLU backward (element-wise)
for i in 0..n {
    if forward_activation[i] > 0.0 {
        grad_input[i] = grad_output[i];  // Pass gradient through
    } else {
        grad_input[i] = 0.0;  // Block gradient
    }
}
```

**Note:** ReLU backward is often fused with pooling backward for efficiency (as in `maxpool_backward_relu`).

---

## Parameter Efficiency

### Comparison: CNN vs MLP

**MNIST MLP (from Tutorial 02):**
```
Architecture: 784 → 512 → 10
Parameters:
  - Dense1: 784 × 512 + 512 = 401,920
  - Dense2: 512 × 10 + 10 = 5,130
  Total: 407,050 parameters
Test accuracy: ~97%
```

**MNIST CNN (this tutorial):**
```
Architecture: 1×28×28 → Conv(8) → Pool → Dense(10)
Parameters:
  - Conv: 8 × 1 × 3 × 3 + 8 = 80
  - Dense: 1568 × 10 + 10 = 15,690
  Total: 15,770 parameters
Test accuracy: ~98%
```

**Dramatic improvement:**
- **96% fewer parameters** (407K → 16K)
- **Better accuracy** (97% → 98%)
- **Faster training** (fewer parameters to update)

### Why is CNN More Efficient?

**1. Parameter sharing:**

MLP:
```
Each of 784 input pixels has separate weights to each of 512 hidden neurons.
Total: 784 × 512 = 401,408 unique weights

If a vertical edge appears at position (5,5) or (10,10),
different weights must learn to detect it.
```

CNN:
```
One 3×3 filter (9 weights) slides over entire image.
Total: 8 filters × 9 weights = 72 weights

Same vertical edge detector applied everywhere.
Position-invariant pattern detection.
```

**2. Local connectivity:**

MLP:
```
Every input pixel connected to every hidden neuron.
Connections: 784 × 512 = 401,408
Most connections learn weak, irrelevant patterns.
```

CNN:
```
Each output pixel connected to 3×3 = 9 input pixels.
Connections: 8 filters × 9 positions = 72 per output pixel
Focuses on local spatial relationships.
```

**3. Hierarchical features:**

MLP:
```
Hidden layer must learn all patterns from raw pixels.
No built-in notion of spatial structure.
Inefficient representation.
```

CNN:
```
Conv layer learns simple patterns (edges).
Pooling layer aggregates patterns into regions.
Dense layer combines high-level features.
Efficient hierarchical representation.
```

### Parameter Count Breakdown

**Convolutional layer:**
```
Filters: out_channels × in_channels × kernel_h × kernel_w
  = 8 × 1 × 3 × 3 = 72 weights
Biases: out_channels = 8
Total: 80 parameters
```

**Dense layer:**
```
Weights: input_size × output_size
  = 1568 × 10 = 15,680
Biases: output_size = 10
Total: 15,690 parameters
```

**Total: 15,770 parameters**

**Memory footprint:**
```
Parameters: 15,770 × 4 bytes (f32) = 63 KB
Activations (batch=32): ~1.1 MB (see Dimension Calculations section)
Total: ~1.2 MB (fits easily in CPU cache!)
```

---

## Verification and Expected Outputs

### Checkpoint 1: Conv2D Output Dimensions

**After first conv layer:**
```rust
// Check dimensions
let input_shape = [batch_size, 1, 28, 28];
let output_shape = [batch_size, 8, 28, 28];

// Verify calculation
let expected_h = (28 + 2*1 - 3) / 1 + 1;  // = 28
let expected_w = (28 + 2*1 - 3) / 1 + 1;  // = 28
assert_eq!(expected_h, 28);
assert_eq!(expected_w, 28);
```

**Expected:** 8 feature maps, each 28×28 (same as input spatial size).

### Checkpoint 2: ReLU Activation

**After ReLU:**
```rust
// Check that negative values are zeroed
for i in 0..conv_output.len() {
    assert!(conv_output[i] >= 0.0, "ReLU output must be non-negative");
}

// Check that positive values are unchanged
// (compare to conv output before ReLU)
```

**Expected:** All values ≥ 0, typically in range [0, 10] depending on input.

### Checkpoint 3: MaxPool Output Dimensions

**After 2×2 max pooling:**
```rust
let pool_input_shape = [batch_size, 8, 28, 28];
let pool_output_shape = [batch_size, 8, 14, 14];

// Verify calculation
let expected_h = 28 / 2;  // = 14
let expected_w = 28 / 2;  // = 14
assert_eq!(expected_h, 14);
assert_eq!(expected_w, 14);
```

**Expected:** 8 feature maps, each 14×14 (half the spatial size).

### Checkpoint 4: Flattening

**After flatten:**
```rust
let flattened_size = 8 * 14 * 14;
assert_eq!(flattened_size, 1568);

// Verify flattened vector length
assert_eq!(dense_input.len(), batch_size * 1568);
```

**Expected:** Vector of length 1568 per sample in batch.

### Checkpoint 5: Final Output

**After dense + softmax:**
```rust
// Check output is valid probability distribution
for b in 0..batch_size {
    let mut sum = 0.0;
    for c in 0..10 {
        let p = output[b * 10 + c];
        assert!(p >= 0.0 && p <= 1.0, "Probability out of range");
        sum += p;
    }
    assert!((sum - 1.0).abs() < 1e-5, "Probabilities don't sum to 1");
}
```

**Expected:** 10 probabilities per sample, each in [0,1], summing to 1.0.

### Expected Training Output

**From `mnist_cnn.rs` training logs:**

```
Epoch 1/3, Loss: 0.3247, Val Loss: 0.1823, Val Acc: 95.2%, Time: 23.4s
Epoch 2/3, Loss: 0.1456, Val Loss: 0.1124, Val Acc: 97.1%, Time: 22.8s
Epoch 3/3, Loss: 0.0982, Val Loss: 0.0876, Val Acc: 97.8%, Time: 23.1s
Best model saved: mnist_cnn_model_best.bin
Test Accuracy: 97.9% (9790/10000)
```

**Expected accuracy milestones:**
- **After 1 epoch:** ~95% validation accuracy
- **After 3 epochs:** ~98% validation accuracy
- **Test set:** ~98% final accuracy

**Compare to MLP (Tutorial 02):**
- MLP after 3 epochs: ~96.5%
- CNN after 3 epochs: ~98%
- **1.5% absolute improvement with 96% fewer parameters!**

### Common Issues and Debugging

**Issue 1: Dimension mismatch in flatten**
```
Error: "Cannot reshape [batch, 8, 28, 28] to [batch, 1568]"
Cause: Forgot to apply max pooling (28×28 instead of 14×14)
Fix: Ensure maxpool_forward is called before flattening
```

**Issue 2: All feature maps look the same**
```
Symptom: Conv outputs have identical values across all 8 channels
Cause: Filters not initialized with different random weights
Fix: Check weight initialization in Conv2DLayer::new()
```

**Issue 3: NaN values after training**
```
Symptom: Loss becomes NaN after a few batches
Cause: Learning rate too high, or gradient explosion
Fix: Reduce learning rate (try 0.001 instead of 0.01)
      Or check for division by zero in softmax
```

**Issue 4: Poor test accuracy (<90%)**
```
Symptom: Training loss decreases but test accuracy stuck at 85%
Cause: Overfitting (memorizing training data)
Fix: Add validation split, early stopping
      Or reduce model capacity (fewer filters)
```

---

## Exercises

### Beginner Level

**Exercise 1: Experiment with filter count**

Modify `CONV_OUT` in `mnist_cnn.rs`:
```rust
const CONV_OUT: usize = 4;  // Try 4, 8, 16, 32
```

**Expected results:**
- 4 filters: ~96% accuracy, faster training
- 8 filters: ~98% accuracy (baseline)
- 16 filters: ~98.5% accuracy, slower training
- 32 filters: ~98.5% accuracy, much slower, diminishing returns

**Question:** Why doesn't accuracy improve much beyond 16 filters?

**Exercise 2: Visualize learned filters**

After training, save filter weights and visualize:
```rust
// Save first filter (8×1×3×3)
for i in 0..8 {
    let filter = &conv_layer.weights()[i*9..(i+1)*9];
    println!("Filter {}: {:?}", i, filter);
}
```

**Expected:** Filters should show edge patterns (not random noise).

**Exercise 3: Try different padding**

Modify padding in Conv2D initialization:
```rust
let conv_layer = Conv2DLayer::new(1, 8, 3, 0, 1, 28, 28, &mut rng);  // padding=0
```

**Expected:**
- Output shape: 8×26×26 (shrinks by 2 pixels)
- After pooling: 8×13×13 = 1352 (instead of 1568)
- Must update `FC_IN` constant accordingly
- Accuracy: Similar (~98%), slightly less spatial resolution

### Intermediate Level

**Exercise 4: Add a second conv layer**

**Modify architecture:**
```
1×28×28 → Conv(8, 3×3, pad=1) → ReLU → MaxPool(2×2)
        → Conv(16, 3×3, pad=1) → ReLU → MaxPool(2×2) → Dense(10)
```

**Expected dimensions:**
```
Input:      [B, 1, 28, 28]
Conv1:      [B, 8, 28, 28]
Pool1:      [B, 8, 14, 14]
Conv2:      [B, 16, 14, 14]  # 16 filters on 8 input channels
Pool2:      [B, 16, 7, 7]
Flatten:    [B, 784]          # 16 × 7 × 7 = 784
Dense:      [B, 10]
```

**Expected results:**
- Parameters: ~8K (still much less than MLP!)
- Accuracy: ~98.5-99% (slight improvement)
- Training time: 2× slower (two conv layers)

**Exercise 5: Experiment with kernel size**

Try 5×5 filters instead of 3×3:
```rust
const KERNEL: usize = 5;
const PAD: isize = 2;  // Maintain same output size
```

**Expected:**
- Parameters: 8 × 1 × 5 × 5 = 200 (vs 72 for 3×3)
- Receptive field: Larger (sees 5×5 patches instead of 3×3)
- Accuracy: Similar (~98%), slightly better for large features
- Training time: ~2× slower (more computation per position)

**Question:** Why do most modern CNNs use 3×3 filters instead of 5×5 or 7×7?

**Exercise 6: Replace MaxPool with AveragePool**

**Implement average pooling:**
```rust
fn avgpool_forward(conv_act: &[f32], pool_out: &mut [f32]) {
    for py in 0..POOL_H {
        for px in 0..POOL_W {
            let iy0 = py * POOL;
            let ix0 = px * POOL;

            let mut sum = 0.0;
            for dy in 0..POOL {
                for dx in 0..POOL {
                    sum += conv_act[(iy0 + dy) * IMG_W + (ix0 + dx)];
                }
            }
            pool_out[py * POOL_W + px] = sum / (POOL * POOL) as f32;
        }
    }
}
```

**Expected:**
- Accuracy: ~97% (slightly worse than MaxPool)
- MaxPool better for sparse features (detects "is feature present anywhere?")
- AvgPool better for dense features (preserves magnitude information)

### Advanced Level

**Exercise 7: Implement strided convolution instead of pooling**

**Replace:**
```
Conv(8, 3×3, stride=1, pad=1) → ReLU → MaxPool(2×2)
```

**With:**
```
Conv(8, 3×3, stride=2, pad=1) → ReLU
```

**Expected dimensions:**
```
Input:   [B, 1, 28, 28]
Conv:    [B, 8, 14, 14]  # Stride=2 downsamples directly
Output:  [B, 8, 14, 14]  # No pooling needed
```

**Expected:**
- Parameters: Same (80)
- Speed: Faster (no separate pooling step)
- Accuracy: ~97.5% (slightly worse than MaxPool)
- Modern trend: Strided convs replacing pooling in many architectures

**Exercise 8: Visualize feature maps**

**Save intermediate activations:**
```rust
// After conv + ReLU
let feature_maps = &conv_output[0..8*28*28];  // First sample in batch
// Save to file for visualization
```

**Then visualize in Python:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Load feature maps
fmaps = np.fromfile('feature_maps.bin', dtype=np.float32).reshape(8, 28, 28)

# Plot all 8 feature maps
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(fmaps[i], cmap='viridis')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
plt.show()
```

**Expected:** See different patterns activated for edges, curves, textures.

**Exercise 9: Implement batch normalization**

**Add batch norm after conv, before ReLU:**
```rust
// Normalize across batch dimension
for c in 0..CONV_OUT {
    let mut mean = 0.0;
    let mut var = 0.0;

    // Compute mean
    for b in 0..batch_size {
        for y in 0..28 {
            for x in 0..28 {
                mean += conv_output[b, c, y, x];
            }
        }
    }
    mean /= (batch_size * 28 * 28) as f32;

    // Compute variance
    // ... (similar loop)

    // Normalize
    for b in 0..batch_size {
        for y in 0..28 {
            for x in 0..28 {
                conv_output[b, c, y, x] = (conv_output[b, c, y, x] - mean) / (var + 1e-5).sqrt();
            }
        }
    }
}
```

**Expected:**
- Faster convergence (less sensitive to learning rate)
- Better accuracy (~98.5%)
- More stable training

---

## Next Steps

### What You've Learned

Congratulations! You now understand:

✅ How convolution operations detect local spatial patterns
✅ Why parameter sharing makes CNNs efficient for images
✅ How pooling downsamples while preserving features
✅ How to track dimensions through conv and pool layers
✅ What learned filters and feature maps look like
✅ Why CNNs achieve better accuracy with fewer parameters than MLPs

### Next Tutorial: CIFAR-10 CNN

**Building on this foundation:**
- **Multi-channel inputs:** RGB images (3 channels) instead of grayscale (1 channel)
- **Deeper networks:** Multiple conv layers with increasing filter counts
- **Harder problem:** 10 classes of natural images (not just digits)
- **Data augmentation:** Random flips, crops to improve generalization

**CIFAR-10 architecture preview:**
```
3×32×32 → Conv(16, 3×3) → ReLU → MaxPool(2×2)
        → Conv(32, 3×3) → ReLU → MaxPool(2×2)
        → Conv(64, 3×3) → ReLU → MaxPool(2×2)
        → Dense(10)
```

**Expected:** ~70% accuracy (vs 10% random guessing)

---

## Related Documentation

**Mathematical foundations:**
- [Backpropagation Guide](../backpropagation/README.md) - Gradient computation theory
- [Conv2D Layer Backpropagation](../backpropagation/conv2d_layer.md) - Mathematical derivations of convolutional gradients
- [Dense Layer Backpropagation](../backpropagation/dense_layer.md) - Fully connected layer gradients
- [Activation Functions](../activation_functions.md) - ReLU, LeakyReLU, GELU, and alternatives
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) - Notation conventions

**Architecture design:**
- [CIFAR-10 Architecture Design](../cifar10_architecture_design.md) - Design principles for deeper CNNs
- [Hyperparameters Guide](../hyperparameters.md) - Learning rate, batch size, optimizer selection
- [Configuration System](../architecture_config.md) - JSON-based hyperparameter configs

**Implementation details:**
- `mnist_cnn.rs` - Full MNIST CNN implementation (this tutorial's code)
- `src/layers/trait.rs` - Layer trait interface
- `src/layers/conv2d.rs` - Full Conv2D layer implementation with backpropagation
- `src/layers/dense.rs` - Dense layer with BLAS acceleration
- `src/utils/activations.rs` - Activation functions (ReLU, softmax, etc.)
- `src/data/mnist.rs` - MNIST IDX format loader
- `tests/test_conv2d.rs` - Conv2D layer correctness tests
- `tests/test_backward_pass.rs` - Gradient validation

**Training infrastructure:**
- `config/training/mnist_cnn_default.json` - Default CNN training configuration
- `config/mnist_mlp_adam.json` - Adam optimizer example
- `config/mnist_mlp_cosine.json` - Cosine annealing scheduler example

**Related tutorials:**
- [Tutorial 01: XOR MLP](01_xor_mlp.md) - Build the foundational 2→4→1 network
- [Tutorial 02: MNIST MLP](02_mnist_mlp.md) - Scale to 784→512→10 digit classifier

**Related architectures:**
- `mnist_mlp.rs` - Feedforward network for comparison (previous tutorial)
- `mnist_attention_pool.rs` - Transformer-style attention mechanism (alternative to convolution)
- `cifar10_cnn.rs` - Multi-channel RGB CNN for color images (coming soon in Tutorial 04)

---

## Experimentation Ideas

**Architecture variations:**
1. Try different numbers of filters: 4, 8, 16, 32
2. Experiment with kernel sizes: 3×3, 5×5, 7×7
3. Add more conv layers (deeper networks)
4. Replace pooling with strided convolutions

**Training improvements:**
1. Try different optimizers: Adam, AdamW, RMSprop (see `config/*.json`)
2. Experiment with learning rate schedules (cosine annealing, step decay)
3. Add data augmentation (random flips, shifts)
4. Implement dropout for regularization

**Analysis:**
1. Visualize learned filters and feature maps
2. Plot training curves (loss, accuracy over time)
3. Analyze failure cases (which digits are confused?)
4. Compare CNN vs MLP on same dataset

---

## Summary

**Congratulations!** You've mastered Convolutional Neural Networks and understand how they revolutionized computer vision. You now know:

✅ How convolution operations detect local spatial patterns
✅ Why parameter sharing makes CNNs efficient for images (31× fewer parameters!)
✅ How pooling downsamples while preserving features
✅ How to track dimensions through conv and pool layers
✅ What learned filters and feature maps look like
✅ Why CNNs achieve better accuracy (~98%) with fewer parameters than MLPs (~97%)

**This prepares you for:**
- Multi-channel RGB image processing (CIFAR-10, ImageNet)
- Deeper convolutional architectures (VGG, ResNet)
- Object detection and segmentation
- Modern vision transformers (ViT, DINO)

**Keep experimenting!** Try the exercises above, visualize your learned filters, and explore the codebase. When you're ready for color images and deeper networks, move on to CIFAR-10 CNN (Tutorial 04, coming soon). Happy learning! 🚀

---

**Navigation:**
← [Previous Tutorial: MNIST MLP](02_mnist_mlp.md) | [Tutorial Index](README.md)
