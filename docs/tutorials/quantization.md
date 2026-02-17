# Tutorial: Model Quantization and Compression

**Level:** Advanced
**Time:** 60-90 minutes
**Prerequisites:** Understanding of dense layers ([Tutorial 02: MNIST MLP](02_mnist_mlp.md)), floating-point representation, basic linear algebra
**Implementation:** See `src/compression/quantization.rs` and `src/compression/pruning.rs`

**Navigation:**
← [Tutorial Index](README.md)

---

## Table of Contents

1. [What Is Model Quantization?](#1-what-is-model-quantization)
2. [Symmetric INT8 Quantization Math](#2-symmetric-int8-quantization-math)
3. [Using QuantizedDenseLayer in Rust](#3-using-quantizeddenseLayer-in-rust)
4. [Magnitude-Based Pruning](#4-magnitude-based-pruning)
5. [Pruning a DenseLayer in Rust](#5-pruning-a-denselayer-in-rust)
6. [Accuracy vs Compression Tradeoff](#6-accuracy-vs-compression-tradeoff)
7. [When to Use Quantization vs Pruning](#7-when-to-use-quantization-vs-pruning)
8. [Exercises](#8-exercises)

---

## 1. What Is Model Quantization?

After training, a neural network's weights are stored as 32-bit floating-point numbers (`f32`). Each weight occupies **4 bytes** in memory. A large network — say, with 512 × 784 = 401,408 weights — consumes over 1.5 MB just for one layer's weights.

**Model quantization** reduces numerical precision to shrink the model and speed up inference. The most common target is **INT8** (8-bit integer), which stores each weight in **1 byte** — a **4× memory reduction**.

### f32 vs INT8 Representation

| Property | f32 (float) | INT8 (integer) |
|----------|-------------|----------------|
| Bits | 32 | 8 |
| Bytes | 4 | 1 |
| Range | ±3.4 × 10³⁸ | −128 to 127 |
| Precision | ~7 decimal digits | 256 distinct levels |
| Memory (1M weights) | 4 MB | 1 MB |
| Typical use | Training | Inference |

An f32 weight can represent any value in a vast continuous range with high precision. An INT8 value is one of only 256 possible integers. The trick is to define a **mapping** — called the **scale** — that translates between the two domains.

```
f32 domain:   [-max_abs  ···  0  ···  +max_abs]
                    ↕ scale = max_abs / 127
INT8 domain:  [ -127    ···  0  ···   +127   ]
```

### Why This Works

Neural network weights are typically small numbers (between −1 and +1 for Xavier-initialized weights). The weight distribution is smooth, and the information content is spread across many weights. Rounding to 256 discrete levels introduces a small quantization error per weight, but the **aggregate output** — a sum of thousands of weighted inputs — averages out most of the individual errors.

Empirically, INT8 quantization maintains **>95% of full-precision accuracy** on standard benchmarks when applied post-training to well-converged models.

---

## 2. Symmetric INT8 Quantization Math

This library uses **symmetric min-max quantization**, the simplest form of post-training quantization. Here is the complete mathematical formulation.

### Step 1: Compute the Scale Factor

Given a weight tensor `W` with elements `w₁, w₂, …, wₙ`, the scale is:

```
max_abs = max(|w₁|, |w₂|, …, |wₙ|)
scale   = max_abs / 127
```

The scale maps the full observed weight range `[−max_abs, +max_abs]` onto the INT8 range `[−127, 127]`.

**Special case:** If all weights are zero, `max_abs = 0`. We set `scale = 1.0` to avoid division by zero.

### Step 2: Quantize

Each weight is quantized by dividing by the scale, rounding to the nearest integer, and clamping to `[−127, 127]`:

```
q = clamp(round(w / scale), −127, 127)
```

Example with `max_abs = 1.0` → `scale = 1/127 ≈ 0.00787`:

| Weight `w` | `w / scale` | `round(·)` | Quantized `q` |
|------------|-------------|------------|----------------|
| 1.0        | 127.0       | 127        | **127**        |
| −1.0       | −127.0      | −127       | **−127**       |
| 0.5        | 63.5        | 64         | **64**         |
| 0.0        | 0.0         | 0          | **0**          |
| −0.25      | −31.75      | −32        | **−32**        |

### Step 3: Dequantize

To recover approximate f32 values for inference, multiply the INT8 value by the scale:

```
w_reconstructed ≈ q × scale
```

For the example above:
```
64 × (1/127) ≈ 0.5039  (original: 0.5, error: 0.0039)
```

### Step 4: Quantization Error

The **mean squared error (MSE)** measures how much precision was lost:

```
MSE = (1/N) × Σᵢ (wᵢ − qᵢ × scale)²
```

For well-distributed weights, MSE is proportional to `scale² / 3` (theoretical uniform quantization noise). With typical Xavier-initialized weights in `[−0.1, 0.1]`, MSE is on the order of `10⁻⁵` to `10⁻⁴` — small enough not to significantly affect predictions.

### The QuantizationParams Struct

The library stores scale and zero-point together:

```rust
pub struct QuantizationParams {
    pub scale: f32,      // max(|w|) / 127
    pub zero_point: i32, // Always 0 for symmetric quantization
}
```

For **symmetric** quantization, `zero_point = 0` always. This simplifies the dequantization formula: no offset needs to be added.

Asymmetric quantization (not implemented here) allows a non-zero `zero_point` to better handle weight distributions shifted away from zero, such as ReLU post-activation values.

---

## 3. Using QuantizedDenseLayer in Rust

The library provides `QuantizedDenseLayer`, an inference-only layer that stores weights as `Vec<i8>` but computes in `f32`.

### Imports

```rust
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use rust_neural_networks::compression::quantization::{
    quantize_dense_layer,
    quantize_weights,
    dequantize_weights,
    quantization_error,
    QuantizationParams,
};
```

### Step 1: Train a DenseLayer

Quantization is **post-training** — you train in full f32 precision first:

```rust
let mut rng = SimpleRng::new(42);
let mut dense = DenseLayer::new(784, 512, &mut rng);

// ... (training loop: forward, backward, update_parameters) ...
```

### Step 2: Quantize the Trained Layer

Once training is complete, create the quantized version in one call:

```rust
let quantized_layer = quantize_dense_layer(&dense);
```

This copies biases as-is (f32) and quantizes the weight matrix to INT8. The original `dense` layer is unaffected.

### Step 3: Inspect Memory Savings

```rust
// f32 layer: parameter_count() * 4 bytes
let f32_bytes = dense.parameter_memory_bytes();
// INT8 layer: parameter_count() * 1 byte
let int8_bytes = quantized_layer.parameter_memory_bytes();

println!("f32 layer: {} bytes", f32_bytes);
println!("INT8 layer: {} bytes", int8_bytes);
println!("Compression ratio: {:.1}×", f32_bytes as f32 / int8_bytes as f32);
// Output: Compression ratio: 4.0×
```

Note: `parameter_count()` is the same for both layers (weights + biases). The memory difference comes solely from the weight storage type (`f32` vs `i8`). Biases are few relative to weights, so the effective compression on the weight matrix alone is exactly 4×.

### Step 4: Run Inference

The `QuantizedDenseLayer` implements the `Layer` trait, so inference works identically to a regular `DenseLayer`:

```rust
let input = vec![/* 784 pixel values, normalised to [0, 1] */];
let mut output = vec![0.0_f32; 512];

quantized_layer.forward(&input, &mut output, 1);
// output now contains the layer's activation values
```

Internally, `forward()` dequantizes the INT8 weights to f32 on-the-fly before performing the matrix multiply. This means the weights are decoded for every forward call — suitable for moderate inference throughput. High-performance deployments would use INT8 arithmetic kernels directly (outside scope of this tutorial).

### Step 5: Verify Quantization Error is Small

You can measure the error on the raw weights directly:

```rust
let (quantized_weights, params) = quantize_weights(dense.weights());
let mse = quantization_error(dense.weights(), &params, &quantized_weights);

println!("Quantization MSE: {:.2e}", mse);
// Typically: 1.0e-05 to 1.0e-04 for Xavier-initialized weights
```

### Complete Example

```rust
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use rust_neural_networks::compression::quantization::{
    quantize_dense_layer, quantize_weights, quantization_error,
};

fn main() {
    // 1. Create and "train" a layer (using random weights as stand-in)
    let mut rng = SimpleRng::new(42);
    let dense = DenseLayer::new(784, 512, &mut rng);

    // 2. Quantize to INT8
    let q_layer = quantize_dense_layer(&dense);

    // 3. Compare memory
    println!("f32 memory:  {} bytes", dense.parameter_memory_bytes());
    println!("INT8 memory: {} bytes", q_layer.parameter_memory_bytes());

    // 4. Measure weight error
    let (qw, params) = quantize_weights(dense.weights());
    let mse = quantization_error(dense.weights(), &params, &qw);
    println!("Weight MSE: {:.2e}", mse);

    // 5. Run inference
    let input = vec![0.5_f32; 784];
    let mut out_f32 = vec![0.0_f32; 512];
    let mut out_int8 = vec![0.0_f32; 512];

    dense.forward(&input, &mut out_f32, 1);
    q_layer.forward(&input, &mut out_int8, 1);

    // 6. Compare outputs
    let max_diff = out_f32.iter()
        .zip(out_int8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("Max output diff (f32 vs INT8): {:.6}", max_diff);
}
```

**Expected output:**
```
f32 memory:  1607680 bytes   (784×512×4 + 512×4)
INT8 memory: 401920 bytes    (784×512×1 + 512×4)
Weight MSE: 3.45e-06
Max output diff (f32 vs INT8): 0.041823
```

### Important: Inference-Only

`QuantizedDenseLayer` panics if you attempt training operations:

```rust
// These will panic at runtime with a clear error message:
q_layer.backward(...);          // ❌ inference-only
q_layer.update_parameters(...); // ❌ inference-only
```

Always quantize **after** training, not before or during.

---

## 4. Magnitude-Based Pruning

**Pruning** removes unimportant weights by setting them to zero. Weights close to zero contribute little to the network's output — they can be eliminated without significantly changing predictions.

### Why Pruning Works

Consider a weight `w = 0.0001`. When multiplied by a typical activation value (e.g., 0.5), its contribution to the output is only `0.00005`. Across thousands of samples, such weights have negligible aggregate effect on the loss. Setting them to zero removes their contribution entirely with minimal accuracy cost.

### Magnitude-Based Threshold Selection

The simplest pruning strategy: remove all weights with `|w| < threshold`.

```
for each weight w:
    if |w| < threshold:
        w ← 0.0
```

The weight distribution of a trained network follows a bell curve centred near zero, with most weights small and a few very large:

```
Frequency
    │          ████
    │        ██████████
    │      ████████████████
    │  ████████████████████████
    └──────────────────────────── |w|
       0    threshold         max
              ↑
         Weights left of this line are pruned
```

### Sparsity

**Sparsity** is the fraction of zero weights:

```
sparsity = (number of zero weights) / (total number of weights)
```

A network with 50% sparsity has half its weights set to zero. Sparse weights can be compressed using sparse formats (e.g., CSR/CSC matrices) or skipped during matrix multiplication on hardware that supports sparse operations.

### The PruningReport

After pruning, you receive a summary:

```rust
pub struct PruningReport {
    pub total_params: usize,   // Total weight count examined
    pub pruned_params: usize,  // Weights set to zero
    pub sparsity: f32,         // pruned_params / total_params
    pub threshold: f32,        // Threshold applied
}
```

Example output:
```
PruningReport { total_params: 401408, pruned_params: 200704, sparsity: 0.5000, threshold: 0.020000 }
```

### Choosing a Threshold

**Strategy 1: Manual threshold** — Pick a small value like `0.01` or `0.05`. Inspect the PruningReport to see how many weights were removed. Evaluate accuracy on the validation set. Adjust and repeat.

**Strategy 2: Target sparsity** — Use `suggest_threshold()` to compute the threshold that achieves a desired sparsity level:

```rust
use rust_neural_networks::compression::pruning::suggest_threshold;

let weights = layer.weights();
// Find threshold that would prune 50% of weights
let threshold = suggest_threshold(weights, 0.5);
```

Internally, `suggest_threshold` sorts the absolute values and returns the value at the `target_sparsity` percentile:

```
sorted |w|: [0.001, 0.003, 0.005, 0.010, 0.020, 0.050, 0.100, ...]
                                    ↑
             At 50th percentile → threshold ≈ 0.010
```

### Accuracy Impact

Pruning is **lossy** — sparsity always introduces some accuracy drop. The key insight is that accuracy degrades slowly at first, then sharply near high sparsity:

| Sparsity | Typical Accuracy Drop |
|----------|-----------------------|
| 10%      | ~0.0% (negligible)    |
| 30%      | ~0.1%                 |
| 50%      | ~0.5%                 |
| 70%      | ~1–2%                 |
| 90%      | ~5–15%                |

(Actual numbers depend on the network and task. Larger networks tolerate higher sparsity.)

---

## 5. Pruning a DenseLayer in Rust

### Imports

```rust
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use rust_neural_networks::compression::pruning::{
    prune_dense_layer,
    prune_by_magnitude,
    compute_sparsity,
    suggest_threshold,
    PruningReport,
};
```

### Option A: Prune Raw Weights In-Place

Use `prune_by_magnitude` when you want to prune a weight slice directly:

```rust
let mut weights = vec![0.5_f32, -0.01, 0.3, 0.005, -0.8, 0.02];
let report = prune_by_magnitude(&mut weights, 0.05);

println!("{}", report);
// PruningReport { total_params: 6, pruned_params: 3, sparsity: 0.5000, threshold: 0.050000 }

// Verify: -0.01, 0.005, 0.02 are now 0.0
assert_eq!(weights[1], 0.0);
assert_eq!(weights[3], 0.0);
assert_eq!(weights[5], 0.0);
```

### Option B: Prune a DenseLayer (Recommended)

Use `prune_dense_layer` to create a new pruned layer. The original layer is not modified:

```rust
let mut rng = SimpleRng::new(42);
let trained_layer = DenseLayer::new(784, 512, &mut rng);
// ... training ...

// Prune 50% of the smallest weights
let threshold = suggest_threshold(trained_layer.weights(), 0.5);
let (pruned_layer, report) = prune_dense_layer(&trained_layer, threshold);

println!("Threshold used: {:.6}", report.threshold);
println!("Weights pruned: {}/{}", report.pruned_params, report.total_params);
println!("Sparsity: {:.1}%", report.sparsity * 100.0);
```

Note: Biases are **not pruned** — only the weight matrix is examined. Biases are few in number and often important for the output, especially when many weights are zeroed.

### Option C: Prune Then Quantize

Pruning and quantization compose well — apply both for maximum compression:

```rust
use rust_neural_networks::compression::pruning::prune_dense_layer;
use rust_neural_networks::compression::quantization::quantize_dense_layer;

let mut rng = SimpleRng::new(42);
let trained = DenseLayer::new(784, 512, &mut rng);

// Step 1: Prune 30% of small weights
let (pruned, prune_report) = prune_dense_layer(&trained, 0.02);
println!("After pruning: {}", prune_report);

// Step 2: Quantize the pruned layer to INT8
let quantized = quantize_dense_layer(&pruned);

println!("Original memory:   {} bytes", trained.parameter_memory_bytes());
println!("Pruned+INT8 memory: {} bytes", quantized.parameter_memory_bytes());
// INT8 achieves 4× regardless; pruned zeros compress further with sparse storage
```

### Measuring Sparsity After Pruning

```rust
use rust_neural_networks::compression::pruning::compute_sparsity;

let sparsity = compute_sparsity(pruned_layer.weights());
println!("Weight sparsity: {:.1}%", sparsity * 100.0);
// Output: Weight sparsity: 50.0%
```

### Complete Pruning Workflow

```rust
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use rust_neural_networks::compression::pruning::{
    prune_dense_layer, suggest_threshold, compute_sparsity,
};

fn main() {
    // 1. Create (or load) a trained layer
    let mut rng = SimpleRng::new(42);
    let trained = DenseLayer::new(4, 2, &mut rng);

    println!("=== Before Pruning ===");
    println!("Weights: {:?}", trained.weights());
    println!("Sparsity: {:.1}%", compute_sparsity(trained.weights()) * 100.0);

    // 2. Choose threshold for ~50% sparsity
    let threshold = suggest_threshold(trained.weights(), 0.5);
    println!("\nTarget sparsity: 50%, threshold: {:.6}", threshold);

    // 3. Prune
    let (pruned, report) = prune_dense_layer(&trained, threshold);
    println!("\n=== After Pruning ===");
    println!("{}", report);
    println!("Weights: {:?}", pruned.weights());
    println!("Sparsity: {:.1}%", compute_sparsity(pruned.weights()) * 100.0);

    // 4. Validate: forward pass still produces finite values
    let input = vec![1.0_f32, 0.5, -0.5, 0.25];
    let mut output = vec![0.0_f32; 2];
    pruned.forward(&input, &mut output, 1);
    println!("\nOutput: {:?}", output);
    assert!(output.iter().all(|x| x.is_finite()), "Pruned layer output must be finite");
}
```

---

## 6. Accuracy vs Compression Tradeoff

This table summarises the expected tradeoffs for a 784→512→10 MNIST MLP (a representative model with ~401K parameters in the first layer and ~5K in the second).

### Memory Comparison

| Technique | Weight Storage | Memory (784×512 layer) | Compression |
|-----------|---------------|------------------------|-------------|
| Full precision (f32) | 4 bytes/weight | 1,605 KB | 1.0× |
| INT8 quantization | 1 byte/weight | 401 KB | **4.0×** |
| 50% pruning (f32) | 4 bytes/weight¹ | 803 KB | 2.0×² |
| 50% pruning + INT8 | 1 byte/weight¹ | 201 KB | **8.0×²** |
| 90% pruning + INT8 | 1 byte/weight¹ | 40 KB | **40.0×²** |

¹ *Weight values are zero but still occupy storage in dense format. Compression ratios marked with ² assume subsequent sparse storage (e.g., CSR format) that actually omits zero values.*

### Accuracy Tradeoff (MNIST, representative results)

| Technique | Accuracy | Accuracy Drop |
|-----------|----------|---------------|
| Full precision (baseline) | 97.8% | — |
| INT8 quantization only | 97.6% | −0.2% |
| 30% pruning only | 97.7% | −0.1% |
| 50% pruning only | 97.3% | −0.5% |
| 70% pruning only | 96.8% | −1.0% |
| 50% pruning + INT8 | 97.1% | −0.7% |
| 90% pruning + INT8 | 94.5% | −3.3% |

*Note: Actual numbers depend on the specific model, training quality, and dataset. These figures are representative ballpark values for educational purposes.*

### Inference Speed

INT8 quantization does not directly accelerate inference in this library, because `forward()` dequantizes weights to f32 before computing. True INT8 speedups require hardware support for INT8 arithmetic (e.g., x86 VNNI, ARM dotprod), which is beyond the scope of this educational implementation.

Pruning similarly does not speed up inference in a naive dense matrix multiply — the zero weights are still multiplied (just multiplied by zero). Speedup from pruning requires sparse matrix multiplication routines.

### When Do the Savings Actually Matter?

| Benefit | Quantization | Pruning |
|---------|--------------|---------|
| Memory reduction (dense storage) | ✅ 4× guaranteed | ❌ None¹ |
| Memory reduction (sparse storage) | ✅ 4× guaranteed | ✅ Proportional to sparsity |
| Inference speedup | ❌ Not in this library² | ❌ Not in this library² |
| Model file size | ✅ 4× | ✅ With compression |
| Supports fine-tuning after | ❌ | ✅ |
| Accuracy sensitivity | Low | Moderate |

¹ *Pruned weights are still f32 zeros in dense storage — they take up 4 bytes each.*
² *Full speedup requires hardware INT8 kernels or sparse BLAS, not implemented here.*

---

## 7. When to Use Quantization vs Pruning

### Use Quantization When:

- **Memory is your primary constraint.** Quantization gives a guaranteed 4× weight memory reduction (INT8 vs f32) with minimal accuracy loss, no tuning required.

- **Deploying to edge devices.** Microcontrollers and embedded systems often have kilobytes, not megabytes, of RAM. INT8 models fit where f32 models don't.

- **Hardware supports INT8 inference.** Modern CPUs (x86 AVX-512 VNNI), mobile chips (ARM dotprod), and most ML accelerators run INT8 arithmetic 2–4× faster than f32. Quantization unlocks this speedup.

- **Post-training, no retraining budget.** Quantization works on an already-trained model. Pruning followed by fine-tuning generally yields better accuracy, but requires more compute.

- **Model is already well-converged.** Post-training quantization works best on high-quality models. Undertrained models may lose more accuracy to quantization error.

### Use Pruning When:

- **Accuracy at high compression is the priority.** Pruning combined with fine-tuning (pruning then continuing to train the sparse network) yields much better accuracy than quantization at the same compression ratio.

- **The network is over-parameterised.** Large networks often have significant redundancy. Pruning removes redundant weights, effectively distilling the model.

- **Interpretability matters.** Sparse networks have fewer active connections, making them easier to visualise and analyse.

- **Storage compression is needed (beyond memory).** A 90%-sparse network, when stored in CSR format or compressed with ZIP, can achieve far better than 4× file size reduction.

- **You have compute for fine-tuning.** The best pruning results come from iterative magnitude pruning: prune → fine-tune → prune → fine-tune. This is more expensive than quantization but recovers more accuracy.

### Combine Both for Maximum Compression:

Pruning + quantization is the standard approach for extreme deployment:

1. **Train** the model to full accuracy in f32.
2. **Prune** 50–70% of the smallest weights.
3. **Fine-tune** the sparse model for a few epochs to recover accuracy.
4. **Quantize** the fine-tuned sparse model to INT8.

This pipeline can achieve 10–20× compression with <1% accuracy drop on image classifiers.

### Decision Guide

```
                 ┌─────────────────────────────────────┐
                 │  Goal: Reduce model size/cost        │
                 └──────────────┬──────────────────────┘
                                │
               ┌────────────────┴──────────────────┐
               │ Do you have compute for fine-tuning?│
               └────────┬──────────────┬────────────┘
                        │ No           │ Yes
                        ▼             ▼
               ┌───────────────┐  ┌──────────────────────┐
               │  Quantization │  │ Prune + fine-tune     │
               │  (4× memory,  │  │ (10–20× compression,  │
               │   simple,     │  │  better accuracy,     │
               │   no tuning)  │  │  more work)           │
               └───────────────┘  └──────────────────────┘
```

---

## 8. Exercises

### Beginner

1. **Measure compression ratio.** Create a `DenseLayer` with 100 inputs and 50 outputs. Quantize it. Compute `parameter_memory_bytes()` for both. Verify the ratio is approximately 4×. What happens to the ratio as the layer grows larger? Why?

2. **Explore the weight distribution.** Use `quantize_weights()` on a slice of Xavier-initialized weights. Plot (or print) a histogram of the absolute values. How many weights fall below 0.01? Below 0.1?

3. **Verify symmetric zero-point.** Check that `params.zero_point == 0` for any weight distribution you create. Explain why this is always true for symmetric quantization.

### Intermediate

4. **Threshold sensitivity analysis.** Create a `DenseLayer` (4 inputs, 2 outputs, seed=42). Apply `prune_by_magnitude` with thresholds `[0.01, 0.05, 0.10, 0.20, 0.50]`. Record the sparsity after each. Run a forward pass on input `[1.0, 0.5, -0.5, 0.25]` and compare outputs. At which threshold do outputs change by more than 10%?

5. **Suggest vs manual threshold.** For a 784→512 layer (initialise with `SimpleRng::new(0)`), compute `suggest_threshold(weights, 0.5)`. Then manually try thresholds `[0.01, 0.02, 0.03, 0.04, 0.05]` and report sparsity for each. Which manual threshold is closest to the suggested one?

6. **Sequential pruning + quantization.** Prune a layer at 30% sparsity. Then quantize the pruned layer. Compute the quantization MSE (`quantization_error`) for both the original layer and the pruned one. Explain why they might differ.

### Advanced

7. **Iterative pruning.** Implement a loop that alternately prunes and "fine-tunes" (for this exercise, simulate fine-tuning by adding small random noise to non-zero weights). Observe how iterative pruning can achieve higher sparsity than a single-pass prune with comparable output values.

8. **Custom activation distribution.** Quantization error depends on the weight distribution. Compare the MSE for:
   - Weights drawn from `Uniform(−1, 1)` (worst case)
   - Weights drawn from `Uniform(−0.1, 0.1)` (Xavier-like)
   - Weights with most values near zero but a few large outliers
   Which distribution is hardest to quantize? Why?

9. **Sparse storage format.** Implement a simple Coordinate (COO) sparse format for a pruned weight matrix: store only the `(row, col, value)` triples of non-zero weights. How many bytes does this take at 50% sparsity vs dense f32 storage? At what sparsity level does COO become more memory-efficient than dense INT8?

---

## Summary

| Concept | Formula | Range |
|---------|---------|-------|
| Scale | `max(|w|) / 127` | `[0, max_f32 / 127]` |
| Quantize | `clamp(round(w / scale), −127, 127)` | `[−127, 127]` as i8 |
| Dequantize | `q × scale` | Approximately original range |
| MSE | `(1/N) Σ (wᵢ − qᵢ·scale)²` | `[0, scale² / 3]` roughly |
| Sparsity | `zero_count / total_count` | `[0.0, 1.0]` |
| Threshold | `sorted_abs_values[target_sparsity × N]` | `[0, max_abs]` |

**Key takeaways:**

- INT8 quantization achieves **exactly 4× weight memory reduction** with typically **<1% accuracy drop** on well-trained models.
- Magnitude-based pruning achieves variable compression depending on sparsity, with **accuracy degrading gradually** up to ~50–70% sparsity.
- Both techniques are **complementary**: apply pruning first, then quantize for maximum compression.
- In this educational library, neither technique accelerates inference — full speedup requires hardware INT8 kernels or sparse BLAS.

---

**Navigation:**
← [Tutorial Index](README.md)

**Related documentation:**
- [Hyperparameters Guide](../hyperparameters.md) - Learning rate, batch size, optimizer selection
- [Activation Functions](../activation_functions.md) - ReLU, sigmoid, and alternatives
- [Dense Layer Backprop](../backpropagation/dense_layer.md) - Gradient computation for DenseLayer
- [Mathematical Documentation Guide](../MATHEMATICAL_DOCUMENTATION_GUIDE.md) - Notation conventions
