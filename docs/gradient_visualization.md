# Gradient Visualization Tool

This document explains how to use the interactive gradient visualization tool to understand gradient flow through neural networks during training. The tool helps learners diagnose vanishing and exploding gradient problems and understand training dynamics.

## Table of Contents

- [Overview](#overview)
- [Understanding Gradient Flow](#understanding-gradient-flow)
  - [What Are Gradients?](#what-are-gradients)
  - [Gradient Flow in Neural Networks](#gradient-flow-in-neural-networks)
  - [Why Gradient Magnitude Matters](#why-gradient-magnitude-matters)
- [Gradient Problems](#gradient-problems)
  - [Vanishing Gradients](#vanishing-gradients)
  - [Exploding Gradients](#exploding-gradients)
- [Using the Visualization Tool](#using-the-visualization-tool)
  - [Installation Requirements](#installation-requirements)
  - [Basic Usage](#basic-usage)
  - [Command-Line Options](#command-line-options)
  - [Output Files](#output-files)
- [Interpreting Visualizations](#interpreting-visualizations)
  - [Static Plots](#static-plots)
  - [Combined Layer Comparison](#combined-layer-comparison)
  - [Animated Gradient Flow](#animated-gradient-flow)
  - [Gradient Health Checks](#gradient-health-checks)
- [Gradient Data Format](#gradient-data-format)
- [Troubleshooting Gradient Issues](#troubleshooting-gradient-issues)
- [Gradient Clipping](#gradient-clipping)
- [Best Practices](#best-practices)
- [Related Tools and References](#related-tools-and-references)

## Overview

The gradient visualization tool analyzes gradient magnitudes flowing through network layers during training. It helps answer critical questions:

- Are my gradients vanishing (becoming too small)?
- Are my gradients exploding (becoming too large)?
- Which layers have the healthiest gradient flow?
- How do gradients evolve during training?
- Why is my network not learning effectively?

**Key features:**
- **Per-layer gradient tracking**: Monitor weight and bias gradients separately for each layer
- **Automatic problem detection**: Warns about vanishing/exploding gradients with specific thresholds
- **Static and animated visualizations**: See gradient evolution over training epochs
- **Log-scale plots**: Handle wide ranges of gradient magnitudes effectively
- **Statistical summaries**: Min, max, mean, and standard deviation per layer

## Understanding Gradient Flow

### What Are Gradients?

Gradients are the partial derivatives of the loss function with respect to model parameters (weights and biases). They tell us:

- **Direction**: Which way to adjust parameters to reduce loss
- **Magnitude**: How strongly parameters should be adjusted

During backpropagation, gradients flow backward through the network from the output layer to the input layer, updating each layer's parameters along the way.

**Mathematical notation:**
```
∂Loss/∂W = gradient with respect to weights
∂Loss/∂b = gradient with respect to biases
```

### Gradient Flow in Neural Networks

Consider a simple network: Input → Layer 1 → Layer 2 → Output

**Forward pass:** Data flows forward through layers
```
x → Layer1(W1, b1) → h1 → Layer2(W2, b2) → output → Loss
```

**Backward pass:** Gradients flow backward through layers
```
∂Loss/∂output → ∂Loss/∂W2, ∂Loss/∂b2 → ∂Loss/∂h1 → ∂Loss/∂W1, ∂Loss/∂b1
```

Each layer computes:
1. Gradient with respect to its own parameters (for parameter updates)
2. Gradient with respect to its inputs (to propagate to previous layers)

### Why Gradient Magnitude Matters

**Healthy gradients** (magnitude ~0.001 to ~1.0):
- Parameters update at a reasonable rate
- Network learns effectively
- Training converges smoothly

**Too small (vanishing)** (magnitude < 1e-5):
- Parameters barely update
- Learning slows or stops
- Early layers may not learn at all

**Too large (exploding)** (magnitude > 100):
- Parameters update too aggressively
- Training becomes unstable
- May cause NaN or Inf values

## Gradient Problems

### Vanishing Gradients

**Problem:** Gradients become extremely small as they propagate backward through layers, especially in deep networks.

**Symptoms:**
- Training loss stops decreasing
- Early layers show very small weight updates
- Gradients consistently below 1e-5
- Network learns slowly or not at all

**Common causes:**
1. **Activation functions with small derivatives**
   - Sigmoid: gradient ≤ 0.25, saturates for |x| > 4
   - Tanh: gradient ≤ 1.0, saturates for |x| > 2
   - Solution: Use ReLU, Leaky ReLU, or GELU

2. **Deep networks**
   - Gradients multiply through many layers
   - Each layer's derivative < 1.0 compounds the problem
   - Solution: Use residual connections (ResNet), batch normalization

3. **Poor weight initialization**
   - Weights too small cause small activations and gradients
   - Solution: Use He initialization for ReLU, Xavier for tanh/sigmoid

**Example from visualization:**
```
⚠️  WARNING: Vanishing weight gradients detected in 'hidden1'
   Occurrences: 5/10 epochs
   Affected epochs: [6, 7, 8, 9, 10]
   Minimum gradient: 3.45e-07
```

**Solutions:**
- Switch activation: Sigmoid/Tanh → ReLU or Leaky ReLU
- Add batch normalization between layers
- Use residual connections (skip connections)
- Reduce network depth
- Check weight initialization (use He or Xavier initialization)
- Increase learning rate cautiously

### Exploding Gradients

**Problem:** Gradients become extremely large during backpropagation, causing unstable training.

**Symptoms:**
- Loss increases instead of decreasing
- Loss becomes NaN or Inf
- Parameters become NaN
- Gradients consistently above 100
- Training diverges

**Common causes:**
1. **Learning rate too high**
   - Large gradients × large learning rate = huge parameter updates
   - Solution: Reduce learning rate (try 10x smaller)

2. **Poor weight initialization**
   - Weights too large cause large activations and gradients
   - Solution: Use proper initialization (He, Xavier)

3. **Deep networks with unbounded activations**
   - ReLU can produce very large activations
   - Solution: Add batch normalization, use gradient clipping

4. **Recurrent networks (RNN/LSTM)**
   - Gradients multiply over many time steps
   - Solution: Use gradient clipping (see Gradient Clipping section)

**Example from visualization:**
```
⚠️  WARNING: Exploding weight gradients detected in 'output'
   Occurrences: 3/10 epochs
   Affected epochs: [1, 2, 3]
   Maximum gradient: 234.56
```

**Solutions:**
- **Gradient clipping** (recommended): Limit gradient norm to safe values (e.g., max_norm = 5.0)
- Reduce learning rate (try 0.1x or 0.01x)
- Check weight initialization
- Add batch normalization
- Use gradient checkpointing for very deep networks
- For RNNs: Use LSTM/GRU instead of vanilla RNN

## Using the Visualization Tool

### Installation Requirements

The gradient visualization tool requires Python 3 with the following packages:

```bash
# Install required packages
pip install matplotlib numpy pillow

# Or install all Python dependencies for the project
pip install -r requirements.txt
```

**Required packages:**
- `matplotlib`: Plotting and visualization
- `numpy`: Numerical computations and data processing
- `pillow`: For saving animated GIFs

### Basic Usage

**Step 1: Generate gradient logs during training**

The Rust training code automatically logs gradient data to CSV files in the `logs/` directory:
- `logs/gradients_mlp.csv` - MLP model gradients
- `logs/gradients_cnn.csv` - CNN model gradients

**Step 2: Run the visualization tool**

```bash
# Visualize MLP gradients (default)
python visualize_gradients.py

# Visualize CNN gradients
python visualize_gradients.py --model cnn

# Generate animated gradient flow
python visualize_gradients.py --animate

# Customize animation speed (frames per second)
python visualize_gradients.py --animate --fps 4

# Use rolling window animation (show last N epochs)
python visualize_gradients.py --animate --window 10
```

### Command-Line Options

```
Usage: python visualize_gradients.py [options]

Options:
  --model, -m MODEL      Model type to visualize (mlp, cnn)
                         Default: mlp

  --animate, -a          Generate animated GIF showing gradient evolution
                         Creates gradient_flow_animated.gif

  --fps, -f FPS          Frames per second for animation
                         Default: 2
                         Typical range: 1-10

  --window, -w SIZE      Rolling window size for animation
                         Shows last SIZE epochs at each frame
                         Default: progressive (shows all epochs up to current)

  --help, -h             Show help message and exit
```

### Output Files

The tool generates several output files:

1. **`gradient_flow.png`** - Static per-layer gradient plots
   - Separate subplots for each layer
   - Weight gradients (left column)
   - Bias gradients (right column)
   - Log scale for handling wide ranges

2. **`gradient_flow_combined.png`** - Combined comparison plot
   - All layers on the same axes
   - Easy comparison of gradient magnitudes across layers
   - Identifies which layers have gradient problems

3. **`gradient_flow_animated.gif`** - Animated gradient evolution
   - Shows how gradients change over training epochs
   - Progressive reveal of training dynamics
   - Useful for presentations and understanding convergence

**Example usage:**

```bash
# Generate all visualizations for MLP model
python visualize_gradients.py --model mlp

# Output:
# ✓ Gradient flow visualization saved to: gradient_flow.png
# ✓ Combined gradient comparison saved to: gradient_flow_combined.png

# Generate animation
python visualize_gradients.py --model mlp --animate --fps 2

# Output:
# ✓ Animated gradient flow saved to: gradient_flow_animated.gif
#   Total frames: 10
#   Duration: 5.0 seconds
#   File size: 125.3 KB
```

## Interpreting Visualizations

### Static Plots

**Layout:**
- Each row represents one layer (e.g., `hidden1`, `output`)
- Left column: Weight gradients
- Right column: Bias gradients
- Y-axis: Gradient magnitude (log scale)
- X-axis: Training epoch

**What to look for:**

**Healthy gradient flow:**
```
Gradients in range [1e-3, 1.0]
Gradients decrease smoothly as training progresses
All layers show similar gradient magnitudes
Mean line (red dashed) is stable
```

**Vanishing gradients:**
```
Gradients drop below 1e-5
Gradients approach zero over epochs
Earlier layers have smaller gradients than later layers
Flat lines near bottom of log scale
```

**Exploding gradients:**
```
Gradients spike above 100
Sharp upward spikes in gradient magnitude
Erratic, unstable gradient values
Lines jump to top of log scale
```

### Combined Layer Comparison

The combined plot shows all layers on the same axes, making it easy to compare:

**Example interpretation:**

```
If output layer gradients >> hidden layer gradients:
  → Vanishing gradient problem in early layers
  → Early layers not learning effectively

If all layer gradients are similar:
  → Healthy gradient flow
  → All layers learning at similar rates

If any layer shows spikes:
  → Exploding gradient in that specific layer
  → May need layer-specific gradient clipping
```

### Animated Gradient Flow

The animated GIF shows gradient evolution frame-by-frame:

**Progressive mode** (`--window` not specified):
- Each frame shows epochs 0 to N
- See how gradients accumulate over training
- Good for understanding long-term trends

**Rolling window mode** (`--window 10`):
- Each frame shows last 10 epochs
- Focus on recent gradient behavior
- Good for spotting instabilities

**What to watch for:**
- Smooth curves indicate stable training
- Sudden jumps indicate instability or learning rate issues
- Gradients should generally decrease as training progresses
- All layers should show similar patterns

### Gradient Health Checks

The tool automatically detects gradient problems and prints warnings:

```
==============================================================
GRADIENT HEALTH CHECK
==============================================================
Vanishing threshold: < 1.00e-05
Exploding threshold: > 1.00e+02
--------------------------------------------------------------

✓ 'hidden1': No gradient issues detected

⚠️  WARNING: Vanishing weight gradients detected in 'output'
   Occurrences: 2/10 epochs
   Affected epochs: [9, 10]
   Minimum gradient: 8.34e-06

⚠️  Gradient issues detected - consider:
   • Using gradient clipping to prevent exploding gradients
   • Adjusting learning rate or network architecture for vanishing gradients
   • Reviewing activation functions (ReLU vs sigmoid/tanh)
   • Adding batch normalization or residual connections
==============================================================
```

**Thresholds:**
- **Vanishing threshold:** 1e-5 (gradients below this are considered too small)
- **Exploding threshold:** 100 (gradients above this are considered too large)

You can modify these thresholds by editing `visualize_gradients.py` if your use case requires different values.

## Gradient Data Format

The gradient logs use CSV format with the following structure:

```csv
epoch,layer_name,grad_norm_weights,grad_norm_biases
1,hidden1,0.0523,0.0412
1,output,0.0891,0.0734
2,hidden1,0.0451,0.0389
2,output,0.0767,0.0623
```

**Fields:**
- `epoch`: Training epoch number (1-indexed)
- `layer_name`: Name of the layer (e.g., "hidden1", "output")
- `grad_norm_weights`: L2 norm of weight gradients for this layer
- `grad_norm_biases`: L2 norm of bias gradients for this layer

**L2 norm calculation:**
```
grad_norm = sqrt(sum(grad_i^2))
```

This measures the overall magnitude of all gradients for a layer's parameters, providing a single number that summarizes gradient strength.

**Why use norms?**
- Reduces thousands of individual gradient values to one interpretable number
- Makes visualization tractable (can't plot every weight's gradient)
- Captures overall gradient magnitude for the layer
- Standard practice in gradient monitoring

## Troubleshooting Gradient Issues

### Issue: All gradients are zero

**Possible causes:**
- Dead ReLU neurons (all activations are negative)
- Learning rate is exactly zero
- Bug in backward pass implementation

**Solutions:**
1. Check activation function (try Leaky ReLU instead of ReLU)
2. Verify learning rate is positive and reasonable (e.g., 0.01)
3. Check weight initialization (weights should not all be zero)
4. Verify backward pass is implemented correctly (see `tests/test_gradient_checking.rs`)

### Issue: Gradients start healthy but vanish over time

**Possible causes:**
- Learning rate too high, causing weights to saturate
- Activation functions saturating (sigmoid, tanh)
- Network becoming overconfident (softmax outputs near 0 or 1)

**Solutions:**
1. Reduce learning rate (try 0.1x)
2. Use learning rate scheduler (step decay, exponential decay)
3. Switch to non-saturating activations (ReLU, Leaky ReLU)
4. Add batch normalization

### Issue: Gradients explode in first few epochs

**Possible causes:**
- Learning rate too high
- Poor weight initialization (weights too large)
- Data not normalized

**Solutions:**
1. Reduce learning rate significantly (try 0.01x)
2. Use proper weight initialization:
   - He initialization for ReLU: `std = sqrt(2.0 / fan_in)`
   - Xavier initialization for sigmoid/tanh: `std = sqrt(1.0 / fan_in)`
3. Normalize input data (mean=0, std=1)
4. Use gradient clipping from the start

### Issue: Gradients are unstable (oscillating)

**Possible causes:**
- Batch size too small
- Learning rate not optimal
- Data has outliers or is not shuffled

**Solutions:**
1. Increase batch size (try 2x or 4x)
2. Use learning rate scheduler for gradual decay
3. Check data preprocessing (shuffle, normalize, remove outliers)
4. Use momentum or Adam optimizer instead of vanilla SGD

### Issue: Earlier layers have much smaller gradients than later layers

**Symptom of vanishing gradients in deep networks**

**Solutions:**
1. Add residual connections (skip connections)
2. Use batch normalization after each layer
3. Reduce network depth
4. Use gradients checkpointing
5. Try layer normalization or group normalization

## Gradient Clipping

Gradient clipping is a crucial technique for preventing exploding gradients, especially in recurrent neural networks (RNNs, LSTMs) and very deep networks.

### What is Gradient Clipping?

Gradient clipping limits the magnitude of gradients during training to prevent them from becoming too large.

**Two main approaches:**

1. **Clipping by norm** (recommended):
   - Scales the entire gradient vector if its L2 norm exceeds a threshold
   - Preserves gradient direction while limiting magnitude

2. **Clipping by value**:
   - Clamps individual gradient elements to a specified range
   - Simpler but can change gradient direction

### Using Gradient Clipping in Rust

The project includes gradient clipping utilities in `src/utils/gradient_clipping.rs`:

**Clip by norm (recommended):**
```rust
use rust_neural_networks::utils::gradient_clipping::clip_gradient_norm;

// Compute gradients (from backward pass)
let mut gradients = vec![3.0, 4.0, 5.0];

// Clip gradients to max norm of 5.0
let original_norm = clip_gradient_norm(&mut gradients, 5.0);

println!("Original norm: {}", original_norm);  // 7.07
println!("Clipped gradients: {:?}", gradients);  // Scaled to norm=5.0
```

**Clip by value:**
```rust
use rust_neural_networks::utils::gradient_clipping::clip_gradient_value;

// Clip each gradient element to range [-4.0, 4.0]
let mut gradients = vec![-5.0, 3.0, 10.0, -2.0];
clip_gradient_value(&mut gradients, 4.0);

println!("Clipped gradients: {:?}", gradients);  // [-4.0, 3.0, 4.0, -2.0]
```

### When to Use Gradient Clipping

**Always use for:**
- Recurrent neural networks (RNN, LSTM, GRU)
- Very deep networks (>50 layers)
- Networks with unbounded activations (ReLU)

**Consider using for:**
- Any network showing exploding gradients in visualization
- Training that becomes unstable after initial epochs
- Networks trained on data with outliers

**Typical max_norm values:**
- RNNs/LSTMs: 5.0 to 10.0
- Deep feedforward networks: 10.0 to 50.0
- CNNs: Usually not needed, but 100.0 if required

### Gradient Clipping vs Other Solutions

| Solution | Advantages | Disadvantages | When to Use |
|----------|-----------|---------------|-------------|
| **Gradient Clipping** | • Simple to implement<br>• Works immediately<br>• No architecture changes | • Doesn't fix root cause<br>• Requires tuning threshold | First-line defense for exploding gradients |
| **Lower Learning Rate** | • Addresses root cause<br>• Improves stability | • Slower convergence<br>• May need scheduler | When gradients explode at start |
| **Batch Normalization** | • Prevents internal covariate shift<br>• Often eliminates need for clipping | • Adds complexity<br>• Slower inference | Deep networks, production models |
| **Better Initialization** | • Prevents problem at source<br>• No runtime overhead | • May not be sufficient alone | Always use (He/Xavier) |
| **Residual Connections** | • Enables very deep networks<br>• Improves gradient flow | • Architecture change<br>• More complex | Very deep networks (>20 layers) |

**Best practice:** Combine multiple approaches:
1. Use proper initialization (He/Xavier)
2. Add batch normalization
3. Use gradient clipping as safety net
4. Monitor gradients with visualization tool

## Best Practices

### During Model Development

1. **Visualize gradients early and often**
   - Run visualization after first few epochs
   - Check for gradient problems before full training
   - Saves time by catching issues early

2. **Monitor gradient trends**
   - Gradients should generally decrease as training progresses
   - Sudden changes indicate instability or learning rate issues
   - Use animated visualization to spot patterns

3. **Compare layers**
   - Use combined plot to compare gradient magnitudes
   - Earlier layers should have similar gradients to later layers
   - Large differences indicate vanishing/exploding gradient problems

### Activation Function Selection

Based on gradient behavior:

**Use ReLU or Leaky ReLU if:**
- Seeing vanishing gradients with sigmoid/tanh
- Need fast training convergence
- Building deep networks (>5 layers)

**Use Sigmoid only if:**
- Output layer for binary classification
- Explicitly need bounded outputs (0, 1)
- Network is shallow (1-2 layers)

**Use Tanh only if:**
- Need zero-centered outputs
- RNNs or specific architectural requirements
- Network is shallow

**Use GELU or Swish if:**
- Building transformer models
- State-of-the-art performance is critical
- Computational cost is acceptable

### Learning Rate Tuning

Use gradient visualization to guide learning rate selection:

**If gradients are exploding:**
- Reduce learning rate by 10x or 100x
- Use learning rate scheduler with decay
- Add gradient clipping

**If gradients are vanishing:**
- Increase learning rate cautiously (2x or 5x)
- Check activation functions first
- Verify weight initialization

**If gradients are healthy but training is slow:**
- Increase learning rate slightly
- Use learning rate warmup
- Consider using Adam optimizer instead of SGD

### Network Architecture

Design networks with gradient flow in mind:

**For deep networks (>10 layers):**
- Add residual connections (skip connections)
- Use batch normalization after each layer
- Consider using GELU or Swish activations
- Monitor gradient visualization closely

**For recurrent networks:**
- Always use gradient clipping (max_norm=5.0)
- Prefer LSTM/GRU over vanilla RNN
- Monitor gradients over time steps
- Consider using layer normalization

**For convolutional networks:**
- Batch normalization is essential
- Use ReLU or Leaky ReLU
- Check gradients in early conv layers
- Consider residual connections for deep CNNs (>20 layers)

### Production Checklist

Before deploying a model, verify gradient health:

- ✓ Gradients remain in healthy range (1e-4 to 10.0) throughout training
- ✓ All layers show similar gradient magnitudes
- ✓ No vanishing gradient warnings in final epochs
- ✓ No exploding gradient warnings at any point
- ✓ Gradient trends are smooth (no erratic behavior)
- ✓ Animation shows stable convergence
- ✓ Early stopping triggered by validation loss, not gradient issues

## Related Tools and References

### Project Tools

**Gradient checking tests:**
- `tests/test_gradient_checking.rs` - Numerical gradient validation
- Verifies backward pass implementation is correct
- Compares analytical gradients to numerical approximations

**Gradient clipping utilities:**
- `src/utils/gradient_clipping.rs` - Gradient clipping functions
- `clip_gradient_norm()` - Clip by L2 norm (recommended)
- `clip_gradient_value()` - Clip by value range

**Activation function tests:**
- `tests/test_activation_gradients.rs` - Activation gradient validation
- Ensures activation functions compute gradients correctly

### Related Documentation

- [Activation Functions](activation_functions.md) - Detailed guide to activation functions and their gradients
- [Hyperparameters Configuration](hyperparameters.md) - Learning rate, batch size, and scheduler configuration
- [Backpropagation Documentation](backpropagation/) - Mathematical foundations of gradient computation
- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture

### External Resources

**Foundational Papers:**
- [Glorot & Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a.html) - Understanding vanishing/exploding gradients
- [He et al. (2015)](https://arxiv.org/abs/1502.01852) - Deep Residual Learning (ResNet)
- [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) - Batch Normalization
- [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063) - On the difficulty of training RNNs

**Tutorials:**
- [CS231n: Gradient Flow](http://cs231n.github.io/neural-networks-3/#gradcheck) - Stanford course on gradient flow
- [Distill.pub: The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) - Visualization techniques

**Books:**
- [Deep Learning (Goodfellow et al., 2016)](https://www.deeplearningbook.org/) - Chapter 8: Optimization for Training Deep Models
- [Neural Networks and Deep Learning (Nielsen, 2015)](http://neuralnetworksanddeeplearning.com/) - Chapter 5: Why are deep neural networks hard to train?

### Implementation Notes

**Gradient logging in Rust:**

The training scripts log gradients automatically:

```rust
// Example from mnist_mlp.rs
let grad_norm_weights = compute_gradient_norm(&layer.grad_weights);
let grad_norm_biases = compute_gradient_norm(&layer.grad_biases);

writeln!(grad_log, "{},{},{:.4},{:.4}",
    epoch, layer_name, grad_norm_weights, grad_norm_biases)?;
```

**Log location:** `logs/gradients_<model>.csv`

**Logging frequency:** Once per epoch (after backward pass, before parameter update)

---

## Quick Reference

### Common Gradient Ranges

| Gradient Magnitude | Interpretation | Action |
|-------------------|----------------|--------|
| < 1e-7 | Severe vanishing | Fix immediately (change architecture) |
| 1e-7 to 1e-5 | Vanishing | Adjust activation function or architecture |
| 1e-5 to 1e-3 | Slightly low | Monitor, may be acceptable |
| 1e-3 to 1.0 | **Healthy** | Good gradient flow |
| 1.0 to 10.0 | Slightly high | Monitor, usually acceptable |
| 10.0 to 100.0 | High | Consider gradient clipping |
| > 100.0 | Exploding | Fix immediately (reduce LR, add clipping) |

### Quick Command Reference

```bash
# Basic visualization
python visualize_gradients.py

# Animated gradient flow
python visualize_gradients.py --animate

# Fast animation (10 fps)
python visualize_gradients.py --animate --fps 10

# CNN model visualization
python visualize_gradients.py --model cnn --animate

# Rolling window (last 10 epochs)
python visualize_gradients.py --animate --window 10

# Help
python visualize_gradients.py --help
```

### Quick Fixes

**Vanishing gradients?**
```
1. Switch to ReLU/Leaky ReLU activation
2. Check weight initialization (use He initialization)
3. Add batch normalization
4. Reduce network depth or add residual connections
```

**Exploding gradients?**
```
1. Reduce learning rate (10x smaller)
2. Add gradient clipping (max_norm=5.0)
3. Check weight initialization
4. Add batch normalization
```

**Unstable gradients?**
```
1. Increase batch size
2. Use learning rate scheduler
3. Check data preprocessing (normalize, shuffle)
4. Try Adam optimizer instead of SGD
```

---

**For more information:**
- GitHub Issues: Report bugs or request features
- Documentation: See `docs/` directory for related topics
- Tests: See `tests/test_gradient_*.rs` for implementation details
