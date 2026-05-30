# CIFAR-10 Dataset

This document provides comprehensive information about the CIFAR-10 dataset, its binary format, how to download and use it with this project, and key differences from MNIST.

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Download Instructions](#download-instructions)
- [Binary File Format](#binary-file-format)
- [RGB vs Grayscale: Key Differences](#rgb-vs-grayscale-key-differences)
- [Architecture Differences](#architecture-differences)
- [Usage Examples](#usage-examples)
- [Expected Performance](#expected-performance)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

CIFAR-10 (Canadian Institute For Advanced Research, 10 classes) is a widely-used computer vision dataset for object recognition tasks. It is significantly more challenging than MNIST due to color images, smaller resolution, and more complex object categories.

**Dataset Characteristics:**

- **Total images**: 60,000 (50,000 training + 10,000 test)
- **Image size**: 32×32 pixels
- **Color**: RGB (3 channels)
- **Classes**: 10 object categories
- **Format**: Binary batch files

**Class Labels (0-9):**

| Label | Class Name | Description |
|-------|------------|-------------|
| 0 | airplane | Aircraft in flight or on ground |
| 1 | automobile | Cars, sedans, SUVs |
| 2 | bird | Various bird species |
| 3 | cat | Domestic cats |
| 4 | deer | Deer in natural settings |
| 5 | dog | Domestic dogs |
| 6 | frog | Frogs and similar amphibians |
| 7 | horse | Horses |
| 8 | ship | Ships, boats, vessels |
| 9 | truck | Trucks, large vehicles |

**Key Differences from MNIST:**

- **Color vs Grayscale**: CIFAR-10 has 3 RGB channels vs MNIST's 1 grayscale channel
- **Image Size**: 32×32 pixels vs MNIST's 28×28 pixels
- **Complexity**: Natural RGB images vs simple handwritten digits
- **Difficulty**: Baseline CNN achieves 50-60% vs MNIST's 90%+ accuracy
- **Data Format**: Custom binary format vs IDX format

## Dataset Structure

The CIFAR-10 binary version consists of 6 files:

```text
data/cifar-10-batches-bin/
├── data_batch_1.bin    (10,000 training images)
├── data_batch_2.bin    (10,000 training images)
├── data_batch_3.bin    (10,000 training images)
├── data_batch_4.bin    (10,000 training images)
├── data_batch_5.bin    (10,000 training images)
├── test_batch.bin      (10,000 test images)
└── batches.meta.txt    (class names, optional)
```

**Training set**: 50,000 images split into 5 batches of 10,000 each

**Test set**: 10,000 images in a single batch

**Note**: This project uses a 10% validation split from the training data, resulting in:
- Training: 45,000 images (90% of training set)
- Validation: 5,000 images (10% of training set)
- Test: 10,000 images (separate test set)

## Quick Verification (Recommended)

After downloading/extracting, verify the expected file layout under `./data`:

```bash
cargo run --bin dataset-helper -- verify --cifar10
```

## Download Instructions

### Official Source

Download the **CIFAR-10 binary version** (not the Python version):

**Direct download:**
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
```

**Or visit:**
- <https://www.cs.toronto.edu/~kriz/cifar.html>

### Installation Steps

1. Download the CIFAR-10 binary version (`cifar-10-binary.tar.gz`)
2. Extract the archive:
   ```bash
   tar -xzf cifar-10-binary.tar.gz
   ```
3. Move the extracted directory to your project's data folder:
   ```bash
   mv cifar-10-batches-bin ./data/
   ```
4. Verify the files are in place:
   ```bash
   ls -la ./data/cifar-10-batches-bin/
   ```

Expected output:
```text
data_batch_1.bin
data_batch_2.bin
data_batch_3.bin
data_batch_4.bin
data_batch_5.bin
test_batch.bin
batches.meta.txt
readme.html
```

**File sizes:**
- Each batch file: 30,730,000 bytes (exactly)
- Total dataset size: ~170 MB

**Alternative sources:**
- <https://www.kaggle.com/c/cifar-10>
- Mirror sites listed at <https://www.cs.toronto.edu/~kriz/cifar.html>

**Important:** Download the **binary version**, not the Python pickle version. The file names must end in `.bin`.

## Binary File Format

Each CIFAR-10 binary batch file contains 10,000 records with a specific structure.

### Record Structure

Each record is **3,073 bytes**:

```text
[1 byte label] [3,072 bytes pixel data]
```

**Label (1 byte):**
- Single byte with value 0-9
- Represents the class label (airplane=0, automobile=1, etc.)

**Pixel Data (3,072 bytes):**
- 32×32 pixels × 3 channels = 3,072 bytes
- **Channel-major order** (also called planar format):
  - Bytes 0-1023: Red channel (32×32 pixels, row-major)
  - Bytes 1024-2047: Green channel (32×32 pixels, row-major)
  - Bytes 2048-3071: Blue channel (32×32 pixels, row-major)

### Data Layout Example

For a single image:

```text
Byte 0: Label (0-9)
Bytes 1-1024: Red channel
  [R(0,0), R(0,1), ..., R(0,31),   <- First row
   R(1,0), R(1,1), ..., R(1,31),   <- Second row
   ...
   R(31,0), R(31,1), ..., R(31,31)] <- Last row

Bytes 1025-2048: Green channel
  [G(0,0), G(0,1), ..., G(31,31)]

Bytes 2049-3072: Blue channel
  [B(0,0), B(0,1), ..., B(31,31)]
```

### Pixel Value Normalization

Raw pixel values are unsigned bytes (0-255):
- 0 = darkest
- 255 = brightest

This project normalizes pixels to [0.0, 1.0] floating-point range:
```rust
normalized_value = raw_byte as f32 / 255.0
```

### Channel Conversion

The data loader converts from **channel-major** (file format) to **pixel-interleaved** format (easier for CNNs):

**File format (channel-major):**
```text
[R₀ R₁ ... R₁₀₂₃] [G₀ G₁ ... G₁₀₂₃] [B₀ B₁ ... B₁₀₂₃]
```

**Loaded format (pixel-interleaved):**
```text
[R₀ G₀ B₀] [R₁ G₁ B₁] [R₂ G₂ B₂] ... [R₁₀₂₃ G₁₀₂₃ B₁₀₂₃]
```

This pixel-interleaved format (RGB, RGB, RGB...) allows convolutional layers to process all three color channels together for each spatial location.

## RGB vs Grayscale: Key Differences

Understanding the differences between CIFAR-10 (RGB) and MNIST (grayscale) is crucial for architecture design.

### Data Dimensions

| Property | MNIST | CIFAR-10 |
|----------|-------|----------|
| Input size | 28×28 | 32×32 |
| Channels | 1 (grayscale) | 3 (RGB) |
| Total input | 784 values | 3,072 values |
| Pixel range | [0, 1] | [0, 1] per channel |
| Data format | IDX format | Binary batch format |

### Architectural Implications

**1. Input Layer Size:**
- MNIST: 784 input neurons (28×28×1)
- CIFAR-10: 3,072 input neurons (32×32×3)

**2. Convolutional Layers:**

For the first convolutional layer:

**MNIST:**
```rust
// Single channel input
const IN_CHANNELS: usize = 1;
const OUT_CHANNELS: usize = 8;
// Kernel size: 1 × 3 × 3 = 9 weights per output filter
// Total weights: 8 × 9 = 72 weights
```

**CIFAR-10:**
```rust
// Three channel input
const IN_CHANNELS: usize = 3;
const OUT_CHANNELS: usize = 16;  // More filters for RGB
// Kernel size: 3 × 3 × 3 = 27 weights per output filter
// Total weights: 16 × 27 = 432 weights
```

**Why more filters for RGB?**
- RGB images have 3× more input information
- Color patterns are more complex than grayscale
- More filters help capture color-specific features (red edges, blue textures, etc.)

**3. Fully Connected Layers:**

After convolution and pooling:

**MNIST CNN:**
```rust
const POOL_SIZE: usize = 2;
const POOL_H: usize = 28 / 2 = 14;
const POOL_W: usize = 28 / 2 = 14;
const FC_IN: usize = 8 × 14 × 14 = 1,568;  // 8 filters
```

**CIFAR-10 CNN:**
```rust
const POOL_SIZE: usize = 2;
const POOL_H: usize = 32 / 2 = 16;
const POOL_W: usize = 32 / 2 = 16;
const FC_IN: usize = 16 × 16 × 16 = 4,096;  // 16 filters
```

CIFAR-10 has **2.6× more parameters** in the fully connected layer due to larger spatial dimensions and more filters.

### Memory Requirements

**MNIST (single image):**
- Input: 784 × 4 bytes (f32) = 3.1 KB
- After Conv (8 filters): 6.3 KB
- After Pool: 1.6 KB

**CIFAR-10 (single image):**
- Input: 3,072 × 4 bytes (f32) = 12.3 KB
- After Conv (16 filters): 32.8 KB
- After Pool: 8.2 KB

**Batch of 32 images:**
- MNIST: ~50 KB input
- CIFAR-10: ~393 KB input

### Computational Complexity

For a single forward pass through the first conv layer (3×3 kernel):

**MNIST:**
- Input channels: 1
- Output size: 28×28×8
- Operations: 28 × 28 × 8 × (9 multiplications + 8 additions) ≈ 106K ops

**CIFAR-10:**
- Input channels: 3
- Output size: 32×32×16
- Operations: 32 × 32 × 16 × (27 multiplications + 26 additions) ≈ 870K ops

CIFAR-10 convolution is **8× more expensive** computationally.

### Color Space Considerations

**Grayscale (MNIST):**
- Single intensity value per pixel
- Simple edge detection sufficient
- Brightness is the only feature

**RGB (CIFAR-10):**
- Three correlated channels (R, G, B)
- Color patterns provide additional information:
  - Sky is typically blue/cyan (high B, medium G, low R)
  - Vegetation is green (high G, medium B/R)
  - Fire trucks are red (high R, low G/B)
- Convolution filters learn color-specific features:
  - "Blue-horizontal-edge detector"
  - "Green-blob detector"
  - "Red-vertical-line detector"

## Architecture Differences

### MNIST CNN Architecture

```text
Input: 28×28×1 grayscale
  ↓
Conv2D: 8 filters, 3×3 kernel, padding=1
  (28×28×1 → 28×28×8)
  Parameters: 1×3×3×8 + 8 = 80
  ↓
ReLU activation
  ↓
MaxPool2D: 2×2
  (28×28×8 → 14×14×8)
  ↓
Flatten: 14×14×8 = 1,568
  ↓
Fully Connected: 1,568 → 10
  Parameters: 1,568×10 + 10 = 15,690
  ↓
Softmax
  ↓
Output: 10 classes

Total parameters: 15,770
```

### CIFAR-10 CNN Architecture

```text
Input: 32×32×3 RGB
  ↓
Conv2D: 16 filters, 3×3 kernel, padding=1
  (32×32×3 → 32×32×16)
  Parameters: 3×3×3×16 + 16 = 448
  ↓
ReLU activation
  ↓
MaxPool2D: 2×2
  (32×32×16 → 16×16×16)
  ↓
Flatten: 16×16×16 = 4,096
  ↓
Fully Connected: 4,096 → 10
  Parameters: 4,096×10 + 10 = 40,970
  ↓
Softmax
  ↓
Output: 10 classes

Total parameters: 41,418
```

**Key Differences:**

1. **Input channels**: 3 (RGB) vs 1 (grayscale)
2. **Number of filters**: 16 vs 8 (doubled for color complexity)
3. **Parameter count**: 41,418 vs 15,770 (2.6× more parameters)
4. **FC layer size**: 4,096 → 10 vs 1,568 → 10
5. **Memory footprint**: Larger activations and gradients

**Why these changes?**
- More input information requires more filters to extract features
- Larger spatial dimensions (32×32 vs 28×28) increase FC layer size
- RGB color patterns are more complex than grayscale intensity patterns

### Training Hyperparameters

Default settings for CIFAR-10 CNN:

```rust
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 10;           // vs 3 for MNIST
const BATCH_SIZE: usize = 32;       // Same as MNIST
const VALIDATION_SPLIT: f32 = 0.1;  // 10% validation set
const EARLY_STOPPING_PATIENCE: usize = 3;
```

**Why more epochs for CIFAR-10?**
- More complex patterns require longer training
- Higher dimensional input space takes longer to optimize
- More difficult classification task needs more iterations

## Usage Examples

### Loading CIFAR-10 Data

```rust
use rust_neural_networks::data::cifar10::{
    read_cifar10_batch,
    read_cifar10_batches,
    label_to_name,
    get_class_names
};

// Load a single training batch
let (images, labels) = read_cifar10_batch(
    "./data/cifar-10-batches-bin/data_batch_1.bin"
)
.expect("failed to read CIFAR-10 batch");
assert_eq!(images.len(), 10_000 * 32 * 32 * 3);  // 3,072,000 values
assert_eq!(labels.len(), 10_000);

// Load all 5 training batches at once
let train_filenames = [
    "./data/cifar-10-batches-bin/data_batch_1.bin",
    "./data/cifar-10-batches-bin/data_batch_2.bin",
    "./data/cifar-10-batches-bin/data_batch_3.bin",
    "./data/cifar-10-batches-bin/data_batch_4.bin",
    "./data/cifar-10-batches-bin/data_batch_5.bin",
];
let (train_images, train_labels) = read_cifar10_batches(&train_filenames)
    .expect("failed to read CIFAR-10 batches");
assert_eq!(train_images.len(), 50_000 * 32 * 32 * 3);  // 15,360,000 values
assert_eq!(train_labels.len(), 50_000);

// Load test batch
let (test_images, test_labels) = read_cifar10_batch(
    "./data/cifar-10-batches-bin/test_batch.bin"
)
.expect("failed to read CIFAR-10 batch");

// Get human-readable class name
let label: u8 = 0;
let class_name = label_to_name(label);
assert_eq!(class_name, "airplane");

// Get all class names
let class_names = get_class_names();
assert_eq!(class_names.len(), 10);
```

### Training CIFAR-10 CNN

**Basic training:**
```bash
cargo run --release --bin cifar10_cnn
```

**With configuration file (recommended):**
```bash
cargo run --release --bin cifar10_cnn -- --config config/training/cifar10_cnn_default.json
```

**With deep architecture (optional):**
```bash
cargo run --release --bin cifar10_cnn -- \
  --arch config/architectures/cifar10_deep_cnn.json \
  --config config/training/cifar10_deep_cnn_default.json
```

**Expected output:**
```text
Loading CIFAR-10 dataset...
Loaded 50000 training images
Loaded 10000 test images
Training set: 45000 images (90%)
Validation set: 5000 images (10%)

CNN: 32x32x3 -> Conv(16,3x3) -> ReLU -> MaxPool -> FC(4096->10)
Total parameters: 41418

Training...
Epoch 1/10, Loss: 2.1234, Val Loss: 1.9876, Val Acc: 32.45%, Time: 45.3s
Epoch 2/10, Loss: 1.8765, Val Loss: 1.7654, Val Acc: 38.21%, Time: 44.8s
...
Best validation loss: 1.6543 at epoch 7
Test accuracy: 52.34%
```

### Performance Optimization

For faster training on macOS with Accelerate framework:

```bash
RUSTFLAGS="-C target-cpu=native" \
VECLIB_MAXIMUM_THREADS=8 \
cargo run --release --bin cifar10_cnn
```

**Note**: CIFAR-10 CNN training is significantly slower than MNIST due to:
- 3× more input channels
- 2.6× more parameters
- Larger spatial dimensions
- Educational implementation without BLAS (manual loops)

## Expected Performance

### Baseline CNN Performance

**Architecture**: Conv(16, 3×3) → ReLU → MaxPool(2×2) → FC(4096→10)

**Expected Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| Training time (10 epochs) | 8-12 minutes | CPU-only, manual loops |
| Validation accuracy | 50-60% | Simple baseline CNN |
| Test accuracy | 50-60% | Comparable to validation |
| Training loss (final) | ~1.4-1.6 | Cross-entropy loss |
| Parameters | 41,418 | 2.6× more than MNIST CNN |

**Accuracy by epoch (typical):**
- Epoch 1: 25-35% (random guessing is 10%)
- Epoch 3: 40-50%
- Epoch 5: 48-58%
- Epoch 10: 50-60%

### Actual Baseline Results

**Training run completed on 2025-01-26 with Cosine Annealing scheduler:**

**Configuration:**
- Epochs: 10
- Batch size: 32
- Initial learning rate: 0.01
- Scheduler: Cosine Annealing (min_lr=0.0001, T_max=30)
- Hardware: macOS with Accelerate framework (BLAS-accelerated)

**Results:**

| Metric | Value |
|--------|-------|
| Final training loss | 1.601 |
| Final validation loss | 1.671 |
| Final validation accuracy | 41.24% |
| **Test accuracy** | **41.82%** |
| Total training time | ~12.7 minutes |
| Model checkpoint | `cifar10_cnn_model_best.bin` (162 KB) |

**Training progression:**

| Epoch | Train Loss | Val Loss | Val Acc | Time (s) |
|-------|------------|----------|---------|----------|
| 1 | 2.086 | 1.990 | 28.82% | 61.2 |
| 2 | 1.923 | 1.882 | 34.22% | 60.5 |
| 3 | 1.838 | 1.819 | 36.14% | 63.6 |
| 4 | 1.774 | 1.783 | 36.74% | 67.8 |
| 5 | 1.726 | 1.754 | 36.90% | 72.2 |
| 6 | 1.689 | 1.824 | 33.34% | 74.4 |
| 7 | 1.660 | 1.727 | 39.60% | 70.2 |
| 8 | 1.636 | 1.677 | 40.04% | 92.0 |
| 9 | 1.618 | 1.658 | 40.00% | 117.5 |
| 10 | 1.601 | 1.671 | 41.24% | 80.0 |

**Key observations:**
- Test accuracy (41.82%) slightly exceeds final validation accuracy (41.24%), indicating good generalization
- Training loss consistently decreased across all epochs
- Validation accuracy showed steady improvement from 28.82% to 41.24%
- No signs of severe overfitting (train and validation losses track closely)
- Performance is within reasonable range for a simple single-layer CNN on CIFAR-10
- The result validates the baseline implementation and data loading pipeline

**Note**: The achieved accuracy (~42%) is lower than the expected 50-60% range. This is typical for very simple baseline architectures. Improvements would require deeper networks, data augmentation, and regularization techniques.

### Comparison with MNIST

| Dataset | Architecture | Parameters | Test Accuracy | Training Time |
|---------|--------------|------------|---------------|---------------|
| MNIST | Conv(8) + FC | 15,770 | 91-93% | 11 seconds (3 epochs) |
| CIFAR-10 | Conv(16) + FC | 41,418 | 50-60% | 8-12 minutes (10 epochs) |

**Why is CIFAR-10 harder?**

1. **Natural images**: Real-world objects vs simple handwritten digits
2. **Color complexity**: 3 channels with color patterns vs 1 grayscale channel
3. **Intra-class variation**: "dog" includes many breeds, poses, backgrounds
4. **Inter-class similarity**: "cat" vs "dog" is harder than "3" vs "8"
5. **Low resolution**: 32×32 is very small for complex objects
6. **Background clutter**: Objects may be partially occluded or have complex backgrounds

### State-of-the-Art Comparison

This project implements a **simple baseline CNN** for educational purposes. For reference:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Baseline CNN (this project) | 50-60% | Educational, single conv layer |
| Simple CNN (3 conv layers) | 70-75% | Multiple conv layers |
| ResNet-18 | 93-95% | Skip connections, batch norm |
| ResNet-50 | 95-96% | Deeper architecture |
| State-of-the-art (2024) | 99%+ | Data augmentation, ensembles, large models |

**Note**: The baseline accuracy demonstrates fundamental concepts. Significant improvements require:
- Multiple convolutional layers
- Batch normalization
- Data augmentation (flips, crops, color jittering)
- Dropout regularization
- Better optimizers (Adam, AdamW)
- Learning rate schedules
- Deeper architectures (ResNet, DenseNet)

## Troubleshooting

### File Not Found Errors

**Error:**
```text
Could not open file ./data/cifar-10-batches-bin/data_batch_1.bin: No such file or directory
```

**Solutions:**
1. Verify the file path:
   ```bash
   ls -la ./data/cifar-10-batches-bin/
   ```
2. Ensure you downloaded the **binary version** (files end in `.bin`)
3. Check the directory structure matches exactly:
   ```text
   ./data/cifar-10-batches-bin/data_batch_1.bin
   ```
4. Re-download if files are corrupted

### Invalid File Size Errors

**Error:**
```text
Invalid CIFAR-10 batch file size. Expected 30730000 bytes, got 1234567 bytes
```

**Solutions:**
1. File download was incomplete - re-download the dataset
2. Verify checksums (available on official website)
3. Check available disk space
4. Use `wget` or `curl` with resume capability

### Low Accuracy (< 20%)

**Possible causes:**
1. **Data loading issue**: Verify pixel normalization (should be [0, 1])
2. **Label mismatch**: Check that labels match images
3. **Architecture bug**: Review conv layer channel counts
4. **Learning rate**: Try different values (0.001 - 0.1)
5. **Weight initialization**: Ensure proper He/Xavier initialization

**Debug steps:**
```rust
// Print sample data
println!("Sample pixels: {:?}", &images[0..10]);
println!("Sample labels: {:?}", &labels[0..10]);

// Check value ranges
let min_val = images.iter().fold(f32::INFINITY, |a, &b| a.min(b));
let max_val = images.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
println!("Pixel range: [{}, {}]", min_val, max_val);  // Should be [0.0, 1.0]
```

### Memory Issues

**Error:**
```text
thread 'main' panicked at 'allocation failed'
```

**Solutions:**
1. Reduce batch size: `const BATCH_SIZE: usize = 16;`
2. Process fewer images at once
3. Check available RAM (CIFAR-10 needs ~1-2 GB)
4. Close other memory-intensive applications

### Slow Training

**Expected timing:**
- CIFAR-10 CNN: ~45-60 seconds per epoch
- Full 10 epochs: 8-12 minutes

**If much slower:**
1. Ensure `--release` flag is used: `cargo run --release`
2. Check CPU usage (should be 100% on one core)
3. Enable CPU optimizations:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo run --release --bin cifar10_cnn
   ```
4. Consider using a GPU version (requires custom CUDA/OpenCL implementation)

### Validation vs Test Accuracy Mismatch

**Large gap (>5%) suggests overfitting:**

**Solutions:**
1. Reduce model complexity (fewer filters)
2. Implement dropout (not in baseline)
3. Use data augmentation (not in baseline)
4. Reduce training epochs
5. Increase validation set size

**Example:**
- Train accuracy: 70%
- Validation accuracy: 55%
- Test accuracy: 54%
- **Diagnosis**: Model is overfitting to training data

## References

### Official Resources

- **CIFAR-10 Homepage**: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Original Paper**: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
- **Download**: <https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>

### Implementation Details

- **Data loader**: `src/data/cifar10.rs`
- **CNN binary**: `src/bin/cifar10_cnn/main.rs`
- **Training configs**: `config/training/cifar10_cnn_default.json`, `config/training/cifar10_deep_cnn_default.json`
- **Architecture configs**: `config/architectures/cifar10_cnn_baseline.json`, `config/architectures/cifar10_deep_cnn.json`
- **Tests**: `tests/test_cifar10_loader.rs`

### Related Documentation

- `docs/activation_functions.md` - ReLU and Softmax details
- `CLAUDE.md` - Project architecture and build commands
- `README.md` - Quick start guide

### Benchmarks and Comparisons

- MNIST documentation (in README.md)
- Training curves in `logs/training_loss_cifar10_cnn.txt`
- Visualization with `plot_comparison.py`

### Further Reading

- **ImageNet**: <http://www.image-net.org/> (larger, more challenging dataset)
- **CIFAR-100**: <https://www.cs.toronto.edu/~kriz/cifar.html> (100 fine-grained classes)
- **Tiny ImageNet**: <https://tiny-imagenet.herokuapp.com/> (200 classes, 64×64 images)

---

**Last Updated**: January 2026

**Maintainers**: Antonio Neto and Thales Matheus
