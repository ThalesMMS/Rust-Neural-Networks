# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational neural network implementations in Rust with Python utilities for visualization. The project provides a shared library (`rust_neural_networks`) with reusable layer abstractions and utilities, used by four binary targets that implement different architectures for MNIST and XOR classification.

## Agent Behavior & Troubleshooting

> **⚠️ IMPORTANT FOR AGENTS:** Read this before running commands.

1.  **Rust Environment**: If `cargo` is not found in standard paths, explicitly check `/opt/homebrew/bin/cargo` or try sourcing `~/.cargo/env`.
2.  **Code Coverage**: The shared library in `src/lib.rs` and `src/` modules is testable via standard unit and integration tests. Binary entry points (`main()`) and I/O code remain difficult to test without mocking, so accept lower coverage for binary-specific code.
3.  **BLAS Safety**: When modifying matrix operations in `DenseLayer` or direct BLAS calls, double-check all dimensions (M, N, K) and Leading Dimensions (lda, ldb, ldc). Incorrect values passed to the `unsafe` CBLAS FFI will cause immediate segmentation faults.
4.  **Layer Trait**: All layer implementations (`DenseLayer`, `Conv2DLayer`) must implement the `Layer` trait from `src/layers/trait.rs`. This provides a consistent interface for forward/backward propagation and parameter updates across different layer types.
5.  **Validation Split**: All MNIST models use a 10% validation split (`VALIDATION_SPLIT = 0.1`), splitting the 60K training set into 54K training and 6K validation samples. The test set remains completely separate.

## Build and Test Commands

```bash
# Build all binaries (release mode required for BLAS performance)
cargo build --release

# Run specific models
cargo run --release --bin mnist_mlp            # MLP with BLAS acceleration
cargo run --release --bin mnist_cnn            # CNN with manual convolutions
cargo run --release --bin mnist_attention_pool # Transformer-style attention
cargo run --release --bin mlp_simple           # XOR example

# Run tests
cargo test --verbose

# Run a single test file
cargo test --test test_matrix_ops --verbose

# Run a specific test
cargo test test_sgemm_identity --verbose

# Lint and format
cargo fmt -- --check
cargo clippy --all-targets --all-features

# Optimized execution (macOS)
RUSTFLAGS="-C target-cpu=native" VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin mnist_mlp
```

## Architecture

### Shared Library Architecture

The project uses a shared library (`src/lib.rs`) to reduce code duplication and provide consistent abstractions:

-   **`src/layers/`** - Layer trait and implementations (`DenseLayer`, `Conv2DLayer`)
-   **`src/utils/`** - Shared utilities (SimpleRng, activation functions, etc.)
-   **Binary targets** - Training scripts in root directory (`mnist_mlp.rs`, `mnist_cnn.rs`, etc.) that import and use the shared library

Each binary imports layers and utilities from `rust_neural_networks` crate:
```rust
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::{SimpleRng, relu_inplace, softmax_rows};
```

#### Layer Trait

The core abstraction is the `Layer` trait (`src/layers/trait.rs`) which defines:
-   `forward()` - Forward propagation through the layer
-   `backward()` - Backward propagation and gradient computation
-   `update_parameters()` - Apply gradient descent updates
-   Helper methods: `input_size()`, `output_size()`, `parameter_count()`

All layer types implement this trait, enabling modular network construction and consistent training loops.

### Four Models

| Binary | Architecture | Key Features |
|--------|--------------|--------------|
| `mnist_mlp` | 784→512→10 | BLAS-accelerated GEMM via macOS Accelerate |
| `mnist_cnn` | Conv(8,3×3) + MaxPool + FC | Manual loop convolutions |
| `mnist_attention_pool` | Patch→Attention→FFN | 49 tokens, D=64, FF=128 (increased for capacity) |
| `mlp_simple` | 2→4→1 | XOR with sigmoid, 1M epochs |

### Language Separation

-   **Rust**: Training, inference, data loading (performance-critical)
-   **Python**: Visualization (`plot_comparison.py`), interactive GUI (`digit_recognizer.py`)
-   **Interface**: Binary model format (`mnist_model.bin`) + CSV logs in `logs/`

### BLAS Integration

The `DenseLayer` in the shared library uses BLAS (Basic Linear Algebra Subprograms) for matrix operations via the `cblas` crate:
-   **macOS**: Accelerate framework (via `blas-src` with `accelerate` feature)
-   **Linux**: OpenBLAS (via `openblas-src` with static linking)
-   **Windows**: OpenBLAS (requires system installation or vcpkg)

The BLAS `sgemm` operation provides 10-100× speedup over naive matrix multiplication implementations.

### Test Suite

Tests in `tests/` validate the shared library components. The library architecture (with `src/lib.rs`) enables standard unit and integration testing of layers and utilities. Binary-specific code (I/O, `main()` functions) has lower test coverage as expected.

**Test organization:**
-   **Unit tests**: In-module tests within `src/` files using `#[cfg(test)]`
-   **Integration tests**: Files in `tests/` that import the library as external users would

Key test files:
-   `test_matrix_ops.rs` - GEMM, bias operations
-   `test_backward_pass.rs` - Gradient computation
-   `test_gradient_checking.rs` - Numerical gradient validation
-   `test_activations.rs` - Sigmoid, ReLU, softmax
-   Layer-specific tests for `DenseLayer` and `Conv2DLayer`

### Data Requirements

MNIST data files (IDX format) expected in `data/` directory. Download instructions in `docs/2b MNIST-Data-Setup.md`.

### Custom Implementations

-   **RNG**: Xorshift PRNG for reproducibility without external dependencies (in `src/utils/rng.rs`)
-   **Model serialization**: Little-endian f32 binary format (not ONNX/protobuf)
-   **Data loading**: Built-in IDX parser, no external loading libraries

### Training Infrastructure

All MNIST models include proper machine learning training practices:

**Data Splitting:**
-   `VALIDATION_SPLIT = 0.1` - 10% of training data reserved for validation
-   60K training set → 54K training + 6K validation
-   10K test set remains completely separate for final evaluation

**Early Stopping:**
-   `EARLY_STOPPING_PATIENCE = 3` - Stops after 3 epochs without improvement
-   `EARLY_STOPPING_MIN_DELTA = 0.001` - Minimum improvement threshold
-   Prevents overfitting and reduces unnecessary training time

**Model Checkpointing:**
-   Best model automatically saved based on validation loss (MLP/CNN) or accuracy (Attention)
-   Checkpoint filenames: `mnist_model_best.bin`, `mnist_cnn_model_best.bin`, `mnist_attention_model_best.bin`
-   Final model saved at end of training to standard filename

**Validation Metrics:**
-   Validation loss and accuracy computed at end of each epoch
-   CSV logs include validation metrics: `epoch,train_loss,train_time,val_loss,val_accuracy`
-   Enables detection of overfitting through training/validation curve comparison

**Training Output Format:**
```
Epoch X/Y, Loss: Z, Val Loss: W, Val Acc: V%, Time: Ts
```

When validation loss plateaus, early stopping message:
```
Early stopping triggered after epoch X (no improvement for 3 epochs)
```
