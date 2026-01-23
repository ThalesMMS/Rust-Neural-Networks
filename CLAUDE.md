# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational neural network implementations in Rust with Python utilities for visualization. Four independent binary targets implement different architectures for MNIST and XOR classification.

## Agent Behavior & Troubleshooting

> **⚠️ IMPORTANT FOR AGENTS:** Read this before running commands.

1.  **Rust Environment**: If `cargo` is not found in standard paths, explicitly check `/opt/homebrew/bin/cargo` or try sourcing `~/.cargo/env`.
2.  **Code Coverage Limits**: Do **not** attempt to achieve >20% code coverage. The binary-only architecture (no `lib.rs`) makes testing `main()` and I/O functions impossible via standard unit tests. Accept low coverage metrics as expected behavior.
3.  **BLAS Safety**: When modifying `sgemm_wrapper` or matrix operations, double-check all dimensions (M, N, K) and Leading Dimensions (lda, ldb, ldc). Incorrect values passed to the `unsafe` CBLAS FFI will cause immediate segmentation faults.
4.  **Test Duplication**: Tests in `tests/` **must** duplicate logic from `src/*.rs` intentionally. Do not refactor code into a shared library; the goal is self-contained educational binaries.

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

### Multi-Binary Design

Each binary (`src/*.rs`) is self-contained with replicated implementations of common patterns (RNG, matrix ops, activations). This is intentional for educational clarity—there is no shared library.

### Four Models

| Binary | Architecture | Key Features |
|--------|--------------|--------------|
| `mnist_mlp` | 784→512→10 | BLAS-accelerated GEMM via macOS Accelerate |
| `mnist_cnn` | Conv(8,3×3) + MaxPool + FC | Manual loop convolutions |
| `mnist_attention_pool` | Patch→Attention→FFN | 49 tokens, D=16, FF=32 (small for CPU) |
| `mlp_simple` | 2→4→1 | XOR with sigmoid, 1M epochs |

Exportar para as Planilhas

### Language Separation

-   **Rust**: Training, inference, data loading (performance-critical)
    
-   **Python**: Visualization (`plot_comparison.py`), interactive GUI (`digit_recognizer.py`)
    
-   **Interface**: Binary model format (`mnist_model.bin`) + CSV logs in `logs/`
    

### BLAS Integration (MLP only)

Uses `blas-src` with Accelerate framework on macOS. For Linux/Windows, configure OpenBLAS or Intel MKL in `Cargo.toml`. The `sgemm` operation provides 10-100× speedup over naive implementations.

### Test Suite

Tests in `tests/` copy functions from binaries for validation (not a shared library pattern). **Note:** Unit tests inside `src/*.rs` files (via `#[cfg(test)]`) are permitted for coverage, but integration tests in `tests/` are preferred for logic verification.

Key test files:

-   `test_matrix_ops.rs` - GEMM, bias operations
    
-   `test_backward_pass.rs` - Gradient computation
    
-   `test_gradient_checking.rs` - Numerical gradient validation
    
-   `test_activations.rs` - Sigmoid, ReLU, softmax
    

### Data Requirements

MNIST data files (IDX format) expected in `data/` directory. Download instructions in `docs/2b MNIST-Data-Setup.md`.

### Custom Implementations

-   **RNG**: Xorshift PRNG for reproducibility without external dependencies
    
-   **Model serialization**: Little-endian f32 binary format (not ONNX/protobuf)
    
-   **Data loading**: Built-in IDX parser, no external loading libraries