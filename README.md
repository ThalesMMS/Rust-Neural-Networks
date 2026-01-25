# Neural Network Models in Rust + Python (MLP, CNN, Attention)

Authors: Antonio Neto and Thales Matheus

This project implements several small neural nets in Rust for two problems:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2->4->1)

Python utilities are included for visualization and digit recognition. Swift code was moved to the companion repository.

The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Contents

Source code (Rust):

- Binaries: `mnist_mlp.rs`, `mnist_cnn.rs`, `mnist_attention_pool.rs`, `mlp_simple.rs`
- Shared library: `src/lib.rs`, `src/layers/`, `src/utils/`
- Tests: `tests/` (integration tests)
- `Cargo.toml` / `Cargo.lock`

Scripts (Python):

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plots training loss from `logs/`)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files)
- `logs/` (training loss CSV logs with validation metrics)
- `mnist_model.bin`, `mnist_cnn_model.bin`, `mnist_attention_model.bin` (final saved models)
- `*_best.bin` files (best model checkpoints based on validation performance)

Documentation:

- `docs/layer_abstraction_design.md` (detailed architecture guide)
- `CLAUDE.md` (development guidelines)

## Architecture

This project uses a **shared library architecture** with the `rust_neural_networks` library providing reusable layer abstractions and utilities. Each binary (`mnist_mlp`, `mnist_cnn`, `mlp_simple`, `mnist_attention_pool`) imports layers and utilities from the shared library instead of duplicating code.

### Layer Trait

All layer types implement the `Layer` trait (`src/layers/trait.rs`), which provides a common interface:

```rust
pub trait Layer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);
    fn backward(&self, input: &[f32], grad_output: &[f32],
                grad_input: &mut [f32], batch_size: usize);
    fn update_parameters(&mut self, learning_rate: f32);

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn parameter_count(&self) -> usize;
}
```

### Layer Implementations

**DenseLayer** (`src/layers/dense.rs`):
- Fully connected (linear) layer with BLAS acceleration
- Xavier/Glorot initialization for stable training
- Used in all MLP models and as final classifier in CNN

**Conv2DLayer** (`src/layers/conv2d.rs`):
- 2D convolutional layer for spatial feature extraction
- Configurable kernel size, padding, and stride
- Manual loop implementation for educational clarity
- Used in CNN model

### Shared Utilities

**SimpleRng** (`src/utils/rng.rs`):
- Xorshift PRNG for reproducible weight initialization
- No external dependencies, consistent across all models

**Activation Functions** (`src/utils/activations.rs`):
- `sigmoid`, `sigmoid_derivative` - Used in XOR example
- `relu_inplace` - Rectified Linear Unit for hidden layers
- `softmax_rows` - Softmax for multi-class classification

### Benefits

- **~500 lines of code eliminated** through shared abstractions
- **Single source of truth** for layer implementations and utilities
- **Consistent behavior** across all models
- **Easier testing** with comprehensive shared test suite
- **Faster development** of new models by composing existing layers

For detailed architecture documentation, see `docs/layer_abstraction_design.md`.

## MNIST MLP model

Architecture:

- Input: 784 neurons (28x28 pixels)
- Hidden: 512 neurons (ReLU)
- Output: 10 neurons (Softmax)

Default training parameters:

- Learning rate: 0.01
- Batch size: 64
- Epochs: 10
- Validation split: 10% (6,000 samples from 60K training set)
- Early stopping patience: 3 epochs

Expected accuracy: ~94-97% depending on backend and hyperparameters.

Training features:

- **Train/Validation Split**: The 60K MNIST training set is automatically split into 54K training samples and 6K validation samples (`VALIDATION_SPLIT = 0.1`).
- **Validation Tracking**: Validation loss and accuracy are computed and displayed at the end of each epoch.
- **Early Stopping**: Training stops if validation loss doesn't improve by at least 0.001 for 3 consecutive epochs (`EARLY_STOPPING_PATIENCE = 3`, `EARLY_STOPPING_MIN_DELTA = 0.001`).
- **Best Model Checkpointing**: The model with the lowest validation loss is automatically saved to `mnist_model_best.bin`.

Example output:
```
Epoch 1/10, Loss: 0.2543, Val Loss: 0.1832, Val Acc: 94.50%, Time: 0.34s
Epoch 2/10, Loss: 0.1432, Val Loss: 0.1245, Val Acc: 96.12%, Time: 0.33s
...
Early stopping triggered after epoch 5 (no improvement for 3 epochs)
```

Rust MNIST notes:

- Uses `f32` tensors and batched GEMM via BLAS for speed.
- On macOS the default BLAS backend is Accelerate (via `blas-src`).
- Threading is controlled by `VECLIB_MAXIMUM_THREADS` when using Accelerate.
- Test set (10K samples) remains completely separate and is only used for final evaluation.

## MNIST CNN model

Architecture:

- Input: 28x28 image
- Conv: 8 filters (3x3) + ReLU
- MaxPool: 2x2
- FC: 1568 -> 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 3
- Validation split: 10% (6,000 samples)
- Early stopping patience: 3 epochs

Training features: Same as MLP (validation split, early stopping, best model checkpointing to `mnist_cnn_model_best.bin`)

## MNIST attention model

Architecture:

- 4x4 patches => 49 tokens
- Token projection + sinusoidal position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49x49 scores)
- Feed-forward MLP per token (D -> FF -> D)
- Mean-pool tokens -> 10 classes

Default training parameters:

- D model: 64
- FF dim: 128
- Learning rate: 0.01
- Batch size: 32
- Epochs: 8
- Validation split: 10% (6,000 samples)
- Early stopping patience: 3 epochs

Expected accuracy: ~88-91% depending on random seed initialization.

Training features: Same as MLP (validation split, early stopping, best model checkpointing to `mnist_attention_model_best.bin`)

## XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Build and run (Rust)

Build:

```
cargo build --release
```

Run MNIST MLP:

```
cargo run --release --bin mnist_mlp
```

Run XOR:

```
cargo run --release --bin mlp_simple
```

Run MNIST CNN:

```
cargo run --release --bin mnist_cnn
```

Run MNIST attention:

```
cargo run --release --bin mnist_attention_pool
```

Performance tips:

```
RUSTFLAGS="-C target-cpu=native" VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin mnist_mlp
```

Linux/Windows note: the Rust MNIST build is configured for Accelerate on macOS. For other platforms, swap the BLAS backend in `Cargo.toml` (e.g., OpenBLAS) and ensure the library is installed.

## Benchmarks (local runs)

All runs used the default settings unless noted. Training time is reported as total training time; for CNN/attention it is the sum of per-epoch times. XOR accuracy is computed with a 0.5 threshold on the final outputs.

| Model | Language | Command | Epochs | Batch | Train time (s) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST MLP | Rust | `cargo run --release --bin mnist_mlp` | 10 | 64 | 3.33 | 94.17 | BLAS (Accelerate) |
| MNIST CNN | Rust | `cargo run --release --bin mnist_cnn` | 3 | 32 | 11.24 | 91.93 | Conv8/3x3 + MaxPool |
| MNIST Attention | Rust | `cargo run --release --bin mnist_attention_pool` | 8 | 32 | 960 | 91.08 | D=64, FF=128, sinusoidal pos encoding |
| XOR MLP | Rust | `cargo run --release --bin mlp_simple` | 1,000,000 | - | 0.74 | 100.00 | Threshold 0.5 |

Note: results vary by hardware and build flags.

## MNIST dataset

Expected files under `data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Download from:

- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- http://yann.lecun.com/exdb/mnist/

## Visualization

To plot training curves (including validation metrics):

```
python plot_comparison.py
```

The CSV logs in `logs/` now include validation columns:
- Format: `epoch,train_loss,train_time,val_loss,val_accuracy`
- Enables plotting both training and validation curves to visualize overfitting

## Digit recognizer UI

The drawing app loads `mnist_model.bin` and runs inference:

```
python digit_recognizer.py
```

Install dependencies:

```
pip install -r requirements.txt
```

## References

- https://github.com/djbyrne/mlp.c
- http://yann.lecun.com/exdb/mnist/
