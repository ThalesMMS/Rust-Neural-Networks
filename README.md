# Rust Neural Network Models

Authors: Antonio Neto and Thales Matheus

## Overview

This repository contains small neural networks in Rust for:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2->4->1)

Python utilities are included for visualization and digit recognition. The Swift implementation lives in the companion Swift-Neural-Networks repository. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Repository layout

Rust source:

- `mnist_mlp.rs`, `mnist_cnn.rs`, `mnist_attention_pool.rs`, `mlp_simple.rs` (standalone binaries)
- `src/` (shared layers, optimizers, utils, config)
- `tests/` (integration tests)
- `Cargo.toml` / `Cargo.lock`

Configs:

- `config/` (learning-rate scheduler JSON files)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plot training/validation curves from `logs/`)
- `visualize_attention.py` (attention visualization utility)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files)
- `logs/` (training metrics logs)
- `mnist_model.bin`, `mnist_model_best.bin` (example and best-checkpoint files)
- `mnist_cnn_model_best.bin`, `mnist_attention_model_best.bin` (generated during training)

## Models

### MNIST MLP

Architecture:

- Input: 784 neurons (28x28 pixels)
- Hidden: 512 neurons (ReLU)
- Output: 10 neurons (Softmax)

Default training parameters:

- Learning rate: 0.01
- Batch size: 64
- Epochs: 10
- Validation split: 10%
- Early stopping patience: 3 (min delta 0.001)

Expected accuracy: ~94-97% depending on hardware and hyperparameters.

### MNIST CNN

Architecture:

- Input: 28x28 image
- Conv: 8 filters (3x3) + ReLU
- MaxPool: 2x2
- FC: 1568 -> 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 3
- Validation split: 10%
- Early stopping patience: 3

### MNIST attention model

Architecture:

- 4x4 patches -> 49 tokens
- Token projection + sinusoidal position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49x49 scores)
- Feed-forward MLP per token (64 -> 128 -> 64)
- Mean-pool tokens -> 10 classes

Default training parameters:

- D model: 64
- FF dim: 128
- Learning rate: 0.01
- Batch size: 32
- Epochs: 8
- Validation split: 10%
- Early stopping patience: 3

Expected accuracy: ~88-91% depending on seed and hyperparameters.

### XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Training behavior

- Training uses a fixed train/validation split and reports validation metrics per epoch.
- Best checkpoints are saved when validation improves.
- An optional learning-rate scheduler can be provided via a JSON config file (see `config/`).

## Build and run (Rust)

Build:

```bash
cargo build --release
```

Run MNIST MLP:

```bash
cargo run --release --bin mnist_mlp
```

Run with a learning-rate schedule:

```bash
cargo run --release --bin mnist_mlp -- config/mnist_mlp_cosine.json
```

Run XOR:

```bash
cargo run --release --bin mlp_simple
```

Run MNIST CNN:

```bash
cargo run --release --bin mnist_cnn
```

Run MNIST attention:

```bash
cargo run --release --bin mnist_attention_pool
```

Performance tips:

```bash
RUSTFLAGS="-C target-cpu=native" VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin mnist_mlp
```

Linux/Windows note: the default BLAS backend is Accelerate on macOS. For other platforms, swap the BLAS backend in `Cargo.toml` (e.g., OpenBLAS) and ensure the library is installed.

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

```bash
python plot_comparison.py
```

## Digit recognizer UI

The drawing app loads `mnist_model.bin` and runs inference:

```bash
python digit_recognizer.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## References

- https://github.com/djbyrne/mlp.c
- http://yann.lecun.com/exdb/mnist/
