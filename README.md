# Rust Neural Network Models

Authors: Antonio Neto and Thales Matheus

## Overview

This repository contains small neural networks in Rust for:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- CIFAR-10 object classification (CNN)
- XOR toy example (2->4->1)

Python utilities are included for visualization and digit recognition. The Swift implementation lives in the companion Swift-Neural-Networks repository. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Setup (start here)

Run the preflight command first. It prints platform-specific setup notes (BLAS, Metal/CUDA, Python tooling, datasets, WASM) and recommended Cargo commands.

```bash
cargo run --bin preflight

# Machine-readable output (for CI/scripts)
cargo run --bin preflight -- --format json
```

This command is safe to run in CI: it does **not** require special privileges and does **not** perform any network access.

## CPU-only (golden path)

## Smoke tests (quick verification)

## Maintainer verification checklist (manual matrix)

Use this checklist before releases (or after big dependency changes) to ensure the documented setup paths still work.

- **Always (any platform)**
  - `cargo run --bin preflight`
  - `cargo test`

- **macOS**
  - CPU-only: `cargo test`
  - Metal (requires Apple Silicon or supported GPU + Xcode CLT): `cargo test --features gpu-metal`

- **Linux**
  - CPU-only: `cargo test`
  - CUDA (requires NVIDIA GPU + driver + CUDA toolkit):
    - `cargo run --bin preflight` (should show CUDA as PASS)
    - `cargo test --features gpu-cuda`

- **Windows (MSVC)**
  - CPU-only (requires OpenBLAS via vcpkg):
    - `cargo run --bin preflight` (should find `vcpkg`)
    - `cargo test`
  - CUDA (requires NVIDIA GPU + driver + CUDA toolkit):
    - `cargo run --bin preflight` (should show CUDA as PASS)
    - `cargo test --features gpu-cuda`

- **WASM (any OS)**
  - `cargo run --bin preflight` (should find wasm-pack + wasm32 target)
  - `cd wasm && wasm-pack build --target web`

Notes:
- CUDA paths require appropriate hardware; CI can still run `preflight` and CPU tests.
- If anything fails, start by re-running `cargo run --bin preflight` and follow the remediation hints.


These are lightweight commands you can run to confirm each supported path is working.

### CPU (all platforms)

```bash
cargo run --bin preflight
cargo test
```

### Metal (macOS)

```bash
cargo run --bin preflight
cargo test --features gpu-metal
```

### CUDA (Linux / Windows)

```bash
cargo run --bin preflight
cargo test --features gpu-cuda
```

### WebAssembly demo

```bash
cargo run --bin preflight
cd wasm
wasm-pack build --target web
```

Then serve the `wasm/` directory (see the WebAssembly Demo section below).


If you want the simplest setup with the fewest external dependencies, stick to the default **CPU-only** build (no GPU feature flags).

```bash
# 1) Sanity-check your environment (Rust toolchain, BLAS notes, Python tooling, datasets, WASM)
cargo run --bin preflight

# 2) Run unit tests (CPU-only)
cargo test
```

Minimal runnable example:

```bash
# Note: this runs a tiny XOR MLP training loop (no datasets required)
cargo run --bin mlp_simple
```

> Notes
> - On macOS, BLAS is provided by **Accelerate** automatically.
> - On Linux, **OpenBLAS** is built automatically (you still need basic build tools like a C compiler).
> - On Windows, you typically need to install OpenBLAS (see the Windows section below).

## Platform setup notes

### macOS (CPU and Metal)

- **CPU-only (default):** no extra feature flags.
  - BLAS: uses Apple **Accelerate** via `blas-src` automatically.
- **Metal (optional):** enable the Cargo feature `gpu-metal`.
  - Prereq: Xcode Command Line Tools.

Commands:

```bash
# CPU-only
cargo test

# Metal (macOS only)
cargo test --features gpu-metal
```

If Metal builds fail due to missing toolchain components, install Xcode Command Line Tools:

```bash
xcode-select --install
```

You can also re-run the preflight tool to confirm Metal/toolchain detection:

```bash
cargo run --bin preflight
```

### Linux (CPU and CUDA)

- **CPU-only (default):** no extra feature flags.
  - BLAS: uses **OpenBLAS** via `openblas-src` (built automatically as part of the Rust build).
  - Prereqs: a C/C++ toolchain and basic build tools (e.g., `gcc`/`clang`, `make`).
- **CUDA (optional):** enable the Cargo feature `gpu-cuda`.
  - Prereqs: NVIDIA driver + CUDA toolkit (so `nvidia-smi` and `nvcc` are available).

Commands:

```bash
# CPU-only
cargo test

# CUDA (Linux only; requires CUDA toolkit)
cargo test --features gpu-cuda
```

If you hit build/link errors on Linux, re-run preflight for actionable hints (toolchain, CUDA detection):

```bash
cargo run --bin preflight
```

### Windows (CPU and CUDA)

- **Prereq (Rust toolchain):** install the *MSVC* toolchain (recommended) via Rustup.
- **CPU-only (default):** no extra feature flags.
  - BLAS: uses **OpenBLAS** via `openblas-src` configured for a *system-provided* OpenBLAS on Windows.
  - Recommended path: install OpenBLAS via **vcpkg**.
- **CUDA (optional):** enable the Cargo feature `gpu-cuda`.
  - Prereqs: NVIDIA driver + CUDA toolkit (`nvcc`), and (optionally) `nvidia-smi`.

Commands (PowerShell):

```powershell
# CPU-only
cargo test

# CUDA (Windows only; requires CUDA toolkit)
cargo test --features gpu-cuda
```

#### Install OpenBLAS (vcpkg)

1. Install vcpkg: https://github.com/microsoft/vcpkg
2. Install OpenBLAS:

```powershell
vcpkg install openblas
```

3. Ensure the `VCPKG_ROOT` environment variable is set and the vcpkg tool is on your PATH.

If linking still fails, re-run preflight for the exact remediation text:

```powershell
cargo run --bin preflight
```

#### CUDA notes

If `nvcc` is not found, install the CUDA toolkit from NVIDIA and ensure its `bin/` directory is on PATH. Preflight will report what it can detect:

```powershell
cargo run --bin preflight
```

## Dataset setup (MNIST / CIFAR-10)

Before running any training binaries, verify that the expected datasets are present under `./data`:

```bash
# Verify MNIST + CIFAR-10 (default)
cargo run --bin dataset-helper -- verify

# Or verify individually
cargo run --bin dataset-helper -- verify --mnist
cargo run --bin dataset-helper -- verify --cifar10
```

If a dataset is missing or mis-extracted (e.g., extra directory nesting, archives not unpacked), the helper prints the expected paths and the official source URLs.

## Learning Tutorials

**New to neural networks?** Start with our curated learning paths (Beginner / Intermediate / Advanced) and corresponding reproducibility checks:

→ **[Curated Learning Paths](docs/learning_paths.md)** - Tiered paths, prerequisites, and smoke-check commands

You can also browse the full tutorial list directly:

→ **[Step-by-Step Tutorials](docs/tutorials/README.md)** - Progressive tutorial series with worked examples

**Tutorial series:**
1. **[XOR MLP](docs/tutorials/01_xor_mlp.md)** (30-45 min) - Build your first network, understand backpropagation
2. **[MNIST MLP](docs/tutorials/02_mnist_mlp.md)** (60-90 min) - Scale to real data with 784→512→10 classifier
3. **[MNIST CNN](docs/tutorials/03_mnist_cnn.md)** (90-120 min) - Add convolutions for spatial feature extraction

Each tutorial includes:
- **Incremental construction** - Build networks layer by layer with explanations
- **Worked examples** - Concrete numerical calculations you can verify
- **Expected outputs** - Checkpoints to confirm your understanding
- **Exercises** - Modifications to deepen learning (beginner to advanced)

Perfect for students, researchers, and practitioners wanting to understand neural networks beyond high-level APIs.

## Architecture Dashboard

**Compare model architectures side-by-side** with interactive visualizations and performance metrics.

The Architecture Dashboard provides a comprehensive comparison tool for understanding how different neural network architectures perform on MNIST digit classification. Compare MLP, CNN, and Attention models across multiple dimensions:

- **Training curves** - Loss and accuracy progression over epochs
- **Architecture diagrams** - Visual representation of layer structure and connectivity
- **Performance metrics** - Final accuracy, training time, parameter count, and efficiency
- **Gradient flow** - Understand training dynamics and identify vanishing/exploding gradients

**Quick start:**

```bash
# Train all models (generates logs in logs/)
cargo run --release --bin mnist_mlp
cargo run --release --bin mnist_cnn
cargo run --release --bin mnist_attention_pool

# Launch dashboard
python architecture_dashboard.py
```

The dashboard opens in your browser at `http://localhost:8050` with interactive plots and comparisons.

→ **[Architecture Dashboard Documentation](docs/architecture_dashboard.md)** - Full guide with examples and analysis

## Repository layout

Rust source:

- `mnist_mlp.rs`, `mnist_cnn.rs`, `mnist_attention_pool.rs`, `cifar10_cnn.rs`, `mlp_simple.rs` (standalone binaries)
- `hyperparameter_sweep.rs` (hyperparameter sweep orchestrator binary)
- `src/` (shared layers, optimizers, utils, config, sweep)
- `tests/` (integration tests)
- `Cargo.toml` / `Cargo.lock`

Configs:

- `config/training/` (training hyperparameters for all models)
- `config/architectures/` (network architecture definitions)
- `config/sweeps/` (hyperparameter sweep configurations)
- `config/` (learning-rate scheduler configs, activation configs)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plot training/validation curves from `logs/`)
- `compare_sweep_results.py` (compare hyperparameter sweep results with plots)
- `visualize_attention.py` (attention visualization utility)
- `visualize_gradients.py` (gradient flow visualization and analysis)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files, CIFAR-10 binary files)
- `logs/` (training metrics logs)
- `runs/` (experiment registry: one folder per run with `run.json` + artifacts)
- `mnist_model.bin`, `mnist_model_best.bin` (example and best-checkpoint files)
- `mnist_cnn_model_best.bin`, `mnist_attention_model_best.bin` (generated during training)

## Experiment registry (runs/)

Training binaries continue to write CSV logs under `logs/`, but they also write a structured *run record* under `runs/`.

Each run creates a directory like:

```
runs/<run_id>/
  run.json
  artifacts/...
```

The `run.json` record includes:

- `run_id`, `timestamp`, `run_name` (optional)
- `model_type`
- config snapshot (the full training config JSON)
- the actual RNG `seed`
- final metrics (loss/accuracy/time/epochs)
- artifact paths (training CSV log, checkpoints, etc.)
- environment metadata (Rust version, OS, git commit if available)

### List runs

```bash
cargo run --bin registry -- list
# optionally:
cargo run --bin registry -- list --registry-dir runs
```

### Compare runs

```bash
cargo run --bin registry -- compare <run_id_1> <run_id_2>
```

### Export sweep-compatible summaries for Python plotting

`compare_sweep_results.py` expects a flat JSON/CSV table of sweep results.
You can export that table from the registry runs:

```bash
# JSON (default)
cargo run --bin registry -- export-sweep --registry-dir runs --format json > sweep_results.json

# CSV
cargo run --bin registry -- export-sweep --registry-dir runs --format csv > sweep_results.csv

# then:
python compare_sweep_results.py sweep_results.json
```

### Hyperparameter sweeps

The sweep orchestrator also understands the registry output. It will prefer reading final metrics from `runs/<run_id>/run.json`, but remains backward-compatible with parsing the newest `logs/*.csv` when a run record is missing.

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

### CIFAR-10 CNN

**Baseline Architecture (Currently Implemented):**

- Input: 32x32x3 RGB image (3072 pixels)
- Conv: 16 filters (3x3) + ReLU + padding=1
- MaxPool: 2x2
- FC: 4096 -> 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 10
- Validation split: 10%
- Early stopping patience: 3 (min delta 0.001)

Expected accuracy: ~50-60% depending on hardware and hyperparameters.

**Deep CNN Architecture (Designed, Not Yet Trained):**

A deeper 17-layer architecture has been designed to improve CIFAR-10 performance:

- 6 convolutional layers in 3 progressive blocks (32→64→128 filters)
- Batch normalization after each conv layer
- Stride-based downsampling (no MaxPool)
- Dropout regularization (0.3, 0.5)
- Classifier: 8192 → 256 → 10
- Total parameters: ~1.2M (vs ~65K baseline)

Architecture config: `config/architectures/cifar10_deep_cnn.json`
Training config: `config/training/cifar10_deep_cnn_default.json`
Design rationale: `docs/cifar10_architecture_design.md`

**Status:** Architecture fully designed and tested. Train it by passing the architecture JSON at runtime:

```bash
# Verify dataset is present under ./data
cargo run --bin dataset-helper -- verify --cifar10

# Train
cargo run --release --bin cifar10_cnn -- --arch config/architectures/cifar10_deep_cnn.json
```

Target performance: 70%+ test accuracy.

**Note:** CIFAR-10 is significantly harder than MNIST. The baseline CNN architecture is intentionally simple for educational purposes. State-of-the-art models typically achieve 90%+ accuracy with deeper architectures, data augmentation, and more training. The deep architecture design demonstrates how architectural choices (depth, normalization, regularization) can significantly improve performance.

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

## Architecture configuration

The project supports defining neural network architectures via JSON configuration files, enabling rapid experimentation without code changes.

**Features:**
- Define architectures by specifying a sequence of layers (Dense, Conv2D, BatchNorm, Dropout)
- Automatic validation of layer connections and parameters
- Example configs provided in `config/architectures/`

**Example configs:**
- `mlp_simple.json` - Simple 784→256→10 MLP
- `mlp_medium.json` - Medium 784→512→256→10 MLP
- `cnn_simple.json` - Convolutional network with Conv2D + Dense layers

**Usage in code:**
```rust
use rust_neural_networks::architecture::{load_architecture, build_model};

let config = load_architecture("config/architectures/mlp_simple.json")?;
let layers = build_model(&config, &mut rng)?;
```

For detailed documentation including layer types, parameters, validation rules, and more examples, see [`docs/architecture_config.md`](docs/architecture_config.md).

## Hyperparameters configuration

The project supports externalizing all training hyperparameters to JSON configuration files, enabling experimentation without recompilation.

**Features:**
- Control learning rate, batch size, epochs, validation split, and early stopping
- Configure learning rate schedulers (step decay, exponential, cosine annealing)
- Choose activation functions (ReLU, LeakyReLU, ELU, GELU, Swish, Tanh)
- Select optimizers (SGD, Adam, AdamW, RMSprop) with specific hyperparameters
- Default configs provided for all models in `config/training/`
- Automatic validation with helpful error messages

**Example config:**
```json
{
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001,
  "optimizer_type": "sgd",
  "momentum": 0.9,
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu"
}
```

**Optimizer selection:**

The project supports four optimizers with different characteristics:

- **SGD (Stochastic Gradient Descent)** - Simple gradient descent with optional momentum
  ```json
  {
    "optimizer_type": "sgd",
    "momentum": 0.9
  }
  ```
  - Best for: Simple problems, full control over learning dynamics
  - Characteristics: No adaptive learning rates, requires careful LR tuning
  - Typical learning rate: 0.01-0.1

- **Adam (Adaptive Moment Estimation)** - Combines momentum and adaptive learning rates
  ```json
  {
    "optimizer_type": "adam",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  }
  ```
  - Best for: General purpose, sparse gradients, noisy data
  - Characteristics: Fast convergence, adapts LR per parameter
  - Typical learning rate: 0.001-0.01

- **AdamW (Adam with Weight Decay)** - Adam with decoupled weight decay regularization
  ```json
  {
    "optimizer_type": "adamw",
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01
  }
  ```
  - Best for: Large models, when regularization is important, modern deep learning
  - Characteristics: Better generalization than Adam, decoupled weight decay
  - Typical learning rate: 0.001-0.01

- **RMSprop (Root Mean Square Propagation)** - Adaptive learning rate with moving average
  ```json
  {
    "optimizer_type": "rmsprop",
    "rmsprop_decay": 0.9,
    "rmsprop_epsilon": 1e-8
  }
  ```
  - Best for: RNNs, non-stationary objectives
  - Characteristics: Handles sparse gradients well, simpler than Adam
  - Typical learning rate: 0.001-0.01

**Default configs:**
- `config/training/mnist_mlp_default.json` - MNIST MLP training parameters
- `config/training/mnist_cnn_default.json` - MNIST CNN training parameters
- `config/training/mnist_attention_default.json` - MNIST Attention training parameters
- `config/training/cifar10_cnn_default.json` - CIFAR-10 CNN training parameters
- `config/training/mlp_simple_default.json` - XOR MLP training parameters

**Usage with CLI:**

All binaries accept a `--config` flag to specify a custom configuration file:

```bash
# Use default config (loaded automatically)
cargo run --release --bin mnist_mlp

# Use custom config file
cargo run --release --bin mnist_mlp -- --config config/training/mnist_mlp_default.json

# Experiment with different learning rates
cargo run --release --bin mnist_cnn -- --config config/training/my_experiment.json

# CIFAR-10 with custom config
cargo run --release --bin cifar10_cnn -- --config config/training/cifar10_aggressive.json
```

**Quick experimentation example:**

Create a custom config to experiment with higher learning rate:

```bash
# Copy default config
cp config/training/mnist_mlp_default.json config/training/mnist_mlp_fast.json

# Edit the config (change learning_rate to 0.1)
# Then run with the new config
cargo run --release --bin mnist_mlp -- --config config/training/mnist_mlp_fast.json
```

**Experiment with different optimizers:**

```bash
# Try AdamW optimizer (recommended for deep networks)
cargo run --release --bin mnist_mlp -- --config config/mnist_mlp_adamw.json

# Try Adam optimizer (fast convergence)
cargo run --release --bin mnist_mlp -- --config config/mnist_mlp_adam.json

# Try RMSprop optimizer (good for RNNs)
cargo run --release --bin mnist_mlp -- --config config/mnist_mlp_rmsprop.json

# Use SGD with momentum (traditional approach)
cargo run --release --bin mnist_mlp -- --config config/mnist_mlp_sgd_momentum.json
```

For comprehensive documentation including all hyperparameters, validation rules, scheduler types, optimizer details, and experimentation guide, see [`docs/hyperparameters.md`](docs/hyperparameters.md).

## Hyperparameter sweeps

The project includes a hyperparameter sweep utility that automates running multiple training configurations and comparing results, enabling systematic hyperparameter optimization without code changes.

**Features:**
- Define parameter ranges (learning rate, batch size, scheduler type, etc.) in a single JSON config
- Automatically generates all combinations (Cartesian product) of parameter values
- Runs training for each configuration sequentially
- Aggregates results to structured JSON with metrics for each run
- Python visualization utility generates comparison plots and recommendations

**Example sweep config:**

```json
{
  "base_config": "config/training/mnist_mlp_default.json",
  "target_binary": "mnist_mlp",
  "description": "Learning rate and batch size sweep for MNIST MLP",

  "learning_rate": [0.001, 0.01, 0.1],
  "batch_size": [32, 64, 128],
  "scheduler_type": ["step_decay", "exponential"]
}
```

This example generates **18 configurations** (3 learning rates × 3 batch sizes × 2 scheduler types).

**Example sweep configs:**
- `config/sweeps/mnist_mlp_sweep.json` - Comprehensive MNIST MLP sweep

**Usage:**

Run a hyperparameter sweep:

```bash
# Run full sweep
cargo run --release --bin hyperparameter_sweep -- \
  --target mnist_mlp \
  --sweep config/sweeps/mnist_mlp_sweep.json

# Run quick sweep with reduced epochs for testing
cargo run --release --bin hyperparameter_sweep -- \
  --target mnist_mlp \
  --sweep config/sweeps/mnist_mlp_sweep.json \
  --quick
```

**Expected output:**

```
=== Hyperparameter Sweep Configuration ===
Target binary: mnist_mlp
Sweep config: config/sweeps/mnist_mlp_sweep.json

Parameter ranges:
  learning_rate: [0.001, 0.01, 0.1]
  batch_size: [32, 64, 128]
  scheduler_type: ["step_decay", "exponential"]

Total configurations: 18

=== Running Training Configurations ===
[1/18] Running config 1...
  LR: 0.001, Batch: 32, Scheduler: step_decay
  Config: /tmp/sweep_config_1.json
  Completed in 45.2s

[2/18] Running config 2...
  ...

=== Sweep Results ===
┌──────┬────────────┬───────────┬──────────┬──────────┬──────────┬──────────┐
│ Rank │ Config ID  │ Learn.Rate│ BatchSize│ Scheduler│ Val Loss │ Val Acc  │
├──────┼────────────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│  1   │     7      │   0.01    │    64    │step_decay│  0.0823  │  97.45%  │
│  2   │    13      │   0.01    │   128    │exponential│ 0.0891  │  97.12%  │
│  3   │     4      │   0.001   │    64    │exponential│ 0.0934  │  96.88%  │
...
└──────┴────────────┴───────────┴──────────┴──────────┴──────────┴──────────┘

Best configuration (by validation loss): Config 7
  Learning rate: 0.01
  Batch size: 64
  Scheduler: step_decay
  Validation loss: 0.0823
  Validation accuracy: 97.45%

Results saved to: ./logs/sweep_results_20260211_143052.json
```

**Visualize results:**

Use the Python comparison utility to generate plots and recommendations:

```bash
# Generate comparison plots
python compare_sweep_results.py logs/sweep_results_20260211_143052.json

# Output:
# - Console: Summary table, ranked configurations, recommendations
# - File: logs/sweep_comparison_20260211_143052.png (6 comparison plots)
```

**Comparison plots include:**
1. Learning rate vs validation loss (log scale)
2. Learning rate vs validation accuracy (log scale)
3. Batch size vs validation loss
4. Batch size vs validation accuracy
5. Loss vs accuracy by scheduler type (color-coded)
6. Training time vs accuracy (colored by validation loss)

**Recommendations provided:**
- **Best Accuracy**: Configuration with highest validation accuracy
- **Fastest Training**: Configuration with shortest training time
- **Best Balance**: Optimal tradeoff between accuracy and training time

**Sweep results format:**

The results JSON contains detailed metrics for each configuration:

```json
[
  {
    "config_id": 1,
    "learning_rate": 0.01,
    "batch_size": 64,
    "epochs": 10,
    "scheduler_type": "step_decay",
    "final_train_loss": 0.0234,
    "final_val_loss": 0.0823,
    "final_val_accuracy": 97.45,
    "training_time_seconds": 45.2
  },
  ...
]
```

**Tips:**
- Start with a coarse sweep (2-3 values per parameter) to identify promising regions
- Use `--quick` flag during development to test sweep configs with reduced epochs
- Each training run is independent - failed runs don't stop the entire sweep
- Results are automatically ranked by validation loss
- Check `logs/` directory for individual training logs from each configuration

## Build and run (Rust)

Build:

```bash
cargo build --release
```

Run MNIST MLP (uses default config):

```bash
cargo run --release --bin mnist_mlp
```

Run MNIST MLP with custom config (e.g., different learning rate scheduler):

```bash
cargo run --release --bin mnist_mlp -- --config config/mnist_mlp_cosine.json
```

Run XOR (uses default config):

```bash
cargo run --release --bin mlp_simple
```

Run MNIST CNN (uses default config):

```bash
cargo run --release --bin mnist_cnn
```

Run MNIST CNN with custom config:

```bash
cargo run --release --bin mnist_cnn -- --config config/training/mnist_cnn_default.json
```

Run MNIST attention (uses default config):

```bash
cargo run --release --bin mnist_attention_pool
```

Run CIFAR-10 CNN:

```bash
# Verify dataset is present under ./data
cargo run --bin dataset-helper -- verify --cifar10

# Train
cargo run --release --bin cifar10_cnn
```

Run with a learning-rate schedule:

```bash
cargo run --release --bin cifar10_cnn -- config/cifar10_cnn_baseline.json
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
| CIFAR-10 CNN | Rust | `cargo run --release --bin cifar10_cnn` | 5/10 | 32 | 30.37 | 10.00 | Conv16/3x3 -> Dense, RGB input; early stopped |
| XOR MLP | Rust | `cargo run --release --bin mlp_simple` | 1,000,000 | - | 0.74 | 100.00 | Threshold 0.5 |

Note: results vary by hardware and build flags. The CIFAR-10 CNN run used the documented default command after `cargo run --bin dataset-helper -- verify --cifar10`; the default config requested 10 epochs but early stopping ended the run after epoch 5, so the training time is the sum of the five logged epoch times.

## MNIST dataset

Expected files under `data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Download from:

- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- http://yann.lecun.com/exdb/mnist/

## CIFAR-10 dataset

Expected files under `data/cifar-10-batches-bin/`:

- `data_batch_1.bin` through `data_batch_5.bin` (50,000 training images)
- `test_batch.bin` (10,000 test images)
- `batches.meta.txt` (class label names)

Download the CIFAR-10 binary version from:

- https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-10 binary version)
- Direct link: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

Extract the archive and place the `cifar-10-batches-bin/` directory inside the `data/` directory.

CIFAR-10 contains 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

For more details on the CIFAR-10 format and RGB handling, see `docs/cifar10_dataset.md`.

## WebAssembly Demo

Try the neural network in your browser with the interactive WebAssembly demo! Draw digits and see real-time predictions, all running client-side with no server required.

### WASM setup (build/run)

Prerequisites:

- Rust toolchain (cargo/rustc)
- wasm-pack
- rustup target `wasm32-unknown-unknown`
- (Optional) node/npm for some wasm-pack workflows

You can also run the preflight tool to check these automatically:

```bash
cargo run --bin preflight
```

**Quick Start:**

```bash
# 1. Install wasm-pack (if needed)
# https://rustwasm.github.io/wasm-pack/installer/

# 2. Ensure the WASM target is installed
rustup target add wasm32-unknown-unknown

# 3. Build the WASM module
cd wasm
wasm-pack build --target web --release

# 4. Copy WASM package to demo directory
cp -r pkg ../demo/

# 5. Start a local HTTP server
cd ../demo
python3 -m http.server 8080

# 6. Open in browser
# Visit http://localhost:8080/index.html
```

**Features:**
- **Interactive canvas** - Draw digits with mouse or touch
- **Real-time predictions** - Instant feedback as you draw
- **Client-side inference** - All computation happens in the browser (1-3ms per prediction)
- **No installation** - Just open a webpage to try the model
- **Cross-platform** - Works on desktop and mobile devices
- **Privacy-preserving** - No data leaves your browser

**Browser Requirements:**
- Chrome 61+, Firefox 60+, Safari 11+, or Edge 79+
- WebAssembly and ES6 module support required
- Works on iOS and Android mobile browsers

**Architecture:**
- Pure Rust inference engine compiled to WebAssembly (~30KB)
- No BLAS dependencies (pure Rust matrix operations)
- Optimized for size and speed (opt-level="s", LTO enabled)
- JavaScript integration layer for model loading and UI
- Binary model format compatible with Rust training code

**What's included:**
```
wasm/
├── src/
│   ├── lib.rs              # WASM bindings (MnistClassifier API)
│   ├── model.rs            # Neural network inference
│   ├── layer.rs            # Dense layer implementation
│   ├── matrix_ops.rs       # Pure Rust linear algebra
│   └── activations.rs      # ReLU, softmax, etc.
└── pkg/                    # Build output (WASM + JS)

demo/
├── index.html              # Interactive demo page
├── style.css               # Responsive styling
├── app.js                  # Main application controller
├── wasm_wrapper.js         # WASM lifecycle management
├── model_loader.js         # Binary model parsing
└── mnist_model.bin         # Trained model (3.1MB)
```

**Performance:**
- WASM compilation: ~50-100ms (one-time startup)
- Model loading: ~200-300ms (one-time download)
- Inference: 1-3ms per prediction (200+ FPS capable)
- Total first-load: ~500ms on fast connection

**Deployment:**
The demo is a static site that can be deployed to GitHub Pages, Netlify, Vercel, or any static hosting service. A GitHub Actions workflow is included for automated deployment.

For comprehensive documentation including build instructions, architecture details, browser compatibility, troubleshooting, and deployment guides, see [`docs/wasm_demo.md`](docs/wasm_demo.md).

## Troubleshooting

When something fails to build or run, start by running:

```bash
cargo run --bin preflight
```

It prints platform-specific remediation steps and the recommended `cargo` commands/feature flags.

### Missing OpenBLAS / BLAS link errors

Symptoms:
- Windows: linker errors mentioning `openblas`, `blas`, or missing `.lib` files
- Linux: build failures while compiling OpenBLAS (toolchain issues)

Remedies:
- **Windows (MSVC)**: install OpenBLAS via vcpkg and ensure `VCPKG_ROOT` is set:

  ```powershell
  git clone https://github.com/microsoft/vcpkg
  .\vcpkg\bootstrap-vcpkg.bat
  .\vcpkg\vcpkg.exe install openblas:x64-windows
  $env:VCPKG_ROOT = (Resolve-Path .\vcpkg)
  ```

- **Linux**: ensure you have a working C toolchain available (`gcc`/`clang`, `make`, etc.). Re-run the build; `openblas-src` builds OpenBLAS from source by default.
- **macOS**: BLAS uses Accelerate via `blas-src` (no OpenBLAS install needed). If you see linker/toolchain failures, install Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

### Missing Xcode Command Line Tools (macOS)

Symptoms:
- `xcode-select: error: tool 'xcodebuild' requires Xcode`
- compilation/link failures on macOS

Remedy:

```bash
xcode-select --install
```

After installing, re-run:

```bash
cargo run --bin preflight
```

### CUDA not found (nvcc / driver)

Symptoms:
- `nvcc: command not found`
- build errors when using `--features gpu-cuda`

Remedies:
- Install the NVIDIA driver + CUDA toolkit appropriate for your OS.
- Verify detection:

  ```bash
  nvcc --version
  nvidia-smi
  ```

If CUDA is not installed, stick to the CPU path:

```bash
cargo test
```

### Metal feature build failures (macOS)

Symptoms:
- errors when building with `--features gpu-metal`

Remedies:
- Ensure Xcode Command Line Tools are installed (see above).
- If you don't need Metal acceleration, use the CPU-only path:

  ```bash
  cargo test
  ```

### wasm-pack / wasm32 target missing

Symptoms:
- `wasm-pack: command not found`
- `error: toolchain ... does not have the target 'wasm32-unknown-unknown'`

Remedies:

```bash
cargo install wasm-pack
rustup target add wasm32-unknown-unknown
```

### Python module not found / scripts fail

Symptoms:
- `ModuleNotFoundError: No module named ...`
- plotting/visualization scripts fail

Remedy (from project root):

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python3 -m pip install -r requirements.txt
```

### Dataset missing (MNIST / CIFAR-10)

Symptoms:
- training/bench binaries fail because files under `./data` are missing

Remedy:

```bash
cargo run --bin dataset-helper -- verify
```

If verification fails, follow the output instructions to download/prepare the datasets.

## Visualization

To plot training curves (including validation metrics):

```bash
python plot_comparison.py
```

## Gradient Visualization

The gradient visualization tool helps understand gradient flow during training and detect vanishing/exploding gradient problems.

**Features:**
- Visualize gradient magnitudes per layer over epochs
- Detect and warn about vanishing gradients (< 1e-5) and exploding gradients (> 100)
- Generate animated visualizations showing gradient evolution
- Statistical analysis of gradient health

**Usage:**

```bash
# Basic usage - creates static plots from gradient logs
python visualize_gradients.py

# Specify which model's gradients to visualize
python visualize_gradients.py --model mlp     # Uses logs/gradients_mlp.csv
python visualize_gradients.py --model cnn     # Uses logs/gradients_cnn.csv

# Create animated visualization
python visualize_gradients.py --animate

# Custom thresholds for gradient detection
python visualize_gradients.py --vanishing-threshold 1e-6 --exploding-threshold 50
```

**Outputs:**
- `gradient_flow.png` - Per-layer gradient magnitude plots
- `gradient_flow_combined.png` - All layers on same axes for comparison
- `gradient_flow_animation.gif` - Animated gradient evolution (with `--animate`)

**Example workflow:**

```bash
# 1. Train a model (generates gradient logs)
cargo run --release --bin mnist_mlp

# 2. Visualize gradients
python visualize_gradients.py --model mlp

# 3. Create animation
python visualize_gradients.py --model mlp --animate
```

The tool automatically detects gradient issues and prints warnings with suggested remediation strategies. For detailed documentation on gradient flow analysis and troubleshooting, see [`docs/gradient_visualization.md`](docs/gradient_visualization.md).

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
