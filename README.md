# Rust Neural Network Models

Authors: Antonio Neto and Thales Matheus

## Overview

This repository contains small neural networks in Rust for:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- CIFAR-10 object classification (CNN)
- XOR toy example (2->4->1)

Python utilities are included for visualization and digit recognition. The Swift implementation lives in the companion Swift-Neural-Networks repository. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Repository layout

Rust source:

- `mnist_mlp.rs`, `mnist_cnn.rs`, `mnist_attention_pool.rs`, `cifar10_cnn.rs`, `mlp_simple.rs` (standalone binaries)
- `src/` (shared layers, optimizers, utils, config)
- `tests/` (integration tests)
- `Cargo.toml` / `Cargo.lock`

Configs:

- `config/training/` (training hyperparameters for all models)
- `config/architectures/` (network architecture definitions)
- `config/` (learning-rate scheduler configs, activation configs)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plot training/validation curves from `logs/`)
- `visualize_attention.py` (attention visualization utility)
- `visualize_gradients.py` (gradient flow visualization and analysis)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files, CIFAR-10 binary files)
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

### CIFAR-10 CNN

Architecture:

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

Note: CIFAR-10 is significantly harder than MNIST. The baseline CNN architecture is intentionally simple for educational purposes. State-of-the-art models typically achieve 90%+ accuracy with deeper architectures, data augmentation, and more training.

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
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu"
}
```

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

For comprehensive documentation including all hyperparameters, validation rules, scheduler types, and experimentation guide, see [`docs/hyperparameters.md`](docs/hyperparameters.md).

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
| CIFAR-10 CNN | Rust | `cargo run --release --bin cifar10_cnn` | 10 | 32 | TBD | TBD (expected 50-60) | Conv16/3x3 + MaxPool, RGB input |
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

**Quick Start:**

```bash
# 1. Build the WASM module (one-time setup)
cd wasm
wasm-pack build --target web --release

# 2. Copy WASM package to demo directory
cp -r pkg ../demo/

# 3. Start a local HTTP server
cd ../demo
python3 -m http.server 8080

# 4. Open in browser
# Visit http://localhost:8080/index.html
```

**Features:**
- ✅ **Interactive canvas** - Draw digits with mouse or touch
- ✅ **Real-time predictions** - Instant feedback as you draw
- ✅ **Client-side inference** - All computation happens in the browser (1-3ms per prediction)
- ✅ **No installation** - Just open a webpage to try the model
- ✅ **Cross-platform** - Works on desktop and mobile devices
- ✅ **Privacy-preserving** - No data leaves your browser

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
