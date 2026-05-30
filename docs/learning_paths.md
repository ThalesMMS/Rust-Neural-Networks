# Curated learning paths

## Contributing: adding a tutorial + reproducibility check

This repo treats tutorials as *docs* plus an optional *smoke check* that proves the tutorial still runs.

### 1) Add (or update) the tutorial doc

- Put new tutorial pages under `docs/tutorials/`.
- Follow the existing tutorial header pattern (Prerequisites / Estimated runtime / Commands / Expected output).
- Link it from:
  - `docs/tutorials/README.md` (the full index)
  - and, if it belongs in a tier, `docs/learning_paths.md`.

### 2) Add a reproducibility check to the manifest

Tutorial smoke checks are defined in `tutorial_repro_manifest.json` at the repo root.

Each check has:

- `id`: unique within the path (use kebab-case)
- `command`: argv array (no shell), e.g. `["cargo","run","--release","--bin","xor_mlp"]`
- `workdir`: working directory relative to repo root (usually `"."`)
- `timeout_seconds`: keep CI-friendly by default; make long-running checks opt-in
- output expectation (at least one):
  - `stdout_contains`: list of substrings that must appear in stdout/stderr
- optional `require`: list of capabilities needed for the check:
  - `"gpu"` → skipped unless `--include-gpu` is passed
  - `"wasm"` → skipped unless `--include-wasm` is passed

Guidelines:

- Prefer small, deterministic commands (1–60s) for default CPU checks.
- If a tutorial needs a dataset, include (or reference) the project’s dataset preflight step in the *path*, and keep the check itself minimal.
- If the tutorial produces files, write them into the runner-provided artifacts directory via the `TUTORIAL_REPRO_ARTIFACTS_DIR` environment variable.

### 3) Verify locally

List available paths:

```bash
cargo run --bin tutorial_repro -- list
```

Dry-run a path to confirm commands/workdirs/timeouts:

```bash
cargo run --bin tutorial_repro -- run --path beginner --dry-run
```

Run it for real:

```bash
cargo run --bin tutorial_repro -- run --path beginner
```

If you added GPU/WASM checks, verify they skip by default and run when opted in:

```bash
cargo run --bin tutorial_repro -- run --path advanced          # should SKIP gpu/wasm checks
cargo run --bin tutorial_repro -- run --path advanced --include-gpu
cargo run --bin tutorial_repro -- run --path advanced --include-wasm
```

### 4) Keep docs and checks in sync

Whenever you update a tutorial command in a doc, update the corresponding manifest check (or vice versa). The goal is that a learner can copy/paste the documented commands and the runner still passes.

This repository includes a set of tutorials and deep-dive docs. The paths below give you a **recommended order** (Beginner → Intermediate → Advanced), with a small number of runnable commands per step.

> Goal: make it easy for learners/educators to follow a coherent sequence, and for contributors to keep the tutorials reproducible.

---

## Before you start (recommended)

### 1) Preflight your environment

Run the repo preflight check once after cloning:

```bash
cargo run --bin preflight
```

### 2) Datasets (MNIST / CIFAR-10)

Some tutorials expect datasets to be present. If a command fails due to missing data, see:

- `docs/cifar10_dataset.md`
- Tutorial notes in `docs/tutorials/02_mnist_mlp.md` and `docs/tutorials/03_mnist_cnn.md`

---

## Beginner path (60–120 min)

Focus: fundamentals, training loop, saving/loading, and getting a first end-to-end result.

1. **XOR with an MLP (fundamentals)**
   - Read: `docs/tutorials/01_xor_mlp.md`
   - Run:
     ```bash
     cargo run --bin mlp_simple
     ```

2. **MNIST with an MLP (first real dataset)**
   - Read: `docs/tutorials/02_mnist_mlp.md`
   - Run:
     ```bash
     cargo run --bin mnist_mlp
     ```

---

## Intermediate path (2–4 hours)

Focus: CNNs, feature extraction, and interpretability/visualization.

1. **MNIST CNN**
   - Read: `docs/tutorials/03_mnist_cnn.md`
   - Run:
     ```bash
     cargo run --bin mnist_cnn
     ```

2. **Autograd engine (how gradients are computed)**
   - Read: `docs/tutorials/05_autograd_engine.md`
   - Run (smoke check):
     ```bash
     cargo test
     ```

3. **Gradient visualization**
   - Read: `docs/gradient_visualization.md`
   - Run (if you have the required deps installed per the doc):
     ```bash
     cargo run --bin gradient_visualization
     ```

---

## Advanced path (4–8+ hours)

Focus: sequence models, transformers, larger vision datasets, and deployment.

1. **Char-level RNN/LSTM**
   - Read: `docs/tutorials/04_rnn_lstm_char_level.md`
   - Run:
     ```bash
     cargo run --bin rnn_lstm_char
     ```

2. **Autoencoder**
   - Read: `docs/tutorials/06_autoencoder.md`
   - Run:
     ```bash
     cargo run --bin autoencoder
     ```

3. **Vision Transformer (ViT)**
   - Read: `docs/tutorials/07_vision_transformer.md`
   - Run:
     ```bash
     cargo run --bin vision_transformer
     ```

4. **WASM demo (deployment)**
   - Read: `docs/wasm_demo.md`
   - Run from the WASM package directory (see doc for exact steps):
     ```bash
     # example (follow docs/wasm_demo.md)
     wasm-pack build
     ```

---

## Optional branches (pick as needed)

These don’t have to be completed in order, and some may require extra setup/time.

- **Quantization**: `docs/tutorials/quantization.md`
- **GAN**: `docs/gan_tutorial.md`
- **ResNet**: `docs/resnet_tutorial.md`
- **Data augmentation**: `docs/data_augmentation_tutorial.md`
- **Training controls & experiments**: `docs/training_controls_experiments.md`

---

## Reproducibility checks

A reproducibility runner will execute a small set of smoke checks per path (CPU-only by default) to ensure tutorials keep working.

For now, use the per-step commands above, starting with `cargo run --bin preflight`.
