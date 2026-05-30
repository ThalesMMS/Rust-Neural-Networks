# Repository artifacts policy

This repository contains both **curated assets** (kept in git) and **generated artifacts** (outputs of training/build/benchmarks that should not be committed).

This document defines the standard locations for artifacts and what belongs in source control.

## Standard directories (relative to repo root)

This document covers all artifact types used in this repository:

- datasets: `data/`
- logs: `logs/`
- runs/experiment registry: `runs/`
- model outputs: `artifacts/models/`
- plots/figures: `artifacts/plots/`
- benchmark output: `benchmarks/results/` (and `target/criterion/` for criterion reports)
- WASM build output: `wasm/pkg/`, `wasm/target/`

### Datasets (curated)

- `data/`
  - Purpose: curated datasets required to run examples locally.
  - Status: **tracked** (keep small and well-scoped).

Notes:
- CIFAR-10 binaries live under `data/cifar-10-batches-bin/`.

### Training logs and metrics (generated)

- `logs/`
  - Purpose: CSV/JSON training metrics, sweep summaries, debug output.
  - Status: **generated** (do not commit).

### Experiment registry (generated)

- `runs/`
  - Purpose: structured run records (`run.json`) and per-run artifacts.
  - Typical structure:

    ```text
    runs/<run_id>/
      run.json
      artifacts/
    ```

  - Status: **generated** (do not commit).

### Model outputs / checkpoints (generated)

- `artifacts/models/`
  - Purpose: model checkpoints and other exported model binaries.
  - Suggested patterns:
    - `*.bin` (repository-specific)
    - `*.onnx`, `*.pt`, `*.safetensors` (if added later)
  - Status: **generated** (do not commit).

Important:
- The repo contains a small number of **curated model binaries** used by demos/documentation. Those are exceptions and remain tracked (see “Curated assets (exceptions)” below).

### Plots / figures (mixed)

- `artifacts/plots/`
  - Purpose: generated plots (PNG/SVG/PDF) from training or analysis scripts.
  - Status: **generated** by default (do not commit).

If a figure is referenced by documentation and is intended to be stable, treat it as a curated asset and commit it under `docs/assets/` instead.

### Benchmarks output (generated)

- `benchmarks/results/`
  - Purpose: benchmark output artifacts (CSV/JSON), summaries, comparisons.
  - Status: **generated** (do not commit).

### WASM/demo build outputs (generated)

- `wasm/pkg/`, `wasm/target/`
  - Purpose: `wasm-pack` build output.
  - Status: **generated** (do not commit).

If you add additional front-end build output (e.g., `dist/`), keep it under the `wasm/` subtree and treat it as generated.

## Curated assets (exceptions)

These files/directories are intentionally kept in git:

- `data/cifar-10-batches-bin/` (dataset used by examples)
- `demo/mnist_model.bin` (demo model)
- `gradient_flow.png`, `gradient_flow_animated.gif`, `gradient_flow_combined.png` (documentation assets)

If you need to add new curated assets, prefer placing them under `docs/assets/` and keep them small.

## Reproducibility: demos and benchmarks

### Demos

- Most demos can be re-built and re-run from source.
- Generated outputs should be written to the standard artifact directories:
  - logs/metrics: `logs/`
  - plots/figures: `artifacts/plots/`
  - model checkpoints/exports: `artifacts/models/`

Curated exception:
- `demo/mnist_model.bin` is a **curated** pre-trained model that ships with the repository for the demo.
  - If you retrain, write your new checkpoint to `artifacts/models/` (do not overwrite the curated file).

### Benchmarks

This repository uses both:

- **Criterion benches** under `benches/` (run with `cargo bench`), and
- a helper binary `run_benchmarks` (run with `cargo run --release --bin run_benchmarks`).

Where results go:

- `cargo bench` (criterion) writes its reports under `target/criterion/` (generated; do not commit).
- `run_benchmarks -- --save-results` writes JSON results under `benchmarks/results/` (generated; do not commit).
- Regression checking compares against `benchmarks/baseline.json` (tracked baseline).

### Large files guidance

- Avoid committing large generated outputs (models, logs, plots, benchmark dumps).
- For sharing results/models:
  - attach artifacts to GitHub Releases, or
  - use external storage (S3/GCS/Drive) and link to it from docs.

## Contributor commit guidance

### What not to commit

Do **not** commit generated artifacts, including:

- training output:
  - `logs/`
  - `runs/`
  - `artifacts/models/`
  - `artifacts/plots/`
- benchmark output:
  - `benchmarks/results/`
  - `target/criterion/` (criterion reports)
- build products:
  - `target/` (Rust build output)
  - `wasm/pkg/`, `wasm/target/` (wasm-pack output)

If you need to share a model checkpoint or large benchmark result, attach it to a GitHub Release or upload it to external storage and link it from documentation.

### How to check before you commit

1. Review what will be committed:

   ```bash
   git status
   git diff --stat
   ```

2. Look for large files in your working tree (example: files > 10MB):

   ```bash
   find . -type f -size +10M -not -path './.git/*' -print
   ```

3. If you already staged something by accident, unstage it:

   ```bash
   git restore --staged <path>
   ```

### Curated exceptions

A small set of curated assets is intentionally tracked (see “Curated assets (exceptions)”). If you add a new curated figure or small model needed for docs, place it under `docs/assets/` and keep it small.

## Cleanup tooling

This repository provides a cleanup helper that removes only **generated** artifact directories defined in this policy.

- Script: `scripts/clean_artifacts.sh`
- Default behavior: **dry-run** (prints what would be removed)

### What it deletes

When run with `--all`, the script deletes these directories if present:

- `logs/`
- `runs/`
- `artifacts/` (including `artifacts/models/` and `artifacts/plots/`)
- `benchmarks/results/`
- `wasm/pkg/`
- `wasm/target/`

It is designed to avoid curated assets such as `data/`, `demo/mnist_model.bin`, and documentation figures.

### How to run

From the repository root:

```bash
# See what would be removed
scripts/clean_artifacts.sh

# Same as above
scripts/clean_artifacts.sh --dry-run

# Actually remove generated artifacts
scripts/clean_artifacts.sh --all
```

### How to recover/regenerate

- Logs/plots/models: re-run your training/demo scripts; outputs should be written to `logs/` and `artifacts/`.
- Benchmarks:
  - Criterion reports: re-run `cargo bench` (outputs under `target/criterion/`).
  - Saved benchmark results: re-run `cargo run --release --bin run_benchmarks -- --save-results` (outputs under `benchmarks/results/`).
- WASM build output: re-run your `wasm-pack` build (outputs under `wasm/pkg/` and `wasm/target/`).

## Summary

- Tracked: source code, configs, small curated datasets/assets.
- Not tracked: anything reproducible from code (training outputs, logs, benchmarks, WASM build products).
