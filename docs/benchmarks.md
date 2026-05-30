# Benchmarks: expected ranges and reproducibility

This repository contains lightweight **performance benchmarks** for common model components and architectures.

The benchmarks are intended to:

- provide **rough performance expectations** on CPU,
- help catch **major regressions** (optionally), and
- be **reproducible** with documented assumptions.

They are **not** intended to be a precise cross-machine leaderboard.

## Benchmark entrypoints

This repo uses both:

1. **Helper binary** (recommended)

   ```bash
   cargo run --release --bin run_benchmarks
   ```

   - Prints benchmark results as JSON to stdout.
   - Can optionally save an artifact file under `benchmarks/results/`.
   - Can optionally check for regressions against `benchmarks/baseline.json`.

2. **Criterion benches** under `benches/`

   ```bash
   cargo bench
   ```

   - Writes criterion reports under `target/criterion/`.
   - Useful for local profiling, but not used as the primary “documented numbers”.

## Where outputs go (artifacts policy)

- `run_benchmarks -- --save-results` writes JSON to: `benchmarks/results/` (generated; do not commit)
- `cargo bench` writes reports to: `target/criterion/` (generated; do not commit)

See also: [`docs/artifacts.md`](./artifacts.md).

## Baseline environment (CPU-only)

The current baseline numbers were captured on:

- OS: macOS 26.5 (Darwin 25.5.0)
- Hardware: Apple M4 (10 cores), 16GB RAM
- Toolchain: rustc 1.95.0 / cargo 1.95.0 (Homebrew)
- Mode: `--release`
- Features: default (CPU-only)

Your results will vary with:

- CPU model and thermal state
- OS scheduler/load
- Rust compiler version
- `--release` vs `--debug`
- backend features (e.g., Metal/CUDA)

## Expected variability and “acceptable ranges”

### What to expect

- Across machines: **very large variance** is normal (often multiple ×).
- On the same machine: modest variance is normal between runs.

### Regression checking

If you want a coarse automated check (best effort), the benchmark runner supports comparing the output to a tracked baseline threshold file:

```bash
cargo run --release --bin run_benchmarks -- --check-regression
```

The file `benchmarks/baseline.json` is intentionally lenient and encodes wide limits:

- maximum acceptable latency (ms)
- minimum acceptable throughput (samples/s)

This is meant to detect **catastrophic slowdowns** (e.g., an accidental O(n²) change), not small performance changes.

## How to reproduce the baseline run

From the repository root:

```bash
# Optional: check your environment first
cargo run --bin preflight

# Run and print results JSON
cargo run --release --bin run_benchmarks

# Save a timestamped JSON file under benchmarks/results/
cargo run --release --bin run_benchmarks -- --save-results

# Compare against the tracked baseline thresholds
cargo run --release --bin run_benchmarks -- --check-regression
```

## Interpreting the output

The `run_benchmarks` JSON output reports, per architecture and batch size, values such as:

- `mean_ms` latency per operation (forward / training steps depending on benchmark)
- `samples_per_second` throughput

When comparing runs:

- Compare the **same** build mode (`--release`) and batch size.
- Treat minor changes as noise; focus on order-of-magnitude changes.

## Optional GPU backends

If you enable GPU backends (e.g., Metal/CUDA), expect results to differ dramatically.

At the moment, the repo’s tracked `benchmarks/baseline.json` is intended as a **CPU-only** sanity bound.
