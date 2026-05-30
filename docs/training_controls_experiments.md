# Training Controls Experiments (Config-Driven)

This document provides **two small, reproducible experiments** demonstrating the effect of config-driven training controls on training behavior.

The goal is not to produce state-of-the-art accuracy; it is to show visible changes in training curves/metrics when toggling individual controls.

## How to run

Each experiment has a baseline config and a modified config. Run the same binary twice and compare the printed epoch metrics.

Example command:

```bash
cargo run --release --bin mnist_mlp -- --config config/training/mnist_mlp_experiment_baseline.json
cargo run --release --bin mnist_mlp -- --config config/training/mnist_mlp_experiment_warmup_l2_clip.json
```

Collect the console output and plot `train_loss` / `val_loss` versus epoch (or simply compare the per-epoch values).

---

## Experiment 1 — MNIST MLP: baseline vs warmup + L2 + norm clipping

### Baseline
- Expected behavior: faster initial loss drop, but potentially more oscillation/instability early on.

Config: `config/training/mnist_mlp_experiment_baseline.json`

### With warmup + L2 + clipping
- Changes:
  - `warmup` (linear, 3 epochs)
  - `regularization.l2_lambda = 0.0001`
  - `gradient_clipping` (norm, max_norm=1.0)
- Expected behavior:
  - smoother early training (warmup)
  - reduced overfitting / slightly improved validation stability (L2)
  - fewer large loss spikes if gradients explode (clipping)

Config: `config/training/mnist_mlp_experiment_warmup_l2_clip.json`

**What to compare:**
- training loss slope for epochs 1–3 (warmup reduces early step size)
- validation loss trend near the end of training (L2 tends to reduce overfitting)
- presence/absence of sudden loss spikes (clipping reduces spikes)

---

## Experiment 2 — CIFAR-10 CNN: constant LR baseline vs cyclical LR

### Baseline (constant LR)
- Expected behavior: steady improvement; may plateau if LR is not ideal throughout training.

Config: `config/training/cifar10_cnn_experiment_baseline.json`

### With cyclical learning rate
- Changes:
  - enable `cyclical_lr` (triangular)
  - set `scheduler_type` to `constant` (required when `cyclical_lr` is used)
- Expected behavior:
  - more pronounced oscillations in loss (as LR cycles)
  - potential improvements in escaping plateaus (depends on run)

Config: `config/training/cifar10_cnn_experiment_cyclical_lr.json`

**What to compare:**
- oscillatory loss pattern (cyclical LR)
- whether validation accuracy continues to improve vs plateauing

---

## Notes
- For best comparison, keep everything except the control being tested identical.
- Run-to-run variance exists; if curves look too similar, increase `epochs` or adjust LR ranges.
