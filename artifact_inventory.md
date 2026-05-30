# Artifact inventory (generated/large outputs)

This file was generated during task **002-repository-artifact-management-policy-and-cleanup** subtask **0.1**.

## Common artifact directories found

- `./wasm/target/` (Rust build output; **~412MB** in this worktree)
- `./demo/pkg/` (WASM/wasm-bindgen output)
- `./data/` (datasets; **~281MB**)
- `./benchmarks/` (benchmark-related directory exists)
- `./config/benchmarks/` (benchmark config)
- `./src/data/` (source directory named data; likely **curated/source**, not generated)

## Common artifact files found (by extension)

### Model / weight binaries
- `./mnist_model.bin`
- `./demo/mnist_model.bin`
- `./transformer_mnist_model.bin`
- `./mnist_cnn_model_best.bin`
- `./mnist_model_best.bin`
- `./mnist_attention_model_best.bin`

### WASM build products
- `./demo/pkg/mnist_wasm_bg.wasm`
- `./wasm/target/wasm32-unknown-unknown/release/mnist_wasm.wasm`
- `./wasm/target/wasm32-unknown-unknown/release/deps/mnist_wasm.wasm`

### Plots/images
- `./gradient_flow.png`
- `./gradient_flow_combined.png`

### Dataset binaries (CIFAR-10)
- `./data/cifar-10-batches-bin/*.bin` (data_batch_1..5, test_batch)

## Largest disk usage hotspots (top-level)

- `./wasm/target/` (~412MB)
- `./data/` (~281MB)

## Notes / initial classification (tentative)

- `wasm/target/` and `demo/pkg/` are build outputs and should be ignored/cleaned.
- `data/` appears to contain raw datasets (MNIST idx files, CIFAR-10 binary batches). These are typically **not** committed and should likely be ignored + documented for download.
- Root `*.bin` model files and `*.png` plots look like generated artifacts (trained weights + training diagnostics) and should likely be ignored/moved into a standard `artifacts/` directory.
- `src/data/` is likely part of the source tree and must **not** be blanket-ignored just because it is named `data`.
