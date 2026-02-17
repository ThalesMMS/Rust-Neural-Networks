//! Profiling binary for the 784→512→10 MNIST-style MLP.
//!
//! Runs a short training loop over synthetic data with per-layer timing
//! instrumentation using [`Profiler`].  At the end of the run it writes:
//!
//! - `./logs/profile_report.html` – self-contained Chart.js HTML report.
//! - `./logs/profile_data.csv` – raw per-layer, per-epoch CSV.
//!
//! No real MNIST files are required; the binary generates synthetic inputs
//! via [`SimpleRng`] so it can run in any environment.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin profile_training
//! ```

use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::profiling::{LayerProfile, Profiler};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::rng::SimpleRng;
use std::process;
use std::time::Instant;

// ============================================================================
// Architecture constants
// ============================================================================

const NUM_INPUTS: usize = 784;
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;

// Number of synthetic samples and training epochs.
const NUM_SAMPLES: usize = 1_024;
const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.01;

// ============================================================================
// Synthetic data helpers
// ============================================================================

/// Generate `n` random f32 values in `[0.0, 1.0)`.
fn random_f32_vec(n: usize, rng: &mut SimpleRng) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

/// Generate `n` random class labels in `0..NUM_OUTPUTS`.
fn random_labels(n: usize, rng: &mut SimpleRng) -> Vec<u8> {
    (0..n)
        .map(|_| (rng.next_f32() * NUM_OUTPUTS as f32) as u8 % NUM_OUTPUTS as u8)
        .collect()
}

// ============================================================================
// Cross-entropy loss gradient (inline – avoids importing training module)
// ============================================================================

/// Compute the softmax cross-entropy gradient in place.
///
/// `probs` must already contain softmax probabilities (batch_size × NUM_OUTPUTS).
/// Subtracts one from the true class probability for each sample and scales by
/// `1 / batch_size`.  Returns the mean cross-entropy loss for the batch.
fn ce_grad_inplace(probs: &mut [f32], labels: &[u8], batch_size: usize) -> f32 {
    let mut loss = 0.0f32;
    let inv_n = 1.0 / batch_size as f32;
    for i in 0..batch_size {
        let label = labels[i] as usize;
        let row_start = i * NUM_OUTPUTS;
        let p = probs[row_start + label].max(1e-9);
        loss -= p.ln();
        probs[row_start + label] -= 1.0;
        for j in 0..NUM_OUTPUTS {
            probs[row_start + j] *= inv_n;
        }
    }
    loss / batch_size as f32
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // ---- Ensure logs directory exists ----
    if let Err(e) = std::fs::create_dir_all("./logs") {
        eprintln!("Warning: could not create ./logs directory: {}", e);
    }

    println!("=== profile_training ===");
    println!(
        "Architecture : {}→{}→{}",
        NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS
    );
    println!("Synthetic samples : {}", NUM_SAMPLES);
    println!("Batch size        : {}", BATCH_SIZE);
    println!("Epochs            : {}", NUM_EPOCHS);
    println!();

    // ---- Initialise layers ----
    let mut rng = SimpleRng::new(42);
    let mut hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, &mut rng);
    let mut output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, &mut rng);

    // ---- Generate synthetic training data ----
    let train_images = random_f32_vec(NUM_SAMPLES * NUM_INPUTS, &mut rng);
    let train_labels = random_labels(NUM_SAMPLES, &mut rng);

    // ---- Profiler ----
    let mut profiler = Profiler::new();

    // Pre-allocate batch buffers (reused each iteration).
    let mut batch_input = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut unused_grad = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];

    let layer_names = [
        format!("Dense {}→{}", NUM_INPUTS, NUM_HIDDEN),
        format!("Dense {}→{}", NUM_HIDDEN, NUM_OUTPUTS),
    ];

    for epoch in 1..=NUM_EPOCHS {
        profiler.start_epoch(epoch);

        // Accumulators for LayerProfile fields (summed across all batches).
        let mut hidden_fwd_ms = 0.0f64;
        let mut hidden_bwd_ms = 0.0f64;
        let mut hidden_upd_ms = 0.0f64;
        let mut output_fwd_ms = 0.0f64;
        let mut output_bwd_ms = 0.0f64;
        let mut output_upd_ms = 0.0f64;

        let mut epoch_loss = 0.0f32;
        let epoch_start = Instant::now();

        for batch_start in (0..NUM_SAMPLES).step_by(BATCH_SIZE) {
            let batch_count = (NUM_SAMPLES - batch_start).min(BATCH_SIZE);

            // Copy mini-batch inputs.
            let src_start = batch_start * NUM_INPUTS;
            let src_end = src_start + batch_count * NUM_INPUTS;
            batch_input[..batch_count * NUM_INPUTS]
                .copy_from_slice(&train_images[src_start..src_end]);

            // ---- Forward: hidden layer (timed) ----
            let t0 = Instant::now();
            hidden_layer.forward(&batch_input, &mut a1, batch_count);
            hidden_fwd_ms += t0.elapsed().as_secs_f64() * 1_000.0;
            relu_inplace(&mut a1[..batch_count * NUM_HIDDEN]);

            // ---- Forward: output layer (timed) ----
            let t0 = Instant::now();
            output_layer.forward(&a1, &mut a2, batch_count);
            output_fwd_ms += t0.elapsed().as_secs_f64() * 1_000.0;
            softmax_rows(
                &mut a2[..batch_count * NUM_OUTPUTS],
                batch_count,
                NUM_OUTPUTS,
            );

            // ---- Loss gradient ----
            dz2[..batch_count * NUM_OUTPUTS].copy_from_slice(&a2[..batch_count * NUM_OUTPUTS]);
            let batch_loss = ce_grad_inplace(
                &mut dz2[..batch_count * NUM_OUTPUTS],
                &train_labels[batch_start..batch_start + batch_count],
                batch_count,
            );
            epoch_loss += batch_loss;

            // ---- Backward: output layer (timed) ----
            let t0 = Instant::now();
            output_layer.backward(&a1, &dz2, &mut dz1, batch_count);
            output_bwd_ms += t0.elapsed().as_secs_f64() * 1_000.0;

            // ReLU derivative applied to hidden gradient.
            for i in 0..batch_count * NUM_HIDDEN {
                if a1[i] <= 0.0 {
                    dz1[i] = 0.0;
                }
            }

            // ---- Backward: hidden layer (timed) ----
            let t0 = Instant::now();
            hidden_layer.backward(
                &batch_input,
                &dz1,
                &mut unused_grad[..batch_count * NUM_INPUTS],
                batch_count,
            );
            hidden_bwd_ms += t0.elapsed().as_secs_f64() * 1_000.0;

            // ---- Parameter updates (timed) ----
            let t0 = Instant::now();
            output_layer.update_parameters(LEARNING_RATE);
            output_upd_ms += t0.elapsed().as_secs_f64() * 1_000.0;

            let t0 = Instant::now();
            hidden_layer.update_parameters(LEARNING_RATE);
            hidden_upd_ms += t0.elapsed().as_secs_f64() * 1_000.0;
        }

        let epoch_elapsed = epoch_start.elapsed().as_secs_f64();
        let num_batches = (NUM_SAMPLES as f64 / BATCH_SIZE as f64).ceil() as usize;
        let avg_loss = epoch_loss / num_batches as f32;

        println!(
            "Epoch {}/{}, Loss: {:.6}, Time: {:.3}s",
            epoch, NUM_EPOCHS, avg_loss, epoch_elapsed
        );

        // ---- Record per-layer profiles ----
        profiler.record_layer(LayerProfile {
            layer_name: layer_names[0].clone(),
            forward_ms: hidden_fwd_ms,
            backward_ms: hidden_bwd_ms,
            update_ms: hidden_upd_ms,
            flops_forward: hidden_layer.flops_forward(NUM_SAMPLES),
            flops_backward: hidden_layer.flops_backward(NUM_SAMPLES),
            param_memory_bytes: hidden_layer.parameter_memory_bytes(),
        });

        profiler.record_layer(LayerProfile {
            layer_name: layer_names[1].clone(),
            forward_ms: output_fwd_ms,
            backward_ms: output_bwd_ms,
            update_ms: output_upd_ms,
            flops_forward: output_layer.flops_forward(NUM_SAMPLES),
            flops_backward: output_layer.flops_backward(NUM_SAMPLES),
            param_memory_bytes: output_layer.parameter_memory_bytes(),
        });

        profiler.end_epoch();
    }

    // ---- Print summary table ----
    println!();
    println!("=== Profiling Summary ===");
    profiler.print_summary();

    // ---- Export CSV ----
    let csv_path = "./logs/profile_data.csv";
    if let Err(e) = profiler.save_csv(csv_path) {
        eprintln!("Warning: failed to save CSV to '{}': {}", csv_path, e);
    } else {
        println!("Profiling CSV saved to '{}'", csv_path);
    }

    // ---- Export HTML report ----
    let html_path = "./logs/profile_report.html";
    if let Err(e) = profiler.generate_html_report(html_path) {
        eprintln!(
            "Warning: failed to generate HTML report '{}': {}",
            html_path, e
        );
        process::exit(1);
    } else {
        println!("Profiling HTML report saved to '{}'", html_path);
    }

    println!();
    println!(
        "Done. Open '{}' in a browser to view the dashboard.",
        html_path
    );
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_f32_vec_length() {
        let mut rng = SimpleRng::new(1);
        let v = random_f32_vec(100, &mut rng);
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_random_labels_range() {
        let mut rng = SimpleRng::new(2);
        let labels = random_labels(500, &mut rng);
        assert_eq!(labels.len(), 500);
        assert!(labels.iter().all(|&l| (l as usize) < NUM_OUTPUTS));
    }

    #[test]
    fn test_ce_grad_inplace_basic() {
        // One-sample batch with NUM_OUTPUTS=10 classes; true label is class 0.
        // Construct a uniform "softmax" output: each probability = 0.1.
        let mut probs = vec![0.1f32; NUM_OUTPUTS];
        let labels = vec![0u8];
        let loss = ce_grad_inplace(&mut probs, &labels, 1);
        // Loss should be -ln(0.1)
        assert!((loss - (-(0.1f32.ln()))).abs() < 1e-5);
        // Gradient for true class: (0.1 - 1.0) * (1/1) = -0.9
        assert!((probs[0] - (-0.9f32)).abs() < 1e-5);
        // Gradient for other classes: 0.1 * (1/1) = 0.1 (unchanged)
        assert!((probs[1] - 0.1f32).abs() < 1e-5);
    }

    #[test]
    fn test_profile_training_layers_initialise() {
        let mut rng = SimpleRng::new(42);
        let hidden = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, &mut rng);
        let output = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, &mut rng);
        assert_eq!(hidden.input_size(), NUM_INPUTS);
        assert_eq!(hidden.output_size(), NUM_HIDDEN);
        assert_eq!(output.input_size(), NUM_HIDDEN);
        assert_eq!(output.output_size(), NUM_OUTPUTS);
    }
}
