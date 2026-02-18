//! Interactive step-through debugging for neural network training.
//!
//! This module provides a [`StepDebugger`] that lets students and educators
//! observe backpropagation in action by stepping through forward pass, loss
//! computation, backward pass, and parameter updates with educational
//! explanations and tensor statistics at each stage.
//!
//! # Overview
//!
//! - [`TensorStats`] — computes shape, mean, std, min, max, zero%, and NaN
//!   count from an `&[f32]` slice.
//! - [`StepMode`] — tracks the user's interaction preference (step-by-step,
//!   continue to end of epoch, or continue all).
//! - [`StepDebugger`] — the main struct. All methods are no-ops when
//!   `enabled=false`, ensuring zero cost in normal training.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::step_debug::StepDebugger;
//!
//! // Disabled debugger: all calls are no-ops.
//! let mut dbg = StepDebugger::new(false);
//! dbg.on_epoch_start(1);
//! ```

use std::fmt;
use std::io::{self, BufRead, Write};

// ─────────────────────────────────────────────────────────────────────────────
// TensorStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a tensor (flat `&[f32]` slice).
///
/// Computed in a single O(n) pass with no allocations.
#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    /// Number of elements in the tensor.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Fraction of elements that are exactly 0.0.
    pub zero_frac: f32,
    /// Number of NaN values.
    pub nan_count: usize,
}

impl TensorStats {
    /// Computes summary statistics from a flat slice.
    ///
    /// Returns `None` if the slice is empty.
    pub fn compute(data: &[f32]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let count = data.len();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut zeros = 0usize;
        let mut nans = 0usize;

        for &v in data {
            if v.is_nan() {
                nans += 1;
                continue;
            }
            let vd = v as f64;
            sum += vd;
            sum_sq += vd * vd;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            if v == 0.0 {
                zeros += 1;
            }
        }

        let valid = count - nans;
        let mean = if valid > 0 {
            (sum / valid as f64) as f32
        } else {
            f32::NAN
        };
        let variance = if valid > 0 {
            (sum_sq / valid as f64 - (sum / valid as f64).powi(2)).max(0.0)
        } else {
            0.0
        };
        let std = variance.sqrt() as f32;

        // Handle the edge case where all values are NaN
        if nans == count {
            min = f32::NAN;
            max = f32::NAN;
        }

        Some(TensorStats {
            count,
            mean,
            std,
            min,
            max,
            zero_frac: zeros as f32 / count as f32,
            nan_count: nans,
        })
    }
}

impl fmt::Display for TensorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean={:.3}, std={:.3}, min={:.3}, max={:.3}, zeros={:.1}%",
            self.mean,
            self.std,
            self.min,
            self.max,
            self.zero_frac * 100.0,
        )?;
        if self.nan_count > 0 {
            write!(f, ", NaNs={}", self.nan_count)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StepMode
// ─────────────────────────────────────────────────────────────────────────────

/// The user's current interaction preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepMode {
    /// Pause after every step for user input.
    StepByStep,
    /// Run to the end of the current epoch without pausing.
    ContinueEpoch,
    /// Run to the end of training without pausing.
    ContinueAll,
}

// ─────────────────────────────────────────────────────────────────────────────
// StepDebugger
// ─────────────────────────────────────────────────────────────────────────────

/// Interactive step-through debugger for training loops.
///
/// When `enabled=false`, all methods return immediately — the compiler can
/// eliminate the dead branches entirely.
pub struct StepDebugger {
    enabled: bool,
    mode: StepMode,
    step: usize,
    epoch: usize,
    batch: usize,
}

impl StepDebugger {
    /// Creates a new debugger.
    ///
    /// When `enabled=false`, every method is a no-op.
    pub fn new(enabled: bool) -> Self {
        if enabled {
            println!("\n=== Step-Through Debug Mode ===");
            println!("Controls: Enter=next step, c=continue epoch, a=continue all, q=quit\n");
        }
        StepDebugger {
            enabled,
            mode: StepMode::StepByStep,
            step: 0,
            epoch: 0,
            batch: 0,
        }
    }

    /// Called at the start of each epoch.
    ///
    /// Resets `ContinueEpoch` back to `StepByStep` so the user sees the
    /// first batch of each new epoch in detail.
    pub fn on_epoch_start(&mut self, epoch: usize) {
        if !self.enabled {
            return;
        }
        self.epoch = epoch;
        if self.mode == StepMode::ContinueEpoch {
            self.mode = StepMode::StepByStep;
        }
    }

    /// Sets the display context for the current batch.
    pub fn set_context(
        &mut self,
        epoch: usize,
        batch: usize,
        total_batches: usize,
        batch_size: usize,
    ) {
        if !self.enabled {
            return;
        }
        self.epoch = epoch;
        self.batch = batch;
        self.step = 0;
        if self.mode == StepMode::StepByStep {
            println!(
                "================================================================================\n\
                 EPOCH {}, BATCH {}/{} (batch_size={})\n\
                 ================================================================================",
                epoch, batch, total_batches, batch_size
            );
        }
    }

    /// Called before a layer's forward pass.
    #[allow(clippy::too_many_arguments)]
    pub fn before_forward(
        &mut self,
        layer_name: &str,
        input: &[f32],
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        param_count: usize,
    ) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;
        let input_len = batch_size * in_features;
        let stats = TensorStats::compute(&input[..input_len]);

        println!(
            "\n--- STEP {}: FORWARD PASS -> '{}' ({} -> {}, {} params) ---",
            self.step, layer_name, in_features, out_features, param_count
        );
        println!("  output = input x weights + biases");
        if let Some(s) = stats {
            println!("  Input:  shape=({}, {}), {}", batch_size, in_features, s);
        }
        self.wait_for_input();
    }

    /// Called after a layer's forward pass.
    pub fn after_forward(
        &mut self,
        layer_name: &str,
        output: &[f32],
        batch_size: usize,
        out_features: usize,
    ) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        let output_len = batch_size * out_features;
        let stats = TensorStats::compute(&output[..output_len]);

        if let Some(s) = stats {
            println!(
                "  Output ('{}'): shape=({}, {}), {}",
                layer_name, batch_size, out_features, s
            );
        }
    }

    /// Called after an activation function is applied.
    pub fn after_activation(
        &mut self,
        name: &str,
        data: &[f32],
        batch_size: usize,
        features: usize,
    ) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;

        let data_len = batch_size * features;
        let stats = TensorStats::compute(&data[..data_len]);

        println!("\n--- STEP {}: ACTIVATION '{}' ---", self.step, name);

        // Educational explanation for each activation
        match name {
            "ReLU" => println!(
                "  ReLU(x) = max(0, x) -- zeroes out negative values, adding non-linearity."
            ),
            "Softmax" => {
                println!("  Softmax converts logits to probabilities summing to 1.0 per sample.")
            }
            "Sigmoid" => println!("  Sigmoid(x) = 1/(1+e^-x) -- squashes values to (0, 1) range."),
            "Tanh" => println!(
                "  Tanh(x) = (e^x - e^-x)/(e^x + e^-x) -- squashes values to (-1, 1) range."
            ),
            "LeakyReLU" => {
                println!("  LeakyReLU(x) = x if x>0, alpha*x otherwise -- prevents dead neurons.")
            }
            "GELU" => println!(
                "  GELU(x) = x * Phi(x) -- smooth approximation of ReLU used in transformers."
            ),
            _ => {}
        }

        if let Some(s) = stats {
            println!("  Output: shape=({}, {}), {}", batch_size, features, s);
            if name == "Softmax" {
                println!(
                    "  (mean={:.3} ~ 1/{}, expected for uniform predictions)",
                    s.mean, features
                );
            }
        }
        self.wait_for_input();
    }

    /// Called after loss computation.
    pub fn after_loss(
        &mut self,
        loss: f32,
        gradients: &[f32],
        batch_size: usize,
        num_classes: usize,
    ) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;

        let grad_len = batch_size * num_classes;
        let grad_stats = TensorStats::compute(&gradients[..grad_len]);

        println!("\n--- STEP {}: LOSS (Cross-Entropy) ---", self.step);
        println!("  Batch loss: {:.4}", loss);
        let random_guess = (num_classes as f32).ln();
        println!(
            "  (random guessing = ln({}) = {:.4})",
            num_classes, random_guess
        );
        if let Some(s) = grad_stats {
            println!(
                "  Gradient (dL/d_logits): shape=({}, {}), {}",
                batch_size, num_classes, s
            );
        }
        self.wait_for_input();
    }

    /// Called after a layer's backward pass.
    pub fn after_backward(
        &mut self,
        layer_name: &str,
        grad_output: &[f32],
        grad_input: &[f32],
        batch_size: usize,
        out_features: usize,
        in_features: usize,
    ) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;

        let grad_out_len = batch_size * out_features;
        let grad_in_len = batch_size * in_features;
        let out_stats = TensorStats::compute(&grad_output[..grad_out_len]);
        let in_stats = TensorStats::compute(&grad_input[..grad_in_len]);

        println!(
            "\n--- STEP {}: BACKWARD PASS <- '{}' ---",
            self.step, layer_name
        );
        println!("  grad_input = grad_output x weights^T (propagating error signal backward)");
        if let Some(s) = out_stats {
            println!(
                "  Grad output: shape=({}, {}), {}",
                batch_size, out_features, s
            );
        }
        if let Some(s) = in_stats {
            println!(
                "  Grad input:  shape=({}, {}), {}",
                batch_size, in_features, s
            );
        }
        self.wait_for_input();
    }

    /// Called after applying the ReLU derivative to gradients.
    pub fn after_relu_derivative(&mut self, data: &[f32], batch_size: usize, features: usize) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;

        let data_len = batch_size * features;
        let stats = TensorStats::compute(&data[..data_len]);

        println!("\n--- STEP {}: ACTIVATION DERIVATIVE 'ReLU' ---", self.step);
        println!("  grad[i] = 0 where activation was <= 0 (gradient killed for inactive neurons)");
        if let Some(s) = stats {
            println!("  Output: shape=({}, {}), {}", batch_size, features, s);
        }
        self.wait_for_input();
    }

    /// Called after parameter updates for all layers.
    pub fn after_update(&mut self, layers: &[(&str, f32, f32)], learning_rate: f32) {
        if !self.enabled || self.mode != StepMode::StepByStep {
            return;
        }
        self.step += 1;

        println!("\n--- STEP {}: PARAMETER UPDATE ---", self.step);
        for &(name, w_norm, b_norm) in layers {
            println!(
                "  {}: lr={:.6}, weight_grad_norm={:.4}, bias_grad_norm={:.4}",
                name, learning_rate, w_norm, b_norm
            );
        }
        println!("  Parameters updated: weights -= lr * grad_weights, biases -= lr * grad_biases");
        self.wait_for_input();
    }

    /// Waits for user input when in `StepByStep` mode.
    ///
    /// - Enter → next step
    /// - `c`   → continue to end of epoch
    /// - `a`   → continue all (no more pauses)
    /// - `q`   → quit immediately
    fn wait_for_input(&mut self) {
        if self.mode != StepMode::StepByStep {
            return;
        }
        print!("[Step {}] Enter=next, c=epoch, a=all, q=quit > ", self.step);
        io::stdout().flush().ok();

        let stdin = io::stdin();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            return;
        }
        let trimmed = line.trim();
        match trimmed {
            "c" => self.mode = StepMode::ContinueEpoch,
            "a" => self.mode = StepMode::ContinueAll,
            "q" => {
                println!("Step debugger: quitting.");
                std::process::exit(0);
            }
            _ => {} // Enter or anything else → next step
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_known_values() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::compute(&data).unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 5.0).abs() < 1e-5);
        assert!((stats.zero_frac - 0.0).abs() < 1e-5);
        assert_eq!(stats.nan_count, 0);
        // std of [1,2,3,4,5]: sqrt(2.0)
        let expected_std = 2.0f64.sqrt() as f32;
        assert!((stats.std - expected_std).abs() < 1e-4);
    }

    #[test]
    fn test_tensor_stats_zeros() {
        let data = [0.0f32, 0.0, 0.0, 1.0];
        let stats = TensorStats::compute(&data).unwrap();
        assert!((stats.zero_frac - 0.75).abs() < 1e-5);
        assert!((stats.mean - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_nans() {
        let data = [1.0f32, f32::NAN, 3.0];
        let stats = TensorStats::compute(&data).unwrap();
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.count, 3);
        // mean of valid values: (1+3)/2 = 2.0
        assert!((stats.mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_single_element() {
        let data = [42.0f32];
        let stats = TensorStats::compute(&data).unwrap();
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 42.0).abs() < 1e-5);
        assert!((stats.std - 0.0).abs() < 1e-5);
        assert!((stats.min - 42.0).abs() < 1e-5);
        assert!((stats.max - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_empty() {
        let data: [f32; 0] = [];
        assert!(TensorStats::compute(&data).is_none());
    }

    #[test]
    fn test_tensor_stats_display() {
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let stats = TensorStats::compute(&data).unwrap();
        let display = format!("{}", stats);
        assert!(display.contains("mean="));
        assert!(display.contains("std="));
        assert!(display.contains("min="));
        assert!(display.contains("max="));
        assert!(display.contains("zeros="));
        // No NaNs, so NaN field should not appear
        assert!(!display.contains("NaN"));
    }

    #[test]
    fn test_tensor_stats_display_with_nans() {
        let data = [1.0f32, f32::NAN];
        let stats = TensorStats::compute(&data).unwrap();
        let display = format!("{}", stats);
        assert!(display.contains("NaNs=1"));
    }

    #[test]
    fn test_debugger_disabled_is_noop() {
        // Just ensure no panics when calling methods on a disabled debugger.
        let mut dbg = StepDebugger::new(false);
        dbg.on_epoch_start(1);
        dbg.set_context(1, 1, 10, 64);
        dbg.before_forward("test", &[0.0; 4], 1, 4, 2, 10);
        dbg.after_forward("test", &[0.0; 2], 1, 2);
        dbg.after_activation("ReLU", &[0.0; 2], 1, 2);
        dbg.after_loss(2.3, &[0.0; 2], 1, 2);
        dbg.after_backward("test", &[0.0; 2], &[0.0; 4], 1, 2, 4);
        dbg.after_relu_derivative(&[0.0; 2], 1, 2);
        dbg.after_update(&[("test", 0.01, 0.005)], 0.01);
    }
}
