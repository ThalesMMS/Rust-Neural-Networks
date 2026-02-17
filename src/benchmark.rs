//! Automated benchmark utilities for neural network layers.
//!
//! This module provides timing harnesses and result structures for measuring
//! the performance of neural network layer forward passes and training steps.
//! Benchmarks use `std::time::Instant` and include warmup phases to allow
//! JIT/branch-predictor effects to stabilize before measurements are taken.
//!
//! # Example
//!
//! ```ignore
//! use rust_neural_networks::benchmark::{BenchmarkConfig, measure_layer_forward};
//! use rust_neural_networks::layers::DenseLayer;
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! let mut rng = SimpleRng::new(42);
//! let layer = DenseLayer::new(784, 512, &mut rng);
//!
//! let config = BenchmarkConfig::default();
//! let input = vec![0.5f32; 64 * 784]; // batch_size=64
//! let result = measure_layer_forward("MLP Dense", &layer, &input, 64, &config);
//!
//! println!("Mean latency: {:.3} ms", result.forward_stats.mean_ms);
//! println!("Throughput: {:.0} samples/sec", result.forward_stats.samples_per_second);
//! ```

use crate::layers::Layer;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for benchmark runs.
///
/// Controls the number of warmup and measurement iterations as well as
/// the batch sizes to benchmark. Warmup iterations are not timed and allow
/// CPU caches and branch predictors to warm up before measurements begin.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::benchmark::BenchmarkConfig;
///
/// let config = BenchmarkConfig::default();
/// assert_eq!(config.warmup_iterations, 5);
/// assert_eq!(config.measure_iterations, 20);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of iterations to run before timing begins (not measured).
    pub warmup_iterations: usize,
    /// Number of iterations to time for statistics.
    pub measure_iterations: usize,
    /// Batch sizes to use when generating synthetic data.
    pub batch_sizes: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measure_iterations: 20,
            batch_sizes: vec![1, 16, 64, 128],
        }
    }
}

// ============================================================================
// Result Structures
// ============================================================================

/// Statistical summary of benchmark measurements.
///
/// All latency values are in milliseconds. `samples_per_second` is derived
/// from `mean_ms` and the batch size used during measurement.
///
/// # Fields
///
/// * `mean_ms` - Arithmetic mean of per-iteration latencies
/// * `median_ms` - Median of per-iteration latencies
/// * `std_dev_ms` - Population standard deviation of latencies
/// * `min_ms` - Minimum observed latency
/// * `max_ms` - Maximum observed latency
/// * `samples_per_second` - Estimated throughput (batch_size / mean latency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    /// Arithmetic mean latency in milliseconds.
    pub mean_ms: f64,
    /// Median latency in milliseconds.
    pub median_ms: f64,
    /// Population standard deviation of latency in milliseconds.
    pub std_dev_ms: f64,
    /// Minimum observed latency in milliseconds.
    pub min_ms: f64,
    /// Maximum observed latency in milliseconds.
    pub max_ms: f64,
    /// Estimated throughput in samples per second.
    pub samples_per_second: f64,
}

/// Complete result from benchmarking a single layer configuration.
///
/// Records metadata about the layer and the statistical results for both
/// the forward-only pass and the optional full training step
/// (forward + backward + parameter update).
///
/// # Examples
///
/// ```ignore
/// let result = measure_layer_forward("MLP", &layer, &input, 64, &config);
/// println!("{}", serde_json::to_string_pretty(&result).unwrap());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Human-readable name identifying this benchmark (e.g. "MLP Dense 784→512").
    pub name: String,
    /// Batch size used during this benchmark run.
    pub batch_size: usize,
    /// Number of input features per sample.
    pub input_size: usize,
    /// Number of output features per sample.
    pub output_size: usize,
    /// Total number of trainable parameters in the layer.
    pub parameter_count: usize,
    /// Statistics for the forward pass only.
    pub forward_stats: BenchmarkStats,
    /// Statistics for the full training step (forward + backward + update),
    /// or `None` if only a forward benchmark was run.
    pub training_stats: Option<BenchmarkStats>,
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute `BenchmarkStats` from a slice of per-iteration durations in milliseconds.
///
/// # Panics
///
/// Panics if `samples` is empty.
fn compute_stats(samples: &[f64], batch_size: usize) -> BenchmarkStats {
    assert!(!samples.is_empty(), "benchmark samples must not be empty");

    let n = samples.len() as f64;
    let mean_ms = samples.iter().sum::<f64>() / n;

    // Population standard deviation
    let variance = samples.iter().map(|&x| (x - mean_ms).powi(2)).sum::<f64>() / n;
    let std_dev_ms = variance.sqrt();

    // Min and max
    let min_ms = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Median (requires sorted copy)
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median_ms = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };

    // Throughput: batch_size samples per mean_ms milliseconds → scale to per second
    let samples_per_second = if mean_ms > 0.0 {
        (batch_size as f64 / mean_ms) * 1_000.0
    } else {
        f64::INFINITY
    };

    BenchmarkStats {
        mean_ms,
        median_ms,
        std_dev_ms,
        min_ms,
        max_ms,
        samples_per_second,
    }
}

// ============================================================================
// Public benchmark functions
// ============================================================================

/// Benchmark the forward pass of a layer using synthetic data.
///
/// Runs `config.warmup_iterations` untimed warmup passes followed by
/// `config.measure_iterations` timed passes. Returns a [`BenchmarkResult`]
/// with populated `forward_stats` and `training_stats` set to `None`.
///
/// # Arguments
///
/// * `name` - Human-readable label for the benchmark (e.g. `"MLP 784→512"`)
/// * `layer` - Reference to any type implementing [`Layer`]
/// * `input` - Pre-allocated input data with length `batch_size × layer.input_size()`
/// * `batch_size` - Number of samples in the batch
/// * `config` - Benchmark configuration (warmup/measure iteration counts)
///
/// # Panics
///
/// Panics if `input.len() != batch_size * layer.input_size()`.
///
/// # Examples
///
/// ```ignore
/// use rust_neural_networks::benchmark::{BenchmarkConfig, measure_layer_forward};
/// use rust_neural_networks::layers::DenseLayer;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(784, 512, &mut rng);
/// let config = BenchmarkConfig::default();
/// let input = vec![0.1f32; 64 * 784];
/// let result = measure_layer_forward("MLP", &layer, &input, 64, &config);
/// assert!(result.forward_stats.mean_ms >= 0.0);
/// ```
pub fn measure_layer_forward(
    name: &str,
    layer: &dyn Layer,
    input: &[f32],
    batch_size: usize,
    config: &BenchmarkConfig,
) -> BenchmarkResult {
    let output_size = layer.output_size();
    let mut output = vec![0.0f32; batch_size * output_size];

    // Warmup phase – not measured
    for _ in 0..config.warmup_iterations {
        layer.forward(input, &mut output, batch_size);
    }

    // Measurement phase
    let mut durations_ms = Vec::with_capacity(config.measure_iterations);
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        layer.forward(input, &mut output, batch_size);
        let elapsed = start.elapsed();
        durations_ms.push(elapsed.as_secs_f64() * 1_000.0);
    }

    let forward_stats = compute_stats(&durations_ms, batch_size);

    BenchmarkResult {
        name: name.to_string(),
        batch_size,
        input_size: layer.input_size(),
        output_size,
        parameter_count: layer.parameter_count(),
        forward_stats,
        training_stats: None,
    }
}

/// Benchmark a full training step (forward + backward + parameter update) for a layer.
///
/// Runs `config.warmup_iterations` untimed warmup steps followed by
/// `config.measure_iterations` timed steps. The forward-only stats are also
/// recorded separately so callers can compare forward vs training overhead.
///
/// # Arguments
///
/// * `name` - Human-readable label for the benchmark
/// * `layer` - Mutable reference to any type implementing [`Layer`]
/// * `input` - Pre-allocated input data (`batch_size × input_size`)
/// * `grad_output` - Pre-allocated upstream gradient (`batch_size × output_size`)
/// * `batch_size` - Number of samples in the batch
/// * `learning_rate` - Learning rate used for `update_parameters`
/// * `config` - Benchmark configuration
///
/// # Panics
///
/// Panics if `input.len() != batch_size * layer.input_size()` or
/// `grad_output.len() != batch_size * layer.output_size()`.
///
/// # Examples
///
/// ```ignore
/// use rust_neural_networks::benchmark::{BenchmarkConfig, measure_layer_training_step};
/// use rust_neural_networks::layers::DenseLayer;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let mut layer = DenseLayer::new(784, 512, &mut rng);
/// let config = BenchmarkConfig::default();
/// let input = vec![0.1f32; 64 * 784];
/// let grad_output = vec![0.01f32; 64 * 512];
/// let result = measure_layer_training_step(
///     "MLP train", &mut layer, &input, &grad_output, 64, 0.01, &config);
/// assert!(result.training_stats.is_some());
/// ```
pub fn measure_layer_training_step(
    name: &str,
    layer: &mut dyn Layer,
    input: &[f32],
    grad_output: &[f32],
    batch_size: usize,
    learning_rate: f32,
    config: &BenchmarkConfig,
) -> BenchmarkResult {
    let input_size = layer.input_size();
    let output_size = layer.output_size();

    let mut output = vec![0.0f32; batch_size * output_size];
    let mut grad_input = vec![0.0f32; batch_size * input_size];

    // ---- Warmup phase (not measured) ----
    for _ in 0..config.warmup_iterations {
        layer.forward(input, &mut output, batch_size);
        layer.backward(input, grad_output, &mut grad_input, batch_size);
        layer.update_parameters(learning_rate);
    }

    // ---- Measure forward-only pass ----
    let mut forward_durations_ms = Vec::with_capacity(config.measure_iterations);
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        layer.forward(input, &mut output, batch_size);
        let elapsed = start.elapsed();
        forward_durations_ms.push(elapsed.as_secs_f64() * 1_000.0);
    }

    // ---- Measure full training step ----
    let mut training_durations_ms = Vec::with_capacity(config.measure_iterations);
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        layer.forward(input, &mut output, batch_size);
        layer.backward(input, grad_output, &mut grad_input, batch_size);
        layer.update_parameters(learning_rate);
        let elapsed = start.elapsed();
        training_durations_ms.push(elapsed.as_secs_f64() * 1_000.0);
    }

    let forward_stats = compute_stats(&forward_durations_ms, batch_size);
    let training_stats = compute_stats(&training_durations_ms, batch_size);

    BenchmarkResult {
        name: name.to_string(),
        batch_size,
        input_size,
        output_size,
        parameter_count: layer.parameter_count(),
        forward_stats,
        training_stats: Some(training_stats),
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal stub layer for testing the benchmark harness without BLAS.
    struct StubLayer {
        input_size: usize,
        output_size: usize,
    }

    impl StubLayer {
        fn new(input_size: usize, output_size: usize) -> Self {
            Self {
                input_size,
                output_size,
            }
        }
    }

    impl Layer for StubLayer {
        fn forward(&self, _input: &[f32], output: &mut [f32], _batch_size: usize) {
            output.fill(0.0);
        }

        fn backward(
            &self,
            _input: &[f32],
            _grad_output: &[f32],
            grad_input: &mut [f32],
            _batch_size: usize,
        ) {
            grad_input.fill(0.0);
        }

        fn update_parameters(&mut self, _learning_rate: f32) {}

        fn input_size(&self) -> usize {
            self.input_size
        }

        fn output_size(&self) -> usize {
            self.output_size
        }

        fn parameter_count(&self) -> usize {
            self.input_size * self.output_size + self.output_size
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measure_iterations, 20);
        assert!(!config.batch_sizes.is_empty());
    }

    #[test]
    fn test_benchmark_config_serde_roundtrip() {
        let config = BenchmarkConfig {
            warmup_iterations: 3,
            measure_iterations: 10,
            batch_sizes: vec![1, 32, 64],
        };
        let json = serde_json::to_string(&config).expect("serialize config");
        let deserialized: BenchmarkConfig =
            serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(deserialized.warmup_iterations, 3);
        assert_eq!(deserialized.measure_iterations, 10);
        assert_eq!(deserialized.batch_sizes, vec![1, 32, 64]);
    }

    #[test]
    fn test_compute_stats_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&samples, 32);
        assert!((stats.mean_ms - 3.0).abs() < 1e-9);
        assert!((stats.median_ms - 3.0).abs() < 1e-9);
        assert!((stats.min_ms - 1.0).abs() < 1e-9);
        assert!((stats.max_ms - 5.0).abs() < 1e-9);
        assert!(stats.std_dev_ms >= 0.0);
        assert!(stats.samples_per_second > 0.0);
    }

    #[test]
    fn test_compute_stats_even_count() {
        let samples = vec![1.0, 3.0, 5.0, 7.0];
        let stats = compute_stats(&samples, 16);
        // median of [1,3,5,7] → (3+5)/2 = 4.0
        assert!((stats.median_ms - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_measure_layer_forward_produces_valid_result() {
        let layer = StubLayer::new(16, 8);
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![4],
        };
        let input = vec![0.1f32; 4 * 16];
        let result = measure_layer_forward("StubForward", &layer, &input, 4, &config);

        assert_eq!(result.name, "StubForward");
        assert_eq!(result.batch_size, 4);
        assert_eq!(result.input_size, 16);
        assert_eq!(result.output_size, 8);
        assert!(result.forward_stats.mean_ms >= 0.0);
        assert!(result.forward_stats.min_ms >= 0.0);
        assert!(result.forward_stats.max_ms >= result.forward_stats.min_ms);
        assert!(result.forward_stats.samples_per_second > 0.0);
        assert!(result.training_stats.is_none());
    }

    #[test]
    fn test_measure_layer_training_step_produces_valid_result() {
        let mut layer = StubLayer::new(16, 8);
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![4],
        };
        let input = vec![0.1f32; 4 * 16];
        let grad_output = vec![0.01f32; 4 * 8];
        let result = measure_layer_training_step(
            "StubTrain",
            &mut layer,
            &input,
            &grad_output,
            4,
            0.01,
            &config,
        );

        assert_eq!(result.name, "StubTrain");
        assert!(result.forward_stats.mean_ms >= 0.0);
        assert!(result.training_stats.is_some());
        let training = result.training_stats.unwrap();
        assert!(training.mean_ms >= 0.0);
        assert!(training.samples_per_second > 0.0);
    }

    #[test]
    fn test_benchmark_result_serde_roundtrip() {
        let stats = BenchmarkStats {
            mean_ms: 1.5,
            median_ms: 1.4,
            std_dev_ms: 0.1,
            min_ms: 1.2,
            max_ms: 1.9,
            samples_per_second: 42666.0,
        };
        let result = BenchmarkResult {
            name: "test".to_string(),
            batch_size: 64,
            input_size: 784,
            output_size: 512,
            parameter_count: 401408,
            forward_stats: stats.clone(),
            training_stats: Some(stats),
        };
        let json = serde_json::to_string(&result).expect("serialize result");
        let deserialized: BenchmarkResult =
            serde_json::from_str(&json).expect("deserialize result");
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.batch_size, 64);
        assert!(deserialized.training_stats.is_some());
    }
}
