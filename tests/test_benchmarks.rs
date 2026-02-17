//! Integration tests for the benchmark module.
//!
//! This file tests the public API of the benchmark module including:
//! - BenchmarkConfig: default values, JSON deserialization
//! - BenchmarkResult: field population after a benchmark run
//! - BenchmarkStats: statistical correctness (mean, min, max, std_dev)
//! - measure_layer_forward(): correctness with a real DenseLayer
//! - Throughput and latency positivity

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

use rust_neural_networks::benchmark::{
    measure_layer_forward, measure_layer_training_step, BenchmarkConfig, BenchmarkResult,
    BenchmarkStats,
};
use rust_neural_networks::layers::DenseLayer;
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// BenchmarkConfig Tests
// ============================================================================

mod benchmark_config_tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default_values() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measure_iterations, 20);
        assert!(!config.batch_sizes.is_empty());
    }

    #[test]
    fn test_benchmark_config_load_from_json() {
        let json = r#"{
            "warmup_iterations": 3,
            "measure_iterations": 10,
            "batch_sizes": [1, 32, 64]
        }"#;
        let config: BenchmarkConfig =
            serde_json::from_str(json).expect("should deserialize BenchmarkConfig from JSON");
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measure_iterations, 10);
        assert_eq!(config.batch_sizes, vec![1, 32, 64]);
    }

    #[test]
    fn test_benchmark_config_serde_roundtrip() {
        let original = BenchmarkConfig {
            warmup_iterations: 7,
            measure_iterations: 15,
            batch_sizes: vec![8, 16, 32],
        };
        let json = serde_json::to_string(&original).expect("should serialize BenchmarkConfig");
        let restored: BenchmarkConfig =
            serde_json::from_str(&json).expect("should deserialize BenchmarkConfig");
        assert_eq!(restored.warmup_iterations, 7);
        assert_eq!(restored.measure_iterations, 15);
        assert_eq!(restored.batch_sizes, vec![8, 16, 32]);
    }

    #[test]
    fn test_benchmark_config_single_batch_size() {
        let json = r#"{
            "warmup_iterations": 1,
            "measure_iterations": 3,
            "batch_sizes": [64]
        }"#;
        let config: BenchmarkConfig =
            serde_json::from_str(json).expect("should deserialize single-batch config");
        assert_eq!(config.batch_sizes.len(), 1);
        assert_eq!(config.batch_sizes[0], 64);
    }
}

// ============================================================================
// BenchmarkStats Tests
// ============================================================================

mod benchmark_stats_tests {
    use super::*;

    /// Helper to create a BenchmarkStats with known values for assertion tests.
    fn make_stats(
        mean: f64,
        median: f64,
        std_dev: f64,
        min: f64,
        max: f64,
        tput: f64,
    ) -> BenchmarkStats {
        BenchmarkStats {
            mean_ms: mean,
            median_ms: median,
            std_dev_ms: std_dev,
            min_ms: min,
            max_ms: max,
            samples_per_second: tput,
        }
    }

    #[test]
    fn test_benchmark_stats_fields_are_accessible() {
        let stats = make_stats(2.5, 2.0, 0.5, 1.0, 5.0, 1000.0);
        assert_eq!(stats.mean_ms, 2.5);
        assert_eq!(stats.median_ms, 2.0);
        assert_eq!(stats.std_dev_ms, 0.5);
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 5.0);
        assert_eq!(stats.samples_per_second, 1000.0);
    }

    #[test]
    fn test_benchmark_stats_serde_roundtrip() {
        let original = make_stats(1.5, 1.4, 0.1, 1.2, 1.9, 42666.0);
        let json = serde_json::to_string(&original).expect("should serialize BenchmarkStats");
        let restored: BenchmarkStats =
            serde_json::from_str(&json).expect("should deserialize BenchmarkStats");
        assert!((restored.mean_ms - 1.5).abs() < 1e-9);
        assert!((restored.median_ms - 1.4).abs() < 1e-9);
        assert!((restored.std_dev_ms - 0.1).abs() < 1e-9);
        assert!((restored.min_ms - 1.2).abs() < 1e-9);
        assert!((restored.max_ms - 1.9).abs() < 1e-9);
        assert!((restored.samples_per_second - 42666.0).abs() < 1e-3);
    }
}

// ============================================================================
// BenchmarkResult Field Population Tests
// ============================================================================

mod benchmark_result_tests {
    use super::*;

    #[test]
    fn test_benchmark_result_fields_populated_after_forward_run() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(16, 8, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![4],
        };
        let input = vec![0.5f32; 4 * 16];
        let result = measure_layer_forward("TestForward", &layer, &input, 4, &config);

        // Name field is set correctly
        assert_eq!(result.name, "TestForward");
        // Batch size is propagated
        assert_eq!(result.batch_size, 4);
        // Layer dimensions are recorded
        assert_eq!(result.input_size, 16);
        assert_eq!(result.output_size, 8);
        // Parameter count matches DenseLayer formula: weights + biases
        assert_eq!(result.parameter_count, 16 * 8 + 8);
        // training_stats should be None for a forward-only benchmark
        assert!(result.training_stats.is_none());
    }

    #[test]
    fn test_benchmark_result_fields_populated_after_training_step() {
        let mut rng = SimpleRng::new(99);
        let mut layer = DenseLayer::new(16, 8, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measure_iterations: 3,
            batch_sizes: vec![4],
        };
        let input = vec![0.1f32; 4 * 16];
        let grad_output = vec![0.01f32; 4 * 8];
        let result = measure_layer_training_step(
            "TestTraining",
            &mut layer,
            &input,
            &grad_output,
            4,
            0.01,
            &config,
        );

        assert_eq!(result.name, "TestTraining");
        assert_eq!(result.batch_size, 4);
        assert_eq!(result.input_size, 16);
        assert_eq!(result.output_size, 8);
        assert_eq!(result.parameter_count, 16 * 8 + 8);
        // training_stats should be populated for a training-step benchmark
        assert!(result.training_stats.is_some());
    }

    #[test]
    fn test_benchmark_result_serde_roundtrip() {
        let mut rng = SimpleRng::new(7);
        let layer = DenseLayer::new(8, 4, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measure_iterations: 3,
            batch_sizes: vec![2],
        };
        let input = vec![0.3f32; 2 * 8];
        let result = measure_layer_forward("SerdeTest", &layer, &input, 2, &config);

        let json = serde_json::to_string(&result).expect("should serialize BenchmarkResult");
        let restored: BenchmarkResult =
            serde_json::from_str(&json).expect("should deserialize BenchmarkResult");

        assert_eq!(restored.name, "SerdeTest");
        assert_eq!(restored.batch_size, 2);
        assert_eq!(restored.input_size, 8);
        assert_eq!(restored.output_size, 4);
    }
}

// ============================================================================
// Stats Correctness Tests (via measure_layer_forward output)
// ============================================================================

mod stats_correctness_tests {
    use super::*;

    #[test]
    fn test_stats_mean_is_finite_and_non_negative() {
        let mut rng = SimpleRng::new(1);
        let layer = DenseLayer::new(32, 16, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 10,
            batch_sizes: vec![8],
        };
        let input = vec![0.1f32; 8 * 32];
        let result = measure_layer_forward("StatsTest", &layer, &input, 8, &config);

        assert!(result.forward_stats.mean_ms.is_finite());
        assert!(result.forward_stats.mean_ms >= 0.0);
    }

    #[test]
    fn test_stats_min_le_mean_le_max() {
        let mut rng = SimpleRng::new(2);
        let layer = DenseLayer::new(32, 16, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 10,
            batch_sizes: vec![8],
        };
        let input = vec![0.5f32; 8 * 32];
        let result = measure_layer_forward("MinMaxTest", &layer, &input, 8, &config);

        let s = &result.forward_stats;
        assert!(s.min_ms <= s.mean_ms, "min_ms should be <= mean_ms");
        assert!(s.mean_ms <= s.max_ms, "mean_ms should be <= max_ms");
        assert!(s.min_ms <= s.max_ms, "min_ms should be <= max_ms");
    }

    #[test]
    fn test_stats_std_dev_is_non_negative() {
        let mut rng = SimpleRng::new(3);
        let layer = DenseLayer::new(32, 16, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 10,
            batch_sizes: vec![8],
        };
        let input = vec![0.2f32; 8 * 32];
        let result = measure_layer_forward("StdDevTest", &layer, &input, 8, &config);

        assert!(result.forward_stats.std_dev_ms >= 0.0);
        assert!(result.forward_stats.std_dev_ms.is_finite());
    }

    #[test]
    fn test_stats_median_within_min_max_range() {
        let mut rng = SimpleRng::new(4);
        let layer = DenseLayer::new(32, 16, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 10,
            batch_sizes: vec![8],
        };
        let input = vec![0.4f32; 8 * 32];
        let result = measure_layer_forward("MedianTest", &layer, &input, 8, &config);

        let s = &result.forward_stats;
        assert!(s.median_ms >= s.min_ms, "median should be >= min");
        assert!(s.median_ms <= s.max_ms, "median should be <= max");
    }
}

// ============================================================================
// Throughput and Latency Positivity Tests
// ============================================================================

mod throughput_latency_tests {
    use super::*;

    #[test]
    fn test_forward_latency_is_positive() {
        let mut rng = SimpleRng::new(10);
        let layer = DenseLayer::new(64, 32, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![16],
        };
        let input = vec![0.1f32; 16 * 64];
        let result = measure_layer_forward("LatencyTest", &layer, &input, 16, &config);

        assert!(
            result.forward_stats.mean_ms >= 0.0,
            "mean latency must be non-negative"
        );
        assert!(
            result.forward_stats.min_ms >= 0.0,
            "min latency must be non-negative"
        );
        assert!(
            result.forward_stats.max_ms >= 0.0,
            "max latency must be non-negative"
        );
    }

    #[test]
    fn test_forward_throughput_is_positive() {
        let mut rng = SimpleRng::new(11);
        let layer = DenseLayer::new(64, 32, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![16],
        };
        let input = vec![0.1f32; 16 * 64];
        let result = measure_layer_forward("ThroughputTest", &layer, &input, 16, &config);

        assert!(
            result.forward_stats.samples_per_second > 0.0,
            "throughput must be positive"
        );
    }

    #[test]
    fn test_training_step_throughput_is_positive() {
        let mut rng = SimpleRng::new(12);
        let mut layer = DenseLayer::new(64, 32, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measure_iterations: 3,
            batch_sizes: vec![8],
        };
        let input = vec![0.2f32; 8 * 64];
        let grad_output = vec![0.01f32; 8 * 32];
        let result = measure_layer_training_step(
            "TrainThroughputTest",
            &mut layer,
            &input,
            &grad_output,
            8,
            0.01,
            &config,
        );

        // Forward stats throughput is positive
        assert!(
            result.forward_stats.samples_per_second > 0.0,
            "forward throughput must be positive"
        );
        // Training stats throughput is positive
        let training = result
            .training_stats
            .expect("training_stats should be Some");
        assert!(
            training.samples_per_second > 0.0,
            "training throughput must be positive"
        );
        assert!(
            training.mean_ms >= 0.0,
            "training mean latency must be non-negative"
        );
    }

    #[test]
    fn test_measure_layer_forward_with_various_input_sizes() {
        // Test with a larger DenseLayer resembling a real MLP hidden layer
        let mut rng = SimpleRng::new(20);
        let layer = DenseLayer::new(128, 64, &mut rng);

        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measure_iterations: 5,
            batch_sizes: vec![32],
        };
        let batch_size = 32usize;
        let input = vec![0.1f32; batch_size * 128];
        let result = measure_layer_forward("LargerDense", &layer, &input, batch_size, &config);

        assert_eq!(result.name, "LargerDense");
        assert_eq!(result.input_size, 128);
        assert_eq!(result.output_size, 64);
        assert_eq!(result.batch_size, batch_size);
        assert_eq!(result.parameter_count, 128 * 64 + 64);
        assert!(result.forward_stats.samples_per_second > 0.0);
        assert!(result.forward_stats.mean_ms >= 0.0);
        assert!(result.forward_stats.min_ms <= result.forward_stats.max_ms);
    }
}
