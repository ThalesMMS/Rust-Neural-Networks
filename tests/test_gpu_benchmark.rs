//! GPU performance benchmarks comparing GPU vs CPU throughput.
//!
//! Measures speedup ratios for:
//! - DenseLayer 784→512 forward/backward at batch_size=64/256/1024
//! - Conv2DLayer 3×3 kernel on 28×28 and 32×32 inputs
//! - Full forward pass through 784→512→10 MLP
//!
//! Uses the BenchmarkConfig infrastructure from src/benchmark.rs.

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

use rust_neural_networks::benchmark::{
    measure_layer_forward, measure_layer_training_step, BenchmarkConfig,
};
use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;
use std::sync::Arc;
use std::time::Instant;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use rust_neural_networks::gpu::create_gpu_backend;

// ============================================================================
// Helper: benchmark config for GPU tests
// ============================================================================

/// Create a BenchmarkConfig preconfigured for GPU performance comparisons.
///
/// The returned config uses 3 warmup iterations, 10 measurement iterations,
/// and batch sizes [64, 256, 1024].
///
/// # Examples
///
/// ```
/// let cfg = gpu_benchmark_config();
/// assert_eq!(cfg.warmup_iterations, 3);
/// assert_eq!(cfg.measure_iterations, 10);
/// assert_eq!(cfg.batch_sizes, vec![64, 256, 1024]);
/// ```
fn gpu_benchmark_config() -> BenchmarkConfig {
    BenchmarkConfig {
        warmup_iterations: 3,
        measure_iterations: 10,
        batch_sizes: vec![64, 256, 1024],
    }
}

/// Prints a formatted CPU vs GPU timing row for a label.
///
/// The printed line shows the label, CPU time (ms), GPU time (ms), and the computed speedup.
///
/// # Returns
///
/// `cpu_ms / gpu_ms` if `gpu_ms` is greater than zero, `f64::INFINITY` if `gpu_ms` is zero.
///
/// # Examples
///
/// ```
/// let speedup = print_speedup("Dense 784→512 forward", 120.0, 10.0);
/// assert_eq!(speedup, 12.0);
/// ```
fn print_speedup(label: &str, cpu_ms: f64, gpu_ms: f64) -> f64 {
    let speedup = if gpu_ms > 0.0 {
        cpu_ms / gpu_ms
    } else {
        f64::INFINITY
    };
    println!(
        "  {:<45} CPU: {:8.3} ms  GPU: {:8.3} ms  Speedup: {:6.2}x",
        label, cpu_ms, gpu_ms, speedup
    );
    speedup
}

// ============================================================================
// DenseLayer Forward/Backward Benchmarks
// ============================================================================

/// Benchmark the forward pass of a 784→512 DenseLayer on CPU and GPU across multiple batch sizes.
///
/// Measures mean execution time for CPU and GPU implementations for batch sizes 64, 256, and 1024,
/// prints per-batch speedups, and asserts that GPU timings and throughput are positive. If no GPU
/// backend is available the test exits early without failing.
///
/// # Examples
///
/// ```no_run
/// if let Some(backend) = create_gpu_backend() {
///     println!("GPU device: {}", backend.device_info().name);
/// } else {
///     println!("No GPU backend available, skipping benchmark");
/// }
/// ```
#[test]
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
fn test_gpu_benchmark_dense_forward() {
    let backend = match create_gpu_backend() {
        Some(b) => b,
        None => {
            println!("No GPU backend available, skipping benchmark");
            return;
        }
    };

    println!("\n=== DenseLayer 784→512 Forward Pass Benchmark ===");
    println!("  GPU: {}", backend.device_info().name);

    let config = gpu_benchmark_config();
    let mut rng = SimpleRng::new(42);

    for &batch_size in &[64, 256, 1024] {
        // CPU layer
        let cpu_layer = DenseLayer::new(784, 512, &mut rng);
        let input = vec![0.5f32; batch_size * 784];
        let cpu_result =
            measure_layer_forward("CPU Dense 784→512", &cpu_layer, &input, batch_size, &config);

        // GPU layer
        let gpu_layer = DenseLayer::new_with_gpu(784, 512, &mut rng, Arc::clone(&backend));
        let gpu_result =
            measure_layer_forward("GPU Dense 784→512", &gpu_layer, &input, batch_size, &config);

        let label = format!("Dense forward batch_size={}", batch_size);
        let speedup = print_speedup(
            &label,
            cpu_result.forward_stats.mean_ms,
            gpu_result.forward_stats.mean_ms,
        );

        // GPU should produce valid results
        assert!(gpu_result.forward_stats.mean_ms > 0.0);
        assert!(gpu_result.forward_stats.samples_per_second > 0.0);

        // For large batches, expect meaningful speedup
        if batch_size >= 256 {
            println!("    (batch_size={} speedup: {:.2}x)", batch_size, speedup);
        }
    }
}

/// Benchmarks the training step of a 784→512 DenseLayer on CPU and GPU and reports per-batch speedups.
///
/// The test measures mean training time for batch sizes 64, 256, and 1024 using a fixed RNG and a
/// standard benchmark configuration (see `gpu_benchmark_config`). If no GPU backend is available
/// the test exits early and prints a notice. For each batch size it prints the GPU device name,
/// the CPU and GPU mean training times, and the computed CPU→GPU speedup; the test asserts that
/// the measured GPU training time is greater than zero.
///
/// # Examples
///
/// ```no_run
/// // Running the full benchmark requires the test harness and an available GPU backend.
/// // This example demonstrates the early-availability check used by the test.
/// if let Some(_backend) = crate::create_gpu_backend() {
///     // The benchmark is implemented as a test; run via `cargo test --tests`.
///     // Calling the test function directly is not necessary when using the test harness.
/// }
/// ```
#[test]
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
fn test_gpu_benchmark_dense_training_step() {
    let backend = match create_gpu_backend() {
        Some(b) => b,
        None => {
            println!("No GPU backend available, skipping benchmark");
            return;
        }
    };

    println!("\n=== DenseLayer 784→512 Training Step Benchmark ===");
    println!("  GPU: {}", backend.device_info().name);

    let config = gpu_benchmark_config();
    let mut rng = SimpleRng::new(42);

    for &batch_size in &[64, 256, 1024] {
        let input = vec![0.5f32; batch_size * 784];
        let grad_output = vec![0.01f32; batch_size * 512];

        // CPU
        let mut cpu_layer = DenseLayer::new(784, 512, &mut rng);
        let cpu_result = measure_layer_training_step(
            "CPU Dense train",
            &mut cpu_layer,
            &input,
            &grad_output,
            batch_size,
            0.01,
            &config,
        );

        // GPU
        let mut gpu_layer = DenseLayer::new_with_gpu(784, 512, &mut rng, Arc::clone(&backend));
        let gpu_result = measure_layer_training_step(
            "GPU Dense train",
            &mut gpu_layer,
            &input,
            &grad_output,
            batch_size,
            0.01,
            &config,
        );

        let cpu_train_ms = cpu_result.training_stats.as_ref().unwrap().mean_ms;
        let gpu_train_ms = gpu_result.training_stats.as_ref().unwrap().mean_ms;

        let label = format!("Dense train batch_size={}", batch_size);
        print_speedup(&label, cpu_train_ms, gpu_train_ms);

        assert!(gpu_train_ms > 0.0);
    }
}

// ============================================================================
// Conv2DLayer Benchmarks
// ============================================================================

#[test]
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
fn test_gpu_benchmark_conv2d() {
    let backend = match create_gpu_backend() {
        Some(b) => b,
        None => {
            println!("No GPU backend available, skipping benchmark");
            return;
        }
    };

    println!("\n=== Conv2DLayer 3×3 Kernel Benchmark ===");
    println!("  GPU: {}", backend.device_info().name);

    let config = BenchmarkConfig {
        warmup_iterations: 3,
        measure_iterations: 10,
        batch_sizes: vec![64],
    };
    let mut rng = SimpleRng::new(42);
    let batch_size = 64;

    // Test configurations: (in_channels, input_size, label)
    let configs = [
        (1, 28, "1ch 28×28 (MNIST)"),
        (3, 32, "3ch 32×32 (CIFAR-10)"),
    ];

    for &(in_channels, input_hw, label) in &configs {
        let out_channels = 8;
        let kernel_size = 3;

        // CPU Conv2D
        let cpu_layer = Conv2DLayer::new(
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            input_hw,
            input_hw,
            &mut rng,
        );
        let input = vec![0.5f32; batch_size * in_channels * input_hw * input_hw];
        let cpu_result = measure_layer_forward(
            &format!("CPU Conv2D {}", label),
            &cpu_layer,
            &input,
            batch_size,
            &config,
        );

        // GPU Conv2D
        let gpu_layer = Conv2DLayer::new_with_gpu(
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            input_hw,
            input_hw,
            &mut rng,
            Arc::clone(&backend),
        );
        let gpu_result = measure_layer_forward(
            &format!("GPU Conv2D {}", label),
            &gpu_layer,
            &input,
            batch_size,
            &config,
        );

        let _speedup = print_speedup(
            &format!("Conv2D {} batch={}", label, batch_size),
            cpu_result.forward_stats.mean_ms,
            gpu_result.forward_stats.mean_ms,
        );

        assert!(gpu_result.forward_stats.mean_ms > 0.0);
    }
}

// ============================================================================
// Full MLP Forward Pass Benchmark (784→512→10)
// ============================================================================

/// Benchmarks the forward pass of a two-layer MLP (784 → 512 → 10) on CPU and GPU across batch sizes,
/// measuring mean forward time and printing per-batch speedup.
///
/// The test runs a small number of warmup iterations followed by timed iterations for each of
/// batch sizes [64, 256, 1024]. If no GPU backend can be created at runtime the benchmark is
/// skipped with a printed notice.
///
/// # Examples
///
/// ```
/// // Run the specific benchmark test and show output:
/// // cargo test --test test_gpu_benchmark -- --nocapture test_gpu_benchmark_full_mlp_forward
/// ```
#[test]
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
fn test_gpu_benchmark_full_mlp_forward() {
    let backend = match create_gpu_backend() {
        Some(b) => b,
        None => {
            println!("No GPU backend available, skipping benchmark");
            return;
        }
    };

    println!("\n=== Full MLP 784→512→10 Forward Pass Benchmark ===");
    println!("  GPU: {}", backend.device_info().name);

    let mut rng = SimpleRng::new(42);
    let warmup = 3;
    let iterations = 10;

    for &batch_size in &[64, 256, 1024] {
        let input = vec![0.5f32; batch_size * 784];

        // --- CPU MLP ---
        let cpu_layer1 = DenseLayer::new(784, 512, &mut rng);
        let cpu_layer2 = DenseLayer::new(512, 10, &mut rng);

        let mut hidden = vec![0.0f32; batch_size * 512];
        let mut output = vec![0.0f32; batch_size * 10];

        // Warmup
        for _ in 0..warmup {
            cpu_layer1.forward(&input, &mut hidden, batch_size);
            // Apply ReLU in-place
            hidden.iter_mut().for_each(|x| {
                if *x < 0.0 {
                    *x = 0.0
                }
            });
            cpu_layer2.forward(&hidden, &mut output, batch_size);
        }

        // Measure CPU
        let mut cpu_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            cpu_layer1.forward(&input, &mut hidden, batch_size);
            hidden.iter_mut().for_each(|x| {
                if *x < 0.0 {
                    *x = 0.0
                }
            });
            cpu_layer2.forward(&hidden, &mut output, batch_size);
            cpu_times.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let cpu_mean = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;

        // --- GPU MLP ---
        let gpu_layer1 = DenseLayer::new_with_gpu(784, 512, &mut rng, Arc::clone(&backend));
        let gpu_layer2 = DenseLayer::new_with_gpu(512, 10, &mut rng, Arc::clone(&backend));

        // Warmup
        for _ in 0..warmup {
            gpu_layer1.forward(&input, &mut hidden, batch_size);
            hidden.iter_mut().for_each(|x| {
                if *x < 0.0 {
                    *x = 0.0
                }
            });
            gpu_layer2.forward(&hidden, &mut output, batch_size);
        }

        // Measure GPU
        let mut gpu_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            gpu_layer1.forward(&input, &mut hidden, batch_size);
            hidden.iter_mut().for_each(|x| {
                if *x < 0.0 {
                    *x = 0.0
                }
            });
            gpu_layer2.forward(&hidden, &mut output, batch_size);
            gpu_times.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let gpu_mean = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;

        let label = format!("MLP 784→512→10 batch_size={}", batch_size);
        print_speedup(&label, cpu_mean, gpu_mean);
    }
}

// ============================================================================
// Speedup Goal Verification
// ============================================================================

#[test]
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
fn test_gpu_benchmark_speedup_summary() {
    let backend = match create_gpu_backend() {
        Some(b) => b,
        None => {
            println!("No GPU backend available, skipping speedup summary");
            return;
        }
    };

    println!("\n=== GPU Speedup Summary ===");
    println!("  GPU: {}", backend.device_info().name);
    println!("  Target: 10x+ speedup for large batch operations\n");

    let config = BenchmarkConfig {
        warmup_iterations: 5,
        measure_iterations: 20,
        batch_sizes: vec![1024],
    };
    let mut rng = SimpleRng::new(42);
    let batch_size = 1024;

    // Dense 784→512 forward at batch=1024
    let input = vec![0.5f32; batch_size * 784];

    let cpu_layer = DenseLayer::new(784, 512, &mut rng);
    let cpu_result = measure_layer_forward("CPU", &cpu_layer, &input, batch_size, &config);

    let gpu_layer = DenseLayer::new_with_gpu(784, 512, &mut rng, Arc::clone(&backend));
    let gpu_result = measure_layer_forward("GPU", &gpu_layer, &input, batch_size, &config);

    let speedup = print_speedup(
        "Dense 784→512 forward (batch=1024)",
        cpu_result.forward_stats.mean_ms,
        gpu_result.forward_stats.mean_ms,
    );

    println!(
        "\n  GPU throughput: {:.0} samples/sec",
        gpu_result.forward_stats.samples_per_second
    );
    println!(
        "  CPU throughput: {:.0} samples/sec",
        cpu_result.forward_stats.samples_per_second
    );

    // Report but don't hard-fail on speedup target (hardware-dependent)
    if speedup >= 10.0 {
        println!("\n  ✓ 10x speedup goal ACHIEVED ({:.1}x)", speedup);
    } else if speedup >= 2.0 {
        println!(
            "\n  ~ Partial speedup ({:.1}x) - may improve with larger batches or on discrete GPU",
            speedup
        );
    } else {
        println!(
            "\n  ! Minimal speedup ({:.1}x) - GPU overhead may dominate at this scale",
            speedup
        );
    }

    // GPU must at least produce valid positive results
    assert!(gpu_result.forward_stats.mean_ms > 0.0);
    assert!(gpu_result.forward_stats.samples_per_second > 0.0);
}
