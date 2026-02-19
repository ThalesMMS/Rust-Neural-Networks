// Tests for GPU backend: fallback behaviour, device detection, and integration.
//
// These tests verify that:
// 1. The library works correctly without GPU features compiled (CPU fallback).
// 2. Layers gracefully use the CPU path when no GPU backend is set.
// 3. GPU device detection returns valid info on supported hardware.
// 4. End-to-end training with DenseLayer produces decreasing loss (XOR problem).

use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

// ============================================================================
// Helper: softmax for a single row
// ============================================================================

/// Normalizes a slice of unnormalized scores into a probability distribution in place.
///
/// The input `data` is replaced with non-negative values that sum to 1, representing probabilities.
///
/// # Examples
///
/// ```
/// let mut scores = [1.0f32, 2.0, 3.0];
/// softmax(&mut scores);
/// let sum: f32 = scores.iter().copied().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// for &p in &scores {
///     assert!(p >= 0.0);
/// }
/// ```
fn softmax(data: &mut [f32]) {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in data.iter_mut() {
        *v *= inv;
    }
}

// ============================================================================
// 1. CPU fallback: DenseLayer works without any GPU backend
// ============================================================================

#[cfg(test)]
mod cpu_fallback_tests {
    use super::*;

    #[test]
    fn test_dense_layer_forward_without_gpu() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let mut output = vec![0.0f32; 3];

        layer.forward(&input, &mut output, 1);

        // Output should be non-zero (weights are randomly initialized)
        let has_nonzero = output.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "forward pass should produce non-zero output");
    }

    /// Verifies that DenseLayer.backward computes non-zero gradients with respect to the input on CPU.
    ///
    /// Runs a single forward and backward pass on a small example and asserts that the computed
    /// input gradients contain at least one non-zero element.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut rng = SimpleRng::new(42);
    /// let layer = DenseLayer::new(4, 3, &mut rng);
    ///
    /// let input = vec![1.0f32, 0.5, -0.3, 0.8];
    /// let mut output = vec![0.0f32; 3];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// let grad_output = vec![0.1f32, -0.2, 0.3];
    /// let mut grad_input = vec![0.0f32; 4];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// let has_nonzero = grad_input.iter().any(|&v| v.abs() > 1e-10);
    /// assert!(has_nonzero, "backward pass should produce non-zero gradients");
    /// ```
    #[test]
    fn test_dense_layer_backward_without_gpu() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 3, &mut rng);

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let mut output = vec![0.0f32; 3];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![0.1f32, -0.2, 0.3];
        let mut grad_input = vec![0.0f32; 4];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        let has_nonzero = grad_input.iter().any(|&v| v.abs() > 1e-10);
        assert!(
            has_nonzero,
            "backward pass should produce non-zero gradients"
        );
    }

    /// Verifies that DenseLayer.forward produces identical outputs for identical inputs across a batch.
    ///
    /// Creates a DenseLayer, runs a forward pass with a batch where every input row is identical,
    /// and asserts each output row equals the first output row within a 1e-5 tolerance.
    #[test]
    fn test_dense_layer_batch_forward_without_gpu() {
        let mut rng = SimpleRng::new(123);
        let layer = DenseLayer::new(2, 3, &mut rng);

        let batch_size = 4;
        let input = vec![1.0f32; batch_size * 2];
        let mut output = vec![0.0f32; batch_size * 3];

        layer.forward(&input, &mut output, batch_size);

        // All rows should have the same output (same input)
        for row in 1..batch_size {
            for col in 0..3 {
                assert!(
                    (output[row * 3 + col] - output[col]).abs() < 1e-5,
                    "batch rows with identical input should produce identical output"
                );
            }
        }
    }

    /// Ensures that calling update_parameters on a DenseLayer modifies its forward output after a forward+backward pass.
    ///
    /// This test performs a forward pass, a backward pass to accumulate gradients, records the output,
    /// calls `update_parameters`, and verifies the subsequent forward output differs from the recorded one.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = DenseLayer::new(2, 2, &mut rng);
    ///
    /// let input = vec![1.0f32, 1.0];
    /// let mut out1 = vec![0.0f32; 2];
    /// layer.forward(&input, &mut out1, 1);
    ///
    /// let grad_output = vec![1.0f32, -1.0f32];
    /// let mut grad_input = vec![0.0f32; 2];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// layer.update_parameters(0.1);
    ///
    /// let mut out2 = vec![0.0f32; 2];
    /// layer.forward(&input, &mut out2, 1);
    ///
    /// assert!(out1.iter().zip(out2.iter()).any(|(a,b)| (a - b).abs() > 1e-10));
    /// ```
    #[test]
    fn test_dense_layer_parameter_update_without_gpu() {
        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(2, 2, &mut rng);

        // Forward + backward to accumulate gradients
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        layer.forward(&input, &mut output, 1);

        let grad_output = vec![1.0f32, -1.0];
        let mut grad_input = vec![0.0f32; 2];
        layer.backward(&input, &grad_output, &mut grad_input, 1);

        // Record output before update
        let mut output_before = vec![0.0f32; 2];
        layer.forward(&input, &mut output_before, 1);

        // Update parameters
        layer.update_parameters(0.1);

        // Output should change after parameter update
        let mut output_after = vec![0.0f32; 2];
        layer.forward(&input, &mut output_after, 1);

        let changed = output_before
            .iter()
            .zip(output_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "parameters should change after update");
    }
}

// ============================================================================
// 2. GPU detection (works regardless of feature flags)
// ============================================================================

#[cfg(test)]
mod gpu_detection_tests {
    /// Verifies the crate compiles and executes when neither the `gpu-metal` nor `gpu-cuda` features are enabled.
    ///
    /// The test succeeds if the cfg-gated block for non-GPU builds compiles (demonstrating the library works on CPU-only builds).
    ///
    /// # Examples
    ///
    /// ```
    /// // Run the test with GPU features disabled:
    /// // cargo test --no-default-features
    /// ```
    #[test]
    fn test_no_gpu_without_features() {
        // When compiled without gpu-metal/gpu-cuda features, the gpu module
        // is not available. This test simply confirms the library compiles
        // and runs without GPU features.
        #[cfg(not(any(feature = "gpu-metal", feature = "gpu-cuda")))]
        {
            // gpu module is not compiled; library should work fine on CPU.
            // This block compiling at all is the assertion.
        }
    }

    /// Verifies Metal GPU detection returns valid device info when available.
    ///
    /// If a Metal-capable GPU is detected, asserts that the backend type is Metal, the device name is non-empty, and the device is marked supported. If no GPU is detected the test passes (CI without GPU is allowed).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::{detect_gpu, BackendType};
    ///
    /// if let Some(info) = detect_gpu() {
    ///     assert_eq!(info.backend_type, BackendType::Metal);
    ///     assert!(!info.device_name.is_empty());
    ///     assert!(info.is_supported);
    /// }
    /// ```
    #[cfg(feature = "gpu-metal")]
    #[test]
    fn test_metal_gpu_detection() {
        use rust_neural_networks::gpu::{detect_gpu, BackendType};

        if let Some(info) = detect_gpu() {
            assert_eq!(info.backend_type, BackendType::Metal);
            assert!(
                !info.device_name.is_empty(),
                "device name should not be empty"
            );
            assert!(info.is_supported, "detected device should be supported");
        }
        // If no GPU detected, that's OK — test passes on CI without GPU
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn test_cuda_gpu_detection() {
        use rust_neural_networks::gpu::{detect_gpu, BackendType};

        if let Some(info) = detect_gpu() {
            assert_eq!(info.backend_type, BackendType::Cuda);
            assert!(
                !info.device_name.is_empty(),
                "device name should not be empty"
            );
            assert!(info.is_supported, "detected device should be supported");
        }
    }

    /// Verifies that creating a GPU backend yields device information with a non-empty name and marked supported when a backend is available.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::create_gpu_backend;
    ///
    /// if let Some(backend) = create_gpu_backend() {
    ///     let info = backend.device_info();
    ///     assert!(!info.name.is_empty());
    ///     assert!(info.is_supported);
    /// }
    /// ```
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    #[test]
    fn test_create_gpu_backend_returns_valid_device_info() {
        use rust_neural_networks::gpu::create_gpu_backend;

        if let Some(backend) = create_gpu_backend() {
            let info = backend.device_info();
            assert!(!info.name.is_empty(), "device name should not be empty");
            assert!(info.is_supported, "backend device should be supported");
        }
    }
}

// ============================================================================
// 3. GPU backend integration (conditional on feature flags)
// ============================================================================

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
#[cfg(test)]
mod gpu_integration_tests {
    use super::*;
    use rust_neural_networks::gpu::create_gpu_backend;
    use std::sync::Arc;

    #[test]
    fn test_dense_layer_forward_with_gpu_backend() {
        let backend = match create_gpu_backend() {
            Some(b) => b,
            None => return, // No GPU available; skip test
        };

        let mut rng = SimpleRng::new(42);
        let mut layer = DenseLayer::new(4, 3, &mut rng);
        layer.set_gpu_backend(Arc::clone(&backend));

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let mut output = vec![0.0f32; 3];

        layer.forward(&input, &mut output, 1);

        let has_nonzero = output.iter().any(|&v| v.abs() > 1e-10);
        assert!(
            has_nonzero,
            "GPU forward pass should produce non-zero output"
        );
    }

    /// Verifies that a DenseLayer using a GPU backend produces the same forward outputs as the CPU implementation.
    ///
    /// This test creates two identically initialized DenseLayer instances, attaches a GPU backend to one,
    /// runs a forward pass on the same input, and asserts element-wise equality within a small tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// // Skips if no GPU backend is available.
    /// let backend = match create_gpu_backend() { Some(b) => b, None => return };
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer_cpu = DenseLayer::new(4, 3, &mut rng);
    ///
    /// let mut rng2 = SimpleRng::new(42);
    /// let mut layer_gpu = DenseLayer::new(4, 3, &mut rng2);
    /// layer_gpu.set_gpu_backend(Arc::clone(&backend));
    ///
    /// let input = vec![1.0f32, 0.5, -0.3, 0.8];
    /// let mut output_cpu = vec![0.0f32; 3];
    /// let mut output_gpu = vec![0.0f32; 3];
    ///
    /// layer_cpu.forward(&input, &mut output_cpu, 1);
    /// layer_gpu.forward(&input, &mut output_gpu, 1);
    ///
    /// for (cpu, gpu) in output_cpu.iter().zip(output_gpu.iter()) {
    ///     assert!((cpu - gpu).abs() < 1e-4);
    /// }
    /// ```
    #[test]
    fn test_dense_layer_gpu_matches_cpu() {
        let backend = match create_gpu_backend() {
            Some(b) => b,
            None => return,
        };

        let mut rng = SimpleRng::new(42);
        let layer_cpu = DenseLayer::new(4, 3, &mut rng);

        // Create identical layer for GPU
        let mut rng2 = SimpleRng::new(42);
        let mut layer_gpu = DenseLayer::new(4, 3, &mut rng2);
        layer_gpu.set_gpu_backend(Arc::clone(&backend));

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let mut output_cpu = vec![0.0f32; 3];
        let mut output_gpu = vec![0.0f32; 3];

        layer_cpu.forward(&input, &mut output_cpu, 1);
        layer_gpu.forward(&input, &mut output_gpu, 1);

        for (i, (&cpu, &gpu)) in output_cpu.iter().zip(output_gpu.iter()).enumerate() {
            assert!(
                (cpu - gpu).abs() < 1e-4,
                "GPU and CPU outputs should match at index {}: cpu={}, gpu={}",
                i,
                cpu,
                gpu
            );
        }
    }
}

// ============================================================================
// 4. End-to-end: XOR training with loss decrease
// ============================================================================

#[cfg(test)]
mod xor_training_tests {
    use super::*;
    /// Trains a small MLP on the XOR problem for a fixed number of epochs and asserts the training loss decreases.
    ///
    /// The test builds a 2→8→2 network with ReLU hidden activation and softmax + cross-entropy output, uses a fixed RNG seed and a batch of the 4 XOR samples, and verifies the final loss is smaller than the initial loss and is reduced by at least 50%.
    ///
    /// # Examples
    ///
    /// ```
    /// // Run the test with: `cargo test test_xor_training_loss_decreases`
    /// ```
    #[test]
    fn test_xor_training_loss_decreases() {
        let mut rng = SimpleRng::new(12345);

        // XOR dataset: 4 samples, 2 inputs, 2 outputs (one-hot)
        let inputs: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let labels: Vec<usize> = vec![0, 1, 1, 0]; // XOR truth table

        let mut hidden = DenseLayer::new(2, 8, &mut rng);
        let mut output_layer = DenseLayer::new(8, 2, &mut rng);

        let batch_size = 4;
        let epochs = 200;
        let learning_rate = 0.5f32;
        let mut losses = Vec::new();

        for _epoch in 0..epochs {
            // Forward pass through hidden layer
            let mut hidden_out = vec![0.0f32; batch_size * 8];
            hidden.forward(&inputs, &mut hidden_out, batch_size);

            // ReLU activation
            for v in hidden_out.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let hidden_activated = hidden_out.clone();

            // Forward pass through output layer
            let mut logits = vec![0.0f32; batch_size * 2];
            output_layer.forward(&hidden_activated, &mut logits, batch_size);

            // Softmax + cross-entropy loss
            let mut probs = logits.clone();
            for row in 0..batch_size {
                softmax(&mut probs[row * 2..(row + 1) * 2]);
            }

            let mut epoch_loss = 0.0f32;
            let mut delta = vec![0.0f32; batch_size * 2];
            for (i, &label) in labels.iter().enumerate() {
                let p = probs[i * 2 + label].max(1e-9);
                epoch_loss -= p.ln();
                for j in 0..2 {
                    delta[i * 2 + j] = probs[i * 2 + j];
                    if j == label {
                        delta[i * 2 + j] -= 1.0;
                    }
                    delta[i * 2 + j] /= batch_size as f32;
                }
            }
            losses.push(epoch_loss / batch_size as f32);

            // Backward pass
            let mut grad_hidden = vec![0.0f32; batch_size * 8];
            output_layer.backward(&hidden_activated, &delta, &mut grad_hidden, batch_size);

            // ReLU backward
            for (g, &h) in grad_hidden.iter_mut().zip(hidden_out.iter()) {
                if h <= 0.0 {
                    *g = 0.0;
                }
            }

            let mut grad_input = vec![0.0f32; batch_size * 2];
            hidden.backward(&inputs, &grad_hidden, &mut grad_input, batch_size);

            // Update parameters
            output_layer.update_parameters(learning_rate);
            hidden.update_parameters(learning_rate);
        }

        // Verify loss decreased overall
        let first_loss = losses[0];
        let last_loss = *losses.last().unwrap();
        assert!(
            last_loss < first_loss,
            "loss should decrease: first={:.4}, last={:.4}",
            first_loss,
            last_loss
        );

        // Check that loss decreased significantly (at least 50%)
        assert!(
            last_loss < first_loss * 0.5,
            "loss should decrease significantly: first={:.4}, last={:.4}",
            first_loss,
            last_loss
        );
    }

    /// Trains a small 2→8→2 network on the XOR problem using an available GPU backend and asserts that loss decreases.
    ///
    /// This test attaches a GPU backend to two DenseLayer instances, trains on a four-sample XOR batch for 200 epochs, and verifies the final loss is lower than the initial loss and is less than half of it. The test returns early (skips) if no GPU backend can be created.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Runs the GPU-backed XOR training test; the test will return early if no GPU is available.
    /// test_xor_training_loss_decreases_gpu();
    /// ```
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    #[test]
    fn test_xor_training_loss_decreases_gpu() {
        use rust_neural_networks::gpu::create_gpu_backend;
        use std::sync::Arc;

        let backend = match create_gpu_backend() {
            Some(b) => b,
            None => return, // No GPU; skip
        };

        let mut rng = SimpleRng::new(12345);

        let inputs: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let labels: Vec<usize> = vec![0, 1, 1, 0];

        let mut hidden = DenseLayer::new(2, 8, &mut rng);
        let mut output_layer = DenseLayer::new(8, 2, &mut rng);

        hidden.set_gpu_backend(Arc::clone(&backend));
        output_layer.set_gpu_backend(Arc::clone(&backend));

        let batch_size = 4;
        let epochs = 200;
        let learning_rate = 0.5f32;
        let mut losses = Vec::new();

        for _epoch in 0..epochs {
            let mut hidden_out = vec![0.0f32; batch_size * 8];
            hidden.forward(&inputs, &mut hidden_out, batch_size);

            for v in hidden_out.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let hidden_activated = hidden_out.clone();

            let mut logits = vec![0.0f32; batch_size * 2];
            output_layer.forward(&hidden_activated, &mut logits, batch_size);

            let mut probs = logits.clone();
            for row in 0..batch_size {
                softmax(&mut probs[row * 2..(row + 1) * 2]);
            }

            let mut epoch_loss = 0.0f32;
            let mut delta = vec![0.0f32; batch_size * 2];
            for (i, &label) in labels.iter().enumerate() {
                let p = probs[i * 2 + label].max(1e-9);
                epoch_loss -= p.ln();
                for j in 0..2 {
                    delta[i * 2 + j] = probs[i * 2 + j];
                    if j == label {
                        delta[i * 2 + j] -= 1.0;
                    }
                    delta[i * 2 + j] /= batch_size as f32;
                }
            }
            losses.push(epoch_loss / batch_size as f32);

            let mut grad_hidden = vec![0.0f32; batch_size * 8];
            output_layer.backward(&hidden_activated, &delta, &mut grad_hidden, batch_size);

            for (g, &h) in grad_hidden.iter_mut().zip(hidden_out.iter()) {
                if h <= 0.0 {
                    *g = 0.0;
                }
            }

            let mut grad_input = vec![0.0f32; batch_size * 2];
            hidden.backward(&inputs, &grad_hidden, &mut grad_input, batch_size);

            output_layer.update_parameters(learning_rate);
            hidden.update_parameters(learning_rate);
        }

        let first_loss = losses[0];
        let last_loss = *losses.last().unwrap();
        assert!(
            last_loss < first_loss,
            "GPU loss should decrease: first={:.4}, last={:.4}",
            first_loss,
            last_loss
        );

        assert!(
            last_loss < first_loss * 0.5,
            "GPU loss should decrease significantly: first={:.4}, last={:.4}",
            first_loss,
            last_loss
        );
    }
}