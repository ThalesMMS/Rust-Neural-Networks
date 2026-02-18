//! Tests for PatchEmbeddingLayer

use rust_neural_networks::layers::{Layer, PatchEmbeddingLayer};
use rust_neural_networks::utils::rng::SimpleRng;

#[test]
fn test_patch_embedding_creation() {
    let mut rng = SimpleRng::new(42);
    let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

    assert_eq!(layer.patch_dim(), 48);
    assert_eq!(layer.d_model(), 128);
    assert_eq!(layer.input_size(), 48);
    assert_eq!(layer.output_size(), 128);
    assert!(layer.parameter_count() > 0);
}

#[test]
fn test_patch_embedding_forward_shape() {
    let mut rng = SimpleRng::new(42);
    let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

    // Simulate 4 images x 64 patches = 256 tokens
    let batch_tokens = 4 * 64;
    let input = vec![0.5f32; batch_tokens * 48];
    let mut output = vec![0.0f32; batch_tokens * 128];

    layer.forward(&input, &mut output, batch_tokens);

    assert_eq!(output.len(), batch_tokens * 128);
    assert!(output.iter().all(|&x| x.is_finite()));
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_patch_embedding_backward_shape() {
    let mut rng = SimpleRng::new(42);
    let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

    let batch_tokens = 64; // 1 image * 64 patches
    let input = vec![0.5f32; batch_tokens * 48];
    let mut output = vec![0.0f32; batch_tokens * 128];

    layer.forward(&input, &mut output, batch_tokens);

    let grad_output = vec![1.0f32; batch_tokens * 128];
    let mut grad_input = vec![0.0f32; batch_tokens * 48];

    layer.backward(&input, &grad_output, &mut grad_input, batch_tokens);

    assert_eq!(grad_input.len(), batch_tokens * 48);
    assert!(grad_input.iter().all(|&x| x.is_finite()));
    assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_patch_embedding_parameter_count() {
    let mut rng = SimpleRng::new(42);
    let layer = PatchEmbeddingLayer::new(48, 128, &mut rng);

    // 48 * 128 weights + 128 biases = 6272
    assert_eq!(layer.parameter_count(), 48 * 128 + 128);
}

#[test]
fn test_patch_embedding_numerical_gradient() {
    let mut rng = SimpleRng::new(42);
    let layer = PatchEmbeddingLayer::new(4, 3, &mut rng);

    let batch_size = 2;
    let input = vec![1.0f32, 0.5, -0.5, 0.3, 0.1, -0.1, 0.8, -0.2];
    let mut output = vec![0.0f32; batch_size * 3];

    layer.forward(&input, &mut output, batch_size);

    let grad_output = vec![1.0f32; batch_size * 3];
    let mut grad_input = vec![0.0f32; batch_size * 4];

    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Numerical gradient check
    let eps = 1e-4;
    let mut numerical_grad = vec![0.0f32; batch_size * 4];

    for idx in 0..input.len() {
        let mut input_plus = input.clone();
        let mut input_minus = input.clone();
        input_plus[idx] += eps;
        input_minus[idx] -= eps;

        let mut out_plus = vec![0.0f32; batch_size * 3];
        let mut out_minus = vec![0.0f32; batch_size * 3];

        layer.forward(&input_plus, &mut out_plus, batch_size);
        layer.forward(&input_minus, &mut out_minus, batch_size);

        // Sum of outputs (since grad_output is all 1s)
        let sum_plus: f32 = out_plus.iter().sum();
        let sum_minus: f32 = out_minus.iter().sum();

        numerical_grad[idx] = (sum_plus - sum_minus) / (2.0 * eps);
    }

    // Compare analytical vs numerical gradients
    let mut max_diff = 0.0f32;
    for i in 0..grad_input.len() {
        let diff = (grad_input[i] - numerical_grad[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(
        max_diff < 1e-2,
        "Numerical gradient check failed: max_diff = {}",
        max_diff
    );
}

#[test]
fn test_patch_extraction_cifar10() {
    // Create a synthetic 32x32x3 image with known pattern
    let img_h = 32;
    let img_w = 32;
    let img_c = 3;
    let patch_size = 4;
    let grid = img_h / patch_size; // 8
    let num_patches = grid * grid; // 64
    let patch_dim = patch_size * patch_size * img_c; // 48

    // Fill image with pixel coordinates for easy verification
    let mut image = vec![0.0f32; img_h * img_w * img_c];
    for y in 0..img_h {
        for x in 0..img_w {
            for c in 0..img_c {
                image[(y * img_w + x) * img_c + c] = (y * 100 + x * 10 + c) as f32;
            }
        }
    }

    // Extract patches
    let mut patches = vec![0.0f32; num_patches * patch_dim];

    // Inline patch extraction (mirrors extract_patches_rgb logic)
    for py in 0..grid {
        for px in 0..grid {
            let token_idx = py * grid + px;
            let patch_offset = token_idx * patch_dim;

            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let img_y = py * patch_size + dy;
                    let img_x = px * patch_size + dx;
                    for c in 0..img_c {
                        let pixel_idx = (img_y * img_w + img_x) * img_c + c;
                        let patch_idx = patch_offset + (dy * patch_size + dx) * img_c + c;
                        patches[patch_idx] = image[pixel_idx];
                    }
                }
            }
        }
    }

    // Verify patch count
    assert_eq!(num_patches, 64);
    assert_eq!(patch_dim, 48);

    // Verify first patch (top-left corner: rows 0-3, cols 0-3)
    // First pixel of first patch should be image[0,0] = (0*100 + 0*10 + 0) = 0
    assert_eq!(patches[0], 0.0); // R
    assert_eq!(patches[1], 1.0); // G
    assert_eq!(patches[2], 2.0); // B

    // Verify second patch (top row, second column: rows 0-3, cols 4-7)
    // First pixel should be image[0,4] = (0*100 + 4*10 + c)
    let second_patch_start = patch_dim;
    assert_eq!(patches[second_patch_start], 40.0); // R of pixel (0,4)
    assert_eq!(patches[second_patch_start + 1], 41.0); // G
    assert_eq!(patches[second_patch_start + 2], 42.0); // B
}

#[test]
fn test_patch_embedding_update() {
    let mut rng = SimpleRng::new(42);
    let mut layer = PatchEmbeddingLayer::new(16, 32, &mut rng);

    let batch_size = 4;
    let input = vec![1.0f32; batch_size * 16];
    let mut output_before = vec![0.0f32; batch_size * 32];
    layer.forward(&input, &mut output_before, batch_size);

    // Backward
    let grad_output = vec![1.0f32; batch_size * 32];
    let mut grad_input = vec![0.0f32; batch_size * 16];
    layer.backward(&input, &grad_output, &mut grad_input, batch_size);

    // Update
    layer.update_parameters(0.01);

    // Forward again - output should differ
    let mut output_after = vec![0.0f32; batch_size * 32];
    layer.forward(&input, &mut output_after, batch_size);

    let changed = output_before
        .iter()
        .zip(output_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "Parameters should change after update");
}
