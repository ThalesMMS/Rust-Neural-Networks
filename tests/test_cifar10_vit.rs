//! Integration tests for the CIFAR-10 Vision Transformer architecture

use rust_neural_networks::layers::{DenseLayer, Layer, PatchEmbeddingLayer, TransformerEncoder};
use rust_neural_networks::optimizers::Adam;
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::positional_encoding::sinusoidal_positional_encoding;
use rust_neural_networks::utils::rng::SimpleRng;

// ViT constants (matching cifar10_vit.rs)
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_CHANNELS: usize = 3;
const PATCH_SIZE: usize = 4;
const GRID: usize = IMG_H / PATCH_SIZE; // 8
const NUM_PATCHES: usize = GRID * GRID; // 64
const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * IMG_CHANNELS; // 48
const D_MODEL: usize = 32; // Smaller for tests
const NUM_HEADS: usize = 4;
const D_FF: usize = 64;
const NUM_BLOCKS: usize = 2;
const NUM_CLASSES: usize = 10;

/// Helper: extract RGB patches from synthetic images (mirrors cifar10_vit.rs logic)
fn extract_patches_rgb(images: &[f32], batch_size: usize, patches: &mut [f32]) {
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    for b in 0..batch_size {
        let img_offset = b * num_inputs;
        for py in 0..GRID {
            for px in 0..GRID {
                let token_idx = py * GRID + px;
                let patch_offset = (b * NUM_PATCHES + token_idx) * PATCH_DIM;
                for dy in 0..PATCH_SIZE {
                    for dx in 0..PATCH_SIZE {
                        let img_y = py * PATCH_SIZE + dy;
                        let img_x = px * PATCH_SIZE + dx;
                        for c in 0..IMG_CHANNELS {
                            let pixel_idx = img_offset + (img_y * IMG_W + img_x) * IMG_CHANNELS + c;
                            let patch_idx =
                                patch_offset + (dy * PATCH_SIZE + dx) * IMG_CHANNELS + c;
                            patches[patch_idx] = images[pixel_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Helper: run full ViT forward pass and return logits
fn vit_forward(
    images: &[f32],
    batch_size: usize,
    patch_embedding: &PatchEmbeddingLayer,
    transformer_encoder: &TransformerEncoder,
    classifier: &DenseLayer,
    pos_encoding: &[f32],
) -> Vec<f32> {
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    assert_eq!(images.len(), batch_size * num_inputs);

    let mut patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut pooled = vec![0.0f32; batch_size * D_MODEL];
    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];

    // Extract patches
    extract_patches_rgb(images, batch_size, &mut patches);

    // Patch embedding
    patch_embedding.forward(&patches, &mut patch_embeds, batch_size * NUM_PATCHES);

    // ReLU
    relu_inplace(&mut patch_embeds);

    // Positional encoding
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            let offset = (b * NUM_PATCHES + t) * D_MODEL;
            let pos_offset = t * D_MODEL;
            for d in 0..D_MODEL {
                patch_embeds[offset + d] += pos_encoding[pos_offset + d];
            }
        }
    }

    // Transformer encoder
    transformer_encoder.forward(&patch_embeds, &mut transformer_out, batch_size);

    // Mean pooling
    for b in 0..batch_size {
        for d in 0..D_MODEL {
            let mut sum = 0.0;
            for t in 0..NUM_PATCHES {
                sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
            }
            pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
        }
    }

    // Classifier
    classifier.forward(&pooled, &mut logits, batch_size);

    logits
}

#[test]
fn test_vit_forward_pass() {
    let mut rng = SimpleRng::new(42);
    let patch_embedding = PatchEmbeddingLayer::new(PATCH_DIM, D_MODEL, &mut rng);
    let transformer_encoder =
        TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, &mut rng);
    let classifier = DenseLayer::new(D_MODEL, NUM_CLASSES, &mut rng);
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    let batch_size = 4;
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    let images = vec![0.5f32; batch_size * num_inputs];

    let logits = vit_forward(
        &images,
        batch_size,
        &patch_embedding,
        &transformer_encoder,
        &classifier,
        &pos_encoding,
    );

    // Check output shape
    assert_eq!(logits.len(), batch_size * NUM_CLASSES);

    // Check no NaN or Inf
    assert!(
        logits.iter().all(|&x| x.is_finite()),
        "Logits contain NaN or Inf"
    );
}

#[test]
fn test_vit_backward_pass() {
    let mut rng = SimpleRng::new(42);
    let patch_embedding = PatchEmbeddingLayer::new(PATCH_DIM, D_MODEL, &mut rng);
    let transformer_encoder =
        TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, &mut rng);
    let classifier = DenseLayer::new(D_MODEL, NUM_CLASSES, &mut rng);
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    let batch_size = 2;
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    let images = vec![0.3f32; batch_size * num_inputs];

    // Forward pass
    let mut patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut pooled = vec![0.0f32; batch_size * D_MODEL];
    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];
    let mut probs = vec![0.0f32; batch_size * NUM_CLASSES];

    extract_patches_rgb(&images, batch_size, &mut patches);
    patch_embedding.forward(&patches, &mut patch_embeds, batch_size * NUM_PATCHES);
    relu_inplace(&mut patch_embeds);
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            let offset = (b * NUM_PATCHES + t) * D_MODEL;
            let pos_offset = t * D_MODEL;
            for d in 0..D_MODEL {
                patch_embeds[offset + d] += pos_encoding[pos_offset + d];
            }
        }
    }
    transformer_encoder.forward(&patch_embeds, &mut transformer_out, batch_size);
    for b in 0..batch_size {
        for d in 0..D_MODEL {
            let mut sum = 0.0;
            for t in 0..NUM_PATCHES {
                sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
            }
            pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
        }
    }
    classifier.forward(&pooled, &mut logits, batch_size);
    probs.copy_from_slice(&logits);
    softmax_rows(&mut probs, batch_size, NUM_CLASSES);

    // Backward pass
    let labels = [3u8, 7];
    let mut grad_logits = vec![0.0f32; batch_size * NUM_CLASSES];
    for i in 0..batch_size {
        for c in 0..NUM_CLASSES {
            grad_logits[i * NUM_CLASSES + c] = probs[i * NUM_CLASSES + c];
        }
        grad_logits[i * NUM_CLASSES + labels[i] as usize] -= 1.0;
    }
    for val in &mut grad_logits {
        *val /= batch_size as f32;
    }

    let mut grad_pooled = vec![0.0f32; batch_size * D_MODEL];
    classifier.backward(&pooled, &grad_logits, &mut grad_pooled, batch_size);

    let mut grad_transformer = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            for d in 0..D_MODEL {
                grad_transformer[(b * NUM_PATCHES + t) * D_MODEL + d] =
                    grad_pooled[b * D_MODEL + d] / NUM_PATCHES as f32;
            }
        }
    }

    let mut grad_patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    transformer_encoder.backward(
        &patch_embeds,
        &grad_transformer,
        &mut grad_patch_embeds,
        batch_size,
    );

    let mut grad_patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    patch_embedding.backward(
        &patches,
        &grad_patch_embeds,
        &mut grad_patches,
        batch_size * NUM_PATCHES,
    );

    // Verify gradients are finite and non-zero
    assert!(
        grad_patches.iter().all(|&x| x.is_finite()),
        "Patch gradients contain NaN/Inf"
    );
    assert!(
        grad_patches.iter().any(|&x| x.abs() > 1e-10),
        "Patch gradients are all zero"
    );
}

#[test]
fn test_vit_parameter_update() {
    let mut rng = SimpleRng::new(42);
    let mut patch_embedding = PatchEmbeddingLayer::new(PATCH_DIM, D_MODEL, &mut rng);
    let mut transformer_encoder =
        TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, &mut rng);
    let mut classifier = DenseLayer::new(D_MODEL, NUM_CLASSES, &mut rng);
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    let batch_size = 2;
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    let images = vec![0.3f32; batch_size * num_inputs];
    let labels = [3u8, 7];

    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Compute loss twice: before and after one update
    let compute_loss = |patch_emb: &PatchEmbeddingLayer,
                        trans_enc: &TransformerEncoder,
                        cls: &DenseLayer|
     -> f32 {
        let logits = vit_forward(
            &images,
            batch_size,
            patch_emb,
            trans_enc,
            cls,
            &pos_encoding,
        );
        let mut probs = logits.clone();
        softmax_rows(&mut probs, batch_size, NUM_CLASSES);

        let mut loss = 0.0f32;
        for i in 0..batch_size {
            let prob = probs[i * NUM_CLASSES + labels[i] as usize].max(1e-7);
            loss -= prob.ln();
        }
        loss / batch_size as f32
    };

    let loss_before = compute_loss(&patch_embedding, &transformer_encoder, &classifier);

    // Forward pass (for backward)
    let mut patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut pooled = vec![0.0f32; batch_size * D_MODEL];
    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];
    let mut probs = vec![0.0f32; batch_size * NUM_CLASSES];

    extract_patches_rgb(&images, batch_size, &mut patches);
    patch_embedding.forward(&patches, &mut patch_embeds, batch_size * NUM_PATCHES);
    relu_inplace(&mut patch_embeds);
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            let offset = (b * NUM_PATCHES + t) * D_MODEL;
            let pos_offset = t * D_MODEL;
            for d in 0..D_MODEL {
                patch_embeds[offset + d] += pos_encoding[pos_offset + d];
            }
        }
    }
    transformer_encoder.forward(&patch_embeds, &mut transformer_out, batch_size);
    for b in 0..batch_size {
        for d in 0..D_MODEL {
            let mut sum = 0.0;
            for t in 0..NUM_PATCHES {
                sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
            }
            pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
        }
    }
    classifier.forward(&pooled, &mut logits, batch_size);
    probs.copy_from_slice(&logits);
    softmax_rows(&mut probs, batch_size, NUM_CLASSES);

    // Backward
    let mut grad_logits = vec![0.0f32; batch_size * NUM_CLASSES];
    for i in 0..batch_size {
        for c in 0..NUM_CLASSES {
            grad_logits[i * NUM_CLASSES + c] = probs[i * NUM_CLASSES + c];
        }
        grad_logits[i * NUM_CLASSES + labels[i] as usize] -= 1.0;
    }
    for val in &mut grad_logits {
        *val /= batch_size as f32;
    }
    let mut grad_pooled = vec![0.0f32; batch_size * D_MODEL];
    classifier.backward(&pooled, &grad_logits, &mut grad_pooled, batch_size);

    let mut grad_transformer = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            for d in 0..D_MODEL {
                grad_transformer[(b * NUM_PATCHES + t) * D_MODEL + d] =
                    grad_pooled[b * D_MODEL + d] / NUM_PATCHES as f32;
            }
        }
    }
    let mut grad_patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    transformer_encoder.backward(
        &patch_embeds,
        &grad_transformer,
        &mut grad_patch_embeds,
        batch_size,
    );
    let mut grad_patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    patch_embedding.backward(
        &patches,
        &grad_patch_embeds,
        &mut grad_patches,
        batch_size * NUM_PATCHES,
    );

    // Update
    patch_embedding.update_with_optimizer(&mut optimizer);
    transformer_encoder.update_with_optimizer(&mut optimizer);
    classifier.update_with_optimizer(&mut optimizer);

    let loss_after = compute_loss(&patch_embedding, &transformer_encoder, &classifier);

    assert!(
        loss_after < loss_before,
        "Loss should decrease after update: before={}, after={}",
        loss_before,
        loss_after
    );
}

#[test]
fn test_vit_mean_pool() {
    // Test that mean pooling correctly averages over tokens
    let batch_size = 2;
    let transformer_out = vec![1.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut pooled = vec![0.0f32; batch_size * D_MODEL];

    for b in 0..batch_size {
        for d in 0..D_MODEL {
            let mut sum = 0.0;
            for t in 0..NUM_PATCHES {
                sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
            }
            pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
        }
    }

    // All ones -> mean should be 1.0
    for &val in &pooled {
        assert!(
            (val - 1.0).abs() < 1e-6,
            "Mean pool of all-ones should be 1.0, got {}",
            val
        );
    }

    // Test with varying values
    let mut varying_out = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    for b in 0..batch_size {
        for t in 0..NUM_PATCHES {
            for d in 0..D_MODEL {
                varying_out[(b * NUM_PATCHES + t) * D_MODEL + d] = t as f32;
            }
        }
    }

    for b in 0..batch_size {
        for d in 0..D_MODEL {
            let mut sum = 0.0;
            for t in 0..NUM_PATCHES {
                sum += varying_out[(b * NUM_PATCHES + t) * D_MODEL + d];
            }
            pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
        }
    }

    // Mean of 0..63 = 31.5
    let expected_mean = (0..NUM_PATCHES).sum::<usize>() as f32 / NUM_PATCHES as f32;
    for &val in &pooled {
        assert!(
            (val - expected_mean).abs() < 1e-4,
            "Expected mean {}, got {}",
            expected_mean,
            val
        );
    }
}

#[test]
fn test_sinusoidal_encoding_applied() {
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    let mut tokens = vec![0.0f32; NUM_PATCHES * D_MODEL];
    let tokens_before = tokens.clone();

    // Add positional encoding
    for t in 0..NUM_PATCHES {
        let offset = t * D_MODEL;
        let pos_offset = t * D_MODEL;
        for d in 0..D_MODEL {
            tokens[offset + d] += pos_encoding[pos_offset + d];
        }
    }

    // Tokens should be modified
    assert_ne!(
        tokens, tokens_before,
        "Positional encoding should change tokens"
    );

    // All values should be finite
    assert!(tokens.iter().all(|&x| x.is_finite()));

    // Different positions should have different encodings
    let pos0 = &tokens[0..D_MODEL];
    let pos1 = &tokens[D_MODEL..2 * D_MODEL];
    assert_ne!(
        pos0, pos1,
        "Different positions should have different encodings"
    );
}

#[test]
fn test_vit_attention_weights_accessible() {
    let mut rng = SimpleRng::new(42);
    let patch_embedding = PatchEmbeddingLayer::new(PATCH_DIM, D_MODEL, &mut rng);
    let transformer_encoder =
        TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, &mut rng);
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    let batch_size = 1;
    let num_inputs = IMG_H * IMG_W * IMG_CHANNELS;
    let images = vec![0.5f32; batch_size * num_inputs];

    let mut patches = vec![0.0f32; batch_size * NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; batch_size * NUM_PATCHES * D_MODEL];

    extract_patches_rgb(&images, batch_size, &mut patches);
    patch_embedding.forward(&patches, &mut patch_embeds, batch_size * NUM_PATCHES);
    relu_inplace(&mut patch_embeds);

    for t in 0..NUM_PATCHES {
        let offset = t * D_MODEL;
        for d in 0..D_MODEL {
            patch_embeds[offset + d] += pos_encoding[t * D_MODEL + d];
        }
    }

    transformer_encoder.forward(&patch_embeds, &mut transformer_out, batch_size);

    // Access attention weights from first block
    let blocks = transformer_encoder.blocks();
    assert!(!blocks.is_empty(), "Transformer should have blocks");

    let attn_weights = blocks[0].attention_layer().get_attention_weights();

    // Expected shape: [batch_size * num_heads * seq_len * seq_len]
    let expected_len = batch_size * NUM_HEADS * NUM_PATCHES * NUM_PATCHES;
    assert_eq!(
        attn_weights.len(),
        expected_len,
        "Attention weights shape mismatch: got {}, expected {}",
        attn_weights.len(),
        expected_len
    );

    // All attention weights should be non-negative (after softmax)
    assert!(
        attn_weights.iter().all(|&x| x >= 0.0),
        "Attention weights should be non-negative"
    );

    // Each row should sum to approximately 1.0 (softmax output)
    for head_row in 0..(batch_size * NUM_HEADS * NUM_PATCHES) {
        let row_start = head_row * NUM_PATCHES;
        let row_sum: f32 = attn_weights[row_start..row_start + NUM_PATCHES]
            .iter()
            .sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "Attention row {} sum = {}, expected ~1.0",
            head_row,
            row_sum
        );
    }
}
