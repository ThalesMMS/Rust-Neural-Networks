// cifar10_vit.rs
// Vision Transformer (ViT) for CIFAR-10 classification.
//
// ============================================================================
// VISION TRANSFORMER ARCHITECTURE FOR CIFAR-10 IMAGE CLASSIFICATION
// ============================================================================
//
// This implementation demonstrates a complete Vision Transformer architecture
// for CIFAR-10 RGB image classification using the PatchEmbeddingLayer and
// TransformerEncoder from the rust_neural_networks library.
//
// ARCHITECTURE OVERVIEW:
//   - Split 32x32x3 RGB image into 4x4 patches => 8x8 = 64 tokens
//   - PatchEmbeddingLayer: project each 48-dim patch to d_model=128 dimensions
//   - ReLU activation
//   - Add sinusoidal positional embeddings (64 positions, d_model dims)
//   - TransformerEncoder: 4 stacked TransformerBlocks
//     * Each block: Multi-Head Attention (4 heads) + FFN (256-dim hidden)
//     * Pre-LN architecture with residual connections
//   - Mean pooling: average over 64 tokens to get image-level representation
//   - Linear classifier: d_model -> 10 classes
//
// TRAINING CONFIGURATION:
//   - Optimizer: Adam (lr=0.001, beta1=0.9, beta2=0.999)
//   - Epochs: 20
//   - Batch size: 64
//   - Validation split: 10%
//   - Early stopping: patience=5, min_delta=0.001
//
// Requires CIFAR-10 binary files in ./data/cifar-10-batches-bin/:
//   data_batch_1.bin through data_batch_5.bin, test_batch.bin

use std::env::args;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::cifar10::read_cifar10_batches;
use rust_neural_networks::layers::{DenseLayer, Layer, PatchEmbeddingLayer, TransformerEncoder};
use rust_neural_networks::optimizers::{Adam, Optimizer};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::parse_step_flag;
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::create_scheduler_from_config;
use rust_neural_networks::utils::{sinusoidal_positional_encoding, SimpleRng};

// CIFAR-10 constants (images are 32x32 RGB in pixel-interleaved format)
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_CHANNELS: usize = 3;
const NUM_INPUTS: usize = IMG_H * IMG_W * IMG_CHANNELS; // 3072
const NUM_CLASSES: usize = 10;
const TRAIN_SAMPLES: usize = 50_000;
const TEST_SAMPLES: usize = 10_000;

// Patch grid and tokenization
const PATCH_SIZE: usize = 4; // 4x4 pixel patches
const GRID: usize = IMG_H / PATCH_SIZE; // 8x8 grid of patches
const NUM_PATCHES: usize = GRID * GRID; // 64 tokens
const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * IMG_CHANNELS; // 48 features per patch

// Model hyperparameters
const D_MODEL: usize = 128; // Token embedding dimension
const NUM_HEADS: usize = 4; // Number of attention heads
const D_FF: usize = 256; // Feed-forward hidden dimension
const NUM_BLOCKS: usize = 4; // Number of transformer blocks

// Training hyperparameters (defaults, can be overridden by config)
const LEARNING_RATE: f32 = 0.001;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 64;
const VALIDATION_SPLIT: f32 = 0.1;
const EARLY_STOPPING_PATIENCE: usize = 5;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001;

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/cifar10_vit_default.json";

// ============================================================================
// Patch Extraction (RGB pixel-interleaved format)
// ============================================================================

/// Extract 4x4x3 patches from 32x32x3 CIFAR-10 images.
///
/// Input format: pixel-interleaved RGB (R,G,B for each pixel consecutively).
/// For a 32x32 image, layout is [row][col][channel].
/// Each 4x4 patch extracts 4*4*3 = 48 values.
///
/// Input: [batch_size, IMG_H * IMG_W * IMG_CHANNELS]
/// Output: [batch_size, NUM_PATCHES, PATCH_DIM] = [batch_size, 64, 48]
fn extract_patches_rgb(images: &[f32], batch_size: usize, patches: &mut [f32]) {
    assert_eq!(images.len(), batch_size * NUM_INPUTS);
    assert_eq!(patches.len(), batch_size * NUM_PATCHES * PATCH_DIM);

    for b in 0..batch_size {
        let img_offset = b * NUM_INPUTS;
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

// ============================================================================
// Training Loop
// ============================================================================

fn main() {
    println!("=== Vision Transformer (ViT) — CIFAR-10 Classifier ===\n");

    // Parse command-line arguments
    let args_vec: Vec<String> = args().collect();

    // Load configuration (first argument after program name)
    let config_path = args_vec
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_CONFIG_PATH);
    let config = load_config(config_path).unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config from {}: {}", config_path, e);
        eprintln!("Proceeding with built-in default hyperparameters\n");
        load_config(DEFAULT_CONFIG_PATH).unwrap_or_else(|_| {
            panic!("Could not load default config from {}", DEFAULT_CONFIG_PATH)
        })
    });

    // Parse step-through mode flag (second argument after program name)
    let step_mode = parse_step_flag(&args_vec);
    let mut debugger = StepDebugger::new(step_mode);

    // Extract hyperparameters from config
    let learning_rate = config.learning_rate.unwrap_or(LEARNING_RATE);
    let epochs = config.epochs.unwrap_or(EPOCHS);
    let batch_size = config.batch_size.unwrap_or(BATCH_SIZE);
    let validation_split = config.validation_split.unwrap_or(VALIDATION_SPLIT);
    let early_stopping_patience = config
        .early_stopping_patience
        .unwrap_or(EARLY_STOPPING_PATIENCE);
    let early_stopping_min_delta = config
        .early_stopping_min_delta
        .unwrap_or(EARLY_STOPPING_MIN_DELTA);

    // Extract optimizer config
    let beta1 = config.adam_beta1.unwrap_or(0.9);
    let beta2 = config.adam_beta2.unwrap_or(0.999);
    let epsilon = config.adam_epsilon.unwrap_or(1e-8);

    println!("Configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Validation split: {:.1}%", validation_split * 100.0);
    println!(
        "  Model: d_model={}, heads={}, d_ff={}, blocks={}, patch={}x{}",
        D_MODEL, NUM_HEADS, D_FF, NUM_BLOCKS, PATCH_SIZE, PATCH_SIZE
    );
    println!(
        "  Patches: {}x{} = {} tokens, patch_dim={}",
        GRID, GRID, NUM_PATCHES, PATCH_DIM
    );
    println!();

    // Load CIFAR-10 data
    println!("Loading CIFAR-10 data...");
    let train_files: Vec<&str> = vec![
        "data/cifar-10-batches-bin/data_batch_1.bin",
        "data/cifar-10-batches-bin/data_batch_2.bin",
        "data/cifar-10-batches-bin/data_batch_3.bin",
        "data/cifar-10-batches-bin/data_batch_4.bin",
        "data/cifar-10-batches-bin/data_batch_5.bin",
    ];
    let (train_images, train_labels) = read_cifar10_batches(&train_files).unwrap_or_else(|e| {
        eprintln!("Error reading CIFAR-10 training data: {}", e);
        process::exit(1);
    });

    let (test_images, test_labels) =
        read_cifar10_batches(&["data/cifar-10-batches-bin/test_batch.bin"]).unwrap_or_else(|e| {
            eprintln!("Error reading CIFAR-10 test data: {}", e);
            process::exit(1);
        });

    // Split training data into train and validation sets
    let val_samples = (TRAIN_SAMPLES as f32 * validation_split) as usize;
    let train_samples = TRAIN_SAMPLES - val_samples;

    println!("Training samples: {}", train_samples);
    println!("Validation samples: {}", val_samples);
    println!("Test samples: {}\n", TEST_SAMPLES);

    // Initialize RNG
    let mut rng = SimpleRng::new(42);

    // Initialize layers
    println!("Initializing model layers...");

    // Patch embedding: Linear projection from PATCH_DIM to D_MODEL
    let mut patch_embedding = PatchEmbeddingLayer::new(PATCH_DIM, D_MODEL, &mut rng);

    // Transformer encoder: stacked transformer blocks
    let mut transformer_encoder =
        TransformerEncoder::new(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, &mut rng);

    // Classifier: Linear projection from D_MODEL to NUM_CLASSES
    let mut classifier = DenseLayer::new(D_MODEL, NUM_CLASSES, &mut rng);

    println!("Model architecture:");
    println!(
        "  Patch embedding: {} -> {} ({} params)",
        PATCH_DIM,
        D_MODEL,
        patch_embedding.parameter_count()
    );
    println!(
        "  Transformer encoder: {} blocks ({} params)",
        NUM_BLOCKS,
        transformer_encoder.parameter_count()
    );
    println!(
        "  Classifier: {} -> {} ({} params)",
        D_MODEL,
        NUM_CLASSES,
        classifier.parameter_count()
    );
    let total_params = patch_embedding.parameter_count()
        + transformer_encoder.parameter_count()
        + classifier.parameter_count();
    println!("  Total parameters: {}\n", total_params);

    // Generate sinusoidal positional encodings
    let pos_encoding = sinusoidal_positional_encoding(NUM_PATCHES, D_MODEL);

    // Initialize optimizers
    let mut patch_emb_optimizer = Adam::new(learning_rate, beta1, beta2, epsilon);
    let mut transformer_optimizer = Adam::new(learning_rate, beta1, beta2, epsilon);
    let mut classifier_optimizer = Adam::new(learning_rate, beta1, beta2, epsilon);

    // Initialize learning rate scheduler
    let mut lr_scheduler = create_scheduler_from_config(learning_rate, epochs, Some(&config_path));

    // Allocate workspace buffers
    let max_batch = batch_size;
    let mut patches = vec![0.0f32; max_batch * NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; max_batch * NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; max_batch * NUM_PATCHES * D_MODEL];
    let mut pooled = vec![0.0f32; max_batch * D_MODEL];
    let mut logits = vec![0.0f32; max_batch * NUM_CLASSES];
    let mut probs = vec![0.0f32; max_batch * NUM_CLASSES];

    // Gradients
    let mut grad_logits = vec![0.0f32; max_batch * NUM_CLASSES];
    let mut grad_pooled = vec![0.0f32; max_batch * D_MODEL];
    let mut grad_transformer = vec![0.0f32; max_batch * NUM_PATCHES * D_MODEL];
    let mut grad_patch_embeds = vec![0.0f32; max_batch * NUM_PATCHES * D_MODEL];
    let mut grad_patches = vec![0.0f32; max_batch * NUM_PATCHES * PATCH_DIM];

    // Create logs directory
    fs::create_dir_all("logs").unwrap();
    let log_file = File::create("logs/cifar10_vit_log.csv").unwrap();
    let mut log_writer = BufWriter::new(log_file);
    writeln!(
        log_writer,
        "epoch,train_loss,train_time,val_loss,val_accuracy"
    )
    .unwrap();

    // Training state
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;

    // Training loop
    println!("Starting training...\n");
    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        // Step-through mode: notify epoch start
        debugger.on_epoch_start(epoch + 1);

        // Get current learning rate from scheduler and update optimizers
        let current_lr = lr_scheduler.get_lr();
        patch_emb_optimizer.set_learning_rate(current_lr);
        transformer_optimizer.set_learning_rate(current_lr);
        classifier_optimizer.set_learning_rate(current_lr);

        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_samples).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_usize(i + 1);
            indices.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        let num_batches = train_samples.div_ceil(batch_size);

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(train_samples);
            let current_batch_size = batch_end - batch_start;

            // Step-through mode: set batch context
            debugger.set_context(epoch + 1, batch_idx + 1, num_batches, current_batch_size);

            // Gather batch
            let batch_images: Vec<f32> = (batch_start..batch_end)
                .flat_map(|i| {
                    let idx = indices[i];
                    let start = idx * NUM_INPUTS;
                    &train_images[start..start + NUM_INPUTS]
                })
                .copied()
                .collect();

            let batch_labels: Vec<u8> = (batch_start..batch_end)
                .map(|i| train_labels[indices[i]])
                .collect();

            // Extract patches
            extract_patches_rgb(
                &batch_images,
                current_batch_size,
                &mut patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
            );

            // Forward pass: Patch embedding
            patch_embedding.forward(
                &patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
                &mut patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
                current_batch_size * NUM_PATCHES,
            );

            // Apply ReLU activation
            relu_inplace(&mut patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL]);

            // Add positional encoding
            for b in 0..current_batch_size {
                for t in 0..NUM_PATCHES {
                    let offset = (b * NUM_PATCHES + t) * D_MODEL;
                    let pos_offset = t * D_MODEL;
                    for d in 0..D_MODEL {
                        patch_embeds[offset + d] += pos_encoding[pos_offset + d];
                    }
                }
            }

            // Forward pass: Transformer encoder
            transformer_encoder.forward(
                &patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
                &mut transformer_out[..current_batch_size * NUM_PATCHES * D_MODEL],
                current_batch_size,
            );

            // Mean pooling over sequence dimension
            for b in 0..current_batch_size {
                for d in 0..D_MODEL {
                    let mut sum = 0.0;
                    for t in 0..NUM_PATCHES {
                        sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
                    }
                    pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
                }
            }

            // Forward pass: Classifier
            classifier.forward(
                &pooled[..current_batch_size * D_MODEL],
                &mut logits[..current_batch_size * NUM_CLASSES],
                current_batch_size,
            );

            // Softmax
            probs[..current_batch_size * NUM_CLASSES]
                .copy_from_slice(&logits[..current_batch_size * NUM_CLASSES]);
            softmax_rows(
                &mut probs[..current_batch_size * NUM_CLASSES],
                current_batch_size,
                NUM_CLASSES,
            );

            // Compute loss (cross-entropy)
            let mut batch_loss = 0.0;
            for i in 0..current_batch_size {
                let label = batch_labels[i] as usize;
                let prob = probs[i * NUM_CLASSES + label].max(1e-7);
                batch_loss -= prob.ln();
            }
            batch_loss /= current_batch_size as f32;
            epoch_loss += batch_loss;

            // ============================================================
            // Backward pass
            // ============================================================

            // Gradient of loss w.r.t. logits (softmax + cross-entropy)
            for i in 0..current_batch_size {
                for c in 0..NUM_CLASSES {
                    grad_logits[i * NUM_CLASSES + c] = probs[i * NUM_CLASSES + c];
                }
                let label = batch_labels[i] as usize;
                grad_logits[i * NUM_CLASSES + label] -= 1.0;
            }
            for val in &mut grad_logits[..current_batch_size * NUM_CLASSES] {
                *val /= current_batch_size as f32;
            }

            // Backward: Classifier
            classifier.backward(
                &pooled[..current_batch_size * D_MODEL],
                &grad_logits[..current_batch_size * NUM_CLASSES],
                &mut grad_pooled[..current_batch_size * D_MODEL],
                current_batch_size,
            );

            // Backward: Mean pooling (distribute gradient)
            for b in 0..current_batch_size {
                for t in 0..NUM_PATCHES {
                    for d in 0..D_MODEL {
                        grad_transformer[(b * NUM_PATCHES + t) * D_MODEL + d] =
                            grad_pooled[b * D_MODEL + d] / NUM_PATCHES as f32;
                    }
                }
            }

            // Backward: Transformer encoder
            transformer_encoder.backward(
                &patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
                &grad_transformer[..current_batch_size * NUM_PATCHES * D_MODEL],
                &mut grad_patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
                current_batch_size,
            );

            // Backward: Patch embedding
            // (ReLU and positional encoding gradients pass through as-is for positive values)
            patch_embedding.backward(
                &patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
                &grad_patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
                &mut grad_patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
                current_batch_size * NUM_PATCHES,
            );

            // Update parameters
            patch_embedding.update_with_optimizer(&mut patch_emb_optimizer);
            transformer_encoder.update_with_optimizer(&mut transformer_optimizer);
            classifier.update_with_optimizer(&mut classifier_optimizer);
        }

        let avg_train_loss = epoch_loss / num_batches as f32;
        let train_time = epoch_start.elapsed().as_secs_f32();

        // Validation
        let (val_loss, val_accuracy) = evaluate(
            &train_images[train_samples * NUM_INPUTS..],
            &train_labels[train_samples..],
            val_samples,
            batch_size,
            &patch_embedding,
            &transformer_encoder,
            &classifier,
            &pos_encoding,
            &mut patches,
            &mut patch_embeds,
            &mut transformer_out,
            &mut pooled,
            &mut logits,
            &mut probs,
        );

        println!(
            "Epoch {}/{}, Loss: {:.4}, Val Loss: {:.4}, Val Acc: {:.2}%, Time: {:.1}s",
            epoch + 1,
            epochs,
            avg_train_loss,
            val_loss,
            val_accuracy * 100.0,
            train_time
        );

        // Log to CSV
        writeln!(
            log_writer,
            "{},{:.6},{:.3},{:.6},{:.6}",
            epoch + 1,
            avg_train_loss,
            train_time,
            val_loss,
            val_accuracy
        )
        .unwrap();
        log_writer.flush().unwrap();

        // Update learning rate
        lr_scheduler.step();

        // Early stopping check
        if val_loss < best_val_loss - early_stopping_min_delta {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= early_stopping_patience {
                println!(
                    "\nEarly stopping triggered after epoch {} (no improvement for {} epochs)",
                    epoch + 1,
                    early_stopping_patience
                );
                break;
            }
        }
    }

    // Final test evaluation
    println!("\n=== Final Test Evaluation ===");
    let (test_loss, test_accuracy) = evaluate(
        &test_images,
        &test_labels,
        TEST_SAMPLES,
        batch_size,
        &patch_embedding,
        &transformer_encoder,
        &classifier,
        &pos_encoding,
        &mut patches,
        &mut patch_embeds,
        &mut transformer_out,
        &mut pooled,
        &mut logits,
        &mut probs,
    );

    println!("Test Loss: {:.4}", test_loss);
    println!("Test Accuracy: {:.2}%", test_accuracy * 100.0);

    // Save attention maps from first 16 test images
    save_attention_maps(
        &test_images,
        16,
        &patch_embedding,
        &transformer_encoder,
        &pos_encoding,
    );

    println!("\nTraining logs saved to logs/cifar10_vit_log.csv");
}

// ============================================================================
// Evaluation Function
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn evaluate(
    images: &[f32],
    labels: &[u8],
    num_samples: usize,
    batch_size: usize,
    patch_embedding: &PatchEmbeddingLayer,
    transformer_encoder: &TransformerEncoder,
    classifier: &DenseLayer,
    pos_encoding: &[f32],
    patches: &mut [f32],
    patch_embeds: &mut [f32],
    transformer_out: &mut [f32],
    pooled: &mut [f32],
    logits: &mut [f32],
    probs: &mut [f32],
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut correct = 0;

    let num_batches = num_samples.div_ceil(batch_size);

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_samples);
        let current_batch_size = batch_end - batch_start;

        let batch_images = &images[batch_start * NUM_INPUTS..batch_end * NUM_INPUTS];
        let batch_labels = &labels[batch_start..batch_end];

        // Extract patches
        extract_patches_rgb(
            batch_images,
            current_batch_size,
            &mut patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
        );

        // Forward: Patch embedding
        patch_embedding.forward(
            &patches[..current_batch_size * NUM_PATCHES * PATCH_DIM],
            &mut patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
            current_batch_size * NUM_PATCHES,
        );

        // ReLU activation
        relu_inplace(&mut patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL]);

        // Add positional encoding
        for b in 0..current_batch_size {
            for t in 0..NUM_PATCHES {
                let offset = (b * NUM_PATCHES + t) * D_MODEL;
                let pos_offset = t * D_MODEL;
                for d in 0..D_MODEL {
                    patch_embeds[offset + d] += pos_encoding[pos_offset + d];
                }
            }
        }

        // Forward: Transformer encoder
        transformer_encoder.forward(
            &patch_embeds[..current_batch_size * NUM_PATCHES * D_MODEL],
            &mut transformer_out[..current_batch_size * NUM_PATCHES * D_MODEL],
            current_batch_size,
        );

        // Mean pooling
        for b in 0..current_batch_size {
            for d in 0..D_MODEL {
                let mut sum = 0.0;
                for t in 0..NUM_PATCHES {
                    sum += transformer_out[(b * NUM_PATCHES + t) * D_MODEL + d];
                }
                pooled[b * D_MODEL + d] = sum / NUM_PATCHES as f32;
            }
        }

        // Forward: Classifier
        classifier.forward(
            &pooled[..current_batch_size * D_MODEL],
            &mut logits[..current_batch_size * NUM_CLASSES],
            current_batch_size,
        );

        // Softmax
        probs[..current_batch_size * NUM_CLASSES]
            .copy_from_slice(&logits[..current_batch_size * NUM_CLASSES]);
        softmax_rows(
            &mut probs[..current_batch_size * NUM_CLASSES],
            current_batch_size,
            NUM_CLASSES,
        );

        // Compute loss and accuracy
        for i in 0..current_batch_size {
            let label = batch_labels[i] as usize;
            let prob = probs[i * NUM_CLASSES + label].max(1e-7);
            total_loss -= prob.ln();

            let predicted = (0..NUM_CLASSES)
                .max_by(|&a, &b| {
                    probs[i * NUM_CLASSES + a]
                        .partial_cmp(&probs[i * NUM_CLASSES + b])
                        .unwrap()
                })
                .unwrap();
            if predicted == label {
                correct += 1;
            }
        }
    }

    let avg_loss = total_loss / num_samples as f32;
    let accuracy = correct as f32 / num_samples as f32;

    (avg_loss, accuracy)
}

// ============================================================================
// Attention Map Saving
// ============================================================================

/// Save attention weight maps from the first transformer block for visualization.
///
/// Runs a forward pass on the first `num_images` test images and writes the
/// attention weights to `logs/vit_attention_maps.csv`.
fn save_attention_maps(
    images: &[f32],
    num_images: usize,
    patch_embedding: &PatchEmbeddingLayer,
    transformer_encoder: &TransformerEncoder,
    pos_encoding: &[f32],
) {
    let num_images = num_images.min(images.len() / NUM_INPUTS);
    if num_images == 0 {
        return;
    }

    println!("\nSaving attention maps for {} test images...", num_images);

    // Run forward pass one image at a time to extract attention weights
    let mut patches = vec![0.0f32; NUM_PATCHES * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; NUM_PATCHES * D_MODEL];
    let mut transformer_out = vec![0.0f32; NUM_PATCHES * D_MODEL];

    let attn_file = File::create("logs/vit_attention_maps.csv").unwrap();
    let mut attn_writer = BufWriter::new(attn_file);

    // Header
    writeln!(
        attn_writer,
        "# ViT Attention Maps: {} images, {} heads, {} tokens",
        num_images, NUM_HEADS, NUM_PATCHES
    )
    .unwrap();

    for img_idx in 0..num_images {
        let img_data = &images[img_idx * NUM_INPUTS..(img_idx + 1) * NUM_INPUTS];

        // Extract patches
        extract_patches_rgb(img_data, 1, &mut patches);

        // Patch embedding
        patch_embedding.forward(&patches, &mut patch_embeds, NUM_PATCHES);

        // ReLU
        relu_inplace(&mut patch_embeds);

        // Add positional encoding
        for t in 0..NUM_PATCHES {
            let offset = t * D_MODEL;
            let pos_offset = t * D_MODEL;
            for d in 0..D_MODEL {
                patch_embeds[offset + d] += pos_encoding[pos_offset + d];
            }
        }

        // Forward through transformer (to populate attention caches)
        transformer_encoder.forward(&patch_embeds, &mut transformer_out, 1);

        // Extract attention weights from first block
        let blocks = transformer_encoder.blocks();
        if !blocks.is_empty() {
            let attn_weights = blocks[0].attention_layer().get_attention_weights();

            // Write image header
            writeln!(attn_writer, "# Image: {}", img_idx).unwrap();

            // Write flattened attention weights [num_heads * seq_len * seq_len]
            let values: Vec<String> = attn_weights.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(attn_writer, "{}", values.join(",")).unwrap();
        }
    }

    attn_writer.flush().unwrap();
    println!("Attention maps saved to logs/vit_attention_maps.csv");
}
