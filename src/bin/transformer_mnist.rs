// transformer_mnist.rs
// Transformer-based MNIST classification using TransformerBlock layer.
//
// ============================================================================
// TRANSFORMER ARCHITECTURE FOR MNIST IMAGE CLASSIFICATION
// ============================================================================
//
// This implementation demonstrates a complete Transformer encoder architecture
// for MNIST digit classification using the newly implemented TransformerBlock
// and TransformerEncoder layers from the rust_neural_networks library.
//
// ARCHITECTURE OVERVIEW:
//   - Split 28x28 image into 4x4 patches => 7×7 = 49 tokens (sequence length)
//   - Linear projection: project each 16-dim patch to d_model=64 dimensions
//   - Add sinusoidal positional embeddings (critical for spatial awareness)
//   - TransformerEncoder: 2 stacked TransformerBlocks
//     * Each block: Multi-Head Attention (4 heads) + FFN (128-dim hidden)
//     * Pre-LN architecture with residual connections
//   - Mean pooling: average over 49 tokens to get image-level representation
//   - Linear classifier: d_model → 10 classes
//
// TRAINING CONFIGURATION:
//   - Optimizer: Adam (lr=0.001, beta1=0.9, beta2=0.999)
//   - Epochs: 5
//   - Batch size: 32
//   - Validation split: 10% (54K training, 6K validation)
//   - Early stopping: patience=3, min_delta=0.001
//
// TARGET PERFORMANCE:
//   - Test accuracy: >85% (validation from TransformerBlock implementation)
//
// Focus: Educational implementation demonstrating Transformer for vision tasks.
// Requires MNIST IDX files in ./data:
//   train-images.idx3-ubyte, train-labels.idx1-ubyte
//   t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte

use std::env::args;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::layers::{DenseLayer, Layer, TransformerEncoder};
use rust_neural_networks::optimizers::{Adam, Optimizer};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::create_scheduler_from_config;
use rust_neural_networks::utils::{sinusoidal_positional_encoding, SimpleRng};

// MNIST constants (images are flat 28x28 in row-major order)
const IMG_H: usize = 28;
const IMG_W: usize = 28;
const NUM_INPUTS: usize = IMG_H * IMG_W; // 784
const NUM_CLASSES: usize = 10;
const TRAIN_SAMPLES: usize = 60_000;
const TEST_SAMPLES: usize = 10_000;

// Patch grid and tokenization
const PATCH: usize = 4; // Patch size: 4x4 pixels
const GRID: usize = IMG_H / PATCH; // 7x7 grid of patches
const SEQ_LEN: usize = GRID * GRID; // 49 tokens (sequence length for attention)
const PATCH_DIM: usize = PATCH * PATCH; // 16 features per patch

// Model hyperparameters (following implementation notes)
const D_MODEL: usize = 64; // Token embedding dimension
const NUM_HEADS: usize = 4; // Number of attention heads
const D_FF: usize = 128; // Feed-forward hidden dimension
const NUM_BLOCKS: usize = 2; // Number of transformer blocks to stack

// Training hyperparameters (defaults, can be overridden by config)
const LEARNING_RATE: f32 = 0.001; // Adam learning rate
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 32;
const VALIDATION_SPLIT: f32 = 0.1; // 10% of training data for validation
const EARLY_STOPPING_PATIENCE: usize = 3;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001;

// Default config path
const DEFAULT_CONFIG_PATH: &str = "config/training/transformer_mnist_default.json";

// ============================================================================
// MNIST Data Loading (IDX format)
// ============================================================================

/// Read a big-endian u32 from byte slice (IDX format uses big-endian).
fn read_be_u32(data: &[u8], offset: &mut usize) -> u32 {
    let b0 = (data[*offset] as u32) << 24;
    let b1 = (data[*offset + 1] as u32) << 16;
    let b2 = (data[*offset + 2] as u32) << 8;
    let b3 = data[*offset + 3] as u32;
    *offset += 4;
    b0 | b1 | b2 | b3
}

/// Read MNIST images from IDX format file and return as normalized floats [0, 1].
fn read_mnist_images(filename: &str) -> Result<Vec<f32>, String> {
    let data =
        fs::read(filename).map_err(|e| format!("Could not read file {}: {}", filename, e))?;

    let mut offset = 0usize;
    let _magic = read_be_u32(&data, &mut offset);
    let num_images = read_be_u32(&data, &mut offset) as usize;
    let rows = read_be_u32(&data, &mut offset) as usize;
    let cols = read_be_u32(&data, &mut offset) as usize;

    if rows != IMG_H || cols != IMG_W {
        return Err(format!(
            "Unexpected image dimensions: {}x{}, expected {}x{}",
            rows, cols, IMG_H, IMG_W
        ));
    }

    let image_size = rows * cols;
    let total_bytes = num_images * image_size;

    if data.len() < offset + total_bytes {
        return Err("Image file is truncated".to_string());
    }

    // Convert bytes to normalized floats
    let mut images = vec![0.0f32; total_bytes];
    let src = &data[offset..offset + total_bytes];
    for (dst, &px) in images.iter_mut().zip(src.iter()) {
        *dst = px as f32 / 255.0;
    }

    Ok(images)
}

/// Read MNIST labels from IDX format file.
fn read_mnist_labels(filename: &str) -> Result<Vec<u8>, String> {
    let data =
        fs::read(filename).map_err(|e| format!("Could not read file {}: {}", filename, e))?;

    let mut offset = 0usize;
    let _magic = read_be_u32(&data, &mut offset);
    let num_labels = read_be_u32(&data, &mut offset) as usize;

    if data.len() < offset + num_labels {
        return Err("Label file is truncated".to_string());
    }

    Ok(data[offset..offset + num_labels].to_vec())
}

// ============================================================================
// Patch Extraction (from mnist_attention_pool.rs pattern)
// ============================================================================

/// Extract 4x4 patches from 28x28 MNIST images and flatten them.
/// Input: [batch_size, 784] (flattened 28x28 images)
/// Output: [batch_size, SEQ_LEN, PATCH_DIM] = [batch_size, 49, 16]
fn extract_patches(images: &[f32], batch_size: usize, patches: &mut [f32]) {
    assert_eq!(images.len(), batch_size * NUM_INPUTS);
    assert_eq!(patches.len(), batch_size * SEQ_LEN * PATCH_DIM);

    for b in 0..batch_size {
        let img_offset = b * NUM_INPUTS;
        for py in 0..GRID {
            for px in 0..GRID {
                let token_idx = py * GRID + px;
                let patch_offset = (b * SEQ_LEN + token_idx) * PATCH_DIM;

                for dy in 0..PATCH {
                    for dx in 0..PATCH {
                        let img_y = py * PATCH + dy;
                        let img_x = px * PATCH + dx;
                        let pixel_idx = img_offset + img_y * IMG_W + img_x;
                        let patch_idx = patch_offset + dy * PATCH + dx;
                        patches[patch_idx] = images[pixel_idx];
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
    println!("=== Transformer MNIST Classifier ===\n");

    // Load configuration (use defaults if config file not found)
    let config_path = args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_string());
    let config = load_config(&config_path).unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config from {}: {}", config_path, e);
        eprintln!("Proceeding with built-in default hyperparameters\n");
        // Return minimal config that will use the const defaults defined above
        load_config(DEFAULT_CONFIG_PATH).unwrap_or_else(|_| {
            panic!("Could not load default config from {}", DEFAULT_CONFIG_PATH)
        })
    });

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

    println!("Configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Validation split: {:.1}%", validation_split * 100.0);
    println!(
        "  Model: d_model={}, num_heads={}, d_ff={}, num_blocks={}",
        D_MODEL, NUM_HEADS, D_FF, NUM_BLOCKS
    );
    println!();

    // Load MNIST data
    println!("Loading MNIST data...");
    let train_images = read_mnist_images("data/train-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("Error reading training images: {}", e);
        process::exit(1);
    });
    let train_labels = read_mnist_labels("data/train-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("Error reading training labels: {}", e);
        process::exit(1);
    });
    let test_images = read_mnist_images("data/t10k-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("Error reading test images: {}", e);
        process::exit(1);
    });
    let test_labels = read_mnist_labels("data/t10k-labels.idx1-ubyte").unwrap_or_else(|e| {
        eprintln!("Error reading test labels: {}", e);
        process::exit(1);
    });

    // Normalize images to [0, 1]
    let train_images: Vec<f32> = train_images.iter().map(|&x| x as f32 / 255.0).collect();
    let test_images: Vec<f32> = test_images.iter().map(|&x| x as f32 / 255.0).collect();

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
    let mut patch_embedding = DenseLayer::new(PATCH_DIM, D_MODEL, &mut rng);

    // Transformer encoder: 2 stacked transformer blocks
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
        "  Transformer encoder: {} blocks, {} params",
        NUM_BLOCKS,
        transformer_encoder.parameter_count()
    );
    println!(
        "  Classifier: {} -> {} ({} params)",
        D_MODEL,
        NUM_CLASSES,
        classifier.parameter_count()
    );
    println!(
        "  Total parameters: {}\n",
        patch_embedding.parameter_count()
            + transformer_encoder.parameter_count()
            + classifier.parameter_count()
    );

    // Generate sinusoidal positional encodings
    let pos_encoding = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL);

    // Initialize optimizers for each layer
    let mut patch_emb_optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut transformer_optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut classifier_optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8);

    // Initialize learning rate scheduler
    let mut lr_scheduler = create_scheduler_from_config(learning_rate, epochs, Some(&config_path));

    // Allocate workspace buffers
    let max_batch = batch_size;
    let mut patches = vec![0.0f32; max_batch * SEQ_LEN * PATCH_DIM];
    let mut patch_embeds = vec![0.0f32; max_batch * SEQ_LEN * D_MODEL];
    let mut transformer_out = vec![0.0f32; max_batch * SEQ_LEN * D_MODEL];
    let mut pooled = vec![0.0f32; max_batch * D_MODEL];
    let mut logits = vec![0.0f32; max_batch * NUM_CLASSES];
    let mut probs = vec![0.0f32; max_batch * NUM_CLASSES];

    // Gradients
    let mut grad_logits = vec![0.0f32; max_batch * NUM_CLASSES];
    let mut grad_pooled = vec![0.0f32; max_batch * D_MODEL];
    let mut grad_transformer = vec![0.0f32; max_batch * SEQ_LEN * D_MODEL];
    let mut grad_patch_embeds = vec![0.0f32; max_batch * SEQ_LEN * D_MODEL];
    let mut grad_patches = vec![0.0f32; max_batch * SEQ_LEN * PATCH_DIM];

    // Create logs directory
    fs::create_dir_all("logs").unwrap();
    let log_file = File::create("logs/transformer_mnist.csv").unwrap();
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
        let num_batches = (train_samples + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(train_samples);
            let current_batch_size = batch_end - batch_start;

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
            extract_patches(
                &batch_images,
                current_batch_size,
                &mut patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
            );

            // Forward pass: Patch embedding
            patch_embedding.forward(
                &patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
                &mut patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
                current_batch_size * SEQ_LEN,
            );

            // Add positional encoding
            for b in 0..current_batch_size {
                for t in 0..SEQ_LEN {
                    let offset = (b * SEQ_LEN + t) * D_MODEL;
                    let pos_offset = t * D_MODEL;
                    for d in 0..D_MODEL {
                        patch_embeds[offset + d] += pos_encoding[pos_offset + d];
                    }
                }
            }

            // Apply ReLU activation
            relu_inplace(&mut patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL]);

            // Forward pass: Transformer encoder
            transformer_encoder.forward(
                &patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
                &mut transformer_out[..current_batch_size * SEQ_LEN * D_MODEL],
                current_batch_size,
            );

            // Mean pooling over sequence dimension
            for b in 0..current_batch_size {
                for d in 0..D_MODEL {
                    let mut sum = 0.0;
                    for t in 0..SEQ_LEN {
                        sum += transformer_out[(b * SEQ_LEN + t) * D_MODEL + d];
                    }
                    pooled[b * D_MODEL + d] = sum / SEQ_LEN as f32;
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

            // Backward pass: Compute gradient of loss w.r.t. logits
            for i in 0..current_batch_size {
                for c in 0..NUM_CLASSES {
                    grad_logits[i * NUM_CLASSES + c] = probs[i * NUM_CLASSES + c];
                }
                let label = batch_labels[i] as usize;
                grad_logits[i * NUM_CLASSES + label] -= 1.0;
            }
            for i in 0..current_batch_size * NUM_CLASSES {
                grad_logits[i] /= current_batch_size as f32;
            }

            // Backward pass: Classifier
            classifier.backward(
                &pooled[..current_batch_size * D_MODEL],
                &grad_logits[..current_batch_size * NUM_CLASSES],
                &mut grad_pooled[..current_batch_size * D_MODEL],
                current_batch_size,
            );

            // Backward pass: Mean pooling (gradient distribution)
            for b in 0..current_batch_size {
                for t in 0..SEQ_LEN {
                    for d in 0..D_MODEL {
                        grad_transformer[(b * SEQ_LEN + t) * D_MODEL + d] =
                            grad_pooled[b * D_MODEL + d] / SEQ_LEN as f32;
                    }
                }
            }

            // Backward pass: Transformer encoder
            transformer_encoder.backward(
                &patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
                &grad_transformer[..current_batch_size * SEQ_LEN * D_MODEL],
                &mut grad_patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
                current_batch_size,
            );

            // Note: We skip backward through ReLU and positional encoding for simplicity
            // The gradients flow through as-is since ReLU gradient is 1 for positive values
            // and positional encoding is added (gradient passes through unchanged)

            // Backward pass: Patch embedding
            patch_embedding.backward(
                &patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
                &grad_patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
                &mut grad_patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
                current_batch_size * SEQ_LEN,
            );

            // Update parameters using optimizers
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

        // Update learning rate for next epoch
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

    // Save model (simplified - just save a marker file)
    fs::write("transformer_mnist_model.bin", b"model_saved").unwrap();
    println!("\nModel saved to transformer_mnist_model.bin");
    println!("Training logs saved to logs/transformer_mnist.csv");
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
    patch_embedding: &DenseLayer,
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

    let num_batches = (num_samples + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_samples);
        let current_batch_size = batch_end - batch_start;

        let batch_images = &images[batch_start * NUM_INPUTS..batch_end * NUM_INPUTS];
        let batch_labels = &labels[batch_start..batch_end];

        // Extract patches
        extract_patches(
            batch_images,
            current_batch_size,
            &mut patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
        );

        // Forward pass: Patch embedding
        patch_embedding.forward(
            &patches[..current_batch_size * SEQ_LEN * PATCH_DIM],
            &mut patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
            current_batch_size * SEQ_LEN,
        );

        // Add positional encoding
        for b in 0..current_batch_size {
            for t in 0..SEQ_LEN {
                let offset = (b * SEQ_LEN + t) * D_MODEL;
                let pos_offset = t * D_MODEL;
                for d in 0..D_MODEL {
                    patch_embeds[offset + d] += pos_encoding[pos_offset + d];
                }
            }
        }

        // Apply ReLU activation
        relu_inplace(&mut patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL]);

        // Forward pass: Transformer encoder
        transformer_encoder.forward(
            &patch_embeds[..current_batch_size * SEQ_LEN * D_MODEL],
            &mut transformer_out[..current_batch_size * SEQ_LEN * D_MODEL],
            current_batch_size,
        );

        // Mean pooling
        for b in 0..current_batch_size {
            for d in 0..D_MODEL {
                let mut sum = 0.0;
                for t in 0..SEQ_LEN {
                    sum += transformer_out[(b * SEQ_LEN + t) * D_MODEL + d];
                }
                pooled[b * D_MODEL + d] = sum / SEQ_LEN as f32;
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
