// resnet_cifar10.rs
// ResNet-18 variant for CIFAR-10 on CPU using shared layers/utilities.
//
// Architecture:
//   Stem: Conv(3→16, 3×3, p=1, s=1) → BN → ReLU
//   Stage 1: 2× ResBlock(16→16, stride=1, 32×32)
//   Stage 2: ResBlock(16→32, stride=2, 32→16) + ResBlock(32→32, stride=1, 16×16)
//   Stage 3: ResBlock(32→64, stride=2, 16→8)  + ResBlock(64→64, stride=1, 8×8)
//   Stage 4: ResBlock(64→128, stride=2, 8→4)  + ResBlock(128→128, stride=1, 4×4)
//   Head: GlobalAvgPool(4, 4, 128) → Dense(128→10)
//
// Expected data files:
//   ./data/cifar-10-batches-bin/data_batch_1.bin through data_batch_5.bin
//   ./data/cifar-10-batches-bin/test_batch.bin
//
// Output:
//   - logs/training_loss_resnet_cifar10.csv (epoch,train_loss,time,val_loss,val_accuracy,lr)
//   - resnet_cifar10_model_best.bin  (best checkpoint by validation loss)
//
// Note: educational implementation. ResidualBlocks handle skip connections internally.
// The layer stack is defined in config/architectures/resnet_cifar10.json.

use std::env;
use std::fs;
use std::io::Write;
use std::process;
use std::time::Instant;

use rust_neural_networks::architecture::{build_model, ArchitectureConfig};
use rust_neural_networks::config::{load_config, TrainingConfig};
use rust_neural_networks::data::cifar10::{read_cifar10_batch, read_cifar10_batches};
use rust_neural_networks::layers::{batchnorm::BatchNormLayer, Layer, ResidualBlock};
use rust_neural_networks::optimizers::rmsprop::RMSprop;
use rust_neural_networks::optimizers::{Adam, AdamW, Optimizer, SGD};
use rust_neural_networks::persistence::save_layers_to_file;
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::{
    compute_softmax_cross_entropy, evaluate_batch_accuracy, gather_batch, parse_config_path,
    parse_step_flag, print_training_config, CsvTrainingLogger, EarlyStopping, EarlyStoppingAction,
    TrainingMetrics,
};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use rust_neural_networks::utils::lr_scheduler::create_scheduler_from_config;
use rust_neural_networks::utils::rng::SimpleRng;

// ── CIFAR-10 constants ────────────────────────────────────────────────────────
const IMG_H: usize = 32;
const IMG_W: usize = 32;
const IMG_CHANNELS: usize = 3; // RGB
const NUM_INPUTS: usize = IMG_H * IMG_W * IMG_CHANNELS; // 3072
const NUM_CLASSES: usize = 10;

// ── Training hyperparameter defaults ─────────────────────────────────────────
const LEARNING_RATE: f32 = 0.001;
const EPOCHS: usize = 50;
const BATCH_SIZE: usize = 64;
const VALIDATION_SPLIT: f32 = 0.1;
const EARLY_STOPPING_PATIENCE: usize = 10;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001;

// ── File paths ────────────────────────────────────────────────────────────────
const DEFAULT_CONFIG_PATH: &str = "config/training/resnet_cifar10_default.json";
const ARCHITECTURE_JSON: &str = include_str!("../../config/architectures/resnet_cifar10.json");
const MODEL_CHECKPOINT_PATH: &str = "resnet_cifar10_model_best.bin";
const LOG_PATH: &str = "./logs/training_loss_resnet_cifar10.csv";

// ── Model ─────────────────────────────────────────────────────────────────────

/// ResNet-18 CIFAR-10 variant.
struct ResNet {
    layers: Vec<Box<dyn Layer>>,
    /// Per-layer flag: `true` if ReLU should be applied after that layer's output.
    apply_relu: Vec<bool>,
}

/// Builds the ResNet-18 CIFAR-10 variant from the architecture config.
///
/// [`ResidualBlock`] is a first-class config layer, so the JSON architecture can
/// represent skip connections without manually constructing the layer stack here.
///
/// # Arguments
///
/// * `rng` – Random number generator for weight initialisation.
///
/// # Returns
///
/// A [`ResNet`] ready for forward/backward passes on CIFAR-10-shaped inputs.
///
/// # Examples
///
/// ```ignore
/// let mut rng = SimpleRng::new(42);
/// let model = init_resnet(&mut rng);
/// assert_eq!(model.layers.len(), 12);
/// ```
fn init_resnet(rng: &mut SimpleRng) -> ResNet {
    let architecture: ArchitectureConfig =
        serde_json::from_str(ARCHITECTURE_JSON).expect("built-in ResNet architecture JSON");
    let layers = build_model(&architecture, rng).expect("built-in ResNet architecture config");
    let apply_relu = vec![
        false, // stem conv
        true,  // stem batchnorm
        false, false, false, false, false, false, false, false, false, false,
    ];
    assert_eq!(
        layers.len(),
        apply_relu.len(),
        "ResNet ReLU flags must match architecture layer count"
    );

    // ── Architecture summary ──────────────────────────────────────────────────
    println!("\nResNet-18 CIFAR-10 Architecture:");
    println!("  Total layers: {}", layers.len());
    let total_params: usize = layers.iter().map(|l| l.parameter_count()).sum();
    for (i, (layer, relu)) in layers.iter().zip(apply_relu.iter()).enumerate() {
        println!(
            "  Layer {:2}: in={:6} out={:6} params={:9}{}",
            i + 1,
            layer.input_size(),
            layer.output_size(),
            layer.parameter_count(),
            if *relu { " +ReLU" } else { "" }
        );
    }
    println!("  Total parameters: {}", total_params);
    println!();

    ResNet { layers, apply_relu }
}

// ── Training-mode toggle ──────────────────────────────────────────────────────

/// Propagates training mode to all layers that support it.
///
/// Sets training mode on [`BatchNormLayer`] (stem) and [`ResidualBlock`] layers
/// (which propagate the mode to their internal BatchNorm sub-layers).
///
/// # Arguments
///
/// * `model`    – Mutable reference to the ResNet model.
/// * `training` – `true` for training mode, `false` for inference mode.
fn set_training_mode(model: &mut ResNet, training: bool) {
    for layer in model.layers.iter_mut() {
        let any_layer = layer.as_any_mut();
        if let Some(bn) = any_layer.downcast_mut::<BatchNormLayer>() {
            bn.set_training(training);
        } else if let Some(rb) = any_layer.downcast_mut::<ResidualBlock>() {
            rb.set_training(training);
        }
        // Conv2DLayer, GlobalAvgPoolLayer, DenseLayer have no training-mode behaviour.
    }
}

// ── Activation buffer ─────────────────────────────────────────────────────────

/// Stores intermediate layer outputs for forward/backward passes.
struct LayerActivations {
    data: Vec<Vec<f32>>,
    /// Mirrors `ResNet::apply_relu`; populated during forward pass.
    apply_relu: Vec<bool>,
}

impl LayerActivations {
    fn new(num_layers: usize) -> Self {
        Self {
            data: vec![Vec::new(); num_layers],
            apply_relu: vec![false; num_layers],
        }
    }
}

// ── Forward pass ──────────────────────────────────────────────────────────────

/// Runs a forward pass through all layers, applying ReLU where required.
///
/// [`ResidualBlock`] layers handle their own internal activations; only the
/// stem BatchNorm gets an external ReLU (indicated by `model.apply_relu[i]`).
///
/// # Arguments
///
/// * `model`      – ResNet containing all layers and their ReLU flags.
/// * `batch_size` – Number of samples in the batch.
/// * `input`      – Flat input slice (length = `batch_size * NUM_INPUTS`).
/// * `activations`– Storage for layer outputs (will be populated).
///
/// # Returns
///
/// Index of the layer whose output is the final model output.
fn forward_pass(
    model: &mut ResNet,
    batch_size: usize,
    input: &[f32],
    activations: &mut LayerActivations,
) -> usize {
    if model.layers.is_empty() {
        return 0;
    }

    // Layer 0: stem Conv2D (no external ReLU).
    {
        let layer = &model.layers[0];
        let out = layer.output_size() * batch_size;
        activations.data[0].resize(out, 0.0);
        layer.forward(input, &mut activations.data[0], batch_size);
        activations.apply_relu[0] = model.apply_relu[0]; // false
    }

    // Layers 1…N-1.
    for i in 1..model.layers.len() {
        let out = model.layers[i].output_size() * batch_size;
        activations.data[i].resize(out, 0.0);

        // Split borrow to satisfy the borrow checker.
        let (prev_data, curr_data) = activations.data.split_at_mut(i);
        let prev_output = &prev_data[i - 1];
        let curr_output = &mut curr_data[0];

        model.layers[i].forward(prev_output, curr_output, batch_size);

        activations.apply_relu[i] = model.apply_relu[i];
        if model.apply_relu[i] {
            relu_inplace(curr_output);
        }
    }

    model.layers.len() - 1
}

// ── Backward pass ─────────────────────────────────────────────────────────────

/// Runs a backward pass through all layers in reverse order.
///
/// Applies the ReLU gradient at layers that had ReLU applied after them
/// (currently only the stem BN, index 1).
///
/// # Arguments
///
/// * `model`        – ResNet containing all layers.
/// * `batch_size`   – Number of samples in the batch.
/// * `input`        – Original model input (needed for layer-0 backward).
/// * `activations`  – Cached layer outputs from the forward pass.
/// * `initial_grad` – Gradient from the loss w.r.t. the final layer output.
/// * `grad_buffer1` – Working gradient buffer (ping).
/// * `grad_buffer2` – Working gradient buffer (pong).
fn backward_pass(
    model: &mut ResNet,
    batch_size: usize,
    input: &[f32],
    activations: &LayerActivations,
    initial_grad: &[f32],
    grad_buffer1: &mut Vec<f32>,
    grad_buffer2: &mut Vec<f32>,
) {
    if model.layers.is_empty() {
        return;
    }

    let num_layers = model.layers.len();

    grad_buffer1.clear();
    grad_buffer1.extend_from_slice(initial_grad);

    for i in (0..num_layers).rev() {
        let layer = &model.layers[i];
        let input_size = layer.input_size() * batch_size;
        grad_buffer2.resize(input_size, 0.0);

        let layer_input = if i == 0 {
            input
        } else {
            &activations.data[i - 1]
        };

        layer.backward(layer_input, grad_buffer1, grad_buffer2, batch_size);

        // Apply ReLU gradient for layers that had ReLU applied after them
        // (the output stored in activations.data[i] is post-ReLU).
        if activations.apply_relu[i] {
            for j in 0..grad_buffer2.len().min(activations.data[i].len()) {
                if activations.data[i][j] <= 0.0 {
                    grad_buffer2[j] = 0.0;
                }
            }
        }

        std::mem::swap(grad_buffer1, grad_buffer2);
    }
}

// ── Accuracy evaluation ────────────────────────────────────────────────────────

/// Computes classification accuracy (%) on the provided dataset.
///
/// Runs in inference mode (BatchNorm uses running statistics) in batches of
/// [`BATCH_SIZE`] and computes argmax accuracy.
///
/// # Examples
///
/// ```ignore
/// let acc = test_accuracy(&mut model, &test_images, &test_labels);
/// assert!(acc >= 0.0 && acc <= 100.0);
/// ```
fn test_accuracy(model: &mut ResNet, images: &[f32], labels: &[u8]) -> f32 {
    let num_samples = labels.len();
    let mut correct = 0usize;

    set_training_mode(model, false);

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let num_layers = model.layers.len();
    let mut activations = LayerActivations::new(num_layers);

    for start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch = (num_samples - start).min(BATCH_SIZE);
        let len = batch * NUM_INPUTS;
        batch_inputs[..len].copy_from_slice(&images[start * NUM_INPUTS..start * NUM_INPUTS + len]);

        let output_idx = forward_pass(model, batch, &batch_inputs, &mut activations);
        let logits = &activations.data[output_idx];

        for b in 0..batch {
            let base = b * NUM_CLASSES;
            let mut best = logits[base];
            let mut arg = 0usize;
            for j in 1..NUM_CLASSES {
                let v = logits[base + j];
                if v > best {
                    best = v;
                    arg = j;
                }
            }
            if arg as u8 == labels[start + b] {
                correct += 1;
            }
        }
    }

    100.0 * (correct as f32) / (num_samples as f32)
}

// ── Model serialisation ───────────────────────────────────────────────────────

/// Saves a model checkpoint to a binary file.
///
/// Uses the shared layer-stack checkpoint format. ResidualBlocks are still saved
/// as architecture metadata only because their internal weights are not publicly
/// accessible via the current library API.
fn save_model(model: &ResNet, filename: &str) {
    save_layers_to_file(filename, &model.layers).expect("Failed to save model layers");
    println!("Model checkpoint saved to: {}", filename);
}

// ── Optimizer factory ─────────────────────────────────────────────────────────

/// Creates an optimizer from the training configuration.
///
/// Supported types: `"sgd"`, `"adam"`, `"adamw"` (default), `"rmsprop"`.
fn create_optimizer(config: &TrainingConfig, lr: f32) -> Box<dyn Optimizer> {
    let optimizer_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    let beta1 = config.adam_beta1.unwrap_or(0.9);
    let beta2 = config.adam_beta2.unwrap_or(0.999);
    let epsilon = config.adam_epsilon.unwrap_or(1e-8);
    let weight_decay = config.adamw_weight_decay.unwrap_or(0.01);

    match optimizer_type {
        "sgd" => Box::new(SGD::new(lr)),
        "adam" => Box::new(Adam::new(lr, beta1, beta2, epsilon)),
        "adamw" => Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay)),
        "rmsprop" => Box::new(RMSprop::new(lr, 0.9, epsilon)),
        _ => {
            eprintln!(
                "Unknown optimizer type: '{}', defaulting to AdamW",
                optimizer_type
            );
            Box::new(AdamW::new(lr, beta1, beta2, epsilon, weight_decay))
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

/// Trains a ResNet-18 variant on CIFAR-10, with validation, early stopping, and
/// CSV logging.  The best model checkpoint (by validation loss) is saved to
/// `resnet_cifar10_model_best.bin`.
fn main() {
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);
    let step_mode = parse_step_flag(&args);

    println!("=== ResNet-18 CIFAR-10 Training ===");
    println!("Loading training configuration from: {}", config_path);

    let config = match load_config(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config from '{}': {}", config_path, e);
            eprintln!("Please ensure the config file exists and is valid JSON.");
            process::exit(1);
        }
    };

    // Extract hyperparameters (config overrides constants above).
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

    let enable_augmentation = config.enable_augmentation.unwrap_or(false);
    let horizontal_flip_prob = config.horizontal_flip_prob;
    let random_crop_padding = config.random_crop_padding;
    let brightness_jitter = config.brightness_jitter;
    let contrast_jitter = config.contrast_jitter;
    let saturation_jitter = config.saturation_jitter;

    print_training_config(
        &config,
        learning_rate,
        epochs,
        batch_size,
        validation_split,
        early_stopping_patience,
        early_stopping_min_delta,
    );

    // ── Load CIFAR-10 ─────────────────────────────────────────────────────────
    println!("Loading CIFAR-10...");

    let train_filenames = vec![
        "./data/cifar-10-batches-bin/data_batch_1.bin",
        "./data/cifar-10-batches-bin/data_batch_2.bin",
        "./data/cifar-10-batches-bin/data_batch_3.bin",
        "./data/cifar-10-batches-bin/data_batch_4.bin",
        "./data/cifar-10-batches-bin/data_batch_5.bin",
    ];

    let (all_train_images, all_train_labels) = read_cifar10_batches(&train_filenames)
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });
    let (test_images, test_labels) =
        read_cifar10_batch("./data/cifar-10-batches-bin/test_batch.bin").unwrap_or_else(|err| {
            eprintln!("{err}");
            process::exit(1);
        });

    // ── Train/validation split ────────────────────────────────────────────────
    let total_train_samples = all_train_images.len() / NUM_INPUTS;
    let validation_samples = (total_train_samples as f32 * validation_split) as usize;
    let actual_train_samples = total_train_samples - validation_samples;

    let mut rng = SimpleRng::new(42);
    rng.reseed_from_time();

    let mut split_indices: Vec<usize> = (0..total_train_samples).collect();
    rng.shuffle_usize(&mut split_indices);

    let mut train_images = vec![0.0f32; actual_train_samples * NUM_INPUTS];
    let mut train_labels = vec![0u8; actual_train_samples];
    let mut val_images = vec![0.0f32; validation_samples * NUM_INPUTS];
    let mut val_labels = vec![0u8; validation_samples];

    for (i, &src_index) in split_indices.iter().enumerate() {
        let src_start = src_index * NUM_INPUTS;
        if i < validation_samples {
            let dst_start = i * NUM_INPUTS;
            val_images[dst_start..dst_start + NUM_INPUTS]
                .copy_from_slice(&all_train_images[src_start..src_start + NUM_INPUTS]);
            val_labels[i] = all_train_labels[src_index];
        } else {
            let dst_i = i - validation_samples;
            let dst_start = dst_i * NUM_INPUTS;
            train_images[dst_start..dst_start + NUM_INPUTS]
                .copy_from_slice(&all_train_images[src_start..src_start + NUM_INPUTS]);
            train_labels[dst_i] = all_train_labels[src_index];
        }
    }

    let train_n = actual_train_samples;
    let test_n = test_labels.len();
    println!(
        "Data split: {} training, {} validation, {} test samples",
        actual_train_samples, validation_samples, test_n
    );

    // ── Build model ───────────────────────────────────────────────────────────
    let mut model = init_resnet(&mut rng);

    // One optimizer per layer to avoid shared state issues across layers.
    let mut layer_optimizers: Vec<Box<dyn Optimizer>> = (0..model.layers.len())
        .map(|_| create_optimizer(&config, learning_rate))
        .collect();

    let optimizer_type = config.optimizer_type.as_deref().unwrap_or("adamw");
    println!("Optimizer: {}", optimizer_type.to_uppercase());

    let mut scheduler = create_scheduler_from_config(learning_rate, epochs, Some(&config_path));

    // ── Logging ───────────────────────────────────────────────────────────────
    fs::create_dir_all("./logs").ok();
    let mut logger = CsvTrainingLogger::new(LOG_PATH).unwrap_or_else(|_| {
        eprintln!("Could not create {}", LOG_PATH);
        process::exit(1);
    });
    logger.write_header().unwrap_or_else(|_| {
        eprintln!("Could not write CSV header to {}", LOG_PATH);
        process::exit(1);
    });

    // ── Training buffers (reused each batch to avoid allocations) ─────────────
    let mut batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut batch_labels = vec![0u8; batch_size];

    let num_layers = model.layers.len();
    let mut activations = LayerActivations::new(num_layers);
    let mut grad_buffer1 = Vec::new();
    let mut grad_buffer2 = Vec::new();

    let mut logits = vec![0.0f32; batch_size * NUM_CLASSES];
    let mut delta = vec![0.0f32; batch_size * NUM_CLASSES];

    let mut val_batch_inputs = vec![0.0f32; batch_size * NUM_INPUTS];
    let mut val_activations = LayerActivations::new(num_layers);
    let mut val_logits = vec![0.0f32; batch_size * NUM_CLASSES];

    let mut indices: Vec<usize> = (0..train_n).collect();
    let mut early_stopping = EarlyStopping::new(early_stopping_patience, early_stopping_min_delta);

    let mut debugger = StepDebugger::new(step_mode);

    println!(
        "Training ResNet-18 CIFAR-10: epochs={} batch={} lr={}",
        epochs, batch_size, learning_rate
    );

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..epochs {
        debugger.on_epoch_start(epoch + 1);
        let start_time = Instant::now();
        rng.shuffle_usize(&mut indices);
        let current_lr = scheduler.get_lr();

        for opt in layer_optimizers.iter_mut() {
            opt.set_learning_rate(current_lr);
        }

        set_training_mode(&mut model, true);

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(batch_size) {
            let batch = (train_n - batch_start).min(batch_size);
            let batch_idx = batch_start / batch_size;
            let total_batches = train_n.div_ceil(batch_size);

            debugger.set_context(epoch + 1, batch_idx + 1, total_batches, batch);

            gather_batch(
                &train_images,
                &train_labels,
                &indices,
                batch_start,
                batch,
                &mut batch_inputs,
                &mut batch_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                if enable_augmentation {
                    horizontal_flip_prob
                } else {
                    None
                },
                if enable_augmentation {
                    random_crop_padding
                } else {
                    None
                },
                if enable_augmentation {
                    brightness_jitter
                } else {
                    None
                },
                if enable_augmentation {
                    contrast_jitter
                } else {
                    None
                },
                if enable_augmentation {
                    saturation_jitter
                } else {
                    None
                },
                if enable_augmentation {
                    Some(&mut rng)
                } else {
                    None
                },
            );

            let output_idx = forward_pass(&mut model, batch, &batch_inputs, &mut activations);

            let logits_slice = &mut activations.data[output_idx];
            logits[..logits_slice.len()].copy_from_slice(logits_slice);

            softmax_rows(&mut logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);
            let batch_loss = compute_softmax_cross_entropy(
                &logits[..batch * NUM_CLASSES],
                &batch_labels,
                batch,
                NUM_CLASSES,
                &mut delta,
                1.0f32,
            );
            total_loss += batch_loss;

            backward_pass(
                &mut model,
                batch,
                &batch_inputs,
                &activations,
                &delta[..batch * NUM_CLASSES],
                &mut grad_buffer1,
                &mut grad_buffer2,
            );

            for (layer, opt) in model.layers.iter_mut().zip(layer_optimizers.iter_mut()) {
                layer.update_with_optimizer(opt.as_mut());
            }

            // Progress indicator every 100 batches.
            let batch_idx = batch_start / batch_size;
            let total_batches = train_n.div_ceil(batch_size);
            if batch_idx % 100 == 0 || batch_idx == total_batches - 1 {
                let progress_pct = (batch_idx as f32 / total_batches as f32) * 100.0;
                print!(
                    "\r  Epoch {}/{}: Batch {}/{} ({:.1}%), Loss: {:.4}",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    total_batches,
                    progress_pct,
                    total_loss / (batch_idx + 1) as f32
                );
                std::io::stdout().flush().unwrap();
            }
        }

        println!(); // newline after progress indicator
        let secs = start_time.elapsed().as_secs_f32();
        let avg_loss = total_loss / train_n as f32;

        // ── Validation ────────────────────────────────────────────────────────
        set_training_mode(&mut model, false);

        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;

        for batch_start in (0..validation_samples).step_by(batch_size) {
            let batch_count = (validation_samples - batch_start).min(batch_size);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_images[input_start..input_start + input_len]);

            let output_idx = forward_pass(
                &mut model,
                batch_count,
                &val_batch_inputs,
                &mut val_activations,
            );

            let logits_slice = &mut val_activations.data[output_idx];
            val_logits[..logits_slice.len()].copy_from_slice(logits_slice);

            softmax_rows(
                &mut val_logits[..batch_count * NUM_CLASSES],
                batch_count,
                NUM_CLASSES,
            );

            let batch_val_labels = &val_labels[batch_start..batch_start + batch_count];
            let (batch_loss, batch_correct) = evaluate_batch_accuracy(
                &val_logits[..batch_count * NUM_CLASSES],
                batch_val_labels,
                batch_count,
                NUM_CLASSES,
            );
            val_total_loss += batch_loss;
            val_correct += batch_correct;
        }

        let val_average_loss = val_total_loss / validation_samples as f32;
        let val_accuracy = val_correct as f32 / validation_samples as f32 * 100.0;

        println!(
            "Epoch {}, Loss: {:.6}, Val Loss: {:.6}, Val Acc: {:.2}%, Time: {:.2}s",
            epoch + 1,
            avg_loss,
            val_average_loss,
            val_accuracy,
            secs
        );

        let metrics = TrainingMetrics {
            train_loss: avg_loss,
            val_loss: val_average_loss,
            val_accuracy,
            train_time: secs,
            learning_rate: current_lr,
        };
        logger.write_epoch(epoch + 1, &metrics).ok();

        // ── Early stopping & checkpointing ────────────────────────────────────
        match early_stopping.check(val_average_loss) {
            EarlyStoppingAction::Improved => {
                save_model(&model, MODEL_CHECKPOINT_PATH);
            }
            EarlyStoppingAction::Stop => {
                println!(
                    "\nEarly stopping triggered! No improvement for {} epochs. \
                     Best validation loss: {:.6}",
                    early_stopping_patience, early_stopping.best_val_loss
                );
                break;
            }
            EarlyStoppingAction::Continue => {}
        }

        scheduler.step();
    }

    // ── Final test evaluation ─────────────────────────────────────────────────
    println!("Testing...");
    let acc = test_accuracy(&mut model, &test_images, &test_labels);
    println!("Test Accuracy: {:.2}%", acc);
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the ResNet architecture has the expected number of layers.
    #[test]
    fn test_resnet_layer_count() {
        let mut rng = SimpleRng::new(42);
        let model = init_resnet(&mut rng);
        // stem_conv + stem_bn + 8 ResBlocks + GAP + Dense = 12 layers
        assert_eq!(model.layers.len(), 12, "Expected 12 layers in ResNet");
        assert_eq!(
            model.apply_relu.len(),
            12,
            "apply_relu must match layers count"
        );
    }

    /// Verify dimension flow: input size of first layer and output of last layer.
    #[test]
    fn test_resnet_dimensions() {
        let mut rng = SimpleRng::new(99);
        let model = init_resnet(&mut rng);

        // First layer (stem Conv2D) expects CIFAR-10 input.
        assert_eq!(
            model.layers[0].input_size(),
            NUM_INPUTS,
            "Stem Conv2D input_size should match NUM_INPUTS"
        );

        // Last layer (Dense) outputs NUM_CLASSES logits.
        let last = model.layers.len() - 1;
        assert_eq!(
            model.layers[last].output_size(),
            NUM_CLASSES,
            "Dense output should match NUM_CLASSES"
        );
    }

    /// Verify that stem BN (layer 1) has ReLU flag set and all others don't.
    #[test]
    fn test_resnet_relu_flags() {
        let mut rng = SimpleRng::new(7);
        let model = init_resnet(&mut rng);

        // Only stem BN (index 1) should have apply_relu = true.
        for (i, &flag) in model.apply_relu.iter().enumerate() {
            if i == 1 {
                assert!(flag, "Layer 1 (stem BN) should have apply_relu=true");
            } else {
                assert!(!flag, "Layer {} should have apply_relu=false", i);
            }
        }
    }

    /// Verify set_training_mode reaches internal BatchNorm layers.
    #[test]
    fn test_resnet_training_mode() {
        let mut rng = SimpleRng::new(3);
        let mut model = init_resnet(&mut rng);

        // Switch to training mode and verify stem BN is in training mode.
        set_training_mode(&mut model, true);
        let stem_bn = model.layers[1]
            .as_any()
            .downcast_ref::<BatchNormLayer>()
            .expect("Layer 1 should be BatchNormLayer");
        assert!(stem_bn.is_training(), "Stem BN should be in training mode");

        // Switch to inference mode.
        set_training_mode(&mut model, false);
        let stem_bn = model.layers[1]
            .as_any()
            .downcast_ref::<BatchNormLayer>()
            .expect("Layer 1 should be BatchNormLayer");
        assert!(
            !stem_bn.is_training(),
            "Stem BN should be in inference mode"
        );
    }
}
