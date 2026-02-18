//! Shared training utilities for neural network binaries.
//!
//! This module centralises the training infrastructure that was previously duplicated
//! across every binary (`mnist_mlp`, `mnist_cnn`, `cifar10_cnn`, `mnist_attention_pool`):
//!
//! - [`EarlyStopping`] – tracks validation loss and fires when no improvement is seen for
//!   `patience` epochs.
//! - [`CsvTrainingLogger`] – opens a CSV file and writes one row of metrics per epoch.
//! - [`TrainingMetrics`] – a plain value type that groups per-epoch scalar metrics.
//! - [`parse_config_path`] – extracts `--config <path>` from `argv`, falling back to a
//!   caller-supplied default.
//! - [`print_training_config`] – prints a [`TrainingConfig`] to stdout in the standard
//!   format used by all binaries.
//! - [`compute_softmax_cross_entropy`] – computes the cross-entropy loss and the
//!   softmax-minus-one-hot gradient in a single pass, unifying
//!   `compute_delta_and_loss` (mnist_mlp) and `softmax_xent_backward` (mnist_cnn).
//! - [`evaluate_batch_accuracy`] – computes cross-entropy loss and argmax accuracy from
//!   a batch of already-softmaxed probabilities, unifying the inline validation loops.
//! - [`gather_batch`] – copies a mini-batch of images and labels with optional data
//!   augmentation (flip, crop, brightness, contrast, saturation), replacing the
//!   binary-local `gather_batch` functions in every binary.
//! - [`CsvGradientLogger`] – opens a CSV file and writes one row of gradient norms per layer
//!   per epoch, enabling real-time gradient flow visualisation across all model types.

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::config::TrainingConfig;
use crate::utils::rng::SimpleRng;

// ─────────────────────────────────────────────────────────────────────────────
// TrainingMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar metrics produced at the end of a training epoch.
///
/// All binaries compute the same five quantities each epoch; this struct
/// avoids repeated long parameter lists.
#[derive(Debug, Clone, Copy)]
pub struct TrainingMetrics {
    /// Average cross-entropy loss over the training set for this epoch.
    pub train_loss: f32,
    /// Average cross-entropy loss over the validation set for this epoch.
    pub val_loss: f32,
    /// Validation accuracy as a percentage (0.0–100.0).
    pub val_accuracy: f32,
    /// Wall-clock time (seconds) spent on the training portion of the epoch.
    pub train_time: f32,
    /// Learning rate that was active during this epoch's parameter updates.
    pub learning_rate: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// EarlyStopping
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome returned by [`EarlyStopping::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingAction {
    /// Validation loss improved by at least `min_delta`; the caller should save
    /// a model checkpoint.
    Improved,
    /// No improvement, but patience has not been exceeded; training continues.
    Continue,
    /// No improvement for `patience` consecutive epochs; training should stop.
    Stop,
}

/// Tracks the best observed validation loss and fires after `patience` epochs
/// without improvement.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::{EarlyStopping, EarlyStoppingAction};
///
/// let mut es = EarlyStopping::new(3, 0.001);
///
/// // First epoch always improves (starting from f32::INFINITY).
/// assert_eq!(es.check(1.0), EarlyStoppingAction::Improved);
///
/// // No improvement for 3 consecutive epochs triggers Stop.
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Stop);
/// ```
pub struct EarlyStopping {
    /// The lowest validation loss observed so far.
    pub best_val_loss: f32,
    /// How many consecutive epochs have passed without a sufficient improvement.
    pub epochs_without_improvement: usize,
    patience: usize,
    min_delta: f32,
}

impl EarlyStopping {
    /// Creates a new [`EarlyStopping`] monitor.
    ///
    /// - `patience` – number of epochs without improvement before [`EarlyStoppingAction::Stop`]
    ///   is returned.
    /// - `min_delta` – the minimum decrease in validation loss that counts as an improvement.
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            best_val_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            patience,
            min_delta,
        }
    }

    /// Updates internal state with the latest `val_loss` and returns the appropriate action.
    ///
    /// If `val_loss < best_val_loss - min_delta`, the best loss is updated, the counter resets,
    /// and [`EarlyStoppingAction::Improved`] is returned.  Otherwise the counter increments; if
    /// it reaches `patience`, [`EarlyStoppingAction::Stop`] is returned; otherwise
    /// [`EarlyStoppingAction::Continue`].
    pub fn check(&mut self, val_loss: f32) -> EarlyStoppingAction {
        if val_loss < self.best_val_loss - self.min_delta {
            self.best_val_loss = val_loss;
            self.epochs_without_improvement = 0;
            EarlyStoppingAction::Improved
        } else {
            self.epochs_without_improvement += 1;
            if self.epochs_without_improvement >= self.patience {
                EarlyStoppingAction::Stop
            } else {
                EarlyStoppingAction::Continue
            }
        }
    }

    /// Returns `true` if training should halt.
    ///
    /// Equivalent to `self.epochs_without_improvement >= patience`.
    pub fn should_stop(&self) -> bool {
        self.epochs_without_improvement >= self.patience
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CsvTrainingLogger
// ─────────────────────────────────────────────────────────────────────────────

/// Writes per-epoch training metrics to a CSV file.
///
/// The CSV format matches what all binaries produce:
/// ```text
/// epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate
/// 1,0.354321,12.3,0.312456,91.23,0.01
/// ```
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::training::{CsvTrainingLogger, TrainingMetrics};
///
/// let mut logger = CsvTrainingLogger::new("./logs/training_loss_mlp.csv").unwrap();
/// logger.write_header().unwrap();
/// let metrics = TrainingMetrics {
///     train_loss: 0.5,
///     val_loss: 0.4,
///     val_accuracy: 88.5,
///     train_time: 10.0,
///     learning_rate: 0.01,
/// };
/// logger.write_epoch(1, &metrics).unwrap();
/// ```
pub struct CsvTrainingLogger {
    writer: BufWriter<File>,
}

impl CsvTrainingLogger {
    /// Opens (or creates) the file at `path` and wraps it in a buffered writer.
    ///
    /// The logs directory must already exist or the caller must create it beforehand
    /// (e.g. with `std::fs::create_dir_all("./logs")`).
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Writes the standard CSV header line.
    ///
    /// This should be called once immediately after construction.
    pub fn write_header(&mut self) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate"
        )
    }

    /// Appends one row for `epoch` (1-based) using the supplied `metrics`.
    pub fn write_epoch(
        &mut self,
        epoch: usize,
        metrics: &TrainingMetrics,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "{},{},{},{},{},{}",
            epoch,
            metrics.train_loss,
            metrics.train_time,
            metrics.val_loss,
            metrics.val_accuracy,
            metrics.learning_rate,
        )
    }

    /// Flushes the internal buffer, ensuring all data is written to disk.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CsvGradientLogger
// ─────────────────────────────────────────────────────────────────────────────

/// Writes per-epoch per-layer gradient norms to a CSV file.
///
/// The CSV format records one row per layer per epoch, allowing downstream tools
/// to visualise gradient flow and detect vanishing or exploding gradients:
/// ```text
/// epoch,layer_name,grad_norm_weights,grad_norm_biases
/// 1,layer_0_dense,0.042314,0.003217
/// 1,layer_1_dense,0.021756,0.001843
/// ```
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::training::CsvGradientLogger;
///
/// let mut logger = CsvGradientLogger::new("./logs/gradients_mlp.csv").unwrap();
/// logger.write_header().unwrap();
/// logger.write_layer(1, "layer_0_dense", 0.042314, 0.003217).unwrap();
/// logger.write_layer(1, "layer_1_dense", 0.021756, 0.001843).unwrap();
/// logger.flush().unwrap();
/// ```
pub struct CsvGradientLogger {
    writer: BufWriter<File>,
}

impl CsvGradientLogger {
    /// Opens (or creates) the file at `path` and wraps it in a buffered writer.
    ///
    /// The logs directory must already exist or the caller must create it beforehand
    /// (e.g. with `std::fs::create_dir_all("./logs")`).
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Writes the standard CSV header line.
    ///
    /// This should be called once immediately after construction.
    pub fn write_header(&mut self) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "epoch,layer_name,grad_norm_weights,grad_norm_biases"
        )
    }

    /// Appends one row for the given `epoch` (1-based) and `layer_name` with the
    /// L2 gradient norms for that layer's weights and biases.
    pub fn write_layer(
        &mut self,
        epoch: usize,
        layer_name: &str,
        weight_norm: f32,
        bias_norm: f32,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "{},{},{},{}",
            epoch, layer_name, weight_norm, bias_norm,
        )
    }

    /// Flushes the internal buffer, ensuring all data is written to disk.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_config_path
// ─────────────────────────────────────────────────────────────────────────────

/// Extracts a configuration file path from command-line arguments.
///
/// Scans `args` for `--config <path>` and returns the `<path>` component.
/// If the flag is absent, `default_path` is returned.
///
/// # Parameters
///
/// - `args` – the full `argv` slice (including the binary name at index 0).
/// - `default_path` – path to use when no `--config` flag is present.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::parse_config_path;
///
/// let args = vec!["binary".to_string(), "--config".to_string(), "my.json".to_string()];
/// assert_eq!(parse_config_path(&args, "default.json"), "my.json");
///
/// let args_empty: Vec<String> = vec!["binary".to_string()];
/// assert_eq!(parse_config_path(&args_empty, "default.json"), "default.json");
/// ```
pub fn parse_config_path(args: &[String], default_path: &str) -> String {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    default_path.to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_step_flag
// ─────────────────────────────────────────────────────────────────────────────

/// Checks whether the `--step` flag is present in command-line arguments.
///
/// Returns `true` if any argument is exactly `"--step"`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::parse_step_flag;
///
/// let args = vec!["binary".to_string(), "--step".to_string()];
/// assert!(parse_step_flag(&args));
///
/// let args_empty: Vec<String> = vec!["binary".to_string()];
/// assert!(!parse_step_flag(&args_empty));
/// ```
pub fn parse_step_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--step")
}

// ─────────────────────────────────────────────────────────────────────────────
// print_training_config
// ─────────────────────────────────────────────────────────────────────────────

/// Prints the active training configuration to stdout in a human-readable format.
///
/// Matches the output format used by all binaries so that training runs are
/// easy to compare across models.
///
/// # Parameters
///
/// - `config` – the loaded [`TrainingConfig`] to display.
/// - `learning_rate` – the resolved learning rate (after applying config defaults).
/// - `epochs` – the resolved number of epochs.
/// - `batch_size` – the resolved batch size.
/// - `validation_split` – the resolved validation split fraction.
/// - `early_stopping_patience` – the resolved early stopping patience.
/// - `early_stopping_min_delta` – the resolved early stopping min delta.
pub fn print_training_config(
    config: &TrainingConfig,
    learning_rate: f32,
    epochs: usize,
    batch_size: usize,
    validation_split: f32,
    early_stopping_patience: usize,
    early_stopping_min_delta: f32,
) {
    let enable_augmentation = config.enable_augmentation.unwrap_or(false);

    println!("\nConfiguration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Validation split: {:.1}%", validation_split * 100.0);
    println!("  Early stopping patience: {}", early_stopping_patience);
    println!("  Early stopping min delta: {}", early_stopping_min_delta);
    println!("  Scheduler type: {}", config.scheduler_type);
    if let Some(ref activation) = config.activation_function {
        println!("  Activation function: {}", activation);
    }
    println!(
        "  Data augmentation: {}",
        if enable_augmentation {
            "enabled"
        } else {
            "disabled"
        }
    );
    if enable_augmentation {
        if let Some(prob) = config.horizontal_flip_prob {
            println!("    Horizontal flip probability: {}", prob);
        }
        if let Some(padding) = config.random_crop_padding {
            println!("    Random crop padding: {}", padding);
        }
        if let Some(delta) = config.brightness_jitter {
            println!("    Brightness jitter: {}", delta);
        }
        if let Some(delta) = config.contrast_jitter {
            println!("    Contrast jitter: {}", delta);
        }
        if let Some(delta) = config.saturation_jitter {
            println!("    Saturation jitter: {}", delta);
        }
    }
    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_softmax_cross_entropy
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the cross-entropy loss and writes the softmax gradient into `delta`.
///
/// This function unifies `compute_delta_and_loss` (mnist_mlp) and
/// `softmax_xent_backward` (mnist_cnn).  It expects that softmax has already
/// been applied to `probs` before calling this function.
///
/// For each sample `i` in `0..rows`:
/// - Accumulates `−ln(probs[i * cols + label[i]])` into the returned loss.
/// - Writes `(probs[i * cols + j] − 𝟙{j == label[i]}) * scale` into
///   `delta[i * cols + j]` for each class `j`.
///
/// # Parameters
///
/// - `probs`  – flattened row-major softmax probabilities, length ≥ `rows * cols`.
/// - `labels` – true class labels; only the first `rows` entries are used.
/// - `rows`   – number of samples to process.
/// - `cols`   – number of classes per sample.
/// - `delta`  – mutable output buffer for gradients; must be length ≥ `rows * cols`.
/// - `scale`  – multiplicative scale applied to every gradient value.
///   Use `1.0` for unscaled gradients (mnist_mlp style) or `1.0 / batch_size`
///   for pre-normalised gradients (mnist_cnn style).
///
/// # Returns
///
/// Total (summed, not averaged) cross-entropy loss over the `rows` samples.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::compute_softmax_cross_entropy;
///
/// let probs = [0.9f32, 0.1f32]; // one sample, two-class softmax
/// let labels = [0u8];
/// let mut delta = [0.0f32; 2];
/// let loss = compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);
///
/// let expected_loss = -(0.9f32).ln();
/// assert!((loss - expected_loss).abs() < 1e-6);
/// // gradient: true class probability minus 1, others unchanged
/// assert!((delta[0] - (-0.1f32)).abs() < 1e-6);
/// assert!((delta[1] - 0.1f32).abs() < 1e-6);
/// ```
pub fn compute_softmax_cross_entropy(
    probs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
    scale: f32,
) -> f32 {
    let epsilon = 1e-9f32;
    let mut total_loss = 0.0f32;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;

        let prob = probs[row_start + label].max(epsilon);
        total_loss -= prob.ln();

        let row = &probs[row_start..row_start + cols];
        let delta_row = &mut delta[row_start..row_start + cols];
        for (j, &p) in row.iter().enumerate() {
            let mut g = p;
            if j == label {
                g -= 1.0;
            }
            delta_row[j] = g * scale;
        }
    }

    total_loss
}

// ─────────────────────────────────────────────────────────────────────────────
// evaluate_batch_accuracy
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the cross-entropy loss and the number of correct predictions for a
/// batch of softmax probabilities.
///
/// This function consolidates the inline validation loops that appear identically
/// in every binary.  The caller is responsible for applying softmax to `probs`
/// before calling this function.
///
/// # Parameters
///
/// - `probs`  – flattened row-major softmax probabilities, length ≥ `rows * cols`.
/// - `labels` – true class labels corresponding to each row.
/// - `rows`   – number of samples (rows) in the batch.
/// - `cols`   – number of classes per sample.
///
/// # Returns
///
/// `(total_loss, correct_count)` where:
/// - `total_loss` is the summed cross-entropy loss (not yet divided by `rows`).
/// - `correct_count` is the number of samples whose argmax equals the true label.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::evaluate_batch_accuracy;
///
/// // One sample, two classes, correct prediction.
/// let probs = [0.9f32, 0.1f32];
/// let labels = [0u8];
/// let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
/// assert!(loss > 0.0);
/// assert_eq!(correct, 1);
/// ```
pub fn evaluate_batch_accuracy(
    probs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
) -> (f32, usize) {
    let epsilon = 1e-9f32;
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;

        // Cross-entropy loss
        let prob = probs[row_start + label].max(epsilon);
        total_loss -= prob.ln();

        // Argmax prediction
        let row = &probs[row_start..row_start + cols];
        let mut predicted = 0usize;
        let mut max_prob = row[0];
        for (j, &value) in row.iter().enumerate().skip(1) {
            if value > max_prob {
                max_prob = value;
                predicted = j;
            }
        }
        if predicted == label {
            correct += 1;
        }
    }

    (total_loss, correct)
}

// ─────────────────────────────────────────────────────────────────────────────
// gather_batch
// ─────────────────────────────────────────────────────────────────────────────

/// Copies a mini-batch of images and labels into output buffers, applying optional augmentations.
///
/// This function unifies the `gather_batch` functions from all binary targets
/// (`mnist_mlp`, `mnist_cnn`, `cifar10_cnn`, `mnist_attention_pool`). It is generic
/// over image dimensions so the same function works for MNIST (1×28×28) and
/// CIFAR-10 (3×32×32) without modification.
///
/// Augmentations are applied in the following order when their parameters are provided:
/// 1. Random crop (`crop_padding`)
/// 2. Random horizontal flip (`flip_prob`)
/// 3. Brightness jitter (`brightness_jitter`)
/// 4. Contrast jitter (`contrast_jitter`)
/// 5. Saturation jitter (`saturation_jitter`) – pass `None` for grayscale images
///
/// # Parameters
///
/// - `images` – flattened dataset buffer; image `i` occupies
///   `images[i * image_size..(i+1) * image_size]` where
///   `image_size = img_width * img_height * img_channels`.
/// - `labels` – class label for each image.
/// - `indices` – index permutation produced by the epoch shuffle.
/// - `start` – first entry in `indices` to read.
/// - `count` – number of samples to gather into the output buffers.
/// - `out_inputs` – output buffer; must be at least `count * image_size` long.
/// - `out_labels` – label output buffer; must be at least `count` long.
/// - `img_width` – image width in pixels.
/// - `img_height` – image height in pixels.
/// - `img_channels` – number of channels (1 for grayscale MNIST, 3 for RGB CIFAR-10).
/// - `flip_prob` – probability (0.0–1.0) of random horizontal flip per image.
/// - `crop_padding` – pixels of padding added before random cropping back to original size.
/// - `brightness_jitter` – maximum brightness adjustment delta (±delta).
/// - `contrast_jitter` – maximum contrast adjustment delta (±delta).
/// - `saturation_jitter` – maximum saturation adjustment delta (±delta); use `None` for
///   grayscale images because the operation is a no-op on single-channel data.
/// - `rng` – optional random number generator; required for any augmentation to take effect.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::gather_batch;
///
/// // Tiny MNIST-like dataset: 3 grayscale 2×2 images (4 pixels each).
/// let images = vec![1.0f32, 2.0, 3.0, 4.0,   // image 0
///                   5.0,    6.0, 7.0, 8.0,   // image 1
///                   9.0,   10.0,11.0,12.0];  // image 2
/// let labels = vec![0u8, 1u8, 2u8];
/// let indices = vec![2usize, 0, 1];
/// let mut out_inputs = vec![0.0f32; 4 * 2]; // 2 images × 4 pixels
/// let mut out_labels = vec![0u8; 2];
///
/// // Gather 2 samples (indices[0..2] → images 2 and 0), no augmentation.
/// gather_batch(
///     &images, &labels, &indices, 0, 2,
///     &mut out_inputs, &mut out_labels,
///     2, 2, 1, // img_width=2, img_height=2, img_channels=1
///     None, None, None, None, None, None,
/// );
///
/// assert_eq!(out_labels, vec![2u8, 0u8]);
/// assert_eq!(out_inputs[0], 9.0); // first pixel of image 2
/// assert_eq!(out_inputs[4], 1.0); // first pixel of image 0
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gather_batch(
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    start: usize,
    count: usize,
    out_inputs: &mut [f32],
    out_labels: &mut [u8],
    img_width: usize,
    img_height: usize,
    img_channels: usize,
    flip_prob: Option<f32>,
    crop_padding: Option<usize>,
    brightness_jitter: Option<f32>,
    contrast_jitter: Option<f32>,
    saturation_jitter: Option<f32>,
    mut rng: Option<&mut SimpleRng>,
) {
    use crate::data::augmentation::{
        random_brightness, random_contrast, random_crop, random_horizontal_flip, random_saturation,
    };

    let image_size = img_width * img_height * img_channels;

    for i in 0..count {
        let src_index = indices[start + i];
        let src_start = src_index * image_size;
        let dst_start = i * image_size;

        // Copy base image to output buffer
        out_inputs[dst_start..dst_start + image_size]
            .copy_from_slice(&images[src_start..src_start + image_size]);
        out_labels[i] = labels[src_index];

        // Apply augmentations if parameters are provided and RNG is available
        if let Some(ref mut rng_ref) = rng {
            let image_slice = &mut out_inputs[dst_start..dst_start + image_size];

            // 1. Apply random crop if padding is specified
            if let Some(padding) = crop_padding {
                // random_crop returns a new Vec; copy it back into the output slice
                let cropped = random_crop(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    padding,
                    img_width,  // crop back to original width
                    img_height, // crop back to original height
                    rng_ref,
                );
                image_slice.copy_from_slice(&cropped);
            }

            // 2. Apply random horizontal flip if probability is specified
            if let Some(prob) = flip_prob {
                random_horizontal_flip(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    prob,
                    rng_ref,
                );
            }

            // 3. Apply brightness jitter if specified
            if let Some(brightness_delta) = brightness_jitter {
                random_brightness(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    brightness_delta,
                    rng_ref,
                );
            }

            // 4. Apply contrast jitter if specified
            if let Some(contrast_delta) = contrast_jitter {
                random_contrast(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    contrast_delta,
                    rng_ref,
                );
            }

            // 5. Apply saturation jitter if specified (only meaningful for RGB; no-op for grayscale)
            if let Some(saturation_delta) = saturation_jitter {
                random_saturation(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    saturation_delta,
                    rng_ref,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_softmax_cross_entropy ─────────────────────────────────────────

    #[test]
    fn test_compute_softmax_cross_entropy_basic() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta = [0.0f32; 2];
        let loss = compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);

        let expected_loss = -(0.9f32).ln();
        assert!((loss - expected_loss).abs() < 1e-6, "loss mismatch");
        assert!(
            (delta[0] - (-0.1f32)).abs() < 1e-6,
            "gradient at true class"
        );
        assert!((delta[1] - 0.1f32).abs() < 1e-6, "gradient at other class");
    }

    #[test]
    fn test_compute_softmax_cross_entropy_scale() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta_scaled = [0.0f32; 2];
        let mut delta_unscaled = [0.0f32; 2];

        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta_unscaled, 1.0);
        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta_scaled, 0.5);

        assert!((delta_scaled[0] - delta_unscaled[0] * 0.5).abs() < 1e-7);
        assert!((delta_scaled[1] - delta_unscaled[1] * 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_compute_softmax_cross_entropy_multibatch() {
        let probs = [0.1f32, 0.2, 0.7, 0.3, 0.4, 0.3];
        let labels = [2u8, 1u8];
        let mut delta = [0.0f32; 6];
        let loss = compute_softmax_cross_entropy(&probs, &labels, 2, 3, &mut delta, 1.0);

        assert!(loss > 0.0);
        // Sample 0: label=2, prob=0.7, gradient at class 2 = 0.7 - 1.0 = -0.3
        assert!((delta[2] - (0.7 - 1.0)).abs() < 1e-6, "delta[2] wrong");
        // Sample 1: label=1, prob=0.4, gradient at class 1 = 0.4 - 1.0 = -0.6
        assert!((delta[4] - (0.4 - 1.0)).abs() < 1e-6, "delta[4] wrong");
    }

    // ── evaluate_batch_accuracy ───────────────────────────────────────────────

    #[test]
    fn test_evaluate_batch_accuracy_correct() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
        assert!(loss > 0.0);
        assert_eq!(correct, 1);
    }

    #[test]
    fn test_evaluate_batch_accuracy_wrong() {
        let probs = [0.1f32, 0.9f32];
        let labels = [0u8]; // true is 0 but argmax is 1
        let (_loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
        assert_eq!(correct, 0);
    }

    #[test]
    fn test_evaluate_batch_accuracy_multibatch() {
        // 2 samples, 3 classes
        // sample 0: probs=[0.1, 0.2, 0.7], label=2  → correct (argmax=2)
        // sample 1: probs=[0.5, 0.3, 0.2], label=1  → wrong   (argmax=0)
        let probs = [0.1f32, 0.2, 0.7, 0.5, 0.3, 0.2];
        let labels = [2u8, 1u8];
        let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 2, 3);
        assert!(loss > 0.0);
        assert_eq!(correct, 1);
    }

    // ── EarlyStopping ─────────────────────────────────────────────────────────

    #[test]
    fn test_early_stopping_improves() {
        let mut es = EarlyStopping::new(3, 0.001);
        assert_eq!(es.check(1.0), EarlyStoppingAction::Improved);
        assert_eq!(es.best_val_loss, 1.0);
        assert_eq!(es.epochs_without_improvement, 0);
    }

    #[test]
    fn test_early_stopping_continues() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0); // Improved
        assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
        assert_eq!(es.epochs_without_improvement, 1);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0); // Improved
        es.check(1.5); // Continue (1 without improvement)
        es.check(1.5); // Continue (2 without improvement)
        assert_eq!(es.check(1.5), EarlyStoppingAction::Stop); // Stop (3 without improvement)
        assert!(es.should_stop());
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0); // Improved
        es.check(1.5); // Continue (1 without improvement)
        assert_eq!(es.check(0.5), EarlyStoppingAction::Improved); // resets counter
        assert_eq!(es.epochs_without_improvement, 0);
        assert!(!es.should_stop());
    }

    // ── parse_config_path ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_config_path_default() {
        let args: Vec<String> = vec!["binary".to_string()];
        assert_eq!(parse_config_path(&args, "default.json"), "default.json");
    }

    #[test]
    fn test_parse_config_path_flag() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "custom.json".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "custom.json");
    }

    #[test]
    fn test_parse_config_path_extra_args() {
        let args = vec![
            "binary".to_string(),
            "--verbose".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
            "--other".to_string(),
        ];
        assert_eq!(parse_config_path(&args, "default.json"), "my.json");
    }

    #[test]
    fn test_parse_config_path_flag_at_end_ignored() {
        // --config with no following argument should fall back to default
        let args = vec!["binary".to_string(), "--config".to_string()];
        assert_eq!(parse_config_path(&args, "default.json"), "default.json");
    }

    // ── parse_step_flag ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_step_flag_present() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
            "--step".to_string(),
        ];
        assert!(parse_step_flag(&args));
    }

    #[test]
    fn test_parse_step_flag_absent() {
        let args = vec![
            "binary".to_string(),
            "--config".to_string(),
            "my.json".to_string(),
        ];
        assert!(!parse_step_flag(&args));
    }

    // ── gather_batch ──────────────────────────────────────────────────────────

    #[test]
    fn test_gather_batch_basic_copy() {
        // 3 grayscale 2×2 images (4 pixels each): indices[0..2] = [2, 0]
        let images = vec![
            1.0f32, 2.0, 3.0, 4.0, // image 0
            5.0, 6.0, 7.0, 8.0, // image 1
            9.0, 10.0, 11.0, 12.0, // image 2
        ];
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![2usize, 0, 1];
        let mut out_inputs = vec![0.0f32; 4 * 2];
        let mut out_labels = vec![0u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            2,    // img_width
            2,    // img_height
            1,    // img_channels (grayscale)
            None, // flip_prob
            None, // crop_padding
            None, // brightness_jitter
            None, // contrast_jitter
            None, // saturation_jitter
            None, // rng
        );

        assert_eq!(out_labels[0], 2u8, "first sample should be image 2");
        assert_eq!(out_labels[1], 0u8, "second sample should be image 0");
        assert!(
            (out_inputs[0] - 9.0f32).abs() < 1e-6,
            "first pixel of image 2"
        );
        assert!(
            (out_inputs[4] - 1.0f32).abs() < 1e-6,
            "first pixel of image 0"
        );
    }

    #[test]
    fn test_gather_batch_no_augmentation_without_rng() {
        // Even if augmentation params are Some, without an rng they must be ignored.
        let images = vec![0.5f32; 4 * 3]; // 3 images, 4 pixels each
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![0usize, 1, 2];
        let mut out_inputs = vec![0.0f32; 4 * 2];
        let mut out_labels = vec![0u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            Some(0.5), // flip_prob provided but rng=None → no augmentation
            Some(1),   // crop_padding provided but rng=None → no augmentation
            Some(0.2), // brightness_jitter provided but rng=None → no augmentation
            Some(0.2), // contrast_jitter
            Some(0.2), // saturation_jitter
            None,      // rng: no generator → augmentation is skipped
        );

        // All pixels are 0.5; without augmentation they must remain 0.5
        for &v in &out_inputs[..8] {
            assert!(
                (v - 0.5f32).abs() < 1e-6,
                "without rng, pixels should be unchanged"
            );
        }
    }

    #[test]
    fn test_gather_batch_with_flip_augmentation() {
        use crate::utils::rng::SimpleRng;

        // One 4-pixel-wide, 1-pixel-high, 1-channel image with a known pattern.
        // horizontal flip of [1,2,3,4] should give [4,3,2,1].
        let images = vec![1.0f32, 2.0, 3.0, 4.0];
        let labels = vec![0u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; 4];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            4,         // img_width  (4 pixels per row)
            1,         // img_height (1 row)
            1,         // img_channels (grayscale)
            Some(1.0), // always flip
            None,
            None,
            None,
            None,
            Some(&mut rng),
        );

        assert_eq!(out_labels[0], 0u8);
        // After a guaranteed horizontal flip the order must be reversed
        assert!(
            (out_inputs[0] - 4.0f32).abs() < 1e-6,
            "flipped pixel 0 should be 4.0, got {}",
            out_inputs[0]
        );
        assert!(
            (out_inputs[3] - 1.0f32).abs() < 1e-6,
            "flipped pixel 3 should be 1.0, got {}",
            out_inputs[3]
        );
    }
}
