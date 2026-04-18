use super::*;

/// Train an attention-based MNIST model for EPOCHS epochs and record per-epoch training loss and validation accuracy.
///
/// Initializes the model with the specified positional encoding, runs mini-batch SGD over the training set for EPOCHS (reusing internal buffers), and evaluates validation accuracy after each epoch.
///
/// # Arguments
///
/// - `lr` — learning rate used for SGD.
/// - `pos_type` — positional encoding strategy to initialize token embeddings (`PosEncodingType`).
///
/// # Returns
///
/// A tuple `(final_accuracy, epoch_losses, epoch_accs)` where:
/// - `final_accuracy` is the validation-set accuracy (percentage) after the last epoch,
/// - `epoch_losses` is a vector of average training losses per epoch (length EPOCHS),
/// - `epoch_accs` is a vector of validation-set accuracies (percentage) after each epoch (length EPOCHS).
///
/// # Examples
///
/// ```
/// // assume `train_images`, `train_labels`, `val_images`, `val_labels` are available slices
/// // and `rng` is a mutable SimpleRng
/// let lr = 0.01;
/// let pos_type = PosEncodingType::Sinusoidal;
/// let (final_acc, epoch_losses, epoch_accs) = train_model_with_config(
///     &train_images,
///     &train_labels,
///     &val_images,
///     &val_labels,
///     lr,
///     pos_type,
///     &mut rng,
/// );
/// assert_eq!(epoch_losses.len(), EPOCHS);
/// assert_eq!(epoch_accs.len(), EPOCHS);
/// ```
#[allow(dead_code)]
pub(crate) fn train_model_with_config(
    train_images: &[f32],
    train_labels: &[u8],
    val_images: &[f32],
    val_labels: &[u8],
    lr: f32,
    pos_type: PosEncodingType,
    rng: &mut SimpleRng,
) -> (f32, Vec<f32>, Vec<f32>) {
    let train_n = train_labels.len();

    let mut model = init_model_with_pos_encoding(rng, pos_type);

    // Shuffled indices for mini-batch sampling.
    let mut indices: Vec<usize> = (0..train_n).collect();

    // Training buffers (reused each batch to avoid allocations).
    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut batch_labels = vec![0u8; BATCH_SIZE];
    let mut buf = BatchBuffers::new();
    let mut grads = Grads::new();

    let mut epoch_losses = Vec::new();
    let mut epoch_accs = Vec::new();

    for _epoch in 0..EPOCHS {
        rng.shuffle_usize(&mut indices);

        let mut total_loss = 0.0f32;

        for batch_start in (0..train_n).step_by(BATCH_SIZE) {
            let batch_count = (train_n - batch_start).min(BATCH_SIZE);

            gather_batch(
                train_images,
                train_labels,
                &indices,
                batch_start,
                batch_count,
                &mut batch_inputs,
                &mut batch_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                None,
                None,
                None,
                None,
                None, // saturation_jitter: not applicable for grayscale MNIST
                None,
            );

            // Forward pass + loss.
            let batch_loss =
                forward_batch(&model, &batch_inputs, &batch_labels, batch_count, &mut buf);
            total_loss += batch_loss;

            // Backward pass + SGD update.
            backward_batch(&model, batch_count, &mut buf, &mut grads);
            apply_sgd(&mut model, &grads, lr);
        }

        let avg_loss = total_loss / train_n as f32;
        let acc = test_accuracy(&model, val_images, val_labels);

        epoch_losses.push(avg_loss);
        epoch_accs.push(acc);
    }

    let final_acc = *epoch_accs.last().unwrap_or(&0.0);
    (final_acc, epoch_losses, epoch_accs)
}

// Backward compatibility wrapper for LR experiments.
/// Train the attention model using Sinusoidal positional embeddings and the given learning rate.
///
/// # Returns
///
/// A tuple `(final_validation_accuracy, epoch_losses, epoch_accuracies)`:
/// - `final_validation_accuracy`: final validation-set accuracy as a percentage.
/// - `epoch_losses`: vector of average training losses per epoch.
/// - `epoch_accuracies`: vector of validation accuracies (percentages) per epoch.
///
/// # Examples
///
/// ```ignore
/// // Prepare `train_images`, `train_labels`, `val_images`, `val_labels` and a RNG before calling.
/// let mut rng = SimpleRng::new(42);
/// let (final_acc, losses, accs) = train_model_with_lr(
///     &train_images,
///     &train_labels,
///     &val_images,
///     &val_labels,
///     0.01,
///     &mut rng,
/// );
/// assert_eq!(losses.len(), accs.len());
/// ```
#[allow(dead_code)]
pub(crate) fn train_model_with_lr(
    train_images: &[f32],
    train_labels: &[u8],
    val_images: &[f32],
    val_labels: &[u8],
    lr: f32,
    rng: &mut SimpleRng,
) -> (f32, Vec<f32>, Vec<f32>) {
    train_model_with_config(
        train_images,
        train_labels,
        val_images,
        val_labels,
        lr,
        PosEncodingType::Sinusoidal,
        rng,
    )
}

/// Constructs a learning-rate scheduler from an optional configuration file.
///
/// If `config_path` is `Some(path)`, attempts to load and instantiate the scheduler
/// specified by the config's `scheduler_type` (supported: `"step_decay"`, `"exponential"`,
/// `"cosine_annealing"`). On unknown types, config load failures, or invalid
/// scheduler parameters, falls back to a `ConstantLR` initialized with `LEARNING_RATE`
/// and emits an error message to stderr. When `config_path` is `None`, returns a
/// `ConstantLR` initialized with `LEARNING_RATE`.
///
/// # Parameters
///
/// - `config_path`: Optional path to a scheduler configuration file.
///
/// # Returns
///
/// A boxed `LRScheduler` instance configured according to the provided file, or a
/// `ConstantLR` fallback when no valid configuration is available.
///
/// # Examples
///
/// ```
/// // Returns ConstantLR when no config path is provided.
/// let _sched = build_scheduler(None);
///
/// // When a real config path is available, this will attempt to load it:
/// // let sched = build_scheduler(Some("configs/lr_scheduler.toml"));
/// ```
#[allow(dead_code)]
pub(crate) fn build_scheduler(config_path: Option<&str>) -> Box<dyn LRScheduler> {
    build_scheduler_with_lr(config_path, LEARNING_RATE)
}

pub(crate) fn build_scheduler_with_lr(
    config_path: Option<&str>,
    learning_rate: f32,
) -> Box<dyn LRScheduler> {
    if let Some(path) = config_path {
        match load_config(path) {
            Ok(config) => match config.scheduler_type.as_str() {
                "step_decay" => match StepDecay::from_config(learning_rate, &config) {
                    Ok(scheduler) => Box::new(scheduler),
                    Err(err) => {
                        eprintln!("Invalid StepDecay config: {}", err);
                        Box::new(ConstantLR::new(learning_rate))
                    }
                },
                "exponential" => match config.decay_rate {
                    Some(decay_rate) => Box::new(ExponentialDecay::new(learning_rate, decay_rate)),
                    None => {
                        eprintln!("Invalid ExponentialDecay config: decay_rate is required");
                        Box::new(ConstantLR::new(learning_rate))
                    }
                },
                "cosine_annealing" => match (config.min_lr, config.T_max) {
                    (Some(min_lr), Some(t_max)) if t_max > 0 => {
                        Box::new(CosineAnnealing::new(learning_rate, min_lr, t_max))
                    }
                    (Some(_), Some(_)) => {
                        eprintln!("Invalid CosineAnnealing config: T_max must be > 0");
                        Box::new(ConstantLR::new(learning_rate))
                    }
                    _ => {
                        eprintln!("Invalid CosineAnnealing config: min_lr and T_max are required");
                        Box::new(ConstantLR::new(learning_rate))
                    }
                },
                _ => {
                    eprintln!("Unknown scheduler type: {}", config.scheduler_type);
                    Box::new(ConstantLR::new(learning_rate))
                }
            },
            Err(e) => {
                eprintln!("Failed to load config from {}: {}", path, e);
                Box::new(ConstantLR::new(learning_rate))
            }
        }
    } else {
        Box::new(ConstantLR::new(learning_rate))
    }
}
