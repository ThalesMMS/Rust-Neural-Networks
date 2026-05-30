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

mod batch;
mod cli;
mod early_stopping;
mod gpu;
mod logging;
mod loss;
mod metrics;

pub use self::batch::gather_batch;
pub use self::cli::{
    parse_config_path, parse_registry_dir, parse_run_name, parse_seed_override, parse_step_flag,
    print_training_config,
};
pub use self::early_stopping::{EarlyStopping, EarlyStoppingAction};
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
pub use self::gpu::upgrade_layers_to_gpu;
pub use self::logging::{CsvGradientLogger, CsvTrainingLogger};
pub use self::loss::{compute_softmax_cross_entropy, evaluate_batch_accuracy};
pub use self::metrics::TrainingMetrics;
