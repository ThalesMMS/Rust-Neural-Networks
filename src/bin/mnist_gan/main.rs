//! MNIST GAN – Generative Adversarial Network for MNIST digit generation.
//!
//! # Architecture
//!
//! ## Generator (noise → image)
//! - Input:  NOISE_DIM (100) latent noise vector, values in [-1, 1]
//! - Layer1: 100 → 256, LeakyReLU(α=0.2)
//! - Layer2: 256 → 512, LeakyReLU(α=0.2)
//! - Layer3: 512 → 784, Tanh
//! - Output: 784-dimensional image in [-1, 1]
//!
//! ## Discriminator (image → real/fake probability)
//! - Input:  784-dimensional image in [-1, 1]
//! - Layer1: 784 → 512, LeakyReLU(α=0.2)
//! - Layer2: 512 → 256, LeakyReLU(α=0.2)
//! - Layer3: 256 → 1,   Sigmoid
//! - Output: scalar probability that the image is real
//!
//! # Training
//!
//! The GAN uses the standard minimax loss with one-sided label smoothing.
//! Real MNIST images are rescaled from [0, 1] to [-1, 1] to match the
//! generator's Tanh output range.  The discriminator is trained first on
//! real images (target = 1.0 - label_smoothing) then on generated fakes
//! (target = 0.0).  The generator is then trained to fool the discriminator
//! (target = 1.0 for the generator step).
//!
//! See `docs/gan_tutorial.md` for a detailed explanation of GAN theory.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

use rust_neural_networks::config::load_config;
use rust_neural_networks::data::mnist::read_mnist_images;
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::optimizers::{Adam, Optimizer};
use rust_neural_networks::training::parse_config_path;
use rust_neural_networks::utils::rng::SimpleRng;

// BLAS back-end (mirrors dense.rs setup)
#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;
use cblas::{sgemm, Layout, Transpose};

// ============================================================================
// Architecture Constants
// ============================================================================

/// Dimension of the latent noise vector fed to the generator.
const NOISE_DIM: usize = 100;

/// Number of units in the first generator hidden layer.
const G_HIDDEN1: usize = 256;

/// Number of units in the second generator hidden layer.
const G_HIDDEN2: usize = 512;

/// Number of pixels in a flattened 28×28 MNIST image.
const IMG_SIZE: usize = 784;

/// Number of units in the first discriminator hidden layer.
const D_HIDDEN1: usize = 512;

/// Number of units in the second discriminator hidden layer.
const D_HIDDEN2: usize = 256;

/// Negative-slope coefficient for Leaky ReLU in both networks.
const LEAKY_RELU_ALPHA: f32 = 0.2;

/// Default configuration file path for GAN training.
const DEFAULT_CONFIG_PATH: &str = "config/training/mnist_gan_default.json";

// ============================================================================
// Utility Functions
// ============================================================================

mod discriminator;
mod generator;
mod io;
mod math;
mod training;

use discriminator::*;
use generator::*;
use io::*;
use math::*;
use training::*;

/// Binary entrypoint that configures and runs training of a fully connected GAN on MNIST.
///
/// This function:
/// - parses CLI args to locate a configuration file (falls back to a default),
/// - loads hyperparameters (or uses built-in defaults on config load failure),
/// - loads MNIST training images (exits on load failure),
/// - initializes deterministic RNGs and constructs the Generator and Discriminator,
/// - prints model architecture and parameter counts, and
/// - runs the training loop, finally printing total runtime.
///
/// # Examples
///
/// ```no_run
/// // Runs the MNIST GAN training as the program entrypoint.
/// mnist_gan::main();
/// ```
fn main() {
    let program_start = Instant::now();
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_path(&args, DEFAULT_CONFIG_PATH);

    println!("=== MNIST GAN Training ===");
    println!("Loading configuration from: {}", config_path);

    // Extract hyperparameters; fall back to GAN defaults if config is unavailable.
    let (g_lr, d_lr, beta1, beta2, noise_dim, label_smoothing, epochs, batch_size) =
        match load_config(&config_path) {
            Ok(cfg) => {
                let lr = cfg.learning_rate.unwrap_or(0.0002);
                (
                    cfg.g_lr.unwrap_or(lr),
                    cfg.d_lr.unwrap_or(lr),
                    cfg.adam_beta1.unwrap_or(0.5),
                    cfg.adam_beta2.unwrap_or(0.999),
                    cfg.noise_dim.unwrap_or(NOISE_DIM),
                    cfg.label_smoothing.unwrap_or(0.1),
                    cfg.epochs.unwrap_or(50),
                    cfg.batch_size.unwrap_or(64),
                )
            }
            Err(e) => {
                eprintln!(
                    "Warning: could not load config from '{}': {}",
                    config_path, e
                );
                eprintln!("Using built-in GAN defaults.");
                (0.0002, 0.0002, 0.5, 0.999, NOISE_DIM, 0.1, 50, 64)
            }
        };

    if !g_lr.is_finite() || g_lr <= 0.0 {
        eprintln!("Invalid g_lr: {}. Expected a finite positive value.", g_lr);
        process::exit(1);
    }
    if !d_lr.is_finite() || d_lr <= 0.0 {
        eprintln!("Invalid d_lr: {}. Expected a finite positive value.", d_lr);
        process::exit(1);
    }
    if !(beta1.is_finite() && beta1 > 0.0 && beta1 < 1.0) {
        eprintln!("Invalid beta1: {}. Expected a value in (0, 1).", beta1);
        process::exit(1);
    }
    if !(beta2.is_finite() && beta2 > 0.0 && beta2 < 1.0) {
        eprintln!("Invalid beta2: {}. Expected a value in (0, 1).", beta2);
        process::exit(1);
    }
    if noise_dim == 0 {
        eprintln!(
            "Invalid noise_dim: {}. Expected a positive value.",
            noise_dim
        );
        process::exit(1);
    }
    if noise_dim != NOISE_DIM {
        eprintln!(
            "Invalid noise_dim: {}. This binary is compiled with NOISE_DIM={}.",
            noise_dim, NOISE_DIM
        );
        process::exit(1);
    }
    if !(label_smoothing.is_finite() && label_smoothing >= 0.0 && label_smoothing < 1.0) {
        eprintln!(
            "Invalid label_smoothing: {}. Expected a value in [0, 1).",
            label_smoothing
        );
        process::exit(1);
    }
    if epochs == 0 {
        eprintln!("Invalid epochs: {}. Expected a positive value.", epochs);
        process::exit(1);
    }
    if batch_size == 0 {
        eprintln!(
            "Invalid batch_size: {}. Expected a positive value.",
            batch_size
        );
        process::exit(1);
    }

    println!(
        "G lr: {}, D lr: {}, beta1: {}, beta2: {}, noise_dim: {}",
        g_lr, d_lr, beta1, beta2, noise_dim
    );
    println!(
        "label_smoothing: {}, epochs: {}, batch_size: {}",
        label_smoothing, epochs, batch_size
    );

    // Load MNIST training data
    println!("Loading MNIST training data...");
    let load_start = Instant::now();
    let train_images = read_mnist_images("./data/train-images.idx3-ubyte").unwrap_or_else(|e| {
        eprintln!("{e}");
        process::exit(1);
    });
    let num_samples = train_images.len() / IMG_SIZE;
    let val_count = (num_samples as f32 * VALIDATION_SPLIT) as usize;
    let train_count = num_samples - val_count;
    if train_count == 0 || val_count == 0 {
        eprintln!(
            "Invalid MNIST split: {} total samples gives {} train and {} validation samples.",
            num_samples, train_count, val_count
        );
        process::exit(1);
    }
    let train_image_len = train_count * IMG_SIZE;
    let total_image_len = num_samples * IMG_SIZE;
    let train_images_train = &train_images[..train_image_len];
    let val_images = &train_images[train_image_len..total_image_len];
    let num_train = train_count;
    println!(
        "Loaded {} images in {:.2}s ({} train, {} validation)",
        num_samples,
        load_start.elapsed().as_secs_f64(),
        num_train,
        val_images.len() / IMG_SIZE
    );

    // Initialise networks
    let mut rng = SimpleRng::new(42);
    let mut generator = Generator::new(&mut rng, g_lr, beta1, beta2);
    let mut discriminator = Discriminator::new(&mut rng, d_lr, beta1, beta2);

    println!(
        "Generator:     {} → {} → {} → {} ({} params)",
        NOISE_DIM,
        G_HIDDEN1,
        G_HIDDEN2,
        IMG_SIZE,
        generator.layer1.parameter_count()
            + generator.layer2.parameter_count()
            + generator.layer3.parameter_count()
    );
    println!(
        "Discriminator: {} → {} → {} → 1  ({} params)",
        IMG_SIZE,
        D_HIDDEN1,
        D_HIDDEN2,
        discriminator.layer1.parameter_count()
            + discriminator.layer2.parameter_count()
            + discriminator.layer3.parameter_count()
    );

    println!("Starting GAN training...");
    let mut shuffle_rng = SimpleRng::new(1337);
    train_gan(
        &mut generator,
        &mut discriminator,
        train_images_train,
        num_train,
        &mut shuffle_rng,
        epochs,
        batch_size,
        val_images,
        val_images.len() / IMG_SIZE,
        label_smoothing,
        noise_dim,
    );

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Total program time: {:.2} seconds", total_time);
    println!("===========================");
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests;
