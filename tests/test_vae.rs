// Integration tests for the VariationalAutoencoder.
// Tests construction, forward/backward pass, loss computation, and training.
// Mirrors the patterns established in test_autoencoder.rs.

use rust_neural_networks::autoencoder::vae::VariationalAutoencoder;
use rust_neural_networks::utils::rng::SimpleRng;

#[path = "test_vae/losses.rs"]
mod losses;
#[path = "test_vae/stability.rs"]
mod stability;
#[path = "test_vae/structure.rs"]
mod structure;
#[path = "test_vae/training.rs"]
mod training;
