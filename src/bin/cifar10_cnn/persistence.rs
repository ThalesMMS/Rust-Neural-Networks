use super::*;
use rust_neural_networks::persistence::{load_layers_from_file, save_layers_to_file};

/// Serialize a Cnn's layers and parameters to a binary file.
///
/// The file contains a compact, layer-indexed binary representation of the model
/// suitable for later deserialization. This function will panic on file creation,
/// write failures, or unsupported layer types.
///
/// # Examples
///
/// ```
/// # use crate::{SimpleRng, init_cnn, save_model};
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng, None).unwrap();
/// save_model(&model, "cifar10_cnn_model.bin");
/// ```
pub(crate) fn save_model(model: &Cnn, filename: &str) {
    save_layers_to_file(filename, &model.layers).expect("Failed to save model layers");
    println!("Model saved to: {}", filename);
}

/// Reconstructs a `Cnn` from a binary file containing serialized layer data.
///
/// The file must follow the shared layer-stack checkpoint layout: a little-endian
/// `u32` layer count followed by one-byte layer type IDs and type-specific payloads.
///
/// # Arguments
///
/// * `filename` - Path to the binary model file.
///
/// # Returns
///
/// A `Cnn` whose `layers` vector is populated from the file's serialized data.
///
/// # Panics
///
/// Panics if the file cannot be opened or read, if the file is truncated or
/// malformed, or if an unknown layer type ID is encountered.
///
/// # Examples
///
/// ```ignore
/// let model = load_model("cifar10_cnn_model_best.bin");
/// ```
#[allow(dead_code)]
pub(crate) fn load_model(filename: &str) -> Cnn {
    let layers = load_layers_from_file(filename).expect("Failed to load model layers");
    println!("Model loaded from: {}", filename);
    Cnn { layers }
}
