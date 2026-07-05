//! Generic inference over any checkpoint written by the shared layer-stack
//! persistence format (`rust_neural_networks::persistence`), used by
//! `cifar10_cnn`. Mirrors the exact activation convention in
//! `src/bin/cifar10_cnn/model.rs::forward_pass`: ReLU is applied after every
//! `Conv2DLayer`, never after `DenseLayer` (the final classifier is a bare
//! Dense layer, so it naturally gets no ReLU). Softmax is applied to the
//! final logits for display purposes (training only needs argmax).
//!
//! Because this walks whatever layer stack the checkpoint contains, it
//! generalizes to deeper architectures (e.g. a future ResNet checkpoint)
//! without new code, as long as they follow the same activation convention.

use super::{to_prediction, Prediction};
use rust_neural_networks::layers::Conv2DLayer;
use rust_neural_networks::persistence::load_layers_from_file;
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};

const NUM_PIXELS: usize = 32 * 32 * 3;

#[tauri::command]
pub fn predict_cifar10(checkpoint_path: String, pixels: Vec<f32>) -> Result<Prediction, String> {
    if pixels.len() != NUM_PIXELS {
        return Err(format!("expected {NUM_PIXELS} pixels (32x32 RGB), got {}", pixels.len()));
    }

    let path = crate::paths::resolve_relative(&checkpoint_path)?;
    let layers = load_layers_from_file(&path)
        .map_err(|e| format!("failed to load {checkpoint_path}: {e}"))?;
    if layers.is_empty() {
        return Err("checkpoint contains no layers".to_string());
    }

    let mut current = pixels;
    for layer in &layers {
        let mut output = vec![0.0f32; layer.output_size()];
        layer.forward(&current, &mut output, 1);
        if layer.as_any().downcast_ref::<Conv2DLayer>().is_some() {
            relu_inplace(&mut output);
        }
        current = output;
    }

    let num_classes = current.len();
    softmax_rows(&mut current, 1, num_classes);
    Ok(to_prediction(current))
}
