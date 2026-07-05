//! Inference-only port of `mnist_mlp`'s checkpoint format (identical to
//! `wasm/src/model.rs`): three little-endian i32 dims followed by weight and
//! bias arrays, feeding a 784->512(ReLU)->10(softmax) network.

use super::{read_f32_vec, read_i32_le, to_prediction, Prediction};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use std::fs;

const NUM_INPUTS: usize = 784;
const NUM_OUTPUTS: usize = 10;

#[tauri::command]
pub fn predict_mlp(checkpoint_path: String, pixels: Vec<f32>) -> Result<Prediction, String> {
    if pixels.len() != NUM_INPUTS {
        return Err(format!("expected {NUM_INPUTS} pixels, got {}", pixels.len()));
    }

    let path = crate::paths::resolve_relative(&checkpoint_path)?;
    let bytes = fs::read(&path).map_err(|e| format!("failed to read {checkpoint_path}: {e}"))?;
    let mut offset = 0usize;
    let input_size = read_i32_le(&bytes, &mut offset)? as usize;
    let hidden_size = read_i32_le(&bytes, &mut offset)? as usize;
    let output_size = read_i32_le(&bytes, &mut offset)? as usize;

    if input_size != NUM_INPUTS || output_size != NUM_OUTPUTS {
        return Err(format!(
            "unexpected model dimensions: {input_size} -> {hidden_size} -> {output_size}"
        ));
    }

    let hidden_weights = read_f32_vec(&bytes, &mut offset, input_size * hidden_size)?;
    let hidden_biases = read_f32_vec(&bytes, &mut offset, hidden_size)?;
    let output_weights = read_f32_vec(&bytes, &mut offset, hidden_size * output_size)?;
    let output_biases = read_f32_vec(&bytes, &mut offset, output_size)?;

    let hidden_layer =
        DenseLayer::new_with_weights(input_size, hidden_size, hidden_weights, hidden_biases);
    let output_layer =
        DenseLayer::new_with_weights(hidden_size, output_size, output_weights, output_biases);

    let mut hidden = vec![0.0f32; hidden_size];
    hidden_layer.forward(&pixels, &mut hidden, 1);
    relu_inplace(&mut hidden);

    let mut output = vec![0.0f32; output_size];
    output_layer.forward(&hidden, &mut output, 1);
    softmax_rows(&mut output, 1, output_size);

    Ok(to_prediction(output))
}
