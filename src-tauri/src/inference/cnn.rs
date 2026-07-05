//! Inference-only port of `mnist_cnn`'s bespoke checkpoint format (see
//! `src/bin/mnist_cnn/io.rs`): conv metadata + weights/biases, then FC
//! metadata + weights/biases. Forward pass mirrors
//! `src/bin/mnist_cnn/model.rs`: conv -> ReLU -> 2x2 max-pool -> dense -> softmax.

use super::{read_f32_vec, read_i32_le, to_prediction, Prediction};
use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::utils::activations::{relu_inplace, softmax_rows};
use std::fs;

const NUM_INPUTS: usize = 784;
const NUM_CLASSES: usize = 10;

fn maxpool_2x2(input: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let out_h = height / 2;
    let out_w = width / 2;
    let mut out = vec![0.0f32; channels * out_h * out_w];
    for c in 0..channels {
        let in_base = c * height * width;
        let out_base = c * out_h * out_w;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut best = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let iy = oy * 2 + dy;
                        let ix = ox * 2 + dx;
                        let v = input[in_base + iy * width + ix];
                        if v > best {
                            best = v;
                        }
                    }
                }
                out[out_base + oy * out_w + ox] = best;
            }
        }
    }
    out
}

#[tauri::command]
pub fn predict_cnn(checkpoint_path: String, pixels: Vec<f32>) -> Result<Prediction, String> {
    if pixels.len() != NUM_INPUTS {
        return Err(format!("expected {NUM_INPUTS} pixels, got {}", pixels.len()));
    }

    let path = crate::paths::resolve_relative(&checkpoint_path)?;
    let bytes = fs::read(&path).map_err(|e| format!("failed to read {checkpoint_path}: {e}"))?;
    let mut offset = 0usize;

    let out_channels = read_i32_le(&bytes, &mut offset)? as usize;
    let in_channels = read_i32_le(&bytes, &mut offset)? as usize;
    let kernel_size = read_i32_le(&bytes, &mut offset)? as usize;
    let padding = read_i32_le(&bytes, &mut offset)? as isize;
    let stride = read_i32_le(&bytes, &mut offset)? as usize;
    let input_height = read_i32_le(&bytes, &mut offset)? as usize;
    let input_width = read_i32_le(&bytes, &mut offset)? as usize;

    let conv_weights =
        read_f32_vec(&bytes, &mut offset, out_channels * in_channels * kernel_size * kernel_size)?;
    let conv_biases = read_f32_vec(&bytes, &mut offset, out_channels)?;

    let fc_input_size = read_i32_le(&bytes, &mut offset)? as usize;
    let fc_output_size = read_i32_le(&bytes, &mut offset)? as usize;
    let fc_weights = read_f32_vec(&bytes, &mut offset, fc_input_size * fc_output_size)?;
    let fc_biases = read_f32_vec(&bytes, &mut offset, fc_output_size)?;

    if fc_output_size != NUM_CLASSES {
        return Err(format!("unexpected output size: {fc_output_size}"));
    }

    let conv_layer = Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        input_height,
        input_width,
        conv_weights,
        conv_biases,
    );
    let fc_layer = DenseLayer::new_with_weights(fc_input_size, fc_output_size, fc_weights, fc_biases);

    let mut conv_out = vec![0.0f32; out_channels * input_height * input_width];
    conv_layer.forward(&pixels, &mut conv_out, 1);
    relu_inplace(&mut conv_out);

    let pooled = maxpool_2x2(&conv_out, out_channels, input_height, input_width);
    if pooled.len() != fc_input_size {
        return Err(format!(
            "pooled feature size {} does not match FC input size {fc_input_size}",
            pooled.len()
        ));
    }

    let mut logits = vec![0.0f32; fc_output_size];
    fc_layer.forward(&pooled, &mut logits, 1);
    softmax_rows(&mut logits, 1, fc_output_size);

    Ok(to_prediction(logits))
}
