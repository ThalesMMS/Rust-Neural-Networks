//! MNIST model structure for WebAssembly inference
//!
//! This module provides a complete MNIST digit recognition model that can be loaded
//! from a binary file and used for inference in the browser.

use crate::activations::{relu_inplace, softmax_rows};
use crate::layer::DenseLayer;

/// MNIST model architecture constants
pub const NUM_INPUTS: usize = 784; // 28x28 pixels
pub const NUM_HIDDEN: usize = 512; // Hidden layer size
pub const NUM_OUTPUTS: usize = 10; // Digits 0-9

/// MNIST neural network with one hidden layer.
///
/// Architecture: 784 -> 512 (ReLU) -> 10 (Softmax)
///
/// This model is designed for inference only (no training support).
/// It loads pre-trained weights from a binary file format.
pub struct MnistModel {
    hidden_layer: DenseLayer,
    output_layer: DenseLayer,
}

impl MnistModel {
    /// Creates a new MNIST model from binary data.
    ///
    /// The binary format is:
    /// 1. Three i32 (little-endian): input_size, hidden_size, output_size
    /// 2. Hidden layer weights (input_size × hidden_size floats, little-endian)
    /// 3. Hidden layer biases (hidden_size floats, little-endian)
    /// 4. Output layer weights (hidden_size × output_size floats, little-endian)
    /// 5. Output layer biases (output_size floats, little-endian)
    ///
    /// # Arguments
    ///
    /// * `bytes` - Binary data containing the model weights and biases
    ///
    /// # Returns
    ///
    /// Returns `Ok(MnistModel)` if the model is loaded successfully,
    /// or `Err(String)` if the data is invalid or corrupted.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mnist_wasm::model::MnistModel;
    ///
    /// let model_data = std::fs::read("mnist_model.bin").unwrap();
    /// let model = MnistModel::from_bytes(&model_data).unwrap();
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut offset = 0;

        // Helper to read i32 (little-endian)
        let read_i32 = |data: &[u8], offset: &mut usize| -> Result<i32, String> {
            if *offset + 4 > data.len() {
                return Err("Unexpected end of file while reading dimensions".to_string());
            }
            let value = i32::from_le_bytes([
                data[*offset],
                data[*offset + 1],
                data[*offset + 2],
                data[*offset + 3],
            ]);
            *offset += 4;
            Ok(value)
        };

        // Helper to read f32 (little-endian)
        let read_f32 = |data: &[u8], offset: &mut usize| -> Result<f32, String> {
            if *offset + 4 > data.len() {
                return Err("Unexpected end of file while reading weights".to_string());
            }
            let value = f32::from_le_bytes([
                data[*offset],
                data[*offset + 1],
                data[*offset + 2],
                data[*offset + 3],
            ]);
            *offset += 4;
            Ok(value)
        };

        // Read dimensions
        let input_size = read_i32(bytes, &mut offset)? as usize;
        let hidden_size = read_i32(bytes, &mut offset)? as usize;
        let output_size = read_i32(bytes, &mut offset)? as usize;

        // Validate dimensions
        if input_size != NUM_INPUTS {
            return Err(format!(
                "Invalid input size: expected {}, got {}",
                NUM_INPUTS, input_size
            ));
        }
        if hidden_size != NUM_HIDDEN {
            return Err(format!(
                "Invalid hidden size: expected {}, got {}",
                NUM_HIDDEN, hidden_size
            ));
        }
        if output_size != NUM_OUTPUTS {
            return Err(format!(
                "Invalid output size: expected {}, got {}",
                NUM_OUTPUTS, output_size
            ));
        }

        // Read hidden layer weights
        let mut hidden_weights = Vec::with_capacity(input_size * hidden_size);
        for _ in 0..(input_size * hidden_size) {
            hidden_weights.push(read_f32(bytes, &mut offset)?);
        }

        // Read hidden layer biases
        let mut hidden_biases = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            hidden_biases.push(read_f32(bytes, &mut offset)?);
        }

        // Read output layer weights
        let mut output_weights = Vec::with_capacity(hidden_size * output_size);
        for _ in 0..(hidden_size * output_size) {
            output_weights.push(read_f32(bytes, &mut offset)?);
        }

        // Read output layer biases
        let mut output_biases = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            output_biases.push(read_f32(bytes, &mut offset)?);
        }

        // Create layers
        let hidden_layer = DenseLayer::new(input_size, hidden_size, hidden_weights, hidden_biases);
        let output_layer = DenseLayer::new(hidden_size, output_size, output_weights, output_biases);

        Ok(Self {
            hidden_layer,
            output_layer,
        })
    }

    /// Runs inference on a single input sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened 28×28 image (784 pixels), normalized to [0, 1]
    ///
    /// # Returns
    ///
    /// Returns a vector of 10 probabilities (one for each digit 0-9).
    /// The probabilities sum to 1.0.
    ///
    /// # Panics
    ///
    /// Panics if input length is not 784.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::model::MnistModel;
    ///
    /// // Create a dummy model for testing (not a real trained model)
    /// let input_size = 784;
    /// let hidden_size = 512;
    /// let output_size = 10;
    ///
    /// // Build a minimal binary model (header only, weights as zeros for this example)
    /// let mut model_bytes = Vec::new();
    /// model_bytes.extend_from_slice(&(input_size as i32).to_le_bytes());
    /// model_bytes.extend_from_slice(&(hidden_size as i32).to_le_bytes());
    /// model_bytes.extend_from_slice(&(output_size as i32).to_le_bytes());
    ///
    /// // Add weights and biases (all zeros for this example)
    /// for _ in 0..(input_size * hidden_size) {
    ///     model_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    /// }
    /// for _ in 0..hidden_size {
    ///     model_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    /// }
    /// for _ in 0..(hidden_size * output_size) {
    ///     model_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    /// }
    /// for _ in 0..output_size {
    ///     model_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    /// }
    ///
    /// let model = MnistModel::from_bytes(&model_bytes).unwrap();
    ///
    /// // Run prediction on a zero image
    /// let input = vec![0.0; 784];
    /// let probabilities = model.predict(&input);
    ///
    /// assert_eq!(probabilities.len(), 10);
    /// // With zero weights and biases, all outputs should be equal (1/10)
    /// let sum: f32 = probabilities.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1.0");
    /// ```
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            NUM_INPUTS,
            "Input must be 784 pixels (28x28 flattened image)"
        );

        // Allocate buffers for hidden and output activations
        let mut hidden = vec![0.0; NUM_HIDDEN];
        let mut output = vec![0.0; NUM_OUTPUTS];

        // Forward pass: input -> hidden (with ReLU)
        self.hidden_layer.forward(input, &mut hidden, 1);
        relu_inplace(&mut hidden);

        // Forward pass: hidden -> output (with softmax)
        self.output_layer.forward(&hidden, &mut output, 1);
        softmax_rows(&mut output, 1, NUM_OUTPUTS);

        output
    }

    /// Returns the predicted digit class (0-9) for the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened 28×28 image (784 pixels), normalized to [0, 1]
    ///
    /// # Returns
    ///
    /// The digit with the highest probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnist_wasm::model::MnistModel;
    ///
    /// // Create a dummy model (same as predict example)
    /// let input_size = 784;
    /// let hidden_size = 512;
    /// let output_size = 10;
    ///
    /// let mut model_bytes = Vec::new();
    /// model_bytes.extend_from_slice(&(input_size as i32).to_le_bytes());
    /// model_bytes.extend_from_slice(&(hidden_size as i32).to_le_bytes());
    /// model_bytes.extend_from_slice(&(output_size as i32).to_le_bytes());
    ///
    /// for _ in 0..(input_size * hidden_size + hidden_size + hidden_size * output_size + output_size) {
    ///     model_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    /// }
    ///
    /// let model = MnistModel::from_bytes(&model_bytes).unwrap();
    ///
    /// let input = vec![0.0; 784];
    /// let predicted_class = model.predict_class(&input);
    ///
    /// assert!(predicted_class < 10, "Predicted class should be 0-9");
    /// ```
    pub fn predict_class(&self, input: &[f32]) -> usize {
        let probabilities = self.predict(input);

        // Find the index of the maximum probability
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a minimal valid model binary for testing
    fn create_test_model_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write dimensions
        bytes.extend_from_slice(&(NUM_INPUTS as i32).to_le_bytes());
        bytes.extend_from_slice(&(NUM_HIDDEN as i32).to_le_bytes());
        bytes.extend_from_slice(&(NUM_OUTPUTS as i32).to_le_bytes());

        // Write hidden layer weights (all zeros for simplicity)
        for _ in 0..(NUM_INPUTS * NUM_HIDDEN) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write hidden layer biases (all zeros)
        for _ in 0..NUM_HIDDEN {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write output layer weights (all zeros)
        for _ in 0..(NUM_HIDDEN * NUM_OUTPUTS) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write output layer biases (all zeros)
        for _ in 0..NUM_OUTPUTS {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        bytes
    }

    #[test]
    fn test_model_from_bytes() {
        let bytes = create_test_model_bytes();
        let model = MnistModel::from_bytes(&bytes);
        assert!(model.is_ok(), "Model should load successfully");
    }

    #[test]
    fn test_model_from_bytes_invalid_dimensions() {
        let mut bytes = Vec::new();
        // Write invalid dimensions
        bytes.extend_from_slice(&100i32.to_le_bytes());
        bytes.extend_from_slice(&200i32.to_le_bytes());
        bytes.extend_from_slice(&10i32.to_le_bytes());

        let model = MnistModel::from_bytes(&bytes);
        assert!(model.is_err(), "Model should fail with invalid dimensions");
    }

    #[test]
    fn test_model_from_bytes_truncated() {
        let mut bytes = Vec::new();
        // Write only dimensions, no weights
        bytes.extend_from_slice(&(NUM_INPUTS as i32).to_le_bytes());
        bytes.extend_from_slice(&(NUM_HIDDEN as i32).to_le_bytes());
        bytes.extend_from_slice(&(NUM_OUTPUTS as i32).to_le_bytes());

        let model = MnistModel::from_bytes(&bytes);
        assert!(model.is_err(), "Model should fail with truncated data");
    }

    #[test]
    fn model_predict() {
        let bytes = create_test_model_bytes();
        let model = MnistModel::from_bytes(&bytes).unwrap();

        // Create a test input (all zeros)
        let input = vec![0.0; NUM_INPUTS];

        // Run prediction
        let probabilities = model.predict(&input);

        // Check output shape
        assert_eq!(probabilities.len(), NUM_OUTPUTS);

        // Check that probabilities sum to 1.0
        let sum: f32 = probabilities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1.0, got {}",
            sum
        );

        // Check that all probabilities are non-negative
        for prob in &probabilities {
            assert!(*prob >= 0.0, "Probability should be non-negative");
            assert!(*prob <= 1.0, "Probability should be <= 1.0");
        }
    }

    #[test]
    fn test_model_predict_class() {
        let bytes = create_test_model_bytes();
        let model = MnistModel::from_bytes(&bytes).unwrap();

        let input = vec![0.0; NUM_INPUTS];
        let predicted_class = model.predict_class(&input);

        // With zero weights, the prediction should be some digit 0-9
        assert!(
            predicted_class < 10,
            "Predicted class should be between 0 and 9"
        );
    }

    #[test]
    #[should_panic(expected = "Input must be 784 pixels")]
    fn test_model_predict_wrong_input_size() {
        let bytes = create_test_model_bytes();
        let model = MnistModel::from_bytes(&bytes).unwrap();

        // Try to predict with wrong input size
        let input = vec![0.0; 100];
        model.predict(&input);
    }

    #[test]
    fn test_model_predict_with_nonzero_input() {
        let bytes = create_test_model_bytes();
        let model = MnistModel::from_bytes(&bytes).unwrap();

        // Create input with some non-zero values
        let mut input = vec![0.0; NUM_INPUTS];
        for i in 0..100 {
            input[i] = 0.5;
        }

        let probabilities = model.predict(&input);

        // Basic sanity checks
        assert_eq!(probabilities.len(), NUM_OUTPUTS);
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
