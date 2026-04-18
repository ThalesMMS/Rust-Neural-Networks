//! WASM module for MNIST inference in the browser
//!
//! This module provides WebAssembly bindings for neural network inference.
//! It exposes a MnistClassifier that can be loaded from binary model data
//! and used to classify handwritten digits in real-time.

use wasm_bindgen::prelude::*;

// Pure Rust matrix operations (BLAS-free for WASM compatibility)
pub mod matrix_ops;

// Activation functions for neural networks
pub mod activations;

// Dense layer implementation for WASM
pub mod layer;

// MNIST model structure for inference
pub mod model;

use model::MnistModel;

/// Set up panic hook for better error messages in the browser console
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// WebAssembly-compatible MNIST digit classifier.
///
/// This struct wraps the MnistModel and provides JavaScript-compatible
/// methods for loading models and making predictions in the browser.
///
/// # Example Usage (JavaScript)
///
/// ```javascript
/// import init, { MnistClassifier } from './mnist_wasm.js';
///
/// async function main() {
///     // Initialize WASM module
///     await init();
///
///     // Load model from binary file
///     const response = await fetch('mnist_model.bin');
///     const modelBytes = new Uint8Array(await response.arrayBuffer());
///     const classifier = MnistClassifier.new(modelBytes);
///
///     // Prepare image data (28×28 pixels, normalized to [0, 1])
///     const imageData = new Float32Array(784);
///     // ... fill with pixel values ...
///
///     // Get probabilities for all digits
///     const probabilities = classifier.predict(imageData);
///     console.log('Probabilities:', probabilities);
///
///     // Get predicted digit class
///     const digit = classifier.predict_digit(imageData);
///     console.log('Predicted digit:', digit);
/// }
/// ```
#[wasm_bindgen]
pub struct MnistClassifier {
    model: MnistModel,
}

#[wasm_bindgen]
impl MnistClassifier {
    /// Creates a new MNIST classifier from binary model data.
    ///
    /// The binary format must match the format saved by the Rust training code:
    /// - 3 × i32 (little-endian): input_size, hidden_size, output_size
    /// - Hidden layer weights (input_size × hidden_size floats)
    /// - Hidden layer biases (hidden_size floats)
    /// - Output layer weights (hidden_size × output_size floats)
    /// - Output layer biases (output_size floats)
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Uint8Array containing the binary model data
    ///
    /// # Returns
    ///
    /// Returns a new MnistClassifier instance.
    ///
    /// # Errors
    ///
    /// Throws a JavaScript error if the model data is invalid or corrupted.
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const response = await fetch('mnist_model.bin');
    /// const modelBytes = new Uint8Array(await response.arrayBuffer());
    /// const classifier = MnistClassifier.new(modelBytes);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<MnistClassifier, JsValue> {
        let model = MnistModel::from_bytes(model_bytes)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        Ok(MnistClassifier { model })
    }

    /// Predicts probabilities for all 10 digit classes (0-9).
    ///
    /// # Arguments
    ///
    /// * `image_data` - Float32Array of 784 pixels (28×28 flattened image),
    ///   normalized to [0, 1] range
    ///
    /// # Returns
    ///
    /// Returns a Float32Array of 10 probabilities, one for each digit class.
    /// The probabilities sum to 1.0.
    ///
    /// # Errors
    ///
    /// Throws a JavaScript error if the input size is not 784.
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const imageData = new Float32Array(784);
    /// // ... fill imageData with normalized pixel values [0, 1] ...
    ///
    /// const probabilities = classifier.predict(imageData);
    /// console.log('Digit 0:', probabilities[0]);
    /// console.log('Digit 1:', probabilities[1]);
    /// // ... etc ...
    /// ```
    pub fn predict(&self, image_data: &[f32]) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != 784 {
            return Err(JsValue::from_str(&format!(
                "Invalid input size: expected 784 pixels, got {}",
                image_data.len()
            )));
        }

        let probabilities = self.model.predict(image_data);
        Ok(probabilities)
    }

    /// Predicts the most likely digit class (0-9).
    ///
    /// # Arguments
    ///
    /// * `image_data` - Float32Array of 784 pixels (28×28 flattened image),
    ///   normalized to [0, 1] range
    ///
    /// # Returns
    ///
    /// Returns the predicted digit (0-9) as an integer.
    ///
    /// # Errors
    ///
    /// Throws a JavaScript error if the input size is not 784.
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const imageData = new Float32Array(784);
    /// // ... fill imageData with normalized pixel values [0, 1] ...
    ///
    /// const digit = classifier.predict_digit(imageData);
    /// console.log('Predicted digit:', digit);
    /// ```
    pub fn predict_digit(&self, image_data: &[f32]) -> Result<usize, JsValue> {
        if image_data.len() != 784 {
            return Err(JsValue::from_str(&format!(
                "Invalid input size: expected 784 pixels, got {}",
                image_data.len()
            )));
        }

        let digit = self.model.predict_class(image_data);
        Ok(digit)
    }

    /// Returns the expected input size (always 784 for MNIST).
    ///
    /// This is a helper method for JavaScript code to validate input dimensions.
    ///
    /// # Returns
    ///
    /// Returns 784 (28×28 pixels).
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const inputSize = classifier.input_size();
    /// console.log('Expected input size:', inputSize); // 784
    /// ```
    pub fn input_size(&self) -> usize {
        784
    }

    /// Returns the number of output classes (always 10 for digits 0-9).
    ///
    /// This is a helper method for JavaScript code to validate output dimensions.
    ///
    /// # Returns
    ///
    /// Returns 10 (digits 0-9).
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const numClasses = classifier.num_classes();
    /// console.log('Number of classes:', numClasses); // 10
    /// ```
    pub fn num_classes(&self) -> usize {
        10
    }
}

// Tests for wasm-bindgen functions can only run on wasm32 target
#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    /// Helper function to create a minimal valid model binary for testing
    fn create_test_model_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();

        let input_size = 784;
        let hidden_size = 512;
        let output_size = 10;

        // Write dimensions
        bytes.extend_from_slice(&(input_size as i32).to_le_bytes());
        bytes.extend_from_slice(&(hidden_size as i32).to_le_bytes());
        bytes.extend_from_slice(&(output_size as i32).to_le_bytes());

        // Write hidden layer weights (all zeros for simplicity)
        for _ in 0..(input_size * hidden_size) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write hidden layer biases (all zeros)
        for _ in 0..hidden_size {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write output layer weights (all zeros)
        for _ in 0..(hidden_size * output_size) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Write output layer biases (all zeros)
        for _ in 0..output_size {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        bytes
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_new() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes);
        assert!(classifier.is_ok(), "Classifier should be created successfully");
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_new_invalid_data() {
        let bytes = vec![0u8; 10]; // Too short to be valid
        let classifier = MnistClassifier::new(&bytes);
        assert!(classifier.is_err(), "Should fail with invalid data");
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_predict() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        let image = vec![0.0f32; 784];
        let result = classifier.predict(&image);

        assert!(result.is_ok(), "Prediction should succeed");
        let probabilities = result.unwrap();
        assert_eq!(probabilities.len(), 10, "Should return 10 probabilities");

        // Check probabilities sum to ~1.0
        let sum: f32 = probabilities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_predict_invalid_size() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        let image = vec![0.0f32; 100]; // Wrong size
        let result = classifier.predict(&image);

        assert!(result.is_err(), "Should fail with wrong input size");
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_predict_digit() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        let image = vec![0.0f32; 784];
        let result = classifier.predict_digit(&image);

        assert!(result.is_ok(), "Digit prediction should succeed");
        let digit = result.unwrap();
        assert!(digit < 10, "Predicted digit should be 0-9, got {}", digit);
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_predict_digit_invalid_size() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        let image = vec![0.0f32; 100]; // Wrong size
        let result = classifier.predict_digit(&image);

        assert!(result.is_err(), "Should fail with wrong input size");
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_input_size() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        assert_eq!(classifier.input_size(), 784, "Input size should be 784");
    }

    #[wasm_bindgen_test]
    fn test_mnist_classifier_num_classes() {
        let bytes = create_test_model_bytes();
        let classifier = MnistClassifier::new(&bytes).unwrap();

        assert_eq!(classifier.num_classes(), 10, "Should have 10 classes");
    }
}
