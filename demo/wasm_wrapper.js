/**
 * WASM Wrapper for MNIST Classifier
 *
 * Provides a high-level JavaScript API for initializing the WebAssembly module,
 * loading the trained model, and making predictions on digit images.
 *
 * This wrapper combines WASM initialization, model loading, and provides
 * error handling for browser-based inference.
 */

import init, { MnistClassifier } from './pkg/mnist_wasm.js';

/**
 * Wrapper class for MNIST digit classification using WebAssembly.
 *
 * Handles WASM module initialization, model loading, and provides
 * a simplified API for making predictions.
 *
 * @example
 * const wrapper = new MnistWasmWrapper();
 * await wrapper.initialize('mnist_model.bin');
 *
 * const imageData = new Float32Array(784);
 * // ... fill imageData with normalized pixel values ...
 *
 * const probabilities = wrapper.predict(imageData);
 * const digit = wrapper.predictDigit(imageData);
 */
export class MnistWasmWrapper {
    /**
     * Creates a new MNIST WASM wrapper.
     * Call initialize() before using prediction methods.
     */
    constructor() {
        this.wasmInitialized = false;
        this.classifier = null;
    }

    /**
     * Initializes the WASM module and loads the model.
     *
     * This must be called before any prediction methods can be used.
     * The method handles both WASM initialization and model loading.
     *
     * @param {string} modelUrl - URL path to the binary model file
     * @returns {Promise<void>}
     * @throws {Error} If WASM initialization or model loading fails
     *
     * @example
     * const wrapper = new MnistWasmWrapper();
     * await wrapper.initialize('mnist_model.bin');
     */
    async initialize(modelUrl) {
        try {
            // Step 1: Initialize WASM module
            await init();
            this.wasmInitialized = true;

            // Step 2: Load model binary
            const response = await fetch(modelUrl);

            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            }

            const arrayBuffer = await response.arrayBuffer();
            const modelBytes = new Uint8Array(arrayBuffer);

            // Step 3: Create classifier instance
            this.classifier = new MnistClassifier(modelBytes);

        } catch (error) {
            this.wasmInitialized = false;
            this.classifier = null;
            throw new Error(`Failed to initialize MNIST classifier: ${error.message}`);
        }
    }

    /**
     * Checks if the wrapper is ready for predictions.
     *
     * @returns {boolean} True if WASM is initialized and model is loaded
     *
     * @example
     * if (wrapper.isReady()) {
     *     const predictions = wrapper.predict(imageData);
     * }
     */
    isReady() {
        return this.wasmInitialized && this.classifier !== null;
    }

    /**
     * Predicts probabilities for all 10 digit classes (0-9).
     *
     * @param {Float32Array} imageData - Flattened 28×28 image (784 pixels),
     *                                   normalized to [0, 1] range
     * @returns {Float32Array} Array of 10 probabilities (one per digit class)
     * @throws {Error} If wrapper is not initialized or input is invalid
     *
     * @example
     * const imageData = new Float32Array(784);
     * // ... fill with normalized pixel values ...
     *
     * const probabilities = wrapper.predict(imageData);
     * console.log('Probability of digit 5:', probabilities[5]);
     */
    predict(imageData) {
        if (!this.isReady()) {
            throw new Error('MNIST classifier not initialized. Call initialize() first.');
        }

        if (!(imageData instanceof Float32Array)) {
            throw new Error('Input must be a Float32Array');
        }

        if (imageData.length !== 784) {
            throw new Error(`Invalid input size: expected 784 pixels, got ${imageData.length}`);
        }

        try {
            const probabilities = this.classifier.predict(imageData);
            return new Float32Array(probabilities);
        } catch (error) {
            throw new Error(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Predicts probabilities for all 10 digit classes (0-9) and returns the hidden-layer
     * (Dense1) post-ReLU activations.
     *
     * @param {Float32Array} imageData - Flattened 28×28 image (784 pixels), normalized to [0, 1]
     * @returns {{ probabilities: Float32Array, hidden: Float32Array }}
     */
    predictWithHidden(imageData) {
        if (!this.isReady()) {
            throw new Error('MNIST classifier not initialized. Call initialize() first.');
        }

        if (!(imageData instanceof Float32Array)) {
            throw new Error('Input must be a Float32Array');
        }

        if (imageData.length !== 784) {
            throw new Error(`Invalid input size: expected 784 pixels, got ${imageData.length}`);
        }

        try {
            const result = this.classifier.predict_with_hidden(imageData);
            return {
                probabilities: new Float32Array(result.probabilities),
                hidden: new Float32Array(result.hidden)
            };
        } catch (error) {
            throw new Error(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Predicts the most likely digit class (0-9).
     *
     * @param {Float32Array} imageData - Flattened 28×28 image (784 pixels),
     *                                   normalized to [0, 1] range
     * @returns {number} Predicted digit (0-9)
     * @throws {Error} If wrapper is not initialized or input is invalid
     *
     * @example
     * const imageData = new Float32Array(784);
     * // ... fill with normalized pixel values ...
     *
     * const digit = wrapper.predictDigit(imageData);
     * console.log('Predicted digit:', digit);
     */
    predictDigit(imageData) {
        if (!this.isReady()) {
            throw new Error('MNIST classifier not initialized. Call initialize() first.');
        }

        if (!(imageData instanceof Float32Array)) {
            throw new Error('Input must be a Float32Array');
        }

        if (imageData.length !== 784) {
            throw new Error(`Invalid input size: expected 784 pixels, got ${imageData.length}`);
        }

        try {
            return this.classifier.predict_digit(imageData);
        } catch (error) {
            throw new Error(`Digit prediction failed: ${error.message}`);
        }
    }

    /**
     * Returns the expected input size (always 784 for MNIST).
     *
     * @returns {number} Expected input size in pixels
     *
     * @example
     * const inputSize = wrapper.getInputSize();
     * console.log('Expected input:', inputSize); // 784
     */
    getInputSize() {
        if (!this.isReady()) {
            return 784; // Default MNIST size
        }
        return this.classifier.input_size();
    }

    /**
     * Returns the number of output classes (always 10 for digits 0-9).
     *
     * @returns {number} Number of digit classes
     *
     * @example
     * const numClasses = wrapper.getNumClasses();
     * console.log('Number of classes:', numClasses); // 10
     */
    getNumClasses() {
        if (!this.isReady()) {
            return 10; // Default MNIST classes
        }
        return this.classifier.num_classes();
    }

    /**
     * Cleans up resources.
     *
     * Call this when the classifier is no longer needed to free memory.
     *
     * @example
     * wrapper.dispose();
     */
    dispose() {
        if (this.classifier) {
            this.classifier.free();
            this.classifier = null;
        }
        this.wasmInitialized = false;
    }
}

/**
 * Creates and initializes a new MNIST WASM wrapper.
 *
 * This is a convenience function that combines construction and initialization.
 *
 * @param {string} modelUrl - URL path to the binary model file
 * @returns {Promise<MnistWasmWrapper>} Initialized wrapper ready for predictions
 * @throws {Error} If initialization fails
 *
 * @example
 * const wrapper = await createMnistClassifier('mnist_model.bin');
 * const probabilities = wrapper.predict(imageData);
 */
export async function createMnistClassifier(modelUrl) {
    const wrapper = new MnistWasmWrapper();
    await wrapper.initialize(modelUrl);
    return wrapper;
}
