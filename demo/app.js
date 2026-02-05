/**
 * MNIST Digit Recognizer - Main Application Controller
 *
 * Coordinates WASM initialization, canvas drawing, and prediction display.
 * Provides an interactive interface for drawing digits and seeing real-time
 * classification results from the trained neural network.
 */

import { MnistWasmWrapper } from './wasm_wrapper.js';

/**
 * Main application controller for MNIST digit recognition demo.
 *
 * Handles:
 * - WASM module initialization and model loading
 * - Canvas drawing with mouse/touch input
 * - Image preprocessing (scaling to 28x28, normalization)
 * - Real-time prediction and UI updates
 *
 * @example
 * const app = new DigitRecognizerApp('drawingCanvas', 'mnist_model.bin');
 * await app.initialize();
 */
export class DigitRecognizerApp {
    /**
     * Creates a new digit recognizer application.
     *
     * @param {string} canvasId - ID of the canvas element for drawing
     * @param {string} predictionsContainerId - ID of the predictions container element
     * @param {string} clearButtonId - ID of the clear button element
     * @param {string} predictButtonId - ID of the predict button element
     * @param {string} statusMessageId - ID of the status message element
     * @param {Object} options - Configuration options
     * @param {number} options.canvasSize - Display size of canvas (default: 280)
     * @param {number} options.imageSize - Output image size (default: 28)
     * @param {number} options.brushSize - Drawing brush size (default: 20)
     * @param {string} options.modelUrl - URL path to the binary model file (default: 'mnist_model.bin')
     */
    constructor(canvasId, predictionsContainerId, clearButtonId, predictButtonId, statusMessageId, options = {}) {
        this.canvasId = canvasId;
        this.predictionsContainerId = predictionsContainerId;
        this.clearButtonId = clearButtonId;
        this.predictButtonId = predictButtonId;
        this.statusMessageId = statusMessageId;
        this.modelUrl = options.modelUrl || 'mnist_model.bin';

        // Configuration
        this.canvasSize = options.canvasSize || 280;
        this.imageSize = options.imageSize || 28;
        this.brushSize = options.brushSize || 20;

        // Canvas and drawing state
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.lastX = null;
        this.lastY = null;

        // Hidden canvas for 28x28 image
        this.hiddenCanvas = null;
        this.hiddenCtx = null;

        // WASM classifier
        this.classifier = null;

        // UI elements
        this.predictionsContainer = null;
        this.clearButton = null;
        this.predictButton = null;
        this.statusMessage = null;
    }

    /**
     * Initializes the application: sets up canvas and loads WASM model.
     *
     * This must be called before the app can be used.
     * Progress updates are sent via onStatusUpdate callback if set.
     *
     * @returns {Promise<void>}
     * @throws {Error} If initialization fails
     *
     * @example
     * const app = new DigitRecognizerApp('canvas', 'model.bin');
     * app.onStatusUpdate = (msg) => console.log(msg);
     * await app.initialize();
     */
    async initialize() {
        try {
            this.updateStatus('Initializing canvas...');
            this.setupCanvas();

            this.updateStatus('Loading WASM module and model...');
            this.classifier = new MnistWasmWrapper();
            await this.classifier.initialize(this.modelUrl);

            this.updateStatus('Ready! Draw a digit on the canvas.');
        } catch (error) {
            this.updateStatus(`Error: ${error.message}`);
            throw error;
        }
    }

    /**
     * Sets up the main drawing canvas and hidden 28x28 canvas.
     * Initializes mouse and touch event listeners and UI elements.
     *
     * @private
     */
    setupCanvas() {
        // Get main canvas
        this.canvas = document.getElementById(this.canvasId);
        if (!this.canvas) {
            throw new Error(`Canvas element '${this.canvasId}' not found`);
        }

        this.canvas.width = this.canvasSize;
        this.canvas.height = this.canvasSize;
        this.ctx = this.canvas.getContext('2d');

        // Style canvas
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvasSize, this.canvasSize);
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Create hidden canvas for 28x28 image
        this.hiddenCanvas = document.createElement('canvas');
        this.hiddenCanvas.width = this.imageSize;
        this.hiddenCanvas.height = this.imageSize;
        this.hiddenCtx = this.hiddenCanvas.getContext('2d');

        // Get UI elements
        this.predictionsContainer = document.getElementById(this.predictionsContainerId);
        this.clearButton = document.getElementById(this.clearButtonId);
        this.predictButton = document.getElementById(this.predictButtonId);
        this.statusMessage = document.getElementById(this.statusMessageId);

        if (!this.predictionsContainer || !this.clearButton || !this.predictButton || !this.statusMessage) {
            throw new Error('Required UI elements not found');
        }

        // Wire up button click handlers
        this.clearButton.addEventListener('click', () => this.clearCanvas());
        this.predictButton.addEventListener('click', () => this.predict());

        // Bind canvas event listeners
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseUp(e));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.handleTouchEnd(e));
    }

    /**
     * Handles mouse button press - starts drawing.
     *
     * @private
     * @param {MouseEvent} event - Mouse event
     */
    handleMouseDown(event) {
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastX = event.clientX - rect.left;
        this.lastY = event.clientY - rect.top;
    }

    /**
     * Handles mouse movement - draws on canvas if mouse is pressed.
     *
     * @private
     * @param {MouseEvent} event - Mouse event
     */
    handleMouseMove(event) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        this.drawLine(this.lastX, this.lastY, x, y);

        this.lastX = x;
        this.lastY = y;
    }

    /**
     * Handles mouse button release - stops drawing and triggers prediction.
     *
     * @private
     * @param {MouseEvent} event - Mouse event
     */
    handleMouseUp(event) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.predict();
        }
    }

    /**
     * Handles touch start - begins drawing on mobile devices.
     *
     * @private
     * @param {TouchEvent} event - Touch event
     */
    handleTouchStart(event) {
        event.preventDefault();
        const touch = event.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        this.isDrawing = true;
        this.lastX = touch.clientX - rect.left;
        this.lastY = touch.clientY - rect.top;
    }

    /**
     * Handles touch movement - draws on canvas.
     *
     * @private
     * @param {TouchEvent} event - Touch event
     */
    handleTouchMove(event) {
        event.preventDefault();
        if (!this.isDrawing) return;

        const touch = event.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;

        this.drawLine(this.lastX, this.lastY, x, y);

        this.lastX = x;
        this.lastY = y;
    }

    /**
     * Handles touch end - stops drawing and triggers prediction.
     *
     * @private
     * @param {TouchEvent} event - Touch event
     */
    handleTouchEnd(event) {
        event.preventDefault();
        if (this.isDrawing) {
            this.isDrawing = false;
            this.predict();
        }
    }

    /**
     * Draws a line on the canvas from (x1, y1) to (x2, y2).
     *
     * @private
     * @param {number} x1 - Start x coordinate
     * @param {number} y1 - Start y coordinate
     * @param {number} x2 - End x coordinate
     * @param {number} y2 - End y coordinate
     */
    drawLine(x1, y1, x2, y2) {
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
    }

    /**
     * Clears the canvas and resets predictions.
     *
     * @example
     * app.clearCanvas();
     */
    clearCanvas() {
        // Clear main canvas
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvasSize, this.canvasSize);

        // Clear hidden canvas
        this.hiddenCtx.fillStyle = 'black';
        this.hiddenCtx.fillRect(0, 0, this.imageSize, this.imageSize);

        // Reset predictions display
        const zeroPredictions = new Float32Array(10).fill(0);
        this.updatePredictions(zeroPredictions, -1);
    }

    /**
     * Extracts the drawing as a 28x28 grayscale image, normalized to [0, 1].
     *
     * @returns {Float32Array} Flattened 784-element array of pixel values
     *
     * @example
     * const imageData = app.getImageData();
     * console.log(imageData.length); // 784
     */
    getImageData() {
        // Draw scaled-down version to hidden canvas
        this.hiddenCtx.fillStyle = 'black';
        this.hiddenCtx.fillRect(0, 0, this.imageSize, this.imageSize);
        this.hiddenCtx.drawImage(this.canvas, 0, 0, this.canvasSize, this.canvasSize,
                                             0, 0, this.imageSize, this.imageSize);

        // Get pixel data
        const imageData = this.hiddenCtx.getImageData(0, 0, this.imageSize, this.imageSize);
        const pixels = imageData.data;

        // Convert to grayscale and normalize to [0, 1]
        const normalized = new Float32Array(this.imageSize * this.imageSize);
        for (let i = 0; i < normalized.length; i++) {
            // RGBA format: take R channel (grayscale, so R=G=B)
            // Normalize from [0, 255] to [0, 1]
            normalized[i] = pixels[i * 4] / 255.0;
        }

        return normalized;
    }

    /**
     * Runs prediction on the current drawing and updates the UI.
     *
     * Extracts the canvas image, calls the WASM classifier,
     * and updates the prediction display.
     *
     * @example
     * app.predict();
     */
    predict() {
        if (!this.classifier || !this.classifier.isReady()) {
            this.updateStatus('Model not loaded yet');
            return;
        }

        try {
            // Get normalized image data
            const imageData = this.getImageData();

            // Run prediction
            const probabilities = this.classifier.predict(imageData);
            const predictedDigit = this.classifier.predictDigit(imageData);

            // Update UI
            this.updatePredictions(probabilities, predictedDigit);
        } catch (error) {
            this.updateStatus(`Prediction error: ${error.message}`);
        }
    }

    /**
     * Updates the status message element.
     *
     * @private
     * @param {string} message - Status message
     */
    updateStatus(message) {
        if (this.statusMessage) {
            this.statusMessage.textContent = message;
        }
    }

    /**
     * Updates the predictions display with probability bars.
     *
     * @private
     * @param {Float32Array} probabilities - Array of 10 probabilities (0-9)
     * @param {number} predictedDigit - The predicted digit (0-9) or -1 for no prediction
     */
    updatePredictions(probabilities, predictedDigit) {
        if (!this.predictionsContainer) return;

        // Get all prediction rows
        const rows = this.predictionsContainer.querySelectorAll('.prediction-row');

        rows.forEach((row, digit) => {
            const probability = probabilities[digit];
            const percentage = (probability * 100).toFixed(1);

            // Update probability bar width
            const bar = row.querySelector('.prediction-bar');
            if (bar) {
                bar.style.width = `${percentage}%`;

                // Highlight the predicted digit
                if (digit === predictedDigit) {
                    bar.style.backgroundColor = '#4CAF50'; // Green for predicted
                } else {
                    bar.style.backgroundColor = '#2196F3'; // Blue for others
                }
            }

            // Update probability label
            const label = row.querySelector('.probability-label');
            if (label) {
                label.textContent = `${percentage}%`;
            }
        });
    }

    /**
     * Cleans up resources when the app is no longer needed.
     *
     * @example
     * app.dispose();
     */
    dispose() {
        if (this.classifier) {
            this.classifier.dispose();
            this.classifier = null;
        }
    }
}

/**
 * Creates and initializes a digit recognizer application.
 *
 * This is a convenience function that combines construction and initialization.
 *
 * @param {string} canvasId - ID of the canvas element
 * @param {string} predictionsContainerId - ID of the predictions container element
 * @param {string} clearButtonId - ID of the clear button element
 * @param {string} predictButtonId - ID of the predict button element
 * @param {string} statusMessageId - ID of the status message element
 * @param {Object} options - Configuration options
 * @returns {Promise<DigitRecognizerApp>} Initialized application
 *
 * @example
 * const app = await createDigitRecognizer(
 *     'canvas', 'predictions', 'clear-btn', 'predict-btn', 'status', { modelUrl: 'model.bin' }
 * );
 */
export async function createDigitRecognizer(
    canvasId, predictionsContainerId, clearButtonId, predictButtonId, statusMessageId, options = {}
) {
    const app = new DigitRecognizerApp(
        canvasId, predictionsContainerId, clearButtonId, predictButtonId, statusMessageId, options
    );
    await app.initialize();
    return app;
}
