/**
 * MNIST Model Loader for WebAssembly Demo
 *
 * Fetches and parses binary model files in the format:
 * - 3 x i32 (little-endian): input_size, hidden_size, output_size
 * - input_size × hidden_size x f32 (little-endian): hidden layer weights
 * - hidden_size x f32 (little-endian): hidden layer biases
 * - hidden_size × output_size x f32 (little-endian): output layer weights
 * - output_size x f32 (little-endian): output layer biases
 */

/**
 * Loads an MNIST model from a binary file.
 *
 * @param {string} url - Path to the binary model file
 * @returns {Promise<Object>} Model data with dimensions, weights, and biases
 * @throws {Error} If fetch fails or model data is invalid
 *
 * @example
 * const model = await loadModel('mnist_model.bin');
 * console.log(`Model: ${model.input_size} -> ${model.hidden_size} -> ${model.output_size}`);
 */
export async function loadModel(url) {
    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        return parseModelBinary(arrayBuffer);
    } catch (error) {
        throw new Error(`Error loading model from ${url}: ${error.message}`);
    }
}

/**
 * Parses binary model data into a structured object.
 *
 * @param {ArrayBuffer} arrayBuffer - Raw binary model data
 * @returns {Object} Parsed model with dimensions, weights, and biases
 * @throws {Error} If binary data is invalid or corrupted
 *
 * @example
 * const arrayBuffer = await fetch('model.bin').then(r => r.arrayBuffer());
 * const model = parseModelBinary(arrayBuffer);
 */
export function parseModelBinary(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    let offset = 0;

    // Helper function to read i32 (little-endian)
    const readI32 = () => {
        if (offset + 4 > arrayBuffer.byteLength) {
            throw new Error('Unexpected end of file while reading dimensions');
        }
        const value = dataView.getInt32(offset, true); // true = little-endian
        offset += 4;
        return value;
    };

    // Helper function to read f32 (little-endian)
    const readF32 = () => {
        if (offset + 4 > arrayBuffer.byteLength) {
            throw new Error('Unexpected end of file while reading weights');
        }
        const value = dataView.getFloat32(offset, true); // true = little-endian
        offset += 4;
        return value;
    };

    // Helper function to read array of f32 values
    const readF32Array = (count) => {
        const array = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            array[i] = readF32();
        }
        return array;
    };

    // Read dimensions
    const input_size = readI32();
    const hidden_size = readI32();
    const output_size = readI32();

    // Validate dimensions
    if (input_size <= 0 || hidden_size <= 0 || output_size <= 0) {
        throw new Error(`Invalid model dimensions: ${input_size} x ${hidden_size} x ${output_size}`);
    }

    // Expected MNIST dimensions
    const EXPECTED_INPUT = 784;   // 28x28 pixels
    const EXPECTED_HIDDEN = 512;  // Hidden layer size
    const EXPECTED_OUTPUT = 10;   // Digits 0-9

    if (input_size !== EXPECTED_INPUT) {
        throw new Error(`Invalid input size: expected ${EXPECTED_INPUT}, got ${input_size}`);
    }
    if (hidden_size !== EXPECTED_HIDDEN) {
        throw new Error(`Invalid hidden size: expected ${EXPECTED_HIDDEN}, got ${hidden_size}`);
    }
    if (output_size !== EXPECTED_OUTPUT) {
        throw new Error(`Invalid output size: expected ${EXPECTED_OUTPUT}, got ${output_size}`);
    }

    // Calculate expected sizes
    const hiddenWeightsCount = input_size * hidden_size;
    const hiddenBiasesCount = hidden_size;
    const outputWeightsCount = hidden_size * output_size;
    const outputBiasesCount = output_size;

    const expectedSize =
        12 + // 3 x i32 for dimensions
        (hiddenWeightsCount + hiddenBiasesCount + outputWeightsCount + outputBiasesCount) * 4; // f32 arrays

    if (arrayBuffer.byteLength < expectedSize) {
        throw new Error(
            `Model file too small: expected at least ${expectedSize} bytes, got ${arrayBuffer.byteLength}`
        );
    }

    // Read hidden layer weights (input_size × hidden_size)
    const hiddenWeights = readF32Array(hiddenWeightsCount);

    // Read hidden layer biases
    const hiddenBiases = readF32Array(hiddenBiasesCount);

    // Read output layer weights (hidden_size × output_size)
    const outputWeights = readF32Array(outputWeightsCount);

    // Read output layer biases
    const outputBiases = readF32Array(outputBiasesCount);

    // Return structured model data
    return {
        input_size,
        hidden_size,
        output_size,
        hiddenWeights,
        hiddenBiases,
        outputWeights,
        outputBiases,
        metadata: {
            totalParameters: hiddenWeightsCount + hiddenBiasesCount + outputWeightsCount + outputBiasesCount,
            fileSizeBytes: arrayBuffer.byteLength,
            architecture: `${input_size} -> ${hidden_size} (ReLU) -> ${output_size} (Softmax)`
        }
    };
}

/**
 * Validates that model data has the expected structure.
 *
 * @param {Object} model - Model object to validate
 * @returns {boolean} True if model is valid
 * @throws {Error} If model structure is invalid
 *
 * @example
 * const model = await loadModel('model.bin');
 * validateModel(model); // throws if invalid
 */
export function validateModel(model) {
    if (!model || typeof model !== 'object') {
        throw new Error('Model must be an object');
    }

    const requiredFields = [
        'input_size', 'hidden_size', 'output_size',
        'hiddenWeights', 'hiddenBiases', 'outputWeights', 'outputBiases'
    ];

    for (const field of requiredFields) {
        if (!(field in model)) {
            throw new Error(`Missing required field: ${field}`);
        }
    }

    // Validate array lengths
    const expectedHiddenWeights = model.input_size * model.hidden_size;
    const expectedHiddenBiases = model.hidden_size;
    const expectedOutputWeights = model.hidden_size * model.output_size;
    const expectedOutputBiases = model.output_size;

    if (model.hiddenWeights.length !== expectedHiddenWeights) {
        throw new Error(
            `Invalid hidden weights length: expected ${expectedHiddenWeights}, got ${model.hiddenWeights.length}`
        );
    }

    if (model.hiddenBiases.length !== expectedHiddenBiases) {
        throw new Error(
            `Invalid hidden biases length: expected ${expectedHiddenBiases}, got ${model.hiddenBiases.length}`
        );
    }

    if (model.outputWeights.length !== expectedOutputWeights) {
        throw new Error(
            `Invalid output weights length: expected ${expectedOutputWeights}, got ${model.outputWeights.length}`
        );
    }

    if (model.outputBiases.length !== expectedOutputBiases) {
        throw new Error(
            `Invalid output biases length: expected ${expectedOutputBiases}, got ${model.outputBiases.length}`
        );
    }

    return true;
}
