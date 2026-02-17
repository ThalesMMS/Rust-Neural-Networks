/**
 * Gradient Data Loader - CSV Parser and Data Transformation Module
 *
 * Parses gradient log CSV files produced by Rust training binaries,
 * transforms data into per-layer structures, and provides classification
 * utilities for detecting vanishing and exploding gradient conditions.
 *
 * Expected CSV format (produced by Rust training binaries):
 *   epoch,layer_name,grad_norm_weights,grad_norm_biases
 *
 * @example
 * const loader = new GradientDataLoader();
 * loader.parseCSV(loader.generateDemoData());
 * console.log(loader.getLayerNames()); // ['dense_layer_0', 'dense_layer_1']
 */

/** Gradient norm threshold below which gradients are considered vanishing. */
export const VANISHING_THRESHOLD = 1e-5;

/** Gradient norm threshold above which gradients are considered exploding. */
export const EXPLODING_THRESHOLD = 100.0;

/**
 * Loads, parses, and queries gradient flow data from CSV training logs.
 *
 * Handles:
 * - CSV parsing with header validation
 * - Per-layer data organization (epochs, weight norms, bias norms)
 * - Gradient health classification (healthy / vanishing / exploding)
 * - Descriptive statistics (min, max, mean, std)
 * - Demo data generation for offline exploration
 * - CSV and JSON export of gradient statistics
 */
export class GradientDataLoader {
    /**
     * Creates a new GradientDataLoader.
     *
     * @param {Object} options - Configuration options
     * @param {number} options.vanishingThreshold - Threshold below which gradients are vanishing (default: 1e-5)
     * @param {number} options.explodingThreshold - Threshold above which gradients are exploding (default: 100.0)
     */
    constructor(options = {}) {
        /** @type {number} */
        this.vanishingThreshold = options.vanishingThreshold ?? VANISHING_THRESHOLD;

        /** @type {number} */
        this.explodingThreshold = options.explodingThreshold ?? EXPLODING_THRESHOLD;

        /**
         * Internal storage: layer name → { epochs, gradWeights, gradBiases }
         * @type {Map<string, {epochs: number[], gradWeights: number[], gradBiases: number[]}>}
         * @private
         */
        this._layers = new Map();
    }

    /**
     * Parses a CSV string in the format produced by the Rust gradient logger.
     *
     * Expected header: `epoch,layer_name,grad_norm_weights,grad_norm_biases`
     * Skips blank lines and logs a warning for malformed rows.
     *
     * @param {string} csvText - Raw CSV text
     * @returns {number} Number of data rows successfully parsed
     * @throws {Error} If the CSV header is missing or malformed
     *
     * @example
     * const loader = new GradientDataLoader();
     * const rowCount = loader.parseCSV(csvText);
     */
    parseCSV(csvText) {
        if (typeof csvText !== 'string') {
            throw new Error('parseCSV expects a string argument');
        }

        // Reset internal state before loading new data
        this._layers.clear();

        const lines = csvText.split('\n');
        if (lines.length === 0) {
            throw new Error('CSV text is empty');
        }

        // Validate header
        const header = lines[0].trim().toLowerCase();
        if (!header.startsWith('epoch')) {
            throw new Error(`Unexpected CSV header: "${lines[0].trim()}". Expected "epoch,layer_name,grad_norm_weights,grad_norm_biases"`);
        }

        let parsedCount = 0;

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const parts = line.split(',');
            if (parts.length !== 4) {
                // Silently skip malformed rows to match Python loader behaviour
                continue;
            }

            const epoch = parseInt(parts[0].trim(), 10);
            const layerName = parts[1].trim();
            const gradNormWeights = parseFloat(parts[2].trim());
            const gradNormBiases = parseFloat(parts[3].trim());

            if (!isFinite(epoch) || !layerName || !isFinite(gradNormWeights) || !isFinite(gradNormBiases)) {
                continue;
            }

            // Initialise layer entry on first encounter
            if (!this._layers.has(layerName)) {
                this._layers.set(layerName, {
                    epochs: [],
                    gradWeights: [],
                    gradBiases: []
                });
            }

            const layerData = this._layers.get(layerName);
            layerData.epochs.push(epoch);
            layerData.gradWeights.push(gradNormWeights);
            layerData.gradBiases.push(gradNormBiases);

            parsedCount++;
        }

        return parsedCount;
    }

    /**
     * Returns a sorted list of all layer names found in the parsed data.
     *
     * @returns {string[]} Sorted array of layer names
     *
     * @example
     * loader.getLayerNames(); // ['dense_layer_0', 'dense_layer_1']
     */
    getLayerNames() {
        return Array.from(this._layers.keys()).sort();
    }

    /**
     * Returns the raw data arrays for a given layer.
     *
     * @param {string} layerName - Layer name as it appears in the CSV
     * @returns {{epochs: number[], gradWeights: number[], gradBiases: number[]}|null}
     *   Layer data object, or null if the layer is not found
     *
     * @example
     * const data = loader.getLayerData('dense_layer_0');
     * // data.epochs     → [1, 2, 3, ...]
     * // data.gradWeights → [0.45, 0.38, ...]
     * // data.gradBiases  → [0.02, 0.018, ...]
     */
    getLayerData(layerName) {
        if (!this._layers.has(layerName)) {
            return null;
        }
        const d = this._layers.get(layerName);
        return {
            epochs: [...d.epochs],
            gradWeights: [...d.gradWeights],
            gradBiases: [...d.gradBiases]
        };
    }

    /**
     * Classifies a single gradient norm value into a health category.
     *
     * Uses the thresholds configured on this loader instance
     * (defaulting to VANISHING_THRESHOLD and EXPLODING_THRESHOLD).
     *
     * @param {number} value - Gradient norm value to classify
     * @returns {'vanishing'|'healthy'|'exploding'} Health classification
     *
     * @example
     * loader.classifyGradient(1e-8);  // 'vanishing'
     * loader.classifyGradient(0.5);   // 'healthy'
     * loader.classifyGradient(200);   // 'exploding'
     */
    classifyGradient(value) {
        if (value < this.vanishingThreshold) {
            return 'vanishing';
        }
        if (value > this.explodingThreshold) {
            return 'exploding';
        }
        return 'healthy';
    }

    /**
     * Computes descriptive statistics for a layer's gradient norms.
     *
     * Returns separate stats for weight gradients and bias gradients,
     * plus the health classification of the final (most recent) value.
     *
     * @param {string} layerName - Layer name as it appears in the CSV
     * @returns {{
     *   weights: {min: number, max: number, mean: number, std: number},
     *   biases:  {min: number, max: number, mean: number, std: number},
     *   currentHealth: 'vanishing'|'healthy'|'exploding'
     * }|null} Statistics object, or null if the layer is not found
     *
     * @example
     * const stats = loader.getLayerStats('dense_layer_0');
     * // stats.weights.mean → 0.25
     * // stats.currentHealth → 'healthy'
     */
    getLayerStats(layerName) {
        const data = this._layers.get(layerName);
        if (!data) {
            return null;
        }

        const weightsStats = this._computeStats(data.gradWeights);
        const biasesStats = this._computeStats(data.gradBiases);

        // Health is based on the most recent weight gradient value
        const latestWeightGrad = data.gradWeights[data.gradWeights.length - 1] ?? 0;
        const currentHealth = this.classifyGradient(latestWeightGrad);

        return {
            weights: weightsStats,
            biases: biasesStats,
            currentHealth
        };
    }

    /**
     * Generates a realistic demo gradient CSV for a two-layer MLP over 20 epochs.
     *
     * The demo shows typical healthy gradient decay with small perturbations,
     * useful for exploring the visualizer without needing real training logs.
     *
     * Layer schedule (weight grad norms, approximate):
     * - `dense_layer_0` (hidden): starts ~0.5, decays to ~0.05 (healthy)
     * - `dense_layer_1` (output): starts ~0.3, decays to ~0.02 (healthy)
     *
     * @param {string} [model='mlp'] - Model preset: 'mlp' or 'cnn'
     * @returns {string} CSV text ready to pass to parseCSV()
     *
     * @example
     * const csv = loader.generateDemoData();
     * loader.parseCSV(csv);
     */
    generateDemoData(model = 'mlp') {
        const lines = ['epoch,layer_name,grad_norm_weights,grad_norm_biases'];
        const epochs = 20;

        let layerDefs;
        if (model === 'cnn') {
            layerDefs = [
                { name: 'conv_layer_0',   initW: 0.6,  initB: 0.04, decayW: 0.88, decayB: 0.90 },
                { name: 'dense_layer_0',  initW: 0.45, initB: 0.03, decayW: 0.91, decayB: 0.92 },
                { name: 'dense_layer_1',  initW: 0.28, initB: 0.02, decayW: 0.93, decayB: 0.94 }
            ];
        } else {
            // MLP: two fully-connected layers
            layerDefs = [
                { name: 'dense_layer_0', initW: 0.50, initB: 0.035, decayW: 0.90, decayB: 0.91 },
                { name: 'dense_layer_1', initW: 0.30, initB: 0.020, decayW: 0.92, decayB: 0.93 }
            ];
        }

        // Simple deterministic pseudo-random perturbation (avoids Math.random for reproducibility)
        const jitter = (seed, scale) => {
            const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
            return (x - Math.floor(x) - 0.5) * scale;
        };

        for (let epoch = 1; epoch <= epochs; epoch++) {
            for (const layer of layerDefs) {
                const wNorm = layer.initW * Math.pow(layer.decayW, epoch - 1)
                    * (1 + jitter(epoch * 17 + layerDefs.indexOf(layer) * 7, 0.15));
                const bNorm = layer.initB * Math.pow(layer.decayB, epoch - 1)
                    * (1 + jitter(epoch * 31 + layerDefs.indexOf(layer) * 13, 0.12));

                const wClamped = Math.max(wNorm, 1e-9);
                const bClamped = Math.max(bNorm, 1e-9);

                lines.push(`${epoch},${layer.name},${wClamped.toExponential(6)},${bClamped.toExponential(6)}`);
            }
        }

        return lines.join('\n');
    }

    /**
     * Exports the currently loaded data as a CSV string in the same format
     * as the original Rust-generated training logs.
     *
     * @returns {string} CSV text with header and one row per epoch/layer combination
     * @throws {Error} If no data has been loaded (parseCSV not yet called)
     *
     * @example
     * const csvText = loader.exportAsCSV();
     * // 'epoch,layer_name,grad_norm_weights,grad_norm_biases\n1,dense_layer_0,...'
     */
    exportAsCSV() {
        if (this._layers.size === 0) {
            throw new Error('No data loaded. Call parseCSV() before exporting.');
        }

        const lines = ['epoch,layer_name,grad_norm_weights,grad_norm_biases'];

        for (const layerName of this.getLayerNames()) {
            const data = this._layers.get(layerName);
            for (let i = 0; i < data.epochs.length; i++) {
                lines.push(`${data.epochs[i]},${layerName},${data.gradWeights[i]},${data.gradBiases[i]}`);
            }
        }

        return lines.join('\n');
    }

    /**
     * Exports the currently loaded data as a JSON string with full per-layer statistics.
     *
     * The exported JSON contains:
     * - `layers`: per-layer statistics (min/max/mean/std, health classification, issue counts)
     * - `meta`: export timestamp, threshold values, total layers, total epochs
     *
     * @returns {string} Pretty-printed JSON string
     * @throws {Error} If no data has been loaded
     *
     * @example
     * const json = loader.exportAsJSON();
     * const data = JSON.parse(json);
     * data.layers['dense_layer_0'].weights.mean; // 0.25
     */
    exportAsJSON() {
        if (this._layers.size === 0) {
            throw new Error('No data loaded. Call parseCSV() before exporting.');
        }

        const layerNames = this.getLayerNames();
        const output = {
            meta: {
                exported_at: new Date().toISOString(),
                vanishing_threshold: this.vanishingThreshold,
                exploding_threshold: this.explodingThreshold,
                total_layers: layerNames.length,
                total_epochs: this._getTotalEpochs()
            },
            layers: {}
        };

        for (const layerName of layerNames) {
            const data = this._layers.get(layerName);
            const stats = this.getLayerStats(layerName);

            // Count health classifications per epoch
            const vanishingCount = data.gradWeights.filter(v => v < this.vanishingThreshold).length;
            const explodingCount = data.gradWeights.filter(v => v > this.explodingThreshold).length;
            const healthyCount = data.gradWeights.length - vanishingCount - explodingCount;

            output.layers[layerName] = {
                weights: stats.weights,
                biases: stats.biases,
                gradient_health: stats.currentHealth,
                issue_counts: {
                    healthy: healthyCount,
                    vanishing: vanishingCount,
                    exploding: explodingCount
                },
                epochs: data.epochs.length
            };
        }

        return JSON.stringify(output, null, 2);
    }

    /**
     * Updates the vanishing and exploding thresholds.
     *
     * @param {number} vanishing - New vanishing threshold (must be positive)
     * @param {number} exploding - New exploding threshold (must be > vanishing)
     * @throws {Error} If the thresholds are invalid
     */
    setThresholds(vanishing, exploding) {
        if (vanishing <= 0) {
            throw new Error(`Vanishing threshold must be positive, got ${vanishing}`);
        }
        if (exploding <= vanishing) {
            throw new Error(`Exploding threshold (${exploding}) must be greater than vanishing threshold (${vanishing})`);
        }
        this.vanishingThreshold = vanishing;
        this.explodingThreshold = exploding;
    }

    /**
     * Returns the number of distinct epochs seen across all layers.
     *
     * @returns {number} Max epoch count across all layers
     * @private
     */
    _getTotalEpochs() {
        let max = 0;
        for (const data of this._layers.values()) {
            if (data.epochs.length > max) {
                max = data.epochs.length;
            }
        }
        return max;
    }

    /**
     * Computes min, max, mean, and standard deviation for an array of numbers.
     *
     * @param {number[]} values - Array of numeric values
     * @returns {{min: number, max: number, mean: number, std: number}}
     * @private
     */
    _computeStats(values) {
        if (!values || values.length === 0) {
            return { min: 0, max: 0, mean: 0, std: 0 };
        }

        let min = values[0];
        let max = values[0];
        let sum = 0;

        for (const v of values) {
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }

        const mean = sum / values.length;

        let variance = 0;
        for (const v of values) {
            const diff = v - mean;
            variance += diff * diff;
        }
        variance /= values.length;

        return {
            min,
            max,
            mean,
            std: Math.sqrt(variance)
        };
    }
}
