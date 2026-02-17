/**
 * Gradient Flow Visualizer - Main Application Controller
 *
 * Coordinates gradient data loading, Chart.js visualization, and interactive
 * UI controls for exploring gradient health during neural network training.
 *
 * Handles:
 * - Epoch slider for animated time-travel through training history
 * - Log-scale gradient norm chart with colored health zones
 * - Per-layer health status cards (green / orange / red)
 * - Demo data loading for offline exploration
 * - Educational annotations toggle
 * - CSV/JSON export and file drag-and-drop upload
 *
 * @example
 * const app = new GradientVizApp({ gradientChartId: 'gradient-chart' });
 * await app.initialize();
 */

import { GradientDataLoader } from './gradient_data_loader.js';

/**
 * Chart line colors – must match the CSS variables in gradient_viz_style.css.
 * @type {Object<string,string>}
 */
const CHART_COLORS = {
    weights:       '#4A90E2',
    biases:        '#9B59B6',
    vanishingZone: 'rgba(217, 83, 79, 0.13)',
    healthyZone:   'rgba(92, 184, 92, 0.09)',
    explodingZone: 'rgba(217, 83, 79, 0.13)',
};

/**
 * Custom Chart.js plugin that draws colored background zones for gradient health.
 *
 * Renders three horizontal bands before chart datasets are painted:
 * - Red band  below the vanishing threshold (vanishing gradients)
 * - Green band between vanishing and exploding thresholds (healthy gradients)
 * - Red band  above the exploding threshold  (exploding gradients)
 *
 * Plugin options are supplied via `chart.options.plugins.backgroundZones`:
 * ```js
 * options: {
 *   plugins: {
 *     backgroundZones: { vanishingThreshold: 1e-5, explodingThreshold: 100 }
 *   }
 * }
 * ```
 */
const backgroundZonesPlugin = {
    id: 'backgroundZones',

    /**
     * Draws health-zone rectangles before chart datasets are rendered.
     *
     * @param {Object} chart - Chart.js chart instance
     */
    beforeDraw(chart) {
        const pluginOpts = chart.options.plugins && chart.options.plugins.backgroundZones;
        if (!pluginOpts) return;

        const { ctx, chartArea, scales } = chart;
        if (!chartArea || !scales.y) return;

        const { top, bottom, left, right } = chartArea;
        const { vanishingThreshold, explodingThreshold } = pluginOpts;

        // Convert threshold values to pixel positions on the logarithmic Y axis.
        // Higher values are towards the top (smaller pixel Y), lower towards bottom.
        const vanishingPx = scales.y.getPixelForValue(vanishingThreshold);
        const explodingPx = scales.y.getPixelForValue(explodingThreshold);

        ctx.save();

        // Vanishing zone: from chart bottom up to vanishingThreshold line
        const vanTop    = Math.max(top, Math.min(bottom, vanishingPx));
        const vanBottom = bottom;
        if (vanBottom > vanTop) {
            ctx.fillStyle = CHART_COLORS.vanishingZone;
            ctx.fillRect(left, vanTop, right - left, vanBottom - vanTop);
        }

        // Healthy zone: between vanishing and exploding threshold lines
        const healthTop    = Math.max(top, Math.min(bottom, explodingPx));
        const healthBottom = Math.min(bottom, Math.max(top, vanishingPx));
        if (healthBottom > healthTop) {
            ctx.fillStyle = CHART_COLORS.healthyZone;
            ctx.fillRect(left, healthTop, right - left, healthBottom - healthTop);
        }

        // Exploding zone: from chart top down to explodingThreshold line
        const expTop    = top;
        const expBottom = Math.min(bottom, Math.max(top, explodingPx));
        if (expBottom > expTop) {
            ctx.fillStyle = CHART_COLORS.explodingZone;
            ctx.fillRect(left, expTop, right - left, expBottom - expTop);
        }

        ctx.restore();
    }
};

/**
 * Main application controller for the Gradient Flow Visualizer dashboard.
 *
 * Manages the full lifecycle:
 * 1. DOM element resolution from element IDs
 * 2. Event listener setup (slider, dropdown, buttons, drag-and-drop)
 * 3. Demo and user-provided CSV loading via GradientDataLoader
 * 4. Chart.js line chart with logarithmic Y-axis and background health zones
 * 5. Per-layer health status cards that respond to epoch slider changes
 * 6. Threshold adjustment triggering re-classification and re-render
 * 7. CSV/JSON export via download link
 *
 * @example
 * const app = new GradientVizApp({
 *     gradientChartId:      'gradient-chart',
 *     epochSliderId:        'epoch-slider',
 *     layerSelectorId:      'layer-selector',
 * });
 * await app.initialize();
 */
export class GradientVizApp {
    /**
     * Creates a new GradientVizApp.
     *
     * All parameters are element IDs with sensible defaults that match the
     * IDs used in gradient_viz.html.
     *
     * @param {Object} [options={}] - Element ID mappings
     * @param {string} [options.dropZoneId='drop-zone']
     * @param {string} [options.fileInputId='file-input']
     * @param {string} [options.uploadStatusId='upload-status']
     * @param {string} [options.modelSelectorId='model-selector']
     * @param {string} [options.loadDemoBtnId='load-demo-btn']
     * @param {string} [options.epochSliderId='epoch-slider']
     * @param {string} [options.epochDisplayId='epoch-display']
     * @param {string} [options.vanishingThresholdId='vanishing-threshold']
     * @param {string} [options.explodingThresholdId='exploding-threshold']
     * @param {string} [options.applyThresholdsBtnId='apply-thresholds-btn']
     * @param {string} [options.exportCsvBtnId='export-csv-btn']
     * @param {string} [options.exportJsonBtnId='export-json-btn']
     * @param {string} [options.gradientChartId='gradient-chart']
     * @param {string} [options.layerSelectorId='layer-selector']
     * @param {string} [options.layerListId='layer-list']
     * @param {string} [options.healthEpochLabelId='health-epoch-label']
     * @param {string} [options.toggleAnnotationsBtnId='toggle-annotations-btn']
     * @param {string} [options.annotationsBodyId='annotations-body']
     */
    constructor(options = {}) {
        // Map of logical keys → element IDs
        this._ids = {
            dropZone:             options.dropZoneId             || 'drop-zone',
            fileInput:            options.fileInputId            || 'file-input',
            uploadStatus:         options.uploadStatusId         || 'upload-status',
            modelSelector:        options.modelSelectorId        || 'model-selector',
            loadDemoBtn:          options.loadDemoBtnId          || 'load-demo-btn',
            epochSlider:          options.epochSliderId          || 'epoch-slider',
            epochDisplay:         options.epochDisplayId         || 'epoch-display',
            vanishingThreshold:   options.vanishingThresholdId   || 'vanishing-threshold',
            explodingThreshold:   options.explodingThresholdId   || 'exploding-threshold',
            applyThresholdsBtn:   options.applyThresholdsBtnId   || 'apply-thresholds-btn',
            exportCsvBtn:         options.exportCsvBtnId         || 'export-csv-btn',
            exportJsonBtn:        options.exportJsonBtnId        || 'export-json-btn',
            gradientChart:        options.gradientChartId        || 'gradient-chart',
            layerSelector:        options.layerSelectorId        || 'layer-selector',
            layerList:            options.layerListId            || 'layer-list',
            healthEpochLabel:     options.healthEpochLabelId     || 'health-epoch-label',
            toggleAnnotationsBtn: options.toggleAnnotationsBtnId || 'toggle-annotations-btn',
            annotationsBody:      options.annotationsBodyId      || 'annotations-body',
        };

        /** @type {GradientDataLoader} Data parsing and statistics. */
        this._loader = new GradientDataLoader();

        /**
         * Active Chart.js chart instance, or null if none has been created yet.
         * @type {Object|null}
         */
        this._chart = null;

        // Application state
        /** @type {string|null} Currently selected layer name. */
        this._selectedLayer = null;

        /** @type {number} Epoch currently shown in the chart. */
        this._currentEpoch = 20;

        /** @type {number} Maximum epoch found in the loaded dataset. */
        this._maxEpoch = 20;

        /** @type {Object<string, HTMLElement>} Resolved DOM element references. */
        this._els = {};
    }

    /**
     * Initializes the app: resolves DOM elements, attaches event listeners,
     * and loads demo data automatically.
     *
     * @returns {Promise<void>}
     * @throws {Error} If Chart.js is not available or initialization fails
     *
     * @example
     * const app = new GradientVizApp();
     * await app.initialize();
     */
    async initialize() {
        this._resolveElements();
        this._setupEventHandlers();
        await this.loadDemoData();
    }

    /**
     * Loads gradient demo data for the currently selected model preset
     * ('mlp' or 'cnn') and renders the full visualization.
     *
     * @returns {Promise<void>}
     */
    async loadDemoData() {
        const model = (this._els.modelSelector && this._els.modelSelector.value) || 'mlp';
        const csvText = this._loader.generateDemoData(model);
        this.loadFromCSV(csvText);
        this._updateStatus('Showing demo data \u2014 upload your own CSV to explore');
    }

    /**
     * Reads a File object via FileReader and loads its contents as gradient CSV.
     *
     * Validates that the file has a .csv extension before reading.
     *
     * @param {File} file - CSV file from a file input or drag-and-drop
     */
    loadFromFile(file) {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.csv')) {
            this._updateStatus('Please select a .csv file', 'error');
            return;
        }

        this._updateStatus(`Loading \u201c${file.name}\u201d\u2026`, 'loading');

        const reader = new FileReader();

        reader.onload = (event) => {
            try {
                this.loadFromCSV(event.target.result);
                this._updateStatus(`Loaded: ${file.name}`, 'ready');
            } catch (error) {
                this._updateStatus(`Parse error: ${error.message}`, 'error');
            }
        };

        reader.onerror = () => {
            this._updateStatus('Failed to read file', 'error');
        };

        reader.readAsText(file);
    }

    /**
     * Parses gradient CSV text and renders the complete visualization.
     *
     * Replaces any existing loaded data. The CSV must contain the header line:
     * `epoch,layer_name,grad_norm_weights,grad_norm_biases`
     *
     * @param {string} csvText - Raw CSV string
     * @throws {Error} If the CSV header is missing or parsing fails
     */
    loadFromCSV(csvText) {
        this._loader.parseCSV(csvText);

        // Determine maximum epoch across all layers
        this._maxEpoch = 1;
        for (const layerName of this._loader.getLayerNames()) {
            const data = this._loader.getLayerData(layerName);
            if (!data) continue;
            for (const epoch of data.epochs) {
                if (epoch > this._maxEpoch) {
                    this._maxEpoch = epoch;
                }
            }
        }

        this._currentEpoch = this._maxEpoch;
        this._syncEpochSlider();
        this._populateLayerSelector();

        // Select first layer by default
        const layerNames = this._loader.getLayerNames();
        this._selectedLayer = layerNames.length > 0 ? layerNames[0] : null;
        if (this._els.layerSelector && this._selectedLayer) {
            this._els.layerSelector.value = this._selectedLayer;
        }

        this._setExportEnabled(true);
        this._renderAll();
    }

    /**
     * Renders the gradient norm line chart for the selected layer up to the
     * given epoch.
     *
     * Creates a Chart.js line chart with:
     * - Logarithmic Y-axis
     * - Background health zones (vanishing / healthy / exploding)
     * - Two datasets: weight gradient norms (solid) and bias gradient norms (dashed)
     * - Tooltip labels formatted in scientific notation
     *
     * Any previous chart instance is destroyed before the new one is created.
     *
     * @param {string} layerName - Layer to visualize
     * @param {number} upToEpoch - Show data only up to and including this epoch
     */
    renderWeightChart(layerName, upToEpoch) {
        if (!layerName) return;

        const data = this._loader.getLayerData(layerName);
        if (!data) return;

        // Slice dataset to the requested epoch range
        const epochs      = [];
        const gradWeights = [];
        const gradBiases  = [];

        for (let i = 0; i < data.epochs.length; i++) {
            if (data.epochs[i] <= upToEpoch) {
                epochs.push(data.epochs[i]);
                gradWeights.push(data.gradWeights[i]);
                gradBiases.push(data.gradBiases[i]);
            }
        }

        // Compute a robust log-scale Y range based on positive values only
        const allPositive = [...gradWeights, ...gradBiases].filter(v => v > 0);
        const minVal = allPositive.length > 0 ? Math.min(...allPositive) : 1e-6;
        const maxVal = allPositive.length > 0 ? Math.max(...allPositive) : 10;
        const yMin   = Math.max(1e-15, Math.pow(10, Math.floor(Math.log10(minVal)) - 1));
        const yMax   = Math.pow(10, Math.ceil(Math.log10(maxVal)) + 1);

        const { vanishingThreshold, explodingThreshold } = this._loader;

        const chartConfig = {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label:           'Weight Gradient Norm',
                        data:            gradWeights,
                        borderColor:     CHART_COLORS.weights,
                        backgroundColor: CHART_COLORS.weights + '33',
                        pointRadius:     4,
                        pointHoverRadius: 7,
                        tension:         0.3,
                        fill:            false,
                    },
                    {
                        label:           'Bias Gradient Norm',
                        data:            gradBiases,
                        borderColor:     CHART_COLORS.biases,
                        backgroundColor: CHART_COLORS.biases + '33',
                        pointRadius:     4,
                        pointHoverRadius: 7,
                        tension:         0.3,
                        fill:            false,
                        borderDash:      [6, 4],
                    },
                ],
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation:           { duration: 200 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label(ctx) {
                                const val = ctx.parsed.y;
                                return `${ctx.dataset.label}: ${val.toExponential(3)}`;
                            },
                        },
                    },
                    // Options for our custom backgroundZonesPlugin
                    backgroundZones: {
                        vanishingThreshold,
                        explodingThreshold,
                    },
                },
                scales: {
                    x: {
                        type:  'linear',
                        title: { display: true, text: 'Epoch' },
                        ticks: { stepSize: 1, maxTicksLimit: 20 },
                    },
                    y: {
                        type:  'logarithmic',
                        title: { display: true, text: 'Gradient Norm (log scale)' },
                        min:   yMin,
                        max:   yMax,
                        ticks: {
                            callback(value) {
                                // Show ticks only at powers of ten for a clean log axis
                                const exponent = Math.log10(value);
                                if (Math.abs(exponent - Math.round(exponent)) < 1e-9) {
                                    return value.toExponential(0);
                                }
                                return null;
                            },
                        },
                    },
                },
            },
            // Register the background-zones plugin for this chart instance only
            plugins: [backgroundZonesPlugin],
        };

        // Destroy any existing chart before creating a new one
        if (this._chart) {
            this._chart.destroy();
            this._chart = null;
        }

        if (this._els.gradientChart) {
            // Chart is loaded from CDN and available as a global
            /* global Chart */
            this._chart = new Chart(this._els.gradientChart, chartConfig);
        }
    }

    /**
     * Renders the per-layer health status cards in the layer list container.
     *
     * Each card displays:
     * - Layer name (title)
     * - Weight gradient norm at the current epoch
     * - Health badge (Healthy / Vanishing / Exploding)
     * - Min / mean / max statistics over all loaded epochs
     *
     * Cards use CSS classes `layer-healthy`, `layer-vanishing`, or
     * `layer-exploding` (defined in gradient_viz_style.css) to provide
     * color-coded borders. Clicking a card selects that layer in the chart.
     */
    renderLayerList() {
        const container = this._els.layerList;
        if (!container) return;

        container.innerHTML = '';

        const layerNames = this._loader.getLayerNames();
        if (layerNames.length === 0) return;

        for (const layerName of layerNames) {
            const valueAtEpoch = this._getValueAtEpoch(layerName, this._currentEpoch);
            const stats        = this._loader.getLayerStats(layerName);

            const currentGrad   = valueAtEpoch ? valueAtEpoch.gradWeights : 0;
            const healthAtEpoch = this._loader.classifyGradient(currentGrad);

            const isSelected = layerName === this._selectedLayer;

            // Build card element
            const card = document.createElement('div');
            card.className = `layer-card layer-${healthAtEpoch}`;
            card.setAttribute('role', 'button');
            card.setAttribute('tabindex', '0');
            card.setAttribute('aria-label', `Select layer ${layerName}`);
            card.setAttribute('aria-pressed', String(isSelected));

            if (isSelected) {
                card.style.outline       = '2px solid var(--primary-color, #4A90E2)';
                card.style.outlineOffset = '2px';
            }

            const badgeLabel  = healthAtEpoch.charAt(0).toUpperCase() + healthAtEpoch.slice(1);
            const gradDisplay = currentGrad > 0 ? currentGrad.toExponential(2) : 'N/A';
            const meanDisplay = stats ? stats.weights.mean.toExponential(2) : 'N/A';
            const minDisplay  = stats ? stats.weights.min.toExponential(2)  : 'N/A';
            const maxDisplay  = stats ? stats.weights.max.toExponential(2)  : 'N/A';

            card.innerHTML = `
                <div class="layer-card-name">${_escapeHtml(layerName)}</div>
                <div class="layer-card-stats">
                    epoch&nbsp;${this._currentEpoch}:&nbsp;${gradDisplay}<br>
                    min:&nbsp;${minDisplay}&nbsp;&nbsp;max:&nbsp;${maxDisplay}<br>
                    mean:&nbsp;${meanDisplay}
                </div>
                <div class="layer-card-badge">
                    <span class="health-badge badge-${healthAtEpoch}">${badgeLabel}</span>
                </div>
            `.trim();

            card.addEventListener('click', () => this._selectLayer(layerName));
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this._selectLayer(layerName);
                }
            });

            container.appendChild(card);
        }
    }

    /**
     * Reads the vanishing and exploding threshold inputs, updates the data
     * loader's thresholds, then re-renders the chart and layer list.
     *
     * Displays an error status message if the inputs are invalid.
     */
    updateThresholds() {
        const vanishingEl = this._els.vanishingThreshold;
        const explodingEl = this._els.explodingThreshold;
        if (!vanishingEl || !explodingEl) return;

        const newVanishing = parseFloat(vanishingEl.value);
        const newExploding = parseFloat(explodingEl.value);

        if (!isFinite(newVanishing) || newVanishing <= 0) {
            this._updateStatus('Vanishing threshold must be a positive number', 'error');
            return;
        }
        if (!isFinite(newExploding) || newExploding <= newVanishing) {
            this._updateStatus('Exploding threshold must be greater than vanishing threshold', 'error');
            return;
        }

        try {
            this._loader.setThresholds(newVanishing, newExploding);
        } catch (error) {
            this._updateStatus(`Invalid thresholds: ${error.message}`, 'error');
            return;
        }

        this._renderAll();
    }

    /**
     * Exports the current gradient data as a CSV file and triggers a download.
     */
    exportCSV() {
        try {
            const csvText = this._loader.exportAsCSV();
            this._triggerDownload(csvText, 'gradient_flow.csv', 'text/csv;charset=utf-8;');
        } catch (error) {
            this._updateStatus(`Export error: ${error.message}`, 'error');
        }
    }

    /**
     * Exports gradient statistics as a JSON file and triggers a download.
     */
    exportJSON() {
        try {
            const jsonText = this._loader.exportAsJSON();
            this._triggerDownload(jsonText, 'gradient_stats.json', 'application/json');
        } catch (error) {
            this._updateStatus(`Export error: ${error.message}`, 'error');
        }
    }

    // ==================== Private Helpers ====================

    /**
     * Resolves all DOM element IDs into cached `this._els` references.
     *
     * @private
     */
    _resolveElements() {
        for (const [key, id] of Object.entries(this._ids)) {
            this._els[key] = document.getElementById(id) || null;
        }
    }

    /**
     * Attaches all event listeners to controls.
     *
     * @private
     */
    _setupEventHandlers() {
        // Load Demo button
        if (this._els.loadDemoBtn) {
            this._els.loadDemoBtn.addEventListener('click', () => this.loadDemoData());
        }

        // Epoch slider – update chart and health cards on every tick
        if (this._els.epochSlider) {
            this._els.epochSlider.addEventListener('input', () => {
                this._currentEpoch = parseInt(this._els.epochSlider.value, 10);
                this._syncEpochDisplay();
                this._renderAll();
            });
        }

        // Layer selector dropdown – re-render chart for chosen layer
        if (this._els.layerSelector) {
            this._els.layerSelector.addEventListener('change', () => {
                const chosen = this._els.layerSelector.value;
                if (chosen) {
                    this._selectedLayer = chosen;
                    this.renderWeightChart(this._selectedLayer, this._currentEpoch);
                    this.renderLayerList();
                }
            });
        }

        // Apply Thresholds button
        if (this._els.applyThresholdsBtn) {
            this._els.applyThresholdsBtn.addEventListener('click', () => this.updateThresholds());
        }

        // Export buttons
        if (this._els.exportCsvBtn) {
            this._els.exportCsvBtn.addEventListener('click', () => this.exportCSV());
        }
        if (this._els.exportJsonBtn) {
            this._els.exportJsonBtn.addEventListener('click', () => this.exportJSON());
        }

        // Toggle annotations panel
        if (this._els.toggleAnnotationsBtn && this._els.annotationsBody) {
            this._els.toggleAnnotationsBtn.addEventListener('click', () => this._toggleAnnotations());
        }

        // File input (browse-to-upload)
        if (this._els.fileInput) {
            this._els.fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.loadFromFile(file);
                    // Reset so the same file can be selected again
                    e.target.value = '';
                }
            });
        }

        // Drag-and-drop on the drop zone
        if (this._els.dropZone) {
            this._els.dropZone.addEventListener('dragenter', (e) => {
                e.preventDefault();
                this._els.dropZone.classList.add('dragover');
            });

            this._els.dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                this._els.dropZone.classList.add('dragover');
            });

            this._els.dropZone.addEventListener('dragleave', (e) => {
                // Only remove highlight when leaving the drop zone entirely (not a child element)
                if (!this._els.dropZone.contains(e.relatedTarget)) {
                    this._els.dropZone.classList.remove('dragover');
                }
            });

            this._els.dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                this._els.dropZone.classList.remove('dragover');
                const file = e.dataTransfer && e.dataTransfer.files[0];
                if (file) {
                    this.loadFromFile(file);
                }
            });

            // Keyboard accessibility: Enter/Space activates the file input
            this._els.dropZone.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    if (this._els.fileInput) {
                        this._els.fileInput.click();
                    }
                }
            });
        }
    }

    /**
     * Populates the layer selector `<select>` with the loaded layer names.
     *
     * @private
     */
    _populateLayerSelector() {
        const select = this._els.layerSelector;
        if (!select) return;

        select.innerHTML = '';
        for (const layerName of this._loader.getLayerNames()) {
            const option = document.createElement('option');
            option.value       = layerName;
            option.textContent = layerName;
            select.appendChild(option);
        }
    }

    /**
     * Renders the chart and layer list for the current selection and epoch.
     *
     * @private
     */
    _renderAll() {
        if (this._selectedLayer) {
            this.renderWeightChart(this._selectedLayer, this._currentEpoch);
        }
        this.renderLayerList();
        this._syncHealthEpochLabel();
    }

    /**
     * Selects a layer, updates the dropdown, and re-renders the visualization.
     *
     * @private
     * @param {string} layerName - Layer name to select
     */
    _selectLayer(layerName) {
        this._selectedLayer = layerName;
        if (this._els.layerSelector) {
            this._els.layerSelector.value = layerName;
        }
        this.renderWeightChart(layerName, this._currentEpoch);
        this.renderLayerList();
    }

    /**
     * Synchronises the epoch slider's min/max/value attributes and display label
     * with the current `_maxEpoch` and `_currentEpoch` state values.
     *
     * @private
     */
    _syncEpochSlider() {
        if (this._els.epochSlider) {
            this._els.epochSlider.min   = '1';
            this._els.epochSlider.max   = String(this._maxEpoch);
            this._els.epochSlider.value = String(this._currentEpoch);
        }
        this._syncEpochDisplay();
    }

    /**
     * Updates the epoch counter text element to show `_currentEpoch`.
     *
     * @private
     */
    _syncEpochDisplay() {
        if (this._els.epochDisplay) {
            this._els.epochDisplay.textContent = String(this._currentEpoch);
        }
    }

    /**
     * Updates the "Gradient health at epoch N" label in the layer list section.
     *
     * @private
     */
    _syncHealthEpochLabel() {
        if (this._els.healthEpochLabel) {
            this._els.healthEpochLabel.textContent = String(this._currentEpoch);
        }
    }

    /**
     * Enables or disables the export CSV/JSON buttons.
     *
     * @private
     * @param {boolean} enabled - Whether to enable the buttons
     */
    _setExportEnabled(enabled) {
        if (this._els.exportCsvBtn)  this._els.exportCsvBtn.disabled  = !enabled;
        if (this._els.exportJsonBtn) this._els.exportJsonBtn.disabled = !enabled;
    }

    /**
     * Returns the gradient values for a layer at the most recent epoch
     * that is less than or equal to `targetEpoch`.
     *
     * @private
     * @param {string} layerName   - Layer name
     * @param {number} targetEpoch - Target epoch (inclusive)
     * @returns {{epoch: number, gradWeights: number, gradBiases: number}|null}
     *   Gradient values at the closest matching epoch, or null if not found
     */
    _getValueAtEpoch(layerName, targetEpoch) {
        const data = this._loader.getLayerData(layerName);
        if (!data || data.epochs.length === 0) return null;

        let result = null;
        for (let i = 0; i < data.epochs.length; i++) {
            if (data.epochs[i] <= targetEpoch) {
                result = {
                    epoch:       data.epochs[i],
                    gradWeights: data.gradWeights[i],
                    gradBiases:  data.gradBiases[i],
                };
            }
        }
        return result;
    }

    /**
     * Updates the upload status message element's text and CSS state class.
     *
     * @private
     * @param {string} message - Status text
     * @param {'ready'|'error'|'loading'|''} [type=''] - CSS modifier suffix
     */
    _updateStatus(message, type = '') {
        const el = this._els.uploadStatus;
        if (!el) return;
        el.textContent = message;
        el.className   = type ? `status-message ${type}` : 'status-message';
    }

    /**
     * Creates a temporary download link and programmatically clicks it to
     * trigger a file download in the browser.
     *
     * @private
     * @param {string} content  - File content as a string
     * @param {string} filename - Suggested filename for the download
     * @param {string} mimeType - MIME type (e.g. 'text/csv;charset=utf-8;')
     */
    _triggerDownload(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href          = url;
        a.download      = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Toggles the educational annotations panel between visible and hidden.
     * Updates the toggle button label and `aria-expanded` attribute accordingly.
     *
     * @private
     */
    _toggleAnnotations() {
        const btn  = this._els.toggleAnnotationsBtn;
        const body = this._els.annotationsBody;
        if (!btn || !body) return;

        const isHidden = body.hasAttribute('hidden');
        if (isHidden) {
            body.removeAttribute('hidden');
            btn.textContent = 'Hide Explanations';
            btn.setAttribute('aria-expanded', 'true');
        } else {
            body.setAttribute('hidden', '');
            btn.textContent = 'Show Explanations';
            btn.setAttribute('aria-expanded', 'false');
        }
    }
}

/**
 * Escapes HTML special characters to prevent XSS when inserting layer names
 * into innerHTML strings.
 *
 * @param {string} text - Potentially unsafe string
 * @returns {string} HTML-safe string
 * @private
 */
function _escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
