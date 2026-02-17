//! Per-layer performance profiling for neural network training.
//!
//! This module provides data structures and a [`Profiler`] for collecting
//! timing measurements (forward pass, backward pass, parameter update) and
//! estimated computational cost (FLOPS, parameter memory) for each layer
//! during a training run.
//!
//! # Overview
//!
//! - [`LayerProfile`] – timing and cost data for a single layer in one epoch.
//! - [`EpochProfile`] – a collection of [`LayerProfile`]s for all layers in
//!   one training epoch.
//! - [`Profiler`] – accumulates [`EpochProfile`]s across all epochs and
//!   provides summary / export helpers.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::profiling::{LayerProfile, Profiler};
//!
//! let mut profiler = Profiler::new();
//!
//! profiler.start_epoch(1);
//! profiler.record_layer(LayerProfile {
//!     layer_name: "Dense 784→512".to_string(),
//!     forward_ms: 1.5,
//!     backward_ms: 2.3,
//!     update_ms: 0.4,
//!     flops_forward: 100_663_296,
//!     flops_backward: 100_663_296,
//!     param_memory_bytes: 1_608_704,
//! });
//! profiler.end_epoch();
//!
//! println!("{}", profiler.summary());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};

// ============================================================================
// LayerProfile
// ============================================================================

/// Timing and computational cost data for a single layer in one training epoch.
///
/// All timing values are in milliseconds. FLOPS counts are estimates based on
/// the layer's arithmetic operations. `param_memory_bytes` is the number of
/// bytes occupied by trainable parameters (weights + biases), calculated as
/// `parameter_count() * size_of::<f32>()`.
///
/// # Fields
///
/// * `layer_name` – human-readable identifier (e.g. `"Dense 784→512"`)
/// * `forward_ms` – cumulative forward-pass time across all mini-batches (ms)
/// * `backward_ms` – cumulative backward-pass time across all mini-batches (ms)
/// * `update_ms` – cumulative parameter-update time across all mini-batches (ms)
/// * `flops_forward` – estimated multiply-add operations in the forward pass
/// * `flops_backward` – estimated multiply-add operations in the backward pass
/// * `param_memory_bytes` – bytes used by all trainable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    /// Human-readable identifier for the layer (e.g. `"Dense 784→512"`).
    pub layer_name: String,
    /// Cumulative forward-pass time in milliseconds.
    pub forward_ms: f64,
    /// Cumulative backward-pass time in milliseconds.
    pub backward_ms: f64,
    /// Cumulative parameter-update time in milliseconds.
    pub update_ms: f64,
    /// Estimated number of floating-point operations in the forward pass.
    pub flops_forward: u64,
    /// Estimated number of floating-point operations in the backward pass.
    pub flops_backward: u64,
    /// Bytes occupied by all trainable parameters (`parameter_count * 4`).
    pub param_memory_bytes: usize,
}

impl LayerProfile {
    /// Total time spent on this layer (forward + backward + update) in ms.
    pub fn total_ms(&self) -> f64 {
        self.forward_ms + self.backward_ms + self.update_ms
    }

    /// Total estimated FLOPS (forward + backward).
    pub fn total_flops(&self) -> u64 {
        self.flops_forward.saturating_add(self.flops_backward)
    }

    /// Parameter memory in kibibytes (1 KiB = 1024 bytes).
    pub fn param_memory_kib(&self) -> f64 {
        self.param_memory_bytes as f64 / 1024.0
    }
}

// ============================================================================
// EpochProfile
// ============================================================================

/// Profiling data for all layers collected during a single training epoch.
///
/// An [`EpochProfile`] is built incrementally via [`Profiler::record_layer`]
/// and finalised by [`Profiler::end_epoch`].
///
/// # Examples
///
/// ```
/// use rust_neural_networks::profiling::{EpochProfile, LayerProfile};
///
/// let profile = EpochProfile {
///     epoch: 1,
///     layers: vec![LayerProfile {
///         layer_name: "Dense".to_string(),
///         forward_ms: 1.0,
///         backward_ms: 2.0,
///         update_ms: 0.5,
///         flops_forward: 1_000_000,
///         flops_backward: 1_000_000,
///         param_memory_bytes: 4096,
///     }],
/// };
///
/// assert!((profile.total_forward_ms() - 1.0).abs() < 1e-9);
/// assert_eq!(profile.total_flops(), 2_000_000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochProfile {
    /// 1-based epoch number.
    pub epoch: usize,
    /// Profiling data for every layer that was recorded during this epoch.
    pub layers: Vec<LayerProfile>,
}

impl EpochProfile {
    /// Sum of `forward_ms` across all layers.
    pub fn total_forward_ms(&self) -> f64 {
        self.layers.iter().map(|l| l.forward_ms).sum()
    }

    /// Sum of `backward_ms` across all layers.
    pub fn total_backward_ms(&self) -> f64 {
        self.layers.iter().map(|l| l.backward_ms).sum()
    }

    /// Sum of `update_ms` across all layers.
    pub fn total_update_ms(&self) -> f64 {
        self.layers.iter().map(|l| l.update_ms).sum()
    }

    /// Total wall-clock time for all layers in this epoch (ms).
    pub fn total_ms(&self) -> f64 {
        self.layers.iter().map(|l| l.total_ms()).sum()
    }

    /// Total estimated FLOPS (forward + backward) across all layers.
    pub fn total_flops(&self) -> u64 {
        self.layers
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.total_flops()))
    }

    /// Total parameter memory in bytes across all layers.
    pub fn total_param_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.param_memory_bytes).sum()
    }
}

// ============================================================================
// Profiler
// ============================================================================

/// Accumulates per-layer, per-epoch profiling data across a full training run.
///
/// Usage pattern:
/// 1. Call [`Profiler::start_epoch`] at the beginning of each epoch.
/// 2. Call [`Profiler::record_layer`] once per layer after timing its passes.
/// 3. Call [`Profiler::end_epoch`] when the epoch is complete.
/// 4. Use [`Profiler::summary`] / [`Profiler::print_summary`] for a formatted
///    table or [`Profiler::to_csv_string`] / [`Profiler::save_csv`] for export.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::profiling::{LayerProfile, Profiler};
///
/// let mut profiler = Profiler::new();
///
/// for epoch in 1..=3 {
///     profiler.start_epoch(epoch);
///     profiler.record_layer(LayerProfile {
///         layer_name: "Dense 784→512".to_string(),
///         forward_ms: 1.5,
///         backward_ms: 2.3,
///         update_ms: 0.4,
///         flops_forward: 100_663_296,
///         flops_backward: 100_663_296,
///         param_memory_bytes: 1_608_704,
///     });
///     profiler.end_epoch();
/// }
///
/// assert_eq!(profiler.epochs().len(), 3);
/// println!("{}", profiler.summary());
/// ```
#[derive(Debug, Default)]
pub struct Profiler {
    /// Completed epoch profiles.
    completed_epochs: Vec<EpochProfile>,
    /// Epoch number for the epoch currently being recorded.
    current_epoch: Option<usize>,
    /// Layers accumulated during the current epoch.
    current_layers: Vec<LayerProfile>,
}

impl Profiler {
    /// Creates an empty [`Profiler`] with no recorded data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signals the start of a new training epoch.
    ///
    /// Any previously accumulated layer data from a forgotten `end_epoch` call
    /// is discarded. The epoch counter is 1-based by convention, matching the
    /// training log format used by the rest of the codebase.
    ///
    /// # Arguments
    ///
    /// * `epoch` – 1-based epoch number.
    pub fn start_epoch(&mut self, epoch: usize) {
        self.current_epoch = Some(epoch);
        self.current_layers.clear();
    }

    /// Appends a completed [`LayerProfile`] to the current epoch.
    ///
    /// Must be called after [`start_epoch`](Profiler::start_epoch) and before
    /// [`end_epoch`](Profiler::end_epoch).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if called without a preceding `start_epoch`.
    pub fn record_layer(&mut self, profile: LayerProfile) {
        debug_assert!(
            self.current_epoch.is_some(),
            "record_layer called without start_epoch"
        );
        self.current_layers.push(profile);
    }

    /// Finalises the current epoch and stores it in the completed list.
    ///
    /// After this call, the accumulated layer data for the epoch is moved into
    /// a new [`EpochProfile`] and appended to the internal history.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if called without a preceding `start_epoch`.
    pub fn end_epoch(&mut self) {
        let epoch = self
            .current_epoch
            .take()
            .expect("end_epoch called without start_epoch");

        let layers = std::mem::take(&mut self.current_layers);
        self.completed_epochs.push(EpochProfile { epoch, layers });
    }

    /// Returns a slice of all completed [`EpochProfile`]s in recording order.
    pub fn epochs(&self) -> &[EpochProfile] {
        &self.completed_epochs
    }

    /// Returns the total number of completed epochs recorded.
    pub fn epoch_count(&self) -> usize {
        self.completed_epochs.len()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Summary helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Produces a formatted text table showing per-layer timing averages,
    /// total FLOPS, and parameter memory across all recorded epochs.
    ///
    /// The table has one row per unique layer (identified by `layer_name`)
    /// showing averages over all completed epochs:
    ///
    /// ```text
    /// Layer Name       | Avg Fwd (ms) | Avg Bwd (ms) | Total FLOPs | Param Mem (KiB)
    /// Dense 784→512    |        1.234 |        2.345 |   201326592 |        1570.0
    /// Dense 512→10     |        0.456 |        0.789 |    10485760 |          20.0
    /// ```
    ///
    /// Returns an empty string if no epochs have been recorded.
    pub fn summary(&self) -> String {
        if self.completed_epochs.is_empty() {
            return String::new();
        }

        // Collect unique layer names in order of first appearance.
        let layer_names: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            let mut names = Vec::new();
            for ep in &self.completed_epochs {
                for l in &ep.layers {
                    if seen.insert(l.layer_name.clone()) {
                        names.push(l.layer_name.clone());
                    }
                }
            }
            names
        };

        let n_epochs = self.completed_epochs.len() as f64;
        let mut out = String::new();

        // Header
        let _ = writeln!(
            out,
            "{:<24} | {:>14} | {:>14} | {:>14} | {:>16}",
            "Layer Name", "Avg Fwd (ms)", "Avg Bwd (ms)", "Total FLOPs", "Param Mem (KiB)"
        );
        let _ = writeln!(out, "{}", "-".repeat(92));

        for name in &layer_names {
            let mut sum_fwd = 0.0_f64;
            let mut sum_bwd = 0.0_f64;
            let mut total_flops = 0u64;
            let mut param_bytes = 0usize;
            let mut count = 0u32;

            for ep in &self.completed_epochs {
                for l in &ep.layers {
                    if &l.layer_name == name {
                        sum_fwd += l.forward_ms;
                        sum_bwd += l.backward_ms;
                        total_flops = total_flops.saturating_add(l.total_flops());
                        param_bytes = l.param_memory_bytes;
                        count += 1;
                    }
                }
            }

            if count == 0 {
                continue;
            }

            let avg_fwd = sum_fwd / n_epochs;
            let avg_bwd = sum_bwd / n_epochs;
            let param_kib = param_bytes as f64 / 1024.0;

            let _ = writeln!(
                out,
                "{:<24} | {:>14.3} | {:>14.3} | {:>14} | {:>16.1}",
                name, avg_fwd, avg_bwd, total_flops, param_kib
            );
        }

        out
    }

    /// Prints the formatted summary table to `stdout`.
    ///
    /// This is a convenience wrapper around [`Profiler::summary`]. If no epochs
    /// have been recorded, nothing is printed.
    pub fn print_summary(&self) {
        let s = self.summary();
        if !s.is_empty() {
            print!("{s}");
        }
    }

    /// Serialises all profiling data as a CSV string.
    ///
    /// # CSV Format
    ///
    /// ```text
    /// epoch,layer_name,forward_ms,backward_ms,update_ms,flops_forward,flops_backward,param_memory_bytes
    /// 1,Dense 784→512,1.5,2.3,0.4,100663296,100663296,1608704
    /// ```
    ///
    /// Returns only the header line if no epochs have been recorded.
    pub fn to_csv_string(&self) -> String {
        let mut out = String::from(
            "epoch,layer_name,forward_ms,backward_ms,update_ms,flops_forward,flops_backward,param_memory_bytes\n",
        );

        for ep in &self.completed_epochs {
            for l in &ep.layers {
                let _ = writeln!(
                    out,
                    "{},{},{},{},{},{},{},{}",
                    ep.epoch,
                    l.layer_name,
                    l.forward_ms,
                    l.backward_ms,
                    l.update_ms,
                    l.flops_forward,
                    l.flops_backward,
                    l.param_memory_bytes
                );
            }
        }

        out
    }

    /// Writes the CSV representation of all profiling data to `path`.
    ///
    /// Creates (or truncates) the file at `path` and writes the same content
    /// as [`to_csv_string`](Profiler::to_csv_string), using a buffered writer
    /// for efficiency — consistent with [`CsvTrainingLogger`](crate::training::CsvTrainingLogger).
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the file cannot be created or written.
    pub fn save_csv(&self, path: &str) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(self.to_csv_string().as_bytes())?;
        Ok(())
    }

    /// Generates a self-contained HTML profiling report and writes it to `path`.
    ///
    /// The report includes:
    /// * A stacked bar chart (Chart.js) of per-layer forward / backward /
    ///   update timing averaged over all epochs.
    /// * A doughnut chart showing the FLOPS distribution by layer.
    /// * An HTML table of parameter memory per layer.
    ///
    /// Chart.js is loaded from the [jsDelivr CDN](https://cdn.jsdelivr.net/npm/chart.js)
    /// so no additional build dependencies are required.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the file cannot be created or written.
    pub fn generate_html_report(&self, path: &str) -> Result<(), std::io::Error> {
        let html = self.build_html_report();
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(html.as_bytes())?;
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────────
    // Private HTML builder
    // ────────────────────────────────────────────────────────────────────────

    fn build_html_report(&self) -> String {
        // Collect unique layer names in first-appearance order.
        let layer_names: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            let mut names = Vec::new();
            for ep in &self.completed_epochs {
                for l in &ep.layers {
                    if seen.insert(l.layer_name.clone()) {
                        names.push(l.layer_name.clone());
                    }
                }
            }
            names
        };

        let n_epochs = self.completed_epochs.len().max(1) as f64;

        // Per-layer averages for the timing bar chart.
        let mut avg_fwd: Vec<f64> = Vec::new();
        let mut avg_bwd: Vec<f64> = Vec::new();
        let mut avg_upd: Vec<f64> = Vec::new();
        let mut total_flops_per_layer: Vec<u64> = Vec::new();
        let mut param_mem_per_layer: Vec<usize> = Vec::new();

        for name in &layer_names {
            let mut sf = 0.0_f64;
            let mut sb = 0.0_f64;
            let mut su = 0.0_f64;
            let mut flops = 0u64;
            let mut pmem = 0usize;

            for ep in &self.completed_epochs {
                for l in &ep.layers {
                    if &l.layer_name == name {
                        sf += l.forward_ms;
                        sb += l.backward_ms;
                        su += l.update_ms;
                        flops = flops.saturating_add(l.total_flops());
                        pmem = l.param_memory_bytes;
                    }
                }
            }

            avg_fwd.push(sf / n_epochs);
            avg_bwd.push(sb / n_epochs);
            avg_upd.push(su / n_epochs);
            total_flops_per_layer.push(flops);
            param_mem_per_layer.push(pmem);
        }

        // JSON arrays for Chart.js injection.
        let labels_json = json_string_array(&layer_names);
        let fwd_json = json_f64_array(&avg_fwd);
        let bwd_json = json_f64_array(&avg_bwd);
        let upd_json = json_f64_array(&avg_upd);
        let flops_json = json_u64_array(&total_flops_per_layer);

        // Memory table rows.
        let mem_table_rows: String = layer_names
            .iter()
            .zip(param_mem_per_layer.iter())
            .map(|(name, bytes)| {
                format!(
                    "  <tr><td>{}</td><td>{}</td><td>{:.2}</td></tr>\n",
                    escape_html(name),
                    bytes,
                    *bytes as f64 / 1024.0
                )
            })
            .collect();

        let epoch_count = self.completed_epochs.len();

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Neural Network Profiling Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; background: #f9f9f9; color: #222; }}
    h1 {{ color: #333; }}
    h2 {{ color: #555; margin-top: 2rem; }}
    .chart-container {{ width: 100%; max-width: 900px; margin: 1rem 0; background: #fff;
                         padding: 1rem; border-radius: 8px; box-shadow: 0 2px 6px #0002; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 600px;
             background: #fff; border-radius: 8px; box-shadow: 0 2px 6px #0002; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 1rem; text-align: right; }}
    th {{ background: #f0f0f0; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .meta {{ color: #888; font-size: 0.9rem; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>Neural Network Profiling Report</h1>
  <p class="meta">Epochs recorded: {epoch_count}</p>

  <h2>Per-Layer Timing Breakdown (avg ms / epoch)</h2>
  <div class="chart-container">
    <canvas id="timingChart"></canvas>
  </div>

  <h2>FLOPS Distribution by Layer</h2>
  <div class="chart-container" style="max-width:500px">
    <canvas id="flopsChart"></canvas>
  </div>

  <h2>Parameter Memory Usage</h2>
  <table>
    <thead>
      <tr><th>Layer</th><th>Bytes</th><th>KiB</th></tr>
    </thead>
    <tbody>
{mem_table_rows}    </tbody>
  </table>

  <script>
    const labels = {labels_json};
    const fwdData = {fwd_json};
    const bwdData = {bwd_json};
    const updData = {upd_json};
    const flopsData = {flops_json};

    new Chart(document.getElementById('timingChart'), {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [
          {{ label: 'Forward (ms)', data: fwdData, backgroundColor: 'rgba(54, 162, 235, 0.7)' }},
          {{ label: 'Backward (ms)', data: bwdData, backgroundColor: 'rgba(255, 99, 132, 0.7)' }},
          {{ label: 'Update (ms)', data: updData, backgroundColor: 'rgba(75, 192, 192, 0.7)' }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }}, title: {{ display: false }} }},
        scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, title: {{ display: true, text: 'ms' }} }} }}
      }}
    }});

    new Chart(document.getElementById('flopsChart'), {{
      type: 'doughnut',
      data: {{
        labels: labels,
        datasets: [{{ data: flopsData, backgroundColor: [
          'rgba(54,162,235,0.7)', 'rgba(255,99,132,0.7)',
          'rgba(75,192,192,0.7)', 'rgba(255,206,86,0.7)',
          'rgba(153,102,255,0.7)', 'rgba(255,159,64,0.7)'
        ] }}]
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ position: 'right' }} }} }}
    }});
  </script>
</body>
</html>
"#
        )
    }
}

// ============================================================================
// Private HTML/JSON helpers
// ============================================================================

/// Escapes HTML special characters to prevent injection in the report.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Formats a slice of strings as a JSON array: `["a","b","c"]`.
fn json_string_array(items: &[String]) -> String {
    let inner: Vec<String> = items
        .iter()
        .map(|s| format!("\"{}\"", escape_html(s)))
        .collect();
    format!("[{}]", inner.join(","))
}

/// Formats a slice of `f64` values as a JSON array: `[1.23,4.56]`.
fn json_f64_array(items: &[f64]) -> String {
    let inner: Vec<String> = items.iter().map(|v| format!("{v:.4}")).collect();
    format!("[{}]", inner.join(","))
}

/// Formats a slice of `u64` values as a JSON array: `[1000,2000]`.
fn json_u64_array(items: &[u64]) -> String {
    let inner: Vec<String> = items.iter().map(|v| v.to_string()).collect();
    format!("[{}]", inner.join(","))
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(name: &str, fwd: f64, bwd: f64, upd: f64) -> LayerProfile {
        LayerProfile {
            layer_name: name.to_string(),
            forward_ms: fwd,
            backward_ms: bwd,
            update_ms: upd,
            flops_forward: 1_000_000,
            flops_backward: 1_000_000,
            param_memory_bytes: 4096,
        }
    }

    #[test]
    fn test_layer_profile_total_ms() {
        let lp = make_layer("Dense", 1.0, 2.0, 0.5);
        assert!((lp.total_ms() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_layer_profile_total_flops() {
        let lp = make_layer("Dense", 0.0, 0.0, 0.0);
        assert_eq!(lp.total_flops(), 2_000_000);
    }

    #[test]
    fn test_layer_profile_param_memory_kib() {
        let lp = make_layer("Dense", 0.0, 0.0, 0.0);
        assert!((lp.param_memory_kib() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_epoch_profile_aggregates() {
        let ep = EpochProfile {
            epoch: 1,
            layers: vec![
                make_layer("A", 1.0, 2.0, 0.5),
                make_layer("B", 3.0, 4.0, 1.0),
            ],
        };
        assert!((ep.total_forward_ms() - 4.0).abs() < 1e-9);
        assert!((ep.total_backward_ms() - 6.0).abs() < 1e-9);
        assert!((ep.total_update_ms() - 1.5).abs() < 1e-9);
        assert!((ep.total_ms() - 11.5).abs() < 1e-9);
        assert_eq!(ep.total_flops(), 4_000_000);
        assert_eq!(ep.total_param_memory_bytes(), 8192);
    }

    #[test]
    fn test_profiler_accumulates_epochs() {
        let mut profiler = Profiler::new();
        for epoch in 1..=3 {
            profiler.start_epoch(epoch);
            profiler.record_layer(make_layer("Dense", 1.0, 2.0, 0.5));
            profiler.end_epoch();
        }
        assert_eq!(profiler.epoch_count(), 3);
        assert_eq!(profiler.epochs()[0].epoch, 1);
        assert_eq!(profiler.epochs()[2].epoch, 3);
    }

    #[test]
    fn test_profiler_start_clears_layers() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("A", 1.0, 1.0, 0.1));
        // Re-start before end_epoch – should discard the layer from epoch 1.
        profiler.start_epoch(2);
        profiler.record_layer(make_layer("B", 2.0, 2.0, 0.2));
        profiler.end_epoch();

        assert_eq!(profiler.epoch_count(), 1);
        assert_eq!(profiler.epochs()[0].epoch, 2);
        assert_eq!(profiler.epochs()[0].layers[0].layer_name, "B");
    }

    #[test]
    fn test_profiler_summary_non_empty() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense 784→512", 1.5, 2.3, 0.4));
        profiler.end_epoch();

        let s = profiler.summary();
        assert!(!s.is_empty());
        assert!(s.contains("Dense 784→512"));
    }

    #[test]
    fn test_profiler_summary_empty_when_no_epochs() {
        let profiler = Profiler::new();
        assert!(profiler.summary().is_empty());
    }

    #[test]
    fn test_to_csv_string_header() {
        let profiler = Profiler::new();
        let csv = profiler.to_csv_string();
        assert!(csv.starts_with(
            "epoch,layer_name,forward_ms,backward_ms,update_ms,flops_forward,flops_backward,param_memory_bytes"
        ));
    }

    #[test]
    fn test_to_csv_string_data_rows() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(LayerProfile {
            layer_name: "Dense".to_string(),
            forward_ms: 1.5,
            backward_ms: 2.3,
            update_ms: 0.4,
            flops_forward: 500_000,
            flops_backward: 500_000,
            param_memory_bytes: 4096,
        });
        profiler.end_epoch();

        let csv = profiler.to_csv_string();
        assert!(csv.contains("1,Dense,1.5,2.3,0.4,500000,500000,4096"));
    }

    #[test]
    fn test_serde_roundtrip_layer_profile() {
        let lp = make_layer("Dense 512→10", 0.5, 1.0, 0.1);
        let json = serde_json::to_string(&lp).expect("serialize");
        let back: LayerProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.layer_name, lp.layer_name);
        assert!((back.forward_ms - lp.forward_ms).abs() < 1e-9);
    }

    #[test]
    fn test_serde_roundtrip_epoch_profile() {
        let ep = EpochProfile {
            epoch: 5,
            layers: vec![make_layer("A", 1.0, 2.0, 0.5)],
        };
        let json = serde_json::to_string(&ep).expect("serialize");
        let back: EpochProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.epoch, 5);
        assert_eq!(back.layers.len(), 1);
    }

    #[test]
    fn test_json_helpers() {
        assert_eq!(
            json_string_array(&["a".to_string(), "b".to_string()]),
            r#"["a","b"]"#
        );
        assert_eq!(json_u64_array(&[0, 1_000_000]), "[0,1000000]");
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("Dense<512>"), "Dense&lt;512&gt;");
        assert_eq!(escape_html("a&b"), "a&amp;b");
    }

    #[test]
    fn test_generate_html_report_content() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense 784→512", 1.5, 2.3, 0.4));
        profiler.end_epoch();

        let html = profiler.build_html_report();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("chart.js"));
        assert!(html.contains("Dense 784"));
        assert!(html.contains("forward_ms") || html.contains("Forward (ms)"));
    }

    #[test]
    fn test_save_csv_creates_file() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense", 1.0, 2.0, 0.5));
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_profiling_save.csv");
        let path_str = path.to_string_lossy().to_string();
        profiler.save_csv(&path_str).expect("save_csv failed");

        let contents = std::fs::read_to_string(&path).expect("read file");
        assert!(contents.contains("epoch,layer_name"));
        assert!(contents.contains("Dense"));
        let _ = std::fs::remove_file(&path);
    }
}
