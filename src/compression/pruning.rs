//! Magnitude-based pruning for neural network weight compression
//!
//! Provides weight pruning by zeroing out low-magnitude parameters, reducing
//! the effective number of non-zero weights (sparsity) to improve inference
//! efficiency or enable downstream sparse compression.
//!
//! # Overview
//!
//! Magnitude-based pruning removes weights whose absolute value falls below a
//! threshold, treating them as negligible contributions to the network output.
//! The resulting sparse weight matrices can be stored and computed more
//! efficiently using sparse representations.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::compression::pruning::{
//!     prune_by_magnitude, compute_sparsity, suggest_threshold,
//! };
//!
//! let mut weights = vec![0.5_f32, -0.01, 0.3, 0.005, -0.8, 0.02];
//! let threshold = 0.05;
//! let report = prune_by_magnitude(&mut weights, threshold);
//!
//! assert!(report.sparsity > 0.0);
//! assert!(report.pruned_params > 0);
//! assert_eq!(report.total_params, 6);
//! ```

use crate::layers::DenseLayer;
use std::fmt;

/// Summary of a pruning operation on a set of weights.
///
/// Reports how many parameters were examined, how many were zeroed out,
/// the resulting sparsity fraction, and the threshold that was applied.
///
/// # Fields
///
/// * `total_params` - Total number of weight parameters examined
/// * `pruned_params` - Number of parameters set to zero by the pruning operation
/// * `sparsity` - Fraction of zero weights after pruning (`pruned_params / total_params`)
/// * `threshold` - The magnitude threshold applied during pruning
#[derive(Debug, Clone, PartialEq)]
pub struct PruningReport {
    /// Total number of weight parameters examined.
    pub total_params: usize,
    /// Number of parameters set to zero by the pruning operation.
    pub pruned_params: usize,
    /// Fraction of zero weights after pruning (`pruned_params / total_params`).
    pub sparsity: f32,
    /// The magnitude threshold applied: weights with `|w| < threshold` were zeroed.
    pub threshold: f32,
}

impl fmt::Display for PruningReport {
    /// Formats the pruning report as a human-readable summary.
    ///
    /// # Example
    ///
    /// ```
    /// use rust_neural_networks::compression::pruning::PruningReport;
    ///
    /// let report = PruningReport {
    ///     total_params: 1000,
    ///     pruned_params: 500,
    ///     sparsity: 0.5,
    ///     threshold: 0.01,
    /// };
    /// let s = format!("{}", report);
    /// assert!(s.contains("sparsity"));
    /// assert!(s.contains("500"));
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PruningReport {{ total_params: {}, pruned_params: {}, sparsity: {:.4}, threshold: {:.6} }}",
            self.total_params, self.pruned_params, self.sparsity, self.threshold
        )
    }
}

/// Prune weights in-place by zeroing those whose absolute value is below `threshold`.
///
/// Each weight `w` with `|w| < threshold` is set to `0.0`. The function returns
/// a [`PruningReport`] summarising the operation.
///
/// # Arguments
///
/// * `weights` - Mutable slice of floating-point weights to prune in-place
/// * `threshold` - Magnitude threshold; weights with `|w| < threshold` are zeroed
///
/// # Returns
///
/// A [`PruningReport`] containing the total parameter count, number of pruned
/// parameters, resulting sparsity, and the applied threshold.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::pruning::prune_by_magnitude;
///
/// let mut weights = vec![0.5_f32, -0.01, 0.3, 0.005, -0.8];
/// let report = prune_by_magnitude(&mut weights, 0.05);
///
/// // The two small weights (0.01, 0.005) should be zeroed
/// assert_eq!(report.pruned_params, 2);
/// assert_eq!(report.total_params, 5);
/// assert_eq!(weights[1], 0.0);
/// assert_eq!(weights[3], 0.0);
/// assert_ne!(weights[0], 0.0);
/// ```
pub fn prune_by_magnitude(weights: &mut [f32], threshold: f32) -> PruningReport {
    let total_params = weights.len();
    let mut pruned_params = 0usize;

    for w in weights.iter_mut() {
        if w.abs() < threshold {
            *w = 0.0;
            pruned_params += 1;
        }
    }

    let sparsity = if total_params > 0 {
        pruned_params as f32 / total_params as f32
    } else {
        0.0
    };

    PruningReport {
        total_params,
        pruned_params,
        sparsity,
        threshold,
    }
}

/// Prune a [`DenseLayer`] by zeroing low-magnitude weights, returning a new layer.
///
/// Creates a copy of the layer's weights, applies magnitude-based pruning to the
/// weight matrix (biases are left unchanged), and constructs a new [`DenseLayer`]
/// via [`DenseLayer::new_with_weights`].
///
/// # Arguments
///
/// * `dense` - Reference to the dense layer to prune
/// * `threshold` - Magnitude threshold; weights with `|w| < threshold` are zeroed
///
/// # Returns
///
/// A tuple of:
/// - `DenseLayer` — New layer with pruned weights and original biases
/// - `PruningReport` — Summary of the pruning applied to the weight matrix
///
/// # Examples
///
/// ```
/// use rust_neural_networks::layers::{DenseLayer, Layer};
/// use rust_neural_networks::utils::rng::SimpleRng;
/// use rust_neural_networks::compression::pruning::prune_dense_layer;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = DenseLayer::new(4, 2, &mut rng);
/// let (pruned_layer, report) = prune_dense_layer(&layer, 0.05);
///
/// assert_eq!(pruned_layer.input_size(), 4);
/// assert_eq!(pruned_layer.output_size(), 2);
/// assert_eq!(report.total_params, 4 * 2); // biases excluded from report
/// assert!(report.sparsity >= 0.0 && report.sparsity <= 1.0);
/// ```
pub fn prune_dense_layer(dense: &DenseLayer, threshold: f32) -> (DenseLayer, PruningReport) {
    let mut pruned_weights = dense.weights().to_vec();
    let report = prune_by_magnitude(&mut pruned_weights, threshold);

    let pruned_layer = DenseLayer::new_with_weights(
        dense.input_size(),
        dense.output_size(),
        pruned_weights,
        dense.biases().to_vec(),
    );

    (pruned_layer, report)
}

/// Compute the fraction of zero-valued weights in a slice.
///
/// Returns a value in `[0.0, 1.0]` representing how sparse the weight tensor is.
/// Returns `0.0` for an empty slice.
///
/// # Arguments
///
/// * `weights` - Slice of floating-point weights to examine
///
/// # Returns
///
/// Sparsity as `zero_count / total_count`, or `0.0` if the slice is empty.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::pruning::compute_sparsity;
///
/// let weights = vec![0.0_f32, 1.0, 0.0, -0.5, 0.0];
/// let sparsity = compute_sparsity(&weights);
/// assert!((sparsity - 0.6).abs() < 1e-6, "Expected 0.6, got {}", sparsity);
///
/// let all_zeros = vec![0.0_f32; 4];
/// assert_eq!(compute_sparsity(&all_zeros), 1.0);
///
/// let no_zeros = vec![1.0_f32, 2.0, 3.0];
/// assert_eq!(compute_sparsity(&no_zeros), 0.0);
///
/// let empty: Vec<f32> = vec![];
/// assert_eq!(compute_sparsity(&empty), 0.0);
/// ```
pub fn compute_sparsity(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }

    let zero_count = weights.iter().filter(|&&w| w == 0.0).count();
    zero_count as f32 / weights.len() as f32
}

/// Compute the magnitude threshold required to achieve a target sparsity level.
///
/// Sorts the absolute values of all weights and returns the value at the
/// `target_sparsity` percentile. Applying this threshold via [`prune_by_magnitude`]
/// will zero out approximately `target_sparsity` fraction of the weights.
///
/// Returns `0.0` if the slice is empty.
///
/// # Arguments
///
/// * `weights` - Slice of floating-point weights to analyse
/// * `target_sparsity` - Desired fraction of weights to prune, in `[0.0, 1.0]`
///
/// # Returns
///
/// A threshold value such that approximately `target_sparsity` of weights have
/// absolute value below it.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::compression::pruning::{suggest_threshold, prune_by_magnitude, compute_sparsity};
///
/// let weights = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
/// let threshold = suggest_threshold(&weights, 0.5);
///
/// // Applying the threshold should yield approximately 50% sparsity
/// let mut w = weights.clone();
/// prune_by_magnitude(&mut w, threshold);
/// let sparsity = compute_sparsity(&w);
/// assert!(sparsity >= 0.4 && sparsity <= 0.6, "Expected ~0.5 sparsity, got {}", sparsity);
/// ```
pub fn suggest_threshold(weights: &[f32], target_sparsity: f32) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }

    // Collect and sort absolute values
    let mut abs_values: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute the index corresponding to target_sparsity percentile
    let n = abs_values.len();
    let target_index = (target_sparsity * n as f32).floor() as usize;
    let index = target_index.min(n - 1);

    abs_values[index]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{DenseLayer, Layer};
    use crate::utils::rng::SimpleRng;

    #[test]
    fn test_prune_by_magnitude_zeros_small_weights() {
        let mut weights = vec![0.5_f32, -0.01, 0.3, 0.005, -0.8];
        let report = prune_by_magnitude(&mut weights, 0.05);

        assert_eq!(weights[0], 0.5);
        assert_eq!(weights[1], 0.0); // -0.01 < 0.05
        assert_eq!(weights[2], 0.3);
        assert_eq!(weights[3], 0.0); // 0.005 < 0.05
        assert_eq!(weights[4], -0.8);
        assert_eq!(report.pruned_params, 2);
        assert_eq!(report.total_params, 5);
    }

    #[test]
    fn test_prune_by_magnitude_report_sparsity() {
        let mut weights = vec![0.0_f32, 0.0, 1.0, 1.0];
        let report = prune_by_magnitude(&mut weights, 0.5);

        // 0.0 values are already zero (|w| < 0.5), 1.0 values are kept
        assert_eq!(report.pruned_params, 2);
        assert_eq!(report.total_params, 4);
        assert!((report.sparsity - 0.5).abs() < 1e-6);
        assert_eq!(report.threshold, 0.5);
    }

    #[test]
    fn test_prune_by_magnitude_empty_slice() {
        let mut weights: Vec<f32> = vec![];
        let report = prune_by_magnitude(&mut weights, 0.1);

        assert_eq!(report.total_params, 0);
        assert_eq!(report.pruned_params, 0);
        assert_eq!(report.sparsity, 0.0);
    }

    #[test]
    fn test_prune_by_magnitude_no_pruning() {
        let mut weights = vec![1.0_f32, 2.0, 3.0];
        let report = prune_by_magnitude(&mut weights, 0.5);

        assert_eq!(report.pruned_params, 0);
        assert_eq!(report.sparsity, 0.0);
        assert_eq!(weights, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_prune_by_magnitude_all_pruned() {
        let mut weights = vec![0.01_f32, 0.02, 0.03];
        let report = prune_by_magnitude(&mut weights, 0.1);

        assert_eq!(report.pruned_params, 3);
        assert!((report.sparsity - 1.0).abs() < 1e-6);
        assert!(weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_pruning_report_display() {
        let report = PruningReport {
            total_params: 1000,
            pruned_params: 500,
            sparsity: 0.5,
            threshold: 0.01,
        };
        let s = format!("{}", report);
        assert!(s.contains("total_params"));
        assert!(s.contains("1000"));
        assert!(s.contains("500"));
        assert!(s.contains("sparsity"));
    }

    #[test]
    fn test_compute_sparsity_mixed() {
        let weights = vec![0.0_f32, 1.0, 0.0, -0.5, 0.0];
        let sparsity = compute_sparsity(&weights);
        assert!((sparsity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_compute_sparsity_all_zeros() {
        let weights = vec![0.0_f32; 4];
        assert_eq!(compute_sparsity(&weights), 1.0);
    }

    #[test]
    fn test_compute_sparsity_no_zeros() {
        let weights = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(compute_sparsity(&weights), 0.0);
    }

    #[test]
    fn test_compute_sparsity_empty() {
        let weights: Vec<f32> = vec![];
        assert_eq!(compute_sparsity(&weights), 0.0);
    }

    #[test]
    fn test_suggest_threshold_half_sparsity() {
        let weights = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let threshold = suggest_threshold(&weights, 0.5);

        let mut w = weights.clone();
        prune_by_magnitude(&mut w, threshold);
        let sparsity = compute_sparsity(&w);
        assert!(
            (0.4..=0.6).contains(&sparsity),
            "Expected ~0.5 sparsity, got {}",
            sparsity
        );
    }

    #[test]
    fn test_suggest_threshold_zero_sparsity() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0];
        let threshold = suggest_threshold(&weights, 0.0);
        // At index 0 of sorted abs values => 1.0
        assert!((threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_suggest_threshold_empty() {
        let weights: Vec<f32> = vec![];
        assert_eq!(suggest_threshold(&weights, 0.5), 0.0);
    }

    #[test]
    fn test_prune_dense_layer_preserves_dimensions() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 2, &mut rng);
        let (pruned_layer, report) = prune_dense_layer(&layer, 0.05);

        assert_eq!(pruned_layer.input_size(), 4);
        assert_eq!(pruned_layer.output_size(), 2);
        assert_eq!(report.total_params, 4 * 2);
        assert!(report.sparsity >= 0.0 && report.sparsity <= 1.0);
    }

    #[test]
    fn test_prune_dense_layer_biases_unchanged() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(4, 2, &mut rng);
        let original_biases = layer.biases().to_vec();
        let (pruned_layer, _) = prune_dense_layer(&layer, 0.5);

        assert_eq!(pruned_layer.biases(), original_biases.as_slice());
    }

    #[test]
    fn test_prune_dense_layer_weights_pruned() {
        // Use known weights that will be pruned
        let weights = vec![0.5_f32, 0.01, 0.3, 0.005]; // 2×2
        let biases = vec![0.1_f32, 0.2];
        let layer = DenseLayer::new_with_weights(2, 2, weights, biases);

        let (pruned_layer, report) = prune_dense_layer(&layer, 0.05);

        // 0.01 and 0.005 should be pruned
        assert_eq!(pruned_layer.weights()[1], 0.0);
        assert_eq!(pruned_layer.weights()[3], 0.0);
        assert_eq!(report.pruned_params, 2);
    }

    #[test]
    fn test_prune_dense_layer_forward_pass_valid() {
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(3, 2, &mut rng);
        let (pruned_layer, _) = prune_dense_layer(&layer, 0.01);

        let input = vec![1.0_f32, 0.5, -0.5];
        let mut output = vec![0.0_f32; 2];
        pruned_layer.forward(&input, &mut output, 1);

        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
