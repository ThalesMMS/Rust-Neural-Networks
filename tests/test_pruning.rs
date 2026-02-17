//! Tests for magnitude-based pruning functionality
//!
//! These tests verify the pruning implementation in the compression module,
//! including weight zeroing, sparsity reporting, threshold suggestion, and
//! layer-level pruning with forward-pass validity.

use rust_neural_networks::compression::pruning::{
    compute_sparsity, prune_by_magnitude, prune_dense_layer, suggest_threshold, PruningReport,
};
use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

/// Verify that weights below the threshold are zeroed out and weights above it are unchanged.
///
/// Uses a hand-crafted weight vector where two values fall below the threshold
/// (0.05) and three values exceed it. Confirms exact zero-zeroing of small weights
/// and exact preservation of large weights.
#[test]
fn test_prune_by_magnitude_zeros_small() {
    let mut weights = vec![0.5_f32, -0.01, 0.3, 0.005, -0.8];
    prune_by_magnitude(&mut weights, 0.05);

    // Large weights survive
    assert_eq!(weights[0], 0.5, "weight 0.5 should survive pruning");
    assert_eq!(weights[2], 0.3, "weight 0.3 should survive pruning");
    assert_eq!(weights[4], -0.8, "weight -0.8 should survive pruning");

    // Small weights are zeroed
    assert_eq!(
        weights[1], 0.0,
        "weight -0.01 should be zeroed (|w| < 0.05)"
    );
    assert_eq!(
        weights[3], 0.0,
        "weight 0.005 should be zeroed (|w| < 0.05)"
    );
}

/// Verify that `PruningReport.sparsity` matches the expected fraction of zeroed weights.
///
/// Provides 4 weights where exactly 2 are below the threshold, expects 50% sparsity.
#[test]
fn test_prune_sparsity_correct() {
    let mut weights = vec![0.5_f32, 0.01, 0.8, 0.02];
    let report = prune_by_magnitude(&mut weights, 0.05);

    assert_eq!(report.total_params, 4);
    assert_eq!(report.pruned_params, 2);
    assert!(
        (report.sparsity - 0.5).abs() < 1e-6,
        "Expected sparsity 0.5, got {}",
        report.sparsity
    );
}

/// Verify that `compute_sparsity` returns the correct fraction for a known input.
///
/// A 5-element vector with 3 zeros should yield sparsity of 0.6.
#[test]
fn test_compute_sparsity() {
    let weights = vec![0.0_f32, 1.0, 0.0, -0.5, 0.0];
    let sparsity = compute_sparsity(&weights);
    assert!(
        (sparsity - 0.6).abs() < 1e-6,
        "Expected sparsity 0.6, got {}",
        sparsity
    );

    let all_zeros = vec![0.0_f32; 4];
    assert_eq!(
        compute_sparsity(&all_zeros),
        1.0,
        "All-zero slice should have sparsity 1.0"
    );

    let no_zeros = vec![1.0_f32, 2.0, 3.0];
    assert_eq!(
        compute_sparsity(&no_zeros),
        0.0,
        "No-zero slice should have sparsity 0.0"
    );

    let empty: Vec<f32> = vec![];
    assert_eq!(
        compute_sparsity(&empty),
        0.0,
        "Empty slice should have sparsity 0.0"
    );
}

/// Verify that a zero threshold prunes nothing (only strictly less-than comparison).
///
/// With threshold = 0.0, only weights with |w| < 0.0 would be pruned — which is
/// impossible — so no weights should be zeroed.
#[test]
fn test_prune_zero_threshold() {
    let original = vec![0.5_f32, -0.01, 0.3, 0.0, -0.8];
    let mut weights = original.clone();
    let report = prune_by_magnitude(&mut weights, 0.0);

    // With threshold = 0, nothing satisfies |w| < 0.0 (not even 0.0 itself)
    assert_eq!(
        report.pruned_params, 0,
        "Zero threshold should prune no weights (strict less-than)"
    );
    assert_eq!(
        report.sparsity, 0.0,
        "Zero threshold should yield 0.0 sparsity"
    );
    // Weights must be unchanged (including existing 0.0)
    for (i, (&w, &orig)) in weights.iter().zip(original.iter()).enumerate() {
        assert_eq!(w, orig, "weights[{}] changed unexpectedly", i);
    }
}

/// Verify that `suggest_threshold` produces a threshold that achieves approximately the
/// target sparsity when applied via `prune_by_magnitude`.
///
/// Uses a uniform weight distribution [0.1, 0.2, ..., 1.0] and targets 50% sparsity,
/// then confirms the resulting sparsity is within ±10% of the target.
#[test]
fn test_suggest_threshold_achieves_target() {
    let weights: Vec<f32> = (1..=10).map(|i| i as f32 * 0.1).collect();
    let target_sparsity = 0.5;
    let threshold = suggest_threshold(&weights, target_sparsity);

    let mut w = weights.clone();
    prune_by_magnitude(&mut w, threshold);
    let sparsity = compute_sparsity(&w);

    assert!(
        sparsity >= 0.4 && sparsity <= 0.6,
        "Expected ~{} sparsity after applying suggested threshold, got {}",
        target_sparsity,
        sparsity
    );
}

/// Verify that a pruned DenseLayer produces finite outputs from its forward pass.
///
/// Creates a Xavier-initialized layer, prunes it with a moderate threshold, then
/// runs a forward pass and checks that all outputs are finite.
#[test]
fn test_prune_dense_layer_produces_finite_outputs() {
    let mut rng = SimpleRng::new(42);
    let layer = DenseLayer::new(8, 4, &mut rng);
    let (pruned_layer, _report) = prune_dense_layer(&layer, 0.05);

    let input = vec![0.5_f32, -0.3, 0.8, -0.1, 0.2, -0.6, 0.4, -0.7];
    let mut output = vec![0.0_f32; 4];
    pruned_layer.forward(&input, &mut output, 1);

    assert!(
        output.iter().all(|&x| x.is_finite()),
        "All forward-pass outputs of pruned layer must be finite, got {:?}",
        output
    );
}

/// Verify that large-magnitude weights survive pruning intact.
///
/// Creates a layer with known large weights (all ≥ 1.0) and applies a small
/// threshold (0.05). All weights should remain non-zero after pruning.
#[test]
fn test_large_weights_survive_pruning() {
    // All weights are large enough to survive a threshold of 0.05
    let weights: Vec<f32> = (1..=6).map(|i| i as f32).collect(); // [1, 2, 3, 4, 5, 6]
    let biases = vec![0.0_f32, 0.0];
    let layer = DenseLayer::new_with_weights(3, 2, weights.clone(), biases);

    let (pruned_layer, report) = prune_dense_layer(&layer, 0.05);

    assert_eq!(
        report.pruned_params, 0,
        "No large weights should be pruned by a small threshold"
    );
    assert_eq!(
        report.sparsity, 0.0,
        "Sparsity should be 0.0 when no weights are pruned"
    );

    for (i, (&orig, &pruned)) in weights
        .iter()
        .zip(pruned_layer.weights().iter())
        .enumerate()
    {
        assert_eq!(
            pruned, orig,
            "weights[{}] should be unchanged: expected {}, got {}",
            i, orig, pruned
        );
    }
}

/// Verify that the PruningReport struct exposes the expected fields correctly.
///
/// Constructs a report directly and checks each field value, including Display output.
#[test]
fn test_pruning_report_fields() {
    let report = PruningReport {
        total_params: 100,
        pruned_params: 40,
        sparsity: 0.4,
        threshold: 0.01,
    };

    assert_eq!(report.total_params, 100);
    assert_eq!(report.pruned_params, 40);
    assert!((report.sparsity - 0.4).abs() < 1e-6);
    assert!((report.threshold - 0.01).abs() < 1e-8);

    let display = format!("{}", report);
    assert!(
        display.contains("sparsity"),
        "Display should mention sparsity"
    );
    assert!(
        display.contains("100"),
        "Display should include total_params"
    );
    assert!(
        display.contains("40"),
        "Display should include pruned_params"
    );
}
