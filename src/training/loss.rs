const EPSILON: f32 = 1e-9;

fn validate_cross_entropy_inputs(
    function_name: &str,
    probs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
) -> usize {
    assert!(cols > 0, "{}: cols must be greater than 0", function_name);
    assert!(
        labels.len() >= rows,
        "{}: labels length {} is less than rows {}",
        function_name,
        labels.len(),
        rows
    );

    let required_len = rows
        .checked_mul(cols)
        .unwrap_or_else(|| panic!("{}: rows * cols overflowed", function_name));
    assert!(
        probs.len() >= required_len,
        "{}: probs length {} is less than rows * cols {}",
        function_name,
        probs.len(),
        required_len
    );

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        assert!(
            (label as usize) < cols,
            "{}: label at row {} ({}) must be less than cols {}",
            function_name,
            row_idx,
            label,
            cols
        );
    }

    required_len
}

fn sanitize_probability(probability: f32) -> f32 {
    if probability.is_finite() {
        probability.clamp(EPSILON, 1.0)
    } else {
        EPSILON
    }
}

/// Computes the cross-entropy loss and writes the softmax gradient into `delta`.
///
/// For each sample `i` in `0..rows`, this calls `sanitize_probability` on the relevant entries in
/// `probs`: it subtracts `ln(sanitize_probability(probs[i * cols + labels[i]]))` from the returned
/// loss and writes `(sanitize_probability(probs[i * cols + j]) - 1.0{j == label}) * scale` into
/// `delta[i * cols + j]` for every class `j`.
///
/// # Parameters
///
/// - `probs` — flattened row-major softmax probabilities; length must be at least `rows * cols`.
/// - `labels` — true class labels; only the first `rows` entries are used and each label must be \< `cols`.
/// - `rows` — number of samples to process.
/// - `cols` — number of classes per sample (must be > 0).
/// - `delta` — mutable output buffer for gradients; length must be at least `rows * cols`.
/// - `scale` — multiplicative factor applied to every gradient value (e.g., `1.0` or `1.0 / batch_size`).
///
/// # Returns
///
/// The total (summed, not averaged) cross-entropy loss over the `rows` samples.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::compute_softmax_cross_entropy;
///
/// let probs = [0.9f32, 0.1f32]; // one sample, two-class softmax
/// let labels = [0u8];
/// let mut delta = [0.0f32; 2];
/// let loss = compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);
///
/// let expected_loss = -(0.9f32).ln();
/// assert!((loss - expected_loss).abs() < 1e-6);
/// // gradient: true class probability minus 1, others unchanged
/// assert!((delta[0] - (-0.1f32)).abs() < 1e-6);
/// assert!((delta[1] - 0.1f32).abs() < 1e-6);
/// ```
pub fn compute_softmax_cross_entropy(
    probs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
    scale: f32,
) -> f32 {
    let required_len =
        validate_cross_entropy_inputs("compute_softmax_cross_entropy", probs, labels, rows, cols);
    assert!(
        delta.len() >= required_len,
        "compute_softmax_cross_entropy: delta length {} is less than rows * cols {}",
        delta.len(),
        required_len
    );

    let mut total_loss = 0.0f32;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;

        let row = &probs[row_start..row_start + cols];
        let delta_row = &mut delta[row_start..row_start + cols];
        let mut label_prob = EPSILON;
        for (j, &p) in row.iter().enumerate() {
            let mut g = sanitize_probability(p);
            if j == label {
                label_prob = g;
                g -= 1.0;
            }
            delta_row[j] = g * scale;
        }

        total_loss -= label_prob.ln();
    }

    total_loss
}

/// Computes the cross-entropy loss and the number of correct predictions for a
/// batch of softmax probabilities.
///
/// This function consolidates the inline validation loops that appear identically
/// in every binary.  The caller is responsible for applying softmax to `probs`
/// before calling this function.
///
/// # Parameters
///
/// - `probs`  – flattened row-major softmax probabilities, length ≥ `rows * cols`.
/// - `labels` – true class labels corresponding to each row.
/// - `rows`   – number of samples (rows) in the batch.
/// - `cols`   – number of classes per sample.
///
/// # Returns
///
/// `(total_loss, correct_count)` where:
/// - `total_loss` is the summed cross-entropy loss (not yet divided by `rows`).
/// - `correct_count` is the number of samples whose argmax equals the true label.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::evaluate_batch_accuracy;
///
/// // One sample, two classes, correct prediction.
/// let probs = [0.9f32, 0.1f32];
/// let labels = [0u8];
/// let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
/// assert!(loss > 0.0);
/// assert_eq!(correct, 1);
/// ```
pub fn evaluate_batch_accuracy(
    probs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
) -> (f32, usize) {
    validate_cross_entropy_inputs("evaluate_batch_accuracy", probs, labels, rows, cols);

    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;

        let row = &probs[row_start..row_start + cols];
        let mut predicted = 0usize;
        let mut max_prob = EPSILON;
        let mut label_prob = EPSILON;
        for (j, &value) in row.iter().enumerate() {
            let value = sanitize_probability(value);
            if j == label {
                label_prob = value;
            }
            if value > max_prob {
                max_prob = value;
                predicted = j;
            }
        }
        total_loss -= label_prob.ln();
        if predicted == label {
            correct += 1;
        }
    }

    (total_loss, correct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_softmax_cross_entropy_basic() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta = [0.0f32; 2];
        let loss = compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);

        let expected_loss = -(0.9f32).ln();
        assert!((loss - expected_loss).abs() < 1e-6, "loss mismatch");
        assert!(
            (delta[0] - (-0.1f32)).abs() < 1e-6,
            "gradient at true class"
        );
        assert!((delta[1] - 0.1f32).abs() < 1e-6, "gradient at other class");
    }

    #[test]
    fn test_compute_softmax_cross_entropy_scale() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta_scaled = [0.0f32; 2];
        let mut delta_unscaled = [0.0f32; 2];

        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta_unscaled, 1.0);
        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta_scaled, 0.5);

        assert!((delta_scaled[0] - delta_unscaled[0] * 0.5).abs() < 1e-7);
        assert!((delta_scaled[1] - delta_unscaled[1] * 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_compute_softmax_cross_entropy_multibatch() {
        let probs = [0.1f32, 0.2, 0.7, 0.3, 0.4, 0.3];
        let labels = [2u8, 1u8];
        let mut delta = [0.0f32; 6];
        let loss = compute_softmax_cross_entropy(&probs, &labels, 2, 3, &mut delta, 1.0);

        assert!(loss > 0.0);
        assert!((delta[2] - (0.7 - 1.0)).abs() < 1e-6, "delta[2] wrong");
        assert!((delta[4] - (0.4 - 1.0)).abs() < 1e-6, "delta[4] wrong");
    }

    #[test]
    fn test_compute_softmax_cross_entropy_sanitizes_invalid_probabilities() {
        let probs = [f32::NAN, f32::INFINITY, -0.5, 1.5];
        let labels = [0u8];
        let mut delta = [0.0f32; 4];

        let loss = compute_softmax_cross_entropy(&probs, &labels, 1, 4, &mut delta, 1.0);

        assert!(loss.is_finite());
        assert!(delta.iter().all(|value| value.is_finite()));
        assert!((delta[0] - (EPSILON - 1.0)).abs() < 1e-6);
        assert!((delta[1] - EPSILON).abs() < 1e-12);
        assert!((delta[2] - EPSILON).abs() < 1e-12);
        assert!((delta[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "compute_softmax_cross_entropy: cols must be greater than 0")]
    fn test_compute_softmax_cross_entropy_rejects_zero_cols() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta = [0.0f32; 2];

        compute_softmax_cross_entropy(&probs, &labels, 1, 0, &mut delta, 1.0);
    }

    #[test]
    #[should_panic(expected = "compute_softmax_cross_entropy: labels length 0 is less than rows 1")]
    fn test_compute_softmax_cross_entropy_rejects_short_labels() {
        let probs = [0.9f32, 0.1f32];
        let labels = [];
        let mut delta = [0.0f32; 2];

        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);
    }

    #[test]
    #[should_panic(
        expected = "compute_softmax_cross_entropy: delta length 1 is less than rows * cols 2"
    )]
    fn test_compute_softmax_cross_entropy_rejects_short_delta() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let mut delta = [0.0f32; 1];

        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);
    }

    #[test]
    #[should_panic(
        expected = "compute_softmax_cross_entropy: label at row 0 (2) must be less than cols 2"
    )]
    fn test_compute_softmax_cross_entropy_rejects_invalid_label() {
        let probs = [0.9f32, 0.1f32];
        let labels = [2u8];
        let mut delta = [0.0f32; 2];

        compute_softmax_cross_entropy(&probs, &labels, 1, 2, &mut delta, 1.0);
    }

    #[test]
    fn test_evaluate_batch_accuracy_correct() {
        let probs = [0.9f32, 0.1f32];
        let labels = [0u8];
        let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
        assert!(loss > 0.0);
        assert_eq!(correct, 1);
    }

    #[test]
    fn test_evaluate_batch_accuracy_wrong() {
        let probs = [0.1f32, 0.9f32];
        let labels = [0u8];
        let (_loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 2);
        assert_eq!(correct, 0);
    }

    #[test]
    fn test_evaluate_batch_accuracy_multibatch() {
        let probs = [0.1f32, 0.2, 0.7, 0.5, 0.3, 0.2];
        let labels = [2u8, 1u8];
        let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 2, 3);
        assert!(loss > 0.0);
        assert_eq!(correct, 1);
    }

    #[test]
    fn test_evaluate_batch_accuracy_sanitizes_invalid_probabilities() {
        let probs = [f32::NAN, f32::INFINITY, -0.5, 1.5];
        let labels = [0u8];

        let (loss, correct) = evaluate_batch_accuracy(&probs, &labels, 1, 4);

        assert!(loss.is_finite());
        assert_eq!(correct, 0);
    }

    #[test]
    #[should_panic(expected = "evaluate_batch_accuracy: probs length 1 is less than rows * cols 2")]
    fn test_evaluate_batch_accuracy_rejects_short_probs() {
        let probs = [0.9f32];
        let labels = [0u8];

        evaluate_batch_accuracy(&probs, &labels, 1, 2);
    }

    #[test]
    #[should_panic(
        expected = "evaluate_batch_accuracy: label at row 0 (2) must be less than cols 2"
    )]
    fn test_evaluate_batch_accuracy_rejects_invalid_label() {
        let probs = [0.9f32, 0.1f32];
        let labels = [2u8];

        evaluate_batch_accuracy(&probs, &labels, 1, 2);
    }
}
