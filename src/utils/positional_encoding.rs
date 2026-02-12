//! Positional encoding utilities for Transformer models
//!
//! This module provides functions for generating positional encodings,
//! which are critical for attention mechanisms to understand token positions.

/// Generates sinusoidal positional encodings for a sequence of tokens.
///
/// Implements the Transformer-style positional encoding from "Attention is All You Need":
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
///
/// where:
/// - `pos` = token position in sequence (0..seq_len-1)
/// - `i` = dimension index (0..d_model-1)
/// - Even dimensions use sin, odd dimensions use cos
///
/// This creates unique, deterministic positional patterns that:
/// - Provide structured positional information with smooth gradients
/// - Allow the model to easily learn relative positions
/// - Enable attention mechanisms to focus on spatially relevant tokens
///
/// # Arguments
///
/// * `seq_len` - Number of tokens in the sequence
/// * `d_model` - Dimension of each token embedding
///
/// # Returns
///
/// A vector of length `seq_len * d_model` containing positional encodings
/// in row-major order (all dimensions for token 0, then all dimensions for token 1, etc.)
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::sinusoidal_positional_encoding;
///
/// // Generate positional encodings for 49 tokens with 64 dimensions each
/// let encodings = sinusoidal_positional_encoding(49, 64);
/// assert_eq!(encodings.len(), 49 * 64);
///
/// // Verify the encoding is deterministic
/// let encodings2 = sinusoidal_positional_encoding(49, 64);
/// assert_eq!(encodings, encodings2);
/// ```
///
/// # Performance Note
///
/// This function allocates a new vector and computes trigonometric functions
/// for each position-dimension pair. For repeated use with the same dimensions,
/// consider caching the result.
pub fn sinusoidal_positional_encoding(seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut pos_encoding = vec![0.0f32; seq_len * d_model];

    for t in 0..seq_len {
        let pos_base = t * d_model;
        for d in 0..d_model {
            // Wavelength increases exponentially with dimension index
            // Using (2 * (d / 2)) to pair dimensions: 0,1 → 0; 2,3 → 2; 4,5 → 4, etc.
            let angle = (t as f32) / 10000.0f32.powf((2 * (d / 2)) as f32 / d_model as f32);

            if d % 2 == 0 {
                // Even dimensions: sine
                pos_encoding[pos_base + d] = angle.sin();
            } else {
                // Odd dimensions: cosine
                pos_encoding[pos_base + d] = angle.cos();
            }
        }
    }

    pos_encoding
}

/// Adds positional encodings to token embeddings in-place.
///
/// This is a convenience function that generates sinusoidal positional encodings
/// and adds them element-wise to the provided token embeddings.
///
/// # Arguments
///
/// * `tokens` - Mutable slice of token embeddings (length = batch_size * seq_len * d_model)
/// * `batch_size` - Number of sequences in the batch
/// * `seq_len` - Number of tokens per sequence
/// * `d_model` - Dimension of each token embedding
///
/// # Panics
///
/// Panics if `tokens.len() != batch_size * seq_len * d_model`
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::add_positional_encoding;
///
/// let batch_size = 2;
/// let seq_len = 10;
/// let d_model = 16;
///
/// // Create token embeddings (normally from a projection layer)
/// let mut tokens = vec![0.0f32; batch_size * seq_len * d_model];
///
/// // Add positional encodings
/// add_positional_encoding(&mut tokens, batch_size, seq_len, d_model);
///
/// // Tokens now contain positional information
/// ```
pub fn add_positional_encoding(
    tokens: &mut [f32],
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
) {
    assert_eq!(
        tokens.len(),
        batch_size * seq_len * d_model,
        "Token buffer size mismatch: expected {} elements, got {}",
        batch_size * seq_len * d_model,
        tokens.len()
    );

    // Generate positional encoding once (same for all batch elements)
    let pos_encoding = sinusoidal_positional_encoding(seq_len, d_model);

    // Add to each sequence in the batch
    for b in 0..batch_size {
        for t in 0..seq_len {
            let token_base = (b * seq_len + t) * d_model;
            let pos_base = t * d_model;

            for d in 0..d_model {
                tokens[token_base + d] += pos_encoding[pos_base + d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let seq_len = 10;
        let d_model = 16;
        let encoding = sinusoidal_positional_encoding(seq_len, d_model);

        assert_eq!(encoding.len(), seq_len * d_model);
    }

    #[test]
    fn test_sinusoidal_encoding_deterministic() {
        let seq_len = 5;
        let d_model = 8;

        let encoding1 = sinusoidal_positional_encoding(seq_len, d_model);
        let encoding2 = sinusoidal_positional_encoding(seq_len, d_model);

        assert_eq!(encoding1, encoding2);
    }

    #[test]
    fn test_sinusoidal_encoding_uniqueness() {
        let seq_len = 5;
        let d_model = 8;
        let encoding = sinusoidal_positional_encoding(seq_len, d_model);

        // Check that different positions have different encodings
        let pos0 = &encoding[0..d_model];
        let pos1 = &encoding[d_model..2 * d_model];
        let pos2 = &encoding[2 * d_model..3 * d_model];

        assert_ne!(pos0, pos1);
        assert_ne!(pos1, pos2);
        assert_ne!(pos0, pos2);
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        // Test with simple dimensions to verify formula
        let seq_len = 2;
        let d_model = 4;
        let encoding = sinusoidal_positional_encoding(seq_len, d_model);

        // Position 0 should have predictable values
        // For pos=0: all angles are 0, so sin(0)=0 and cos(0)=1
        assert_eq!(encoding[0], 0.0); // sin(0 / 10000^0) = sin(0) = 0
        assert_eq!(encoding[1], 1.0); // cos(0 / 10000^0) = cos(0) = 1
        assert_eq!(encoding[2], 0.0); // sin(0 / 10000^(2/4)) = sin(0) = 0
        assert_eq!(encoding[3], 1.0); // cos(0 / 10000^(2/4)) = cos(0) = 1

        // Position 1 should have non-zero values
        let pos1_start = d_model;
        assert_ne!(encoding[pos1_start], 0.0);
        assert_ne!(encoding[pos1_start + 1], 1.0);
    }

    #[test]
    fn test_sinusoidal_encoding_range() {
        let seq_len = 50;
        let d_model = 64;
        let encoding = sinusoidal_positional_encoding(seq_len, d_model);

        // All values should be in range [-1, 1] (sin/cos range)
        for &val in &encoding {
            assert!(val >= -1.0 && val <= 1.0, "Value {} out of range", val);
        }
    }

    #[test]
    fn test_add_positional_encoding_single_batch() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 4;

        let mut tokens = vec![1.0f32; batch_size * seq_len * d_model];
        let original_tokens = tokens.clone();

        add_positional_encoding(&mut tokens, batch_size, seq_len, d_model);

        // Verify tokens were modified (positional encoding was added)
        assert_ne!(tokens, original_tokens);

        // Verify values changed by the expected amount
        let pos_enc = sinusoidal_positional_encoding(seq_len, d_model);
        for i in 0..tokens.len() {
            assert_eq!(tokens[i], original_tokens[i] + pos_enc[i]);
        }
    }

    #[test]
    fn test_add_positional_encoding_multiple_batches() {
        let batch_size = 3;
        let seq_len = 5;
        let d_model = 8;

        let mut tokens = vec![0.5f32; batch_size * seq_len * d_model];
        add_positional_encoding(&mut tokens, batch_size, seq_len, d_model);

        // Verify same positional encoding is added to each batch
        let pos_enc = sinusoidal_positional_encoding(seq_len, d_model);

        for b in 0..batch_size {
            for t in 0..seq_len {
                let token_base = (b * seq_len + t) * d_model;
                let pos_base = t * d_model;

                for d in 0..d_model {
                    let expected = 0.5 + pos_enc[pos_base + d];
                    assert!((tokens[token_base + d] - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Token buffer size mismatch")]
    fn test_add_positional_encoding_wrong_size() {
        let batch_size = 2;
        let seq_len = 5;
        let d_model = 8;

        // Wrong size buffer
        let mut tokens = vec![0.0f32; 10];
        add_positional_encoding(&mut tokens, batch_size, seq_len, d_model);
    }

    #[test]
    fn test_positional_encoding_matches_reference() {
        // Test against the implementation from mnist_attention_pool.rs
        // for a 7×7 grid of patches with D_MODEL=64
        let seq_len = 49;
        let d_model = 64;

        let encoding = sinusoidal_positional_encoding(seq_len, d_model);

        // Verify basic properties that match the reference implementation:
        // 1. Correct size
        assert_eq!(encoding.len(), seq_len * d_model);

        // 2. Position 0 should have alternating 0 and 1 (sin(0)=0, cos(0)=1)
        assert_eq!(encoding[0], 0.0);
        assert_eq!(encoding[1], 1.0);

        // 3. All values in valid range
        for &val in &encoding {
            assert!(val >= -1.0 && val <= 1.0);
        }

        // 4. Different positions should be different
        let pos0 = &encoding[0..d_model];
        let pos48 = &encoding[48 * d_model..49 * d_model];
        assert_ne!(pos0, pos48);
    }
}
