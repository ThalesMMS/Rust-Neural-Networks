// Integration tests for the character-level LSTM model.
// Tests vocabulary building, training loop, and text generation on a small repeating pattern.
// Following patterns from test_rnn.rs and test_lstm.rs.

use rust_neural_networks::layers::{Layer, LstmLayer};
use rust_neural_networks::utils::gradient_clipping::clip_gradient_norm;
use rust_neural_networks::utils::rng::SimpleRng;
use std::collections::HashMap;

// ============================================================================
// Local Character Vocabulary (mirrors rnn_char_level.rs binary logic)
// ============================================================================

/// Character vocabulary with mappings between characters and indices.
struct CharVocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
    vocab_size: usize,
}

impl CharVocab {
    /// Create vocabulary from text: sort unique chars, assign indices.
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();

        let mut char_to_idx = HashMap::new();
        for (idx, &ch) in chars.iter().enumerate() {
            char_to_idx.insert(ch, idx);
        }

        let vocab_size = chars.len();

        Self {
            char_to_idx,
            idx_to_char: chars,
            vocab_size,
        }
    }

    /// Convert character to one-hot vector of length vocab_size.
    fn char_to_onehot(&self, ch: char) -> Vec<f32> {
        let mut onehot = vec![0.0f32; self.vocab_size];
        if let Some(&idx) = self.char_to_idx.get(&ch) {
            onehot[idx] = 1.0;
        }
        onehot
    }

    /// Convert index back to character.
    fn idx_to_char(&self, idx: usize) -> char {
        self.idx_to_char[idx.min(self.vocab_size - 1)]
    }

    /// Sample character from probability distribution using the RNG.
    fn sample_char(&self, probs: &[f32], rng: &mut SimpleRng) -> char {
        let r = rng.next_f32();
        let mut cumsum = 0.0f32;

        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return self.idx_to_char(idx);
            }
        }

        // Fallback to last character
        self.idx_to_char(self.vocab_size - 1)
    }
}

// ============================================================================
// Loss / Gradient Helpers
// ============================================================================

/// Softmax activation: stable implementation using max subtraction.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum_exp).collect()
}

/// Cross-entropy loss: -log(p[target_idx])
fn cross_entropy_loss(probs: &[f32], target_idx: usize) -> f32 {
    -probs[target_idx].max(1e-10).ln()
}

/// Gradient of cross-entropy loss w.r.t. logits (softmax + CE gradient = probs - one_hot).
fn cross_entropy_gradient(probs: &[f32], target_idx: usize) -> Vec<f32> {
    let mut grad = probs.to_vec();
    grad[target_idx] -= 1.0;
    grad
}

/// Generate text from a trained LSTM: prime with seed, then sample new characters.
fn generate_text(
    lstm: &LstmLayer,
    vocab: &CharVocab,
    seed: &str,
    length: usize,
    rng: &mut SimpleRng,
) -> String {
    lstm.reset_state();

    let mut generated = seed.to_string();
    let seed_chars: Vec<char> = seed.chars().collect();

    // Prime the LSTM with the seed text
    for &ch in &seed_chars {
        let input = vocab.char_to_onehot(ch);
        let mut output = vec![0.0f32; vocab.vocab_size];
        lstm.forward(&input, &mut output, 1);
    }

    // Generate new characters autoregressively
    let mut current_char = *seed_chars.last().unwrap_or(&' ');
    for _ in 0..length {
        let input = vocab.char_to_onehot(current_char);
        let mut output = vec![0.0f32; vocab.vocab_size];
        lstm.forward(&input, &mut output, 1);

        let probs = softmax(&output);
        current_char = vocab.sample_char(&probs, rng);
        generated.push(current_char);
    }

    generated
}

// ============================================================================
// Test: Vocabulary Building
// ============================================================================

#[test]
fn test_char_vocab_from_text() {
    let text = "abcabc";
    let vocab = CharVocab::from_text(text);

    // Should have exactly 3 unique characters
    assert_eq!(
        vocab.vocab_size, 3,
        "Vocabulary should contain 3 unique chars"
    );

    // Characters should be sorted: a, b, c
    assert_eq!(vocab.idx_to_char[0], 'a');
    assert_eq!(vocab.idx_to_char[1], 'b');
    assert_eq!(vocab.idx_to_char[2], 'c');

    // Mappings should be consistent
    assert_eq!(vocab.char_to_idx[&'a'], 0);
    assert_eq!(vocab.char_to_idx[&'b'], 1);
    assert_eq!(vocab.char_to_idx[&'c'], 2);
}

#[test]
fn test_char_vocab_deduplication() {
    // Repeated characters should produce the correct unique count
    let text = "aaabbbccc";
    let vocab = CharVocab::from_text(text);
    assert_eq!(
        vocab.vocab_size, 3,
        "Duplicate chars should be deduplicated"
    );
}

#[test]
fn test_char_vocab_single_char() {
    let text = "aaaa";
    let vocab = CharVocab::from_text(text);
    assert_eq!(
        vocab.vocab_size, 1,
        "All same chars should produce vocab of size 1"
    );
    assert_eq!(vocab.idx_to_char[0], 'a');
}

#[test]
fn test_char_vocab_onehot_encoding() {
    let text = "abc";
    let vocab = CharVocab::from_text(text);

    let onehot_a = vocab.char_to_onehot('a');
    assert_eq!(
        onehot_a.len(),
        3,
        "One-hot vector should have length equal to vocab_size"
    );
    assert_eq!(
        onehot_a,
        vec![1.0, 0.0, 0.0],
        "One-hot for 'a' should be [1,0,0]"
    );

    let onehot_b = vocab.char_to_onehot('b');
    assert_eq!(
        onehot_b,
        vec![0.0, 1.0, 0.0],
        "One-hot for 'b' should be [0,1,0]"
    );

    let onehot_c = vocab.char_to_onehot('c');
    assert_eq!(
        onehot_c,
        vec![0.0, 0.0, 1.0],
        "One-hot for 'c' should be [0,0,1]"
    );
}

#[test]
fn test_char_vocab_onehot_sums_to_one() {
    let text = "abcde";
    let vocab = CharVocab::from_text(text);

    for &ch in &['a', 'b', 'c', 'd', 'e'] {
        let onehot = vocab.char_to_onehot(ch);
        let sum: f32 = onehot.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "One-hot vector for '{}' should sum to 1.0, got {}",
            ch,
            sum
        );

        // Exactly one element should be 1.0
        let ones = onehot.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(
            ones, 1,
            "One-hot vector for '{}' should have exactly one 1.0",
            ch
        );
    }
}

#[test]
fn test_char_vocab_unknown_char_ignored() {
    let text = "abc";
    let vocab = CharVocab::from_text(text);

    // Unknown character 'z' should produce an all-zero vector
    let onehot_unknown = vocab.char_to_onehot('z');
    assert!(
        onehot_unknown.iter().all(|&x| x == 0.0),
        "Unknown character should produce all-zero one-hot vector"
    );
}

// ============================================================================
// Test: Softmax Helper
// ============================================================================

#[test]
fn test_softmax_sums_to_one() {
    let logits = vec![1.0f32, 2.0, 3.0, 0.5];
    let probs = softmax(&logits);

    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax output should sum to 1.0, got {}",
        sum
    );
}

#[test]
fn test_softmax_all_positive() {
    let logits = vec![0.5f32, -0.3, 1.2, -1.0];
    let probs = softmax(&logits);

    for &p in &probs {
        assert!(p > 0.0, "All softmax probabilities should be positive");
        assert!(p <= 1.0, "All softmax probabilities should be <= 1.0");
    }
}

#[test]
fn test_softmax_argmax_preserved() {
    // The highest logit should produce the highest probability
    let logits = vec![1.0f32, 5.0, 2.0, 3.0];
    let probs = softmax(&logits);

    let max_idx = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(max_idx, 1, "Argmax should match index of largest logit");
}

// ============================================================================
// Test: Training Loop on Small Repeating Pattern
// ============================================================================

#[test]
fn test_char_level_training_reduces_loss() {
    // Use a small repeating pattern: "abcabcabc"
    let training_text = "abcabcabcabcabc";
    let vocab = CharVocab::from_text(training_text);
    let vocab_size = vocab.vocab_size;

    // Small model for fast testing
    let hidden_size = 16;
    let mut rng = SimpleRng::new(42);
    let mut lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let text_chars: Vec<char> = training_text.chars().collect();
    let sequence_length = 6;
    let learning_rate = 0.01f32;
    let gradient_clip_norm = 5.0f32;

    // Compute initial loss before any training
    let initial_loss = compute_epoch_loss(
        &lstm,
        &vocab,
        &text_chars,
        sequence_length,
        gradient_clip_norm,
    );

    // Train for several epochs
    let epochs = 30;
    for _ in 0..epochs {
        train_one_epoch(
            &mut lstm,
            &vocab,
            &text_chars,
            sequence_length,
            learning_rate,
            gradient_clip_norm,
        );
    }

    // Compute final loss after training
    let final_loss = compute_epoch_loss(
        &lstm,
        &vocab,
        &text_chars,
        sequence_length,
        gradient_clip_norm,
    );

    assert!(
        final_loss < initial_loss,
        "Training should reduce loss: initial={:.4}, final={:.4}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_char_level_training_all_outputs_finite() {
    let training_text = "abab";
    let vocab = CharVocab::from_text(training_text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 8;
    let mut rng = SimpleRng::new(99);
    let mut lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let text_chars: Vec<char> = training_text.chars().collect();
    let sequence_length = 3;
    let learning_rate = 0.005f32;
    let gradient_clip_norm = 5.0f32;

    // Run training for a few epochs and verify outputs stay finite
    for _ in 0..5 {
        lstm.reset_state();

        for t in 0..text_chars.len().saturating_sub(1) {
            let input = vocab.char_to_onehot(text_chars[t]);
            let mut output = vec![0.0f32; vocab_size];
            lstm.forward(&input, &mut output, 1);

            // Outputs must be finite throughout training
            assert!(
                output.iter().all(|&x| x.is_finite()),
                "LSTM output should be finite at time step {}",
                t
            );

            let probs = softmax(&output);
            assert!(
                probs.iter().all(|&x| x.is_finite() && x > 0.0),
                "Softmax probabilities should be finite and positive"
            );
        }

        train_one_epoch(
            &mut lstm,
            &vocab,
            &text_chars,
            sequence_length,
            learning_rate,
            gradient_clip_norm,
        );
    }
}

// ============================================================================
// Test: Text Generation
// ============================================================================

#[test]
fn test_char_level_generation_produces_valid_chars() {
    let training_text = "abcabcabcabc";
    let vocab = CharVocab::from_text(training_text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 16;
    let mut rng = SimpleRng::new(42);
    let mut lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    // Train briefly so the model has learned something
    let text_chars: Vec<char> = training_text.chars().collect();
    let sequence_length = 6;
    let learning_rate = 0.01f32;
    let gradient_clip_norm = 5.0f32;

    for _ in 0..20 {
        train_one_epoch(
            &mut lstm,
            &vocab,
            &text_chars,
            sequence_length,
            learning_rate,
            gradient_clip_norm,
        );
    }

    // Generate text and verify every character is in the vocabulary
    let mut sample_rng = SimpleRng::new(7);
    let generated = generate_text(&lstm, &vocab, "a", 20, &mut sample_rng);

    assert!(!generated.is_empty(), "Generated text should not be empty");

    for ch in generated.chars() {
        assert!(
            vocab.char_to_idx.contains_key(&ch),
            "Generated character '{}' should be in vocabulary",
            ch
        );
    }
}

#[test]
fn test_char_level_generation_length() {
    let text = "ab";
    let vocab = CharVocab::from_text(text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 8;
    let mut rng = SimpleRng::new(1);
    let lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let mut sample_rng = SimpleRng::new(2);
    let seed = "a";
    let generate_length = 10;
    let generated = generate_text(&lstm, &vocab, seed, generate_length, &mut sample_rng);

    // Generated text = seed + generate_length characters
    let expected_len = seed.len() + generate_length;
    assert_eq!(
        generated.len(),
        expected_len,
        "Generated text should have length seed_len + generate_length"
    );
}

#[test]
fn test_char_level_generation_starts_with_seed() {
    let text = "abc";
    let vocab = CharVocab::from_text(text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 8;
    let mut rng = SimpleRng::new(3);
    let lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let mut sample_rng = SimpleRng::new(4);
    let seed = "ab";
    let generated = generate_text(&lstm, &vocab, seed, 5, &mut sample_rng);

    assert!(
        generated.starts_with(seed),
        "Generated text should start with the seed text"
    );
}

// ============================================================================
// Test: BPTT with Gradient Clipping
// ============================================================================

#[test]
fn test_char_level_bptt_gradient_clipping() {
    let training_text = "xyzxyzxyz";
    let vocab = CharVocab::from_text(training_text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 8;
    let mut rng = SimpleRng::new(55);
    let lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let text_chars: Vec<char> = training_text.chars().collect();
    let seq_len = 4;
    let gradient_clip_norm_value = 5.0f32;

    // Run one BPTT backward pass and verify gradient clipping works
    lstm.reset_state();

    // Forward pass through sequence
    let mut outputs = Vec::new();
    let mut targets = Vec::new();

    for t in 0..seq_len.min(text_chars.len() - 1) {
        let input = vocab.char_to_onehot(text_chars[t]);
        let mut output = vec![0.0f32; vocab_size];
        lstm.forward(&input, &mut output, 1);
        let target_idx = vocab.char_to_idx[&text_chars[t + 1]];
        outputs.push(output);
        targets.push(target_idx);
    }

    // Collect output gradients
    let mut all_grad_outputs: Vec<Vec<f32>> = outputs
        .iter()
        .enumerate()
        .map(|(t, out)| {
            let probs = softmax(out);
            cross_entropy_gradient(&probs, targets[t])
        })
        .collect();

    // Flatten and clip gradients by norm
    let mut flat_grads: Vec<f32> = all_grad_outputs.iter().flatten().copied().collect();
    let _norm = clip_gradient_norm(&mut flat_grads, gradient_clip_norm_value);

    // Verify clipped gradient norm is at most clip threshold
    let clipped_norm: f32 = flat_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        clipped_norm <= gradient_clip_norm_value + 1e-5,
        "Clipped gradient norm {} should not exceed clip threshold {}",
        clipped_norm,
        gradient_clip_norm_value
    );

    // Reshape clipped gradients back
    for (t, grad_chunk) in flat_grads.chunks(vocab_size).enumerate() {
        all_grad_outputs[t].copy_from_slice(grad_chunk);
    }

    // BPTT backward pass in reverse order
    let mut dh_next = vec![0.0f32; hidden_size];
    let mut dc_next = vec![0.0f32; hidden_size];

    for t in (0..seq_len.min(text_chars.len() - 1)).rev() {
        let input = vocab.char_to_onehot(text_chars[t]);
        let mut grad_input = vec![0.0f32; vocab_size];

        let (new_dh, new_dc) = lstm.backward_bptt(
            &input,
            &all_grad_outputs[t],
            &mut grad_input,
            &dh_next,
            &dc_next,
            1,
        );

        // Verify returned gradients are finite
        assert!(
            new_dh.iter().all(|&x| x.is_finite()),
            "dh_prev at step {} should be finite",
            t
        );
        assert!(
            new_dc.iter().all(|&x| x.is_finite()),
            "dc_prev at step {} should be finite",
            t
        );

        dh_next = new_dh;
        dc_next = new_dc;
    }
}

#[test]
fn test_char_level_bptt_loss_decreases() {
    // Verify that BPTT + gradient clipping reduces the character-level loss.
    let training_text = "abcabcabc";
    let vocab = CharVocab::from_text(training_text);
    let vocab_size = vocab.vocab_size;

    let hidden_size = 16;
    let mut rng = SimpleRng::new(77);
    let mut lstm = LstmLayer::new(vocab_size, hidden_size, vocab_size, &mut rng);

    let text_chars: Vec<char> = training_text.chars().collect();
    let sequence_length = 6;
    let learning_rate = 0.005f32;
    let gradient_clip_norm_value = 5.0f32;

    // Measure initial loss
    let initial_loss = compute_epoch_loss(
        &lstm,
        &vocab,
        &text_chars,
        sequence_length,
        gradient_clip_norm_value,
    );

    // Train using BPTT for several epochs
    for _ in 0..40 {
        train_one_epoch(
            &mut lstm,
            &vocab,
            &text_chars,
            sequence_length,
            learning_rate,
            gradient_clip_norm_value,
        );
    }

    let final_loss = compute_epoch_loss(
        &lstm,
        &vocab,
        &text_chars,
        sequence_length,
        gradient_clip_norm_value,
    );

    assert!(
        final_loss < initial_loss,
        "BPTT training should reduce loss: initial={:.4}, final={:.4}",
        initial_loss,
        final_loss
    );
}

// ============================================================================
// Helper: Training and Evaluation Functions
// ============================================================================

/// Train the LSTM for one epoch using BPTT with gradient clipping.
fn train_one_epoch(
    lstm: &mut LstmLayer,
    vocab: &CharVocab,
    text_chars: &[char],
    sequence_length: usize,
    learning_rate: f32,
    gradient_clip_norm_value: f32,
) {
    let vocab_size = vocab.vocab_size;
    let num_sequences = (text_chars.len().saturating_sub(1)) / sequence_length;

    for seq_idx in 0..num_sequences {
        let start_idx = seq_idx * sequence_length;
        let end_idx = (start_idx + sequence_length).min(text_chars.len() - 1);

        if end_idx <= start_idx {
            continue;
        }

        lstm.reset_state();

        // Forward pass through sequence
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<usize> = Vec::new();

        for t in start_idx..end_idx {
            let input = vocab.char_to_onehot(text_chars[t]);
            let mut output = vec![0.0f32; vocab_size];
            lstm.forward(&input, &mut output, 1);

            let target_idx = vocab.char_to_idx[&text_chars[t + 1]];
            outputs.push(output);
            targets.push(target_idx);
        }

        let seq_len = outputs.len();

        // Collect output gradients
        let mut all_grad_outputs: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| {
                let probs = softmax(&outputs[t]);
                cross_entropy_gradient(&probs, targets[t])
            })
            .collect();

        // Flatten and clip by norm
        let mut flat_grads: Vec<f32> = all_grad_outputs.iter().flatten().copied().collect();
        clip_gradient_norm(&mut flat_grads, gradient_clip_norm_value);

        // Reshape clipped gradients back
        for (t, grad_chunk) in flat_grads.chunks(vocab_size).enumerate() {
            all_grad_outputs[t].copy_from_slice(grad_chunk);
        }

        // BPTT backward in reverse order
        let mut dh_next = vec![0.0f32; lstm.hidden_size()];
        let mut dc_next = vec![0.0f32; lstm.hidden_size()];

        for t in (0..seq_len).rev() {
            let input = vocab.char_to_onehot(text_chars[start_idx + t]);
            let mut grad_input = vec![0.0f32; vocab_size];

            let (new_dh, new_dc) = lstm.backward_bptt(
                &input,
                &all_grad_outputs[t],
                &mut grad_input,
                &dh_next,
                &dc_next,
                1,
            );

            dh_next = new_dh;
            dc_next = new_dc;
        }

        lstm.update_parameters(learning_rate);
    }
}

/// Compute average cross-entropy loss over all sequences (no parameter updates).
fn compute_epoch_loss(
    lstm: &LstmLayer,
    vocab: &CharVocab,
    text_chars: &[char],
    sequence_length: usize,
    _gradient_clip_norm_value: f32,
) -> f32 {
    let vocab_size = vocab.vocab_size;
    let num_sequences = (text_chars.len().saturating_sub(1)) / sequence_length;
    let mut total_loss = 0.0f32;
    let mut num_chars = 0usize;

    for seq_idx in 0..num_sequences {
        let start_idx = seq_idx * sequence_length;
        let end_idx = (start_idx + sequence_length).min(text_chars.len() - 1);

        if end_idx <= start_idx {
            continue;
        }

        lstm.reset_state();

        for t in start_idx..end_idx {
            let input = vocab.char_to_onehot(text_chars[t]);
            let mut output = vec![0.0f32; vocab_size];
            lstm.forward(&input, &mut output, 1);

            let probs = softmax(&output);
            let target_idx = vocab.char_to_idx[&text_chars[t + 1]];
            total_loss += cross_entropy_loss(&probs, target_idx);
            num_chars += 1;
        }
    }

    if num_chars == 0 {
        return f32::INFINITY;
    }
    total_loss / num_chars as f32
}
