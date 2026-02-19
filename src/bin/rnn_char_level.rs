use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use rust_neural_networks::layers::{Layer, LstmLayer};
use rust_neural_networks::step_debug::StepDebugger;
use rust_neural_networks::training::parse_step_flag;
use rust_neural_networks::utils::gradient_clipping::clip_gradient_norm;
use rust_neural_networks::utils::rng::SimpleRng;

// Character-level RNN for text generation (educational example).
// Demonstrates LSTM with BPTT, gradient clipping, and sequence modeling.

// Training hyperparameters
const HIDDEN_SIZE: usize = 128;
const SEQUENCE_LENGTH: usize = 50; // Length of sequences for BPTT
const LEARNING_RATE: f32 = 0.001;
const EPOCHS: usize = 100;
const BATCH_SIZE: usize = 1; // Process one sequence at a time
const GRADIENT_CLIP_NORM: f32 = 5.0; // Clip gradients to prevent exploding gradients
const SAMPLE_LENGTH: usize = 200; // Length of generated text samples
const SAMPLE_FREQUENCY: usize = 10; // Generate sample every N epochs

// Sample text for training (Shakespeare-like text)
const TRAINING_TEXT: &str = r#"To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience doth make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action."#;

/// Character vocabulary with mappings between characters and indices
struct CharVocab {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
    vocab_size: usize,
}

impl CharVocab {
    /// Create vocabulary from text
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

    /// Convert character to one-hot vector
    fn char_to_onehot(&self, ch: char) -> Vec<f32> {
        let mut onehot = vec![0.0; self.vocab_size];
        if let Some(&idx) = self.char_to_idx.get(&ch) {
            onehot[idx] = 1.0;
        }
        onehot
    }

    /// Convert index to character
    fn idx_to_char(&self, idx: usize) -> char {
        self.idx_to_char[idx.min(self.vocab_size - 1)]
    }

    /// Sample character from probability distribution
    fn sample_char(&self, probs: &[f32], rng: &mut SimpleRng) -> char {
        let r = rng.next_f32();
        let mut cumsum = 0.0;

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

/// Softmax activation for output layer
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum_exp).collect()
}

/// Cross-entropy loss
fn cross_entropy_loss(probs: &[f32], target_idx: usize) -> f32 {
    -probs[target_idx].ln()
}

/// Gradient of cross-entropy loss w.r.t. logits
fn cross_entropy_gradient(probs: &[f32], target_idx: usize) -> Vec<f32> {
    let mut grad = probs.to_vec();
    grad[target_idx] -= 1.0;
    grad
}

/// Generate text sample from trained model
fn generate_sample(
    lstm: &LstmLayer,
    vocab: &CharVocab,
    seed_text: &str,
    length: usize,
    rng: &mut SimpleRng,
) -> String {
    lstm.reset_state();

    let mut generated = seed_text.to_string();
    let seed_chars: Vec<char> = seed_text.chars().collect();

    // Prime the LSTM with seed text
    for &ch in &seed_chars {
        let input = vocab.char_to_onehot(ch);
        let mut output = vec![0.0; vocab.vocab_size];
        lstm.forward(&input, &mut output, 1);
    }

    // Generate new characters
    let mut current_char = *seed_chars.last().unwrap_or(&' ');
    for _ in 0..length {
        let input = vocab.char_to_onehot(current_char);
        let mut output = vec![0.0; vocab.vocab_size];
        lstm.forward(&input, &mut output, 1);

        let probs = softmax(&output);
        current_char = vocab.sample_char(&probs, rng);
        generated.push(current_char);
    }

    generated
}

fn main() {
    println!("=== Character-Level LSTM for Text Generation ===\n");

    // Parse command-line arguments for step mode
    let args: Vec<String> = env::args().collect();
    let step_debug_enabled = parse_step_flag(&args);
    let mut debugger = StepDebugger::new(step_debug_enabled);

    // Create vocabulary from training text
    let vocab = CharVocab::from_text(TRAINING_TEXT);
    println!("Vocabulary size: {}", vocab.vocab_size);
    println!("Training text length: {} characters", TRAINING_TEXT.len());
    println!("Sequence length: {}", SEQUENCE_LENGTH);
    println!("Hidden size: {}", HIDDEN_SIZE);
    println!("Learning rate: {}", LEARNING_RATE);
    println!("Gradient clipping norm: {}\n", GRADIENT_CLIP_NORM);

    // Initialize LSTM layer
    let mut rng = SimpleRng::new(42);
    let mut lstm = LstmLayer::new(vocab.vocab_size, HIDDEN_SIZE, vocab.vocab_size, &mut rng);

    println!("Model parameters: {}\n", lstm.parameter_count());

    // Training data: characters of the text
    let text_chars: Vec<char> = TRAINING_TEXT.chars().collect();
    let num_sequences = (text_chars.len() - 1) / SEQUENCE_LENGTH;

    println!("Starting training for {} epochs...\n", EPOCHS);

    // Create CSV log file
    let mut log_file =
        BufWriter::new(File::create("logs/rnn_char_level.csv").unwrap_or_else(|_| {
            std::fs::create_dir_all("logs").ok();
            File::create("logs/rnn_char_level.csv").expect("Failed to create log file")
        }));
    writeln!(log_file, "epoch,loss,time_ms").expect("Failed to write CSV header");

    // Training loop
    for epoch in 0..EPOCHS {
        debugger.on_epoch_start(epoch + 1);

        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        let mut num_chars = 0;

        // Process sequences
        for seq_idx in 0..num_sequences {
            let start_idx = seq_idx * SEQUENCE_LENGTH;
            let end_idx = (start_idx + SEQUENCE_LENGTH).min(text_chars.len() - 1);

            if end_idx <= start_idx {
                continue;
            }

            // Reset hidden state at start of each sequence
            lstm.reset_state();

            // Forward pass through sequence
            let mut sequence_loss = 0.0;
            let mut outputs = Vec::new();
            let mut targets = Vec::new();

            for t in start_idx..end_idx {
                let current_char = text_chars[t];
                let next_char = text_chars[t + 1];

                let input = vocab.char_to_onehot(current_char);
                let mut output = vec![0.0; vocab.vocab_size];

                lstm.forward(&input, &mut output, BATCH_SIZE);

                let probs = softmax(&output);
                let target_idx = vocab.char_to_idx[&next_char];
                let loss = cross_entropy_loss(&probs, target_idx);

                sequence_loss += loss;
                num_chars += 1;

                outputs.push(output);
                targets.push(target_idx);
            }

            total_loss += sequence_loss;

            // Backward pass through sequence (BPTT)
            // Collect output gradients in forward order
            let seq_len = outputs.len();
            let mut all_grad_outputs: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| {
                    let probs = softmax(&outputs[t]);
                    cross_entropy_gradient(&probs, targets[t])
                })
                .collect();

            // Flatten all gradients for clipping
            let mut flat_grads: Vec<f32> = all_grad_outputs.iter().flatten().copied().collect();

            // Clip gradients by L2 norm to prevent exploding gradients
            let grad_norm = clip_gradient_norm(&mut flat_grads, GRADIENT_CLIP_NORM);

            // Reshape clipped gradients back
            for (t, grad_chunk) in flat_grads.chunks(vocab.vocab_size).enumerate() {
                all_grad_outputs[t].copy_from_slice(grad_chunk);
            }

            // Apply BPTT backward pass in reverse order with dh/dc state propagation
            let mut dh_next = vec![0.0f32; HIDDEN_SIZE];
            let mut dc_next = vec![0.0f32; HIDDEN_SIZE];

            for t in (0..seq_len).rev() {
                let current_char = text_chars[start_idx + t];
                let input = vocab.char_to_onehot(current_char);
                let mut grad_input = vec![0.0; vocab.vocab_size];

                let (new_dh, new_dc) = lstm.backward_bptt(
                    &input,
                    &all_grad_outputs[t],
                    &mut grad_input,
                    &dh_next,
                    &dc_next,
                    BATCH_SIZE,
                );
                dh_next = new_dh;
                dc_next = new_dc;
            }

            // Track maximum gradient norm for monitoring
            let _ = grad_norm;

            // Update parameters after BPTT
            lstm.update_parameters(LEARNING_RATE);
        }

        let avg_loss = total_loss / num_chars as f32;
        let epoch_time = epoch_start.elapsed().as_millis();

        println!(
            "Epoch {}/{}, Loss: {:.4}, Time: {}ms",
            epoch + 1,
            EPOCHS,
            avg_loss,
            epoch_time
        );

        // Log to CSV
        writeln!(log_file, "{},{:.6},{}", epoch + 1, avg_loss, epoch_time)
            .expect("Failed to write to log");

        // Generate sample text periodically
        if (epoch + 1) % SAMPLE_FREQUENCY == 0 || epoch == EPOCHS - 1 {
            println!("\n--- Generated Sample (seed: 'To be') ---");
            let sample = generate_sample(&lstm, &vocab, "To be", SAMPLE_LENGTH, &mut rng);
            println!("{}", sample);
            println!("--- End Sample ---\n");
        }
    }

    println!("\nTraining complete! Logs saved to logs/rnn_char_level.csv");

    // Generate final samples
    println!("\n=== Final Text Generation ===\n");

    let seeds = vec!["To be", "For who", "And by"];
    for seed in seeds {
        println!("--- Seed: '{}' ---", seed);
        let sample = generate_sample(&lstm, &vocab, seed, SAMPLE_LENGTH, &mut rng);
        println!("{}\n", sample);
    }
}
