use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rust_neural_networks::config::load_config;
use rust_neural_networks::utils::lr_scheduler::{
    CosineAnnealing, ExponentialDecay, LRScheduler, StepDecay,
};

// MLP with minibatches for MNIST (Rust port for study).
// Uses manual loops for layer operations (self-contained, educational).
const NUM_INPUTS: usize = 784;
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;
const TRAIN_SAMPLES: usize = 60000;
const TEST_SAMPLES: usize = 10000;
// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;
const VALIDATION_SPLIT: f32 = 0.1; // 10% of training data for validation
const EARLY_STOPPING_PATIENCE: usize = 3; // Number of epochs without improvement before stopping
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001; // Minimum change to be considered an improvement

// ============================================================================
// Internal Abstractions (Inlined for self-contained binary)
// ============================================================================

/// Constant learning rate scheduler (for backward compatibility).
///
/// This scheduler maintains a constant learning rate throughout training.
/// Used when no config file is provided.
struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    /// Create a `ConstantLR` that always yields the given learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// let sched = ConstantLR::new(0.01);
    /// assert_eq!(sched.get_lr(), 0.01);
    /// ```
    fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    /// Retrieve the configured constant learning rate.
    ///
    /// # Examples
    ///
    /// ```
    /// let sched = ConstantLR::new(0.01);
    /// let lr = sched.get_lr();
    /// assert_eq!(lr, 0.01_f32);
    /// ```
    fn get_lr(&self) -> f32 {
        self.lr
    }

    /// Advance the scheduler to the next step. For `ConstantLR` this is a no-op.
    ///
    /// This implementation does not change the learning rate or any internal state.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = ConstantLR::new(0.01);
    /// let lr_before = s.get_lr();
    /// s.step();
    /// assert_eq!(s.get_lr(), lr_before);
    /// ```
    fn step(&mut self) {
        // No-op for constant learning rate
    }

    /// Resets the scheduler to its initial state.
    ///
    /// For ConstantLR this operation has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut scheduler = ConstantLR::new(0.01);
    /// scheduler.step();
    /// scheduler.reset();
    /// assert_eq!(scheduler.get_lr(), 0.01);
    /// ```
    fn reset(&mut self) {
        // No-op for constant learning rate
    }
}

/// Core trait for neural network layers.
pub trait Layer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize);
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    );
    fn update_parameters(&mut self, learning_rate: f32);
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
}

/// Simple random number generator.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    pub fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 {
            0x9e3779b97f4a7c15
        } else {
            nanos
        };
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    pub fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }
}

/// ReLU activation function applied in-place.
pub fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// Softmax activation function applied row-wise.
pub fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
    if cols == 0 {
        return;
    }
    assert_eq!(
        outputs.len(),
        rows * cols,
        "outputs length mismatch in softmax_rows"
    );

    for row in outputs.chunks_exact_mut(cols).take(rows) {
        let mut max_value = row[0];
        for &value in row.iter().skip(1) {
            if value > max_value {
                max_value = value;
            }
        }

        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max_value).exp();
            sum += *value;
        }

        let inv_sum = 1.0f32 / sum;
        for value in row.iter_mut() {
            *value *= inv_sum;
        }
    }
}

/// Fully connected layer (manual implementation, no BLAS).
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    grad_weights: RefCell<Vec<f32>>,
    grad_biases: RefCell<Vec<f32>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> Self {
        let mut weights = vec![0.0; input_size * output_size];
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        for w in &mut weights {
            *w = rng.gen_range_f32(-limit, limit);
        }

        Self {
            input_size,
            output_size,
            weights,
            biases: vec![0.0; output_size],
            grad_weights: RefCell::new(vec![0.0; input_size * output_size]),
            grad_biases: RefCell::new(vec![0.0; output_size]),
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn biases(&self) -> &[f32] {
        &self.biases
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;

            for j in 0..self.output_size {
                let mut sum = self.biases[j];
                for i in 0..self.input_size {
                    sum += input[in_offset + i] * self.weights[i * self.output_size + j];
                }
                output[out_offset + j] = sum;
            }
        }
    }

    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let scale = 1.0 / batch_size as f32;
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        // Zero out grad_input first
        for v in grad_input.iter_mut() {
            *v = 0.0;
        }

        for b in 0..batch_size {
            let in_offset = b * self.input_size;
            let out_offset = b * self.output_size;

            for j in 0..self.output_size {
                let g = grad_output[out_offset + j];
                grad_b[j] += g * scale;

                for i in 0..self.input_size {
                    grad_w[i * self.output_size + j] += input[in_offset + i] * g * scale;
                    grad_input[in_offset + i] += g * self.weights[i * self.output_size + j] * scale;
                }
            }
        }
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        let mut grad_w = self.grad_weights.borrow_mut();
        let mut grad_b = self.grad_biases.borrow_mut();

        for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
            *w -= learning_rate * g;
        }
        for (b, g) in self.biases.iter_mut().zip(grad_b.iter()) {
            *b -= learning_rate * g;
        }

        // Reset gradients
        for g in grad_w.iter_mut() {
            *g = 0.0;
        }
        for g in grad_b.iter_mut() {
            *g = 0.0;
        }
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.output_size
    }
}

// ============================================================================
// Main Logic
// ============================================================================

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: DenseLayer,
    output_layer: DenseLayer,
}

// Network construction 784 -> 512 -> 10.
/// Create a feedforward NeuralNetwork with randomized DenseLayer parameters.
///
/// The returned network contains a hidden DenseLayer sized NUM_INPUTS -> NUM_HIDDEN
/// and an output DenseLayer sized NUM_HIDDEN -> NUM_OUTPUTS. Layer parameters
/// are initialized using the provided RNG (the RNG is reseeded inside the function).
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let nn = initialize_network(&mut rng);
/// assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
/// assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
/// ```
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    let hidden_layer = DenseLayer::new(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = DenseLayer::new(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

/// Computes the total cross-entropy loss for a batch of softmax outputs and writes the corresponding gradient (softmax-backed deltas) into `delta`.
///
/// For each of the first `rows` samples, the function:
/// - Accumulates negative log probability of the true class (using a small epsilon to avoid log(0)) into the returned loss.
/// - Writes a gradient row into `delta` equal to the predicted probabilities with `1.0` subtracted at the true label index (i.e., `p_i` for i != y, and `p_y - 1.0` for the true class).
///
/// # Parameters
///
/// - `outputs`: Flattened row-major softmax output probabilities with length at least `rows * cols`.
/// - `labels`: True class labels for the batch; only the first `rows` entries are used.
/// - `rows`: Number of samples (rows) to process from `outputs` and `labels`.
/// - `cols`: Number of classes (columns) per sample.
/// - `delta`: Mutable flattened buffer where computed gradient rows are written; must have length at least `rows * cols`.
///
/// # Returns
///
/// Total cross-entropy loss summed over the processed `rows` samples.
///
/// # Examples
///
/// ```
/// let outputs = [0.9f32, 0.1f32]; // one sample, two-class softmax
/// let labels = [0u8];
/// let mut delta = [0.0f32; 2];
/// let loss = compute_delta_and_loss(&outputs, &labels, 1, 2, &mut delta);
/// let expected_loss = -(0.9f32).ln();
/// assert!((loss - expected_loss).abs() < 1e-6);
/// // gradient: true class probability minus 1, other classes remain as probabilities
/// assert!((delta[0] - (-0.1f32)).abs() < 1e-6);
/// assert!((delta[1] - 0.1f32).abs() < 1e-6);
/// ```
fn compute_delta_and_loss(
    outputs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
) -> f32 {
    let mut total_loss = 0.0f32;
    let epsilon = 1e-9f32;

    for (row_idx, &label) in labels.iter().enumerate().take(rows) {
        let row_start = row_idx * cols;
        let label = label as usize;
        let prob = outputs[row_start + label].max(epsilon);
        total_loss -= prob.ln();

        let row = &outputs[row_start..row_start + cols];
        let delta_row = &mut delta[row_start..row_start + cols];
        for (j, value) in row.iter().enumerate() {
            let mut v = *value;
            if j == label {
                v -= 1.0;
            }
            delta_row[j] = v;
        }
    }

    total_loss
}

/// Copies a minibatch of images and labels selected by index into the provided output buffers.
///
/// Copies `count` samples whose indices are taken from `indices[start..start + count]`.
/// Each image is NUM_INPUTS consecutive values in `images` and is copied into `out_inputs` at consecutive positions (batch-major).
/// Corresponding labels are copied into `out_labels` in order.
///
/// The caller must ensure `out_inputs` has length at least `count * NUM_INPUTS`, `out_labels` has length at least `count`,
/// and that `indices[start..start + count]` contains valid indices into `images` and `labels`.
///
/// # Examples
///
/// ```
/// let images = vec![0f32; NUM_INPUTS * 2];
/// let labels = vec![1u8, 2u8];
/// let indices = vec![1usize, 0usize];
/// let mut out_inputs = vec![0f32; NUM_INPUTS * 2];
/// let mut out_labels = vec![0u8; 2];
/// gather_batch(&images, &labels, &indices, 0, 2, &mut out_inputs, &mut out_labels);
/// assert_eq!(out_labels, vec![2, 1]);
/// ```
fn gather_batch(
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    start: usize,
    count: usize,
    out_inputs: &mut [f32],
    out_labels: &mut [u8],
) {
    let input_stride = NUM_INPUTS;
    for i in 0..count {
        let src_index = indices[start + i];
        let src_start = src_index * input_stride;
        let dst_start = i * input_stride;
        let src_slice = &images[src_start..src_start + input_stride];
        let dst_slice = &mut out_inputs[dst_start..dst_start + input_stride];
        dst_slice.copy_from_slice(src_slice);
        out_labels[i] = labels[src_index];
    }
}

// Training with shuffling and minibatches.
/// Trains the neural network for the configured number of epochs using minibatch SGD, updating the network in-place and appending per-epoch metrics to ./logs/training_loss_c.txt.
///
/// Evaluates on the provided validation set each epoch, uses the scheduler's current learning rate for parameter updates, saves the best model to "mnist_model_best.bin" when validation loss improves, and supports early stopping based on validation loss and configured patience. Progress (training loss, validation loss, validation accuracy, learning rate, and epoch time) is printed to stdout and recorded in CSV format.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let mut nn = initialize_network(&mut rng);
/// let images = vec![0.0f32; NUM_INPUTS * 1];
/// let labels = vec![0u8; 1];
/// let val_images = vec![0.0f32; NUM_INPUTS * 1];
/// let val_labels = vec![0u8; 1];
/// let mut scheduler = ConstantLR::new(0.01);
/// train(&mut nn, &images, &labels, 1, &val_images, &val_labels, 1, &mut rng, &mut scheduler);
/// ```
#[allow(clippy::too_many_arguments)]
fn train(
    nn: &mut NeuralNetwork,
    images: &[f32],
    labels: &[u8],
    num_samples: usize,
    val_images: &[f32],
    val_labels: &[u8],
    val_num_samples: usize,
    rng: &mut SimpleRng,
    scheduler: &mut dyn LRScheduler,
) {
    // Attempt to create logs dir if not exists
    std::fs::create_dir_all("./logs").ok();

    let file = File::create("./logs/training_loss_c.txt").unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    let mut loss_file = BufWriter::new(file);

    // Write CSV header
    writeln!(
        loss_file,
        "epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate"
    )
    .unwrap_or_else(|_| {
        eprintln!("Failed writing CSV header.");
        process::exit(1);
    });

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut batch_labels = vec![0u8; BATCH_SIZE];
    let mut a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];

    let mut indices: Vec<usize> = (0..num_samples).collect();

    let mut unused_grad = vec![0.0f32; BATCH_SIZE * NUM_INPUTS]; // Preallocate reusable buffer.

    // Early stopping state
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0usize;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        let start_time = Instant::now();
        let current_lr = scheduler.get_lr();

        // Fisher-Yates shuffle.
        if num_samples > 1 {
            for i in (1..num_samples).rev() {
                let j = rng.gen_usize(i + 1);
                indices.swap(i, j);
            }
        }

        for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
            let batch_count = (num_samples - batch_start).min(BATCH_SIZE);

            gather_batch(
                images,
                labels,
                &indices,
                batch_start,
                batch_count,
                &mut batch_inputs,
                &mut batch_labels,
            );

            // Forward: hidden layer.
            let a1_len = batch_count * NUM_HIDDEN;
            nn.hidden_layer.forward(&batch_inputs, &mut a1, batch_count);
            relu_inplace(&mut a1[..a1_len]);

            // Forward: output layer.
            let a2_len = batch_count * NUM_OUTPUTS;
            nn.output_layer.forward(&a1, &mut a2, batch_count);
            assert_eq!(
                a2[..a2_len].len(),
                batch_count * NUM_OUTPUTS,
                "Buffer size mismatch before softmax_rows"
            );
            softmax_rows(&mut a2[..a2_len], batch_count, NUM_OUTPUTS);

            // Output delta and loss.
            let batch_loss = compute_delta_and_loss(
                &a2[..a2_len],
                &batch_labels[..batch_count],
                batch_count,
                NUM_OUTPUTS,
                &mut dz2,
            );
            total_loss += batch_loss;

            // Backward: output layer.
            nn.output_layer.backward(&a1, &dz2, &mut dz1, batch_count);

            // Apply ReLU derivative to hidden layer gradient.
            let dz1_len = batch_count * NUM_HIDDEN;
            for i in 0..dz1_len {
                if a1[i] <= 0.0 {
                    dz1[i] = 0.0;
                }
            }

            // Backward: hidden layer.
            let grad_len = batch_count * NUM_INPUTS;
            nn.hidden_layer.backward(
                &batch_inputs,
                &dz1,
                &mut unused_grad[..grad_len],
                batch_count,
            );

            // Update parameters.
            nn.output_layer.update_parameters(current_lr);
            nn.hidden_layer.update_parameters(current_lr);
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_loss = total_loss / num_samples as f32;

        // Evaluate on validation set
        let mut val_total_loss = 0.0f32;
        let mut val_correct = 0usize;
        let mut val_batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
        let mut val_a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
        let mut val_a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];

        for batch_start in (0..val_num_samples).step_by(BATCH_SIZE) {
            let batch_count = (val_num_samples - batch_start).min(BATCH_SIZE);
            let input_len = batch_count * NUM_INPUTS;
            let input_start = batch_start * NUM_INPUTS;
            val_batch_inputs[..input_len]
                .copy_from_slice(&val_images[input_start..input_start + input_len]);

            // Forward: hidden layer
            let val_a1_len = batch_count * NUM_HIDDEN;
            nn.hidden_layer
                .forward(&val_batch_inputs, &mut val_a1, batch_count);
            relu_inplace(&mut val_a1[..val_a1_len]);

            // Forward: output layer
            let val_a2_len = batch_count * NUM_OUTPUTS;
            nn.output_layer.forward(&val_a1, &mut val_a2, batch_count);
            softmax_rows(&mut val_a2[..val_a2_len], batch_count, NUM_OUTPUTS);

            // Compute loss
            let epsilon = 1e-9f32;
            for row_idx in 0..batch_count {
                let row_start = row_idx * NUM_OUTPUTS;
                let label = val_labels[batch_start + row_idx] as usize;
                let prob = val_a2[row_start + label].max(epsilon);
                val_total_loss -= prob.ln();

                // Compute accuracy
                let row = &val_a2[row_start..row_start + NUM_OUTPUTS];
                let mut predicted = 0usize;
                let mut max_prob = row[0];
                for (i, &value) in row.iter().enumerate().skip(1) {
                    if value > max_prob {
                        max_prob = value;
                        predicted = i;
                    }
                }
                if predicted == label {
                    val_correct += 1;
                }
            }
        }

        let val_average_loss = val_total_loss / val_num_samples as f32;
        let val_accuracy = val_correct as f32 / val_num_samples as f32 * 100.0;

        println!(
            "Epoch {}, Loss: {:.6}, Val Loss: {:.6}, Val Acc: {:.2}%, LR: {:.6}, Time: {:.6}",
            epoch + 1,
            average_loss,
            val_average_loss,
            val_accuracy,
            current_lr,
            duration
        );
        writeln!(
            loss_file,
            "{},{},{},{},{},{}",
            epoch + 1,
            average_loss,
            duration,
            val_average_loss,
            val_accuracy,
            current_lr
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing training loss data.");
            process::exit(1);
        });

        // Early stopping check
        if val_average_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA {
            best_val_loss = val_average_loss;
            epochs_without_improvement = 0;
            // Save best model
            save_model(nn, "mnist_model_best.bin");
        } else {
            epochs_without_improvement += 1;
        }

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE {
            println!(
                "\nEarly stopping triggered! No improvement for {} epochs. Best validation loss: {:.6}",
                EARLY_STOPPING_PATIENCE, best_val_loss
            );
            break;
        }

        // Update learning rate for next epoch
        scheduler.step();
    }
}

// Evaluate accuracy on the test set using batches.
/// Evaluates a trained network on the provided dataset and prints the test accuracy.
///
/// The function processes the dataset in minibatches, performs forward passes (hidden ReLU, output softmax),
/// selects the highest-probability class per sample, counts correct predictions, and prints accuracy as a percentage.
///
/// # Examples
///
/// ```
/// // Assume `nn`, `images`, `labels`, and `num_samples` are prepared:
/// // let mut rng = SimpleRng::new();
/// // let nn = initialize_network(&mut rng);
/// // let images: Vec<f32> = ...; // flattened images
/// // let labels: Vec<u8> = ...; // one label per image
/// test(&nn, &images, &labels, num_samples);
/// ```
fn test(nn: &NeuralNetwork, images: &[f32], labels: &[u8], num_samples: usize) {
    let mut correct_predictions = 0usize;
    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];

    for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch_count = (num_samples - batch_start).min(BATCH_SIZE);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len].copy_from_slice(&images[input_start..input_start + input_len]);

        // Forward: hidden layer.
        let a1_len = batch_count * NUM_HIDDEN;
        nn.hidden_layer.forward(&batch_inputs, &mut a1, batch_count);
        relu_inplace(&mut a1[..a1_len]);

        // Forward: output layer.
        let a2_len = batch_count * NUM_OUTPUTS;
        nn.output_layer.forward(&a1, &mut a2, batch_count);
        assert_eq!(
            a2[..a2_len].len(),
            batch_count * NUM_OUTPUTS,
            "Buffer size mismatch before softmax_rows in test"
        );
        softmax_rows(&mut a2[..a2_len], batch_count, NUM_OUTPUTS);

        for row_idx in 0..batch_count {
            let row_start = row_idx * NUM_OUTPUTS;
            let row = &a2[row_start..row_start + NUM_OUTPUTS];
            let mut predicted = 0usize;
            let mut max_prob = row[0];
            for (i, &value) in row.iter().enumerate().skip(1) {
                if value > max_prob {
                    max_prob = value;
                    predicted = i;
                }
            }
            if predicted == labels[batch_start + row_idx] as usize {
                correct_predictions += 1;
            }
        }
    }

    let accuracy = correct_predictions as f32 / num_samples as f32 * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
}

// Save the model in binary (little-endian i32 + f32).
/// Serializes the neural network to a binary file using little-endian encoding.
///
/// The file contains, in order:
/// 1. Three 32-bit integers: hidden layer input size, hidden layer output size, and output layer output size.
/// 2. All hidden layer weights as 32-bit floats.
/// 3. All hidden layer biases as 32-bit floats.
/// 4. All output layer weights as 32-bit floats.
/// 5. All output layer biases as 32-bit floats.
///
/// The function terminates the process with an error message if the file cannot be created or any write fails.
///
/// # Examples
///
/// ```
/// // Serializes `nn` to "mnist_model.bin".
/// save_model(&nn, "mnist_model.bin");
/// ```
fn save_model(nn: &NeuralNetwork, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {} for writing model", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    let write_i32 = |writer: &mut BufWriter<File>, value: i32| {
        writer.write_all(&value.to_le_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };
    let write_f32 = |writer: &mut BufWriter<File>, value: f32| {
        writer.write_all(&value.to_le_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };

    write_i32(&mut writer, nn.hidden_layer.input_size() as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size() as i32);
    write_i32(&mut writer, nn.output_layer.output_size() as i32);

    for &value in nn.hidden_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in nn.hidden_layer.biases() {
        write_f32(&mut writer, value);
    }
    for &value in nn.output_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in nn.output_layer.biases() {
        write_f32(&mut writer, value);
    }

    println!("Model saved to {}", filename);
}

fn read_be_u32(data: &[u8], offset: &mut usize) -> u32 {
    let b0 = (data[*offset] as u32) << 24;
    let b1 = (data[*offset + 1] as u32) << 16;
    let b2 = (data[*offset + 2] as u32) << 8;
    let b3 = data[*offset + 3] as u32;
    *offset += 4;
    b0 | b1 | b2 | b3
}

// Read IDX images and normalize to [0, 1].
fn read_mnist_images(filename: &str, num_images: usize) -> Vec<f32> {
    let data = std::fs::read(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let mut offset = 0usize;
    let _magic_number = read_be_u32(&data, &mut offset);
    let total_images = read_be_u32(&data, &mut offset) as usize;
    let rows = read_be_u32(&data, &mut offset) as usize;
    let cols = read_be_u32(&data, &mut offset) as usize;
    let image_size = rows * cols;
    let actual_count = num_images.min(total_images);
    let total_bytes = actual_count * image_size;

    if data.len() < offset + total_bytes {
        eprintln!("MNIST image file is truncated");
        process::exit(1);
    }

    let mut images = vec![0.0f32; total_bytes];
    let src = &data[offset..offset + total_bytes];
    for (dst, &pixel) in images.iter_mut().zip(src.iter()) {
        *dst = pixel as f32 / 255.0;
    }

    images
}

// Read IDX labels (0-9).
fn read_mnist_labels(filename: &str, num_labels: usize) -> Vec<u8> {
    let data = std::fs::read(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let mut offset = 0usize;
    let _magic_number = read_be_u32(&data, &mut offset);
    let total_labels = read_be_u32(&data, &mut offset) as usize;
    let actual_count = num_labels.min(total_labels);

    if data.len() < offset + actual_count {
        eprintln!("MNIST label file is truncated");
        process::exit(1);
    }

    data[offset..offset + actual_count].to_vec()
}

/// Program entry point that loads MNIST data, constructs a learning-rate scheduler and neural network, trains and evaluates the model, and saves the trained parameters.
///
/// The function parses an optional CLI config path to select a learning-rate scheduler, measures and reports timings for data loading, training, and testing, and writes the final model to `mnist_model.bin`.
///
/// # Examples
///
/// ```
/// // Run the program entry point (typically executed by the runtime).
/// main();
/// ```
fn main() {
    let program_start = Instant::now();

    // Parse command-line arguments for optional config file
    let args: Vec<String> = env::args().collect();
    let config_path = if args.len() > 1 {
        Some(args[1].as_str())
    } else {
        None
    };

    // Create scheduler based on config or use default constant LR
    let mut scheduler: Box<dyn LRScheduler> = if let Some(path) = config_path {
        match load_config(path) {
            Ok(config) => {
                println!("Loaded config from: {}", path);
                match config.scheduler_type.as_str() {
                    "step_decay" => {
                        let step_size = config.step_size.unwrap_or(3);
                        let gamma = config.gamma.unwrap_or(0.5);
                        println!(
                            "Using StepDecay scheduler: initial_lr={}, step_size={}, gamma={}",
                            LEARNING_RATE, step_size, gamma
                        );
                        Box::new(StepDecay::new(LEARNING_RATE, step_size, gamma))
                    }
                    "exponential" => {
                        let decay_rate = config.decay_rate.unwrap_or(0.95);
                        println!(
                            "Using ExponentialDecay scheduler: initial_lr={}, decay_rate={}",
                            LEARNING_RATE, decay_rate
                        );
                        Box::new(ExponentialDecay::new(LEARNING_RATE, decay_rate))
                    }
                    "cosine_annealing" => {
                        let min_lr = config.min_lr.unwrap_or(0.0001);
                        let t_max = config.T_max.unwrap_or(EPOCHS);
                        println!(
                            "Using CosineAnnealing scheduler: initial_lr={}, min_lr={}, T_max={}",
                            LEARNING_RATE, min_lr, t_max
                        );
                        Box::new(CosineAnnealing::new(LEARNING_RATE, min_lr, t_max))
                    }
                    _ => {
                        eprintln!(
                            "Unknown scheduler type: {}. Using constant learning rate.",
                            config.scheduler_type
                        );
                        Box::new(ConstantLR::new(LEARNING_RATE))
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "Failed to load config from {}: {}. Using constant learning rate.",
                    path, e
                );
                Box::new(ConstantLR::new(LEARNING_RATE))
            }
        }
    } else {
        println!(
            "No config file provided. Using constant learning rate: {}",
            LEARNING_RATE
        );
        Box::new(ConstantLR::new(LEARNING_RATE))
    };

    println!("Loading training data...");
    let load_start = Instant::now();
    let mut train_images = read_mnist_images("./data/train-images.idx3-ubyte", TRAIN_SAMPLES);
    let mut train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte", TRAIN_SAMPLES);

    println!("Loading test data...");
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte", TEST_SAMPLES);
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte", TEST_SAMPLES);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Data loading time: {:.2} seconds", load_time);

    // Split training data into train and validation sets
    let total_train_samples = train_images.len() / NUM_INPUTS;
    let validation_samples = (total_train_samples as f32 * VALIDATION_SPLIT) as usize;
    let actual_train_samples = total_train_samples - validation_samples;

    let split_point_images = actual_train_samples * NUM_INPUTS;
    let split_point_labels = actual_train_samples;

    let val_images = train_images.split_off(split_point_images);
    let val_labels = train_labels.split_off(split_point_labels);

    println!(
        "Data split: {} training samples, {} validation samples, {} test samples",
        actual_train_samples, validation_samples, TEST_SAMPLES
    );

    println!("Initializing neural network...");
    let mut rng = SimpleRng::new(1);
    let mut nn = initialize_network(&mut rng);

    println!("Training neural network...");
    let train_start = Instant::now();
    train(
        &mut nn,
        &train_images,
        &train_labels,
        actual_train_samples,
        &val_images,
        &val_labels,
        validation_samples,
        &mut rng,
        scheduler.as_mut(),
    );
    let train_time = train_start.elapsed().as_secs_f64();
    println!("Total training time: {:.2} seconds", train_time);

    println!("Testing neural network...");
    let test_start = Instant::now();
    let test_samples = test_images.len() / NUM_INPUTS;
    test(&nn, &test_images, &test_labels, test_samples);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Testing time: {:.2} seconds", test_time);

    println!("Saving model...");
    save_model(&nn, "mnist_model.bin");

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Data loading time: {:.2} seconds", load_time);
    println!("Total training time: {:.2} seconds", train_time);
    println!("Testing time: {:.2} seconds", test_time);
    println!("Total program time: {:.2} seconds", total_time);
    println!("========================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_delta_and_loss() {
        let outputs = vec![0.1, 0.2, 0.7, 0.3, 0.4, 0.3];
        let labels = vec![2, 1];
        let mut delta = vec![0.0; 6];

        let loss = compute_delta_and_loss(&outputs, &labels, 2, 3, &mut delta);

        assert!(loss > 0.0);
        assert!((delta[2] - (0.7 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gather_batch() {
        let images = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels = [0, 1];
        let indices = [1, 0];
        let mut out_inputs = [0.0; 6];
        let mut out_labels = [0; 2];

        const TEST_NUM_INPUTS: usize = 3;
        let input_stride = TEST_NUM_INPUTS;

        for i in 0..2 {
            let src_index = indices[i];
            let src_start = src_index * input_stride;
            let dst_start = i * input_stride;
            let src_slice = &images[src_start..src_start + input_stride];
            let dst_slice = &mut out_inputs[dst_start..dst_start + input_stride];
            dst_slice.copy_from_slice(src_slice);
            out_labels[i] = labels[src_index];
        }

        assert_eq!(out_labels[0], 1);
        assert_eq!(out_labels[1], 0);
        assert!((out_inputs[0] - 4.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_initialize_network() {
        let mut rng = SimpleRng::new(42);
        let nn = initialize_network(&mut rng);

        assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
        assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
        assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
    }
}