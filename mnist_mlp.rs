use rust_neural_networks::layers::{DenseLayer, Layer};
use rust_neural_networks::utils::{relu_inplace, softmax_rows, SimpleRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

// MLP with minibatches and GEMM (CPU) for MNIST (Rust port for study).
const NUM_INPUTS: usize = 784;
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;
const TRAIN_SAMPLES: usize = 60000;
const TEST_SAMPLES: usize = 10000;
// Training hyperparameters.
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;

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
/// let mut rng = SimpleRng::new();
/// let nn = initialize_network(&mut rng);
/// assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
/// assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
/// assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
/// ```
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
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
/// assert_eq!(out_labels, vec![1, 2]);
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
/// Trains the neural network using minibatch stochastic gradient descent, logging per-epoch loss and duration.
///
/// This function performs training for a fixed number of epochs using minibatches, updates the
/// network parameters in-place, and appends per-epoch loss and elapsed time to ./logs/training_loss_c.txt.
/// It also prints epoch-level loss and timing to standard output.
///
/// # Examples
///
/// ```
/// // Construct a tiny example network and dataset, then run one training invocation.
/// let mut rng = SimpleRng::new();
/// let mut nn = initialize_network(&mut rng);
/// let images = vec![0.0f32; NUM_INPUTS * 1]; // single example with zeroed pixels
/// let labels = vec![0u8; 1]; // single label
/// train(&mut nn, &images, &labels, 1, &mut rng);
/// ```
fn train(
    nn: &mut NeuralNetwork,
    images: &[f32],
    labels: &[u8],
    num_samples: usize,
    rng: &mut SimpleRng,
) {
    let file = File::create("./logs/training_loss_c.txt").unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    let mut loss_file = BufWriter::new(file);

    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut batch_labels = vec![0u8; BATCH_SIZE];
    let mut a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];
    let mut dz1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];

    let mut indices: Vec<usize> = (0..num_samples).collect();

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        let start_time = Instant::now();

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
            let mut unused_grad = vec![0.0f32; batch_count * NUM_INPUTS];
            nn.hidden_layer
                .backward(&batch_inputs, &dz1, &mut unused_grad, batch_count);

            // Update parameters.
            nn.output_layer.update_parameters(LEARNING_RATE);
            nn.hidden_layer.update_parameters(LEARNING_RATE);
        }

        let duration = start_time.elapsed().as_secs_f32();
        let average_loss = total_loss / num_samples as f32;
        println!(
            "Epoch {}, Loss: {:.6} Time: {:.6}",
            epoch + 1,
            average_loss,
            duration
        );
        writeln!(loss_file, "{},{},{}", epoch + 1, average_loss, duration).unwrap_or_else(|_| {
            eprintln!("Failed writing training loss data.");
            process::exit(1);
        });
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

// Save the model in binary (int + doubles, native endianness).
/// Serializes the neural network to a binary file using native endianness.
///
/// The file contains, in order:
/// 1. Three 32-bit integers: hidden layer input size, hidden layer output size, and output layer output size.
/// 2. All hidden layer weights as 64-bit floats (row-major as provided by `DenseLayer::weights()`).
/// 3. All hidden layer biases as 64-bit floats (in order from `DenseLayer::biases()`).
/// 4. All output layer weights as 64-bit floats.
/// 5. All output layer biases as 64-bit floats.
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
        writer.write_all(&value.to_ne_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };
    let write_f64 = |writer: &mut BufWriter<File>, value: f64| {
        writer.write_all(&value.to_ne_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };

    write_i32(&mut writer, nn.hidden_layer.input_size() as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size() as i32);
    write_i32(&mut writer, nn.output_layer.output_size() as i32);

    for &value in nn.hidden_layer.weights() {
        write_f64(&mut writer, value as f64);
    }
    for &value in nn.hidden_layer.biases() {
        write_f64(&mut writer, value as f64);
    }
    for &value in nn.output_layer.weights() {
        write_f64(&mut writer, value as f64);
    }
    for &value in nn.output_layer.biases() {
        write_f64(&mut writer, value as f64);
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

fn main() {
    let program_start = Instant::now();

    println!("Loading training data...");
    let load_start = Instant::now();
    let train_images = read_mnist_images("./data/train-images.idx3-ubyte", TRAIN_SAMPLES);
    let train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte", TRAIN_SAMPLES);

    println!("Loading test data...");
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte", TEST_SAMPLES);
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte", TEST_SAMPLES);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Data loading time: {:.2} seconds", load_time);

    println!("Initializing neural network...");
    let mut rng = SimpleRng::new(1);
    let mut nn = initialize_network(&mut rng);

    println!("Training neural network...");
    let train_start = Instant::now();
    let train_samples = train_images.len() / NUM_INPUTS;
    train(
        &mut nn,
        &train_images,
        &train_labels,
        train_samples,
        &mut rng,
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