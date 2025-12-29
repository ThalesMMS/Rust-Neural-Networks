extern crate blas_src;

use cblas::{Layout, Transpose, sgemm};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

// Simple RNG for reproducibility without external crates.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    // Explicit seed (if zero, use a fixed value).
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    // Reseed based on the current time.
    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 { 0x9e3779b97f4a7c15 } else { nanos };
    }

    // Basic xorshift to generate u32.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    // Convert to [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    // Uniform sample in [low, high).
    fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    // Integer sample in [0, upper).
    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }
}

// Dense layer: weights (input x output) and biases (row-major).
struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    hidden_layer: LinearLayer,
    output_layer: LinearLayer,
}

// Initialize a layer with Xavier and zero biases.
fn initialize_layer(input_size: usize, output_size: usize, rng: &mut SimpleRng) -> LinearLayer {
    let mut weights = vec![0.0f32; input_size * output_size];
    let limit = (6.0f32 / (input_size + output_size) as f32).sqrt();
    for value in &mut weights {
        *value = rng.gen_range_f32(-limit, limit);
    }

    LinearLayer {
        input_size,
        output_size,
        weights,
        biases: vec![0.0f32; output_size],
    }
}

// Network construction 784 -> 512 -> 10.
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
    let hidden_layer = initialize_layer(NUM_INPUTS, NUM_HIDDEN, rng);
    let output_layer = initialize_layer(NUM_HIDDEN, NUM_OUTPUTS, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

fn sgemm_wrapper(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
) {
    let trans_a = if transpose_a {
        Transpose::Ordinary
    } else {
        Transpose::None
    };
    let trans_b = if transpose_b {
        Transpose::Ordinary
    } else {
        Transpose::None
    };

    unsafe {
        sgemm(
            Layout::RowMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            lda as i32,
            b,
            ldb as i32,
            beta,
            c,
            ldc as i32,
        );
    }
}

fn add_bias(data: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    for row in data.chunks_exact_mut(cols).take(rows) {
        for (value, b) in row.iter_mut().zip(bias) {
            *value += *b;
        }
    }
}

fn relu_inplace(data: &mut [f32]) {
    for value in data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn softmax_rows(outputs: &mut [f32], rows: usize, cols: usize) {
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

fn sum_rows(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }

    for row in data.chunks_exact(cols).take(rows) {
        for (value, sum) in row.iter().zip(out.iter_mut()) {
            *sum += *value;
        }
    }
}

fn compute_delta_and_loss(
    outputs: &[f32],
    labels: &[u8],
    rows: usize,
    cols: usize,
    delta: &mut [f32],
) -> f32 {
    let mut total_loss = 0.0f32;
    let epsilon = 1e-9f32;

    for row_idx in 0..rows {
        let row_start = row_idx * cols;
        let label = labels[row_idx] as usize;
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

fn apply_sgd_update(weights: &mut [f32], grads: &[f32]) {
    for (w, g) in weights.iter_mut().zip(grads.iter()) {
        *w -= LEARNING_RATE * *g;
    }
}

// Training with shuffling and minibatches.
fn train(nn: &mut NeuralNetwork, images: &[f32], labels: &[u8], num_samples: usize, rng: &mut SimpleRng) {
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
    let mut grad_w1 = vec![0.0f32; NUM_INPUTS * NUM_HIDDEN];
    let mut grad_w2 = vec![0.0f32; NUM_HIDDEN * NUM_OUTPUTS];
    let mut grad_b1 = vec![0.0f32; NUM_HIDDEN];
    let mut grad_b2 = vec![0.0f32; NUM_OUTPUTS];

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
            let scale = 1.0f32 / batch_count as f32;

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
            sgemm_wrapper(
                batch_count,
                NUM_HIDDEN,
                NUM_INPUTS,
                &batch_inputs,
                NUM_INPUTS,
                &nn.hidden_layer.weights,
                NUM_HIDDEN,
                &mut a1,
                NUM_HIDDEN,
                false,
                false,
                1.0,
                0.0,
            );
            let a1_len = batch_count * NUM_HIDDEN;
            add_bias(&mut a1[..a1_len], batch_count, NUM_HIDDEN, &nn.hidden_layer.biases);
            relu_inplace(&mut a1[..a1_len]);

            // Forward: output layer.
            sgemm_wrapper(
                batch_count,
                NUM_OUTPUTS,
                NUM_HIDDEN,
                &a1,
                NUM_HIDDEN,
                &nn.output_layer.weights,
                NUM_OUTPUTS,
                &mut a2,
                NUM_OUTPUTS,
                false,
                false,
                1.0,
                0.0,
            );
            let a2_len = batch_count * NUM_OUTPUTS;
            add_bias(&mut a2[..a2_len], batch_count, NUM_OUTPUTS, &nn.output_layer.biases);
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

            // Output-layer gradients: dW2 = A1^T * dZ2.
            sgemm_wrapper(
                NUM_HIDDEN,
                NUM_OUTPUTS,
                batch_count,
                &a1,
                NUM_HIDDEN,
                &dz2,
                NUM_OUTPUTS,
                &mut grad_w2,
                NUM_OUTPUTS,
                true,
                false,
                scale,
                0.0,
            );
            sum_rows(&dz2[..a2_len], batch_count, NUM_OUTPUTS, &mut grad_b2);
            for value in grad_b2.iter_mut() {
                *value *= scale;
            }

            // Hidden-layer gradient: dZ1 = dZ2 * W2^T.
            sgemm_wrapper(
                batch_count,
                NUM_HIDDEN,
                NUM_OUTPUTS,
                &dz2,
                NUM_OUTPUTS,
                &nn.output_layer.weights,
                NUM_OUTPUTS,
                &mut dz1,
                NUM_HIDDEN,
                false,
                true,
                1.0,
                0.0,
            );
            let dz1_len = batch_count * NUM_HIDDEN;
            for i in 0..dz1_len {
                if a1[i] <= 0.0 {
                    dz1[i] = 0.0;
                }
            }

            // Hidden-layer gradients: dW1 = X^T * dZ1.
            sgemm_wrapper(
                NUM_INPUTS,
                NUM_HIDDEN,
                batch_count,
                &batch_inputs,
                NUM_INPUTS,
                &dz1,
                NUM_HIDDEN,
                &mut grad_w1,
                NUM_HIDDEN,
                true,
                false,
                scale,
                0.0,
            );
            sum_rows(&dz1[..dz1_len], batch_count, NUM_HIDDEN, &mut grad_b1);
            for value in grad_b1.iter_mut() {
                *value *= scale;
            }

            apply_sgd_update(&mut nn.output_layer.weights, &grad_w2);
            apply_sgd_update(&mut nn.output_layer.biases, &grad_b2);
            apply_sgd_update(&mut nn.hidden_layer.weights, &grad_w1);
            apply_sgd_update(&mut nn.hidden_layer.biases, &grad_b1);
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
fn test(nn: &NeuralNetwork, images: &[f32], labels: &[u8], num_samples: usize) {
    let mut correct_predictions = 0usize;
    let mut batch_inputs = vec![0.0f32; BATCH_SIZE * NUM_INPUTS];
    let mut a1 = vec![0.0f32; BATCH_SIZE * NUM_HIDDEN];
    let mut a2 = vec![0.0f32; BATCH_SIZE * NUM_OUTPUTS];

    for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch_count = (num_samples - batch_start).min(BATCH_SIZE);
        let input_len = batch_count * NUM_INPUTS;
        let input_start = batch_start * NUM_INPUTS;
        batch_inputs[..input_len]
            .copy_from_slice(&images[input_start..input_start + input_len]);

        sgemm_wrapper(
            batch_count,
            NUM_HIDDEN,
            NUM_INPUTS,
            &batch_inputs,
            NUM_INPUTS,
            &nn.hidden_layer.weights,
            NUM_HIDDEN,
            &mut a1,
            NUM_HIDDEN,
            false,
            false,
            1.0,
            0.0,
        );
        let a1_len = batch_count * NUM_HIDDEN;
        add_bias(&mut a1[..a1_len], batch_count, NUM_HIDDEN, &nn.hidden_layer.biases);
        relu_inplace(&mut a1[..a1_len]);

        sgemm_wrapper(
            batch_count,
            NUM_OUTPUTS,
            NUM_HIDDEN,
            &a1,
            NUM_HIDDEN,
            &nn.output_layer.weights,
            NUM_OUTPUTS,
            &mut a2,
            NUM_OUTPUTS,
            false,
            false,
            1.0,
            0.0,
        );
        let a2_len = batch_count * NUM_OUTPUTS;
        add_bias(&mut a2[..a2_len], batch_count, NUM_OUTPUTS, &nn.output_layer.biases);
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

    write_i32(&mut writer, nn.hidden_layer.input_size as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size as i32);
    write_i32(&mut writer, nn.output_layer.output_size as i32);

    for value in &nn.hidden_layer.weights {
        write_f64(&mut writer, *value as f64);
    }
    for value in &nn.hidden_layer.biases {
        write_f64(&mut writer, *value as f64);
    }
    for value in &nn.output_layer.weights {
        write_f64(&mut writer, *value as f64);
    }
    for value in &nn.output_layer.biases {
        write_f64(&mut writer, *value as f64);
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
    train(&mut nn, &train_images, &train_labels, train_samples, &mut rng);
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
