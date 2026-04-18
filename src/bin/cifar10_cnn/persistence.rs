use super::*;

/// Serialize a Cnn's layers and parameters to a binary file.
///
/// The file contains a compact, layer-indexed binary representation of the model
/// suitable for later deserialization. This function will panic on file creation
/// or write failures and will panic if it encounters an unknown layer type.
///
/// # Examples
///
/// ```
/// # use crate::{SimpleRng, init_cnn, save_model};
/// let mut rng = SimpleRng::new(42);
/// let mut model = init_cnn(&mut rng, None).unwrap();
/// save_model(&model, "cifar10_cnn_model.bin");
/// ```
pub(crate) fn save_model(model: &Cnn, filename: &str) {
    use std::io::Write;

    let mut f = BufWriter::new(File::create(filename).expect("Failed to create model file"));

    // Write number of layers
    let num_layers = model.layers.len() as u32;
    f.write_all(&num_layers.to_le_bytes())
        .expect("Failed to write number of layers");

    // Iterate through each layer and save based on type
    for layer in &model.layers {
        let layer_ref: &dyn Layer = &**layer;
        let any_layer = layer_ref.as_any();

        // Try to downcast and save each layer type
        if let Some(dense_layer) = any_layer.downcast_ref::<DenseLayer>() {
            // Layer type ID: 0 = Dense
            f.write_all(&[0u8]).expect("Failed to write layer type");

            // Save dimensions
            let in_size = dense_layer.input_size() as u32;
            let out_size = dense_layer.output_size() as u32;
            f.write_all(&in_size.to_le_bytes())
                .expect("Failed to write input size");
            f.write_all(&out_size.to_le_bytes())
                .expect("Failed to write output size");

            // Save weights and biases
            for &w in dense_layer.weights() {
                f.write_all(&w.to_le_bytes())
                    .expect("Failed to write weight");
            }
            for &b in dense_layer.biases() {
                f.write_all(&b.to_le_bytes()).expect("Failed to write bias");
            }
        } else if let Some(conv_layer) = any_layer.downcast_ref::<Conv2DLayer>() {
            // Layer type ID: 1 = Conv2D
            f.write_all(&[1u8]).expect("Failed to write layer type");

            // Save configuration
            let in_channels = conv_layer.in_channels() as u32;
            let out_channels = conv_layer.out_channels() as u32;
            let kernel_size = conv_layer.kernel_size() as u32;
            let padding = conv_layer.padding() as i32;
            let stride = conv_layer.stride() as u32;
            let input_height = conv_layer.input_height() as u32;
            let input_width = conv_layer.input_width() as u32;

            f.write_all(&in_channels.to_le_bytes()).unwrap();
            f.write_all(&out_channels.to_le_bytes()).unwrap();
            f.write_all(&kernel_size.to_le_bytes()).unwrap();
            f.write_all(&padding.to_le_bytes()).unwrap();
            f.write_all(&stride.to_le_bytes()).unwrap();
            f.write_all(&input_height.to_le_bytes()).unwrap();
            f.write_all(&input_width.to_le_bytes()).unwrap();

            // Save weights and biases
            for &w in conv_layer.weights() {
                f.write_all(&w.to_le_bytes()).unwrap();
            }
            for &b in conv_layer.biases() {
                f.write_all(&b.to_le_bytes()).unwrap();
            }
        } else if let Some(bn_layer) = any_layer.downcast_ref::<BatchNormLayer>() {
            // Layer type ID: 2 = BatchNorm
            f.write_all(&[2u8]).expect("Failed to write layer type");

            // Save size
            let size = bn_layer.output_size() as u32;
            f.write_all(&size.to_le_bytes()).unwrap();

            // Save learnable parameters
            for &g in bn_layer.gamma() {
                f.write_all(&g.to_le_bytes()).unwrap();
            }
            for &b in bn_layer.beta() {
                f.write_all(&b.to_le_bytes()).unwrap();
            }

            // Save running statistics
            for &m in &bn_layer.running_mean() {
                f.write_all(&m.to_le_bytes()).unwrap();
            }
            for &v in &bn_layer.running_var() {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        } else if let Some(dropout_layer) = any_layer.downcast_ref::<DropoutLayer>() {
            // Layer type ID: 3 = Dropout
            f.write_all(&[3u8]).expect("Failed to write layer type");

            // Dropout has no trainable parameters, just save size and drop_rate for reconstruction
            let size = dropout_layer.output_size() as u32;
            f.write_all(&size.to_le_bytes()).unwrap();

            // Save drop_rate so load_model can reconstruct the layer correctly
            f.write_all(&dropout_layer.drop_rate().to_le_bytes())
                .unwrap();
        } else {
            panic!("Unknown layer type encountered during serialization");
        }
    }

    println!("Model saved to: {}", filename);
}

/// Reconstructs a `Cnn` from a binary file containing serialized layer data.
///
/// The file must follow the exact binary layout produced by the corresponding
/// serializer: a little-endian `u32` layer count followed by one-byte layer
/// type IDs and type-specific fields/arrays for each layer (Dense, Conv2D,
/// BatchNorm, Dropout). Unknown layer type IDs or I/O failures will cause a panic.
///
/// # Arguments
///
/// * `filename` - Path to the binary model file.
///
/// # Returns
///
/// A `Cnn` whose `layers` vector is populated from the file's serialized data.
///
/// # Panics
///
/// Panics if the file cannot be opened or read, if the file is truncated or
/// malformed, or if an unknown layer type ID is encountered.
///
/// # Examples
///
/// ```ignore
/// let model = load_model("cifar10_cnn_model_best.bin");
/// ```
#[allow(dead_code)]
pub(crate) fn load_model(filename: &str) -> Cnn {
    use std::io::Read;

    let mut f = File::open(filename).expect("Failed to open model file");

    // Read number of layers
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)
        .expect("Failed to read number of layers");
    let num_layers = u32::from_le_bytes(buf4) as usize;

    let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(num_layers);

    // Dropout RNG state is not serialized; loaded models get a fresh RNG for resumed training.
    let mut rng = SimpleRng::new(42);
    rng.reseed_from_time();

    for _ in 0..num_layers {
        // Read layer type ID
        let mut type_buf = [0u8; 1];
        f.read_exact(&mut type_buf)
            .expect("Failed to read layer type");
        let layer_type = type_buf[0];

        match layer_type {
            0 => {
                // Dense layer
                f.read_exact(&mut buf4).expect("Failed to read input size");
                let in_size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read output size");
                let out_size = u32::from_le_bytes(buf4) as usize;

                let weight_count = in_size * out_size;
                let mut weights = vec![0.0f32; weight_count];
                for w in weights.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read weight");
                    *w = f32::from_le_bytes(buf4);
                }

                let mut biases = vec![0.0f32; out_size];
                for b in biases.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read bias");
                    *b = f32::from_le_bytes(buf4);
                }

                layers.push(Box::new(DenseLayer::new_with_weights(
                    in_size, out_size, weights, biases,
                )));
            }
            1 => {
                // Conv2D layer
                f.read_exact(&mut buf4).expect("Failed to read in_channels");
                let in_channels = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4)
                    .expect("Failed to read out_channels");
                let out_channels = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read kernel_size");
                let kernel_size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read padding");
                let padding = i32::from_le_bytes(buf4) as isize;
                f.read_exact(&mut buf4).expect("Failed to read stride");
                let stride = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4)
                    .expect("Failed to read input_height");
                let input_height = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read input_width");
                let input_width = u32::from_le_bytes(buf4) as usize;

                let weight_count = out_channels * in_channels * kernel_size * kernel_size;
                let mut weights = vec![0.0f32; weight_count];
                for w in weights.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read weight");
                    *w = f32::from_le_bytes(buf4);
                }

                let mut biases = vec![0.0f32; out_channels];
                for b in biases.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read bias");
                    *b = f32::from_le_bytes(buf4);
                }

                layers.push(Box::new(Conv2DLayer::new_with_weights(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    stride,
                    input_height,
                    input_width,
                    weights,
                    biases,
                )));
            }
            2 => {
                // BatchNorm layer
                f.read_exact(&mut buf4).expect("Failed to read size");
                let size = u32::from_le_bytes(buf4) as usize;

                let mut gamma = vec![0.0f32; size];
                for g in gamma.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read gamma");
                    *g = f32::from_le_bytes(buf4);
                }

                let mut beta = vec![0.0f32; size];
                for b in beta.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read beta");
                    *b = f32::from_le_bytes(buf4);
                }

                let mut running_mean = vec![0.0f32; size];
                for m in running_mean.iter_mut() {
                    f.read_exact(&mut buf4)
                        .expect("Failed to read running_mean");
                    *m = f32::from_le_bytes(buf4);
                }

                let mut running_var = vec![0.0f32; size];
                for v in running_var.iter_mut() {
                    f.read_exact(&mut buf4).expect("Failed to read running_var");
                    *v = f32::from_le_bytes(buf4);
                }

                // Use standard epsilon and momentum defaults matching BatchNormLayer::new
                layers.push(Box::new(BatchNormLayer::new_with_params(
                    size,
                    1e-5,
                    0.1,
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                )));
            }
            3 => {
                // Dropout layer (no trainable parameters)
                f.read_exact(&mut buf4).expect("Failed to read size");
                let size = u32::from_le_bytes(buf4) as usize;
                f.read_exact(&mut buf4).expect("Failed to read drop_rate");
                let drop_rate = f32::from_le_bytes(buf4);

                layers.push(Box::new(DropoutLayer::new(size, drop_rate, &mut rng)));
            }
            _ => {
                panic!(
                    "Unknown layer type {} encountered during deserialization",
                    layer_type
                );
            }
        }
    }

    println!("Model loaded from: {}", filename);
    Cnn { layers }
}
