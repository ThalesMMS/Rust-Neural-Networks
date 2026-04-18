use super::*;

/// Serializes the CNN model to a binary file using little-endian encoding.
///
/// The file contains, in order:
/// 1. Conv layer metadata: out_channels, in_channels, kernel_size, padding, stride, input_height, input_width (as i32)
/// 2. All conv layer weights as 32-bit floats.
/// 3. All conv layer biases as 32-bit floats.
/// 4. FC layer metadata: input_size, output_size (as i32)
/// 5. All FC layer weights as 32-bit floats.
/// 6. All FC layer biases as 32-bit floats.
///
/// The function terminates the process with an error message if the file cannot be created or any write fails.
///
/// # Examples
///
/// ```
/// // Serializes `model` to "mnist_cnn_model_best.bin".
/// save_model(&model, "mnist_cnn_model_best.bin");
/// ```
pub(crate) fn save_model(model: &Cnn, filename: &str) {
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

    // Write conv layer metadata
    write_i32(&mut writer, model.conv_layer.out_channels() as i32);
    write_i32(&mut writer, model.conv_layer.in_channels() as i32);
    write_i32(&mut writer, model.conv_layer.kernel_size() as i32);
    write_i32(&mut writer, model.conv_layer.padding() as i32);
    write_i32(&mut writer, model.conv_layer.stride() as i32);
    write_i32(&mut writer, model.conv_layer.input_height() as i32);
    write_i32(&mut writer, model.conv_layer.input_width() as i32);

    // Write conv layer weights and biases
    for &value in model.conv_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in model.conv_layer.biases() {
        write_f32(&mut writer, value);
    }

    // Write FC layer metadata
    write_i32(&mut writer, model.fc_layer.input_size() as i32);
    write_i32(&mut writer, model.fc_layer.output_size() as i32);

    // Write FC layer weights and biases
    for &value in model.fc_layer.weights() {
        write_f32(&mut writer, value);
    }
    for &value in model.fc_layer.biases() {
        write_f32(&mut writer, value);
    }

    println!("Model saved to {}", filename);
}
