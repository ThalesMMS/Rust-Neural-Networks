use super::*;

/// Save the generator and discriminator weights to a single binary file.
///
/// The generator's weights are written first followed by the discriminator's
/// weights into the specified `filename`. If the file cannot be created, the
/// process terminates with exit code 1 after printing an error to stderr.
///
/// # Examples
///
/// ```no_run
/// // Construct or obtain `gen` and `disc` from your application and then:
/// save_model(&gen, &disc, "model.bin");
/// ```
pub(crate) fn save_model(gen: &Generator, disc: &Discriminator, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open {} for writing", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);
    gen.save(&mut writer);
    disc.save(&mut writer);
    writer.flush().unwrap_or_else(|_| {
        eprintln!("Failed flushing model data");
        process::exit(1);
    });
    writer.get_ref().sync_all().unwrap_or_else(|_| {
        eprintln!("Failed syncing model data");
        process::exit(1);
    });
    println!("Model saved to {}", filename);
}

/// Write `n` generated samples to a CSV file, one row per sample.
///
/// Each row contains IMG_SIZE (784) pixel values formatted to six decimal places.
/// Pixel values are in the range [-1, 1] (Tanh output range).
///
/// On failure to create the file or to write a row, the process prints an error to
/// stderr and exits with code 1.
///
/// # Examples
///
/// ```no_run
/// // assuming `gen` is initialized and `rng` is a dedicated evaluation RNG
/// export_samples(&gen, &mut rng, 10, "samples.csv");
/// ```
pub(crate) fn export_samples(gen: &Generator, rng: &mut SimpleRng, n: usize, path: &str) {
    let noise = sample_noise(rng, n);
    let (_, _, samples) = gen.forward(&noise, n);

    let file = File::create(path).unwrap_or_else(|_| {
        eprintln!("Could not open {} for writing", path);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    for i in 0..n {
        let start = i * IMG_SIZE;
        let row: Vec<String> = samples[start..start + IMG_SIZE]
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect();
        writeln!(writer, "{}", row.join(",")).unwrap_or_else(|_| {
            eprintln!("Failed writing sample row");
            process::exit(1);
        });
    }
    writer.flush().unwrap_or_else(|_| {
        eprintln!("Failed flushing generated samples");
        process::exit(1);
    });
    writer.get_ref().sync_all().unwrap_or_else(|_| {
        eprintln!("Failed syncing generated samples");
        process::exit(1);
    });
    println!("Generated samples saved to {}", path);
}

// ============================================================================
// Main Entry Point
// ============================================================================
