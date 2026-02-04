//! CIFAR-10 dataset loader.
//!
//! This module provides functions to load the CIFAR-10 dataset from binary files.
//! CIFAR-10 consists of 60000 32x32 RGB images in 10 classes (6000 images per class).
//! Training set: 50000 images (5 batches of 10000 each)
//! Test set: 10000 images (1 batch)
//!
//! Binary format: Each file contains 10000 records, each record is 3073 bytes:
//! - 1 byte: label (0-9)
//! - 3072 bytes: image data (1024 red + 1024 green + 1024 blue pixels, row-major order)

use std::fs;

/// CIFAR-10 image dimensions
pub const CIFAR10_IMAGE_SIZE: usize = 32;
pub const CIFAR10_NUM_CHANNELS: usize = 3;
pub const CIFAR10_IMAGE_PIXELS: usize = CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE; // 1024
pub const CIFAR10_IMAGE_BYTES: usize = CIFAR10_IMAGE_PIXELS * CIFAR10_NUM_CHANNELS; // 3072
pub const CIFAR10_NUM_CLASSES: usize = 10;

/// Size of each record in CIFAR-10 binary files (1 byte label + 3072 bytes image data)
const CIFAR10_RECORD_SIZE: usize = 1 + CIFAR10_IMAGE_BYTES; // 3073 bytes

/// Number of images per CIFAR-10 batch file
const CIFAR10_BATCH_SIZE: usize = 10000;

/// Class names for CIFAR-10 labels (indices 0-9)
const CIFAR10_CLASS_NAMES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// Load a single CIFAR-10 binary batch into normalized, pixel-interleaved images and labels.
///
/// The CIFAR-10 batch file contains 10,000 records; each record is 1 label byte followed by
/// 3072 image bytes in channel-major order: 1024 red, 1024 green, then 1024 blue (each channel
/// is a 32×32 image stored row-major). This function returns a tuple (images, labels):
/// - `images`: flattened `Vec<f32>` of length `num_images * 32 * 32 * 3` with values normalized
///   to [0.0, 1.0] and arranged per-pixel in RGB order (`img0_px0_r, img0_px0_g, img0_px0_b, ...`).
/// - `labels`: `Vec<u8>` of class indices (0–9), one per image.
///
/// Returns an error if the file cannot be read or its size does not match the expected
/// CIFAR-10 batch size.
///
/// # Examples
///
/// ```no_run
/// # use std::path::Path;
/// # // Example assumes a CIFAR-10 batch file is available at the given path.
/// let path = "./data/cifar-10-batches-bin/data_batch_1.bin";
/// let (images, labels) = rust_neural_networks::data::cifar10::read_cifar10_batch(path)
///     .expect("failed to read CIFAR-10 batch");
/// assert_eq!(labels.len(), 10000);
/// assert_eq!(images.len(), 10000 * 32 * 32 * 3);
/// ```
pub fn read_cifar10_batch(filename: &str) -> std::io::Result<(Vec<f32>, Vec<u8>)> {
    let data = fs::read(filename)?;

    // Validate file size
    let expected_size = CIFAR10_BATCH_SIZE * CIFAR10_RECORD_SIZE;
    if data.len() != expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid CIFAR-10 batch file size. Expected {} bytes, got {} bytes",
                expected_size,
                data.len()
            ),
        ));
    }

    let num_images = CIFAR10_BATCH_SIZE;
    let mut images = vec![0.0f32; num_images * CIFAR10_IMAGE_BYTES];
    let mut labels = vec![0u8; num_images];

    // Parse each record
    for (i, label) in labels.iter_mut().enumerate() {
        let record_offset = i * CIFAR10_RECORD_SIZE;

        // Read label (first byte of record)
        *label = data[record_offset];

        // Read image data (3072 bytes after label)
        // CIFAR-10 format: channel-major (all R, then all G, then all B)
        // We convert to pixel-interleaved (RGB RGB RGB...) for easier processing
        let pixel_data_start = record_offset + 1;

        for pixel_idx in 0..CIFAR10_IMAGE_PIXELS {
            let r = data[pixel_data_start + pixel_idx] as f32 / 255.0;
            let g = data[pixel_data_start + CIFAR10_IMAGE_PIXELS + pixel_idx] as f32 / 255.0;
            let b = data[pixel_data_start + 2 * CIFAR10_IMAGE_PIXELS + pixel_idx] as f32 / 255.0;

            // Store in pixel-interleaved format
            let output_idx = i * CIFAR10_IMAGE_BYTES + pixel_idx * 3;
            images[output_idx] = r;
            images[output_idx + 1] = g;
            images[output_idx + 2] = b;
        }
    }

    Ok((images, labels))
}

/// Load and concatenate images and labels from multiple CIFAR-10 binary batch files.
///
/// The returned images are normalized `f32` values in pixel-interleaved RGB order
/// (flattened as image_count × 32 × 32 × 3). Labels are one `u8` per image.
///
/// # Arguments
///
/// * `filenames` - Slice of paths to CIFAR-10 binary batch files to read and concatenate.
///
/// # Returns
///
/// `(Vec<f32>, Vec<u8>)` — a tuple where the first element is the concatenated image data and the second is the concatenated labels.
/// Returns an error if any batch file cannot be read or fails validation.
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::data::cifar10::read_cifar10_batches;
///
/// let filenames = &[
///     "./data/cifar-10-batches-bin/data_batch_1.bin",
///     "./data/cifar-10-batches-bin/data_batch_2.bin",
/// ];
/// let (images, labels) = read_cifar10_batches(filenames)
///     .expect("failed to read CIFAR-10 batches");
/// assert_eq!(labels.len(), 20000);
/// assert_eq!(images.len(), 20000 * 32 * 32 * 3);
/// ```
pub fn read_cifar10_batches(filenames: &[&str]) -> std::io::Result<(Vec<f32>, Vec<u8>)> {
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    for filename in filenames {
        let (images, labels) = read_cifar10_batch(filename)?;
        all_images.extend(images);
        all_labels.extend(labels);
    }

    Ok((all_images, all_labels))
}

/// Get the CIFAR-10 class name for a label index.
///
/// # Panics
///
/// Panics if `label` is outside 0–9 with the message
/// "Invalid CIFAR-10 label: {label}. Must be 0-9.".
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::cifar10::label_to_name;
///
/// assert_eq!(label_to_name(0), "airplane");
/// assert_eq!(label_to_name(1), "automobile");
/// assert_eq!(label_to_name(9), "truck");
/// ```
pub fn label_to_name(label: u8) -> &'static str {
    if label as usize >= CIFAR10_NUM_CLASSES {
        panic!("Invalid CIFAR-10 label: {}. Must be 0-9.", label);
    }
    CIFAR10_CLASS_NAMES[label as usize]
}

/// Get a reference to the CIFAR-10 class names in label order (0 through 9).
///
/// # Returns
///
/// A reference to the static array of 10 CIFAR-10 class name strings, ordered by label index.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::cifar10::get_class_names;
///
/// let names = get_class_names();
/// assert_eq!(names.len(), 10);
/// assert_eq!(names[0], "airplane");
/// assert_eq!(names[9], "truck");
/// ```
pub fn get_class_names() -> &'static [&'static str; 10] {
    &CIFAR10_CLASS_NAMES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_to_name() {
        assert_eq!(label_to_name(0), "airplane");
        assert_eq!(label_to_name(1), "automobile");
        assert_eq!(label_to_name(2), "bird");
        assert_eq!(label_to_name(3), "cat");
        assert_eq!(label_to_name(4), "deer");
        assert_eq!(label_to_name(5), "dog");
        assert_eq!(label_to_name(6), "frog");
        assert_eq!(label_to_name(7), "horse");
        assert_eq!(label_to_name(8), "ship");
        assert_eq!(label_to_name(9), "truck");
    }

    #[test]
    #[should_panic(expected = "Invalid CIFAR-10 label")]
    fn test_label_to_name_invalid() {
        label_to_name(10);
    }

    #[test]
    fn test_get_class_names() {
        let names = get_class_names();
        assert_eq!(names.len(), 10);
        assert_eq!(names[0], "airplane");
        assert_eq!(names[9], "truck");
    }

    #[test]
    fn test_cifar10_constants() {
        assert_eq!(CIFAR10_IMAGE_SIZE, 32);
        assert_eq!(CIFAR10_NUM_CHANNELS, 3);
        assert_eq!(CIFAR10_IMAGE_PIXELS, 1024);
        assert_eq!(CIFAR10_IMAGE_BYTES, 3072);
        assert_eq!(CIFAR10_NUM_CLASSES, 10);
        assert_eq!(CIFAR10_RECORD_SIZE, 3073);
    }
}
