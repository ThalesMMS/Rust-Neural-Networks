//! MNIST dataset loader.
//!
//! This module provides functions to load the MNIST dataset from IDX binary files.
//! MNIST consists of 70000 28x28 grayscale images of handwritten digits (0-9).
//! Training set: 60000 images
//! Test set: 10000 images
//!
//! Binary format (IDX):
//! - Images file (idx3-ubyte):
//!   - 4 bytes: magic number (0x00000803 = 2051, big-endian)
//!   - 4 bytes: number of images (big-endian u32)
//!   - 4 bytes: number of rows (28, big-endian u32)
//!   - 4 bytes: number of columns (28, big-endian u32)
//!   - N * 28 * 28 bytes: raw pixel data (u8, row-major order)
//! - Labels file (idx1-ubyte):
//!   - 4 bytes: magic number (0x00000801 = 2049, big-endian)
//!   - 4 bytes: number of labels (big-endian u32)
//!   - N bytes: raw label data (u8, values 0-9)

use crate::utils::error_messages::mnist_data_not_found_message;
use std::fs;

/// MNIST image dimensions
pub const MNIST_IMAGE_SIZE: usize = 28;
pub const MNIST_IMAGE_PIXELS: usize = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE; // 784
pub const MNIST_NUM_CLASSES: usize = 10;

/// Magic numbers for MNIST IDX files (big-endian)
const MNIST_IMAGE_MAGIC: u32 = 2051; // 0x00000803
const MNIST_LABEL_MAGIC: u32 = 2049; // 0x00000801

/// Read a big-endian u32 from a byte slice at the given offset, advancing the offset by 4.
fn read_be_u32(data: &[u8], offset: &mut usize) -> u32 {
    let b0 = (data[*offset] as u32) << 24;
    let b1 = (data[*offset + 1] as u32) << 16;
    let b2 = (data[*offset + 2] as u32) << 8;
    let b3 = data[*offset + 3] as u32;
    *offset += 4;
    b0 | b1 | b2 | b3
}

/// Load MNIST images from an IDX3 binary file and return as normalized floats.
///
/// Reads all images from the IDX3-ubyte file, normalizing pixel values from
/// `[0, 255]` to `[0.0, 1.0]`. The returned vector is flattened in row-major order:
/// `[img0_row0_col0, img0_row0_col1, ..., img0_row27_col27, img1_row0_col0, ...]`.
///
/// # Arguments
///
/// * `filename` - Path to the MNIST images IDX3 file (e.g., `"data/train-images-idx3-ubyte"`).
///
/// # Returns
///
/// A `Vec<f32>` of length `num_images * 28 * 28` with values in `[0.0, 1.0]`,
/// or an `io::Error` if the file cannot be read, has an incorrect magic number,
/// has unexpected dimensions, or is truncated.
///
/// # Examples
///
/// ```no_run
/// let images = rust_neural_networks::data::mnist::read_mnist_images(
///     "data/train-images-idx3-ubyte",
/// )
/// .expect("failed to read MNIST images");
/// assert_eq!(images.len(), 60000 * 28 * 28);
/// ```
pub fn read_mnist_images(filename: &str) -> std::io::Result<Vec<f32>> {
    let data = fs::read(filename).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                mnist_data_not_found_message(filename),
            )
        } else {
            e
        }
    })?;

    if data.len() < 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "MNIST image file too short: {} bytes (need at least 16 for header)",
                data.len()
            ),
        ));
    }

    let mut offset = 0usize;
    let magic = read_be_u32(&data, &mut offset);

    if magic != MNIST_IMAGE_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid MNIST image magic number. Expected {}, got {}",
                MNIST_IMAGE_MAGIC, magic
            ),
        ));
    }

    let num_images = read_be_u32(&data, &mut offset) as usize;
    let rows = read_be_u32(&data, &mut offset) as usize;
    let cols = read_be_u32(&data, &mut offset) as usize;

    if rows != MNIST_IMAGE_SIZE || cols != MNIST_IMAGE_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Unexpected MNIST image dimensions: {}x{}, expected {}x{}",
                rows, cols, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE
            ),
        ));
    }

    let image_pixels = rows * cols;
    let total_bytes = num_images * image_pixels;

    if data.len() < offset + total_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "MNIST image file is truncated. Expected {} bytes of pixel data, got {}",
                total_bytes,
                data.len() - offset
            ),
        ));
    }

    let mut images = vec![0.0f32; total_bytes];
    let src = &data[offset..offset + total_bytes];
    for (dst, &pixel) in images.iter_mut().zip(src.iter()) {
        *dst = pixel as f32 / 255.0;
    }

    Ok(images)
}

/// Load MNIST labels from an IDX1 binary file.
///
/// Reads all labels from the IDX1-ubyte file. Each label is a `u8` in `[0, 9]`
/// representing the digit class.
///
/// # Arguments
///
/// * `filename` - Path to the MNIST labels IDX1 file (e.g., `"data/train-labels-idx1-ubyte"`).
///
/// # Returns
///
/// A `Vec<u8>` of length `num_labels` with class indices `0`–`9`,
/// or an `io::Error` if the file cannot be read, has an incorrect magic number,
/// or is truncated.
///
/// # Examples
///
/// ```no_run
/// let labels = rust_neural_networks::data::mnist::read_mnist_labels(
///     "data/train-labels-idx1-ubyte",
/// )
/// .expect("failed to read MNIST labels");
/// assert_eq!(labels.len(), 60000);
/// assert!(labels.iter().all(|&l| l < 10));
/// ```
pub fn read_mnist_labels(filename: &str) -> std::io::Result<Vec<u8>> {
    let data = fs::read(filename).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                mnist_data_not_found_message(filename),
            )
        } else {
            e
        }
    })?;

    if data.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "MNIST label file too short: {} bytes (need at least 8 for header)",
                data.len()
            ),
        ));
    }

    let mut offset = 0usize;
    let magic = read_be_u32(&data, &mut offset);

    if magic != MNIST_LABEL_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid MNIST label magic number. Expected {}, got {}",
                MNIST_LABEL_MAGIC, magic
            ),
        ));
    }

    let num_labels = read_be_u32(&data, &mut offset) as usize;

    if data.len() < offset + num_labels {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "MNIST label file is truncated. Expected {} label bytes, got {}",
                num_labels,
                data.len() - offset
            ),
        ));
    }

    Ok(data[offset..offset + num_labels].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_constants() {
        assert_eq!(MNIST_IMAGE_SIZE, 28);
        assert_eq!(MNIST_IMAGE_PIXELS, 784);
        assert_eq!(MNIST_NUM_CLASSES, 10);
        assert_eq!(MNIST_IMAGE_MAGIC, 2051);
        assert_eq!(MNIST_LABEL_MAGIC, 2049);
    }

    #[test]
    fn test_read_be_u32() {
        let data = [0x00u8, 0x00, 0x08, 0x03];
        let mut offset = 0;
        assert_eq!(read_be_u32(&data, &mut offset), 2051);
        assert_eq!(offset, 4);
    }

    #[test]
    fn test_read_be_u32_offset_advances() {
        let data = [0x00u8, 0x00, 0x08, 0x03, 0x00, 0x00, 0x08, 0x01];
        let mut offset = 0;
        assert_eq!(read_be_u32(&data, &mut offset), 2051);
        assert_eq!(read_be_u32(&data, &mut offset), 2049);
        assert_eq!(offset, 8);
    }

    #[test]
    fn test_read_mnist_images_wrong_magic() {
        // Build a fake IDX file with wrong magic number
        let mut data = Vec::new();
        // Wrong magic: 0x00000802 instead of 0x00000803
        data.extend_from_slice(&[0x00u8, 0x00, 0x08, 0x02]);
        // num_images = 1
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x01]);
        // rows = 28
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x1C]);
        // cols = 28
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x1C]);
        // pixel data (784 bytes)
        data.extend(vec![0u8; 784]);

        let tmp = write_temp_file(&data);
        let result = read_mnist_images(&tmp);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("magic"),
            "Expected magic number error, got: {}",
            err
        );
    }

    #[test]
    fn test_read_mnist_images_wrong_dimensions() {
        // Build a fake IDX file with 32x32 dimensions instead of 28x28
        let mut data = Vec::new();
        // Correct magic
        data.extend_from_slice(&[0x00u8, 0x00, 0x08, 0x03]);
        // num_images = 1
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x01]);
        // rows = 32 (wrong)
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x20]);
        // cols = 32 (wrong)
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x20]);
        // pixel data (1024 bytes)
        data.extend(vec![128u8; 1024]);

        let tmp = write_temp_file(&data);
        let result = read_mnist_images(&tmp);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("dimensions"),
            "Expected dimensions error, got: {}",
            err
        );
    }

    #[test]
    fn test_read_mnist_images_valid() {
        // Build a minimal valid IDX3 file with 2 images
        let num_images: u32 = 2;
        let rows: u32 = 28;
        let cols: u32 = 28;
        let pixel_count = (num_images * rows * cols) as usize;

        let mut data = Vec::new();
        data.extend_from_slice(&2051u32.to_be_bytes()); // magic
        data.extend_from_slice(&num_images.to_be_bytes());
        data.extend_from_slice(&rows.to_be_bytes());
        data.extend_from_slice(&cols.to_be_bytes());
        // pixel values: first image all 0, second image all 255
        data.extend(vec![0u8; 784]);
        data.extend(vec![255u8; 784]);

        let tmp = write_temp_file(&data);
        let images = read_mnist_images(&tmp).expect("should parse valid IDX3 file");

        assert_eq!(images.len(), pixel_count);
        // First image: all 0.0
        for &px in &images[..784] {
            assert_eq!(px, 0.0f32);
        }
        // Second image: all 1.0
        for &px in &images[784..] {
            assert!((px - 1.0f32).abs() < 1e-6, "expected 1.0, got {}", px);
        }
    }

    #[test]
    fn test_read_mnist_labels_wrong_magic() {
        let mut data = Vec::new();
        // Wrong magic: 0x00000800 instead of 0x00000801
        data.extend_from_slice(&[0x00u8, 0x00, 0x08, 0x00]);
        // num_labels = 1
        data.extend_from_slice(&[0x00u8, 0x00, 0x00, 0x01]);
        data.push(5u8);

        let tmp = write_temp_file(&data);
        let result = read_mnist_labels(&tmp);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("magic"),
            "Expected magic number error, got: {}",
            err
        );
    }

    #[test]
    fn test_read_mnist_labels_valid() {
        let labels_raw: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let num_labels = labels_raw.len() as u32;

        let mut data = Vec::new();
        data.extend_from_slice(&2049u32.to_be_bytes()); // magic
        data.extend_from_slice(&num_labels.to_be_bytes());
        data.extend_from_slice(&labels_raw);

        let tmp = write_temp_file(&data);
        let labels = read_mnist_labels(&tmp).expect("should parse valid IDX1 file");

        assert_eq!(labels, labels_raw);
    }

    #[test]
    fn test_read_mnist_labels_truncated() {
        let mut data = Vec::new();
        data.extend_from_slice(&2049u32.to_be_bytes()); // magic
                                                        // Claims 100 labels but provides none
        data.extend_from_slice(&100u32.to_be_bytes());
        // No label bytes

        let tmp = write_temp_file(&data);
        let result = read_mnist_labels(&tmp);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("truncated"),
            "Expected truncated error, got: {}",
            err
        );
    }

    #[test]
    fn test_read_mnist_images_not_found() {
        let result = read_mnist_images("data/__nonexistent_mnist_images_test__.idx3-ubyte");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(
            msg.contains("MNIST"),
            "Expected MNIST in error message, got: {}",
            msg
        );
        assert!(
            msg.contains("docs/"),
            "Expected docs/ in error message, got: {}",
            msg
        );
    }

    #[test]
    fn test_read_mnist_labels_not_found() {
        let result = read_mnist_labels("data/__nonexistent_mnist_labels_test__.idx1-ubyte");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(
            msg.contains("MNIST"),
            "Expected MNIST in error message, got: {}",
            msg
        );
        assert!(
            msg.contains("docs/"),
            "Expected docs/ in error message, got: {}",
            msg
        );
    }

    /// Write bytes to a temporary file and return the path.
    fn write_temp_file(data: &[u8]) -> String {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("could not create temp file");
        f.write_all(data).expect("could not write temp file");
        // Persist so the file outlives this function; leak the path
        let (_, path) = f.keep().expect("could not persist temp file");
        path.to_str().expect("path is not valid UTF-8").to_string()
    }
}
