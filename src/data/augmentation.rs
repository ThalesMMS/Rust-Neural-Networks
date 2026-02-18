//! Image data augmentation functions for training.
//!
//! This module provides common data augmentation techniques to improve model
//! generalization and prevent overfitting. Augmentations should only be applied
//! to training data, not validation or test sets.
//!
//! All functions assume pixel-interleaved RGB format (RGBRGBRGB...) as used
//! by the CIFAR-10 loader.

use crate::utils::rng::SimpleRng;

/// Randomly flip an image horizontally with given probability.
///
/// This function mirrors the image left-to-right based on a random draw.
/// The decision to flip is made by comparing a random value against `probability`.
/// If flipping occurs, pixels are reordered so that columns are reversed while
/// maintaining the pixel-interleaved RGB layout.
///
/// # Arguments
///
/// * `image` - Mutable slice of pixel data in pixel-interleaved RGB format.
///   Length must equal `width * height * channels`.
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `channels` - Number of color channels (typically 3 for RGB).
/// * `probability` - Probability of flipping (0.0 = never flip, 1.0 = always flip).
/// * `rng` - Random number generator for the flip decision.
///
/// # Panics
///
/// Panics if `image.len() != width * height * channels`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_horizontal_flip;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 32;
/// let height = 32;
/// let channels = 3;
/// let mut image = vec![0.5f32; width * height * channels];
///
/// // 50% chance of horizontal flip
/// random_horizontal_flip(&mut image, width, height, channels, 0.5, &mut rng);
/// ```
pub fn random_horizontal_flip(
    image: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    probability: f32,
    rng: &mut SimpleRng,
) {
    // Validate image buffer size
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    // Decide whether to flip based on probability
    if rng.next_f32() >= probability {
        return; // No flip
    }

    // Flip the image horizontally by reversing each row
    // Each row has 'width' pixels, each pixel has 'channels' values
    let row_size = width * channels;

    for row in 0..height {
        let row_start = row * row_size;

        // Swap pixels from left and right within this row
        for col in 0..(width / 2) {
            let left_pixel_start = row_start + col * channels;
            let right_pixel_start = row_start + (width - 1 - col) * channels;

            // Swap all channels of the two pixels
            for c in 0..channels {
                image.swap(left_pixel_start + c, right_pixel_start + c);
            }
        }
    }
}

/// Randomly crop an image after applying zero-padding.
///
/// This function first pads the input image with zeros on all sides, then randomly
/// selects a crop region of the specified size. This is useful for data augmentation
/// during training, as it introduces translation invariance and prevents overfitting
/// to the exact positioning of objects in training images.
///
/// # Arguments
///
/// * `image` - Slice of pixel data in pixel-interleaved RGB format.
///   Length must equal `width * height * channels`.
/// * `width` - Original image width in pixels.
/// * `height` - Original image height in pixels.
/// * `channels` - Number of color channels (typically 3 for RGB).
/// * `padding` - Number of pixels to pad on each side (top, bottom, left, right).
/// * `crop_width` - Width of the output crop in pixels.
/// * `crop_height` - Height of the output crop in pixels.
/// * `rng` - Random number generator for selecting the crop position.
///
/// # Returns
///
/// A new `Vec<f32>` containing the cropped image data in pixel-interleaved RGB format.
/// Length will be `crop_width * crop_height * channels`.
///
/// # Panics
///
/// * Panics if `image.len() != width * height * channels`.
/// * Panics if `crop_width > width + 2 * padding` or `crop_height > height + 2 * padding`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_crop;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 32;
/// let height = 32;
/// let channels = 3;
/// let image = vec![0.5f32; width * height * channels];
///
/// // Pad with 4 pixels and crop back to original size
/// let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);
/// assert_eq!(cropped.len(), 32 * 32 * 3);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn random_crop(
    image: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    padding: usize,
    crop_width: usize,
    crop_height: usize,
    rng: &mut SimpleRng,
) -> Vec<f32> {
    // Validate input image buffer size
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    // Calculate padded dimensions
    let padded_width = width + 2 * padding;
    let padded_height = height + 2 * padding;

    // Validate crop dimensions
    assert!(
        crop_width <= padded_width,
        "Crop width {} exceeds padded image width {}",
        crop_width,
        padded_width
    );
    assert!(
        crop_height <= padded_height,
        "Crop height {} exceeds padded image height {}",
        crop_height,
        padded_height
    );

    // Create padded image buffer (initialized to zero)
    let mut padded = vec![0.0f32; padded_width * padded_height * channels];

    // Copy original image to center of padded buffer
    for row in 0..height {
        for col in 0..width {
            let src_idx = (row * width + col) * channels;
            let dst_row = row + padding;
            let dst_col = col + padding;
            let dst_idx = (dst_row * padded_width + dst_col) * channels;

            padded[dst_idx..dst_idx + channels]
                .copy_from_slice(&image[src_idx..src_idx + channels]);
        }
    }

    // Randomly select crop position
    let max_x = padded_width - crop_width;
    let max_y = padded_height - crop_height;
    let crop_x = if max_x > 0 {
        rng.gen_usize(max_x + 1)
    } else {
        0
    };
    let crop_y = if max_y > 0 {
        rng.gen_usize(max_y + 1)
    } else {
        0
    };

    // Extract crop region
    let mut cropped = vec![0.0f32; crop_width * crop_height * channels];
    for row in 0..crop_height {
        for col in 0..crop_width {
            let src_row = crop_y + row;
            let src_col = crop_x + col;
            let src_idx = (src_row * padded_width + src_col) * channels;
            let dst_idx = (row * crop_width + col) * channels;

            cropped[dst_idx..dst_idx + channels]
                .copy_from_slice(&padded[src_idx..src_idx + channels]);
        }
    }

    cropped
}

/// Randomly adjust image brightness by adding a constant offset to all pixel values.
///
/// This function uniformly samples a brightness adjustment factor from the range
/// `[-max_delta, max_delta]` and adds it to all color channels. Pixel values are
/// clamped to `[0.0, 1.0]` after adjustment to prevent overflow/underflow.
///
/// # Arguments
///
/// * `image` - Mutable slice of pixel data in pixel-interleaved RGB format.
///   Length must equal `width * height * channels`.
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `channels` - Number of color channels (typically 3 for RGB).
/// * `max_delta` - Maximum absolute brightness adjustment (typical range: 0.1 to 0.3).
/// * `rng` - Random number generator for sampling the adjustment factor.
///
/// # Panics
///
/// Panics if `image.len() != width * height * channels`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_brightness;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 32;
/// let height = 32;
/// let channels = 3;
/// let mut image = vec![0.5f32; width * height * channels];
///
/// // Randomly adjust brightness by up to ±0.2
/// random_brightness(&mut image, width, height, channels, 0.2, &mut rng);
/// ```
pub fn random_brightness(
    image: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    max_delta: f32,
    rng: &mut SimpleRng,
) {
    // Validate image buffer size
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    // Sample brightness adjustment uniformly from [-max_delta, max_delta]
    let delta = rng.gen_range_f32(-max_delta, max_delta);

    // Apply brightness adjustment to all pixels and clamp to [0.0, 1.0]
    for pixel in image.iter_mut() {
        *pixel = (*pixel + delta).clamp(0.0, 1.0);
    }
}

/// Randomly adjust image contrast by scaling pixel values around their mean.
///
/// This function uniformly samples a contrast factor from the range
/// `[1.0 - max_delta, 1.0 + max_delta]` and scales all pixel values around
/// their mean. A factor of 1.0 leaves the image unchanged, values less than 1.0
/// decrease contrast, and values greater than 1.0 increase contrast. Pixel values
/// are clamped to `[0.0, 1.0]` after adjustment.
///
/// # Arguments
///
/// * `image` - Mutable slice of pixel data in pixel-interleaved RGB format.
///   Length must equal `width * height * channels`.
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `channels` - Number of color channels (typically 3 for RGB).
/// * `max_delta` - Maximum absolute deviation from contrast factor 1.0 (typical: 0.2 to 0.5).
/// * `rng` - Random number generator for sampling the contrast factor.
///
/// # Panics
///
/// Panics if `image.len() != width * height * channels`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_contrast;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 32;
/// let height = 32;
/// let channels = 3;
/// let mut image = vec![0.5f32; width * height * channels];
///
/// // Randomly adjust contrast by factor in [0.8, 1.2]
/// random_contrast(&mut image, width, height, channels, 0.2, &mut rng);
/// ```
pub fn random_contrast(
    image: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    max_delta: f32,
    rng: &mut SimpleRng,
) {
    // Validate image buffer size
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    // Compute mean pixel value across all channels
    let mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

    // Sample contrast factor uniformly from [1.0 - max_delta, 1.0 + max_delta]
    let factor = rng.gen_range_f32(1.0 - max_delta, 1.0 + max_delta);

    // Apply contrast adjustment: pixel = mean + factor * (pixel - mean)
    for pixel in image.iter_mut() {
        *pixel = (mean + factor * (*pixel - mean)).clamp(0.0, 1.0);
    }
}

/// Randomly adjust image saturation by interpolating between grayscale and original.
///
/// This function uniformly samples a saturation factor from the range
/// `[1.0 - max_delta, 1.0 + max_delta]`. A factor of 0.0 produces a grayscale image,
/// 1.0 leaves the image unchanged, and values greater than 1.0 increase saturation.
/// The grayscale conversion uses standard luminance weights: 0.299*R + 0.587*G + 0.114*B.
///
/// # Arguments
///
/// * `image` - Mutable slice of pixel data in pixel-interleaved RGB format.
///   Length must equal `width * height * channels`.
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `channels` - Number of color channels (must be 3 for RGB).
/// * `max_delta` - Maximum absolute deviation from saturation factor 1.0 (typical: 0.2 to 0.5).
/// * `rng` - Random number generator for sampling the saturation factor.
///
/// # Panics
///
/// * Panics if `image.len() != width * height * channels`.
/// * Panics if `channels != 3` (saturation only applies to RGB images).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_saturation;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 32;
/// let height = 32;
/// let channels = 3;
/// let mut image = vec![0.5f32; width * height * channels];
///
/// // Randomly adjust saturation by factor in [0.8, 1.2]
/// random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
/// ```
pub fn random_saturation(
    image: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    max_delta: f32,
    rng: &mut SimpleRng,
) {
    // Validate image buffer size
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    // Saturation adjustment only applies to RGB images (3 channels)
    assert_eq!(
        channels, 3,
        "Saturation adjustment requires 3 channels (RGB), got {}",
        channels
    );

    // Sample saturation factor uniformly from [1.0 - max_delta, 1.0 + max_delta]
    let factor = rng.gen_range_f32(1.0 - max_delta, 1.0 + max_delta);

    // Process each pixel
    let num_pixels = width * height;
    for i in 0..num_pixels {
        let pixel_start = i * channels;

        // Extract RGB values
        let r = image[pixel_start];
        let g = image[pixel_start + 1];
        let b = image[pixel_start + 2];

        // Convert to grayscale using standard luminance weights
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;

        // Interpolate between grayscale and original based on saturation factor
        // factor = 0.0 -> full grayscale
        // factor = 1.0 -> original image
        // factor > 1.0 -> enhanced saturation
        image[pixel_start] = (gray + factor * (r - gray)).clamp(0.0, 1.0);
        image[pixel_start + 1] = (gray + factor * (g - gray)).clamp(0.0, 1.0);
        image[pixel_start + 2] = (gray + factor * (b - gray)).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_flip_deterministic() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 2;
        let channels = 3;

        // Create a simple test pattern (pixel values = column index)
        // Row 0: [0,0,0][1,1,1][2,2,2][3,3,3]
        // Row 1: [0,0,0][1,1,1][2,2,2][3,3,3]
        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                for c in 0..channels {
                    image[row * width * channels + col * channels + c] = col as f32;
                }
            }
        }

        // Flip with probability 1.0 (always flip)
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // After flip, columns should be reversed:
        // Row 0: [3,3,3][2,2,2][1,1,1][0,0,0]
        // Row 1: [3,3,3][2,2,2][1,1,1][0,0,0]
        for row in 0..height {
            for col in 0..width {
                let expected_value = (width - 1 - col) as f32;
                for c in 0..channels {
                    let idx = row * width * channels + col * channels + c;
                    assert_eq!(
                        image[idx], expected_value,
                        "Mismatch at row={}, col={}, channel={}",
                        row, col, c
                    );
                }
            }
        }
    }

    #[test]
    fn test_horizontal_flip_probability_zero() {
        let mut rng = SimpleRng::new(123);
        let width = 4;
        let height = 2;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        let original = image.clone();

        // Flip with probability 0.0 (never flip)
        random_horizontal_flip(&mut image, width, height, channels, 0.0, &mut rng);

        // Image should be unchanged
        assert_eq!(image, original);
    }

    #[test]
    fn test_horizontal_flip_single_channel() {
        let mut rng = SimpleRng::new(999);
        let width = 3;
        let height = 1;
        let channels = 1;

        // Single row: [0.1, 0.2, 0.3]
        let mut image = vec![0.1, 0.2, 0.3];

        // Flip with probability 1.0
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // After flip: [0.3, 0.2, 0.1]
        assert_eq!(image, vec![0.3, 0.2, 0.1]);
    }

    #[test]
    fn test_horizontal_flip_cifar10_size() {
        // Test with CIFAR-10 image dimensions (32x32x3)
        let mut rng = SimpleRng::new(777);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];

        // Set a distinctive pattern: top-left corner = 1.0, others = 0.0
        image[0] = 1.0; // Red channel of pixel (0,0)
        image[1] = 1.0; // Green channel of pixel (0,0)
        image[2] = 1.0; // Blue channel of pixel (0,0)

        // Flip with probability 1.0
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // After flip, the bright pixel should be at top-right corner
        let top_right_idx = (width - 1) * channels; // Last pixel of first row
        assert_eq!(image[top_right_idx], 1.0);
        assert_eq!(image[top_right_idx + 1], 1.0);
        assert_eq!(image[top_right_idx + 2], 1.0);

        // Top-left should now be 0.0
        assert_eq!(image[0], 0.0);
        assert_eq!(image[1], 0.0);
        assert_eq!(image[2], 0.0);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_horizontal_flip_invalid_buffer_size() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.0f32; 100]; // Wrong size
        random_horizontal_flip(&mut image, 32, 32, 3, 1.0, &mut rng);
    }

    #[test]
    fn test_random_crop_basic() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        // Create image with distinct values at each position
        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                let value = (row * width + col) as f32;
                for c in 0..channels {
                    image[(row * width + col) * channels + c] = value;
                }
            }
        }

        // Crop with no padding (should extract a 2x2 region)
        let cropped = random_crop(&image, width, height, channels, 0, 2, 2, &mut rng);

        // Verify output dimensions
        assert_eq!(cropped.len(), 2 * 2 * channels);

        // Verify all pixels in crop have consistent channel values
        for i in 0..4 {
            let r = cropped[i * channels];
            let g = cropped[i * channels + 1];
            let b = cropped[i * channels + 2];
            assert_eq!(r, g);
            assert_eq!(g, b);
        }
    }

    #[test]
    fn test_random_crop_with_padding() {
        let mut rng = SimpleRng::new(123);
        let width = 32;
        let height = 32;
        let channels = 3;

        // Create uniform image
        let image = vec![0.7f32; width * height * channels];

        // Pad with 4 pixels and crop back to original size
        let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);

        // Verify output dimensions
        assert_eq!(cropped.len(), 32 * 32 * 3);

        // Since the original image is uniform, crop should contain either:
        // - All 0.7 (if crop is entirely within original image)
        // - Mix of 0.7 and 0.0 (if crop includes padding)
        for &val in &cropped {
            assert!(val == 0.0 || val == 0.7);
        }
    }

    #[test]
    fn test_random_crop_padding_creates_zeros() {
        let mut rng = SimpleRng::new(999);
        let width = 2;
        let height = 2;
        let channels = 3;

        // Create image with all ones
        let image = vec![1.0f32; width * height * channels];

        // Pad with 2 pixels and crop to padded size (should include padding zeros)
        let cropped = random_crop(&image, width, height, channels, 2, 6, 6, &mut rng);

        // Verify output dimensions
        assert_eq!(cropped.len(), 6 * 6 * 3);

        // Count zeros and ones
        let zero_count = cropped.iter().filter(|&&v| v == 0.0).count();
        let one_count = cropped.iter().filter(|&&v| v == 1.0).count();

        // Should have both zeros (from padding) and ones (from original image)
        assert!(zero_count > 0, "Expected padding zeros");
        assert!(one_count > 0, "Expected original image ones");
        assert_eq!(zero_count + one_count, 6 * 6 * 3);
    }

    #[test]
    fn test_random_crop_deterministic() {
        let width = 8;
        let height = 8;
        let channels = 3;
        let image = vec![0.5f32; width * height * channels];

        // Two RNGs with same seed should produce identical crops
        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        let crop1 = random_crop(&image, width, height, channels, 2, 4, 4, &mut rng1);
        let crop2 = random_crop(&image, width, height, channels, 2, 4, 4, &mut rng2);

        assert_eq!(crop1, crop2);
    }

    #[test]
    fn test_random_crop_no_padding_full_image() {
        let mut rng = SimpleRng::new(111);
        let width = 3;
        let height = 3;
        let channels = 1;

        // Create simple pattern
        let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Crop entire image (no padding, crop size = image size)
        let cropped = random_crop(&image, width, height, channels, 0, 3, 3, &mut rng);

        // Should be identical to original
        assert_eq!(cropped, image);
    }

    #[test]
    fn test_random_crop_single_channel() {
        let mut rng = SimpleRng::new(222);
        let width = 4;
        let height = 4;
        let channels = 1;

        let image = vec![1.0f32; width * height * channels];

        let cropped = random_crop(&image, width, height, channels, 1, 3, 3, &mut rng);

        assert_eq!(cropped.len(), (3 * 3));
    }

    #[test]
    fn test_random_crop_cifar10_size() {
        // Test with CIFAR-10 dimensions (common use case)
        let mut rng = SimpleRng::new(333);
        let width = 32;
        let height = 32;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];

        // Typical augmentation: pad by 4 and crop back to 32x32
        let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);

        assert_eq!(cropped.len(), 32 * 32 * 3);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_random_crop_invalid_input_size() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 100]; // Wrong size
        random_crop(&image, 32, 32, 3, 4, 32, 32, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Crop width")]
    fn test_random_crop_crop_too_large() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 4 * 4 * 3];
        // Crop larger than padded image
        random_crop(&image, 4, 4, 3, 1, 10, 10, &mut rng);
    }

    #[test]
    fn test_random_crop_edge_case_zero_padding() {
        let mut rng = SimpleRng::new(444);
        let width = 5;
        let height = 5;
        let channels = 3;

        let image = vec![0.8f32; width * height * channels];

        // Zero padding should still work
        let cropped = random_crop(&image, width, height, channels, 0, 3, 3, &mut rng);

        assert_eq!(cropped.len(), 3 * 3 * 3);

        // All values should be from original image (no padding)
        for &val in &cropped {
            assert_eq!(val, 0.8);
        }
    }

    #[test]
    fn test_random_brightness_increases_values() {
        let mut rng = SimpleRng::new(42);
        let width = 2;
        let height = 2;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        // Store original values
        let original = image.clone();

        // Apply brightness with positive delta range [0.0, 0.2]
        // (by seeding RNG, we can make this deterministic)
        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        // Values should be clamped to [0.0, 1.0]
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }

        // At least some values should have changed
        assert_ne!(image, original);
    }

    #[test]
    fn test_random_brightness_clamps_values() {
        let mut rng = SimpleRng::new(123);
        let width = 2;
        let height = 2;
        let channels = 3;

        // Start with high values near 1.0
        let mut image = vec![0.95f32; width * height * channels];

        // Large brightness delta could push values above 1.0
        random_brightness(&mut image, width, height, channels, 0.5, &mut rng);

        // All values should be clamped to [0.0, 1.0]
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_random_brightness_deterministic() {
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image1 = vec![0.5f32; width * height * channels];
        let mut image2 = vec![0.5f32; width * height * channels];

        // Same seed should produce same results
        let mut rng1 = SimpleRng::new(999);
        let mut rng2 = SimpleRng::new(999);

        random_brightness(&mut image1, width, height, channels, 0.3, &mut rng1);
        random_brightness(&mut image2, width, height, channels, 0.3, &mut rng2);

        assert_eq!(image1, image2);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_random_brightness_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100]; // Wrong size
        random_brightness(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    fn test_random_contrast_changes_values() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        // Create image with varying values
        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 10) as f32 / 10.0;
        }

        let original = image.clone();

        // Apply contrast adjustment
        random_contrast(&mut image, width, height, channels, 0.2, &mut rng);

        // Values should be clamped to [0.0, 1.0]
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }

        // At least some values should have changed
        assert_ne!(image, original);
    }

    #[test]
    fn test_random_contrast_preserves_mean() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 20) as f32 / 20.0;
        }

        let original_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        // Apply small contrast adjustment
        random_contrast(&mut image, width, height, channels, 0.1, &mut rng);

        let new_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        // Mean should be approximately preserved (within tolerance due to clamping)
        assert!((original_mean - new_mean).abs() < 0.1);
    }

    #[test]
    fn test_random_contrast_deterministic() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.5f32; width * height * channels];
        let mut image2 = vec![0.5f32; width * height * channels];

        // Same seed should produce same results
        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        random_contrast(&mut image1, width, height, channels, 0.3, &mut rng1);
        random_contrast(&mut image2, width, height, channels, 0.3, &mut rng2);

        assert_eq!(image1, image2);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_random_contrast_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100]; // Wrong size
        random_contrast(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    fn test_random_saturation_grayscale_conversion() {
        let mut rng = SimpleRng::new(42);
        let width = 2;
        let height = 2;
        let channels = 3;

        // Create a pure red pixel
        let mut image = vec![0.0f32; width * height * channels];
        image[0] = 1.0; // R
        image[1] = 0.0; // G
        image[2] = 0.0; // B

        // Apply saturation adjustment
        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);

        // Values should be clamped to [0.0, 1.0]
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_random_saturation_preserves_grayscale() {
        let mut rng = SimpleRng::new(123);
        let width = 3;
        let height = 3;
        let channels = 3;

        // Create grayscale image (R=G=B for all pixels)
        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            let gray_value = 0.5;
            image[i * channels] = gray_value;
            image[i * channels + 1] = gray_value;
            image[i * channels + 2] = gray_value;
        }

        let _original = image.clone();

        // Saturation adjustment should not affect grayscale images
        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

        // Grayscale pixels should remain grayscale (R=G=B)
        for i in 0..(width * height) {
            let r = image[i * channels];
            let g = image[i * channels + 1];
            let b = image[i * channels + 2];
            assert!((r - g).abs() < 1e-6);
            assert!((g - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_random_saturation_deterministic() {
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image1 = vec![0.0f32; width * height * channels];
        let mut image2 = vec![0.0f32; width * height * channels];

        // Create colorful pattern
        for i in 0..(width * height) {
            image1[i * channels] = (i % 7) as f32 / 10.0;
            image1[i * channels + 1] = (i % 5) as f32 / 10.0;
            image1[i * channels + 2] = (i % 3) as f32 / 10.0;
        }
        image2.copy_from_slice(&image1);

        // Same seed should produce same results
        let mut rng1 = SimpleRng::new(555);
        let mut rng2 = SimpleRng::new(555);

        random_saturation(&mut image1, width, height, channels, 0.4, &mut rng1);
        random_saturation(&mut image2, width, height, channels, 0.4, &mut rng2);

        assert_eq!(image1, image2);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_random_saturation_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100]; // Wrong size
        random_saturation(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Saturation adjustment requires 3 channels")]
    fn test_random_saturation_invalid_channels() {
        let mut rng = SimpleRng::new(1);
        let width = 32;
        let height = 32;
        let channels = 1; // Only 1 channel (grayscale)
        let mut image = vec![0.5f32; width * height * channels];
        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
    }

    #[test]
    fn test_random_saturation_cifar10_size() {
        // Test with CIFAR-10 dimensions (common use case)
        let mut rng = SimpleRng::new(888);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        // Apply saturation adjustment
        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

        // Verify all values are valid
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
