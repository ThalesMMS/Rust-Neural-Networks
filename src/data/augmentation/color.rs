use crate::utils::rng::SimpleRng;

fn checked_pixel_count(width: usize, height: usize) -> usize {
    width.checked_mul(height).unwrap_or_else(|| {
        panic!(
            "Image dimension product overflow: width={}, height={}",
            width, height
        )
    })
}

fn checked_image_len(width: usize, height: usize, channels: usize) -> usize {
    checked_pixel_count(width, height)
        .checked_mul(channels)
        .unwrap_or_else(|| {
            panic!(
                "Image dimension product overflow: width={}, height={}, channels={}",
                width, height, channels
            )
        })
}

fn check_image_buffer(image: &[f32], width: usize, height: usize, channels: usize) {
    let expected = checked_image_len(width, height, channels);
    assert_eq!(
        image.len(),
        expected,
        "Image buffer size mismatch: expected {} elements, got {}",
        expected,
        image.len()
    );
}

/// Randomly adjust image brightness by adding a constant offset to all pixel values.
///
/// Samples `delta` uniformly from `[-max_delta, max_delta]`, adds it to each element of `image`,
/// and clamps results to `[0.0, 1.0]`.
///
/// # Panics
///
/// Panics if checked dimension/index arithmetic such as `width * height * channels`
/// overflows or if `image.len()` does not equal that f32 element count.
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
    check_image_buffer(image, width, height, channels);

    let delta = rng.gen_range_f32(-max_delta, max_delta);

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
/// Panics if checked dimension/index arithmetic such as `width * height * channels`
/// overflows or if `image.len()` does not equal that f32 element count.
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
    check_image_buffer(image, width, height, channels);

    let mean: f32 = image.iter().sum::<f32>() / image.len() as f32;
    let factor = rng.gen_range_f32(1.0 - max_delta, 1.0 + max_delta);

    for pixel in image.iter_mut() {
        *pixel = (mean + factor * (*pixel - mean)).clamp(0.0, 1.0);
    }
}

/// Randomly adjusts an image's saturation by interpolating between its grayscale and original colors.
///
/// Samples a saturation factor from `[1.0 - max_delta, 1.0 + max_delta]` and linearly blends each
/// RGB pixel toward or away from its luminance computed with weights 0.299 (R), 0.587 (G), and 0.114 (B).
/// The result for each channel is clamped to the range [0.0, 1.0].
///
/// # Panics
///
/// Panics if checked dimension/index arithmetic such as `width * height * channels`
/// or the pixel count overflows, if `image.len()` does not equal that f32 element
/// count, or if `channels != 3`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::data::augmentation::random_saturation;
/// use rust_neural_networks::utils::rng::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let width = 2;
/// let height = 1;
/// let channels = 3;
/// let mut image = vec![0.2f32, 0.4, 0.6, 0.8, 0.6, 0.4]; // two RGB pixels
///
/// random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
/// // all values remain within [0.0, 1.0]
/// assert!(image.iter().all(|&v| (0.0..=1.0).contains(&v)));
/// ```
pub fn random_saturation(
    image: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    max_delta: f32,
    rng: &mut SimpleRng,
) {
    check_image_buffer(image, width, height, channels);

    assert_eq!(
        channels, 3,
        "Saturation adjustment requires 3 channels (RGB), got {}",
        channels
    );

    let factor = rng.gen_range_f32(1.0 - max_delta, 1.0 + max_delta);

    let num_pixels = checked_pixel_count(width, height);
    for i in 0..num_pixels {
        let pixel_start = i * channels;

        let r = image[pixel_start];
        let g = image[pixel_start + 1];
        let b = image[pixel_start + 2];

        let gray = 0.299 * r + 0.587 * g + 0.114 * b;

        image[pixel_start] = (gray + factor * (r - gray)).clamp(0.0, 1.0);
        image[pixel_start + 1] = (gray + factor * (g - gray)).clamp(0.0, 1.0);
        image[pixel_start + 2] = (gray + factor * (b - gray)).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_brightness_increases_values() {
        let mut rng = SimpleRng::new(42);
        let width = 2;
        let height = 2;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        let original = image.clone();

        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
        assert_ne!(image, original);
    }

    #[test]
    fn test_random_brightness_clamps_values() {
        let mut rng = SimpleRng::new(123);
        let width = 2;
        let height = 2;
        let channels = 3;

        let mut image = vec![0.95f32; width * height * channels];
        random_brightness(&mut image, width, height, channels, 0.5, &mut rng);

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
        let mut image = vec![0.5f32; 100];
        random_brightness(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Image dimension product overflow")]
    fn test_random_brightness_dimension_overflow() {
        let mut rng = SimpleRng::new(1);
        let mut image = [];
        random_brightness(&mut image, usize::MAX, 2, 3, 0.2, &mut rng);
    }

    #[test]
    fn test_random_contrast_changes_values() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 10) as f32 / 10.0;
        }

        let original = image.clone();
        random_contrast(&mut image, width, height, channels, 0.2, &mut rng);

        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
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
        random_contrast(&mut image, width, height, channels, 0.1, &mut rng);
        let new_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        assert!((original_mean - new_mean).abs() < 0.1);
    }

    #[test]
    fn test_random_contrast_deterministic() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.5f32; width * height * channels];
        let mut image2 = vec![0.5f32; width * height * channels];

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
        let mut image = vec![0.5f32; 100];
        random_contrast(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    fn test_random_saturation_grayscale_conversion() {
        let mut rng = SimpleRng::new(42);
        let width = 2;
        let height = 2;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        image[0] = 1.0;
        image[1] = 0.0;
        image[2] = 0.0;

        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);

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

        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            let gray_value = 0.5;
            image[i * channels] = gray_value;
            image[i * channels + 1] = gray_value;
            image[i * channels + 2] = gray_value;
        }

        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

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

        for i in 0..(width * height) {
            image1[i * channels] = (i % 7) as f32 / 10.0;
            image1[i * channels + 1] = (i % 5) as f32 / 10.0;
            image1[i * channels + 2] = (i % 3) as f32 / 10.0;
        }
        image2.copy_from_slice(&image1);

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
        let mut image = vec![0.5f32; 100];
        random_saturation(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Saturation adjustment requires 3 channels")]
    fn test_random_saturation_invalid_channels() {
        let mut rng = SimpleRng::new(1);
        let width = 32;
        let height = 32;
        let channels = 1;
        let mut image = vec![0.5f32; width * height * channels];
        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
    }

    #[test]
    fn test_random_saturation_cifar10_size() {
        let mut rng = SimpleRng::new(888);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
