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
/// Panics if `probability` is not finite or is outside `[0.0, 1.0]`.
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
    assert!(
        probability.is_finite() && (0.0..=1.0).contains(&probability),
        "Horizontal flip probability must be finite and in [0.0, 1.0], got {}",
        probability
    );
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    if rng.next_f32() >= probability {
        return;
    }

    let row_size = width * channels;

    for row in 0..height {
        let row_start = row * row_size;

        for col in 0..(width / 2) {
            let left_pixel_start = row_start + col * channels;
            let right_pixel_start = row_start + (width - 1 - col) * channels;

            for c in 0..channels {
                image.swap(left_pixel_start + c, right_pixel_start + c);
            }
        }
    }
}

/// Produce a randomly positioned crop from the input image after applying zero-padding.
///
/// The input image is interpreted as a flat, pixel-interleaved buffer (row-major) with
/// `width * height * channels` elements. The function pads the image with zeros on all
/// sides by `padding` pixels, selects a top-left corner uniformly at random (using `rng`)
/// such that a `crop_width x crop_height` window fits inside the padded image, and returns
/// the cropped window as a newly allocated `Vec<f32>` in the same interleaved layout.
///
/// # Panics
///
/// * If `image.len() != width * height * channels`.
/// * If `crop_width > width + 2 * padding` or `crop_height > height + 2 * padding`.
///
/// # Returns
///
/// A `Vec<f32>` containing the cropped pixel data. Its length is `crop_width * crop_height * channels`.
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
    let mut cropped = vec![0.0f32; crop_width * crop_height * channels];
    random_crop_into(
        image,
        width,
        height,
        channels,
        padding,
        crop_width,
        crop_height,
        rng,
        &mut cropped,
    );
    cropped
}

#[allow(clippy::too_many_arguments)]
pub fn random_crop_into(
    image: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    padding: usize,
    crop_width: usize,
    crop_height: usize,
    rng: &mut SimpleRng,
    output: &mut [f32],
) {
    assert_eq!(
        image.len(),
        width * height * channels,
        "Image buffer size mismatch: expected {} bytes, got {}",
        width * height * channels,
        image.len()
    );

    let padded_width = width + 2 * padding;
    let padded_height = height + 2 * padding;

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
    assert_eq!(
        output.len(),
        crop_width * crop_height * channels,
        "Crop output buffer size mismatch: expected {} elements, got {}",
        crop_width * crop_height * channels,
        output.len()
    );

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

    output.fill(0.0);
    for row in 0..crop_height {
        for col in 0..crop_width {
            let padded_row = crop_y + row;
            let padded_col = crop_x + col;
            let dst_idx = (row * crop_width + col) * channels;

            if padded_row >= padding
                && padded_row < padding + height
                && padded_col >= padding
                && padded_col < padding + width
            {
                let src_row = padded_row - padding;
                let src_col = padded_col - padding;
                let src_idx = (src_row * width + src_col) * channels;
                output[dst_idx..dst_idx + channels]
                    .copy_from_slice(&image[src_idx..src_idx + channels]);
            }
        }
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

        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                for c in 0..channels {
                    image[row * width * channels + col * channels + c] = col as f32;
                }
            }
        }

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

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

        random_horizontal_flip(&mut image, width, height, channels, 0.0, &mut rng);

        assert_eq!(image, original);
    }

    #[test]
    #[should_panic(expected = "Horizontal flip probability must be finite")]
    fn test_horizontal_flip_rejects_probability_out_of_range() {
        let mut rng = SimpleRng::new(123);
        let mut image = vec![0.5f32; 4 * 2 * 3];

        random_horizontal_flip(&mut image, 4, 2, 3, 1.5, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Horizontal flip probability must be finite")]
    fn test_horizontal_flip_rejects_nan_probability() {
        let mut rng = SimpleRng::new(123);
        let mut image = vec![0.5f32; 4 * 2 * 3];

        random_horizontal_flip(&mut image, 4, 2, 3, f32::NAN, &mut rng);
    }

    #[test]
    fn test_horizontal_flip_single_channel() {
        let mut rng = SimpleRng::new(999);
        let width = 3;
        let height = 1;
        let channels = 1;

        let mut image = vec![0.1, 0.2, 0.3];

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        assert_eq!(image, vec![0.3, 0.2, 0.1]);
    }

    #[test]
    fn test_horizontal_flip_cifar10_size() {
        let mut rng = SimpleRng::new(777);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        image[0] = 1.0;
        image[1] = 1.0;
        image[2] = 1.0;

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        let top_right_idx = (width - 1) * channels;
        assert_eq!(image[top_right_idx], 1.0);
        assert_eq!(image[top_right_idx + 1], 1.0);
        assert_eq!(image[top_right_idx + 2], 1.0);
        assert_eq!(image[0], 0.0);
        assert_eq!(image[1], 0.0);
        assert_eq!(image[2], 0.0);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_horizontal_flip_invalid_buffer_size() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.0f32; 100];
        random_horizontal_flip(&mut image, 32, 32, 3, 1.0, &mut rng);
    }

    #[test]
    fn test_random_crop_basic() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                let value = (row * width + col) as f32;
                for c in 0..channels {
                    image[(row * width + col) * channels + c] = value;
                }
            }
        }

        let cropped = random_crop(&image, width, height, channels, 0, 2, 2, &mut rng);

        assert_eq!(cropped.len(), 2 * 2 * channels);
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

        let image = vec![0.7f32; width * height * channels];
        let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);

        assert_eq!(cropped.len(), 32 * 32 * 3);
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

        let image = vec![1.0f32; width * height * channels];
        let cropped = random_crop(&image, width, height, channels, 2, 6, 6, &mut rng);

        assert_eq!(cropped.len(), 6 * 6 * 3);

        let zero_count = cropped.iter().filter(|&&v| v == 0.0).count();
        let one_count = cropped.iter().filter(|&&v| v == 1.0).count();

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

        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        let crop1 = random_crop(&image, width, height, channels, 2, 4, 4, &mut rng1);
        let crop2 = random_crop(&image, width, height, channels, 2, 4, 4, &mut rng2);

        assert_eq!(crop1, crop2);
    }

    #[test]
    fn test_random_crop_into_matches_random_crop() {
        let width = 8;
        let height = 8;
        let channels = 3;
        let image = vec![0.5f32; width * height * channels];

        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        let crop = random_crop(&image, width, height, channels, 2, 4, 4, &mut rng1);
        let mut output = vec![0.0f32; 4 * 4 * channels];
        random_crop_into(
            &image,
            width,
            height,
            channels,
            2,
            4,
            4,
            &mut rng2,
            &mut output,
        );

        assert_eq!(output, crop);
    }

    #[test]
    fn test_random_crop_no_padding_full_image() {
        let mut rng = SimpleRng::new(111);
        let width = 3;
        let height = 3;
        let channels = 1;

        let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let cropped = random_crop(&image, width, height, channels, 0, 3, 3, &mut rng);

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

        assert_eq!(cropped.len(), 3 * 3);
    }

    #[test]
    fn test_random_crop_cifar10_size() {
        let mut rng = SimpleRng::new(333);
        let width = 32;
        let height = 32;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];
        let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);

        assert_eq!(cropped.len(), 32 * 32 * 3);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_random_crop_invalid_input_size() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 100];
        random_crop(&image, 32, 32, 3, 4, 32, 32, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Crop width")]
    fn test_random_crop_crop_too_large() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 4 * 4 * 3];
        random_crop(&image, 4, 4, 3, 1, 10, 10, &mut rng);
    }

    #[test]
    fn test_random_crop_edge_case_zero_padding() {
        let mut rng = SimpleRng::new(444);
        let width = 5;
        let height = 5;
        let channels = 3;

        let image = vec![0.8f32; width * height * channels];
        let cropped = random_crop(&image, width, height, channels, 0, 3, 3, &mut rng);

        assert_eq!(cropped.len(), 3 * 3 * 3);
        for &val in &cropped {
            assert_eq!(val, 0.8);
        }
    }
}
