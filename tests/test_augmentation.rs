//! Comprehensive tests for image data augmentation functions
//!
//! This file tests the augmentation module including:
//! - Random horizontal flip
//! - Random crop with padding
//! - Random brightness adjustment
//! - Random contrast adjustment
//! - Random saturation adjustment
//! - Combination of multiple augmentations
//! - Edge cases and error conditions
//!
//! Tests verify that augmentations:
//! - Preserve valid pixel ranges [0.0, 1.0]
//! - Are deterministic with same RNG seed
//! - Handle edge cases correctly
//! - Validate input dimensions properly

use rust_neural_networks::data::augmentation::{
    random_brightness, random_contrast, random_crop, random_horizontal_flip, random_saturation,
};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Horizontal Flip Tests
// ============================================================================

mod horizontal_flip_tests {
    use super::*;

    #[test]
    fn test_flip_always_with_probability_one() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 2;
        let channels = 3;

        // Create distinct pattern: pixel value = column index
        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                let value = col as f32 / 10.0;
                for c in 0..channels {
                    image[row * width * channels + col * channels + c] = value;
                }
            }
        }

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // Verify columns are reversed
        for row in 0..height {
            for col in 0..width {
                let expected = (width - 1 - col) as f32 / 10.0;
                for c in 0..channels {
                    let idx = row * width * channels + col * channels + c;
                    assert!(
                        (image[idx] - expected).abs() < 1e-6,
                        "Mismatch at row={}, col={}, channel={}",
                        row,
                        col,
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_flip_never_with_probability_zero() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 100) as f32 / 100.0;
        }
        let original = image.clone();

        random_horizontal_flip(&mut image, width, height, channels, 0.0, &mut rng);

        assert_eq!(
            image, original,
            "Image should be unchanged with probability 0.0"
        );
    }

    #[test]
    fn test_flip_is_reversible() {
        let mut rng = SimpleRng::new(999);
        let width = 6;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 50) as f32 / 50.0;
        }
        let original = image.clone();

        // Flip twice should restore original
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        assert_eq!(image, original, "Double flip should restore original image");
    }

    #[test]
    fn test_flip_single_column() {
        let mut rng = SimpleRng::new(111);
        let width = 1;
        let height = 5;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        let original = image.clone();

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // Single column should be unchanged
        assert_eq!(image, original);
    }

    #[test]
    fn test_flip_preserves_pixel_range() {
        let mut rng = SimpleRng::new(222);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 256) as f32 / 256.0;
        }

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // All values should remain in valid range
        for &val in &image {
            assert!(
                (0.0..=1.0).contains(&val),
                "Pixel value {} outside valid range [0.0, 1.0]",
                val
            );
        }
    }

    #[test]
    fn test_flip_deterministic_same_seed() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.0f32; width * height * channels];
        let mut image2 = vec![0.0f32; width * height * channels];
        for i in 0..image1.len() {
            let val = (i % 128) as f32 / 128.0;
            image1[i] = val;
            image2[i] = val;
        }

        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        random_horizontal_flip(&mut image1, width, height, channels, 0.5, &mut rng1);
        random_horizontal_flip(&mut image2, width, height, channels, 0.5, &mut rng2);

        assert_eq!(image1, image2, "Same seed should produce identical results");
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_flip_invalid_buffer_size() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.0f32; 100];
        random_horizontal_flip(&mut image, 32, 32, 3, 1.0, &mut rng);
    }

    #[test]
    fn test_flip_odd_width() {
        let mut rng = SimpleRng::new(333);
        let width = 5;
        let height = 3;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for col in 0..width {
            for c in 0..channels {
                image[col * channels + c] = col as f32;
            }
        }

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);

        // Middle column (index 2) should stay in place
        let middle = 2;
        for c in 0..channels {
            assert_eq!(image[middle * channels + c], middle as f32);
        }
    }
}

// ============================================================================
// Random Crop Tests
// ============================================================================

mod random_crop_tests {
    use super::*;

    #[test]
    fn test_crop_output_dimensions() {
        let mut rng = SimpleRng::new(42);
        let width = 32;
        let height = 32;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];

        let cropped = random_crop(&image, width, height, channels, 4, 28, 28, &mut rng);

        assert_eq!(
            cropped.len(),
            28 * 28 * 3,
            "Cropped image should have correct dimensions"
        );
    }

    #[test]
    fn test_crop_no_padding_extracts_subregion() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        // Create image with unique values at each pixel
        let mut image = vec![0.0f32; width * height * channels];
        for row in 0..height {
            for col in 0..width {
                let value = (row * width + col) as f32;
                for c in 0..channels {
                    image[(row * width + col) * channels + c] = value;
                }
            }
        }

        let cropped = random_crop(&image, width, height, channels, 0, 4, 4, &mut rng);

        assert_eq!(cropped.len(), 4 * 4 * 3);

        // All pixels in crop should have values from original image
        for i in 0..(4 * 4) {
            let r = cropped[i * channels];
            let g = cropped[i * channels + 1];
            let b = cropped[i * channels + 2];
            assert_eq!(r, g);
            assert_eq!(g, b);
            assert!(r >= 0.0 && r < (width * height) as f32);
        }
    }

    #[test]
    fn test_crop_with_padding_includes_zeros() {
        let mut rng = SimpleRng::new(999);
        let width = 4;
        let height = 4;
        let channels = 3;

        // All ones in original image
        let image = vec![1.0f32; width * height * channels];

        // Pad heavily and crop to padded size
        let cropped = random_crop(&image, width, height, channels, 4, 12, 12, &mut rng);

        assert_eq!(cropped.len(), 12 * 12 * 3);

        // Should contain both zeros (padding) and ones (original)
        let has_zeros = cropped.contains(&0.0);
        let has_ones = cropped.contains(&1.0);

        assert!(has_zeros, "Should contain padding zeros");
        assert!(has_ones, "Should contain original ones");
    }

    #[test]
    fn test_crop_full_image_no_padding() {
        let mut rng = SimpleRng::new(111);
        let width = 5;
        let height = 5;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 25) as f32 / 25.0;
        }

        // Crop entire image with no padding
        let cropped = random_crop(&image, width, height, channels, 0, width, height, &mut rng);

        assert_eq!(
            cropped, image,
            "Full crop with no padding should match original"
        );
    }

    #[test]
    fn test_crop_preserves_pixel_range() {
        let mut rng = SimpleRng::new(222);
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 200) as f32 / 200.0;
        }

        let cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);

        for &val in &cropped {
            assert!(
                (0.0..=1.0).contains(&val),
                "Pixel value {} outside valid range [0.0, 1.0]",
                val
            );
        }
    }

    #[test]
    fn test_crop_deterministic_same_seed() {
        let width = 32;
        let height = 32;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];

        let mut rng1 = SimpleRng::new(555);
        let mut rng2 = SimpleRng::new(555);

        let crop1 = random_crop(&image, width, height, channels, 4, 28, 28, &mut rng1);
        let crop2 = random_crop(&image, width, height, channels, 4, 28, 28, &mut rng2);

        assert_eq!(crop1, crop2, "Same seed should produce identical crops");
    }

    #[test]
    fn test_crop_different_seeds_may_differ() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 100) as f32 / 100.0;
        }

        let mut rng1 = SimpleRng::new(100);
        let mut rng2 = SimpleRng::new(200);

        let crop1 = random_crop(&image, width, height, channels, 4, 12, 12, &mut rng1);
        let crop2 = random_crop(&image, width, height, channels, 4, 12, 12, &mut rng2);

        // Different seeds may (but not guaranteed to) produce different crops
        // Just verify both are valid
        assert_eq!(crop1.len(), 12 * 12 * 3);
        assert_eq!(crop2.len(), 12 * 12 * 3);
    }

    #[test]
    fn test_crop_single_channel() {
        let mut rng = SimpleRng::new(333);
        let width = 8;
        let height = 8;
        let channels = 1;

        let image = vec![0.7f32; width * height * channels];

        let cropped = random_crop(&image, width, height, channels, 2, 6, 6, &mut rng);

        assert_eq!(cropped.len(), (6 * 6));
    }

    #[test]
    fn test_crop_rectangular_image() {
        let mut rng = SimpleRng::new(444);
        let width = 16;
        let height = 8;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];

        let cropped = random_crop(&image, width, height, channels, 2, 12, 6, &mut rng);

        assert_eq!(cropped.len(), 12 * 6 * 3);
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_crop_invalid_input_size() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 100];
        random_crop(&image, 32, 32, 3, 4, 28, 28, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Crop width")]
    fn test_crop_too_large_width() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 8 * 8 * 3];
        random_crop(&image, 8, 8, 3, 1, 20, 8, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Crop height")]
    fn test_crop_too_large_height() {
        let mut rng = SimpleRng::new(1);
        let image = vec![0.0f32; 8 * 8 * 3];
        random_crop(&image, 8, 8, 3, 1, 8, 20, &mut rng);
    }
}

// ============================================================================
// Brightness Adjustment Tests
// ============================================================================

mod brightness_tests {
    use super::*;

    #[test]
    fn test_brightness_modifies_values() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        let original = image.clone();

        random_brightness(&mut image, width, height, channels, 0.3, &mut rng);

        // Values should have changed
        assert_ne!(image, original);
    }

    #[test]
    fn test_brightness_clamps_to_valid_range() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        // Start near boundaries
        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = if i % 2 == 0 { 0.05 } else { 0.95 };
        }

        // Large delta could push values outside [0.0, 1.0]
        random_brightness(&mut image, width, height, channels, 0.5, &mut rng);

        // All values should be clamped
        for &val in &image {
            assert!(
                (0.0..=1.0).contains(&val),
                "Brightness should clamp to [0.0, 1.0], got {}",
                val
            );
        }
    }

    #[test]
    fn test_brightness_uniform_adjustment() {
        let mut rng = SimpleRng::new(999);
        let width = 4;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        // All pixels should have same value (uniform adjustment)
        let first_value = image[0];
        for &val in &image {
            assert_eq!(
                val, first_value,
                "All pixels should receive same brightness delta"
            );
        }
    }

    #[test]
    fn test_brightness_deterministic_same_seed() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.5f32; width * height * channels];
        let mut image2 = vec![0.5f32; width * height * channels];

        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        random_brightness(&mut image1, width, height, channels, 0.3, &mut rng1);
        random_brightness(&mut image2, width, height, channels, 0.3, &mut rng2);

        assert_eq!(image1, image2, "Same seed should produce identical results");
    }

    #[test]
    fn test_brightness_single_channel() {
        let mut rng = SimpleRng::new(111);
        let width = 8;
        let height = 8;
        let channels = 1;

        let mut image = vec![0.5f32; width * height * channels];

        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_brightness_zero_delta() {
        let mut rng = SimpleRng::new(222);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 50) as f32 / 50.0;
        }
        let original = image.clone();

        // Zero max_delta means no change
        random_brightness(&mut image, width, height, channels, 0.0, &mut rng);

        assert_eq!(image, original, "Zero delta should not modify image");
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_brightness_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100];
        random_brightness(&mut image, 32, 32, 3, 0.2, &mut rng);
    }
}

// ============================================================================
// Contrast Adjustment Tests
// ============================================================================

mod contrast_tests {
    use super::*;

    #[test]
    fn test_contrast_modifies_values() {
        let mut rng = SimpleRng::new(42);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 20) as f32 / 20.0;
        }
        let original = image.clone();

        random_contrast(&mut image, width, height, channels, 0.3, &mut rng);

        // Values should have changed
        assert_ne!(image, original);
    }

    #[test]
    fn test_contrast_clamps_to_valid_range() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 10) as f32 / 10.0;
        }

        // High contrast factor could push values outside [0.0, 1.0]
        random_contrast(&mut image, width, height, channels, 0.9, &mut rng);

        for &val in &image {
            assert!(
                (0.0..=1.0).contains(&val),
                "Contrast should clamp to [0.0, 1.0], got {}",
                val
            );
        }
    }

    #[test]
    fn test_contrast_preserves_mean_approximately() {
        let mut rng = SimpleRng::new(999);
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 30) as f32 / 30.0;
        }

        let original_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        random_contrast(&mut image, width, height, channels, 0.2, &mut rng);

        let new_mean: f32 = image.iter().sum::<f32>() / image.len() as f32;

        // Mean should be approximately preserved (tolerance for clamping)
        assert!(
            (original_mean - new_mean).abs() < 0.15,
            "Mean should be approximately preserved: {} vs {}",
            original_mean,
            new_mean
        );
    }

    #[test]
    fn test_contrast_uniform_image_unchanged() {
        let mut rng = SimpleRng::new(111);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];
        let original = image.clone();

        // Contrast around mean has no effect on uniform image
        random_contrast(&mut image, width, height, channels, 0.5, &mut rng);

        assert_eq!(
            image, original,
            "Uniform image should be unchanged by contrast adjustment"
        );
    }

    #[test]
    fn test_contrast_deterministic_same_seed() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.0f32; width * height * channels];
        let mut image2 = vec![0.0f32; width * height * channels];
        for i in 0..image1.len() {
            let val = (i % 25) as f32 / 25.0;
            image1[i] = val;
            image2[i] = val;
        }

        let mut rng1 = SimpleRng::new(555);
        let mut rng2 = SimpleRng::new(555);

        random_contrast(&mut image1, width, height, channels, 0.4, &mut rng1);
        random_contrast(&mut image2, width, height, channels, 0.4, &mut rng2);

        assert_eq!(image1, image2, "Same seed should produce identical results");
    }

    #[test]
    fn test_contrast_zero_delta() {
        let mut rng = SimpleRng::new(222);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 40) as f32 / 40.0;
        }
        let original = image.clone();

        // Zero delta means factor = 1.0, minimal change (floating point precision)
        random_contrast(&mut image, width, height, channels, 0.0, &mut rng);

        // Check approximate equality (tolerance for floating point precision)
        for (i, (&new_val, &orig_val)) in image.iter().zip(original.iter()).enumerate() {
            assert!(
                (new_val - orig_val).abs() < 1e-6,
                "Value at index {} differs too much: {} vs {}",
                i,
                new_val,
                orig_val
            );
        }
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_contrast_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100];
        random_contrast(&mut image, 32, 32, 3, 0.2, &mut rng);
    }
}

// ============================================================================
// Saturation Adjustment Tests
// ============================================================================

mod saturation_tests {
    use super::*;

    #[test]
    fn test_saturation_modifies_color_values() {
        let mut rng = SimpleRng::new(42);
        let width = 4;
        let height = 4;
        let channels = 3;

        // Create colorful image (not grayscale)
        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            image[i * channels] = 0.8; // R
            image[i * channels + 1] = 0.3; // G
            image[i * channels + 2] = 0.5; // B
        }
        let original = image.clone();

        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

        // Values should have changed
        assert_ne!(image, original);
    }

    #[test]
    fn test_saturation_clamps_to_valid_range() {
        let mut rng = SimpleRng::new(123);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            image[i * channels] = 1.0; // R
            image[i * channels + 1] = 0.0; // G
            image[i * channels + 2] = 0.0; // B
        }

        // High saturation could push values outside [0.0, 1.0]
        random_saturation(&mut image, width, height, channels, 0.9, &mut rng);

        for &val in &image {
            assert!(
                (0.0..=1.0).contains(&val),
                "Saturation should clamp to [0.0, 1.0], got {}",
                val
            );
        }
    }

    #[test]
    fn test_saturation_preserves_grayscale() {
        let mut rng = SimpleRng::new(999);
        let width = 6;
        let height = 6;
        let channels = 3;

        // Create grayscale image (R=G=B)
        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            let gray = 0.6;
            image[i * channels] = gray;
            image[i * channels + 1] = gray;
            image[i * channels + 2] = gray;
        }

        random_saturation(&mut image, width, height, channels, 0.5, &mut rng);

        // Grayscale should remain grayscale
        for i in 0..(width * height) {
            let r = image[i * channels];
            let g = image[i * channels + 1];
            let b = image[i * channels + 2];
            assert!(
                (r - g).abs() < 1e-6 && (g - b).abs() < 1e-6,
                "Grayscale pixel should remain grayscale"
            );
        }
    }

    #[test]
    fn test_saturation_luminance_weights() {
        let mut rng = SimpleRng::new(111);
        let width = 2;
        let height = 2;
        let channels = 3;

        // Pure red pixel
        let mut image = vec![0.0f32; width * height * channels];
        image[0] = 1.0; // R
        image[1] = 0.0; // G
        image[2] = 0.0; // B

        random_saturation(&mut image, width, height, channels, 0.3, &mut rng);

        // After saturation adjustment, values should still be valid
        for item in image.iter().take(3) {
            assert!((0.0..=1.0).contains(item));
        }
    }

    #[test]
    fn test_saturation_deterministic_same_seed() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let mut image1 = vec![0.0f32; width * height * channels];
        let mut image2 = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            image1[i * channels] = 0.7;
            image1[i * channels + 1] = 0.4;
            image1[i * channels + 2] = 0.6;
            image2[i * channels] = 0.7;
            image2[i * channels + 1] = 0.4;
            image2[i * channels + 2] = 0.6;
        }

        let mut rng1 = SimpleRng::new(777);
        let mut rng2 = SimpleRng::new(777);

        random_saturation(&mut image1, width, height, channels, 0.4, &mut rng1);
        random_saturation(&mut image2, width, height, channels, 0.4, &mut rng2);

        assert_eq!(image1, image2, "Same seed should produce identical results");
    }

    #[test]
    fn test_saturation_zero_delta() {
        let mut rng = SimpleRng::new(222);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for i in 0..(width * height) {
            image[i * channels] = 0.8;
            image[i * channels + 1] = 0.3;
            image[i * channels + 2] = 0.5;
        }
        let original = image.clone();

        // Zero delta means factor = 1.0, no change
        random_saturation(&mut image, width, height, channels, 0.0, &mut rng);

        assert_eq!(image, original, "Zero delta should not modify image");
    }

    #[test]
    #[should_panic(expected = "Image buffer size mismatch")]
    fn test_saturation_invalid_buffer() {
        let mut rng = SimpleRng::new(1);
        let mut image = vec![0.5f32; 100];
        random_saturation(&mut image, 32, 32, 3, 0.2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "Saturation adjustment requires 3 channels")]
    fn test_saturation_requires_rgb() {
        let mut rng = SimpleRng::new(1);
        let width = 8;
        let height = 8;
        let channels = 1; // Single channel not supported

        let mut image = vec![0.5f32; width * height * channels];
        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);
    }
}

// ============================================================================
// Combined Augmentation Tests
// ============================================================================

mod combined_augmentation_tests {
    use super::*;

    #[test]
    fn test_flip_then_brightness() {
        let mut rng = SimpleRng::new(42);
        let width = 8;
        let height = 8;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        // All values should remain valid
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_crop_then_contrast() {
        let mut rng = SimpleRng::new(123);
        let width = 32;
        let height = 32;
        let channels = 3;

        let image = vec![0.6f32; width * height * channels];

        let mut cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);
        random_contrast(&mut cropped, 32, 32, channels, 0.3, &mut rng);

        // All values should remain valid
        for &val in &cropped {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_full_augmentation_pipeline() {
        let mut rng = SimpleRng::new(999);
        let width = 32;
        let height = 32;
        let channels = 3;

        // Start with varied image
        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 100) as f32 / 100.0;
        }

        // Apply all augmentations in sequence
        random_horizontal_flip(&mut image, width, height, channels, 0.5, &mut rng);
        let mut cropped = random_crop(&image, width, height, channels, 4, 32, 32, &mut rng);
        random_brightness(&mut cropped, 32, 32, channels, 0.2, &mut rng);
        random_contrast(&mut cropped, 32, 32, channels, 0.2, &mut rng);
        random_saturation(&mut cropped, 32, 32, channels, 0.2, &mut rng);

        // Verify output is valid
        assert_eq!(cropped.len(), 32 * 32 * 3);
        for &val in &cropped {
            assert!(
                (0.0..=1.0).contains(&val),
                "Pipeline output should be in valid range, got {}",
                val
            );
        }
    }

    #[test]
    fn test_deterministic_pipeline_same_seed() {
        let width = 16;
        let height = 16;
        let channels = 3;

        let image = vec![0.5f32; width * height * channels];

        // Pipeline 1
        let mut rng1 = SimpleRng::new(777);
        let mut img1 = image.clone();
        random_horizontal_flip(&mut img1, width, height, channels, 0.5, &mut rng1);
        random_brightness(&mut img1, width, height, channels, 0.2, &mut rng1);
        random_contrast(&mut img1, width, height, channels, 0.2, &mut rng1);

        // Pipeline 2 (same seed)
        let mut rng2 = SimpleRng::new(777);
        let mut img2 = image.clone();
        random_horizontal_flip(&mut img2, width, height, channels, 0.5, &mut rng2);
        random_brightness(&mut img2, width, height, channels, 0.2, &mut rng2);
        random_contrast(&mut img2, width, height, channels, 0.2, &mut rng2);

        assert_eq!(
            img1, img2,
            "Same seed should produce identical pipeline results"
        );
    }

    #[test]
    fn test_multiple_crops_different_results() {
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = (i % 150) as f32 / 150.0;
        }

        let mut rng = SimpleRng::new(111);

        // Multiple crops should potentially differ
        let crop1 = random_crop(&image, width, height, channels, 4, 28, 28, &mut rng);
        let crop2 = random_crop(&image, width, height, channels, 4, 28, 28, &mut rng);

        // Both should be valid
        assert_eq!(crop1.len(), 28 * 28 * 3);
        assert_eq!(crop2.len(), 28 * 28 * 3);
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_minimal_image_1x1() {
        let mut rng = SimpleRng::new(42);
        let width = 1;
        let height = 1;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        // All augmentations should handle 1x1 images
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);
        random_contrast(&mut image, width, height, channels, 0.2, &mut rng);
        random_saturation(&mut image, width, height, channels, 0.2, &mut rng);

        assert_eq!(image.len(), 3);
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_single_row_image() {
        let mut rng = SimpleRng::new(123);
        let width = 16;
        let height = 1;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        assert_eq!(image.len(), 16 * 3);
    }

    #[test]
    fn test_single_column_image() {
        let mut rng = SimpleRng::new(999);
        let width = 1;
        let height = 16;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_brightness(&mut image, width, height, channels, 0.2, &mut rng);

        assert_eq!(image.len(), 16 * 3);
    }

    #[test]
    fn test_large_image() {
        let mut rng = SimpleRng::new(111);
        let width = 128;
        let height = 128;
        let channels = 3;

        let mut image = vec![0.5f32; width * height * channels];

        // Should handle large images efficiently
        random_horizontal_flip(&mut image, width, height, channels, 1.0, &mut rng);
        random_brightness(&mut image, width, height, channels, 0.1, &mut rng);

        assert_eq!(image.len(), 128 * 128 * 3);
    }

    #[test]
    fn test_extreme_brightness_values() {
        let mut rng = SimpleRng::new(222);
        let width = 4;
        let height = 4;
        let channels = 3;

        // Test with values at boundaries
        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = if i % 2 == 0 { 0.0 } else { 1.0 };
        }

        random_brightness(&mut image, width, height, channels, 1.0, &mut rng);

        // Should clamp properly
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_extreme_contrast_values() {
        let mut rng = SimpleRng::new(333);
        let width = 4;
        let height = 4;
        let channels = 3;

        let mut image = vec![0.0f32; width * height * channels];
        for (i, pixel) in image.iter_mut().enumerate() {
            *pixel = if i % 2 == 0 { 0.0 } else { 1.0 };
        }

        random_contrast(&mut image, width, height, channels, 0.99, &mut rng);

        // Should clamp properly
        for &val in &image {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
