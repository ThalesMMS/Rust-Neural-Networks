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
