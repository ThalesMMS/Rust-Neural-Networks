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
