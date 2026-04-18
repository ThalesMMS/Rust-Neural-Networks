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
