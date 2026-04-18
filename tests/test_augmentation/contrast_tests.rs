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
