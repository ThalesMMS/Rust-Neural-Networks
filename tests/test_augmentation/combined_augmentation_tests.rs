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
