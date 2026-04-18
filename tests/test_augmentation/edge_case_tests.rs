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
