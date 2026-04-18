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
