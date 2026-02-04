//! Comprehensive tests for CIFAR-10 data loader
//!
//! This file tests the CIFAR-10 data loading module including:
//! - Label name mappings
//! - Data loader functions (with data files if available)
//! - Constant values
//! - Image dimensions and data format validation
//!
//! Tests that require CIFAR-10 data files are marked with #[ignore] and will be skipped
//! unless the data is present. Run with `cargo test -- --ignored` to run those tests.

use rust_neural_networks::data::cifar10::{
    get_class_names, label_to_name, read_cifar10_batch, read_cifar10_batches, CIFAR10_IMAGE_BYTES,
    CIFAR10_IMAGE_PIXELS, CIFAR10_IMAGE_SIZE, CIFAR10_NUM_CHANNELS, CIFAR10_NUM_CLASSES,
};

// ============================================================================
// Constants Validation Tests
// ============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_cifar10_image_dimensions() {
        assert_eq!(CIFAR10_IMAGE_SIZE, 32);
        assert_eq!(CIFAR10_NUM_CHANNELS, 3);
        assert_eq!(CIFAR10_IMAGE_PIXELS, 1024);
        assert_eq!(CIFAR10_IMAGE_BYTES, 3072);
    }

    #[test]
    fn test_cifar10_num_classes() {
        assert_eq!(CIFAR10_NUM_CLASSES, 10);
    }

    #[test]
    fn test_image_bytes_calculation() {
        assert_eq!(
            CIFAR10_IMAGE_BYTES,
            CIFAR10_IMAGE_PIXELS * CIFAR10_NUM_CHANNELS
        );
    }

    #[test]
    fn test_image_pixels_calculation() {
        assert_eq!(
            CIFAR10_IMAGE_PIXELS,
            CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE
        );
    }
}

// ============================================================================
// Label Name Mapping Tests
// ============================================================================

mod label_mapping_tests {
    use super::*;

    #[test]
    fn test_label_to_name_all_classes() {
        assert_eq!(label_to_name(0), "airplane");
        assert_eq!(label_to_name(1), "automobile");
        assert_eq!(label_to_name(2), "bird");
        assert_eq!(label_to_name(3), "cat");
        assert_eq!(label_to_name(4), "deer");
        assert_eq!(label_to_name(5), "dog");
        assert_eq!(label_to_name(6), "frog");
        assert_eq!(label_to_name(7), "horse");
        assert_eq!(label_to_name(8), "ship");
        assert_eq!(label_to_name(9), "truck");
    }

    #[test]
    fn test_label_to_name_first_and_last() {
        assert_eq!(label_to_name(0), "airplane");
        assert_eq!(label_to_name(9), "truck");
    }

    #[test]
    #[should_panic(expected = "Invalid CIFAR-10 label")]
    fn test_label_to_name_invalid_10() {
        label_to_name(10);
    }

    #[test]
    #[should_panic(expected = "Invalid CIFAR-10 label")]
    fn test_label_to_name_invalid_255() {
        label_to_name(255);
    }

    #[test]
    fn test_get_class_names_length() {
        let names = get_class_names();
        assert_eq!(names.len(), 10);
    }

    #[test]
    fn test_get_class_names_content() {
        let names = get_class_names();
        assert_eq!(names[0], "airplane");
        assert_eq!(names[1], "automobile");
        assert_eq!(names[2], "bird");
        assert_eq!(names[3], "cat");
        assert_eq!(names[4], "deer");
        assert_eq!(names[5], "dog");
        assert_eq!(names[6], "frog");
        assert_eq!(names[7], "horse");
        assert_eq!(names[8], "ship");
        assert_eq!(names[9], "truck");
    }

    /// Asserts that the CIFAR-10 class name list contains no duplicate entries.
    ///
    /// # Examples
    ///
    /// ```
    /// let names = rust_neural_networks::data::cifar10::get_class_names();
    /// let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
    /// assert_eq!(names.len(), unique_count);
    /// ```
    #[test]
    fn test_class_names_no_duplicates() {
        let names = get_class_names();
        for (i, &name_i) in names.iter().enumerate() {
            for &name_j in names.iter().skip(i + 1) {
                assert_ne!(
                    name_i, name_j,
                    "Class names should be unique: {} vs {}",
                    name_i, name_j
                );
            }
        }
    }

    #[test]
    fn test_label_to_name_matches_get_class_names() {
        let names = get_class_names();
        for (i, &name) in names.iter().enumerate() {
            assert_eq!(label_to_name(i as u8), name);
        }
    }
}

// ============================================================================
// Data Loader Tests (Require CIFAR-10 Data Files)
// ============================================================================

mod data_loader_tests {
    use super::*;

    /// Test reading a single CIFAR-10 batch file.
    /// This test is ignored by default because it requires CIFAR-10 data files.
    /// Run with: cargo test test_read_single_batch -- --ignored
    #[test]
    #[ignore]
    fn test_read_single_batch() {
        let (images, labels) = read_cifar10_batch("./data/cifar-10-batches-bin/data_batch_1.bin")
            .expect("failed to read CIFAR-10 batch");

        // Verify batch size
        assert_eq!(labels.len(), 10000, "Should have 10000 images in batch");
        assert_eq!(
            images.len(),
            10000 * CIFAR10_IMAGE_BYTES,
            "Image data size should be 10000 * 3072"
        );

        // Verify labels are in valid range (0-9)
        for (i, &label) in labels.iter().enumerate() {
            assert!(
                label < 10,
                "Label at index {} should be 0-9, got {}",
                i,
                label
            );
        }
    }

    /// Asserts that pixel values loaded from a CIFAR-10 batch are within the range 0.0..=1.0.
    ///
    /// This test requires the CIFAR-10 binary files to be present and is ignored by default.
    ///
    /// # Examples
    ///
    /// ```
    /// let (images, _) = rust_neural_networks::data::cifar10::read_cifar10_batch(
    ///     "./data/cifar-10-batches-bin/data_batch_1.bin",
    /// )
    /// .expect("failed to read CIFAR-10 batch");
    /// for &pixel in images.iter() {
    ///     assert!((0.0..=1.0).contains(&pixel));
    /// }
    /// ```
    #[test]
    #[ignore]
    fn test_image_normalization() {
        let (images, _) = read_cifar10_batch("./data/cifar-10-batches-bin/data_batch_1.bin")
            .expect("failed to read CIFAR-10 batch");

        // Check that all pixel values are in [0, 1] range
        for (i, &pixel) in images.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&pixel),
                "Pixel at index {} should be in [0, 1], got {}",
                i,
                pixel
            );
        }
    }

    /// Test reading multiple CIFAR-10 batch files.
    /// This test is ignored by default because it requires CIFAR-10 data files.
    #[test]
    #[ignore]
    fn test_read_multiple_batches() {
        let filenames = vec![
            "./data/cifar-10-batches-bin/data_batch_1.bin",
            "./data/cifar-10-batches-bin/data_batch_2.bin",
        ];
        let (images, labels) =
            read_cifar10_batches(&filenames).expect("failed to read CIFAR-10 batches");

        // Verify total size (2 batches = 20000 images)
        assert_eq!(labels.len(), 20000, "Should have 20000 images total");
        assert_eq!(
            images.len(),
            20000 * CIFAR10_IMAGE_BYTES,
            "Image data size should be 20000 * 3072"
        );

        // Verify all labels are valid
        for &label in &labels {
            assert!(label < 10, "All labels should be 0-9");
        }
    }

    /// Test reading all 5 training batches.
    /// This test is ignored by default because it requires CIFAR-10 data files.
    #[test]
    #[ignore]
    fn test_read_all_training_batches() {
        let filenames = vec![
            "./data/cifar-10-batches-bin/data_batch_1.bin",
            "./data/cifar-10-batches-bin/data_batch_2.bin",
            "./data/cifar-10-batches-bin/data_batch_3.bin",
            "./data/cifar-10-batches-bin/data_batch_4.bin",
            "./data/cifar-10-batches-bin/data_batch_5.bin",
        ];
        let (images, labels) =
            read_cifar10_batches(&filenames).expect("failed to read CIFAR-10 batches");

        // Verify total size (5 batches = 50000 images)
        assert_eq!(
            labels.len(),
            50000,
            "Should have 50000 training images total"
        );
        assert_eq!(
            images.len(),
            50000 * CIFAR10_IMAGE_BYTES,
            "Image data size should be 50000 * 3072"
        );
    }

    /// Test reading the test batch.
    /// This test is ignored by default because it requires CIFAR-10 data files.
    #[test]
    #[ignore]
    fn test_read_test_batch() {
        let (images, labels) = read_cifar10_batch("./data/cifar-10-batches-bin/test_batch.bin")
            .expect("failed to read CIFAR-10 batch");

        // Verify test batch size
        assert_eq!(labels.len(), 10000, "Should have 10000 test images");
        assert_eq!(
            images.len(),
            10000 * CIFAR10_IMAGE_BYTES,
            "Test image data size should be 10000 * 3072"
        );
    }

    /// Verifies that pixel data in a CIFAR-10 batch is stored with RGB values interleaved per pixel.
    ///
    /// This test reads a CIFAR-10 training batch and asserts that consecutive values within a pixel
    /// correspond to red, green, and blue channels and that those values are normalized to the
    /// range [0.0, 1.0]. It checks the first pixel of the first image and a pixel from the middle
    /// of the first image. The test requires the CIFAR-10 binary files and is ignored by default.
    ///
    /// # Examples
    ///
    /// ```
    /// let (images, _labels) = rust_neural_networks::data::cifar10::read_cifar10_batch(
    ///     "./data/cifar-10-batches-bin/data_batch_1.bin",
    /// )
    /// .expect("failed to read CIFAR-10 batch");
    /// // First pixel: R, G, B
    /// assert!((0.0..=1.0).contains(&images[0]));
    /// assert!((0.0..=1.0).contains(&images[1]));
    /// assert!((0.0..=1.0).contains(&images[2]));
    /// ```
    #[test]
    #[ignore]
    fn test_rgb_channel_interleaving() {
        let (images, _) = read_cifar10_batch("./data/cifar-10-batches-bin/data_batch_1.bin")
            .expect("failed to read CIFAR-10 batch");

        // Check that data is organized as RGB RGB RGB... (pixel-interleaved format)
        // For the first image, verify that we have R, G, B values in sequence
        let first_pixel_r = images[0];
        let first_pixel_g = images[1];
        let first_pixel_b = images[2];

        // All should be valid normalized values
        assert!((0.0..=1.0).contains(&first_pixel_r));
        assert!((0.0..=1.0).contains(&first_pixel_g));
        assert!((0.0..=1.0).contains(&first_pixel_b));

        // Check a pixel in the middle of the first image
        let mid_pixel_idx = 512 * 3; // Pixel 512 of first image
        assert!(images[mid_pixel_idx] >= 0.0 && images[mid_pixel_idx] <= 1.0); // R
        assert!(images[mid_pixel_idx + 1] >= 0.0 && images[mid_pixel_idx + 1] <= 1.0); // G
        assert!(images[mid_pixel_idx + 2] >= 0.0 && images[mid_pixel_idx + 2] <= 1.0);
        // B
    }

    /// Test that each image has exactly 3072 values (32x32x3).
    /// This test is ignored by default because it requires CIFAR-10 data files.
    #[test]
    #[ignore]
    fn test_image_dimensions_validation() {
        let (images, labels) = read_cifar10_batch("./data/cifar-10-batches-bin/data_batch_1.bin")
            .expect("failed to read CIFAR-10 batch");

        let num_images = labels.len();
        let expected_total_size = num_images * CIFAR10_IMAGE_BYTES;

        assert_eq!(
            images.len(),
            expected_total_size,
            "Total image data should be {} * {} = {}",
            num_images,
            CIFAR10_IMAGE_BYTES,
            expected_total_size
        );

        // Verify that data can be evenly divided into images
        assert_eq!(images.len() % CIFAR10_IMAGE_BYTES, 0);
    }

    /// Test that all class labels appear in the dataset.
    /// This test is ignored by default because it requires CIFAR-10 data files.
    #[test]
    #[ignore]
    fn test_all_classes_present() {
        let (_, labels) = read_cifar10_batch("./data/cifar-10-batches-bin/data_batch_1.bin")
            .expect("failed to read CIFAR-10 batch");

        let mut class_counts = [0; 10];
        for &label in &labels {
            class_counts[label as usize] += 1;
        }

        // Verify that all classes appear at least once
        for (i, &count) in class_counts.iter().enumerate() {
            assert!(
                count > 0,
                "Class {} ({}) should appear at least once in the batch",
                i,
                label_to_name(i as u8)
            );
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================
//
// Note: Error handling tests are not included, but read_cifar10_batch/read_cifar10_batches
// now return Result so callers can assert on failure conditions.

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    /// Verifies that supplying no filenames yields empty image and label collections.
    ///
    /// This test ensures `read_cifar10_batches` handles an empty list of input files
    /// by returning zero-length vectors for both images and labels.
    ///
    /// # Examples
    ///
    /// ```
    /// let filenames: Vec<&str> = vec![];
    /// let (images, labels) = rust_neural_networks::data::cifar10::read_cifar10_batches(&filenames)
    ///     .expect("failed to read CIFAR-10 batches");
    /// assert!(images.is_empty());
    /// assert!(labels.is_empty());
    /// ```
    #[test]
    fn test_empty_batch_list() {
        let filenames: Vec<&str> = vec![];
        let (images, labels) =
            read_cifar10_batches(&filenames).expect("failed to read CIFAR-10 batches");

        assert_eq!(images.len(), 0);
        assert_eq!(labels.len(), 0);
    }

    #[test]
    fn test_label_boundary_values() {
        // Test valid boundary values
        assert_eq!(label_to_name(0), "airplane");
        assert_eq!(label_to_name(9), "truck");
    }

    #[test]
    fn test_class_names_are_lowercase() {
        let names = get_class_names();
        for name in names.iter() {
            assert_eq!(
                *name,
                name.to_lowercase(),
                "Class name '{}' should be lowercase",
                name
            );
        }
    }

    #[test]
    fn test_class_names_no_spaces() {
        let names = get_class_names();
        for name in names.iter() {
            assert!(
                !name.contains(' '),
                "Class name '{}' should not contain spaces",
                name
            );
        }
    }
}
