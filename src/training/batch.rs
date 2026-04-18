use crate::utils::rng::SimpleRng;

/// Copies a mini-batch of images and labels into the provided output buffers and optionally applies
/// data augmentations to each copied image.
///
/// Augmentations are applied only when a mutable RNG is supplied and their corresponding
/// parameters are Some. The applied order is: random crop, horizontal flip, brightness jitter,
/// contrast jitter, then saturation jitter. Saturation jitter is RGB-only and is applied only
/// when `img_channels == 3`; it is skipped for all other channel counts.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::gather_batch;
///
/// // Tiny MNIST-like dataset: 3 grayscale 2×2 images (4 pixels each).
/// let images = vec![1.0f32, 2.0, 3.0, 4.0,   // image 0
///                   5.0,    6.0, 7.0, 8.0,   // image 1
///                   9.0,   10.0,11.0,12.0];  // image 2
/// let labels = vec![0u8, 1u8, 2u8];
/// let indices = vec![2usize, 0, 1];
/// let mut out_inputs = vec![0.0f32; 4 * 2]; // 2 images × 4 pixels
/// let mut out_labels = vec![0u8; 2];
///
/// // Gather 2 samples (indices[0..2] → images 2 and 0), no augmentation.
/// gather_batch(
///     &images, &labels, &indices, 0, 2,
///     &mut out_inputs, &mut out_labels,
///     2, 2, 1, // img_width=2, img_height=2, img_channels=1
///     None, None, None, None, None, None,
/// );
///
/// assert_eq!(out_labels, vec![2u8, 0u8]);
/// assert_eq!(out_inputs[0], 9.0); // first pixel of image 2
/// assert_eq!(out_inputs[4], 1.0); // first pixel of image 0
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gather_batch(
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    start: usize,
    count: usize,
    out_inputs: &mut [f32],
    out_labels: &mut [u8],
    img_width: usize,
    img_height: usize,
    img_channels: usize,
    flip_prob: Option<f32>,
    crop_padding: Option<usize>,
    brightness_jitter: Option<f32>,
    contrast_jitter: Option<f32>,
    saturation_jitter: Option<f32>,
    mut rng: Option<&mut SimpleRng>,
) {
    use crate::data::augmentation::{
        random_brightness, random_contrast, random_crop_into, random_horizontal_flip,
        random_saturation,
    };

    let image_size = match img_width
        .checked_mul(img_height)
        .and_then(|size| size.checked_mul(img_channels))
    {
        Some(size) => size,
        None => return,
    };
    let end = match start.checked_add(count) {
        Some(end) if end <= indices.len() => end,
        _ => return,
    };
    let required_output_len = match count.checked_mul(image_size) {
        Some(len) => len,
        None => return,
    };

    if out_inputs.len() < required_output_len || out_labels.len() < count {
        return;
    }
    for &src_index in &indices[start..end] {
        if src_index >= labels.len() {
            return;
        }
        let src_start = match src_index.checked_mul(image_size) {
            Some(start) => start,
            None => return,
        };
        let src_end = match src_start.checked_add(image_size) {
            Some(end) => end,
            None => return,
        };
        if src_end > images.len() {
            return;
        }
    }

    let mut crop_scratch = if crop_padding.is_some() && rng.is_some() {
        vec![0.0f32; image_size]
    } else {
        Vec::new()
    };

    for i in 0..count {
        let src_index = indices[start + i];
        let src_start = src_index * image_size;
        let dst_start = i * image_size;

        out_inputs[dst_start..dst_start + image_size]
            .copy_from_slice(&images[src_start..src_start + image_size]);
        out_labels[i] = labels[src_index];

        if let Some(ref mut rng_ref) = rng {
            let image_slice = &mut out_inputs[dst_start..dst_start + image_size];

            if let Some(padding) = crop_padding {
                random_crop_into(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    padding,
                    img_width,
                    img_height,
                    rng_ref,
                    &mut crop_scratch,
                );
                image_slice.copy_from_slice(&crop_scratch);
            }

            if let Some(prob) = flip_prob {
                random_horizontal_flip(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    prob,
                    rng_ref,
                );
            }

            if let Some(brightness_delta) = brightness_jitter {
                random_brightness(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    brightness_delta,
                    rng_ref,
                );
            }

            if let Some(contrast_delta) = contrast_jitter {
                random_contrast(
                    image_slice,
                    img_width,
                    img_height,
                    img_channels,
                    contrast_delta,
                    rng_ref,
                );
            }

            if let Some(saturation_delta) = saturation_jitter {
                // Saturation jitter operates on RGB luminance, so non-3-channel images skip it.
                if img_channels == 3 {
                    random_saturation(
                        image_slice,
                        img_width,
                        img_height,
                        img_channels,
                        saturation_delta,
                        rng_ref,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slices_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {}: expected {}, got {}",
                idx,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_gather_batch_basic_copy() {
        let images = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![2usize, 0, 1];
        let mut out_inputs = vec![0.0f32; 4 * 2];
        let mut out_labels = vec![0u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(out_labels[0], 2u8, "first sample should be image 2");
        assert_eq!(out_labels[1], 0u8, "second sample should be image 0");
        assert!(
            (out_inputs[0] - 9.0f32).abs() < 1e-6,
            "first pixel of image 2"
        );
        assert!(
            (out_inputs[4] - 1.0f32).abs() < 1e-6,
            "first pixel of image 0"
        );
    }

    #[test]
    fn test_gather_batch_no_augmentation_without_rng() {
        let images = vec![0.5f32; 4 * 3];
        let labels = vec![0u8, 1u8, 2u8];
        let indices = vec![0usize, 1, 2];
        let mut out_inputs = vec![0.0f32; 4 * 2];
        let mut out_labels = vec![0u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            Some(0.5),
            Some(1),
            Some(0.2),
            Some(0.2),
            Some(0.2),
            None,
        );

        for &v in &out_inputs[..8] {
            assert!(
                (v - 0.5f32).abs() < 1e-6,
                "without rng, pixels should be unchanged"
            );
        }
    }

    #[test]
    fn test_gather_batch_with_flip_augmentation() {
        let images = vec![1.0f32, 2.0, 3.0, 4.0];
        let labels = vec![0u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; 4];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            4,
            1,
            1,
            Some(1.0),
            None,
            None,
            None,
            None,
            Some(&mut rng),
        );

        assert_eq!(out_labels[0], 0u8);
        assert!(
            (out_inputs[0] - 4.0f32).abs() < 1e-6,
            "flipped pixel 0 should be 4.0, got {}",
            out_inputs[0]
        );
        assert!(
            (out_inputs[3] - 1.0f32).abs() < 1e-6,
            "flipped pixel 3 should be 1.0, got {}",
            out_inputs[3]
        );
    }

    #[test]
    fn test_gather_batch_applies_random_crop_into() {
        let width = 2;
        let height = 2;
        let channels = 3;
        let image_size = width * height * channels;
        let images: Vec<f32> = (0..image_size).map(|i| (i as f32 + 1.0) / 20.0).collect();
        let labels = vec![4u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; image_size];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);
        let mut expected_rng = SimpleRng::new(99);
        let mut expected = vec![0.0f32; image_size];

        crate::data::augmentation::random_crop_into(
            &images,
            width,
            height,
            channels,
            1,
            width,
            height,
            &mut expected_rng,
            &mut expected,
        );

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            width,
            height,
            channels,
            None,
            Some(1),
            None,
            None,
            None,
            Some(&mut rng),
        );

        assert_eq!(out_labels, vec![4u8]);
        assert_slices_close(&out_inputs, &expected);
    }

    #[test]
    fn test_gather_batch_applies_brightness_jitter() {
        let width = 2;
        let height = 2;
        let channels = 3;
        let image_size = width * height * channels;
        let images = vec![0.5f32; image_size];
        let labels = vec![2u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; image_size];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);
        let mut expected_rng = SimpleRng::new(99);
        let mut expected = images.clone();

        crate::data::augmentation::random_brightness(
            &mut expected,
            width,
            height,
            channels,
            0.2,
            &mut expected_rng,
        );

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            width,
            height,
            channels,
            None,
            None,
            Some(0.2),
            None,
            None,
            Some(&mut rng),
        );

        assert_eq!(out_labels, vec![2u8]);
        assert!(expected
            .iter()
            .zip(&images)
            .any(|(&a, &b)| (a - b).abs() > 1e-6));
        assert_slices_close(&out_inputs, &expected);
    }

    #[test]
    fn test_gather_batch_applies_contrast_jitter() {
        let width = 2;
        let height = 2;
        let channels = 3;
        let image_size = width * height * channels;
        let images: Vec<f32> = (0..image_size).map(|i| 0.1 + i as f32 * 0.05).collect();
        let labels = vec![3u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; image_size];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);
        let mut expected_rng = SimpleRng::new(99);
        let mut expected = images.clone();

        crate::data::augmentation::random_contrast(
            &mut expected,
            width,
            height,
            channels,
            0.4,
            &mut expected_rng,
        );

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            width,
            height,
            channels,
            None,
            None,
            None,
            Some(0.4),
            None,
            Some(&mut rng),
        );

        assert_eq!(out_labels, vec![3u8]);
        assert!(expected
            .iter()
            .zip(&images)
            .any(|(&a, &b)| (a - b).abs() > 1e-6));
        assert_slices_close(&out_inputs, &expected);
    }

    #[test]
    fn test_gather_batch_skips_saturation_for_non_rgb_images() {
        let images = vec![0.1f32, 0.2, 0.3, 0.4];
        let labels = vec![0u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![0.0f32; 4];
        let mut out_labels = vec![0u8; 1];
        let mut rng = SimpleRng::new(99);

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            None,
            None,
            None,
            None,
            Some(0.2),
            Some(&mut rng),
        );

        assert_eq!(out_inputs, images);
    }

    #[test]
    fn test_gather_batch_returns_before_writing_when_index_range_is_invalid() {
        let images = vec![1.0f32, 2.0, 3.0, 4.0];
        let labels = vec![0u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![-1.0f32; 4];
        let mut out_labels = vec![9u8; 1];

        gather_batch(
            &images,
            &labels,
            &indices,
            1,
            1,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(out_inputs, vec![-1.0f32; 4]);
        assert_eq!(out_labels, vec![9u8]);
    }

    #[test]
    fn test_gather_batch_returns_before_partial_write_when_src_index_is_invalid() {
        let images = vec![1.0f32, 2.0, 3.0, 4.0];
        let labels = vec![0u8];
        let indices = vec![0usize, 1];
        let mut out_inputs = vec![-1.0f32; 8];
        let mut out_labels = vec![9u8; 2];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            2,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(out_inputs, vec![-1.0f32; 8]);
        assert_eq!(out_labels, vec![9u8; 2]);
    }

    #[test]
    fn test_gather_batch_returns_before_writing_when_outputs_are_too_small() {
        let images = vec![1.0f32, 2.0, 3.0, 4.0];
        let labels = vec![0u8];
        let indices = vec![0usize];
        let mut out_inputs = vec![-1.0f32; 3];
        let mut out_labels = vec![9u8; 1];

        gather_batch(
            &images,
            &labels,
            &indices,
            0,
            1,
            &mut out_inputs,
            &mut out_labels,
            2,
            2,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_eq!(out_inputs, vec![-1.0f32; 3]);
        assert_eq!(out_labels, vec![9u8]);
    }
}
