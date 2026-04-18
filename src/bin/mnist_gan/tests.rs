use super::*;

/// Verifies that `sigmoid` yields values close to 0 for large negative inputs, 0.5 at 0, and close to 1 for large positive inputs.
///
/// Checks:
/// - `sigmoid(-100.0) < 0.001`
/// - `sigmoid(0.0)` is within `1e-6` of `0.5`
/// - `sigmoid(100.0) > 0.999`
///
/// # Examples
///
/// ```
/// assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
/// ```
#[test]
fn test_sigmoid_bounds() {
    assert!(sigmoid(-100.0) < 0.001);
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!(sigmoid(100.0) > 0.999);
}

#[test]
fn test_sigmoid_midpoint() {
    assert_eq!(sigmoid(0.0), 0.5);
}

#[test]
fn test_bce_loss_near_zero_when_correct() {
    assert!(bce_loss(0.99, 1.0) < 0.05);
}

#[test]
fn test_bce_loss_high_when_wrong() {
    assert!(bce_loss(0.01, 1.0) > 4.0);
}

/// Creates a `Generator` seeded with a fixed `SimpleRng` and verifies its layer dimensions.
///
/// Confirms that layer1 maps `NOISE_DIM` -> `G_HIDDEN1`, layer2 maps `G_HIDDEN1` -> `G_HIDDEN2`,
/// and layer3 maps `G_HIDDEN2` -> `IMG_SIZE`.
///
/// # Examples
///
/// ```
/// let mut rng = SimpleRng::new(42);
/// let gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
/// assert_eq!(gen.layer1.input_size(), NOISE_DIM);
/// assert_eq!(gen.layer1.output_size(), G_HIDDEN1);
/// assert_eq!(gen.layer2.input_size(), G_HIDDEN1);
/// assert_eq!(gen.layer2.output_size(), G_HIDDEN2);
/// assert_eq!(gen.layer3.input_size(), G_HIDDEN2);
/// assert_eq!(gen.layer3.output_size(), IMG_SIZE);
/// ```
#[test]
fn test_generator_create() {
    let mut rng = SimpleRng::new(42);
    let gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    assert_eq!(gen.layer1.input_size(), NOISE_DIM);
    assert_eq!(gen.layer1.output_size(), G_HIDDEN1);
    assert_eq!(gen.layer2.input_size(), G_HIDDEN1);
    assert_eq!(gen.layer2.output_size(), G_HIDDEN2);
    assert_eq!(gen.layer3.input_size(), G_HIDDEN2);
    assert_eq!(gen.layer3.output_size(), IMG_SIZE);
}

#[test]
fn test_discriminator_create() {
    let mut rng = SimpleRng::new(42);
    let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    assert_eq!(disc.layer1.input_size(), IMG_SIZE);
    assert_eq!(disc.layer1.output_size(), D_HIDDEN1);
    assert_eq!(disc.layer2.input_size(), D_HIDDEN1);
    assert_eq!(disc.layer2.output_size(), D_HIDDEN2);
    assert_eq!(disc.layer3.input_size(), D_HIDDEN2);
    assert_eq!(disc.layer3.output_size(), 1);
}

#[test]
fn test_generator_noise_range() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let noise = gen.generate_noise(10);
    assert_eq!(noise.len(), 10 * NOISE_DIM);
    for &v in &noise {
        assert!(
            (-1.0..=1.0).contains(&v),
            "Noise value {} outside [-1, 1]",
            v
        );
    }
}

#[test]
fn test_generator_forward_shape() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let batch_size = 4;
    let noise = gen.generate_noise(batch_size);
    let (a1, a2, output) = gen.forward(&noise, batch_size);
    assert_eq!(a1.len(), batch_size * G_HIDDEN1);
    assert_eq!(a2.len(), batch_size * G_HIDDEN2);
    assert_eq!(output.len(), batch_size * IMG_SIZE);
}

#[test]
fn test_generator_forward_uses_layer1_activation_values() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let batch_size = 2;
    let noise = gen.generate_noise(batch_size);

    let (_, _, output) = gen.forward(&noise, batch_size);

    let mut expected_a1 = vec![0.0f32; batch_size * G_HIDDEN1];
    gen.layer1.forward(&noise, &mut expected_a1, batch_size);
    leaky_relu_inplace(&mut expected_a1, LEAKY_RELU_ALPHA);

    let mut expected_a2 = vec![0.0f32; batch_size * G_HIDDEN2];
    gen.layer2
        .forward(&expected_a1, &mut expected_a2, batch_size);
    leaky_relu_inplace(&mut expected_a2, LEAKY_RELU_ALPHA);

    let mut expected_output = vec![0.0f32; batch_size * IMG_SIZE];
    gen.layer3
        .forward(&expected_a2, &mut expected_output, batch_size);
    tanh_inplace(&mut expected_output);

    for &idx in &[0usize, IMG_SIZE / 2, IMG_SIZE + 7] {
        assert!(
            (output[idx] - expected_output[idx]).abs() < 1e-6,
            "output[{idx}] = {}, expected {}",
            output[idx],
            expected_output[idx]
        );
    }
}

#[test]
fn test_generator_output_in_tanh_range() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let batch_size = 4;
    let noise = gen.generate_noise(batch_size);
    let (_, _, output) = gen.forward(&noise, batch_size);
    for &v in &output {
        assert!(
            v > -1.0 && v < 1.0,
            "Generator output {} outside (-1, 1)",
            v
        );
    }
}

#[test]
fn test_discriminator_forward_shape() {
    let mut rng = SimpleRng::new(42);
    let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let batch_size = 4;
    let images = vec![0.0f32; batch_size * IMG_SIZE];
    let (a1, a2, output) = disc.forward(&images, batch_size);
    assert_eq!(a1.len(), batch_size * D_HIDDEN1);
    assert_eq!(a2.len(), batch_size * D_HIDDEN2);
    assert_eq!(output.len(), batch_size);
}

#[test]
fn test_discriminator_output_in_sigmoid_range() {
    let mut rng = SimpleRng::new(42);
    let disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let batch_size = 4;
    let images = vec![0.0f32; batch_size * IMG_SIZE];
    let (_, _, output) = disc.forward(&images, batch_size);
    for &v in &output {
        assert!(
            v > 0.0 && v < 1.0,
            "Discriminator output {} outside (0, 1)",
            v
        );
    }
}

#[test]
#[should_panic(expected = "compute_diversity requires n_samples > 0")]
fn test_compute_diversity_rejects_zero_samples() {
    let mut rng = SimpleRng::new(42);
    let gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);

    compute_diversity(&gen, &mut rng, 0);
}

#[test]
#[should_panic(expected = "num_train must be > 0")]
fn test_train_gan_rejects_zero_num_train() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let mut disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let train_images = Vec::new();

    train_gan(
        &mut gen,
        &mut disc,
        &train_images,
        0,
        &mut rng,
        1,
        1,
        &train_images,
        1,
        0.1,
        NOISE_DIM,
    );
}

#[test]
#[should_panic(expected = "train_images length")]
fn test_train_gan_rejects_short_train_images() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let mut disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let train_images = vec![0.0f32; IMG_SIZE];

    train_gan(
        &mut gen,
        &mut disc,
        &train_images,
        2,
        &mut rng,
        1,
        1,
        &train_images,
        1,
        0.1,
        NOISE_DIM,
    );
}

#[test]
#[should_panic(expected = "epochs must be > 0")]
fn test_train_gan_rejects_zero_epochs() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let mut disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let train_images = vec![0.0f32; IMG_SIZE];

    train_gan(
        &mut gen,
        &mut disc,
        &train_images,
        1,
        &mut rng,
        0,
        1,
        &train_images,
        1,
        0.1,
        NOISE_DIM,
    );
}

#[test]
#[should_panic(expected = "batch_size must be > 0")]
fn test_train_gan_rejects_zero_batch_size() {
    let mut rng = SimpleRng::new(42);
    let mut gen = Generator::new(&mut rng, 0.0002, 0.5, 0.999);
    let mut disc = Discriminator::new(&mut rng, 0.0002, 0.5, 0.999);
    let train_images = vec![0.0f32; IMG_SIZE];

    train_gan(
        &mut gen,
        &mut disc,
        &train_images,
        1,
        &mut rng,
        1,
        0,
        &train_images,
        1,
        0.1,
        NOISE_DIM,
    );
}
