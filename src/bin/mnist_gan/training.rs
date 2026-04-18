use super::*;

pub(crate) const VALIDATION_SPLIT: f32 = 0.1;
const EARLY_STOPPING_PATIENCE: usize = 3;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.001;

/// Computes the standard deviation of per-sample mean pixel values for generated images.
///
/// Generates `n_samples` images using a dedicated evaluation RNG, computes each image's
/// mean pixel value, and returns the standard deviation across those means. A small value
/// (for example, less than 0.02) indicates low diversity and may signal mode collapse.
///
/// # Examples
///
/// ```ignore
/// // `gen` is a Generator; replace the setup with a real generator instance.
/// # use crate::Generator;
/// # fn make_gen() -> Generator { unimplemented!() }
/// let gen = make_gen();
/// let mut rng = SimpleRng::new(123);
/// let diversity = compute_diversity(&gen, &mut rng, 10);
/// assert!(diversity >= 0.0);
/// ```
pub(crate) fn compute_diversity(
    gen: &Generator,
    eval_rng: &mut SimpleRng,
    n_samples: usize,
) -> f32 {
    assert!(n_samples > 0, "compute_diversity requires n_samples > 0");

    let noise = sample_noise(eval_rng, n_samples);
    let (_, _, samples) = gen.forward(&noise, n_samples);

    // Mean pixel value per sample
    let means: Vec<f32> = (0..n_samples)
        .map(|i| {
            let start = i * IMG_SIZE;
            let end = start + IMG_SIZE;
            samples[start..end].iter().sum::<f32>() / IMG_SIZE as f32
        })
        .collect();

    // Std-dev across samples
    let mean_of_means = means.iter().sum::<f32>() / n_samples as f32;
    let variance = means
        .iter()
        .map(|&m| (m - mean_of_means) * (m - mean_of_means))
        .sum::<f32>()
        / n_samples as f32;
    variance.sqrt()
}

fn validate_gan(
    gen: &Generator,
    disc: &Discriminator,
    val_images: &[f32],
    num_val: usize,
    batch_size: usize,
    real_target: f32,
    fake_target: f32,
    epoch: usize,
) -> (f32, f32) {
    let mut loss_sum = 0.0f32;
    let mut correct = 0usize;
    let mut sample_count = 0usize;
    let mut eval_rng = SimpleRng::new(30_000 + epoch as u64);

    for batch_start in (0..num_val).step_by(batch_size) {
        let bs = (num_val - batch_start).min(batch_size);

        let mut real_imgs = vec![0.0f32; bs * IMG_SIZE];
        for i in 0..bs {
            let src = (batch_start + i) * IMG_SIZE;
            let dst = i * IMG_SIZE;
            for k in 0..IMG_SIZE {
                real_imgs[dst + k] = val_images[src + k] * 2.0 - 1.0;
            }
        }

        let (_, _, real_pred) = disc.forward(&real_imgs, bs);
        for &pred in &real_pred {
            loss_sum += bce_loss(pred, real_target);
            if pred >= 0.5 {
                correct += 1;
            }
        }

        let noise = sample_noise(&mut eval_rng, bs);
        let (_, _, fake_imgs) = gen.forward(&noise, bs);
        let (_, _, fake_pred) = disc.forward(&fake_imgs, bs);
        for &pred in &fake_pred {
            loss_sum += bce_loss(pred, fake_target);
            if pred < 0.5 {
                correct += 1;
            }
        }

        sample_count += bs * 2;
    }

    assert!(
        sample_count > 0,
        "validation sample_count must be > 0 before averaging"
    );
    (
        loss_sum / sample_count as f32,
        correct as f32 / sample_count as f32,
    )
}

// ============================================================================
// GAN Training Loop
// ============================================================================

/// Train the generator and discriminator networks over multiple epochs using shuffled MNIST mini-batches.
///
/// Per epoch this function shuffles the training set, iterates over mini-batches, trains the discriminator
/// on real (with one-sided label smoothing) and generated images, then trains the generator to fool the
/// discriminator. It logs per-epoch metrics to ./logs/mnist_gan_log.csv, checkpoints the best and final models,
/// exports example samples, and flags mode collapse when sample diversity (computed via `compute_diversity`) falls
/// below 0.02.
///
/// # Parameters
///
/// * `label_smoothing` — One-sided smoothing applied to real labels; the discriminator's real target is `1.0 - label_smoothing`.
/// * `noise_dim_check` — Sanity-check value that must equal the compiled `NOISE_DIM`.
///
/// # Examples
///
/// ```no_run
/// // Assuming `gen`, `disc`, `train_images`, and `rng` are initialized appropriately:
/// // train_gan(&mut gen, &mut disc, &train_images, train_images.len() / IMG_SIZE, &mut rng, 10, 64, &val_images, val_images.len() / IMG_SIZE, 0.1, NOISE_DIM);
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_gan(
    gen: &mut Generator,
    disc: &mut Discriminator,
    train_images: &[f32],
    num_train: usize,
    rng: &mut SimpleRng,
    epochs: usize,
    batch_size: usize,
    val_images: &[f32],
    num_val: usize,
    label_smoothing: f32,
    noise_dim_check: usize,
) {
    assert_eq!(
        noise_dim_check, NOISE_DIM,
        "Config noise_dim {} does not match compiled NOISE_DIM {}",
        noise_dim_check, NOISE_DIM
    );
    assert!(num_train > 0, "num_train must be > 0");
    assert!(epochs > 0, "epochs must be > 0");
    assert!(batch_size > 0, "batch_size must be > 0");
    let required_train_len = num_train
        .checked_mul(IMG_SIZE)
        .expect("num_train * IMG_SIZE overflow");
    assert!(
        train_images.len() >= required_train_len,
        "train_images length {} is smaller than required {} for num_train {} and IMG_SIZE {}",
        train_images.len(),
        required_train_len,
        num_train,
        IMG_SIZE
    );
    let required_val_len = num_val
        .checked_mul(IMG_SIZE)
        .expect("num_val * IMG_SIZE overflow");
    assert!(num_val > 0, "num_val must be > 0");
    assert!(
        val_images.len() >= required_val_len,
        "val_images length {} is smaller than required {} for num_val {} and IMG_SIZE {}",
        val_images.len(),
        required_val_len,
        num_val,
        IMG_SIZE
    );
    if !(label_smoothing.is_finite() && label_smoothing >= 0.0 && label_smoothing < 1.0) {
        eprintln!(
            "Invalid label_smoothing: {}. Expected a value in [0.0, 1.0).",
            label_smoothing
        );
        process::exit(1);
    }

    std::fs::create_dir_all("./logs").ok();

    // Open GAN training log
    let log_file = File::create("./logs/mnist_gan_log.csv").unwrap_or_else(|_| {
        eprintln!("Could not open logs/mnist_gan_log.csv for writing");
        process::exit(1);
    });
    let mut log_writer = BufWriter::new(log_file);
    writeln!(
        log_writer,
        "epoch,train_loss,train_time,val_loss,val_accuracy"
    )
    .unwrap_or_else(|_| {
        eprintln!("Failed writing CSV header");
        process::exit(1);
    });

    let real_target = 1.0 - label_smoothing; // one-sided label smoothing
    let fake_target_d = 0.0f32; // discriminator sees fakes as 0
    let real_target_g = 1.0f32; // generator wants D to output 1

    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0usize;
    let mut epochs_without_improvement = 0usize;
    let mut mode_collapse_detected = false;

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut d_loss_sum = 0.0f32;
        let mut g_loss_sum = 0.0f32;
        let mut d_real_sum = 0.0f32;
        let mut d_fake_sum = 0.0f32;
        let mut batch_count = 0usize;
        let mut sample_count = 0usize;

        // Fisher-Yates shuffle
        for i in (1..num_train).rev() {
            let j = rng.gen_usize(i + 1);
            indices.swap(i, j);
        }

        for batch_start in (0..num_train).step_by(batch_size) {
            let bs = (num_train - batch_start).min(batch_size);

            // ── Gather real images and rescale [0, 1] → [−1, 1] ──
            let mut real_imgs = vec![0.0f32; bs * IMG_SIZE];
            for (slot_idx, &sample_idx) in indices[batch_start..batch_start + bs].iter().enumerate()
            {
                let src = sample_idx * IMG_SIZE;
                let dst = slot_idx * IMG_SIZE;
                for k in 0..IMG_SIZE {
                    real_imgs[dst + k] = train_images[src + k] * 2.0 - 1.0;
                }
            }

            // ──────────────────────────────────────────────────────────────
            // Step 1: Train Discriminator on real images
            // ──────────────────────────────────────────────────────────────
            let (d_a1_real, d_a2_real, d_pred_real) = disc.forward(&real_imgs, bs);
            let mut grad_logit_real = vec![0.0f32; bs];
            let mut d_real_acc = 0.0f32;
            for i in 0..bs {
                grad_logit_real[i] = d_pred_real[i] - real_target; // pred - target
                d_real_acc += d_pred_real[i];
                d_loss_sum += bce_loss(d_pred_real[i], real_target);
            }
            disc.backward(&real_imgs, &d_a1_real, &d_a2_real, &grad_logit_real, bs);

            // ──────────────────────────────────────────────────────────────
            // Step 2: Train Discriminator on fake images
            // ──────────────────────────────────────────────────────────────
            let noise = gen.generate_noise(bs);
            let (g_a1, g_a2, fake_imgs) = gen.forward(&noise, bs);

            let (d_a1_fake, d_a2_fake, d_pred_fake) = disc.forward(&fake_imgs, bs);
            let mut grad_logit_fake = vec![0.0f32; bs];
            let mut d_fake_acc = 0.0f32;
            for i in 0..bs {
                grad_logit_fake[i] = d_pred_fake[i] - fake_target_d; // pred - 0 = pred
                d_fake_acc += d_pred_fake[i];
                d_loss_sum += bce_loss(d_pred_fake[i], fake_target_d);
            }
            disc.backward(&fake_imgs, &d_a1_fake, &d_a2_fake, &grad_logit_fake, bs);

            // ──────────────────────────────────────────────────────────────
            // Step 3: Train Generator (fool the discriminator)
            // ──────────────────────────────────────────────────────────────
            // Re-run D forward on the same fake images to get fresh activations
            // (D's parameters may have changed in steps 1 and 2).
            let noise2 = gen.generate_noise(bs);
            let (g_a1_new, g_a2_new, fake_imgs_new) = gen.forward(&noise2, bs);
            let (d_a1_g, d_a2_g, d_pred_g) = disc.forward(&fake_imgs_new, bs);

            // G's adversarial gradient: target=1.0 (G wants D to output 1)
            let mut grad_logit_g = vec![0.0f32; bs];
            for i in 0..bs {
                grad_logit_g[i] = d_pred_g[i] - real_target_g; // pred - 1 (negative → pushes D toward 1)
                g_loss_sum += bce_loss(d_pred_g[i], real_target_g);
            }

            // Propagate gradient through D (without updating D's parameters)
            let d_grad_input = disc.propagate_gradient(&d_a1_g, &d_a2_g, &grad_logit_g, bs);

            // Backpropagate through G and update G
            gen.backward(
                &noise2,
                &g_a1_new,
                &g_a2_new,
                &fake_imgs_new,
                &d_grad_input,
                bs,
            );

            d_real_sum += d_real_acc;
            d_fake_sum += d_fake_acc;
            batch_count += 1;
            sample_count += bs;

            // Suppress unused variable warnings for first D pass activations
            let _ = (&g_a1, &g_a2, &fake_imgs);
        }

        assert!(batch_count > 0, "batch_count must be > 0 before averaging");
        assert!(
            sample_count > 0,
            "sample_count must be > 0 before averaging"
        );
        let n_samples = sample_count as f32;
        let avg_d_loss = d_loss_sum / (n_samples * 2.0); // averaged over real+fake samples
        let avg_g_loss = g_loss_sum / n_samples;
        let train_loss = avg_d_loss + avg_g_loss;
        let avg_d_real = d_real_sum / n_samples;
        let avg_d_fake = d_fake_sum / n_samples;
        let epoch_time = epoch_start.elapsed().as_secs_f32();
        let (val_loss, val_accuracy) = validate_gan(
            gen,
            disc,
            val_images,
            num_val,
            batch_size,
            real_target,
            fake_target_d,
            epoch,
        );

        // Mode collapse detection
        let mut diversity_rng = SimpleRng::new(10_000 + epoch as u64);
        let diversity = compute_diversity(gen, &mut diversity_rng, 100);
        if diversity < 0.02 {
            mode_collapse_detected = true;
            println!(
                "WARNING: Mode collapse detected at epoch {} (diversity: {:.4})",
                epoch + 1,
                diversity
            );
        }

        println!(
            "Epoch {}/{}, G loss: {:.4}, D loss: {:.4}, D(real): {:.4}, D(fake): {:.4}, \
             Val loss: {:.4}, Val acc: {:.4}, Diversity: {:.4}, Time: {:.1}s",
            epoch + 1,
            epochs,
            avg_g_loss,
            avg_d_loss,
            avg_d_real,
            avg_d_fake,
            val_loss,
            val_accuracy,
            diversity,
            epoch_time
        );

        writeln!(
            log_writer,
            "{},{},{:.6},{},{}",
            epoch + 1,
            train_loss,
            epoch_time,
            val_loss,
            val_accuracy
        )
        .unwrap_or_else(|_| {
            eprintln!("Failed writing epoch log");
            process::exit(1);
        });

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA {
            best_val_loss = val_loss;
            best_epoch = epoch + 1;
            save_model(gen, disc, "mnist_gan_best.bin");
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE {
                println!(
                    "Early stopping at epoch {}: validation loss did not improve by {:.4} for {} epochs",
                    epoch + 1,
                    EARLY_STOPPING_MIN_DELTA,
                    EARLY_STOPPING_PATIENCE
                );
                break;
            }
        }
    }

    // Export 16 generated samples to CSV
    let mut export_rng = SimpleRng::new(20_000 + epochs as u64);
    export_samples(gen, &mut export_rng, 16, "./logs/mnist_gan_samples.csv");

    // Save final model
    save_model(gen, disc, "mnist_gan_final.bin");

    println!(
        "\nTraining complete. Best validation loss: {:.4} at epoch {}. Mode collapse detected: {}",
        best_val_loss,
        best_epoch,
        if mode_collapse_detected { "Yes" } else { "No" }
    );
}

// ============================================================================
// Model I/O
// ============================================================================
