//! Integration tests for CIFAR-10 architecture JSONs.
//!
//! These tests ensure the shipped CIFAR-10 architecture configs can be loaded via
//! the shared architecture system and can run a forward pass on a CIFAR-shaped
//! batch.

use rust_neural_networks::architecture::{build_model, load_architecture};
use rust_neural_networks::utils::rng::SimpleRng;

fn cifar10_cnn_binary_path() -> std::path::PathBuf {
    // Cargo exposes the absolute path to built binaries to integration tests.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_cifar10_cnn"))
}

fn forward_through_model(
    arch_path: &str,
    batch_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let arch = load_architecture(arch_path)?;

    let mut rng = SimpleRng::new(42);
    let layers = build_model(&arch, &mut rng)?;
    assert!(!layers.is_empty(), "Expected at least one layer");

    let input_size = 3 * 32 * 32;
    let mut current = vec![0.5f32; input_size * batch_size];

    for layer in layers.iter() {
        let output_size = layer.output_size();
        let mut next = vec![0.0f32; output_size * batch_size];
        layer.forward(&current, &mut next, batch_size);
        current = next;
    }

    Ok(current)
}

#[test]
fn cifar10_baseline_architecture_builds_and_forwards() {
    let output = forward_through_model("config/architectures/cifar10_cnn_baseline.json", 2)
        .expect("baseline cifar10 architecture should build and run forward pass");

    // CIFAR-10 classifier should output 10 logits per sample.
    assert_eq!(output.len(), 2 * 10);
}

#[test]
fn cifar10_deep_architecture_builds_and_forwards() {
    let output = forward_through_model("config/architectures/cifar10_deep_cnn.json", 2)
        .expect("deep cifar10 architecture should build and run forward pass");

    // CIFAR-10 classifier should output 10 logits per sample.
    assert_eq!(output.len(), 2 * 10);
}

#[test]
fn cifar10_cnn_binary_with_deep_arch_reaches_training_loop() {
    let binary = cifar10_cnn_binary_path();

    let output = std::process::Command::new(binary)
        .args(["--arch", "config/architectures/cifar10_deep_cnn.json"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("failed to run cifar10_cnn binary");

    // The run is expected to fail in CI because it requires CIFAR-10 data,
    // but we still want to verify that argument parsing + architecture loading
    // succeed and that we enter the training loop.
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        combined.contains("Loading architecture from: config/architectures/cifar10_deep_cnn.json"),
        "expected architecture path to be logged, got:\n{combined}"
    );

    // If training begins, we should see epoch logging. If CIFAR data is missing,
    // the binary may exit earlier, but argument parsing + architecture loading
    // should still be validated by the assertion above.
    if output.status.success() {
        assert!(
            combined.contains("Epoch 1/") || combined.contains("Epoch 1,"),
            "expected training loop to start (epoch logging), got:\n{combined}"
        );
    }
}

#[test]
fn cifar10_cnn_binary_default_shallow_reaches_training_loop() {
    let binary = cifar10_cnn_binary_path();

    let output = std::process::Command::new(binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("failed to run cifar10_cnn binary");

    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        combined
            .contains("Loading architecture from: config/architectures/cifar10_cnn_baseline.json"),
        "expected default baseline architecture path to be logged, got:\n{combined}"
    );

    if output.status.success() {
        assert!(
            combined.contains("Epoch 1/") || combined.contains("Epoch 1,"),
            "expected training loop to start (epoch logging), got:\n{combined}"
        );
    }
}
