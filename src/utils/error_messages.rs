//! Enhanced error message helpers for common setup and data issues.
//!
//! This module provides human-friendly error messages for situations that students
//! and beginners commonly encounter, such as missing dataset files or BLAS setup
//! problems. Each function returns a multi-line string suitable for embedding in
//! an `io::Error` message or printing directly to stderr.

/// Returns a descriptive error message with MNIST download instructions when an
/// expected data file is not found.
///
/// # Arguments
///
/// * `filename` - The path that was attempted (e.g., `"data/train-images-idx3-ubyte"`).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::error_messages::mnist_data_not_found_message;
///
/// let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
/// assert!(msg.contains("MNIST"));
/// assert!(msg.contains("docs/"));
/// ```
pub fn mnist_data_not_found_message(filename: &str) -> String {
    format!(
        "MNIST data file not found: '{filename}'\n\
         \n\
         The MNIST dataset must be downloaded and placed in the 'data/' directory.\n\
         Expected files:\n\
           data/train-images-idx3-ubyte   (training images)\n\
           data/train-labels-idx1-ubyte   (training labels)\n\
           data/t10k-images-idx3-ubyte    (test images)\n\
           data/t10k-labels-idx1-ubyte    (test labels)\n\
         \n\
         Download instructions: see docs/2b MNIST-Data-Setup.md\n\
         Official source: http://yann.lecun.com/exdb/mnist/\n\
         Mirror:          https://storage.googleapis.com/cvdf-datasets/mnist/",
    )
}

/// Returns a descriptive error message with CIFAR-10 download instructions when an
/// expected data file is not found.
///
/// # Arguments
///
/// * `filename` - The path that was attempted (e.g., `"data/cifar-10-batches-bin/data_batch_1.bin"`).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::error_messages::cifar10_data_not_found_message;
///
/// let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
/// assert!(msg.contains("CIFAR-10"));
/// assert!(msg.contains("docs/"));
/// ```
pub fn cifar10_data_not_found_message(filename: &str) -> String {
    format!(
        "CIFAR-10 data file not found: '{filename}'\n\
         \n\
         The CIFAR-10 dataset must be downloaded and extracted to 'data/cifar-10-batches-bin/'.\n\
         Expected files:\n\
           data/cifar-10-batches-bin/data_batch_1.bin  (training batch 1)\n\
           data/cifar-10-batches-bin/data_batch_2.bin  (training batch 2)\n\
           data/cifar-10-batches-bin/data_batch_3.bin  (training batch 3)\n\
           data/cifar-10-batches-bin/data_batch_4.bin  (training batch 4)\n\
           data/cifar-10-batches-bin/data_batch_5.bin  (training batch 5)\n\
           data/cifar-10-batches-bin/test_batch.bin    (test set)\n\
         \n\
         Download instructions: see docs/cifar10_dataset.md\n\
         Official source: https://www.cs.toronto.edu/~kriz/cifar.html\n\
         Direct download: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    )
}

/// Returns a platform-specific hint for BLAS setup and performance tuning.
///
/// Uses compile-time `cfg` flags to detect the target OS and return tailored
/// guidance for configuring BLAS on that platform.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::utils::error_messages::blas_setup_hint;
///
/// let hint = blas_setup_hint();
/// assert!(!hint.is_empty());
/// ```
pub fn blas_setup_hint() -> String {
    #[cfg(target_os = "macos")]
    {
        "BLAS backend: macOS Accelerate framework (built-in, no installation needed).\n\
         Performance tip: set VECLIB_MAXIMUM_THREADS to use all CPU cores.\n\
         Example: VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin mnist_mlp\n\
         For best results also pass: RUSTFLAGS=\"-C target-cpu=native\""
            .to_string()
    }

    #[cfg(target_os = "linux")]
    {
        "BLAS backend: OpenBLAS.\n\
         Ensure the OpenBLAS shared library is installed:\n\
           Ubuntu/Debian: sudo apt install libopenblas-dev\n\
           RHEL/CentOS:   sudo yum install openblas-devel\n\
           Arch Linux:    sudo pacman -S openblas\n\
         For best results also pass: RUSTFLAGS=\"-C target-cpu=native\""
            .to_string()
    }

    #[cfg(target_os = "windows")]
    {
        "BLAS backend: OpenBLAS via vcpkg.\n\
         Install with: vcpkg install openblas:x64-windows\n\
         Then set the environment variable VCPKG_ROOT to your vcpkg installation.\n\
         For best results also pass: RUSTFLAGS=\"-C target-cpu=native\""
            .to_string()
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "BLAS backend: platform-specific.\n\
         Ensure a BLAS implementation (e.g., OpenBLAS) is installed on your system.\n\
         For best results also pass: RUSTFLAGS=\"-C target-cpu=native\""
            .to_string()
    }
}

/// Prints BLAS platform information to stdout at startup.
///
/// Call this at the beginning of a training binary to inform users which BLAS
/// backend is active and how to tune performance.
pub fn print_blas_info() {
    println!("{}", blas_setup_hint());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_message_contains_filename() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(msg.contains("data/train-images-idx3-ubyte"));
    }

    #[test]
    fn test_mnist_message_contains_download_info() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(msg.contains("docs/2b MNIST-Data-Setup.md"));
        assert!(msg.contains("http"));
    }

    #[test]
    fn test_mnist_message_lists_expected_files() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(msg.contains("train-images-idx3-ubyte"));
        assert!(msg.contains("train-labels-idx1-ubyte"));
        assert!(msg.contains("t10k-images-idx3-ubyte"));
        assert!(msg.contains("t10k-labels-idx1-ubyte"));
    }

    #[test]
    fn test_cifar10_message_contains_filename() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(msg.contains("data/cifar-10-batches-bin/data_batch_1.bin"));
    }

    #[test]
    fn test_cifar10_message_contains_download_info() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(msg.contains("docs/cifar10_dataset.md"));
        assert!(msg.contains("https://"));
    }

    #[test]
    fn test_cifar10_message_lists_batch_files() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(msg.contains("data_batch_1.bin"));
        assert!(msg.contains("data_batch_5.bin"));
        assert!(msg.contains("test_batch.bin"));
    }

    #[test]
    fn test_blas_setup_hint_non_empty() {
        let hint = blas_setup_hint();
        assert!(!hint.is_empty());
    }
}
