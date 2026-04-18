//! Tests for enhanced error messages for common issues.
//!
//! This file tests the error_messages module including:
//! - MNIST data-not-found messages with download instructions
//! - CIFAR-10 data-not-found messages with download instructions
//! - Error propagation from data loaders on missing files

use rust_neural_networks::data::cifar10::read_cifar10_batch;
use rust_neural_networks::data::mnist::{read_mnist_images, read_mnist_labels};
use rust_neural_networks::utils::error_messages::{
    cifar10_data_not_found_message, mnist_data_not_found_message,
};

// ============================================================================
// MNIST Error Message Content Tests
// ============================================================================

mod mnist_message_tests {
    use super::*;

    #[test]
    fn test_mnist_message_contains_filename() {
        let filename = "data/train-images-idx3-ubyte";
        let msg = mnist_data_not_found_message(filename);
        assert!(
            msg.contains(filename),
            "MNIST error message should include the attempted filename, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_contains_dataset_name() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(
            msg.contains("MNIST"),
            "MNIST error message should mention 'MNIST', got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_contains_download_instructions() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(
            msg.contains("docs/"),
            "MNIST error message should reference docs/ directory, got: {}",
            msg
        );
        assert!(
            msg.contains("http"),
            "MNIST error message should contain a URL (download link), got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_contains_setup_doc() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(
            msg.contains("docs/2b MNIST-Data-Setup.md"),
            "MNIST error message should reference setup documentation, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_lists_train_images() {
        let msg = mnist_data_not_found_message("data/some-file");
        assert!(
            msg.contains("train-images-idx3-ubyte"),
            "MNIST error message should list training images file, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_lists_train_labels() {
        let msg = mnist_data_not_found_message("data/some-file");
        assert!(
            msg.contains("train-labels-idx1-ubyte"),
            "MNIST error message should list training labels file, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_lists_test_images() {
        let msg = mnist_data_not_found_message("data/some-file");
        assert!(
            msg.contains("t10k-images-idx3-ubyte"),
            "MNIST error message should list test images file, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_lists_test_labels() {
        let msg = mnist_data_not_found_message("data/some-file");
        assert!(
            msg.contains("t10k-labels-idx1-ubyte"),
            "MNIST error message should list test labels file, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_contains_data_directory() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(
            msg.contains("data/"),
            "MNIST error message should mention the 'data/' directory, got: {}",
            msg
        );
    }

    #[test]
    fn test_mnist_message_with_different_filenames() {
        let files = [
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte",
            "data/t10k-images-idx3-ubyte",
            "data/t10k-labels-idx1-ubyte",
        ];

        for file in &files {
            let msg = mnist_data_not_found_message(file);
            assert!(
                msg.contains(file),
                "MNIST error message should contain the attempted path '{}', got: {}",
                file,
                msg
            );
            assert!(
                msg.contains("MNIST"),
                "MNIST error message should contain 'MNIST' for path '{}', got: {}",
                file,
                msg
            );
        }
    }

    #[test]
    fn test_mnist_message_is_non_empty() {
        let msg = mnist_data_not_found_message("data/train-images-idx3-ubyte");
        assert!(!msg.is_empty(), "MNIST error message should not be empty");
    }
}

// ============================================================================
// CIFAR-10 Error Message Content Tests
// ============================================================================

mod cifar10_message_tests {
    use super::*;

    #[test]
    fn test_cifar10_message_contains_filename() {
        let filename = "data/cifar-10-batches-bin/data_batch_1.bin";
        let msg = cifar10_data_not_found_message(filename);
        assert!(
            msg.contains(filename),
            "CIFAR-10 error message should include the attempted filename, got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_contains_dataset_name() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("CIFAR-10"),
            "CIFAR-10 error message should mention 'CIFAR-10', got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_contains_download_instructions() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("docs/"),
            "CIFAR-10 error message should reference docs/ directory, got: {}",
            msg
        );
        assert!(
            msg.contains("https://"),
            "CIFAR-10 error message should contain a URL (download link), got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_contains_setup_doc() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("docs/cifar10_dataset.md"),
            "CIFAR-10 error message should reference setup documentation, got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_lists_batch_files() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("data_batch_1.bin"),
            "CIFAR-10 error message should list batch 1, got: {}",
            msg
        );
        assert!(
            msg.contains("data_batch_5.bin"),
            "CIFAR-10 error message should list batch 5, got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_lists_test_batch() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("test_batch.bin"),
            "CIFAR-10 error message should list test batch file, got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_contains_data_directory() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            msg.contains("data/cifar-10-batches-bin/"),
            "CIFAR-10 error message should mention the data directory, got: {}",
            msg
        );
    }

    #[test]
    fn test_cifar10_message_with_different_filenames() {
        let files = [
            "data/cifar-10-batches-bin/data_batch_1.bin",
            "data/cifar-10-batches-bin/data_batch_3.bin",
            "data/cifar-10-batches-bin/test_batch.bin",
        ];

        for file in &files {
            let msg = cifar10_data_not_found_message(file);
            assert!(
                msg.contains(file),
                "CIFAR-10 error message should contain the attempted path '{}', got: {}",
                file,
                msg
            );
            assert!(
                msg.contains("CIFAR-10"),
                "CIFAR-10 error message should contain 'CIFAR-10' for path '{}', got: {}",
                file,
                msg
            );
        }
    }

    #[test]
    fn test_cifar10_message_is_non_empty() {
        let msg = cifar10_data_not_found_message("data/cifar-10-batches-bin/data_batch_1.bin");
        assert!(
            !msg.is_empty(),
            "CIFAR-10 error message should not be empty"
        );
    }
}

// ============================================================================
// MNIST Data Loader Error Propagation Tests
// ============================================================================

mod mnist_loader_error_tests {
    use super::*;

    #[test]
    fn test_read_mnist_images_missing_file_returns_error() {
        let result = read_mnist_images("data/nonexistent-images-file.idx3");
        assert!(
            result.is_err(),
            "read_mnist_images should return Err for missing file"
        );
    }

    #[test]
    fn test_read_mnist_images_missing_file_error_contains_mnist() {
        let result = read_mnist_images("data/nonexistent-images-file.idx3");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("MNIST"),
            "Missing MNIST images file error should mention 'MNIST', got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_mnist_images_missing_file_error_contains_download_info() {
        let result = read_mnist_images("data/nonexistent-images-file.idx3");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("docs/"),
            "Missing MNIST images file error should reference docs/, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("http"),
            "Missing MNIST images file error should contain a URL, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_mnist_images_missing_file_error_contains_filename() {
        let missing_path = "data/no-such-file-images.idx3";
        let result = read_mnist_images(missing_path);
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains(missing_path),
            "Missing MNIST images error should contain the attempted path '{}', got: {}",
            missing_path,
            err_msg
        );
    }

    #[test]
    fn test_read_mnist_labels_missing_file_returns_error() {
        let result = read_mnist_labels("data/nonexistent-labels-file.idx1");
        assert!(
            result.is_err(),
            "read_mnist_labels should return Err for missing file"
        );
    }

    #[test]
    fn test_read_mnist_labels_missing_file_error_contains_mnist() {
        let result = read_mnist_labels("data/nonexistent-labels-file.idx1");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("MNIST"),
            "Missing MNIST labels file error should mention 'MNIST', got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_mnist_labels_missing_file_error_contains_download_info() {
        let result = read_mnist_labels("data/nonexistent-labels-file.idx1");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("docs/"),
            "Missing MNIST labels file error should reference docs/, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("http"),
            "Missing MNIST labels file error should contain a URL, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_mnist_labels_missing_file_error_contains_filename() {
        let missing_path = "data/no-such-file-labels.idx1";
        let result = read_mnist_labels(missing_path);
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains(missing_path),
            "Missing MNIST labels error should contain the attempted path '{}', got: {}",
            missing_path,
            err_msg
        );
    }
}

// ============================================================================
// CIFAR-10 Data Loader Error Propagation Tests
// ============================================================================

mod cifar10_loader_error_tests {
    use super::*;

    #[test]
    fn test_read_cifar10_batch_missing_file_returns_error() {
        let result = read_cifar10_batch("data/cifar-10-batches-bin/no_such_batch.bin");
        assert!(
            result.is_err(),
            "read_cifar10_batch should return Err for missing file"
        );
    }

    #[test]
    fn test_read_cifar10_batch_missing_file_error_contains_cifar10() {
        let result = read_cifar10_batch("data/cifar-10-batches-bin/no_such_batch.bin");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CIFAR-10"),
            "Missing CIFAR-10 batch file error should mention 'CIFAR-10', got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_cifar10_batch_missing_file_error_contains_download_info() {
        let result = read_cifar10_batch("data/cifar-10-batches-bin/no_such_batch.bin");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("docs/"),
            "Missing CIFAR-10 batch file error should reference docs/, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("https://"),
            "Missing CIFAR-10 batch file error should contain a URL, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_read_cifar10_batch_missing_file_error_contains_filename() {
        let missing_path = "data/cifar-10-batches-bin/no_such_batch.bin";
        let result = read_cifar10_batch(missing_path);
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains(missing_path),
            "Missing CIFAR-10 batch error should contain the attempted path '{}', got: {}",
            missing_path,
            err_msg
        );
    }

    #[test]
    fn test_read_cifar10_batch_missing_file_error_contains_setup_doc() {
        let result = read_cifar10_batch("data/cifar-10-batches-bin/no_such_batch.bin");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cifar10_dataset.md"),
            "Missing CIFAR-10 batch file error should reference setup doc, got: {}",
            err_msg
        );
    }
}
