use std::io::Write;
use tempfile::NamedTempFile;

fn write_temp_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("failed to create temp config");
    file.write_all(contents.as_bytes())
        .expect("failed to write temp config");
    file
}

#[allow(dead_code)]
mod mnist_mlp_bin {
    include!("../src/bin/mnist_mlp.rs");

    #[cfg(test)]
    mod coverage_tests {
        use super::*;
        use rust_neural_networks::utils::lr_scheduler::ConstantLR;
        use std::path::Path;
        use tempfile::tempdir;

        #[test]
        fn test_scheduler_from_args_without_config() {
            let learning_rate = 0.01;
            let epochs = 10;
            let mut scheduler = scheduler_from_args(learning_rate, epochs, None);
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_scheduler_from_args_step_decay() {
            let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.5
}"#;
            let temp = crate::write_temp_config(config_json);
            let learning_rate = 0.01;
            let epochs = 10;
            let config_path = temp.path().to_str().unwrap();
            let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(config_path));
            assert_eq!(scheduler.get_lr(), learning_rate);
            scheduler.step();
            assert_eq!(scheduler.get_lr(), learning_rate);
            scheduler.step();
            assert!((scheduler.get_lr() - learning_rate * 0.5).abs() < 1e-6);
        }

        #[test]
        fn test_train_single_sample_runs() {
            struct DirGuard {
                old: std::path::PathBuf,
            }

            impl Drop for DirGuard {
                fn drop(&mut self) {
                    let _ = std::env::set_current_dir(&self.old);
                }
            }

            let temp_dir = tempdir().expect("failed to create temp dir");
            let old_dir = std::env::current_dir().expect("failed to get cwd");
            let _guard = DirGuard { old: old_dir };
            std::env::set_current_dir(temp_dir.path()).expect("failed to set cwd");

            let mut rng = SimpleRng::new(1);
            let mut nn = initialize_network(&mut rng);

            let images = vec![0.0f32; NUM_INPUTS];
            let labels = vec![0u8; 1];
            let val_images = vec![0.0f32; NUM_INPUTS];
            let val_labels = vec![0u8; 1];

            let learning_rate = 0.01;
            let epochs = 1;
            let batch_size = 1;
            let early_stopping_patience = 3;
            let early_stopping_min_delta = 0.001;
            let mut scheduler = ConstantLR::new(learning_rate);
            let train_data = DataSet {
                images: &images,
                labels: &labels,
                num_samples: 1,
            };
            let val_data = DataSet {
                images: &val_images,
                labels: &val_labels,
                num_samples: 1,
            };
            let params = TrainHyperparams {
                learning_rate,
                epochs,
                batch_size,
                early_stopping_patience,
                early_stopping_min_delta,
                enable_augmentation: false,
                horizontal_flip_prob: None,
                random_crop_padding: None,
                brightness_jitter: None,
                contrast_jitter: None,
            };
            let mut aug_rng = SimpleRng::new(42);
            train(
                &mut nn,
                &train_data,
                &val_data,
                &mut rng,
                &mut scheduler,
                &params,
                &mut aug_rng,
            );

            assert!(Path::new("logs/training_loss_adam.txt").exists());
            assert!(Path::new("mnist_model_best.bin").exists());
        }
    }
}

#[allow(dead_code)]
mod mnist_cnn_bin {
    include!("../src/bin/mnist_cnn.rs");

    #[cfg(test)]
    mod coverage_tests {
        use super::*;

        #[test]
        fn test_scheduler_from_args_without_config() {
            let learning_rate = 0.01;
            let epochs = 3;
            let mut scheduler = scheduler_from_args(learning_rate, epochs, None);
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_scheduler_from_args_step_decay() {
            let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.5
}"#;
            let temp = crate::write_temp_config(config_json);
            let learning_rate = 0.01;
            let epochs = 3;
            let config_path = temp.path().to_str().unwrap();
            let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(config_path));
            scheduler.step();
            assert_eq!(scheduler.get_lr(), learning_rate);
            scheduler.step();
            assert!((scheduler.get_lr() - learning_rate * 0.5).abs() < 1e-6);
        }
    }
}

#[allow(dead_code)]
mod mlp_simple_bin {
    include!("../src/bin/mlp_simple.rs");

    #[cfg(test)]
    mod coverage_tests {
        use super::*;

        #[test]
        fn test_scheduler_from_args_without_config() {
            let learning_rate = 0.01;
            let epochs = 1000000;
            let mut scheduler = scheduler_from_args(learning_rate, epochs, None);
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_scheduler_from_args_step_decay() {
            let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.5
}"#;
            let temp = crate::write_temp_config(config_json);
            let learning_rate = 0.01;
            let epochs = 1000000;
            let config_path = temp.path().to_str().unwrap();
            let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(config_path));
            scheduler.step();
            assert_eq!(scheduler.get_lr(), learning_rate);
            scheduler.step();
            assert!((scheduler.get_lr() - learning_rate * 0.5).abs() < 1e-6);
        }
    }
}

#[allow(dead_code)]
mod cifar10_cnn_bin {
    include!("../src/bin/cifar10_cnn.rs");

    #[cfg(test)]
    mod coverage_tests {
        use super::*;

        #[test]
        fn test_scheduler_from_args_without_config() {
            let learning_rate = 0.01;
            let epochs = 10;
            let mut scheduler = scheduler_from_args(learning_rate, epochs, None);
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_scheduler_from_args_step_decay() {
            let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.5
}"#;
            let temp = crate::write_temp_config(config_json);
            let learning_rate = 0.01;
            let epochs = 10;
            let config_path = temp.path().to_str().unwrap();
            let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(config_path));
            scheduler.step();
            assert_eq!(scheduler.get_lr(), learning_rate);
            scheduler.step();
            assert!((scheduler.get_lr() - learning_rate * 0.5).abs() < 1e-6);
        }
    }
}

#[allow(dead_code)]
mod mnist_attention_pool_bin {
    include!("../src/bin/mnist_attention_pool.rs");

    #[cfg(test)]
    mod coverage_tests {
        use super::*;

        #[test]
        fn test_build_scheduler_without_config() {
            let mut scheduler = build_scheduler(None);
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_build_scheduler_step_decay() {
            let config_json = r#"{
  "scheduler_type": "step_decay",
  "step_size": 2,
  "gamma": 0.5
}"#;
            let temp = crate::write_temp_config(config_json);
            let mut scheduler = build_scheduler(Some(temp.path().to_str().unwrap()));
            scheduler.step();
            assert_eq!(scheduler.get_lr(), LEARNING_RATE);
            scheduler.step();
            assert!((scheduler.get_lr() - LEARNING_RATE * 0.5).abs() < 1e-6);
        }

        #[test]
        fn test_build_scheduler_exponential() {
            let config_json = r#"{
  "scheduler_type": "exponential",
  "decay_rate": 0.9
}"#;
            let temp = crate::write_temp_config(config_json);
            let mut scheduler = build_scheduler(Some(temp.path().to_str().unwrap()));
            scheduler.step();
            assert!((scheduler.get_lr() - LEARNING_RATE * 0.9).abs() < 1e-6);
        }

        #[test]
        fn test_build_scheduler_cosine() {
            let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0,
  "T_max": 4
}"#;
            let temp = crate::write_temp_config(config_json);
            let mut scheduler = build_scheduler(Some(temp.path().to_str().unwrap()));
            scheduler.step();
            assert!(scheduler.get_lr() < LEARNING_RATE);
        }

        #[test]
        fn test_build_scheduler_unknown_type_fallback() {
            let config_json = r#"{
  "scheduler_type": "unknown"
}"#;
            let temp = crate::write_temp_config(config_json);
            let mut scheduler = build_scheduler(Some(temp.path().to_str().unwrap()));
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }

        #[test]
        fn test_build_scheduler_load_error_fallback() {
            let mut scheduler = build_scheduler(Some("does_not_exist.json"));
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
        }
    }
}
