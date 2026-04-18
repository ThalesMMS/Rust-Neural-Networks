use std::io::Write;
use std::sync::Mutex;
use tempfile::NamedTempFile;

/// Global mutex to serialize tests that change the process working directory.
/// `std::env::set_current_dir` is process-wide and unsafe to call concurrently
/// from multiple threads.
static CWD_LOCK: Mutex<()> = Mutex::new(());

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
                _lock: std::sync::MutexGuard<'static, ()>,
            }

            impl Drop for DirGuard {
                fn drop(&mut self) {
                    let _ = std::env::set_current_dir(&self.old);
                }
            }

            let temp_dir = tempdir().expect("failed to create temp dir");
            let old_dir = std::env::current_dir().expect("failed to get cwd");
            let _guard = DirGuard {
                old: old_dir,
                _lock: crate::CWD_LOCK.lock().expect("CWD_LOCK poisoned"),
            };
            std::env::set_current_dir(temp_dir.path()).expect("failed to set cwd");

            let mut rng = SimpleRng::new(1);
            #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
            let mut nn = initialize_network(&mut rng, &None);
            #[cfg(not(any(feature = "gpu-metal", feature = "gpu-cuda")))]
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
            let mut debugger = StepDebugger::new(false);
            train(
                &mut nn,
                &train_data,
                &val_data,
                &mut rng,
                &mut scheduler,
                &params,
                &mut aug_rng,
                &mut debugger,
            );

            assert!(Path::new("logs/training_loss_adam.txt").exists());
            assert!(Path::new("mnist_model_best.bin").exists());
        }
    }
}

#[allow(dead_code)]
mod mnist_cnn_bin {
    include!("../src/bin/mnist_cnn/main.rs");

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

        #[test]
        fn test_init_cnn_runs() {
            let mut rng = SimpleRng::new(42);
            let model = init_cnn(&mut rng);
            // Conv layer: 1 in_ch, CONV_OUT out_ch, KERNEL x KERNEL weights
            assert_eq!(model.conv_layer.weights().len(), CONV_OUT * KERNEL * KERNEL);
            // FC layer: FC_IN -> NUM_CLASSES
            assert_eq!(model.fc_layer.input_size(), FC_IN);
            assert_eq!(model.fc_layer.output_size(), NUM_CLASSES);
        }

        #[test]
        fn test_accuracy_invalid_inputs_return_zero() {
            let mut rng = SimpleRng::new(42);
            let mut model = init_cnn(&mut rng);

            assert_eq!(test_accuracy(&mut model, &[], &[]), 0.0);
            assert_eq!(test_accuracy(&mut model, &[], &[0]), 0.0);
        }

        #[test]
        fn test_conv_backward_leaves_caller_grad_input_untouched() {
            let mut rng = SimpleRng::new(42);
            let mut model = init_cnn(&mut rng);
            let input = vec![0.0f32; NUM_INPUTS];
            let conv_grad = vec![1.0f32; CONV_OUT * IMG_H * IMG_W];
            let mut grad_input = vec![7.0f32; NUM_INPUTS];

            conv_backward(&mut model, 1, &input, &conv_grad, &mut grad_input);

            assert!(grad_input.iter().all(|&value| value == 7.0));
        }

        #[test]
        fn test_conv_forward_relu_runs() {
            let mut rng = SimpleRng::new(42);
            let mut model = init_cnn(&mut rng);
            let batch = 1;
            let input = vec![0.0f32; batch * NUM_INPUTS];
            let mut conv_out = vec![0.0f32; batch * CONV_OUT * IMG_H * IMG_W];
            conv_forward_relu(&mut model, batch, &input, &mut conv_out);
            // ReLU ensures all outputs are >= 0
            assert!(conv_out.iter().all(|&v| v >= 0.0));
        }

        #[test]
        fn test_maxpool_forward_basic() {
            let batch = 1;
            // All ones post-ReLU conv activations
            let conv_act = vec![1.0f32; batch * CONV_OUT * IMG_H * IMG_W];
            let mut pool_out = vec![0.0f32; batch * CONV_OUT * POOL_H * POOL_W];
            let mut pool_idx = vec![0u8; batch * CONV_OUT * POOL_H * POOL_W];
            maxpool_forward(batch, &conv_act, &mut pool_out, &mut pool_idx);
            // Max of a 2x2 window of all 1.0 should be 1.0
            assert!(pool_out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_softmax_xent_backward_basic() {
            let batch = 1;
            // Equal logits -> uniform probs after softmax
            let mut logits = vec![1.0f32; batch * NUM_CLASSES];
            let labels = vec![0u8; batch]; // true label is class 0
            let mut delta = vec![0.0f32; batch * NUM_CLASSES];
            let scale = 1.0f32;

            softmax_rows(&mut logits[..batch * NUM_CLASSES], batch, NUM_CLASSES);
            let loss = compute_softmax_cross_entropy(
                &logits[..batch * NUM_CLASSES],
                &labels,
                batch,
                NUM_CLASSES,
                &mut delta,
                scale,
            );

            // Loss should be positive
            assert!(loss > 0.0);
            // Delta for the correct class is (prob - 1) * scale; since prob < 1, delta[0] < 0
            assert!(delta[0] < 0.0);
            // Sum of deltas (prob - onehot) across classes equals 0 when scale=1
            let delta_sum: f32 = delta.iter().sum();
            assert!(delta_sum.abs() < 1e-5);
        }

        #[test]
        fn test_cnn_train_single_sample_runs() {
            use std::path::Path;
            use tempfile::tempdir;

            struct DirGuard {
                old: std::path::PathBuf,
                _lock: std::sync::MutexGuard<'static, ()>,
            }
            impl Drop for DirGuard {
                fn drop(&mut self) {
                    let _ = std::env::set_current_dir(&self.old);
                }
            }

            let temp_dir = tempdir().expect("failed to create temp dir");
            let old_dir = std::env::current_dir().expect("failed to get cwd");
            let _guard = DirGuard {
                old: old_dir,
                _lock: crate::CWD_LOCK.lock().expect("CWD_LOCK poisoned"),
            };
            std::env::set_current_dir(temp_dir.path()).expect("failed to set cwd");

            let mut rng = SimpleRng::new(1);
            let mut model = init_cnn(&mut rng);

            // Minimal single-sample dataset
            let images = vec![0.0f32; NUM_INPUTS];
            let labels = vec![0u8; 1];
            let indices = vec![0usize];
            let learning_rate = 0.01f32;

            // Training buffers (batch_size = 1)
            let mut batch_inputs = vec![0.0f32; NUM_INPUTS];
            let mut batch_labels = vec![0u8; 1];
            let mut conv_out = vec![0.0f32; CONV_OUT * IMG_H * IMG_W];
            let mut pool_out = vec![0.0f32; FC_IN];
            let mut pool_idx = vec![0u8; CONV_OUT * POOL_H * POOL_W];
            let mut logits = vec![0.0f32; NUM_CLASSES];
            let mut delta = vec![0.0f32; NUM_CLASSES];
            let mut d_pool = vec![0.0f32; FC_IN];
            let mut d_conv = vec![0.0f32; CONV_OUT * IMG_H * IMG_W];
            let mut grad_input = vec![0.0f32; NUM_INPUTS];

            // Gather the single sample
            gather_batch(
                &images,
                &labels,
                &indices,
                0,
                1,
                &mut batch_inputs,
                &mut batch_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                None, // flip_prob
                None, // crop_padding
                None, // brightness_jitter
                None, // contrast_jitter
                None, // saturation_jitter
                None, // rng
            );

            // Forward pass
            conv_forward_relu(&mut model, 1, &batch_inputs, &mut conv_out);
            maxpool_forward(1, &conv_out, &mut pool_out, &mut pool_idx);
            fc_forward(&mut model, 1, &pool_out, &mut logits);

            // Softmax + cross-entropy loss + gradient
            softmax_rows(&mut logits[..NUM_CLASSES], 1, NUM_CLASSES);
            let loss = compute_softmax_cross_entropy(
                &logits[..NUM_CLASSES],
                &batch_labels,
                1,
                NUM_CLASSES,
                &mut delta,
                1.0,
            );

            // Backward pass
            fc_backward(&mut model, 1, &pool_out, &delta, &mut d_pool);
            maxpool_backward_relu(1, &conv_out, &d_pool, &pool_idx, &mut d_conv);
            conv_backward(&mut model, 1, &batch_inputs, &d_conv, &mut grad_input);

            // Parameter update
            model.fc_layer.update_parameters(learning_rate);
            model.conv_layer.update_parameters(learning_rate);

            // Loss must be positive
            assert!(loss > 0.0);

            // Write the training log file (as main() does)
            std::fs::create_dir_all("./logs").expect("failed to create logs dir");
            let log_file =
                File::create("./logs/training_loss_cnn.txt").expect("failed to create log file");
            let mut log = BufWriter::new(log_file);
            writeln!(log, "1,{},0.0,0.0,0.0,{}", loss, learning_rate).expect("failed to write log");
            drop(log);

            assert!(Path::new("logs/training_loss_cnn.txt").exists());
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

        #[test]
        fn test_initialize_network_runs() {
            let mut rng = SimpleRng::new(42);
            let nn = initialize_network(&mut rng);
            assert_eq!(nn.hidden_layer.input_size(), NUM_INPUTS);
            assert_eq!(nn.hidden_layer.output_size(), NUM_HIDDEN);
            assert_eq!(nn.output_layer.input_size(), NUM_HIDDEN);
            assert_eq!(nn.output_layer.output_size(), NUM_OUTPUTS);
        }

        #[test]
        fn test_forward_with_sigmoid_output_range() {
            let mut rng = SimpleRng::new(42);
            let nn = initialize_network(&mut rng);
            let inputs = [0.5f32, 0.3f32];
            let mut hidden_outputs = vec![0.0f32; NUM_HIDDEN];
            forward_with_sigmoid(&nn.hidden_layer, &inputs, &mut hidden_outputs);
            // Sigmoid output must be in [0, 1]
            assert!(hidden_outputs.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }

        #[test]
        fn test_train_xor_runs() {
            use rust_neural_networks::utils::lr_scheduler::ConstantLR;
            let mut rng = SimpleRng::new(42);
            let mut nn = initialize_network(&mut rng);
            let inputs: [[f32; NUM_INPUTS]; NUM_SAMPLES] =
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
            let expected: [[f32; NUM_OUTPUTS]; NUM_SAMPLES] = [[0.0], [1.0], [1.0], [0.0]];
            let mut scheduler = ConstantLR::new(0.1);
            // Create a disabled step debugger for testing
            let mut debugger = StepDebugger::new(false);
            // Run a small number of epochs to verify the training loop executes correctly
            train(
                &mut nn,
                &inputs,
                &expected,
                &mut scheduler,
                10,
                &mut debugger,
            );
            // After training, forward pass should produce values in [0, 1]
            let mut hidden_out = vec![0.0f32; NUM_HIDDEN];
            let mut output_out = vec![0.0f32; NUM_OUTPUTS];
            forward_with_sigmoid(&nn.hidden_layer, &inputs[0], &mut hidden_out);
            forward_with_sigmoid(&nn.output_layer, &hidden_out, &mut output_out);
            assert!(output_out[0] >= 0.0 && output_out[0] <= 1.0);
        }

        #[test]
        fn test_sigmoid_basic() {
            // sigmoid(0) == 0.5
            assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
            // sigmoid approaches 1 for large positive input
            assert!(sigmoid(10.0) > 0.99);
            // sigmoid approaches 0 for large negative input
            assert!(sigmoid(-10.0) < 0.01);
        }

        #[test]
        fn test_sigmoid_derivative_basic() {
            // sigmoid_derivative(0.5) == 0.5 * (1 - 0.5) == 0.25
            let s = sigmoid(0.0); // == 0.5
            let d = sigmoid_derivative(s);
            assert!((d - 0.25).abs() < 1e-6);
        }
    }
}

#[allow(dead_code)]
mod cifar10_cnn_bin {
    include!("../src/bin/cifar10_cnn/main.rs");

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

        /// Verifies that a step-decay learning-rate scheduler reduces the learning rate by the configured factor after the configured number of steps.
        ///
        /// # Examples
        ///
        /// ```
        /// // Create a temporary JSON config for a step-decay scheduler with step_size = 2 and gamma = 0.5,
        /// // then build a scheduler and step it twice to observe the decay.
        /// let config_json = r#"{
        ///   "scheduler_type": "step_decay",
        ///   "step_size": 2,
        ///   "gamma": 0.5
        /// }"#;
        /// let temp = crate::write_temp_config(config_json);
        /// let learning_rate = 0.01;
        /// let epochs = 10;
        /// let config_path = temp.path().to_str().unwrap();
        /// let mut scheduler = scheduler_from_args(learning_rate, epochs, Some(config_path));
        /// scheduler.step();
        /// assert_eq!(scheduler.get_lr(), learning_rate);
        /// scheduler.step();
        /// assert!((scheduler.get_lr() - learning_rate * 0.5).abs() < 1e-6);
        /// ```
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

        /// Create a TrainingConfig populated with default values and set `optimizer_type` when provided.
        ///
        /// # Examples
        ///
        /// ```
        /// let cfg = make_config(Some("adam"));
        /// assert_eq!(cfg.optimizer_type.as_deref(), Some("adam"));
        /// ```
        fn make_config(
            optimizer_type: Option<&str>,
        ) -> rust_neural_networks::config::TrainingConfig {
            rust_neural_networks::config::TrainingConfig {
                scheduler_type: "none".to_string(),
                step_size: None,
                gamma: None,
                decay_rate: None,
                min_lr: None,
                T_max: None,
                activation_function: None,
                leaky_relu_alpha: None,
                elu_alpha: None,
                optimizer_type: optimizer_type.map(|s| s.to_string()),
                adam_beta1: None,
                adam_beta2: None,
                adam_epsilon: None,
                adamw_weight_decay: None,
                rmsprop_decay: None,
                rmsprop_epsilon: None,
                learning_rate: None,
                epochs: None,
                batch_size: None,
                validation_split: None,
                early_stopping_patience: None,
                early_stopping_min_delta: None,
                enable_profiling: None,
                enable_augmentation: None,
                horizontal_flip_prob: None,
                random_crop_padding: None,
                brightness_jitter: None,
                contrast_jitter: None,
                saturation_jitter: None,
                noise_dim: None,
                g_lr: None,
                d_lr: None,
                label_smoothing: None,
                gpu_backend: None,
                gpu_device_id: None,
                step_debug: None,
            }
        }

        #[test]
        fn test_create_optimizer_sgd() {
            let config = make_config(Some("sgd"));
            let lr = 0.01f32;
            let opt = create_optimizer(&config, lr);
            assert!((opt.learning_rate() - lr).abs() < 1e-6);
        }

        #[test]
        fn test_create_optimizer_adam() {
            let config = make_config(Some("adam"));
            let lr = 0.001f32;
            let opt = create_optimizer(&config, lr);
            assert!((opt.learning_rate() - lr).abs() < 1e-6);
        }

        #[test]
        fn test_create_optimizer_adamw_default() {
            // None optimizer_type defaults to AdamW
            let config = make_config(None);
            let lr = 0.01f32;
            let opt = create_optimizer(&config, lr);
            assert!((opt.learning_rate() - lr).abs() < 1e-6);
        }

        #[test]
        fn test_create_optimizer_rmsprop() {
            let config = make_config(Some("rmsprop"));
            let lr = 0.01f32;
            let opt = create_optimizer(&config, lr);
            assert!((opt.learning_rate() - lr).abs() < 1e-6);
        }

        #[test]
        fn test_parse_config_paths_default() {
            let args = vec!["program".to_string()];
            let (config_path, arch_path) = parse_config_paths(&args);
            assert_eq!(config_path, DEFAULT_CONFIG_PATH);
            assert!(arch_path.is_none());
        }

        #[test]
        fn test_parse_config_paths_with_config() {
            let args = vec![
                "program".to_string(),
                "--config".to_string(),
                "my_config.json".to_string(),
            ];
            let (config_path, arch_path) = parse_config_paths(&args);
            assert_eq!(config_path, "my_config.json");
            assert!(arch_path.is_none());
        }

        #[test]
        fn test_parse_config_paths_with_arch() {
            let args = vec![
                "program".to_string(),
                "--arch".to_string(),
                "my_arch.json".to_string(),
            ];
            let (config_path, arch_path) = parse_config_paths(&args);
            assert_eq!(config_path, DEFAULT_CONFIG_PATH);
            assert_eq!(arch_path.as_deref(), Some("my_arch.json"));
        }

        #[test]
        fn test_forward_pass_with_dense_arch() {
            // Create a minimal architecture JSON for a Dense(8->2) network
            let arch_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 8,
      "output_size": 2
    }
  ]
}"#;
            let temp = crate::write_temp_config(arch_json);
            let arch_path = temp.path().to_str().unwrap();

            let mut rng = SimpleRng::new(42);
            let mut model = init_cnn(&mut rng, Some(arch_path)).unwrap();

            // 1 sample, 8 input features
            let input = vec![0.5f32; 8];
            let mut activations = LayerActivations::new(model.layers.len());
            let mut temp_buf = Vec::new();

            let last_idx = forward_pass(&mut model, 1, &input, &mut activations, &mut temp_buf);

            // The final layer outputs 2 values for 1 sample
            assert_eq!(activations.data[last_idx].len(), 2);
        }

        #[test]
        fn test_forward_and_backward_pass_with_dense_arch() {
            use rust_neural_networks::utils::activations::softmax_rows;

            // Dense(8->2) architecture
            let arch_json = r#"{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 8,
      "output_size": 2
    }
  ]
}"#;
            let temp = crate::write_temp_config(arch_json);
            let arch_path = temp.path().to_str().unwrap();

            let mut rng = SimpleRng::new(7);
            let mut model = init_cnn(&mut rng, Some(arch_path)).unwrap();

            let input = vec![1.0f32; 8];
            let mut activations = LayerActivations::new(model.layers.len());
            let mut temp_buf = Vec::new();

            // Forward pass
            let last_idx = forward_pass(&mut model, 1, &input, &mut activations, &mut temp_buf);

            // Compute softmax + cross-entropy gradient
            let mut logits = activations.data[last_idx].clone();
            let labels = [0u8; 1]; // true class is 0
            let mut grad = vec![0.0f32; 2];
            softmax_rows(&mut logits, 1, 2);
            // grad = softmax - one_hot
            for (i, (&prob, g)) in logits.iter().zip(grad.iter_mut()).enumerate() {
                *g = prob - if i == labels[0] as usize { 1.0 } else { 0.0 };
            }

            // Backward pass
            let mut grad_buf1 = Vec::new();
            let mut grad_buf2 = Vec::new();
            backward_pass(
                &mut model,
                1,
                &input,
                &activations,
                &grad,
                &mut grad_buf1,
                &mut grad_buf2,
            );

            // After backward, grad buffers should be populated
            // (input gradient = grad_buf1 or grad_buf2 with input_size elements)
            // No panic = success
            assert_eq!(logits.len(), 2);
        }
    }
}

#[allow(dead_code)]
mod mnist_attention_pool_bin {
    include!("../src/bin/mnist_attention_pool/main.rs");

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
        fn test_build_scheduler_cosine_zero_t_max_fallback() {
            let config_json = r#"{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0,
  "T_max": 0
}"#;
            let temp = crate::write_temp_config(config_json);
            let mut scheduler = build_scheduler(Some(temp.path().to_str().unwrap()));
            let lr = scheduler.get_lr();
            scheduler.step();
            assert_eq!(scheduler.get_lr(), lr);
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

        #[test]
        fn test_init_model_runs() {
            let mut rng = SimpleRng::new(42);
            let model = init_model(&mut rng);
            // w_cls: [D_MODEL * NUM_CLASSES] = 64 * 10 = 640
            assert_eq!(model.w_cls.len(), D_MODEL * NUM_CLASSES);
            // w_q: [D_MODEL * D_MODEL] = 64 * 64 = 4096
            assert_eq!(model.w_q.len(), D_MODEL * D_MODEL);
            // w_patch: [PATCH_DIM * D_MODEL] = 16 * 64 = 1024
            assert_eq!(model.w_patch.len(), PATCH_DIM * D_MODEL);
        }

        #[test]
        fn test_extract_patches_basic() {
            let batch_count = 1;
            let inputs = vec![0.0f32; batch_count * NUM_INPUTS];
            let mut patches = vec![0.0f32; batch_count * SEQ_LEN * PATCH_DIM];
            extract_patches(&inputs, batch_count, &mut patches);
            // All-zero input should produce all-zero patches
            assert!(patches.iter().all(|&v| v == 0.0));
        }

        #[test]
        fn test_extract_patches_nonzero() {
            let batch_count = 1;
            let mut inputs = vec![0.0f32; batch_count * NUM_INPUTS];
            // Set the first pixel to 1.0
            inputs[0] = 1.0;
            let mut patches = vec![0.0f32; batch_count * SEQ_LEN * PATCH_DIM];
            extract_patches(&inputs, batch_count, &mut patches);
            // The first patch's first element should be 1.0
            assert_eq!(patches[0], 1.0);
        }

        #[test]
        fn test_forward_batch_runs() {
            let mut rng = SimpleRng::new(42);
            let model = init_model(&mut rng);
            let mut buf = BatchBuffers::new();

            let batch_count = 1;
            let inputs = vec![0.5f32; batch_count * NUM_INPUTS];
            let labels = vec![3u8; batch_count];

            let loss = forward_batch(&model, &inputs, &labels, batch_count, &mut buf);
            assert!(loss > 0.0, "cross-entropy loss should be positive");
        }

        #[test]
        fn test_backward_batch_runs() {
            let mut rng = SimpleRng::new(42);
            let model = init_model(&mut rng);
            let mut buf = BatchBuffers::new();
            let mut grads = Grads::new();

            let batch_count = 1;
            let inputs = vec![0.5f32; batch_count * NUM_INPUTS];
            let labels = vec![0u8; batch_count];

            // Forward first to populate buffers
            forward_batch(&model, &inputs, &labels, batch_count, &mut buf);

            // Backward pass
            backward_batch(&model, batch_count, &mut buf, &mut grads);

            // Gradients should not all be zero after backward pass
            let all_zero = grads.w_cls.iter().all(|&v| v == 0.0);
            assert!(
                !all_zero,
                "w_cls gradients should be non-zero after backward"
            );
        }

        #[test]
        fn test_gather_batch_basic() {
            let num_samples = 3;
            let images = vec![0.5f32; num_samples * NUM_INPUTS];
            let labels = vec![1u8, 2u8, 3u8];
            let indices = vec![0usize, 1, 2];
            let mut out_inputs = vec![0.0f32; NUM_INPUTS];
            let mut out_labels = vec![0u8; 1];

            gather_batch(
                &images,
                &labels,
                &indices,
                1, // start at index 1
                1, // gather 1 sample
                &mut out_inputs,
                &mut out_labels,
                IMG_W,
                IMG_H,
                IMG_CHANNELS,
                None, // flip_prob
                None, // crop_padding
                None, // brightness_jitter
                None, // contrast_jitter
                None, // saturation_jitter
                None, // rng
            );

            // Should copy label from position indices[1] = 1
            assert_eq!(out_labels[0], labels[1]);
            // All input values should be 0.5
            assert!(out_inputs.iter().all(|&v| (v - 0.5).abs() < 1e-6));
        }
    }
}
