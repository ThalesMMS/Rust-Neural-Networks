use serde::Serialize;

/// How a binary expects to receive its config file path on the command line.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgStyle {
    /// `--config <path>`
    ConfigFlag,
    /// bare positional argument: `<bin> <path>`
    Positional,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataRequirement {
    Mnist,
    Cifar10,
    None,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDescriptor {
    /// Stable identifier used by the frontend and by Tauri commands.
    pub id: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    /// Cargo `--bin` name (also the filename under `target/release/`).
    pub bin_name: &'static str,
    /// Default config JSON, relative to the project root. `None` when the
    /// binary has no config file (hyperparameters are hardcoded consts).
    pub default_config_path: Option<&'static str>,
    /// For `cifar10_cnn` only: the default `--arch <path>` architecture file.
    pub default_arch_path: Option<&'static str>,
    pub arg_style: ArgStyle,
    pub supports_step: bool,
    /// Whether the binary accepts `--run-name`/`--registry-dir`/`--seed` and
    /// writes a `runs/<id>/run.json` record.
    pub supports_registry_flags: bool,
    pub data_requirement: DataRequirement,
    /// Checkpoint filenames this binary writes, relative to the project root.
    pub checkpoints: &'static [&'static str],
    /// Known quirks/limitations surfaced verbatim in the UI.
    pub caveats: &'static [&'static str],
}

pub const MODEL_REGISTRY: &[ModelDescriptor] = &[
    ModelDescriptor {
        id: "xor",
        display_name: "XOR MLP",
        description: "2\u{2192}4\u{2192}1 toy network with sigmoid activations. Trains in seconds — the fastest way to sanity-check the whole pipeline.",
        category: "Toy example",
        bin_name: "mlp_simple",
        default_config_path: Some("config/training/mlp_simple_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::None,
        checkpoints: &[],
        caveats: &["No checkpoint or CSV log is produced by this binary — progress is only visible in the console."],
    },
    ModelDescriptor {
        id: "mnist_mlp",
        display_name: "MNIST MLP",
        description: "784\u{2192}512\u{2192}10 fully-connected classifier, BLAS-accelerated. The most complete, best-documented model in the repo.",
        category: "MNIST classification",
        bin_name: "mnist_mlp",
        default_config_path: Some("config/training/mnist_mlp_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: true,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_model_best.bin", "mnist_model.bin"],
        caveats: &[],
    },
    ModelDescriptor {
        id: "mnist_cnn",
        display_name: "MNIST CNN",
        description: "Conv(8, 3×3) + ReLU + 2×2 max-pool + fully-connected classifier, implemented with manual convolution loops.",
        category: "MNIST classification",
        bin_name: "mnist_cnn",
        default_config_path: Some("config/training/mnist_cnn_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: true,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_cnn_model_best.bin", "mnist_cnn_model_final.bin"],
        caveats: &[],
    },
    ModelDescriptor {
        id: "mnist_attention",
        display_name: "MNIST Attention",
        description: "Splits each digit into 49 patches (7×7 grid) and runs single-head self-attention + a feed-forward network, mean-pooled into a classifier.",
        category: "MNIST classification",
        bin_name: "mnist_attention_pool",
        default_config_path: Some("config/training/mnist_attention_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::Positional,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_attention_model_best.bin"],
        caveats: &[
            "Shares its checkpoint filename (mnist_attention_model_best.bin) with Transformer MNIST — training one after the other overwrites the other's checkpoint.",
        ],
    },
    ModelDescriptor {
        id: "cifar10_cnn",
        display_name: "CIFAR-10 CNN",
        description: "Convolutional classifier for 32×32 RGB images. Architecture (layer stack) is defined by a separate JSON file, not hardcoded.",
        category: "CIFAR-10",
        bin_name: "cifar10_cnn",
        default_config_path: Some("config/training/cifar10_cnn_default.json"),
        default_arch_path: Some("config/architectures/cifar10_cnn_baseline.json"),
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Cifar10,
        checkpoints: &["cifar10_cnn_model_best.bin"],
        caveats: &[],
    },
    ModelDescriptor {
        id: "resnet_cifar10",
        display_name: "ResNet CIFAR-10",
        description: "ResNet-18-style residual network with batch normalization for CIFAR-10.",
        category: "CIFAR-10",
        bin_name: "resnet_cifar10",
        default_config_path: Some("config/training/resnet_cifar10_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Cifar10,
        checkpoints: &["resnet_cifar10_model_best.bin"],
        caveats: &[],
    },
    ModelDescriptor {
        id: "cifar10_vit",
        display_name: "CIFAR-10 ViT",
        description: "Vision Transformer: patch tokens through stacked self-attention encoder blocks for CIFAR-10 classification.",
        category: "CIFAR-10",
        bin_name: "cifar10_vit",
        default_config_path: Some("config/training/cifar10_vit_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::Positional,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Cifar10,
        checkpoints: &[],
        caveats: &["This binary does not save a model checkpoint — training only produces logs (no inference possible)."],
    },
    ModelDescriptor {
        id: "mnist_autoencoder",
        display_name: "MNIST Autoencoder",
        description: "784\u{2192}256\u{2192}64\u{2192}256\u{2192}784 autoencoder trained to reconstruct MNIST digits from a compressed bottleneck.",
        category: "Generative",
        bin_name: "mnist_autoencoder",
        default_config_path: Some("config/training/mnist_autoencoder_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_ae_model_best.bin", "mnist_ae_model_final.bin"],
        caveats: &[],
    },
    ModelDescriptor {
        id: "mnist_vae",
        display_name: "MNIST VAE",
        description: "Variational autoencoder with a 32-dimensional latent space, trained on the ELBO (reconstruction + KL divergence).",
        category: "Generative",
        bin_name: "mnist_vae",
        default_config_path: Some("config/training/mnist_vae_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_vae_model_best.bin"],
        caveats: &["The final-model save is metadata-only, not full weights — the _best.bin checkpoint is the only usable one."],
    },
    ModelDescriptor {
        id: "rnn_char_level",
        display_name: "Char-level RNN",
        description: "Character-level LSTM trained on a Hamlet excerpt. Periodically prints generated text samples during training.",
        category: "Sequence",
        bin_name: "rnn_char_level",
        default_config_path: None,
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::None,
        checkpoints: &[],
        caveats: &["No config file (hyperparameters are fixed in source) and no checkpoint is saved."],
    },
    ModelDescriptor {
        id: "transformer_mnist",
        display_name: "Transformer MNIST",
        description: "Full Transformer encoder (patch tokens + stacked self-attention blocks) classifying MNIST digits.",
        category: "MNIST classification",
        bin_name: "transformer_mnist",
        default_config_path: Some("config/training/transformer_mnist_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::Positional,
        supports_step: true,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_attention_model_best.bin", "transformer_mnist_model.bin"],
        caveats: &[
            "Shares mnist_attention_model_best.bin with MNIST Attention — training one after the other overwrites the other's checkpoint.",
        ],
    },
    ModelDescriptor {
        id: "mnist_gan",
        display_name: "MNIST GAN",
        description: "Generative adversarial network: a 100-dim-noise generator and a discriminator trained jointly on MNIST digits.",
        category: "Generative",
        bin_name: "mnist_gan",
        default_config_path: Some("config/training/mnist_gan_default.json"),
        default_arch_path: None,
        arg_style: ArgStyle::ConfigFlag,
        supports_step: false,
        supports_registry_flags: false,
        data_requirement: DataRequirement::Mnist,
        checkpoints: &["mnist_gan_best.bin", "mnist_gan_final.bin"],
        caveats: &["Does not support step-through debug mode."],
    },
];

pub fn find_model(id: &str) -> Option<&'static ModelDescriptor> {
    MODEL_REGISTRY.iter().find(|m| m.id == id)
}

#[tauri::command]
pub fn list_models() -> Vec<ModelDescriptor> {
    MODEL_REGISTRY.to_vec()
}
