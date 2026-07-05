// Static, hand-authored mirror of `TrainingConfig` (src/config.rs). Kept as
// data rather than generated from the Rust struct — the struct is small and
// stable, and a schema-generation pipeline would be more machinery than a v1
// UI needs. Nested optional objects (warmup, cyclical_lr, regularization,
// gradient_clipping) are intentionally NOT modeled here; they're rare, and
// power users can edit them directly as JSON in the Configs tab. Saving a
// config here only ever touches the scalar keys below and preserves
// anything else already present in the file untouched.

export type ConfigValue = Record<string, unknown>;

export interface FieldOption {
  value: string;
  label: string;
}

export interface Field {
  key: string;
  label: string;
  type: "number" | "text" | "select" | "boolean";
  options?: FieldOption[];
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
  hint?: string;
  integer?: boolean;
  showIf?: (cfg: ConfigValue) => boolean;
}

export interface Section {
  title: string;
  description?: string;
  /** Hide this whole section unless the model id is in this list. Omit to always show. */
  onlyForModelIds?: string[];
  fields: Field[];
}

const isOptimizer = (names: string[]) => (cfg: ConfigValue) =>
  names.includes(String(cfg.optimizer_type ?? "sgd"));

export const SECTIONS: Section[] = [
  {
    title: "Core hyperparameters",
    fields: [
      { key: "learning_rate", label: "Learning rate", type: "number", step: 0.0001, min: 0 },
      { key: "epochs", label: "Epochs", type: "number", integer: true, min: 1 },
      { key: "batch_size", label: "Batch size", type: "number", integer: true, min: 1 },
      { key: "validation_split", label: "Validation split", type: "number", step: 0.01, min: 0, max: 1 },
      {
        key: "early_stopping_patience",
        label: "Early stopping patience (epochs)",
        type: "number",
        integer: true,
        min: 0,
      },
      { key: "early_stopping_min_delta", label: "Early stopping min delta", type: "number", step: 0.0001, min: 0 },
    ],
  },
  {
    title: "Learning rate scheduler",
    fields: [
      {
        key: "scheduler_type",
        label: "Scheduler",
        type: "select",
        options: [
          { value: "step_decay", label: "Step decay" },
          { value: "exponential", label: "Exponential decay" },
          { value: "cosine_annealing", label: "Cosine annealing" },
          { value: "constant", label: "Constant (no scheduler)" },
        ],
      },
      {
        key: "step_size",
        label: "Step size (epochs between drops)",
        type: "number",
        integer: true,
        min: 1,
        showIf: (c) => c.scheduler_type === "step_decay",
      },
      {
        key: "gamma",
        label: "Gamma (LR multiplier)",
        type: "number",
        step: 0.01,
        min: 0,
        showIf: (c) => c.scheduler_type === "step_decay",
      },
      {
        key: "decay_rate",
        label: "Decay rate (per epoch)",
        type: "number",
        step: 0.01,
        min: 0,
        showIf: (c) => c.scheduler_type === "exponential",
      },
      {
        key: "min_lr",
        label: "Minimum learning rate",
        type: "number",
        step: 0.0001,
        min: 0,
        showIf: (c) => c.scheduler_type === "cosine_annealing",
      },
      {
        key: "T_max",
        label: "T_max (annealing cycle epochs)",
        type: "number",
        integer: true,
        min: 1,
        showIf: (c) => c.scheduler_type === "cosine_annealing",
      },
    ],
  },
  {
    title: "Activation function",
    fields: [
      {
        key: "activation_function",
        label: "Activation",
        type: "select",
        options: [
          { value: "relu", label: "ReLU" },
          { value: "leaky_relu", label: "Leaky ReLU" },
          { value: "elu", label: "ELU" },
          { value: "gelu", label: "GELU" },
          { value: "swish", label: "Swish" },
          { value: "tanh", label: "Tanh" },
        ],
      },
      {
        key: "leaky_relu_alpha",
        label: "Leaky ReLU alpha",
        type: "number",
        step: 0.001,
        min: 0,
        showIf: (c) => c.activation_function === "leaky_relu",
      },
      {
        key: "elu_alpha",
        label: "ELU alpha",
        type: "number",
        step: 0.1,
        min: 0,
        showIf: (c) => c.activation_function === "elu",
      },
    ],
  },
  {
    title: "Optimizer",
    fields: [
      {
        key: "optimizer_type",
        label: "Optimizer",
        type: "select",
        options: [
          { value: "sgd", label: "SGD" },
          { value: "adam", label: "Adam" },
          { value: "adamw", label: "AdamW" },
          { value: "rmsprop", label: "RMSprop" },
        ],
      },
      {
        key: "adam_beta1",
        label: "Beta1",
        type: "number",
        step: 0.001,
        min: 0,
        max: 1,
        showIf: isOptimizer(["adam", "adamw"]),
      },
      {
        key: "adam_beta2",
        label: "Beta2",
        type: "number",
        step: 0.0001,
        min: 0,
        max: 1,
        showIf: isOptimizer(["adam", "adamw"]),
      },
      {
        key: "adam_epsilon",
        label: "Epsilon",
        type: "number",
        step: 1e-9,
        min: 0,
        showIf: isOptimizer(["adam", "adamw"]),
      },
      {
        key: "adamw_weight_decay",
        label: "Weight decay",
        type: "number",
        step: 0.001,
        min: 0,
        showIf: isOptimizer(["adamw"]),
      },
      {
        key: "rmsprop_decay",
        label: "Decay",
        type: "number",
        step: 0.01,
        min: 0,
        max: 1,
        showIf: isOptimizer(["rmsprop"]),
      },
      {
        key: "rmsprop_epsilon",
        label: "Epsilon",
        type: "number",
        step: 1e-9,
        min: 0,
        showIf: isOptimizer(["rmsprop"]),
      },
    ],
  },
  {
    title: "Data augmentation",
    description: "Only applied by binaries that read these fields (MNIST MLP/CNN, CIFAR-10 CNN).",
    fields: [
      { key: "enable_augmentation", label: "Enable augmentation", type: "boolean" },
      {
        key: "horizontal_flip_prob",
        label: "Horizontal flip probability",
        type: "number",
        step: 0.01,
        min: 0,
        max: 1,
        showIf: (c) => Boolean(c.enable_augmentation),
      },
      {
        key: "random_crop_padding",
        label: "Random crop padding (px)",
        type: "number",
        integer: true,
        min: 0,
        showIf: (c) => Boolean(c.enable_augmentation),
      },
      {
        key: "brightness_jitter",
        label: "Brightness jitter",
        type: "number",
        step: 0.01,
        min: 0,
        showIf: (c) => Boolean(c.enable_augmentation),
      },
      {
        key: "contrast_jitter",
        label: "Contrast jitter",
        type: "number",
        step: 0.01,
        min: 0,
        showIf: (c) => Boolean(c.enable_augmentation),
      },
      {
        key: "saturation_jitter",
        label: "Saturation jitter",
        type: "number",
        step: 0.01,
        min: 0,
        showIf: (c) => Boolean(c.enable_augmentation),
      },
    ],
  },
  {
    title: "GAN parameters",
    onlyForModelIds: ["mnist_gan"],
    fields: [
      { key: "noise_dim", label: "Noise dimension", type: "number", integer: true, min: 1 },
      { key: "g_lr", label: "Generator learning rate", type: "number", step: 0.0001, min: 0 },
      { key: "d_lr", label: "Discriminator learning rate", type: "number", step: 0.0001, min: 0 },
      { key: "label_smoothing", label: "Label smoothing", type: "number", step: 0.01, min: 0, max: 1 },
    ],
  },
  {
    title: "GPU acceleration",
    description:
      "Metal/CUDA backends require rebuilding this app with the gpu-metal/gpu-cuda Cargo features — not enabled in this build. Leave as auto/cpu unless you've done that.",
    fields: [
      {
        key: "gpu_backend",
        label: "Backend",
        type: "select",
        options: [
          { value: "auto", label: "Auto" },
          { value: "cpu", label: "CPU" },
          { value: "metal", label: "Metal" },
          { value: "cuda", label: "CUDA" },
        ],
      },
      { key: "gpu_device_id", label: "GPU device id", type: "number", integer: true, min: 0 },
    ],
  },
];

export const DEFAULT_SCHEDULER_TYPE = "step_decay";
