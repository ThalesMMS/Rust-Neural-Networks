# Hyperparameters Configuration

This document describes how to configure training hyperparameters using JSON configuration files, enabling experimentation with different training settings without code changes.

## Overview

The hyperparameters configuration system allows you to control all aspects of neural network training by specifying values in a JSON file. Each model binary can load configuration from a file specified via the `--config` command-line flag, or use a sensible default configuration.

Benefits:
- **Rapid experimentation**: Try different learning rates, batch sizes, and schedulers without recompiling
- **Reproducibility**: Training configurations are versioned alongside code
- **Educational**: Students can explore the effect of hyperparameters by editing config files
- **Validation**: Automatic checks ensure parameter values are valid with helpful error messages

## Configuration Format

A training configuration is a JSON object with hyperparameters for training, scheduling, and activation functions:

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

## CLI Usage

All model binaries accept a `--config` flag to specify a configuration file:

```bash
# Use default config (config/training/<model>_default.json)
cargo run --release --bin mnist_mlp

# Use custom config file
cargo run --release --bin mnist_mlp -- --config config/training/my_experiment.json

# Use custom config with other models
cargo run --release --bin mnist_cnn -- --config config/training/mnist_cnn_fast.json
cargo run --release --bin cifar10_cnn -- --config config/training/cifar10_aggressive.json
```

If `--config` is not specified, each binary uses its default configuration:
- `mnist_mlp` → `config/training/mnist_mlp_default.json`
- `mnist_cnn` → `config/training/mnist_cnn_default.json`
- `mnist_attention_pool` → `config/training/mnist_attention_default.json`
- `cifar10_cnn` → `config/training/cifar10_cnn_default.json`
- `mlp_simple` → `config/training/mlp_simple_default.json`

## Training Hyperparameters

### Learning Rate

Initial learning rate for gradient descent optimization.

**Field:** `learning_rate`
**Type:** Float (f32)
**Required:** Optional (has default)
**Validation:** Must be positive (> 0.0)
**Default:** 0.01
**Typical range:** 0.0001 to 0.1

**Description:**
The learning rate controls the step size during gradient descent. Higher learning rates lead to faster training but may overshoot optimal values. Lower learning rates provide more stable convergence but require more training time.

**Example values:**
- `0.001` - Conservative, stable learning
- `0.01` - Standard starting point for most models
- `0.1` - Aggressive, may cause instability

**Example:**
```json
{
  "learning_rate": 0.01
}
```

### Epochs

Number of complete passes through the training dataset.

**Field:** `epochs`
**Type:** Integer (usize)
**Required:** Optional (has default)
**Validation:** Must be positive (> 0)
**Default:** Model-specific (10 for MNIST models, 1000000 for XOR)
**Typical range:** 3 to 100 (or 1M for simple problems like XOR)

**Description:**
Controls how many times the model sees the entire training dataset. More epochs can improve performance but risk overfitting. Early stopping typically prevents training for the full number of epochs.

**Example values:**
- `3` - Quick training for fast models (CNN)
- `10` - Standard for most MNIST models
- `20` - Extended training for complex models
- `1000000` - Long training for simple problems (XOR)

**Example:**
```json
{
  "epochs": 10
}
```

### Batch Size

Number of samples processed together before updating model parameters.

**Field:** `batch_size`
**Type:** Integer (usize)
**Required:** Optional (has default)
**Validation:** Must be positive (> 0)
**Default:** 32 or 64 depending on model
**Typical range:** 8 to 256

**Description:**
Controls the number of training samples used to compute each gradient update. Larger batches provide more stable gradients but require more memory and may reduce generalization. Smaller batches add noise that can help escape local minima.

**Example values:**
- `8` - Small batch, noisy gradients, less memory
- `32` - Balanced for most models
- `64` - Standard for MLP models
- `128` - Large batch, stable gradients, more memory

**Example:**
```json
{
  "batch_size": 64
}
```

### Validation Split

Fraction of training data to reserve for validation.

**Field:** `validation_split`
**Type:** Float (f32)
**Required:** Optional (has default)
**Validation:** Must be in range [0.0, 1.0]
**Default:** 0.1 (10%)
**Typical range:** 0.1 to 0.2

**Description:**
Reserves a fraction of the training data for validation during training. Validation metrics are computed at the end of each epoch to monitor overfitting. For MNIST models with 60K training samples, a 10% split creates 54K training + 6K validation samples.

**Example values:**
- `0.0` - No validation split (not recommended)
- `0.1` - Standard 10% validation split
- `0.2` - Conservative 20% validation split

**Example:**
```json
{
  "validation_split": 0.1
}
```

**Note:** The XOR model (`mlp_simple`) typically does not use validation split as it's a toy problem.

### Early Stopping Patience

Number of epochs to wait for improvement before stopping training.

**Field:** `early_stopping_patience`
**Type:** Integer (usize)
**Required:** Optional
**Validation:** Must be positive if specified
**Default:** 3
**Typical range:** 2 to 10

**Description:**
Prevents overfitting by stopping training if validation loss doesn't improve for the specified number of epochs. Saves training time and finds the best model automatically.

**Example values:**
- `2` - Aggressive early stopping
- `3` - Standard patience
- `5` - Conservative, allows more training

**Example:**
```json
{
  "early_stopping_patience": 3
}
```

**Note:** Only applies to models using validation split (not XOR).

### Early Stopping Minimum Delta

Minimum change in validation loss to qualify as improvement.

**Field:** `early_stopping_min_delta`
**Type:** Float (f32)
**Required:** Optional
**Validation:** Must be non-negative (>= 0.0)
**Default:** 0.001
**Typical range:** 0.0001 to 0.01

**Description:**
Defines the threshold for considering validation loss improvement. Changes smaller than this value don't reset the patience counter. Helps avoid stopping due to small fluctuations.

**Example values:**
- `0.0001` - Sensitive to small improvements
- `0.001` - Standard threshold
- `0.01` - Only considers significant improvements

**Example:**
```json
{
  "early_stopping_min_delta": 0.001
}
```

## Learning Rate Schedulers

Learning rate schedulers adjust the learning rate during training to improve convergence.

### Step Decay Scheduler

Reduces learning rate by a constant factor at regular intervals.

**Scheduler type:** `"step_decay"`

**Required parameters:**
- `step_size`: Number of epochs between learning rate reductions (positive integer)
- `gamma`: Multiplicative factor for learning rate (non-negative float, typically 0.1 to 0.9)

**Formula:**
```
new_lr = current_lr * gamma    (every step_size epochs)
```

**Example:**
```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "learning_rate": 0.01
}
```

This configuration reduces the learning rate by 50% every 3 epochs:
- Epochs 0-2: lr = 0.01
- Epochs 3-5: lr = 0.005
- Epochs 6-8: lr = 0.0025
- etc.

**Typical use cases:**
- Standard choice for MNIST models
- Good for stable, predictable learning rate decay
- Works well when you know roughly how many epochs to train

### Exponential Decay Scheduler

Reduces learning rate exponentially every epoch.

**Scheduler type:** `"exponential"`

**Required parameters:**
- `decay_rate`: Per-epoch multiplicative factor (non-negative float, typically 0.9 to 0.99)

**Formula:**
```
new_lr = current_lr * decay_rate    (every epoch)
```

**Example:**
```json
{
  "scheduler_type": "exponential",
  "decay_rate": 0.95,
  "learning_rate": 0.01
}
```

This configuration reduces the learning rate by 5% every epoch:
- Epoch 0: lr = 0.01
- Epoch 1: lr = 0.0095
- Epoch 2: lr = 0.009025
- etc.

**Typical use cases:**
- Smooth, continuous learning rate decay
- Good for long training runs
- More gradual than step decay

### Cosine Annealing Scheduler

Adjusts learning rate following a cosine curve, cycling between max and min values.

**Scheduler type:** `"cosine_annealing"`

**Required parameters:**
- `min_lr`: Minimum learning rate (non-negative float)
- `T_max`: Period of the cosine cycle in epochs (positive integer, typically total epochs)

**Formula:**
```
new_lr = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / T_max)) / 2
```

**Example:**
```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001,
  "T_max": 10,
  "learning_rate": 0.01
}
```

This configuration smoothly anneals the learning rate from 0.01 to 0.0001 over 10 epochs following a cosine curve.

**Typical use cases:**
- Used for CIFAR-10 CNN model
- Provides smooth learning rate decay
- Often yields better final accuracy than step decay
- Popular in modern deep learning

## Activation Functions

Activation functions can be configured with optional parameters for advanced activations.

### ReLU (Rectified Linear Unit)

**Function:** `"relu"`
**Parameters:** None
**Formula:** `f(x) = max(0, x)`

**Example:**
```json
{
  "activation_function": "relu"
}
```

**Typical use cases:**
- Default choice for most models
- Fast computation
- Works well for deep networks

### Leaky ReLU

**Function:** `"leaky_relu"`
**Optional parameters:**
- `leaky_relu_alpha`: Slope for negative values (default 0.01, must be non-negative)

**Formula:** `f(x) = max(alpha * x, x)`

**Example:**
```json
{
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.01
}
```

**Typical use cases:**
- Prevents "dying ReLU" problem
- Good for networks with many layers
- Slight improvement over ReLU in some cases

### ELU (Exponential Linear Unit)

**Function:** `"elu"`
**Optional parameters:**
- `elu_alpha`: Controls saturation for negative values (default 1.0, must be positive)

**Formula:**
```
f(x) = x                      if x > 0
     = alpha * (exp(x) - 1)   if x <= 0
```

**Example:**
```json
{
  "activation_function": "elu",
  "elu_alpha": 1.0
}
```

**Typical use cases:**
- Smoother than ReLU
- Can improve learning dynamics
- More computationally expensive than ReLU

### GELU (Gaussian Error Linear Unit)

**Function:** `"gelu"`
**Parameters:** None
**Formula:** `f(x) = x * Φ(x)` where Φ is the standard Gaussian CDF

**Example:**
```json
{
  "activation_function": "gelu"
}
```

**Typical use cases:**
- Popular in transformer models
- Smooth, non-linear activation
- Can improve performance on some tasks

### Swish

**Function:** `"swish"`
**Parameters:** None
**Formula:** `f(x) = x * sigmoid(x)`

**Example:**
```json
{
  "activation_function": "swish"
}
```

**Typical use cases:**
- Self-gated activation
- Can outperform ReLU on some tasks
- More computationally expensive

### Tanh (Hyperbolic Tangent)

**Function:** `"tanh"`
**Parameters:** None
**Formula:** `f(x) = tanh(x)`

**Example:**
```json
{
  "activation_function": "tanh"
}
```

**Typical use cases:**
- Classic activation function
- Outputs in range [-1, 1]
- Can suffer from vanishing gradients

## Validation Rules

The configuration system performs automatic validation to ensure valid training:

### 1. Hyperparameter Validation

Each hyperparameter has constraints that must be satisfied:

- **learning_rate**: Must be positive (> 0.0)
- **epochs**: Must be positive (> 0)
- **batch_size**: Must be positive (> 0)
- **validation_split**: Must be in range [0.0, 1.0]
- **early_stopping_min_delta**: Must be non-negative (>= 0.0)

### 2. Scheduler Parameter Validation

Each scheduler type requires specific parameters:

- **StepDecay**: Requires `step_size` (positive integer) and `gamma` (non-negative float)
- **ExponentialDecay**: Requires `decay_rate` (non-negative float)
- **CosineAnnealing**: Requires `min_lr` (non-negative float) and `T_max` (positive integer)

### 3. Activation Function Validation

Activation function must be one of the valid types:
- `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `tanh`

Activation function parameters must be valid:
- **leaky_relu_alpha**: Must be non-negative (>= 0.0)
- **elu_alpha**: Must be positive (> 0.0)

### 4. Error Messages

When validation fails, helpful error messages are provided:

```bash
Error loading config from 'config/training/invalid.json': learning_rate must be positive
Please ensure the config file exists and is valid JSON.
```

## Default Configurations

Default configuration files are provided for all models in the `config/training/` directory:

### MNIST MLP (`mnist_mlp_default.json`)

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 64,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

**Architecture:** 784 → 512 → 10 (fully-connected)
**Training:** Step decay scheduler, 10 epochs, batch size 64

### MNIST CNN (`mnist_cnn_default.json`)

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 3,
  "batch_size": 32,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

**Architecture:** Conv(8, 3×3) + MaxPool + Dense(10)
**Training:** Step decay scheduler, 3 epochs (trains quickly), batch size 32

### MNIST Attention (`mnist_attention_default.json`)

```json
{
  "scheduler_type": "step_decay",
  "step_size": 3,
  "gamma": 0.5,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 8,
  "batch_size": 32,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

**Architecture:** Patch embedding + Attention + FFN
**Training:** Step decay scheduler, 8 epochs, batch size 32

### CIFAR-10 CNN (`cifar10_cnn_default.json`)

```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001,
  "T_max": 10,
  "activation_function": "relu",
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 32,
  "validation_split": 0.1,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

**Architecture:** Conv(16, 3×3) + MaxPool + Dense(10)
**Training:** Cosine annealing scheduler, 10 epochs, batch size 32

### XOR MLP (`mlp_simple_default.json`)

```json
{
  "scheduler_type": "step_decay",
  "step_size": 100000,
  "gamma": 0.5,
  "activation_function": "sigmoid",
  "learning_rate": 0.01,
  "epochs": 1000000,
  "batch_size": 4
}
```

**Architecture:** 2 → 4 → 1 (simple XOR problem)
**Training:** Step decay with large step size, 1M epochs, batch size 4 (full dataset)
**Note:** No validation split or early stopping (toy problem)

## Experimentation Guide

### Creating Custom Configurations

To experiment with different hyperparameters:

1. **Copy a default config:**
   ```bash
   cp config/training/mnist_mlp_default.json config/training/mnist_mlp_experiment.json
   ```

2. **Edit the parameters:**
   ```json
   {
     "learning_rate": 0.001,
     "epochs": 20,
     "batch_size": 128
   }
   ```

3. **Run with the custom config:**
   ```bash
   cargo run --release --bin mnist_mlp -- --config config/training/mnist_mlp_experiment.json
   ```

### Example Experiments

#### Experiment 1: Higher Learning Rate

Test if a higher learning rate trains faster:

```json
{
  "learning_rate": 0.1,
  "epochs": 10,
  "batch_size": 64
}
```

**Expected result:** May converge faster but could be unstable.

#### Experiment 2: Larger Batch Size

Test effect of batch size on training:

```json
{
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 256
}
```

**Expected result:** More stable gradients, possibly slower convergence, requires more memory.

#### Experiment 3: Different Scheduler

Compare cosine annealing to step decay:

```json
{
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.0001,
  "T_max": 10,
  "learning_rate": 0.01
}
```

**Expected result:** Smoother learning rate decay, possibly better final accuracy.

#### Experiment 4: Alternative Activation

Test Leaky ReLU instead of ReLU:

```json
{
  "activation_function": "leaky_relu",
  "leaky_relu_alpha": 0.01
}
```

**Expected result:** May prevent dying ReLU problem, slight performance change.

### Comparing Results

To compare experiments systematically:

1. Create multiple config files with descriptive names
2. Run each experiment and save logs to different files
3. Use the Python visualization tools to compare training curves
4. Document which hyperparameters worked best for your use case

### Tips for Experimentation

- **Change one parameter at a time** to understand its individual effect
- **Use multiple random seeds** to ensure results are reproducible
- **Monitor both training and validation metrics** to detect overfitting
- **Start with default configs** and make incremental changes
- **Keep a log** of experiments and their results
- **Use early stopping** to save time on poorly-performing configs

## Common Hyperparameter Combinations

### Fast Training (Quick Experiments)

```json
{
  "learning_rate": 0.1,
  "epochs": 3,
  "batch_size": 128,
  "early_stopping_patience": 2
}
```

Use when you want quick feedback during development.

### Stable Training (Best Accuracy)

```json
{
  "learning_rate": 0.001,
  "epochs": 20,
  "batch_size": 32,
  "scheduler_type": "cosine_annealing",
  "min_lr": 0.00001,
  "T_max": 20
}
```

Use when you want the best possible accuracy and have time to train.

### Memory-Constrained

```json
{
  "learning_rate": 0.01,
  "epochs": 15,
  "batch_size": 8
}
```

Use when training on systems with limited memory.

### Large Dataset

```json
{
  "learning_rate": 0.01,
  "epochs": 10,
  "batch_size": 256,
  "validation_split": 0.1
}
```

Use when training on larger datasets where larger batches are beneficial.

## Troubleshooting

### Training Loss Not Decreasing

**Possible causes:**
- Learning rate too high or too low
- Poor initialization
- Incorrect data preprocessing

**Solutions:**
- Try learning rates in range [0.0001, 0.1]
- Use step decay scheduler to gradually reduce learning rate
- Verify data loading and normalization

### Model Overfitting

**Symptoms:**
- Training accuracy high, validation accuracy low
- Training loss decreasing, validation loss increasing

**Solutions:**
- Reduce model complexity (fewer layers or units)
- Increase validation split to 0.2
- Add regularization (not yet supported in config)
- Use early stopping with patience=2

### Training Too Slow

**Solutions:**
- Increase learning rate (0.01 → 0.1)
- Increase batch size (32 → 128)
- Reduce epochs
- Use step decay scheduler with larger gamma

### Unstable Training

**Symptoms:**
- Loss fluctuating wildly
- NaN or Inf values

**Solutions:**
- Decrease learning rate (0.01 → 0.001)
- Decrease batch size for more stable gradients
- Use cosine annealing for smoother learning rate decay
- Check for bugs in data preprocessing

## Advanced Topics

### Learning Rate Warmup

Not yet supported in configuration, but planned for future releases. Warmup gradually increases learning rate from 0 to target value over initial epochs.

### Cyclical Learning Rates

Not yet supported. Cyclical LR repeatedly cycles between minimum and maximum learning rates.

### Adaptive Optimizers

Current models support SGD and Adam optimizers (hardcoded). Future releases may add optimizer configuration.

### Gradient Clipping

Not yet supported in configuration. Gradient clipping prevents exploding gradients by capping gradient magnitude.

## See Also

- [Architecture Configuration](architecture_config.md) - Configure network architectures via JSON
- [README.md](../README.md) - Project overview and quick start guide
- [CLAUDE.md](../CLAUDE.md) - Instructions for AI assistants working with this codebase

## References

- [Learning Rate Schedules](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) - PyTorch documentation
- [Activation Functions](https://arxiv.org/abs/1710.05941) - "Searching for Activation Functions" paper
- [Batch Size Effects](https://arxiv.org/abs/1609.04836) - "On Large-Batch Training for Deep Learning" paper
- [Early Stopping](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf) - "Early Stopping - but when?" paper
