# Architecture Configuration

This document describes how to define neural network architectures using JSON configuration files, enabling architecture experimentation without code changes.

## Overview

The architecture configuration system allows you to define neural networks by specifying a sequence of layers in a JSON file. Each layer is defined by its type (Dense, Conv2D, BatchNorm, Dropout) and the required parameters for that layer type.

Benefits:
- **Rapid experimentation**: Try different architectures without recompiling
- **Reproducibility**: Architecture definitions are versioned alongside code
- **Educational**: Students can experiment with architectures by editing config files
- **Validation**: Automatic checks ensure layer connections are valid

## Configuration Format

An architecture configuration is a JSON object with a single `layers` array:

```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}
```

Layers are applied in the order they appear in the configuration.

## Layer Types

### Pooling Layer (MaxPool / AvgPool)

2D pooling layer that reduces spatial dimensions by applying a fixed window over each channel.

Pooling layers can be specified in two equivalent ways:

1) Use explicit layer types:
- `layer_type`: `"maxpool"`
- `layer_type`: `"avgpool"`

2) Use the generic `"pool"` type with a mode:
- `layer_type`: `"pool"`
- `pool_mode`: `"max"` or `"avg"`

**Required parameters:**
- `pool_size`: Pool window size (positive integer, square window)
- `pool_input_height`: Input height (positive integer)
- `pool_input_width`: Input width (positive integer)
- `pool_channels`: Number of channels (positive integer)

**Optional parameters:**
- `pool_stride`: Stride (default: `pool_size`)
- `pool_padding`: Padding (default: 0)

**Output size calculation:**
```
output_height = ((pool_input_height + 2 * pool_padding - pool_size) / pool_stride) + 1
output_width  = ((pool_input_width  + 2 * pool_padding - pool_size) / pool_stride) + 1
output_size   = pool_channels * output_height * output_width
```

**Examples:**

Max pooling via explicit type:
```json
{
  "layer_type": "maxpool",
  "pool_size": 2,
  "pool_stride": 2,
  "pool_padding": 0,
  "pool_input_height": 28,
  "pool_input_width": 28,
  "pool_channels": 8
}
```

Average pooling via `pool_mode`:
```json
{
  "layer_type": "pool",
  "pool_mode": "avg",
  "pool_size": 2,
  "pool_stride": 2,
  "pool_padding": 0,
  "pool_input_height": 28,
  "pool_input_width": 28,
  "pool_channels": 8
}
```


### Dense Layer

Fully-connected (dense) layer with weight matrix and bias vector. Uses BLAS-accelerated matrix multiplication.

**Required parameters:**
- `layer_type`: `"dense"`
- `input_size`: Number of input neurons (positive integer)
- `output_size`: Number of output neurons (positive integer)

**Example:**
```json
{
  "layer_type": "dense",
  "input_size": 784,
  "output_size": 512
}
```

**Implementation:** Uses `DenseLayer` from `src/layers/dense.rs`

### Conv2D Layer

2D convolutional layer with learnable filters. Operates on flattened image data (input is `in_channels * height * width`).

**Required parameters:**
- `layer_type`: `"conv2d"`
- `in_channels`: Number of input channels (positive integer)
- `out_channels`: Number of filters/output channels (positive integer)
- `kernel_size`: Size of square kernel (positive integer, e.g., 3 for 3×3)
- `input_height`: Height of input image (positive integer)
- `input_width`: Width of input image (positive integer)

**Optional parameters:**
- `padding`: Zero-padding added to input (default: 0, can be negative)
- `stride`: Step size for convolution (default: 1, positive integer)

**Output size calculation:**
```
output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1
output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1
output_size = out_channels * output_height * output_width
```

**Example:**
```json
{
  "layer_type": "conv2d",
  "in_channels": 1,
  "out_channels": 8,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "input_height": 28,
  "input_width": 28
}
```

This example produces an output size of `8 * 28 * 28 = 6272` (same spatial dimensions due to padding=1, stride=1, kernel=3).

**Implementation:** Uses `Conv2DLayer` from `src/layers/conv2d/mod.rs`

### BatchNorm Layer

Batch normalization layer that normalizes activations. Does not change the size of the input.

**Required parameters:**
- `layer_type`: `"batchnorm"`
- `size`: Number of features to normalize (positive integer)

**Optional parameters:**
- `epsilon`: Small constant for numerical stability (default: 1e-5, must be positive)
- `momentum`: Momentum for running mean/variance updates (default: 0.9, range [0.0, 1.0])

**Example:**
```json
{
  "layer_type": "batchnorm",
  "size": 256,
  "epsilon": 1e-5,
  "momentum": 0.9
}
```

**Note:** BatchNorm input size must match the output size of the previous layer.

**Implementation:** Uses `BatchNormLayer` from `src/layers/batchnorm.rs`

### Dropout Layer

Dropout regularization layer that randomly drops units during training. Does not change the size of the input.

**Required parameters:**
- `layer_type`: `"dropout"`
- `size`: Number of features (positive integer)
- `drop_rate`: Probability of dropping each unit (range [0.0, 1.0), exclusive of 1.0)

**Example:**
```json
{
  "layer_type": "dropout",
  "size": 256,
  "drop_rate": 0.5
}
```

**Note:** Dropout input size must match the output size of the previous layer.

**Implementation:** Uses `DropoutLayer` from `src/layers/dropout.rs`

## Validation Rules

The configuration system performs automatic validation to ensure valid architectures:

### 1. Layer-Specific Validation

Each layer type has required parameters that must be present:

- **Dense**: Must have `input_size` and `output_size` (both > 0)
- **Conv2D**: Must have `in_channels`, `out_channels`, `kernel_size`, `input_height`, `input_width` (all > 0), and `stride` > 0 if specified
- **BatchNorm**: Must have `size` > 0, `epsilon` > 0 if specified, `momentum` in [0.0, 1.0] if specified
- **Dropout**: Must have `size` > 0 and `drop_rate` in [0.0, 1.0) (exclusive of 1.0)

### 2. Layer Connection Validation

The output size of each layer must match the input size of the next layer:

```
output_size(layer[i]) == input_size(layer[i+1])
```

For example, this configuration is **valid**:
```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dropout",
      "size": 256,
      "drop_rate": 0.2
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}
```

This configuration is **invalid** (connection mismatch):
```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 128,  // ERROR: 256 != 128
      "output_size": 10
    }
  ]
}
```

### 3. Architecture-Level Validation

- Configuration must contain at least one layer
- All layer types must be recognized (case-insensitive): `"dense"`, `"conv2d"`, `"batchnorm"`, `"dropout"`

## Usage in Code

### Loading a Configuration

```rust
use rust_neural_networks::architecture::load_architecture;

// Load and validate architecture configuration
let config = load_architecture("config/architectures/mlp_simple.json")
    .expect("Failed to load architecture");

println!("Loaded architecture with {} layers", config.layers.len());
```

### Building a Model

```rust
use rust_neural_networks::architecture::{load_architecture, build_model};
use rust_neural_networks::utils::rng::SimpleRng;

// Load configuration
let config = load_architecture("config/architectures/mlp_simple.json")
    .expect("Failed to load architecture");

// Create RNG for weight initialization
let mut rng = SimpleRng::new(42);

// Build model from configuration
let layers = build_model(&config, &mut rng)
    .expect("Failed to build model");

println!("Built model with {} layers", layers.len());
println!("Input size: {}", layers[0].input_size());
println!("Output size: {}", layers.last().unwrap().output_size());
```

### Using Layers in Training

```rust
use rust_neural_networks::layers::Layer;

// Forward pass through all layers
let mut activations = input.clone();
for layer in &mut layers {
    activations = layer.forward(&activations);
}

// Backward pass through all layers (in reverse)
let mut gradient = output_gradient.clone();
for layer in layers.iter_mut().rev() {
    gradient = layer.backward(&gradient);
}

// Update parameters
let learning_rate = 0.01;
for layer in &mut layers {
    layer.update_parameters(learning_rate);
}
```

## Example Configurations

### Simple MLP (2 layers)

File: `config/architectures/mlp_simple.json`

```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}
```

**Architecture:** 784 → 256 → 10
**Use case:** Simple MNIST classifier

### Medium MLP (3 layers)

File: `config/architectures/mlp_medium.json`

```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 256
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}
```

**Architecture:** 784 → 512 → 256 → 10
**Use case:** Deeper MNIST classifier with more capacity

### Simple CNN

File: `config/architectures/cnn_simple.json`

```json
{
  "layers": [
    {
      "layer_type": "conv2d",
      "in_channels": 1,
      "out_channels": 8,
      "kernel_size": 3,
      "padding": 1,
      "stride": 1,
      "input_height": 28,
      "input_width": 28
    },
    {
      "layer_type": "dense",
      "input_size": 6272,
      "output_size": 128
    },
    {
      "layer_type": "dense",
      "input_size": 128,
      "output_size": 10
    }
  ]
}
```

**Architecture:** Conv2D(8 filters, 3×3) → Dense(128) → Dense(10)
**Use case:** MNIST CNN with convolutional feature extraction

**Note:** After convolution with padding=1, stride=1, the spatial dimensions remain 28×28, so the output is `8 * 28 * 28 = 6272` features.

### MLP with Regularization

Example showing BatchNorm and Dropout layers:

```json
{
  "layers": [
    {
      "layer_type": "dense",
      "input_size": 784,
      "output_size": 512
    },
    {
      "layer_type": "batchnorm",
      "size": 512,
      "epsilon": 1e-5,
      "momentum": 0.9
    },
    {
      "layer_type": "dropout",
      "size": 512,
      "drop_rate": 0.5
    },
    {
      "layer_type": "dense",
      "input_size": 512,
      "output_size": 256
    },
    {
      "layer_type": "batchnorm",
      "size": 256
    },
    {
      "layer_type": "dropout",
      "size": 256,
      "drop_rate": 0.3
    },
    {
      "layer_type": "dense",
      "input_size": 256,
      "output_size": 10
    }
  ]
}
```

**Architecture:** Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → BatchNorm → Dropout(0.3) → Dense(10)
**Use case:** Regularized deep network to prevent overfitting

## Testing

The architecture module includes comprehensive tests in `src/architecture.rs` and `tests/test_architecture.rs`.

### Running Architecture Tests

```bash
# Run all architecture tests
cargo test architecture

# Run unit tests in the architecture module
cargo test --lib architecture::tests

# Run integration tests
cargo test --test test_architecture

# Run a specific test
cargo test test_build_model --verbose
```

### Test Coverage

Tests validate:
- Valid architectures (MLP, CNN, with BatchNorm/Dropout)
- Model building from configurations
- Layer connection validation
- Error handling for invalid configs
- Missing required fields
- Invalid parameter ranges
- Layer size mismatches
- Edge cases (single layer, deep networks, default parameters)
- Case-insensitive layer types
- Example config files parsing

## Common Patterns

### Creating a Config Programmatically

```rust
use rust_neural_networks::architecture::{ArchitectureConfig, LayerConfig};

let config = ArchitectureConfig {
    layers: vec![
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: Some(784),
            output_size: Some(256),
            ..Default::default()  // Note: LayerConfig doesn't derive Default
        },
        LayerConfig {
            layer_type: "dense".to_string(),
            input_size: Some(256),
            output_size: Some(10),
            ..Default::default()
        },
    ],
};
```

**Note:** In practice, you'll need to explicitly set all fields to `None` for unused parameters rather than using `..Default::default()`.

### Calculating Conv2D Output Size

When connecting a Conv2D layer to a Dense layer, calculate the output size:

```
output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1
output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1
output_size = out_channels * output_height * output_width
```

**Example:** Conv2D with `in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1, input_height=28, input_width=28`:

```
output_height = ((28 + 2*1 - 3) / 1) + 1 = 28
output_width = ((28 + 2*1 - 3) / 1) + 1 = 28
output_size = 8 * 28 * 28 = 6272
```

The next Dense layer must have `input_size: 6272`.

### Adding Regularization

A common pattern is to add BatchNorm and Dropout after each Dense layer:

```
Dense → BatchNorm → Dropout → Dense → BatchNorm → Dropout → ...
```

Make sure the `size` parameter for BatchNorm and Dropout matches the `output_size` of the preceding Dense layer.

## Error Handling

The configuration system provides detailed error messages for common issues:

### Missing Required Field

```
Error: Layer 0: Dense layer requires 'input_size'
```

Solution: Add the missing field to the layer configuration.

### Layer Connection Mismatch

```
Error: Layer connection mismatch: Layer 0 output size (256) does not match Layer 1 input size (128)
```

Solution: Ensure output size of layer N matches input size of layer N+1.

### Invalid Layer Type

```
Error: Layer 0: Invalid layer type 'invalid'. Must be one of: dense, conv2d, batchnorm, dropout
```

Solution: Use a valid layer type (case-insensitive).

### Invalid Parameter Range

```
Error: Layer 1: drop_rate must be in range [0.0, 1.0)
```

Solution: Ensure parameters are within valid ranges specified for each layer type.

### Empty Architecture

```
Error: Architecture must have at least one layer
```

Solution: Add at least one layer to the `layers` array.

## Best Practices

1. **Start simple**: Begin with basic architectures (2-3 Dense layers) and gradually add complexity
2. **Match layer sizes**: Always ensure layer connections are valid by matching input/output sizes
3. **Use validation**: Let `load_architecture()` catch errors before building the model
4. **Version configs**: Store architecture configs in version control alongside code
5. **Document experiments**: Use descriptive filenames (e.g., `mlp_512_256_dropout.json`)
6. **Test configs**: Write tests that load your configs to ensure they remain valid
7. **Calculate Conv2D sizes**: Double-check Conv2D output size calculations when connecting to Dense layers
8. **Regularize deep networks**: Use BatchNorm and/or Dropout for networks with many parameters

## Limitations

- **Fixed activation functions**: Activation functions (ReLU, sigmoid, softmax) are not configurable and must be applied in the training code
- **Pooling layers**: MaxPool and AvgPool are supported via `layer_type`: `"maxpool"` / `"avgpool"` or `layer_type`: `"pool"` with `pool_mode`: `"max"` / `"avg"`
- **No skip connections**: ResNet-style skip connections require code modifications
- **No attention layers**: Transformer attention layers are not configurable
- **Sequential only**: Only sequential layer stacking is supported (no branching/merging)

## Future Enhancements

Potential improvements to the architecture configuration system:

- [ ] Configurable activation functions per layer
- [x] Pooling layer support (MaxPool, AvgPool)
- [ ] Skip connections and residual blocks
- [ ] Multi-branch architectures (inception-style)
- [ ] Attention layer configurations
- [ ] Weight initialization strategies
- [ ] Layer freezing for transfer learning
- [ ] Architecture visualization tools

## Related Documentation

- [Layer Trait Implementation](../src/layers/trait.rs) - Common interface for all layers
- [Dense Layer](../src/layers/dense.rs) - BLAS-accelerated dense layer implementation
- [Conv2D Layer](../src/layers/conv2d/mod.rs) - 2D convolutional layer implementation
- [BatchNorm Layer](../src/layers/batchnorm.rs) - Batch normalization implementation
- [Dropout Layer](../src/layers/dropout.rs) - Dropout regularization implementation
- [CLAUDE.md](../CLAUDE.md) - Project overview and build instructions
- [README.md](../README.md) - Repository overview and getting started

## Questions or Issues?

If you encounter issues with architecture configurations:

1. Verify your JSON is valid (use a JSON validator)
2. Check that all required fields are present for each layer type
3. Verify layer connections (output[i] matches input[i+1])
4. Review error messages - they specify the exact issue and layer index
5. Consult the example configs in `config/architectures/`
6. Run the test suite: `cargo test architecture`
