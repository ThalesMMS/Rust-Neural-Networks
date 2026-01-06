# Training Pipeline

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/README.md)
> * [mnist_attention_pool.rs](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs)
> * [mnist_cnn.rs](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs)
> * [mnist_mlp.rs](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs)

## Purpose and Scope

This document describes the common training workflow shared across all neural network implementations in the repository. It covers data loading from IDX files, model initialization with Xavier/Glorot distributions, the mini-batch training loop with forward/backward propagation, stochastic gradient descent (SGD) optimization, and evaluation procedures.

For platform-specific BLAS acceleration details, see [BLAS Integration](5a%20BLAS-Integration.md). For model serialization format, see [Binary Model Format](5c%20Binary-Model-Format.md). For architecture-specific implementation details, see the individual model pages: [MNIST MLP](3a%20MNIST-MLP.md), [MNIST CNN](3b%20MNIST-CNN.md), [MNIST Attention Model](3c%20MNIST-Attention-Model.md), and [Simple XOR MLP](3d%20Simple-XOR-MLP.md).

## Training Pipeline Architecture

The following diagram shows the complete training pipeline flow from data loading through model persistence:

```mermaid
flowchart TD

IDX_PARSE["read_mnist_images()<br>read_mnist_labels()<br>read_be_u32()"]
TRAIN_DATA["train_images: Vec<f32><br>train_labels: Vec<u8><br>60,000 samples"]
TEST_DATA["test_images: Vec<f32><br>test_labels: Vec<u8><br>10,000 samples"]
RNG_INIT["SimpleRng::new()<br>reseed_from_time()"]
MODEL_INIT["initialize_network() / init_model()<br>Xavier/Glorot uniform init"]
BUFFERS["Training buffers allocation<br>batch_inputs, a1, a2, grads"]
SHUFFLE["shuffle_usize()<br>Fisher-Yates shuffle"]
GATHER["gather_batch()<br>Mini-batch sampling"]
FORWARD["Forward pass<br>sgemm_wrapper() / conv_forward_relu()<br>ReLU / Softmax activation"]
LOSS["compute_delta_and_loss()<br>softmax_xent_backward()<br>Cross-entropy"]
BACKWARD["Backward pass<br>Gradient computation<br>Chain rule"]
SGD["apply_sgd_update()<br>W -= lr * grad_W"]
TEST_FWD["test() / test_accuracy()<br>Forward-only inference"]
ARGMAX["Argmax prediction<br>Accuracy calculation"]
LOG_WRITE["BufWriter<br>logs/training_loss_*.txt<br>epoch,loss,time"]
MODEL_SAVE["save_model()<br>mnist_model.bin"]

BUFFERS -.-> SHUFFLE
SGD -.-> TEST_FWD
LOSS -.-> LOG_WRITE
SGD -.-> MODEL_SAVE

subgraph Persistence ["Persistence"]
    LOG_WRITE
    MODEL_SAVE
end

subgraph subGraph3 ["Evaluation Phase"]
    TEST_FWD
    ARGMAX
end

subgraph subGraph2 ["Training Loop (per epoch)"]
    SHUFFLE
    GATHER
    FORWARD
    LOSS
    BACKWARD
    SGD
    SHUFFLE -.->|"Next batch"| GATHER
    GATHER -.->|"Epoch complete"| FORWARD
    FORWARD -.-> LOSS
    LOSS -.-> BACKWARD
    BACKWARD -.-> SGD
    SGD -.-> GATHER
    SGD -.-> SHUFFLE
end

subgraph subGraph1 ["Initialization Phase"]
    RNG_INIT
    MODEL_INIT
    BUFFERS
end

subgraph subGraph0 ["Data Loading Phase"]
    IDX_PARSE
    TRAIN_DATA
    TEST_DATA
end
```

**Sources:** [mnist_mlp.rs L1-L664](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L1-L664)

 [mnist_cnn.rs L1-L704](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L1-L704)

 [mnist_attention_pool.rs L1-L1256](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L1-L1256)

## Data Loading

### IDX Format Parsing

All implementations use the IDX binary format for MNIST data. The `read_be_u32()` function parses big-endian 32-bit unsigned integers from the IDX file headers:

```mermaid
flowchart TD

FILE["IDX File<br>Big-endian format"]
READ_BE["read_be_u32()<br>4-byte BE parsing"]
HEADER["Header fields:<br>magic (u32)<br>count (u32)<br>rows (u32)<br>cols (u32)"]
DATA["Pixel data:<br>u8 bytes<br>0-255 range"]
NORM["Normalized floats:<br>f32 / 255.0<br>[0.0, 1.0] range"]

FILE -.-> READ_BE
HEADER -.-> DATA
DATA -.-> NORM
```

**Sources:** [mnist_mlp.rs L563-L570](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L563-L570)

 [mnist_cnn.rs L99-L106](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L99-L106)

 [mnist_attention_pool.rs L104-L111](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L104-L111)

### Image and Label Loading

| Function | Input | Output | Normalization |
| --- | --- | --- | --- |
| `read_mnist_images()` | Filename, count | `Vec<f32>` (N × 784) | `pixel as f32 / 255.0` |
| `read_mnist_labels()` | Filename, count | `Vec<u8>` (N) | Raw labels 0-9 |

The image loader validates dimensions (28×28) and truncates to requested sample count. Data is stored in row-major order as flattened vectors:

```mermaid
flowchart TD

INPUT["28×28 image<br>784 pixels"]
FLAT["Flattened: [p0, p1, ..., p783]<br>Row-major order"]
BATCH["Batch: [img0, img1, ..., imgN]<br>Contiguous memory"]

INPUT -.-> FLAT
FLAT -.-> BATCH
```

**Sources:** [mnist_mlp.rs L572-L600](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L572-L600)

 [mnist_mlp.rs L603-L620](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L603-L620)

 [mnist_cnn.rs L109-L142](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L109-L142)

 [mnist_cnn.rs L145-L162](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L145-L162)

 [mnist_attention_pool.rs L114-L148](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L114-L148)

 [mnist_attention_pool.rs L151-L168](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L151-L168)

## Model Initialization

### Random Number Generation

All implementations use `SimpleRng`, a lightweight xorshift-based pseudo-random number generator for reproducible initialization:

```mermaid
flowchart TD

SEED["Seed: u64<br>0 → fixed constant"]
RESEED["reseed_from_time()<br>SystemTime nanos"]
XORSHIFT["Xorshift algorithm:<br>x ^= x << 13<br>x ^= x >> 7<br>x ^= x << 17"]
F32["next_f32()<br>u32 / MAX → [0,1)"]
RANGE["gen_range_f32()<br>low + (high-low) * r"]

SEED -.-> RESEED
RESEED -.-> XORSHIFT
XORSHIFT -.-> F32
F32 -.-> RANGE
```

**Sources:** [mnist_mlp.rs L21-L69](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L21-L69)

 [mnist_cnn.rs L44-L96](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L44-L96)

 [mnist_attention_pool.rs L49-L101](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L49-L101)

### Xavier/Glorot Initialization

Weights are initialized using Xavier uniform distribution to maintain stable signal magnitude across layers:

| Model | Layer | Formula | Implementation |
| --- | --- | --- | --- |
| MLP | Hidden (784→512) | `limit = sqrt(6 / (784 + 512))` | [mnist_mlp.rs L86-L98](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L86-L98) |
| MLP | Output (512→10) | `limit = sqrt(6 / (512 + 10))` | [mnist_mlp.rs L86-L98](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L86-L98) |
| CNN | Conv (9→8) | `limit = sqrt(6 / (9 + 72))` | [mnist_cnn.rs L236-L244](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L236-L244) |
| CNN | FC (1568→10) | `limit = sqrt(6 / (1568 + 10))` | [mnist_cnn.rs L246-L250](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L246-L250) |
| Attention | All layers | Based on fan-in/fan-out | [mnist_attention_pool.rs L390-L463](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L390-L463) |

Biases are initialized to zero. The initialization functions use `gen_range_f32(-limit, limit)` to sample uniformly:

```mermaid
flowchart TD

CALC_LIMIT["Calculate limit:<br>sqrt(6 / (fan_in + fan_out))"]
INIT_W["Initialize weights:<br>W[i] = uniform(-limit, limit)"]
INIT_B["Initialize biases:<br>b[i] = 0.0"]
STRUCT["Construct model struct:<br>NeuralNetwork / Cnn / AttnModel"]
```

**Sources:** [mnist_mlp.rs L86-L111](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L86-L111)

 [mnist_cnn.rs L230-L258](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L230-L258)

 [mnist_attention_pool.rs L390-L463](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L390-L463)

## Training Loop Structure

### Epoch and Batch Organization

The training loop follows this nested structure:

```mermaid
flowchart TD

EPOCH_START["for epoch in 0..EPOCHS"]
INDEX_INIT["indices = [0, 1, ..., N-1]"]
TIMER_START["start_time = Instant::now()"]
SHUFFLE["rng.shuffle_usize(&mut indices)<br>Fisher-Yates shuffle"]
BATCH_LOOP["for batch_start in (0..N).step_by(BATCH_SIZE)"]
BATCH_COUNT["batch_count = min(BATCH_SIZE, remaining)"]
GATHER["gather_batch() using shuffled indices"]
TRAIN_STEP["Forward → Loss → Backward → SGD"]
ACCUMULATE["total_loss += batch_loss"]
NEXT_BATCH["More batches?"]
EVAL["test_accuracy() on test set"]
LOG["Write epoch, avg_loss, time to log file"]
NEXT_EPOCH["More epochs?"]
DONE["Training complete"]

SHUFFLE -.->|"Yes"| BATCH_LOOP
GATHER -.->|"Yes"| TRAIN_STEP
ACCUMULATE -.->|"No"| NEXT_BATCH
EVAL -.->|"No"| LOG
LOG -.-> NEXT_EPOCH
```

**Sources:** [mnist_mlp.rs L287-L446](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L287-L446)

 [mnist_cnn.rs L645-L699](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L645-L699)

 [mnist_attention_pool.rs L1207-L1253](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L1207-L1253)

### Mini-Batch Sampling

The `gather_batch()` function copies shuffled samples into contiguous batch buffers for efficient processing:

| Implementation | Function Location | Buffer Types |
| --- | --- | --- |
| MLP | [mnist_mlp.rs L239-L258](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L239-L258) | `batch_inputs: [f32; BATCH_SIZE * 784]` |
| CNN | [mnist_cnn.rs L165-L182](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L165-L182) | `batch_inputs: [f32; BATCH_SIZE * 784]` |
| Attention | [mnist_attention_pool.rs L171-L188](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L171-L188) | `batch_inputs: [f32; BATCH_SIZE * 784]` |

```mermaid
flowchart TD

INDICES["Shuffled indices:<br>[17, 42, 5, ...]"]
SOURCE["Source data:<br>images[i * 784..(i+1) * 784]"]
GATHER["gather_batch()<br>Copy subset"]
BATCH["Batch buffer:<br>[img_17, img_42, img_5, ...]<br>Contiguous memory"]

INDICES -.-> GATHER
SOURCE -.-> GATHER
GATHER -.-> BATCH
```

**Sources:** [mnist_mlp.rs L239-L258](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L239-L258)

 [mnist_cnn.rs L165-L182](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L165-L182)

 [mnist_attention_pool.rs L171-L188](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L171-L188)

## Forward Propagation

### MLP Forward Pass (BLAS-Accelerated)

The MLP uses `sgemm_wrapper()` for batched matrix multiplication:

```mermaid
flowchart TD

INPUT["batch_inputs<br>[batch × 784]"]
GEMM1["sgemm_wrapper()<br>A=inputs, B=W1<br>C=a1 (batch × 512)"]
BIAS1["add_bias()<br>a1 += b1"]
RELU1["relu_inplace()<br>a1 = max(0, a1)"]
GEMM2["sgemm_wrapper()<br>A=a1, B=W2<br>C=a2 (batch × 10)"]
BIAS2["add_bias()<br>a2 += b2"]
SOFTMAX["softmax_rows()<br>a2 = exp(a2) / sum"]

INPUT -.-> GEMM1
GEMM1 -.-> BIAS1
BIAS1 -.-> RELU1
RELU1 -.-> GEMM2
GEMM2 -.-> BIAS2
BIAS2 -.-> SOFTMAX
```

**Sources:** [mnist_mlp.rs L314-L351](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L314-L351)

### CNN Forward Pass (Manual Loops)

The CNN uses explicit loops for convolution, pooling, and fully-connected layers:

```mermaid
flowchart TD

INPUT["batch_inputs<br>[batch × 28 × 28]"]
CONV["conv_forward_relu()<br>8 filters 3×3, pad=1<br>Output: [batch × 8 × 28 × 28]"]
POOL["maxpool_forward()<br>2×2 stride 2<br>Output: [batch × 8 × 14 × 14]<br>Stores argmax indices"]
FC["fc_forward()<br>1568 → 10<br>Output: [batch × 10]"]
SOFTMAX["softmax_rows()"]

INPUT -.-> CONV
CONV -.-> POOL
POOL -.-> FC
FC -.-> SOFTMAX
```

**Sources:** [mnist_cnn.rs L667-L669](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L667-L669)

### Attention Forward Pass

The attention model processes image patches as tokens:

```mermaid
flowchart TD

INPUT["batch_inputs<br>[batch × 784]"]
PATCH["extract_patches()<br>4×4 patches<br>[batch × 49 × 16]"]
PROJ["Patch projection + pos embed<br>tok = ReLU(patch * W + b + pos)<br>[batch × 49 × 16]"]
QKV["Q/K/V projections<br>Q = tok * W_q + b_q<br>K = tok * W_k + b_k<br>V = tok * W_v + b_v"]
ATTN["Self-attention<br>scores = Q * K^T / sqrt(d)<br>alpha = softmax(scores)<br>out = alpha * V"]
FFN["Feed-forward MLP<br>ffn1 = ReLU(out * W1 + b1)<br>ffn2 = ffn1 * W2 + b2"]
POOL["Mean pooling<br>[batch × 16]"]
CLS["Classifier<br>logits = pooled * W_cls + b_cls<br>[batch × 10]"]
SOFTMAX["softmax_rows()"]

INPUT -.-> PATCH
PATCH -.-> PROJ
PROJ -.-> QKV
QKV -.-> ATTN
ATTN -.-> FFN
FFN -.-> POOL
POOL -.-> CLS
CLS -.-> SOFTMAX
```

**Sources:** [mnist_attention_pool.rs L489-L661](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L489-L661)

## Loss Computation

### Cross-Entropy Loss

All models use softmax cross-entropy loss with numerical stability via max subtraction:

```mermaid
flowchart TD

LOGITS["logits[b]<br>Raw scores"]
SOFTMAX["softmax_rows()<br>probs = exp(z - max) / sum"]
LABEL["label[b]<br>Ground truth class y"]
LOSS["loss = -log(probs[y])<br>Accumulate over batch"]
DELTA["delta = probs<br>delta[y] -= 1.0<br>Scaled by 1/batch"]

LOGITS -.-> SOFTMAX
SOFTMAX -.-> LABEL
LABEL -.-> LOSS
SOFTMAX -.-> DELTA
```

**Sources:** [mnist_mlp.rs L209-L236](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L209-L236)

 [mnist_cnn.rs L374-L401](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L374-L401)

 [mnist_attention_pool.rs L664-L683](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L664-L683)

The `compute_delta_and_loss()` / `softmax_xent_backward()` functions combine loss calculation and gradient initialization:

| Implementation | Function | Loss Calculation | Delta Calculation |
| --- | --- | --- | --- |
| MLP | [mnist_mlp.rs L209-L236](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L209-L236) | `-log(probs[label])` | `delta = probs - onehot` |
| CNN | [mnist_cnn.rs L374-L401](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L374-L401) | `-log(probs[label])` | `delta = (probs - onehot) * scale` |
| Attention | [mnist_attention_pool.rs L664-L683](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L664-L683) | `-log(probs[label])` | `delta = (probs - onehot) * scale` |

## Backward Propagation

### Gradient Flow Through Layers

Backpropagation applies the chain rule to compute gradients from output to input:

```mermaid
flowchart TD

DLOSS["dL/dlogits<br>delta = probs - onehot"]
DW2["dL/dW2 = a1^T * delta"]
DB2["dL/db2 = sum(delta, axis=0)"]
DA1["dL/da1 = delta * W2^T"]
DRELU["dL/dz1 = da1 * (a1 > 0)"]
DW1["dL/dW1 = X^T * dz1"]
DB1["dL/db1 = sum(dz1, axis=0)"]

DLOSS -.-> DW2
DLOSS -.-> DB2
DLOSS -.-> DA1
DA1 -.-> DRELU

subgraph subGraph1 ["Hidden Layer (ReLU)"]
    DRELU
    DW1
    DB1
    DRELU -.-> DW1
    DRELU -.-> DB1
end

subgraph subGraph0 ["Output Layer"]
    DW2
    DB2
    DA1
end
```

**Sources:** [mnist_mlp.rs L363-L426](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L363-L426)

### MLP Backward Pass

The MLP backward pass uses transposed GEMM operations for efficient gradient computation:

| Step | Operation | SGEMM Call | Output |
| --- | --- | --- | --- |
| Output grad | `dW2 = a1^T * delta` | [mnist_mlp.rs L363-L378](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L363-L378) | `grad_w2 [512 × 10]` |
| Hidden grad | `dz1 = delta * W2^T` | [mnist_mlp.rs L384-L398](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L384-L398) | `dz1 [batch × 512]` |
| ReLU grad | `dz1 *= (a1 > 0)` | [mnist_mlp.rs L401-L405](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L401-L405) | Element-wise mask |
| Input grad | `dW1 = X^T * dz1` | [mnist_mlp.rs L407-L422](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L407-L422) | `grad_w1 [784 × 512]` |

**Sources:** [mnist_mlp.rs L363-L426](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L363-L426)

### CNN Backward Pass

The CNN backward pass routes gradients through pooling and convolution layers:

```mermaid
flowchart TD

DLOSS["dL/dlogits"]
FC_GW["grad_fc_w += pool_out * delta"]
FC_GB["grad_fc_b += sum(delta)"]
FC_DX["d_pool = delta * fc_w^T"]
POOL_SCATTER["Scatter d_pool to argmax positions<br>Use stored pool_idx"]
POOL_RELU["d_conv *= (conv_out > 0)"]
CONV_GW["grad_conv_w += input * d_conv (windowed)"]
CONV_GB["grad_conv_b += sum(d_conv)"]

DLOSS -.-> FC_GW
DLOSS -.-> FC_GB
DLOSS -.-> FC_DX

subgraph subGraph2 ["Conv Backward"]
    CONV_GW
    CONV_GB
end

subgraph subGraph1 ["MaxPool Backward"]
    POOL_SCATTER
    POOL_RELU
end

subgraph subGraph0 ["FC Backward"]
    FC_GW
    FC_GB
    FC_DX
end
```

**Sources:** [mnist_cnn.rs L404-L453](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L404-L453)

 [mnist_cnn.rs L456-L504](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L456-L504)

 [mnist_cnn.rs L507-L557](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L507-L557)

### Attention Backward Pass

The attention backward pass handles complex dependencies through self-attention:

```mermaid
flowchart TD

DLOSS["dL/dlogits"]
CLS_G["dpooled = delta * W_cls^T"]
POOL_G["dffn2 = dpooled / seq_len<br>Broadcast to all tokens"]
FFN2_G["dffn1 = dffn2 * W_ff2^T<br>ReLU mask"]
FFN1_G["dattn = dffn1 * W_ff1^T"]
ATTN_G["dV += alpha^T * dattn<br>dalpha = dattn * V^T"]
SOFT_G["dscores = softmax_grad(dalpha, alpha)"]
QK_G["dQ += dscores * K / sqrt(d)<br>dK += dscores^T * Q / sqrt(d)"]
PROJ_G["dtok = dQW_q^T + dKW_k^T + dV*W_v^T<br>ReLU mask"]
PATCH_G["grad_w_patch += patches * dtok<br>grad_pos += dtok"]

DLOSS -.-> CLS_G

subgraph subGraph4 ["Token Projection"]
    PROJ_G
    PATCH_G
end

subgraph Self-Attention ["Self-Attention"]
    ATTN_G
    SOFT_G
    QK_G
end

subgraph FFN ["FFN"]
    FFN2_G
    FFN1_G
end

subgraph Pooling ["Pooling"]
    POOL_G
end

subgraph Classifier ["Classifier"]
    CLS_G
end
```

**Sources:** [mnist_attention_pool.rs L686-L928](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L686-L928)

## Optimizer

### Stochastic Gradient Descent (SGD)

All implementations use vanilla SGD without momentum or weight decay:

```
W = W - learning_rate * grad_W
b = b - learning_rate * grad_b
```

| Model | Learning Rate | Batch Size | Function |
| --- | --- | --- | --- |
| MLP | 0.01 | 64 | [mnist_mlp.rs L260-L264](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L260-L264) |
| CNN | 0.01 | 32 | [mnist_cnn.rs L680-L692](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L680-L692) |
| Attention | 0.01 | 32 | [mnist_attention_pool.rs L930-L977](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L930-L977) |

The update is applied immediately after each mini-batch's gradient computation:

```mermaid
flowchart TD

FORWARD["Forward pass<br>Compute outputs"]
LOSS["Compute loss<br>and delta"]
BACKWARD["Backward pass<br>Compute gradients"]
UPDATE["SGD update<br>W -= lr * grad_W"]
NEXT["Next batch"]

FORWARD -.-> LOSS
LOSS -.-> BACKWARD
BACKWARD -.-> UPDATE
UPDATE -.-> NEXT
```

**Sources:** [mnist_mlp.rs L428-L431](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L428-L431)

 [mnist_cnn.rs L680-L692](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L680-L692)

 [mnist_attention_pool.rs L930-L977](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L930-L977)

## Evaluation

### Test Accuracy Computation

The `test()` / `test_accuracy()` functions evaluate model performance on held-out test data:

```mermaid
flowchart TD

TEST_DATA["test_images, test_labels"]
BATCH_LOOP["Process in batches of BATCH_SIZE"]
FORWARD["Forward pass (inference only)<br>No gradients computed"]
ARGMAX["Argmax prediction:<br>pred = argmax(logits[b])"]
COMPARE["Compare pred == label[b]"]
COUNT["correct += 1 if match"]
ACCURACY["accuracy = 100 * correct / total"]

FORWARD -.-> ARGMAX
ARGMAX -.-> COMPARE
COMPARE -.-> COUNT
COUNT -.-> BATCH_LOOP
```

**Sources:** [mnist_mlp.rs L450-L520](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L450-L520)

 [mnist_cnn.rs L559-L597](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L559-L597)

 [mnist_attention_pool.rs L979-L1167](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L979-L1167)

The evaluation process differs from training:

* No shuffling of indices
* No gradient computation or backpropagation
* No parameter updates
* Processes sequentially through test set

## Logging and Persistence

### Training Logs

All implementations write training metrics to CSV files in the `logs/` directory:

| Model | Log File | Format | Content |
| --- | --- | --- | --- |
| MLP | `logs/training_loss_c.txt` | CSV | `epoch,loss,time` |
| CNN | `logs/training_loss_cnn.txt` | CSV | `epoch,loss,time` |
| Attention | `logs/training_loss_attention_mnist.txt` | CSV | `epoch,loss,time` |

The logging process uses `BufWriter` for efficient I/O:

```mermaid
flowchart TD

CREATE["File::create()<br>logs/training_loss_*.txt"]
BUFWRITER["BufWriter::new()"]
EPOCH["After each epoch"]
COMPUTE["avg_loss = total_loss / N<br>time = elapsed.as_secs_f32()"]
WRITE["writeln!(log, '{},{},{}', epoch, avg_loss, time)"]

CREATE -.-> BUFWRITER
BUFWRITER -.-> EPOCH
EPOCH -.-> COMPUTE
COMPUTE -.-> WRITE
WRITE -.-> EPOCH
```

**Sources:** [mnist_mlp.rs L268-L272](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L268-L272)

 [mnist_mlp.rs L442-L445](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L442-L445)

 [mnist_cnn.rs L617-L621](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L617-L621)

 [mnist_cnn.rs L698](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_cnn.rs#L698-L698)

 [mnist_attention_pool.rs L1182-L1186](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L1182-L1186)

 [mnist_attention_pool.rs L1252](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_attention_pool.rs#L1252-L1252)

### Model Serialization

The MLP implementation includes `save_model()` for binary serialization:

```mermaid
flowchart TD

MODEL["NeuralNetwork struct"]
FILE["File::create('mnist_model.bin')"]
WRITE_DIM["Write dimensions:<br>input_size (i32)<br>hidden_size (i32)<br>output_size (i32)"]
WRITE_W1["Write hidden_layer.weights<br>as f64 array"]
WRITE_B1["Write hidden_layer.biases<br>as f64 array"]
WRITE_W2["Write output_layer.weights<br>as f64 array"]
WRITE_B2["Write output_layer.biases<br>as f64 array"]

MODEL -.-> FILE
FILE -.-> WRITE_DIM
```

**Sources:** [mnist_mlp.rs L522-L561](https://github.com/ThalesMMS/Rust-Neural-Networks/blob/0e978f90/mnist_mlp.rs#L522-L561)

The serialized model uses native endianness (via `to_ne_bytes()`) and converts f32 weights to f64 for storage. This format is consumed by the Python digit recognizer GUI (see [Digit Recognizer GUI](4a%20Digit-Recognizer-GUI.md)).



)

### On this page

* [Training Pipeline](5b%20Training-Visualization.md)
* [Purpose and Scope](5b%20Training-Visualization.md)
* [Training Pipeline Architecture](5b%20Training-Visualization.md)
* [Data Loading](5b%20Training-Visualization.md)
* [IDX Format Parsing](5b%20Training-Visualization.md)
* [Image and Label Loading](5b%20Training-Visualization.md)
* [Model Initialization](5b%20Training-Visualization.md)
* [Random Number Generation](5b%20Training-Visualization.md)
* [Xavier/Glorot Initialization](5b%20Training-Visualization.md)
* [Training Loop Structure](5b%20Training-Visualization.md)
* [Epoch and Batch Organization](5b%20Training-Visualization.md)
* [Mini-Batch Sampling](5b%20Training-Visualization.md)
* [Forward Propagation](5b%20Training-Visualization.md)
* [MLP Forward Pass (BLAS-Accelerated)](5b%20Training-Visualization.md)
* [CNN Forward Pass (Manual Loops)](5b%20Training-Visualization.md)
* [Attention Forward Pass](5b%20Training-Visualization.md)
* [Loss Computation](5b%20Training-Visualization.md)
* [Cross-Entropy Loss](5b%20Training-Visualization.md)
* [Backward Propagation](5b%20Training-Visualization.md)
* [Gradient Flow Through Layers](5b%20Training-Visualization.md)
* [MLP Backward Pass](5b%20Training-Visualization.md)
* [CNN Backward Pass](5b%20Training-Visualization.md)
* [Attention Backward Pass](5b%20Training-Visualization.md)
* [Optimizer](5b%20Training-Visualization.md)
* [Stochastic Gradient Descent (SGD)](5b%20Training-Visualization.md)
* [Evaluation](5b%20Training-Visualization.md)
* [Test Accuracy Computation](5b%20Training-Visualization.md)
* [Logging and Persistence](5b%20Training-Visualization.md)
* [Training Logs](5b%20Training-Visualization.md)
* [Model Serialization](5b%20Training-Visualization.md)

Ask Devin about Rust-Neural-Networks