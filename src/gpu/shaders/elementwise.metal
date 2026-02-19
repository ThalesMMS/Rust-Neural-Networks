// Metal Shading Language - Element-wise kernels
// Activation functions, bias operations, and row reductions for neural network layers.

#include <metal_stdlib>
using namespace metal;

/// ReLU activation: x = max(0, x)
kernel void relu(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],  // [length]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    data[gid] = max(0.0f, data[gid]);
}

/// ReLU backward: grad_input = grad_output * (input > 0)
kernel void relu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [length]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    grad_input[gid] = input[gid] > 0.0f ? grad_output[gid] : 0.0f;
}

/// Sigmoid activation: x = 1 / (1 + exp(-x))
kernel void sigmoid(
    device float* data [[buffer(0)]],
    device const uint* params [[buffer(1)]],  // [length]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    data[gid] = 1.0f / (1.0f + exp(-data[gid]));
}

/// Sigmoid backward: grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
kernel void sigmoid_backward(
    device const float* sigmoid_output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [length]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    float s = sigmoid_output[gid];
    grad_input[gid] = grad_output[gid] * s * (1.0f - s);
}

/// Add bias to each row: data[b*n + j] += bias[j]
kernel void add_bias(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [batch_size, n]
    uint2 gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint n = params[1];
    uint row = gid.y;
    uint col = gid.x;

    if (row >= batch_size || col >= n) return;
    data[row * n + col] += bias[col];
}

/// Sum rows: out[j] = sum_b data[b*n + j]
kernel void sum_rows(
    device const float* data [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [batch_size, n]
    uint gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint n = params[1];

    if (gid >= n) return;

    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += data[b * n + gid];
    }
    out[gid] = sum;
}
