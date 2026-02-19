// Metal Shading Language - Conv2D kernels
// Forward and backward kernels for 2D convolution operations.
// Data layout: NCHW (batch, channels, height, width) for both input and output.
// Filter layout: (out_channels, in_channels, kernel_h, kernel_w).

#include <metal_stdlib>
using namespace metal;

/// Conv2D forward pass: each thread computes one output pixel.
///
/// params: [batch_size, in_channels, out_channels, input_h, input_w,
///          kernel_h, kernel_w, stride, padding, out_h, out_w]
/// gid.x = output column, gid.y = output row, gid.z = batch * out_channels + oc
kernel void conv2d_forward(
    device const float* input [[buffer(0)]],
    device const float* filters [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint in_channels = params[1];
    uint out_channels = params[2];
    uint input_h = params[3];
    uint input_w = params[4];
    uint kernel_h = params[5];
    uint kernel_w = params[6];
    uint stride = params[7];
    uint padding = params[8];
    uint out_h = params[9];
    uint out_w = params[10];

    uint col = gid.x;
    uint row = gid.y;
    uint boc = gid.z;
    uint b = boc / out_channels;
    uint oc = boc % out_channels;

    if (col >= out_w || row >= out_h || b >= batch_size) return;

    float sum = bias[oc];
    for (uint ic = 0; ic < in_channels; ic++) {
        for (uint kh = 0; kh < kernel_h; kh++) {
            for (uint kw = 0; kw < kernel_w; kw++) {
                int ih = (int)(row * stride + kh) - (int)padding;
                int iw = (int)(col * stride + kw) - (int)padding;
                if (ih >= 0 && ih < (int)input_h && iw >= 0 && iw < (int)input_w) {
                    uint input_idx = b * (in_channels * input_h * input_w)
                                   + ic * (input_h * input_w)
                                   + (uint)ih * input_w + (uint)iw;
                    uint filter_idx = oc * (in_channels * kernel_h * kernel_w)
                                    + ic * (kernel_h * kernel_w)
                                    + kh * kernel_w + kw;
                    sum += input[input_idx] * filters[filter_idx];
                }
            }
        }
    }

    uint output_idx = b * (out_channels * out_h * out_w)
                    + oc * (out_h * out_w)
                    + row * out_w + col;
    output[output_idx] = sum;
}

/// Conv2D backward pass - compute gradient w.r.t. input (transposed convolution).
///
/// Each thread computes the gradient for one input pixel position.
/// gid.x = input column, gid.y = input row, gid.z = batch * in_channels + ic
///
/// params: [batch_size, in_channels, out_channels, input_h, input_w,
///          kernel_h, kernel_w, stride, padding, out_h, out_w]
kernel void conv2d_backward_input(
    device const float* grad_output [[buffer(0)]],
    device const float* filters [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint in_channels = params[1];
    uint out_channels = params[2];
    uint input_h = params[3];
    uint input_w = params[4];
    uint kernel_h = params[5];
    uint kernel_w = params[6];
    uint stride = params[7];
    uint padding = params[8];
    uint out_h = params[9];
    uint out_w = params[10];

    uint ix = gid.x;  // input column
    uint iy = gid.y;  // input row
    uint bic = gid.z; // batch * in_channels + ic
    uint b = bic / in_channels;
    uint ic = bic % in_channels;

    if (ix >= input_w || iy >= input_h || b >= batch_size) return;

    float sum = 0.0f;

    // For each output channel, accumulate gradient contributions
    for (uint oc = 0; oc < out_channels; oc++) {
        uint go_base = b * (out_channels * out_h * out_w) + oc * (out_h * out_w);
        uint f_base = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w);

        for (uint kh = 0; kh < kernel_h; kh++) {
            for (uint kw = 0; kw < kernel_w; kw++) {
                // Which output position (oy, ox) used input position (iy, ix)
                // with kernel offset (kh, kw)?
                // iy = oy * stride + kh - padding => oy = (iy + padding - kh) / stride
                int oy_num = (int)iy + (int)padding - (int)kh;
                int ox_num = (int)ix + (int)padding - (int)kw;

                if (oy_num >= 0 && ox_num >= 0 &&
                    oy_num % (int)stride == 0 && ox_num % (int)stride == 0) {
                    uint oy = (uint)oy_num / stride;
                    uint ox = (uint)ox_num / stride;
                    if (oy < out_h && ox < out_w) {
                        float go = grad_output[go_base + oy * out_w + ox];
                        float fw = filters[f_base + kh * kernel_w + kw];
                        sum += go * fw;
                    }
                }
            }
        }
    }

    uint input_idx = b * (in_channels * input_h * input_w)
                   + ic * (input_h * input_w)
                   + iy * input_w + ix;
    grad_input[input_idx] = sum;
}

/// Conv2D backward pass - compute gradient w.r.t. filters.
///
/// Each thread computes the gradient for one filter weight.
/// gid.x = kw, gid.y = kh, gid.z = oc * (in_channels * kernel_h * kernel_w) + ic * kernel_h * kernel_w + ...
/// Actually: gid.x = kw, gid.y = kh, gid.z = oc * in_channels + ic
///
/// params: [batch_size, in_channels, out_channels, input_h, input_w,
///          kernel_h, kernel_w, stride, padding, out_h, out_w]
kernel void conv2d_backward_filters(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_filters [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint in_channels = params[1];
    uint out_channels = params[2];
    uint input_h = params[3];
    uint input_w = params[4];
    uint kernel_h = params[5];
    uint kernel_w = params[6];
    uint stride = params[7];
    uint padding = params[8];
    uint out_h = params[9];
    uint out_w = params[10];

    uint kw = gid.x;
    uint kh = gid.y;
    uint oc_ic = gid.z; // oc * in_channels + ic
    uint oc = oc_ic / in_channels;
    uint ic = oc_ic % in_channels;

    if (kw >= kernel_w || kh >= kernel_h || oc >= out_channels) return;

    float sum = 0.0f;

    // Sum over batch and output spatial positions
    for (uint b = 0; b < batch_size; b++) {
        uint go_base = b * (out_channels * out_h * out_w) + oc * (out_h * out_w);
        uint in_base = b * (in_channels * input_h * input_w) + ic * (input_h * input_w);

        for (uint oy = 0; oy < out_h; oy++) {
            for (uint ox = 0; ox < out_w; ox++) {
                int ih = (int)(oy * stride + kh) - (int)padding;
                int iw = (int)(ox * stride + kw) - (int)padding;

                if (ih >= 0 && ih < (int)input_h && iw >= 0 && iw < (int)input_w) {
                    float go = grad_output[go_base + oy * out_w + ox];
                    float inp = input[in_base + (uint)ih * input_w + (uint)iw];
                    sum += go * inp;
                }
            }
        }
    }

    uint filter_idx = oc * (in_channels * kernel_h * kernel_w)
                    + ic * (kernel_h * kernel_w)
                    + kh * kernel_w + kw;
    grad_filters[filter_idx] = sum;
}

/// Conv2D backward pass - compute gradient w.r.t. bias.
///
/// Each thread computes the gradient for one output channel's bias.
/// grad_bias[oc] = sum over batch and spatial positions of grad_output.
///
/// params: [batch_size, out_channels, out_h, out_w]
kernel void conv2d_backward_bias(
    device const float* grad_output [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint batch_size = params[0];
    uint out_channels = params[1];
    uint out_h = params[2];
    uint out_w = params[3];

    uint oc = gid;
    if (oc >= out_channels) return;

    float sum = 0.0f;
    uint spatial = out_h * out_w;

    for (uint b = 0; b < batch_size; b++) {
        uint base = b * (out_channels * spatial) + oc * spatial;
        for (uint i = 0; i < spatial; i++) {
            sum += grad_output[base + i];
        }
    }

    grad_bias[oc] = sum;
}
