// CUDA 2D convolution kernels for neural network operations.
// Forward pass and three backward sub-kernels (input, filters, bias gradients).
// Compiled at runtime via NVRTC (inline in cuda_backend.rs).
//
// Data layout: NCHW (batch × channels × height × width)
// Filter layout: (out_channels × in_channels × kernel_h × kernel_w)

/// Conv2D forward pass.
/// Each thread computes one output pixel for one (batch, out_channel) pair.
/// Grid z-dimension encodes batch_size * out_channels.
extern "C" /**
 * @brief Compute the forward pass of a 2D convolution for a batched NCHW input.
 *
 * Performs a standard convolution with explicit stride and padding, adds the bias
 * for each output channel, and stores the result in the output tensor.
 *
 * Data layouts:
 * - input: NCHW (batch_size × in_channels × input_h × input_w)
 * - filters: (out_channels × in_channels × kernel_h × kernel_w)
 * - bias: length out_channels
 * - output: NCHW (batch_size × out_channels × out_h × out_w)
 *
 * Grid/thread mapping:
 * - Each thread computes one output pixel (ow, oh) for a specific (batch, out_channel).
 * - blockIdx.z encodes the combined batch and output-channel index: b = idx / out_channels; oc = idx % out_channels.
 * - Threads whose computed coordinates are outside output bounds return without writing.
 *
 * @param input Pointer to input activations in NCHW layout.
 * @param filters Pointer to convolution filters laid out as (out_channels, in_channels, kernel_h, kernel_w).
 * @param bias Pointer to per-output-channel biases (length out_channels).
 * @param output Pointer to output tensor in NCHW layout where results are written.
 * @param stride Number of pixels to step between receptive field positions.
 * @param padding Number of zero-padding pixels applied to each spatial side of the input.
 */
__global__ void conv2d_forward(
    const float *input, const float *filters, const float *bias, float *output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int b = idx / out_channels;
    int oc = idx % out_channels;

    if (ow >= out_w || oh >= out_h || b >= batch_size) return;

    float sum = bias[oc];
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    float iv = input[((b * in_channels + ic) * input_h + ih) * input_w + iw];
                    float fv = filters[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                    sum += iv * fv;
                }
            }
        }
    }
    output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
}

/// Conv2D backward pass: compute gradient w.r.t. input.
/// Each thread computes one input gradient pixel for one (batch, in_channel) pair.
extern "C" /**
 * @brief Compute gradients with respect to the input for a 2D convolution.
 *
 * Accumulates contributions from the output gradients and convolution filters
 * to produce grad_input for each (batch, in_channel, ih, iw). Handles stride
 * and padding by mapping each input position to corresponding output positions
 * and summing valid contributions.
 *
 * Data layout:
 * - grad_output and grad_input: NCHW (batch × channels × height × width).
 * - filters: (out_channels × in_channels × kernel_h × kernel_w).
 *
 * @param grad_output Pointer to output gradients (shape: batch_size × out_channels × out_h × out_w).
 * @param filters Pointer to convolution filters (shape: out_channels × in_channels × kernel_h × kernel_w).
 * @param grad_input Pointer to memory where computed input gradients are written (shape: batch_size × in_channels × input_h × input_w).
 * @param batch_size Number of examples in the batch.
 * @param in_channels Number of input channels.
 * @param out_channels Number of output channels.
 * @param input_h Input height.
 * @param input_w Input width.
 * @param kernel_h Filter kernel height.
 * @param kernel_w Filter kernel width.
 * @param stride Stride used by the convolution.
 * @param padding Padding applied to the input.
 * @param out_h Output (feature map) height.
 * @param out_w Output (feature map) width.
 */
__global__ void conv2d_backward_input(
    const float *grad_output, const float *filters, float *grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int iw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ih_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int b = idx / in_channels;
    int ic = idx % in_channels;

    if (iw_idx >= input_w || ih_idx >= input_h || b >= batch_size) return;

    float sum = 0.0f;
    for (int oc = 0; oc < out_channels; oc++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int oh = ih_idx + padding - kh;
                int ow = iw_idx + padding - kw;
                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        float gov = grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                        float fv = filters[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                        sum += gov * fv;
                    }
                }
            }
        }
    }
    grad_input[((b * in_channels + ic) * input_h + ih_idx) * input_w + iw_idx] = sum;
}

/// Conv2D backward pass: compute gradient w.r.t. filter weights.
/// Each thread computes one filter gradient element for one (out_channel, in_channel) pair.
extern "C" /**
 * @brief Computes gradients for convolution filter weights by correlating inputs with output gradients.
 *
 * Accumulates dL/dW for each filter element (oc, ic, kh, kw) by summing input(b, ic, ih, iw) * grad_output(b, oc, oh, ow)
 * over all batches and spatial output positions where the corresponding input position is valid given stride and padding.
 *
 * Data layout:
 * - input is NCHW: (batch_size, in_channels, input_h, input_w)
 * - grad_output is NCHW: (batch_size, out_channels, out_h, out_w)
 * - grad_filters is laid out as (out_channels, in_channels, kernel_h, kernel_w)
 *
 * @param input Pointer to input activations.
 * @param grad_output Pointer to gradients w.r.t. the convolution outputs.
 * @param grad_filters Output pointer where computed filter gradients are written.
 * @param batch_size Number of examples in the batch.
 * @param in_channels Number of input channels.
 * @param out_channels Number of output channels.
 * @param input_h Input height.
 * @param input_w Input width.
 * @param kernel_h Kernel (filter) height.
 * @param kernel_w Kernel (filter) width.
 * @param stride Stride used by the convolution.
 * @param padding Padding applied to the input (zero-padding).
 * @param out_h Output height (resulting from convolution).
 * @param out_w Output width (resulting from convolution).
 */
__global__ void conv2d_backward_filters(
    const float *input, const float *grad_output, float *grad_filters,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride, int padding, int out_h, int out_w)
{
    int kw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kh_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int oc = idx / in_channels;
    int ic = idx % in_channels;

    if (kw_idx >= kernel_w || kh_idx >= kernel_h || oc >= out_channels) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int ih = oh * stride - padding + kh_idx;
                int iw = ow * stride - padding + kw_idx;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    float iv = input[((b * in_channels + ic) * input_h + ih) * input_w + iw];
                    float gov = grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                    sum += iv * gov;
                }
            }
        }
    }
    grad_filters[((oc * in_channels + ic) * kernel_h + kh_idx) * kernel_w + kw_idx] = sum;
}

/// Conv2D backward pass: compute gradient w.r.t. bias.
/// Each thread computes one bias gradient (sum over batch and spatial dims).
extern "C" /**
 * @brief Computes gradients for biases by summing output gradients over batch and spatial dimensions.
 *
 * Accumulates grad_output for each output channel across all batches and spatial positions and writes
 * the result into grad_bias[oc].
 *
 * @param grad_output Pointer to output gradients laid out as NCHW where channel dimension equals out_channels.
 *                    Memory is indexed as ((b * out_channels + oc) * out_h + oh) * out_w + ow.
 * @param grad_bias Pointer to an array of length out_channels where the computed bias gradients are written.
 * @param batch_size Number of examples in the batch (N).
 * @param out_channels Number of output channels (C_out).
 * @param out_h Output height (H_out).
 * @param out_w Output width (W_out).
 */
__global__ void conv2d_backward_bias(
    const float *grad_output, float *grad_bias,
    int batch_size, int out_channels, int out_h, int out_w)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_channels) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                sum += grad_output[((b * out_channels + oc) * out_h + oh) * out_w + ow];
            }
        }
    }
    grad_bias[oc] = sum;
}