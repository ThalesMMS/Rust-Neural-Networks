// CUDA element-wise kernels for neural network operations.
// Activation functions, bias operations, and row reductions.
// Compiled at runtime via NVRTC (inline in cuda_backend.rs).

/// ReLU activation: x = max(0, x)
extern "C" /**
 * @brief Applies the rectified linear unit (ReLU) activation in-place to a 1D array.
 *
 * For each index i in [0, n), replaces data[i] with max(0.0f, data[i]).
 *
 * @param data Pointer to an array of floats containing n elements; values are modified in-place.
 * @param n Number of elements in the data array.
 */
__global__ void relu(float *data, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

/// ReLU backward: grad_input = grad_output * (input > 0)
extern "C" /**
 * @brief Computes the element-wise backward pass for the ReLU activation.
 *
 * For each index i in [0, n), writes the gradient with respect to the input:
 * if input[i] > 0, grad_input[i] is set to grad_output[i]; otherwise grad_input[i] is set to 0.
 *
 * @param input Pointer to the original forward input values (used to test > 0).
 * @param grad_output Pointer to the gradient with respect to the ReLU output.
 * @param grad_input Pointer to the buffer where gradients with respect to the input are written.
 * @param n Number of elements to process.
 */
__global__ void relu_backward(const float *input, const float *grad_output,
                                          float *grad_input, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
}

/// Sigmoid activation: x = 1 / (1 + exp(-x))
extern "C" /**
 * @brief Applies the sigmoid activation element-wise to a float array in place.
 *
 * Each element data[i] for 0 <= i < n is replaced with 1 / (1 + exp(-data[i])).
 *
 * @param data Pointer to an array of at least n floats; values are overwritten with their sigmoid.
 * @param n Number of elements to process.
 */
__global__ void sigmoid(float *data, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

/// Sigmoid backward: grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
extern "C" /**
 * @brief Computes the element-wise gradient of a sigmoid activation.
 *
 * For each index i in [0, n), writes grad_input[i] = grad_output[i] * s * (1 - s),
 * where s is the precomputed sigmoid value sigmoid_output[i].
 *
 * @param sigmoid_output Array of sigmoid activation outputs (s).
 * @param grad_output Upstream gradients corresponding to the outputs.
 * @param grad_input Destination array where computed input gradients are stored.
 * @param n Number of elements to process.
 */
__global__ void sigmoid_backward(const float *sigmoid_output,
                                             const float *grad_output,
                                             float *grad_input, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid_output[i];
        grad_input[i] = grad_output[i] * s * (1.0f - s);
    }
}

/// Add bias to each row: data[row * n + col] += bias[col]
extern "C" /**
 * @brief Adds a per-column bias vector to each row of a row-major batch matrix in-place.
 *
 * For each row r and column c where r < batch_size and c < n, increments data[r * n + c] by bias[c].
 *
 * @param data Pointer to the matrix data laid out row-major with shape (batch_size, n); updated in-place.
 * @param bias Pointer to a bias vector of length n; bias[col] is added to every row's column col.
 * @param batch_size Number of rows in the matrix.
 * @param n Number of columns in the matrix (length of the bias).
 */
__global__ void add_bias(float *data, const float *bias, int batch_size, int n) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < n) {
        data[row * n + col] += bias[col];
    }
}

/// Sum rows: out[col] = sum over batch dimension of data[b * n + col]
extern "C" /**
 * @brief Computes column-wise sums across a batch and writes them to the output.
 *
 * Each column index c in [0, n) receives the sum over rows b in [0, batch_size) of data[b * n + c].
 *
 * @param data Pointer to input data laid out row-major with shape (batch_size, n).
 * @param out Pointer to output buffer of length n where per-column sums will be stored.
 * @param batch_size Number of rows (batch size) in the input.
 * @param n Number of columns in the input and length of the output.
 */
__global__ void sum_rows(const float *data, float *out, int batch_size, int n) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += data[b * n + col];
        }
        out[col] = sum;
    }
}