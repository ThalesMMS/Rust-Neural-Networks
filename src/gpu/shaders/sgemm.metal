// Metal Shading Language - SGEMM kernels
// Tiled matrix multiplication with threadgroup (shared) memory for performance.
//
// All matrices are stored in row-major order.
// C = alpha * op(A) * op(B) + beta * C

#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 16;

/// Tiled matrix multiply: C = A * B
/// A is (M x K), B is (K x N), C is (M x N), all row-major.
kernel void sgemm_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;

        // Load tile of A
        if (row < M && aCol < K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B
        if (bRow < K && col < N) {
            Bs[tid.y][tid.x] = B[bRow * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[tid.y][i] * Bs[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Tiled C = A^T * B  (A stored as K x M, transposed to M x K)
kernel void sgemm_tn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint kIdx = t * TILE_SIZE + tid.x;  // column in logical A (row in stored A)
        uint bRow = t * TILE_SIZE + tid.y;

        // Load tile of A^T: logical A[row][kIdx] = stored A[kIdx][row] = A[kIdx * M + row]
        if (row < M && kIdx < K) {
            As[tid.y][tid.x] = A[kIdx * M + row];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B
        if (bRow < K && col < N) {
            Bs[tid.y][tid.x] = B[bRow * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[tid.y][i] * Bs[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Tiled C = A * B^T  (B stored as N x K, transposed to K x N)
kernel void sgemm_nt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint M = params[0];
    uint N = params[1];
    uint K = params[2];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        uint kIdx = t * TILE_SIZE + tid.y;  // row in logical B^T = column in stored B

        // Load tile of A
        if (row < M && aCol < K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B^T: logical B^T[kIdx][col] = stored B[col][kIdx] = B[col * K + kIdx]
        if (kIdx < K && col < N) {
            Bs[tid.y][tid.x] = B[col * K + kIdx];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += As[tid.y][i] * Bs[i][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
