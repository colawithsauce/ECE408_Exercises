#include <cuda_runtime_api.h>
#include <device_types.h>

#define OUT_TILE_DIM 6
#define IN_TILE_DIM 8

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

__global__ void
stencil_kernel(float* A, float* B, unsigned int N, const int* c)
{
    // Get the position of output grid
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // 1-Ordered stencil method
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        B[i * N * N + j * N + k] = c[0] * A[i * N * N + j * N + k] +
                                   c[1] * A[(i + 1) * N * N + j * N + k] +
                                   c[2] * A[(i - 1) * N * N + j * N + k] +
                                   c[3] * A[i * N * N + (j + 1) * N + k] +
                                   c[4] * A[i * N * N + (j - 1) * N + k] +
                                   c[5] * A[i * N * N + j * N + k + 1] +
                                   c[6] * A[i * N * N + j * N + k - 1];
    }
}

__global__ void
stencil_with_tiling(float* in, float* out, unsigned int N, const int* c)
{
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Load from global memory into shared memory.
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }

    __syncthreads();

    // processing caculate
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] =
              c[0] * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
              c[1] * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x] +
              c[2] * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
              c[3] * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
              c[4] * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
              c[5] * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
              c[6] * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1];
        }
    }
}

__global__ void
stencil_with_coarsening(float* in, float* out, unsigned int N, const int* c)
{
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float prev[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float curr[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float next[IN_TILE_DIM][IN_TILE_DIM];

    // 载入 prev,  NOTE: 这里是 iStart - 1
    if (j >= 0 && k >= 0 && j < N && k < N && iStart - 1 >= 0 &&
        iStart - 1 < N) {
        prev[threadIdx.y][threadIdx.x] =
          in[(iStart - 1) * N * N + j * N + k]; // NOTE: use tx and ty here
    }

    // 载入 curr
    if (j >= 0 && k >= 0 && j < N && k < N && iStart >= 0 && iStart < N) {
        curr[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }

    // 循环计算并输出 output
    for (int i = iStart; i != iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && k >= 0 && j < N && k < N) {
            next[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 &&
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] =
                  c[0] * curr[threadIdx.y][threadIdx.x] +
                  c[1] * next[threadIdx.y][threadIdx.x] +
                  c[2] * prev[threadIdx.y][threadIdx.x] +
                  c[3] * curr[threadIdx.y + 1][threadIdx.x] +
                  c[4] * curr[threadIdx.y - 1][threadIdx.x] +
                  c[5] * curr[threadIdx.y][threadIdx.x + 1] +
                  c[6] * curr[threadIdx.y][threadIdx.x - 1];
            }
        }

        __syncthreads();

        prev[threadIdx.y][threadIdx.x] = curr[threadIdx.y][threadIdx.x];
        curr[threadIdx.y][threadIdx.x] = next[threadIdx.y][threadIdx.x];
    }
}

__global__ void
stencil_with_amend_conarsening(float* in,
                               float* out,
                               unsigned int N,
                               const int* c)
{
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float prev, next;
    __shared__ float curr[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 &&
        k < N) {
        prev = in[(iStart - 1) * N * N + j * N + k];
    }

    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        curr[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }

    for (int i = iStart; i != iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            next = in[(i + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (tz >= 1 && tz < IN_TILE_DIM - 1 && ty >= 1 &&
                ty < IN_TILE_DIM - 1 && tx >= 1 && tx < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] =
                  c[0] * curr[ty][tx] + c[1] * prev + c[2] * next +
                  c[3] * curr[ty + 1][tx] + c[4] * curr[ty - 1][tx] +
                  c[5] * curr[ty][tx + 1] + c[6] * curr[ty][tx - 1];
            }
        }

        __syncthreads();

        prev = curr[ty][tx];
        curr[ty][tx] = next;
    }
}
