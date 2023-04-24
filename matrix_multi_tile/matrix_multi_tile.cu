#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../cuda_alias.h"

#define TILE_WIDTH 16

// @param:
//   A_d: M * S matrix
//   B_d: S * N matrix
//   C_d: M * N matrix
//   And we assume that blockDim.x == blocDim.y && blockDim.x == TILE_WIDTH
template <typename T>
__global__ void
matrix_multi_tile_kernel(const T* A_d, const T* B_d, T* C_d, int M, int N, int S, int sza, int width)
{
    // Initialize space for tiling multiplication
    extern __shared__ T Ads_Bds[]; // Defined in the kernel arguments
    T* Ads = Ads_Bds;
    T* Bds = Ads_Bds + sza;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int Row = width * blockIdx.y + threadIdx.y;
    int Col = width * blockIdx.x + threadIdx.x;

    // do matrix multiplication
    T Cvalue = 0;
    for (int ph = 0; ph < ceil(max(M / (float)width, N / (float)width)); ph++) {
        // Load from A and B
        if (Row < width && (ph * width + tx)) {
        }

        // Accumulate this result.
    }

    C_d[(Row * N) + Col] = Cvalue;
}
