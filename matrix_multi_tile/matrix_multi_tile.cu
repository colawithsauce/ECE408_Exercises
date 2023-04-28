#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 16

// @param:
//   A_d: M * S matrix
//   B_d: S * N matrix
//   C_d: M * N matrix
//   And we assume that blockDim.x == blocDim.y && blockDim.x == TILE_WIDTH
__global__ void
matrix_multi_tile_kernel(const double* A_d, const double* B_d, double* C_d, int M, int N, int S, int nSZa, int nBlkWidth)
{
    // Initialize space for tiling multiplication
    extern __shared__ double Ads_Bds[]; // Defined in the kernel arguments
    double* Ads = Ads_Bds;
    double* Bds = Ads_Bds + nSZa;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int Row = nBlkWidth * by + threadIdx.y;
    int Col = nBlkWidth * bx + threadIdx.x;

    // do matrix multiplication
    double Cvalue = 0;
    for (int ph = 0; ph < ceil(max((float)M / nBlkWidth, (float)N / nBlkWidth)); ph++) {
        if (Row < M && (ph * nBlkWidth + tx) < S) {
            Ads[ty * nBlkWidth + tx] = A_d[Row * S + (ph * nBlkWidth + tx)];
        } else {
            Ads[ty * nBlkWidth + tx] = 0;
        }

        if (Col < N && (ph * nBlkWidth + ty) < S) {
            Bds[ty * nBlkWidth + tx] = B_d[(ph * nBlkWidth + ty) * N + Col];
        } else {
            Bds[ty * nBlkWidth + tx] = 0;
        }

        __syncthreads();

        // Accumulate the result in this phase
        for (int i = 0; i != nBlkWidth; i++) {
            Cvalue += Ads[ty * nBlkWidth + i] * Bds[i * nBlkWidth + tx];
        }

        __syncthreads();
    }

    C_d[(Row * N) + Col] = Cvalue;
}
