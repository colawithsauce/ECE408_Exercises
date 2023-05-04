#include "../cuda_alias.h"
#include <bits/types/clock_t.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 2

// @param:
//   A_d: M * S matrix
//   B_d: S * N matrix
//   C_d: M * N matrix
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

__global__ void
matrix_multi_tile_simple_kernel(const double* A_d, const double* B_d, double* C_d, int width)
{
    __shared__ double Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bds[TILE_WIDTH][TILE_WIDTH];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int Row = TILE_WIDTH * by + threadIdx.y;
    int Col = TILE_WIDTH * bx + threadIdx.x;

    double Cvalue = 0.0;
    for (int ph = 0; ph < width / (float)TILE_WIDTH; ph++) {
        Ads[ty][tx] = A_d[Row * width + ph * TILE_WIDTH + tx];
        Bds[ty][tx] = B_d[(ph * TILE_WIDTH + ty) * width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += Ads[ty][k] * Bds[k][tx];
        }

        __syncthreads();
    }

    C_d[Row * width + Col] = Cvalue;
}

cudaError_t
matrix_multi_tile(const double* A_h, const double* B_h, double* C_h, int M, int N, int S)
{
    double *A_d, *B_d, *C_d;
    cudaError_t err = cudaSuccess;

    // dim3 dimGrid = { (unsigned int)ceil(N / 32.0), (unsigned int)ceil(M / 32.0), 1 };
    // dim3 dimBlock = { 32, 32, 1 };

    dim3 dimGrid = { (unsigned int)ceil(N / 2.0), (unsigned int)ceil(M / 2.0), 1 };
    dim3 dimBlock = { 2, 2, 1 };

    err = cudaMalloc((void**)&A_d, M * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&B_d, N * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&C_d, M * N * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMemcpy(A_d, A_h, M * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");
    err = cudaMemcpy(B_d, B_h, N * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    // matrix_multi_tile_kernel<<<dimGrid, dimBlock, 1024 * 2 * sizeof(double)>>>(A_d,
    //     B_d,
    //     C_d,
    //     M,
    //     N,
    //     S,
    //     1024,
    //     32);
    matrix_multi_tile_simple_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, 2);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Can't launch kernel matrix_multi_tile_kernel");

    err = cudaMemcpy(C_h, C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");

Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return err;
}
