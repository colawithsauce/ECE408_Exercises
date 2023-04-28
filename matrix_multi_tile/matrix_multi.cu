#include "../cuda_alias.h"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <locale>

__global__ void
matrix_multi_kernel(double* A_d, double* B_d, double* C_d, int M, int N, int S)
{
    // A_d is M x S, while B_d is S x N
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N) {
        double sum = 0;
        for (int i = 0; i != S; i++) {
            sum += A_d[row * S + i] * B_d[i * S + col];
        }

        C_d[row * N + col] = sum;
    }
}

extern __global__ void
matrix_multi_tile_kernel(const double* A_d, const double* B_d, double* C_d, int M, int N, int S, int nSZa, int nBlkWidth);

cudaError_t
matrix_multi(const double* A_h, const double* B_h, double* C_h_2, int M, int N, int S)
{
    double *A_d, *B_d, *C_d, *C_d1;
    clock_t start = clock();
    double elapsed = 0;
    cudaError_t err = cudaSuccess;

    dim3 dimGrid = { (unsigned int)ceil(N / 128.0), (unsigned int)ceil(M / 128.0),
        1 };
    dim3 dimBlock = { 128, 128, 1 };

    err = cudaMalloc((void**)&A_d, M * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&B_d, N * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&C_d, M * N * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&C_d1, M * N * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMemcpy(A_d, A_h, M * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");
    err = cudaMemcpy(B_d, B_h, N * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    // compare the time consume
    // first
    start = clock();
    matrix_multi_kernel KERNEL_ARGS2(dimGrid, dimBlock)(A_d, B_d, C_d, M, N, S);
    elapsed = 1000 * (double)(clock() - start) / CLOCKS_PER_SEC; // in milliseconds
    printf("normal matrix_multi: %lf ms\n", elapsed);

    err = cudaMemcpy(C_h_2, C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    // second
    start = clock();
    matrix_multi_tile_kernel KERNEL_ARGS3(dimGrid, dimBlock, dimBlock.x * 2)(A_d, B_d, C_d, M, N, S, dimBlock.x, dimBlock.x);
    elapsed = 1000 * (double)(clock() - start) / CLOCKS_PER_SEC; // in milliseconds
    printf("normal matrix_multi: %lf ms\n", elapsed);

    err = cudaMemcpy(C_h_2 + M * N, C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");
Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return err;
}
