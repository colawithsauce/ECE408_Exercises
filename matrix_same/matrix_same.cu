#include "cuda_alias.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define msg "ERROR"

// check two matrix is same or not.
// input:
//  A and B: matrix with type M * N;
// output:
//  flag:
//      0 means same, and others means not same. User should init flag as 1.
__global__ void
matrix_same_kernel(const double* matA_d, const double* matB_d, int M, int N, int* flag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N) {
        if (matA_d[y * N + x] != matB_d[y * N + x]) {
            *flag = 0;
        }
    }
}

bool matrix_same(const double* matA_h, const double* matB_h, int M, int N)
{
    int flag = 1;
    cudaError_t err;
    dim3 dimGrid = { static_cast<unsigned int>(ceil(M / 32.0)), static_cast<unsigned int>(ceil(N / 32.0)), 1 };
    dim3 dimBlock = { 32, 32, 1 };

    double *matA_d, *matB_d;
    err = cudaMalloc((void**)&matA_d, sizeof(double) * M * N);
    CUDA_CHECK(err, "Can't Malloc");
    err = cudaMalloc((void**)&matB_d, sizeof(double) * M * N);
    CUDA_CHECK(err, "Can't Malloc");
    err = cudaMemcpy(matA_d, matA_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't Memcpy");
    err = cudaMemcpy(matB_d, matB_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't Memcpy");

    matrix_same_kernel KERNEL_ARGS2(dimGrid, dimBlock)(matA_d, matB_d, M, N, &flag);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Launch kernel matrix_same_kernel failed");

    cudaFree(matA_d);
    cudaFree(matB_d);
    return flag;

Error:
    cudaFree(matA_d);
    cudaFree(matB_d);

    exit(1);
}
