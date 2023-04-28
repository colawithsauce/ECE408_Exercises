#include "../cuda_alias.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define msg "ERROR"

// check two matrix is same or not.
// input:
//  A and B: matrix with type M * N;
// output:
//  flag:
//      0 means same, and others means not same. User should init flag as 1.
template <class T>
__global__ void
matrix_same_kernel(const T* matA_d, const T* matB_d, int M, int N, int* flag)
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
    dim3 dimGrid = { static_cast<unsigned int>(ceil(M / 128.0)), static_cast<unsigned int>(ceil(N / 128.0)), 1 };
    dim3 dimBlock = { 128, 128, 1 };

    double *matA_d, *matB_d;
    err = cudaMalloc(&matA_d, sizeof(double) * M * N);
    CUDA_CHECK(err, msg);
    err = cudaMalloc(&matB_d, sizeof(double) * M * N);
    CUDA_CHECK(err, msg);
    err = cudaMemcpy(matA_d, matA_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, msg);
    err = cudaMemcpy(matB_d, matB_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, msg);

    matrix_same_kernel KERNEL_ARGS2(dimGrid, dimBlock)(matA_d, matB_d, M, N, &flag);

Error:
    cudaFree(matA_d);
    cudaFree(matB_d);
    return flag;
}
