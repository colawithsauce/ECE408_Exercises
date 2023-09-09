#include "cuda_alias.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define TOLERANCE 0.0000001

// check two matrix is same or not.
// input:
//  A and B: matrix with type M * N;
// output:
//  flag:
//      0 means same, and others means not same. User should init flag as 1.
__global__ void matrix_same_kernel(const double *matA_d, const double *matB_d, int M, int N, int *flag_d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N)
    {
        if (abs(matA_d[y * N + x] - matB_d[y * N + x]) > TOLERANCE)
        {
            *flag_d = 0;
        }
    }
}

bool matrix_same(const double *matA_h, const double *matB_h, int M, int N)
{
    int flag = 1;
    cudaError_t err;
    dim3 dimGrid = {(unsigned int)ceil(M / 32.0), (unsigned int)ceil(N / 32.0), 1};
    dim3 dimBlock = {32, 32, 1};

    printf("Launch kernel with dimGrid: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

    int *flag_d;
    double *matA_d, *matB_d;

    err = cudaMalloc((void **)&matA_d, sizeof(double) * M * N);
    CUDA_CHECK(err, "Can't Malloc");
    err = cudaMalloc((void **)&matB_d, sizeof(double) * M * N);
    CUDA_CHECK(err, "Can't Malloc");
    err = cudaMalloc((void **)&flag_d, sizeof(int));
    CUDA_CHECK(err, "Can't Malloc");

    err = cudaMemcpy(matA_d, matA_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't Memcpy");
    err = cudaMemcpy(matB_d, matB_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't Memcpy");
    err = cudaMemcpy(flag_d, &flag, sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't Memcpy");

    matrix_same_kernel<<<dimGrid, dimBlock>>>(matA_d, matB_d, M, N, flag_d);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Launch kernel matrix_same_kernel failed");

    err = cudaMemcpy(&flag, flag_d, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't Memcpy");

    cudaFree(matA_d);
    cudaFree(matB_d);
    return flag;

Error:
    cudaFree(matA_d);
    cudaFree(matB_d);

    exit(1);
}
