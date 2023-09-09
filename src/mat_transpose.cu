#include <cuda_runtime_api.h>

#include "cuda_alias.hpp"

const int BLOCK_WIDTH = 10;

__global__ void BlockTransposeKernel(double *A_elements, int A_width, int A_height)
{
    __shared__ double blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

cudaError_t BlockTranspose(double *A_elements, int A_width, int A_height)
{
    double *A_d;
    cudaError_t err = cudaSuccess;

    dim3 dimBlock, dimGrid;
    dimBlock = {BLOCK_WIDTH, BLOCK_WIDTH, 1};
    dimGrid = {(unsigned int)ceil((float)A_width / dimBlock.x), (unsigned int)ceil((float)A_height / dimBlock.y), 1};

    err = cudaMalloc(&A_d, A_width * A_height * sizeof(double));
    CUDA_CHECK(err, "can't cudaMalloc!");

    err = cudaMemcpy(A_d, A_elements, A_width * A_height * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy!");

    BlockTransposeKernel<<<dimGrid, dimBlock>>>(A_d, A_width, A_height);

    err = cudaMemcpy(A_elements, A_d, A_width * A_height * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy!");

Error:
    cudaFree(A_d);

    return err;
}
