#include "cuda_alias.hpp"
#include <cuda_runtime_api.h>

const int BLOCK_WIDTH = 10;

__global__ void
BlockTransposeKernel(double* A_elements, int A_width, int A_height)
{
    __shared__ double blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

cudaError_t BlockTranspose(double* A_elements, int A_width, int A_height)
{
}
