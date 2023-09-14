#include <algorithm>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"

const int BLOCK_WIDTH = 16;

__global__ void
matTransposeKernel(double* in_d, double* out_d, int width, int height)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < width && i < height) {
        out_d[i + j * height] = in_d[i * width + j];
    }
}

cudaError_t
matTranspose(double* in_h, double* out_h, int width, int height)
{
    double *in_d, *out_d;
    cudaError_t err = cudaSuccess;

    int T = std::max(width, height);

    dim3 dimBlock, dimGrid;
    dimBlock = { BLOCK_WIDTH, BLOCK_WIDTH, 1 };
    dimGrid = { (unsigned int)ceil((float)T / dimBlock.x),
                (unsigned int)ceil((float)T / dimBlock.y),
                1 };

    err = cudaMalloc(&in_d, width * height * sizeof(double));
    CUDA_CHECK(err, "can't cudaMalloc!");
    err = cudaMalloc(&out_d, width * height * sizeof(double));
    CUDA_CHECK(err, "can't cudaMalloc!");

    err = cudaMemcpy(
      in_d, in_h, width * height * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy!");

    matTransposeKernel<<<dimGrid, dimBlock>>>(in_d, out_d, width, height);

    err = cudaMemcpy(
      out_h, out_d, width * height * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy!");

Error:
    cudaFree(in_d);
    cudaFree(out_d);

    return err;
}
