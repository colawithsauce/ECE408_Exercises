#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"

__global__ void
reduction_kernel(float* in, int size, float func(float, float))
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // we are writing programs for kernel.
    for (int stride = 1; stride <= size / 2; stride *= 2) {
        if (i % (2 * stride) == 0) {
            in[i] = func(in[i], in[i + stride]);
        }
    }
}

float
reduction_kernel_launcher(float* in, int size, float func(float, float))
{
    float result;
    dim3 dimBlock, dimGrid;
    CUDA_INIT_VAR(float, in, size);
    dimBlock = { 32, 1, 1 };
    dimGrid = { (unsigned int)ceil(((float)size) / dimBlock.x), 1, 1 };

    reduction_kernel<<<dimBlock, dimGrid>>>(in_d, size, func);

    cudaMemcpy(
      (void*)&result, (const void*)in_d, sizeof(float), cudaMemcpyDeviceToHost);
    return result;

Error:
    CUDA_FREE_VAR(in);
    return 0;
}
