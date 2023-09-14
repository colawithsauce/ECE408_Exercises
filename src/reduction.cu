#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"

/* NOTE: This is not an multi-block kernel. which means it can't handle more
 * than one block in put data.
 *
 * The multi-block version would be introduced later in this Chapter!!!
 *
 * NOTE: This kernel assume the size of input become the result of pow(2, n) and
 * that the blockDim.x become ~size / 2~.
 * */
__global__ void
reduction_kernel_1(float* in, float* out, int size)
{
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (i % stride == 0) {
            in[i] += in[stride + i];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out = in[0];
    }
}

// __global__ void
// reduction_kernel_2(float* input, float* output)
// {
//     unsigned int i = threadIdx.x;
// }

cudaError_t
reduction_kernel_1_launcher(float* in, float* out, int size)
{
    dim3 dimBlock, dimGrid;
    cudaError_t err = cudaSuccess;

    float *in_d = nullptr, *out_d = nullptr;
    err = cudaMalloc((void**)&in_d, sizeof(float) * size);
    CUDA_CHECK(err, "Malloc failed!");

    err = cudaMalloc((void**)&out_d, sizeof(float) * 1);
    CUDA_CHECK(err, "Malloc failed!");

    err = cudaMemcpy(
      (void*)in_d, (void*)in, sizeof(float) * size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcpy failed!");

    err = cudaMemcpy(
      (void*)out_d, (void*)out, sizeof(float) * 1, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Memcpy failed!");

    dimBlock = { (uint)size / 2, 1, 1 };
    dimGrid = { 1, 1, 1 };

    printf("Launching kernel with dimGrid %u ...\n", dimGrid.x);
    reduction_kernel_1<<<dimGrid, dimBlock>>>(in_d, out_d, size);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Error when calling kernel");

    cudaMemcpy(out, out_d, sizeof(float), cudaMemcpyDeviceToHost);

Error:
    cudaFree(in_d);
    cudaFree(out_d);
    return err;
}
