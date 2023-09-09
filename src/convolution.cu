#include <assert.h>
#include <cuda_runtime_api.h>
#include <device_types.h>
#include <sys/types.h>

#include "cuda_alias.hpp"

/* Normal convolution function:
 *
 *  input: N, M, size
 *
 *  output: P
 *
 *  desc: Caculate convolution P = N * M, where
 *   */
static __global__ void no_opt_kernel(const double *N, const double *M, double *P, int size, const int MASK_WIDTH)
{
    int halo = (MASK_WIDTH - 1) / 2;
    int row_i = threadIdx.y + blockIdx.y * blockDim.y - halo;
    int col_i = threadIdx.x + blockIdx.x * blockDim.x - halo;

    double Pvalue = 0.0f;
    for (int j = 0; j != MASK_WIDTH; ++j)
    {
        for (int i = 0; i != MASK_WIDTH; ++i)
        {
            if (row_i + j >= 0 && col_i + i >= 0 && row_i + j < size && col_i + i < size)
            {
                Pvalue += M[j * MASK_WIDTH + i] * N[(row_i + j) * size + col_i + i];
            }
        }
    }

    P[(row_i + halo) * size + col_i + halo] = Pvalue;
}

static __global__ void optimized_kernel(const double *N, const double *M, double *P, int size, const int MASK_WIDTH)
{
    int halo = MASK_WIDTH / 2;
    // The 3rd strategy.
    int row_o = threadIdx.y + blockIdx.y * blockDim.y;
    int col_o = threadIdx.x + blockIdx.x * blockDim.x;
}

/* Convolution function */
namespace convolution
{
cudaError_t no_opt(const double *N_h, const double *M_h, double *P_h, int size, const int MASK_WIDTH)
{
    assert(MASK_WIDTH % 2 == 1); // Mask width must be odd.

    double *M, *N, *P;
    cudaError_t err = cudaSuccess;

    // TODO: #8 improve it?
    dim3 dimBlock = {(uint)ceil(size / 32.0), (uint)ceil(size / 32.0), 1};
    dim3 dimGrid = {32, 32, 1};

    cudaMalloc((void **)M, sizeof(double) * MASK_WIDTH * MASK_WIDTH);
    CUDA_CHECK(err, "Failed when cuda malloc");
    cudaMalloc((void **)N, sizeof(double) * size * size);
    CUDA_CHECK(err, "Failed when cuda malloc");
    cudaMalloc((void **)P, sizeof(double) * size * size);
    CUDA_CHECK(err, "Failed when cuda malloc");

    cudaMemcpy((void *)M, (const void *)M_h, sizeof(double) * MASK_WIDTH * MASK_WIDTH, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Failed when cuda memcpy");
    cudaMemcpy((void *)N, (const void *)N_h, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Failed when cuda memcpy");
    cudaMemcpy((void *)P, (const void *)P_h, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Failed when cuda memcpy");

    no_opt_kernel<<<dimGrid, dimBlock>>>((const double *)N, (const double *)M, (double *)P, size, MASK_WIDTH);

    cudaMemcpy((void *)P_h, (const void *)P, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Failed when cuda memcpy");

Error:
    cudaFree((void *)M);
    cudaFree((void *)N);
    cudaFree((void *)P);

    return err;
}
} // namespace convolution
