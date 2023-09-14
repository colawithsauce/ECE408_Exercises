#include "cuda_alias.hpp"

#define TILE_WIDTH 16

__global__ void
matrix_multi_tile_simple_kernel(const double* A_d,
                                const double* B_d,
                                double* C_d,
                                int width)
{
    __shared__ double Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bds[TILE_WIDTH][TILE_WIDTH];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int Row = TILE_WIDTH * by + threadIdx.y;
    int Col = TILE_WIDTH * bx + threadIdx.x;

    double Cvalue = 0.0;
    for (int ph = 0; ph < width / (float)TILE_WIDTH; ph++) {
        Ads[ty][tx] = A_d[Row * width + ph * TILE_WIDTH + tx];
        Bds[ty][tx] = B_d[(ph * TILE_WIDTH + ty) * width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += Ads[ty][k] * Bds[k][tx];
        }

        __syncthreads();
    }

    C_d[Row * width + Col] = Cvalue;
}

cudaError_t
matrix_multi_tile_simple(const double* A_h,
                         const double* B_h,
                         double* C_h,
                         int width)
{
    double *A_d, *B_d, *C_d;
    cudaError_t err = cudaSuccess;

    dim3 dimGrid = { (unsigned int)ceil(width / (float)TILE_WIDTH),
                     (unsigned int)ceil(width / (float)TILE_WIDTH),
                     1 };
    dim3 dimBlock = { TILE_WIDTH, TILE_WIDTH, 1 };

    err = cudaMalloc((void**)&A_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&B_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&C_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMemcpy(
      A_d, A_h, width * width * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");
    err = cudaMemcpy(
      B_d, B_h, width * width * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    matrix_multi_tile_simple_kernel<<<dimGrid, dimBlock>>>(
      A_d, B_d, C_d, width);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Can't launch kernel matrix_multi_tile_kernel");

    err = cudaMemcpy(
      C_h, C_d, width * width * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");

Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return err;
}
