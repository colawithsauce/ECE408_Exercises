#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void
mat_mul_coarsening_kernel(const double* M, const double* N, double* P, int width)
{
    __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identity the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty; // Keep same row
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx; // to coarsen tile means to "fatten" the working area each time here.

    // Initialize Pvalue for all output elements
    double Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < width / TILE_WIDTH; ph++) {
        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }
}

/* mat_mul_coarsening:
 *
 *  @desc: Matrix multiplication function integated with tiled
 *  and coarsening techenology.
 *
 *  NOTE: This function works based on following assumptions for params:
 *
 *  1. A_h, B_h, C_h all matrix with shape width x width.
 *
 *  2. TILE_WIDTH is 32, and `width` whould be able to fully divide by 32.
 *
 *  */
cudaError_t
mat_mul_coarsening(const double* A_h, const double* B_h, double* C_h, int width)
{
    double *A_d, *B_d, *C_d;
    cudaError_t err = cudaSuccess;

    dim3 dimGrid = { (unsigned int)ceil(width / (float)TILE_WIDTH), (unsigned int)ceil(width / (float)TILE_WIDTH), 1 };
    dim3 dimBlock = { TILE_WIDTH, TILE_WIDTH, 1 };

    err = cudaMalloc((void**)&A_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&B_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void**)&C_d, width * width * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMemcpy(A_d, A_h, width * width * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");
    err = cudaMemcpy(B_d, B_h, width * width * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    mat_mul_coarsening_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);

    err = cudaGetLastError();
    CUDA_CHECK(err, "Can't launch kernel matrix_multi_tile_kernel");

    err = cudaMemcpy(C_h, C_d, width * width * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");

Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return err;
}
