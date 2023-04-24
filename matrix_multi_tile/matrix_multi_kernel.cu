#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define PRINT(msg) printf("%s:\nat : FILE %s\nat: LINE %d", msg, __FILE__, __LINE__);
#define CUDA_CHECK(err, msg)                                                                                           \
    if (err != cudaSuccess)                                                                                            \
    {                                                                                                                  \
        PRINT(msg);                                                                                                    \
        goto Error;                                                                                                    \
    }

const int TILE_WIDTH = 16;
template <typename T> __global__ void matrix_multi_tile_kernel(T *M, T *N, T *P, int Width)
{
    // Alloc share memory within block
    __shared__ T Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ T Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph)
    {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

__global__ void matrix_multi_kernel(double *A_d, double *B_d, double *C_d, int M, int N, int S)
{
    // A_d is M x S, while B_d is S x N
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N)
    {
        double sum = 0;
        for (int i = 0; i != S; i++)
        {
            sum += A_d[row * S + i] * B_d[i * S + col];
        }

        C_d[row * N + col] = sum;
    }
}

cudaError_t matrix_multi(const double *A_h, const double *B_h, double *C_h, int M, int N, int S)
{
    double *A_d, *B_d, *C_d;
    clock_t start = clock();
    double elapsed = 0;
    cudaError_t err = cudaSuccess;

    dim3 dimGrid = {(unsigned int)ceil(N / 128.0), (unsigned int)ceil(M / 128.0), 1};
    dim3 dimBlock = {128, 128, 1};

    dim3 dimGrid1 = {(unsigned int)ceil(N / 128.0), 1, 1};
    dim3 dimBlock1 = {128, 1, 1};

    dim3 dimGrid2 = {1, (unsigned int)ceil(M / 128.0), 1};
    dim3 dimBlock2 = {1, 128, 1};

    err = cudaMalloc((void **)&A_d, M * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void **)&B_d, N * S * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMalloc((void **)&C_d, M * N * sizeof(double));
    CUDA_CHECK(err, "Can't cudaMalloc");

    err = cudaMemcpy(A_d, A_h, M * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");
    err = cudaMemcpy(B_d, B_h, N * S * sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "Can't cudaMemcpy");

    // compare the time consume

    start = clock();
    matrix_multi_kernel KERNEL_ARGS2(dimGrid, dimBlock)(A_d, B_d, C_d, M, N, S);
    elapsed = 1000 * (double)(clock() - start) / CLOCKS_PER_SEC; // in milliseconds
    printf("normal matrix_multi: %lf ms\n", elapsed);

    err = cudaMemcpy(C_h, C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "Can't cudaMemcpy");

Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return err;
}