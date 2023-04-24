#include <bits/types/clock_t.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdio>
#include <locale>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) \
<<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define PRINT(msg) \
  printf("%s:\nat : FILE %s\nat: LINE %d", msg, __FILE__, __LINE__);
#define CUDA_CHECK(err, msg) \
  if (err != cudaSuccess) {  \
    PRINT(msg);              \
    goto Error;              \
  }

__global__ void matrix_multi_kernel(double* A_d,
                                    double* B_d,
                                    double* C_d,
                                    int M,
                                    int N,
                                    int S) {
  // A_d is M x S, while B_d is S x N
  unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

  if (row < M && col < N) {
    double sum = 0;
    for (int i = 0; i != S; i++) {
      sum += A_d[row * S + i] * B_d[i * S + col];
    }

    C_d[row * N + col] = sum;
  }
}

__global__ void matrix_multi_kernel1(double* A_d,
                                     double* B_d,
                                     double* C_d,
                                     int M,
                                     int N,
                                     int S) {
  unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;

  // write a kernel that has each thread produce one output matrix row. Fill in
  // the execution configuration parameters for the disign.
  if (col < N) {
    for (unsigned int row = 0; row != M; row++) {
      int sum = 0;
      for (unsigned int i = 0; i != S; i++) {
        sum += A_d[row * S + i] * B_d[i * S + col];
      }

      C_d[row * N + col] = sum;
    }
  }
}

__global__ void matrix_multi_kernel2(double* A_d,
                                     double* B_d,
                                     double* C_d,
                                     int M,
                                     int N,
                                     int S) {
  unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;

  // write a kernel that has each thread produce one output matrix column, Fill
  // in the execution configuration parameters for the disign.
  if (row < M) {
    for (unsigned int col = 0; col != N; col++) {
      int sum = 0;
      for (unsigned int i = 0; i != S; i++) {
        sum += A_d[row * S + i] * B_d[i * S + col];
      }

      C_d[row * N + col] = sum;
    }
  }
}

cudaError_t matrix_multi(const double* A_h,
                         const double* B_h,
                         double* C_h,
                         int M,
                         int N,
                         int S) {
  double *A_d, *B_d, *C_d;
  clock_t start = clock();
  double elapsed = 0;
  cudaError_t err = cudaSuccess;

  dim3 dimGrid = {(unsigned int)ceil(N / 128.0), (unsigned int)ceil(M / 128.0),
                  1};
  dim3 dimBlock = {128, 128, 1};

  dim3 dimGrid1 = {(unsigned int)ceil(N / 128.0), 1, 1};
  dim3 dimBlock1 = {128, 1, 1};

  dim3 dimGrid2 = {1, (unsigned int)ceil(M / 128.0), 1};
  dim3 dimBlock2 = {1, 128, 1};

  err = cudaMalloc((void**)&A_d, M * S * sizeof(double));
  CUDA_CHECK(err, "Can't cudaMalloc");

  err = cudaMalloc((void**)&B_d, N * S * sizeof(double));
  CUDA_CHECK(err, "Can't cudaMalloc");

  err = cudaMalloc((void**)&C_d, M * N * sizeof(double));
  CUDA_CHECK(err, "Can't cudaMalloc");

  err = cudaMemcpy(A_d, A_h, M * S * sizeof(double), cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "Can't cudaMemcpy");
  err = cudaMemcpy(B_d, B_h, N * S * sizeof(double), cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "Can't cudaMemcpy");

  // compare the time consume

  start = clock();
  matrix_multi_kernel KERNEL_ARGS2(dimGrid, dimBlock)(A_d, B_d, C_d, M, N, S);
  elapsed =
      1000 * (double)(clock() - start) / CLOCKS_PER_SEC;  // in milliseconds
  printf("normal matrix_multi: %lf ms\n", elapsed);

  start = clock();
  matrix_multi_kernel1 KERNEL_ARGS2(dimGrid1, dimBlock1)(A_d, B_d, C_d, M, N,
                                                         S);
  elapsed =
      1000 * (double)(clock() - start) / CLOCKS_PER_SEC;  // in milliseconds
  printf("row matrix_multi1: %lf ms\n", elapsed);

  start = clock();
  matrix_multi_kernel2 KERNEL_ARGS2(dimGrid2, dimBlock2)(A_d, B_d, C_d, M, N,
                                                         S);
  elapsed =
      1000 * (double)(clock() - start) / CLOCKS_PER_SEC;  // in milliseconds
  printf("column matrix_multi2: %lf ms\n", elapsed);

  err = cudaMemcpy(C_h, C_d, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  CUDA_CHECK(err, "Can't cudaMemcpy");

Error:
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return err;
}
