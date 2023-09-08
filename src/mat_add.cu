#include <cuda_runtime_api.h>

#include "cuda_alias.hpp"

#define msg ""

__global__ void mat_add_kernel(const double* A_d, const double* B_d,
                               double* C_d, int M, int N) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < M && x < N) {
    C_d[y * N + x] = A_d[y * N + x] + B_d[y * N + x];
  }
}

cudaError_t mat_add(const double* A_h, const double* B_h, double* C_h, int M,
                    int N) {
  cudaError_t err = cudaSuccess;
  double *A_d, *B_d, *C_d;

  dim3 dimGrid = {(unsigned int)ceil(N / 32.0), (unsigned int)ceil(M / 32.0),
                  1};
  dim3 dimBlock = {32, 32, 1};

  err = cudaMalloc((void**)&A_d, sizeof(double) * M * N);
  CUDA_CHECK(err, msg);

  err = cudaMalloc((void**)&B_d, sizeof(double) * M * N);
  CUDA_CHECK(err, msg);

  err = cudaMalloc((void**)&C_d, sizeof(double) * M * N);
  CUDA_CHECK(err, msg);

  err = cudaMemcpy(A_d, A_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, msg);
  err = cudaMemcpy(B_d, B_h, sizeof(double) * M * N, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, msg);

  mat_add_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N);
  err = cudaGetLastError();
  CUDA_CHECK(err, "Launch kernel failed");

  err = cudaMemcpy(C_h, C_d, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
  CUDA_CHECK(err, msg);

Error:
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return err;
}
