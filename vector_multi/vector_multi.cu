#include "../cuda_alias.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <stdio.h>
#include <time.h>

// Caculate A = B * v, where B is M x N, v is N x 1, and A is M x 1
__global__ void vector_multi(float *A_d, float *B_d, float *v_d, int M, int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M)
    {
        float sum = 0;
        for (unsigned int x = 0; x != N; ++x)
        {
            sum += B_d[i * N + x] * v_d[x];
        }

        A_d[i] = sum;
    }
}

// launcher of vector_multi
cudaError_t vector_multi_launcher(float *A_h, const float *B_h, const float *v_h, int M, int N)
{
    float *A_d, *B_d, *v_d;
    cudaError_t err = cudaSuccess;
    clock_t start;
    float duration;

    err = cudaMalloc((void **)&A_d, M * sizeof(float));
    CUDA_CHECK(err, "failed to malloc");

    err = cudaMalloc((void **)&B_d, M * N * sizeof(float));
    CUDA_CHECK(err, "failed to malloc");

    err = cudaMalloc((void **)&v_d, N * sizeof(float));
    CUDA_CHECK(err, "failed to malloc");

    err = cudaMemcpy(B_d, B_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "failed to memcpy");

    err = cudaMemcpy(v_d, v_h, N * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "failed to memcpy");

    start = clock();
    vector_multi<<<ceil(M / 32.0), 32>>>(A_d, B_d, v_d, M, N);
    duration = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    printf("vector_multi: %f ms\n", duration);

    err = cudaMemcpy(A_h, A_d, M * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "failed to memcpy");

Error:
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(v_d);

    return err;
}
