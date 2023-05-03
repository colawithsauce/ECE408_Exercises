#include "../cuda_alias.h"
#include "../matrix_print.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <iostream>
#include <random>
#include <string>

const int SIZE = 2;
const int M = SIZE, N = SIZE, S = SIZE;

extern cudaError_t
matrix_multi_tile(const double* A_h, const double* B_h, double* C_h_2, int M, int N, int S);

template <class T>
void fill_randomly(T* mat, int M, int N)
{
    for (int i = 0; i < M * N; ++i) {
        mat[i] = rand() % 100 / 10.0;
    }
}

template <class T>
void fill_eye(T* mat, int M, int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j)
                mat[i * N + j] = 1;
            else
                mat[i * N + j] = 0;
        }
    }
}

bool matrix_same_sync(const double* matA, const double* matB, int M, int N)
{
    for (int i = 0; i < M * N; ++i) {
        if (matA[i] != matB[i])
            return false;
    }

    return true;
}

int main()
{
    cudaError_t err = cudaSuccess;
    int devCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&devCount);

    assert(devCount != 0);

    // malloc the matrix
    double* A_h = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* B_h = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* C_h = (double*)malloc(SIZE * SIZE * sizeof(double)); // this would contains two matrix.

    fill_randomly(A_h, M, N);
    fill_eye(B_h, M, N);

    PRINT_ARR(A_h);
    PRINT_ARR(B_h);

    err = matrix_multi_tile(A_h, B_h, C_h, M, N, S);
    CUDA_CHECK(err, "Matrix Multi tile");

    PRINT_ARR(C_h);

    assert(matrix_same_sync(A_h, C_h, M, N));
Error:
    return 0;
}
