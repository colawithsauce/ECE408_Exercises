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
matrix_multi(const double* A_h, const double* B_h, double* C_h_2, int M, int N, int S);

extern cudaError_t
matrix_multi_tile(const double* A_h, const double* B_h, double* C_h_2, int M, int N, int S);

extern bool matrix_same(const double* matA_h, const double* matB_h, int M, int N);

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
        if (matA != matB)
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
    double* A_h = (double*)malloc(M * S * sizeof(double));
    double* B_h = (double*)malloc(S * N * sizeof(double));
    double* C_h_1 = (double*)malloc(M * N * sizeof(double)); // this would contains two matrix.
    double* C_h_2 = (double*)malloc(M * N * sizeof(double)); // this would contains two matrix.

    fill_randomly(A_h, M, N);
    fill_eye(B_h, M, N);

    // Check if two matrix the same, and TODO the correctness of the result.
    err = matrix_multi(A_h, B_h, C_h_1, M, N, S);
    CUDA_CHECK(err, "Matrix Multi normal");

    err = matrix_multi_tile(A_h, B_h, C_h_2, M, N, S);
    CUDA_CHECK(err, "Matrix Multi tile");

    assert(matrix_same(C_h_1, C_h_2, M, N));
    assert(matrix_same(A_h, B_h, M, N));

    assert(matrix_same_sync(C_h_1, C_h_2, M, N));
    assert(matrix_same_sync(A_h, B_h, M, N));
Error:
    return 0;
}
