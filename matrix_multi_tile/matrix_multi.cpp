#include "cuda_runtime_api.h"
#include "driver_types.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <iostream>
#include <random>
#include <string>

// Matrix multiplication cuda kernel, include from matrix_multi_kernel.cu
__global__ void matrix_multi_kernel(double* A_d, double* B_d, double* C_d, int M, int N, int S);
cudaError_t matrix_multi(const double* A_h, const double* B_h, double* C_h, int M, int N, int S);

template <typename T>
std::string matrix2str(T* mat, int y, int x)
{
    std::string s;
    s += "[";
    for (int i = 0; i != y; i++) {
        s += "[";
        for (int j = 0; j != x; j++) {
            if (j != 0)
                s += ", ";
            s += std::to_string(mat[j + i * x]);
        }

        s += "]";
    }

    s += "]";

    return s;
}

int main(int argc, char** argv)
{
    int M, N, S;
    int x = (argc > 1) ? atoi(argv[1]) : 1024;
    N = S = M = x;

    double *A_h = nullptr, *B_h = nullptr, *C_h = nullptr;
    A_h = new double[M * S];
    B_h = new double[S * N];
    C_h = new double[M * N];

    srand(time(NULL));

    for (int i = 0; i != M; i++) {
        for (int j = 0; j != S; j++) {
            A_h[i * S + j] = rand() % 100 / 10.0;
        }
    }

    for (int i = 0; i != S; i++) {
        for (int j = 0; j != N; j++) {
            B_h[i * N + j] = rand() % 100 / 10.0;
        }
    }

    for (int i = 0; i != M; i++) {
        for (int j = 0; j != N; j++) {
            C_h[i * N + j] = 0.0;
        }
    }

    clock_t start = clock();
    matrix_multi((const double*)A_h, (const double*)B_h, (double*)C_h, M, N, S);
    float duration = (float)(clock() - start) / CLOCKS_PER_SEC;

    std::cout << "Matrix Multi: " << duration << " s" << std::endl;

    /* std::cout << matrix2str((double *)A_h, M, S) << "*" << matrix2str((double *)B_h, S, N) << "=" */
    /*           << matrix2str((double *)C_h, M, N) << std::endl; */

    return 0;
}
