#include "cuda_runtime_api.h"
#include "driver_types.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <iostream>
#include <random>
#include <string>

const int M = 4, N = 4, S = 4;

// Matrix multiplication cuda kernel, include from matrix_multi_kernel.cu
__global__ void matrix_multi_kernel(double *A_d, double *B_d, double *C_d, int M, int N, int S);
cudaError_t matrix_multi(const double *A_h, const double *B_h, double *C_h, int M, int N, int S);

template <typename T> std::string matrix2str(T *mat, int y, int x)
{
    std::string s;
    s += "[";
    for (int i = 0; i != y; i++)
    {
        s += "[";
        for (int j = 0; j != x; j++)
        {
            if (j != 0)
                s += ", ";
            s += std::to_string(mat[j + i * x]);
        }

        s += "]";
    }

    s += "]";

    return s;
}

int main()
{
    double A_h[M][S], B_h[S][N], C_h[M][N];

    srand(time(NULL));

    for (int i = 0; i != M; i++)
    {
        for (int j = 0; j != S; j++)
        {
            A_h[i][j] = rand() % 100 / 10.0;
        }
    }

    for (int i = 0; i != S; i++)
    {
        for (int j = 0; j != M; j++)
        {
            B_h[i][j] = rand() % 100 / 10.0;
        }
    }

    for (int i = 0; i != M; i++)
    {
        for (int j = 0; j != N; j++)
        {
            C_h[i][j] = 0.0;
        }
    }

    matrix_multi((const double *)A_h, (const double *)B_h, (double *)C_h, M, N, S);

    std::cout << matrix2str((double *)A_h, M, S) << "*" << matrix2str((double *)B_h, S, N) << "="
              << matrix2str((double *)C_h, M, N) << std::endl;

    return 0;
}
