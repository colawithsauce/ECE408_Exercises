#include "../matrix_print.h"
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

const int M = 100, N = 100;

// include from vector_multi.cu
__global__ void vector_multi(float *A_d, float *B_d, float *v_d, int M, int N);
cudaError_t vector_multi_launcher(float *A_h, const float *B_h, const float *v_h, int M, int N);

int main()
{
    // define the input and output data
    float A_h[M], B_h[M][N], v_h[N];

    // initialize the input data
    for (int i = 0; i != M; i++)
    {
        for (int j = 0; j != N; j++)
        {
            B_h[i][j] = (float)rand() / (float)RAND_MAX;
            v_h[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    clock_t start = clock();
    vector_multi_launcher((float *)A_h, (float *)B_h, (float *)v_h, M, N);
    float time_milliseconds = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    std::cout << "Time: " << time_milliseconds << " ms" << std::endl;

    // print the result
    /* std::cout << "The result of " << matrix2str((float *)v_h, N, 1) << " * " << matrix2str((float *)B_h, M, N) << " =
     * " */
    /*           << matrix2str((float *)A_h, M, 1) << std::endl; */

    return 0;
}
