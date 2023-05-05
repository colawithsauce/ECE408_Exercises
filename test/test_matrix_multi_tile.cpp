#include "ece408.hpp"
#include "fill_matrix.hpp"
#include <gtest/gtest.h>

TEST(matrix_multi_tile, correctness_of_square__multi_eye)
{
    int M = 16;
    double* matA_h = new double[M * M];
    double* matB_h = new double[M * M];
    double* matC_h = new double[M * M];

    fill_randomly(matA_h, M, M);
    fill_eye(matB_h, M, M);

    matrix_multi_tile((const double*)matA_h, (const double*)matB_h, (double*)matC_h, M, M, M);

    ASSERT_TRUE(matrix_same(matA_h, matC_h, M, M));
}
