#include "ece408.hpp"
#include "fill_matrix.hpp"
#include <gtest/gtest.h>

TEST(matrix_multi_tile_simple, correctness_of_square__multi_eye)
{
    int M = 16;
    double *matA_h = new double[M * M];
    double *matB_h = new double[M * M];
    double *matC_h = new double[M * M];

    fill_randomly(matA_h, M, M);
    fill_eye(matB_h, M, M);

    matrix_multi_tile_simple((const double *)matA_h, (const double *)matB_h, (double *)matC_h, M);

    ASSERT_TRUE(matrix_same(matA_h, matC_h, M, M));

    fill_randomly(matB_h, M, M);
    fill_eye(matA_h, M, M);

    matrix_multi_tile_simple((const double *)matA_h, (const double *)matB_h, (double *)matC_h, M);

    ASSERT_TRUE(matrix_same(matB_h, matC_h, M, M));

    fill_randomly(matA_h, M, M);
    fill_eye(matC_h, M, M);

    matrix_multi_tile_simple((const double *)matA_h, (const double *)matC_h, (double *)matB_h, M);

    ASSERT_TRUE(matrix_same(matA_h, matB_h, M, M));
}
