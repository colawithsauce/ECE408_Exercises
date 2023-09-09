#include <gtest/gtest.h>

#include "ece408.hpp"
#include "fill_matrix.hpp"

TEST(test_mat_mul_coarse, correctness_huge_matrix)
{
    int M = 1024;
    double *matA_h = new double[M * M];
    double *matB_h = new double[M * M];
    double *matC_h = new double[M * M];

    fill_randomly(matA_h, M, M);
    fill_eye(matB_h, M, M);

    mat_mul_coarsening((const double *)matA_h, (const double *)matB_h, (double *)matC_h, M);

    ASSERT_TRUE(matrix_same(matA_h, matC_h, M, M));
}
