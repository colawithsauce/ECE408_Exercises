//
// Created by colawithsauce on 9/9/23.
//

#include "ece408.hpp"
#include "matrix_print.hpp"
#include "tools.hpp"
#include "gtest/gtest.h"

TEST(test_mat_transpose, correctness)
{
    double in_h[2][3] = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    double exp[3][2] = { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } };
    double out_h[3][2] = { 10, 9, 8, 7, 6, 5 };

    matTranspose((double*)in_h, (double*)out_h, 3, 2);
    ASSERT_TRUE(matrix_same((double*)out_h, (double*)exp, 3, 2));
}

TEST(test_mat_transpose, big_matrix)
{
    double A[32][16] = {
        0.40, 0.18, 0.14, 0.88, 0.42, 0.27, 0.58, 0.56, 0.13, 0.23, 0.65, 0.31,
        0.13, 0.69, 0.81, 0.04, 0.35, 0.05, 0.17, 0.00, 0.52, 0.83, 0.45, 0.91,
        0.06, 0.52, 0.96, 0.34, 0.28, 0.87, 0.22, 0.54, 0.17, 0.94, 0.80, 0.19,
        0.14, 0.92, 0.17, 0.52, 0.59, 0.73, 0.35, 0.71, 0.05, 0.49, 0.00, 0.69,
        0.68, 0.00, 0.52, 0.08, 0.84, 0.69, 0.41, 0.52, 0.15, 0.20, 0.39, 0.12,
        0.92, 0.75, 0.76, 0.25, 0.51, 0.07, 0.56, 0.82, 0.92, 0.11, 0.62, 0.60,
        0.75, 0.96, 0.57, 0.59, 0.97, 0.49, 0.86, 0.20, 0.96, 0.93, 0.73, 0.58,
        1.00, 0.65, 0.95, 0.46, 0.20, 0.85, 0.21, 0.44, 0.54, 0.08, 0.50, 0.84,
        0.43, 0.49, 0.95, 0.72, 0.61, 0.24, 0.32, 0.71, 0.38, 0.73, 0.42, 0.74,
        0.80, 0.86, 0.55, 0.78, 0.02, 0.68, 0.67, 0.49, 0.67, 0.93, 0.66, 0.89,
        0.08, 0.79, 0.02, 0.50, 0.89, 0.17, 0.11, 0.20, 0.33, 0.82, 0.25, 0.28,
        0.18, 0.68, 0.33, 0.15, 0.04, 0.44, 0.59, 0.22, 0.52, 0.22, 0.56, 0.02,
        0.11, 0.99, 0.07, 0.81, 0.27, 0.33, 0.21, 0.65, 0.40, 0.95, 0.97, 0.56,
        0.17, 0.62, 0.04, 0.38, 0.91, 0.46, 0.79, 0.78, 0.84, 0.49, 0.29, 0.79,
        0.91, 0.97, 0.87, 0.39, 0.91, 0.65, 0.87, 0.31, 0.80, 0.66, 0.55, 0.13,
        0.21, 0.59, 0.11, 0.28, 0.08, 0.07, 0.77, 0.89, 0.99, 0.67, 0.08, 0.90,
        0.76, 0.77, 0.07, 0.07, 0.10, 0.57, 0.29, 0.98, 0.49, 0.31, 0.49, 0.41,
        0.02, 0.81, 0.94, 0.85, 0.55, 0.43, 0.24, 0.37, 0.59, 0.90, 0.98, 0.47,
        0.21, 0.54, 0.66, 0.11, 0.80, 0.88, 0.59, 0.15, 0.52, 0.01, 0.47, 0.15,
        0.43, 0.37, 0.21, 0.27, 0.77, 0.14, 0.93, 0.17, 0.96, 0.15, 0.21, 0.51,
        0.48, 0.39, 0.95, 0.64, 0.74, 0.43, 0.65, 0.14, 0.40, 0.10, 0.55, 0.37,
        0.92, 0.66, 0.34, 0.20, 0.96, 0.92, 0.91, 0.28, 0.11, 0.41, 0.87, 0.50,
        0.61, 0.74, 0.72, 0.69, 0.66, 0.76, 0.31, 0.25, 0.38, 0.98, 0.01, 0.40,
        0.46, 0.89, 0.05, 0.34, 0.64, 0.93, 0.42, 0.50, 0.22, 0.89, 0.51, 0.32,
        0.29, 0.37, 0.94, 0.37, 0.09, 0.98, 0.56, 0.69, 0.08, 0.71, 0.03, 0.88,
        0.98, 0.33, 0.83, 0.67, 0.76, 0.77, 0.50, 0.76, 0.24, 0.79, 0.90, 0.97,
        0.44, 0.39, 0.60, 0.56, 0.28, 0.36, 0.07, 0.16, 0.46, 0.16, 0.63, 0.70,
        0.76, 0.16, 0.11, 0.73, 0.17, 0.20, 0.27, 0.16, 0.35, 0.57, 0.85, 0.60,
        0.96, 0.45, 0.14, 0.85, 0.31, 0.43, 0.68, 0.07, 0.99, 0.35, 0.46, 0.29,
        0.52, 0.89, 0.91, 0.20, 0.88, 0.37, 0.98, 0.07, 0.51, 0.58, 0.68, 0.75,
        0.08, 0.76, 0.07, 0.35, 0.70, 0.66, 0.54, 0.92, 0.48, 0.11, 0.75, 0.99,
        0.75, 0.09, 0.60, 0.74, 0.66, 0.11, 0.64, 0.33, 0.28, 0.73, 0.78, 0.17,
        0.71, 0.79, 0.89, 0.30, 0.02, 0.46, 0.68, 0.33, 0.66, 0.34, 0.12, 0.95,
        0.87, 0.62, 0.40, 0.73, 0.88, 0.32, 0.53, 0.91, 0.60, 1.00, 0.42, 0.17,
        0.32, 0.98, 0.97, 0.27, 0.01, 0.23, 0.94, 0.39, 0.32, 0.15, 0.53, 0.44,
        0.15, 0.89, 0.23, 0.49, 0.60, 0.84, 0.56, 0.05, 0.82, 0.74, 0.22, 0.79,
        0.03, 0.66, 0.25, 0.69, 0.27, 0.45, 0.22, 0.66, 0.96, 0.10, 0.66, 0.99,
        0.21, 0.51, 0.50, 0.39, 0.63, 0.83, 0.95, 0.06, 0.58, 0.37, 0.97, 0.21,
        0.76, 0.01, 0.34, 0.51, 0.47, 0.52, 0.70, 0.66, 0.88, 0.03, 0.08, 0.31,
        0.07, 0.08, 0.11, 0.88, 0.57, 0.38, 0.55, 0.04, 0.03, 0.78, 0.09, 0.68,
        0.72, 0.00, 0.49, 0.45, 0.05, 0.01, 0.31, 0.10, 0.35, 0.01, 0.63, 0.06,
        0.55, 0.31, 0.39, 0.54, 0.31, 0.60, 0.63, 0.87, 0.87, 1.00, 0.57, 0.65,
        0.81, 0.51, 0.69, 0.02, 0.88, 0.77, 0.23, 0.25
    };

    double C[16][32] = {
        0.40, 0.35, 0.17, 0.68, 0.51, 0.96, 0.43, 0.02, 0.33, 0.11, 0.91, 0.80,
        0.76, 0.55, 0.52, 0.48, 0.96, 0.38, 0.29, 0.76, 0.46, 0.96, 0.88, 0.48,
        0.71, 0.88, 0.32, 0.03, 0.63, 0.88, 0.72, 0.31, 0.18, 0.05, 0.94, 0.00,
        0.07, 0.93, 0.49, 0.68, 0.82, 0.99, 0.46, 0.66, 0.77, 0.43, 0.01, 0.39,
        0.92, 0.98, 0.37, 0.77, 0.16, 0.45, 0.37, 0.11, 0.79, 0.32, 0.15, 0.66,
        0.83, 0.03, 0.00, 0.60, 0.14, 0.17, 0.80, 0.52, 0.56, 0.73, 0.95, 0.67,
        0.25, 0.07, 0.79, 0.55, 0.07, 0.24, 0.47, 0.95, 0.91, 0.01, 0.94, 0.50,
        0.63, 0.14, 0.98, 0.75, 0.89, 0.53, 0.53, 0.25, 0.95, 0.08, 0.49, 0.63,
        0.88, 0.00, 0.19, 0.08, 0.82, 0.58, 0.72, 0.49, 0.28, 0.81, 0.78, 0.13,
        0.07, 0.37, 0.15, 0.64, 0.28, 0.40, 0.37, 0.76, 0.70, 0.85, 0.07, 0.99,
        0.30, 0.91, 0.44, 0.69, 0.06, 0.31, 0.45, 0.87, 0.42, 0.52, 0.14, 0.84,
        0.92, 1.00, 0.61, 0.67, 0.18, 0.27, 0.84, 0.21, 0.10, 0.59, 0.43, 0.74,
        0.11, 0.46, 0.09, 0.24, 0.76, 0.31, 0.51, 0.75, 0.02, 0.60, 0.15, 0.27,
        0.58, 0.07, 0.05, 0.87, 0.27, 0.83, 0.92, 0.69, 0.11, 0.65, 0.24, 0.93,
        0.68, 0.33, 0.49, 0.59, 0.57, 0.90, 0.37, 0.43, 0.41, 0.89, 0.98, 0.79,
        0.16, 0.43, 0.58, 0.09, 0.46, 1.00, 0.89, 0.45, 0.37, 0.08, 0.01, 1.00,
        0.58, 0.45, 0.17, 0.41, 0.62, 0.95, 0.32, 0.66, 0.33, 0.21, 0.29, 0.11,
        0.29, 0.98, 0.21, 0.65, 0.87, 0.05, 0.56, 0.90, 0.11, 0.68, 0.68, 0.60,
        0.68, 0.42, 0.23, 0.22, 0.97, 0.11, 0.31, 0.57, 0.56, 0.91, 0.52, 0.52,
        0.60, 0.46, 0.71, 0.89, 0.15, 0.65, 0.79, 0.28, 0.98, 0.47, 0.27, 0.14,
        0.50, 0.34, 0.69, 0.97, 0.73, 0.07, 0.75, 0.74, 0.33, 0.17, 0.49, 0.66,
        0.21, 0.88, 0.10, 0.65, 0.13, 0.06, 0.59, 0.15, 0.75, 0.20, 0.38, 0.08,
        0.04, 0.40, 0.91, 0.08, 0.49, 0.21, 0.77, 0.40, 0.61, 0.64, 0.08, 0.44,
        0.17, 0.99, 0.08, 0.66, 0.66, 0.32, 0.60, 0.96, 0.76, 0.57, 0.35, 0.81,
        0.23, 0.52, 0.73, 0.20, 0.96, 0.85, 0.73, 0.79, 0.44, 0.95, 0.97, 0.07,
        0.31, 0.54, 0.14, 0.10, 0.74, 0.93, 0.71, 0.39, 0.20, 0.35, 0.76, 0.11,
        0.34, 0.98, 0.84, 0.10, 0.01, 0.38, 0.01, 0.51, 0.65, 0.96, 0.35, 0.39,
        0.57, 0.21, 0.42, 0.02, 0.59, 0.97, 0.87, 0.77, 0.49, 0.66, 0.93, 0.55,
        0.72, 0.42, 0.03, 0.60, 0.27, 0.46, 0.07, 0.64, 0.12, 0.97, 0.56, 0.66,
        0.34, 0.55, 0.63, 0.69, 0.31, 0.34, 0.71, 0.12, 0.59, 0.44, 0.74, 0.50,
        0.22, 0.56, 0.39, 0.89, 0.41, 0.11, 0.17, 0.37, 0.69, 0.50, 0.88, 0.56,
        0.16, 0.29, 0.35, 0.33, 0.95, 0.27, 0.05, 0.99, 0.51, 0.04, 0.06, 0.02,
        0.13, 0.28, 0.05, 0.92, 0.97, 0.54, 0.80, 0.89, 0.52, 0.17, 0.91, 0.99,
        0.02, 0.80, 0.96, 0.92, 0.66, 0.22, 0.98, 0.28, 0.35, 0.52, 0.70, 0.28,
        0.87, 0.01, 0.82, 0.21, 0.47, 0.03, 0.55, 0.88, 0.69, 0.87, 0.49, 0.75,
        0.49, 0.08, 0.86, 0.17, 0.22, 0.62, 0.65, 0.67, 0.81, 0.88, 0.15, 0.66,
        0.76, 0.89, 0.33, 0.36, 0.57, 0.89, 0.66, 0.73, 0.62, 0.23, 0.74, 0.51,
        0.52, 0.78, 0.31, 0.77, 0.81, 0.22, 0.00, 0.76, 0.86, 0.50, 0.55, 0.11,
        0.56, 0.04, 0.87, 0.08, 0.94, 0.59, 0.21, 0.34, 0.31, 0.51, 0.83, 0.07,
        0.85, 0.91, 0.54, 0.78, 0.40, 0.94, 0.22, 0.50, 0.70, 0.09, 0.39, 0.23,
        0.04, 0.54, 0.69, 0.25, 0.20, 0.84, 0.78, 0.20, 0.02, 0.38, 0.31, 0.90,
        0.85, 0.15, 0.51, 0.20, 0.25, 0.32, 0.67, 0.16, 0.60, 0.20, 0.92, 0.17,
        0.73, 0.39, 0.79, 0.39, 0.66, 0.68, 0.54, 0.25
    };

    double B[16][32];

    matTranspose((double*)A, (double*)B, 16, 32);
    ASSERT_TRUE(matrix_same((double*)B, (double*)C, 16, 32));
}
