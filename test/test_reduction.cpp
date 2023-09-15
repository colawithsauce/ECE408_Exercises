#include "ece408.hpp"
#include "gtest/gtest.h"

#define SIZE 128

float
reduction(float* arr, int size)
{
    float result = 0;
    for (int i = 0; i != size; i++) {
        result += arr[i];
    }

    return result;
}

TEST(reduction, correctness)
{
    float result = 0;
    float in[SIZE];

    for (int i = 0; i != SIZE; i++) {
        in[i] = 2;
    }

    cudaError_t err = reduction_kernel_1_launcher(in, &result, SIZE);

    ASSERT_EQ(cudaSuccess, err);

    ASSERT_EQ(result, reduction(in, SIZE));
}
