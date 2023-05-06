#ifndef __ECE408__
#define __ECE408__
#include <cuda_runtime_api.h>

// Matrix Addition
cudaError_t
mat_add(const double* A_h, const double* B_h, double* C_h, int M, int N);

// Vector multiplication
cudaError_t vector_multi_launcher(float* A_h, const float* B_h, const float* v_h, int M, int N);

// Matrix multi with tile techenique, (simplified version)
cudaError_t
matrix_multi_tile_simple(const double* A_h, const double* B_h, double* C_h, int width);

// Matrix multi with tile techenique
cudaError_t
matrix_multi_tile(const double* A_h, const double* B_h, double* C_h, int M, int S, int N);

// Test if matrix is same
bool matrix_same(const double* matA_h, const double* matB_h, int M, int N);

#endif // !__ECE408__
