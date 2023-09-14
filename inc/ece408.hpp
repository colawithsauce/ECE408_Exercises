#ifndef __ECE408__
#define __ECE408__
#include <cuda_runtime_api.h>

// Matrix Addition
cudaError_t
mat_add(const double* A_h, const double* B_h, double* C_h, int M, int N);

// Vector multiplication
cudaError_t
vector_multi_launcher(float* A_h,
                      const float* B_h,
                      const float* v_h,
                      int M,
                      int N);

// Matrix multi with tile techenique, (simplified version)
cudaError_t
matrix_multi_tile_simple(const double* A_h,
                         const double* B_h,
                         double* C_h,
                         int width);

// Matrix multi with tile techenique
cudaError_t
matrix_multi_tile(const double* A_h,
                  const double* B_h,
                  double* C_h,
                  int M,
                  int S,
                  int N);

/* mat_mul_coarsening:
 *
 *  @desc: Matrix multiplication function integated with tiled
 *  and coarsening techenology.
 *
 *  NOTE: This function works based on following assumptions for params:
 *
 *  1. A_h, B_h, C_h all matrix with shape width x width.
 *
 *  2. TILE_WIDTH is 32, and `width` whould be able to fully divide by 32.
 *
 *  */
cudaError_t
mat_mul_coarsening(const double* A_h,
                   const double* B_h,
                   double* C_h,
                   int width);

// Matrix transpose
cudaError_t
matTranspose(double* in_h, double* out_h, int width, int height);

cudaError_t
reduction_kernel_1_launcher(float* in, float* out, int size);

// Test if matrix is same
bool
matrix_same(const double* matA_h, const double* matB_h, int M, int N);

#endif // !__ECE408__
