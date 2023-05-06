__global__ void
mat_add_kernel(const double* A_d, const double* B_d, double* C_d, int M, int N, int SZ_shared);

cudaError_t
mat_add(const double* A_h, const double* B_h, double* C_h, int M, int N);
