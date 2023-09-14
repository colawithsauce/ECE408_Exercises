#include <cstdio>
#include <math.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"
#include "tools.hpp"

const char* err_msg = "ERROR";

namespace tools {

kerArgs
getDimBlock()
{
    kerArgs args;

    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDeviceProperties(&prop, 0);
    CUDA_CHECK(err, err_msg);

    printf("Max shared memory per block: %lu\n", prop.sharedMemPerBlock);
    printf("Max shared memory per SM: %lu\n", prop.sharedMemPerMultiprocessor);

    // We only consider the threads per SM
    args.dimGrid.x = args.dimGrid.y =
      (int)floor(sqrt(prop.maxThreadsPerMultiProcessor / 32.0));
    args.dimGrid.z = 1;

    args.dimBlock = { 32, 32, 1 };

Error:
    return args;
}
} // namespace tools
