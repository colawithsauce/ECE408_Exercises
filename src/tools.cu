#include <cstdio>
#include <math.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <device_types.h>

#include "cuda_alias.hpp"
#include "tools.hpp"

const char* err_msg = "ERROR";

namespace tools {

kerArgs getDimBlock()
{
    kerArgs args;

    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDeviceProperties(&prop, 0);
    CUDA_CHECK(err, err_msg);

    printf("Memory Bus Width:\t%d\tMax threads per SM: %d\n", prop.memoryBusWidth, prop.maxThreadsPerMultiProcessor);
    printf("Max block per SM:\t%d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max thread per Block:\t%d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per Block:%lu\n", prop.sharedMemPerBlock);
    printf("SM count:\t\t%d\n", prop.multiProcessorCount);
    printf("Max thread dimension:\t{%d, %d, %d}\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size:\t\t{%d, %d, %d}\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // We only consider the threads per SM
    args.dimGrid.x = args.dimGrid.y = (int)floor(sqrt(prop.maxThreadsPerMultiProcessor / 32.0));
    args.dimGrid.z = 1;

    args.dimBlock = { 32, 32, 1 };

Error:
    return args;
}
}
