#ifndef __CUDA_ALIAS_H__
#define __CUDA_ALIAS_H__
#include <cstdio>

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream)                              \
<<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define PRINT(msg)                                                             \
    printf("ERROR: %s:\t %s:\nat : FILE %s\nat: LINE %d\n",                    \
           cudaGetErrorString(err),                                            \
           msg,                                                                \
           __FILE__,                                                           \
           __LINE__);
#define CUDA_CHECK(err, msg)                                                   \
    if (err != cudaSuccess) {                                                  \
        PRINT(msg);                                                            \
        goto Error;                                                            \
    }

#endif // !__CUDA_ALIAS_H__
