#include <cuda_runtime_api.h>
namespace tools {
struct kerArgs
{
    dim3 dimBlock, dimGrid;
};

kerArgs
getDimBlock();
} // namespace tools
