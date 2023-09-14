set(SRC
    mat_add.cu
    matrix_multi.cu
    matrix_multi_tile.cu
    matrix_multi_tile_simple.cu
    matrix_same.cu
    vector_multi.cu
    mat_mul_coarse.cu
    convolution.cu
    stencil_kernel.cu
    mat_transpose.cu
    reduction.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/.cpp
)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
    "/opt/cuda/include;/usr/local/cuda/include")

add_library(ece408 SHARED ${SRC})
target_include_directories(ece408 PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(ece408 PRIVATE tools)

add_library(tools SHARED ${CMAKE_CURRENT_SOURCE_DIR}/tools.cu)
target_link_libraries(tools PRIVATE cuda)
target_include_directories(
  tools PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
               ${CMAKE_SOURCE_DIR}/inc)
target_link_directories(tools PRIVATE cuda)
