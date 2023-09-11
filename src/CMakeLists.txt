set(SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_add.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi_tile.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi_tile_simple.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_same.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/vector_multi.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_coarse.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/stencil_kernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_transpose.cu
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
