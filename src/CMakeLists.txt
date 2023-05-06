set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_add.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi_tile.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi_tile_simple.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_same.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_multi.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/.cpp
)

add_library(ece408 SHARED ${SRC})

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
    "/opt/cuda/include;/usr/local/cuda/include")

target_include_directories(ece408
    PUBLIC ${CMAKE_SOURCE_DIR}/inc
)
