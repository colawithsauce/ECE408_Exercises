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
