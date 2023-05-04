set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/matrix_multi_tile.cu)
add_library(matrix_multi_tile SHARED ${SRC})

# target_include_directories(matrix_multi_tile
#     PUBLIC "${CMAKE_SOURCE_DIR}/include")
#
# target_link_libraries(matrix_multi_tile
#     PUBLIC cuda)
