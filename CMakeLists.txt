cmake_minimum_required(VERSION 3.20)
project(ECE408)
enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(matrix_multi_tile)
# add_subdirectory(matrix_same)
#
add_subdirectory(test)
