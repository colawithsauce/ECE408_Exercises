cmake_minimum_required(VERSION 3.20)
project(ECE408)
enable_language(CUDA)

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
    "/opt/cuda/include;/usr/local/cuda/include")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(matrix_multi_tile_simple)
add_subdirectory(matrix_multi_tile)
add_subdirectory(matrix_same)

add_subdirectory(test)
