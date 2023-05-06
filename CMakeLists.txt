cmake_minimum_required(VERSION 3.20)
project(ECE408)
enable_language(CUDA)

# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

add_subdirectory(src)
add_subdirectory(test)
