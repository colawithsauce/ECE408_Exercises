include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Threads)

enable_testing()
include(GoogleTest)

# Set include directories directory wide
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
    "/opt/cuda/include;/usr/local/cuda/include")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    ${CMAKE_SOURCE_DIR}/inc)
link_libraries(ece408)

# Test mat_add
add_executable(test_mat_add "test_mat_add.cpp")
target_link_libraries(test_mat_add GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_mat_add)

# Test matrix_same
add_executable(test_matrix_same "test_matrix_same.cpp")
target_link_libraries(test_matrix_same GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_matrix_same)

# Test matrix_multi_tile_simple
add_executable(test_matrix_multi_tile_simple
               "test_matrix_multi_tile_simple.cpp")
target_link_libraries(test_matrix_multi_tile_simple GTest::gtest_main
                      Threads::Threads)
gtest_discover_tests(test_matrix_multi_tile_simple)

# Test matrix_multi_tile
add_executable(test_matrix_multi_tile "test_matrix_multi_tile.cpp")
target_link_libraries(test_matrix_multi_tile GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_matrix_multi_tile)

# Test mat_mul_coarse
add_executable(test_mat_mul_coarse "test_mat_mul_coarse.cpp")
target_link_libraries(test_mat_mul_coarse GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_mat_mul_coarse)
