find_package(GTest)
find_package(Threads)
enable_testing()
include(GoogleTest)

# Test matrix_multi_tile
add_executable(matrix_mult_tile "test_matrix_multi_tile.cpp")
target_link_libraries(matrix_mult_tile GTest::gtest_main Threads::Threads)
gtest_discover_tests(matrix_mult_tile)

# Test matrix_same
add_executable(matrix_same "test_matrix_same.cpp")
target_link_libraries(matrix_same GTest::gtest_main Threads::Threads)
gtest_discover_tests(matrix_same)
