include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Threads)

enable_testing()
include(GoogleTest)

# Test matrix_same
add_executable(test_matrix_same "test_matrix_same.cpp")
target_link_libraries(test_matrix_same
    GTest::gtest_main
    Threads::Threads
    matrix_same)
gtest_discover_tests(test_matrix_same)

# Test matrix_multi_tile_simple
add_executable(test_matrix_multi_tile_simple "test_matrix_multi_tile_simple.cpp")
target_link_libraries(test_matrix_multi_tile_simple
    GTest::gtest_main
    Threads::Threads
    matrix_multi_tile_simple
    matrix_same)
gtest_discover_tests(test_matrix_multi_tile_simple)

# Test matrix_multi_tile
add_executable(test_matrix_multi_tile "test_matrix_multi_tile.cpp")
target_link_libraries(test_matrix_multi_tile
    GTest::gtest_main
    Threads::Threads
    matrix_multi_tile
    matrix_same)
gtest_discover_tests(test_matrix_multi_tile)
