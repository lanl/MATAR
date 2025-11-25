cmake_minimum_required(VERSION 3.5)

# Find all test files in the current directory except test_main.cpp
file(GLOB TEST_SOURCES "test_*.cpp")

# Create a single test executable that includes all test files
add_executable(matar_tests test_main.cpp ${TEST_SOURCES})
target_link_libraries(matar_tests matar gtest_main)

# Add Kokkos if available
if(KOKKOS)
    target_link_libraries(matar_tests Kokkos::kokkos)
endif()

# Add test discovery
include(GoogleTest)
enable_testing()

gtest_discover_tests(matar_tests)
