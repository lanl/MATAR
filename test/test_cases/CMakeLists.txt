cmake_minimum_required(VERSION 3.1.3)

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
enable_testing()
add_test(NAME matar_tests COMMAND matar_tests)
