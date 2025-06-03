cmake_minimum_required(VERSION 3.1.3)

# Find all test files in the current directory
file(GLOB TEST_SOURCES "*.cpp")

# Create a list of test executables by removing the .cpp extension
foreach(source ${TEST_SOURCES})
    get_filename_component(test_name ${source} NAME_WE)
    add_executable(${test_name} ${source})
    target_link_libraries(${test_name} matar gtest_main)
    
    # Add Kokkos if available
    if(KOKKOS)
        target_link_libraries(${test_name} Kokkos::kokkos)
    endif()
endforeach()

# Add test discovery
enable_testing()
foreach(source ${TEST_SOURCES})
    get_filename_component(test_name ${source} NAME_WE)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
