cmake_minimum_required(VERSION 3.5)

# Find all test files in the current directory except test_main.cpp and MPI-only sources.
file(GLOB TEST_SOURCES "test_*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_mpi_types.cpp")

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

# --- MPI + Kokkos: MPICArrayKokkos / CommunicationPlan (run suite under mpirun) ---
if(KOKKOS)
    find_package(MPI COMPONENTS CXX QUIET)
endif()

if(KOKKOS AND MPI_CXX_FOUND)
    add_executable(matar_mpi_tests mpi_test_main.cpp test_mpi_types.cpp)
    target_link_libraries(matar_mpi_tests PRIVATE matar gtest Kokkos::kokkos MPI::MPI_CXX)
    target_compile_definitions(matar_mpi_tests PRIVATE HAVE_MPI=1 HAVE_KOKKOS=1)

    # Run all MPI tests with multiple ranks (single CTest entry; avoids running shards without MPI).
    add_test(
        NAME MPICArrayKokkos_mpi_suite
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:matar_mpi_tests>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
