include(GoogleTest)

if(MATAR_ENABLE_KOKKOS)
    # Find all test files in the current directory except MPI-only sources.
    file(GLOB TEST_SOURCES "test_*.cpp")
    list(REMOVE_ITEM TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_mpi_types.cpp")

    # Create a single test executable that includes all test files
    add_executable(matar_tests test_main.cpp ${TEST_SOURCES})
    target_link_libraries(matar_tests matar::matar gtest_main)
    # PRE_TEST: enumerate tests when ctest runs, not by executing the
    # binary right after linking (avoids Kokkos initialization during builds).
    gtest_discover_tests(matar_tests DISCOVERY_MODE PRE_TEST)
else()
    # Without Kokkos, only the host-type tests apply (the other test
    # sources, including test_macros.cpp, use Kokkos types/APIs directly).
    # test_precision.cpp guards its Kokkos-only parts with HAVE_KOKKOS.
    add_executable(matar_tests test_nonkokkos_types.cpp test_precision.cpp)
    target_link_libraries(matar_tests matar::matar gtest_main)
    gtest_discover_tests(matar_tests DISCOVERY_MODE PRE_TEST)
endif()

# --- MPI + Kokkos: MPICArrayKokkos / CommunicationPlan (run suite under mpirun) ---
if(MATAR_ENABLE_KOKKOS AND MATAR_ENABLE_MPI)
    add_executable(matar_mpi_tests mpi_test_main.cpp test_mpi_types.cpp)
    target_link_libraries(matar_mpi_tests PRIVATE matar::matar gtest)

    # --oversubscribe: GitHub Actions / small VMs often expose fewer slots than ranks (Open MPI).
    add_test(
        NAME MPICArrayKokkos_mpi_suite
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:matar_mpi_tests>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
