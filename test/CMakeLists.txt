cmake_minimum_required(VERSION 3.5)

project (matartest)

# Required at top level so CMake generates ${CMAKE_BINARY_DIR}/CTestTestfile.cmake.
# Without this, ctest from the build root finds no tests (enable_testing in a subdirectory is not enough).
enable_testing()

# MATAR built with MPI lists MPI::MPI_CXX on imported target `matar`; MatarConfig loads targets during
# find_package(Matar). Only pre-find MPI for those test builds (see cmake_build_test.sh).
option(MATAR_TEST_USE_MPI "Configure tests against MPI-enabled MATAR (requires MPI before find_package)" OFF)
if(MATAR_TEST_USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()
find_package(Matar REQUIRED)

#-------------------------------------------
# Testing stuff
#-------------------------------------------
if (NOT KOKKOS)
  add_subdirectory(googletest)
  add_subdirectory(test_cases)
  add_executable(test_csr CSRArray.cpp)
  target_link_libraries(test_csr matar gtest_main)
  add_executable(test_csc CSCArray.cpp)
  target_link_libraries(test_csc matar gtest_main)
endif()

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new
  add_definitions(-DHAVE_KOKKOS=1)

  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
  endif()

  add_subdirectory(googletest)
  add_subdirectory(test_cases)
  add_executable(test_csr CSRArray.cpp)
  target_link_libraries(test_csr matar gtest_main)
  add_executable(test_csc CSCArray.cpp)
  target_link_libraries(test_csc matar gtest_main)


endif()
