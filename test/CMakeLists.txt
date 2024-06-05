cmake_minimum_required(VERSION 3.1.3)

project (matartest)

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
