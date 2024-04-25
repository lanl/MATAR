cmake_minimum_required(VERSION 3.1.3)

if (KOKKOS)
    find_package(Kokkos REQUIRED)
    add_definitions(-DHAVE_KOKKOS=1)
    
    add_executable(lin_algebra_test linear_algebra_test.cpp)
    target_link_libraries(lin_algebra_test  matar Kokkos::kokkos)

  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
  endif()

endif ()

