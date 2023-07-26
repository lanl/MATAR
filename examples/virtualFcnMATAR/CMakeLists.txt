cmake_minimum_required(VERSION 3.1.3)


if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  find_package(Kokkos REQUIRED)

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

  add_executable(virtual_fcn_test main.cpp classes.cpp)
  target_link_libraries(virtual_fcn_test Kokkos::kokkos)

endif()
