cmake_minimum_required(VERSION 3.18)


if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new

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


  add_executable(pointcloud pointcloud.cpp)
  target_link_libraries(pointcloud ${LINKING_LIBRARIES})
endif(KOKKOS)
