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


  add_executable(pointcloud-gbl pointcloud-gbl.cpp)
  add_executable(pointcloud-rk pointcloud-rk.cpp)

  target_link_libraries(pointcloud-gbl ${LINKING_LIBRARIES})
  target_link_libraries(pointcloud-rk ${LINKING_LIBRARIES})
endif(KOKKOS)
