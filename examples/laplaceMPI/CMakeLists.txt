cmake_minimum_required(VERSION 3.1.3)

if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new
  find_package(MPI REQUIRED)
  
  #add_executable(laplace_mpi laplace_mpi.cpp)
  #add_executable(laplace_mpi simple_mpi.cpp)
  #add_executable(laplace_mpi mpi_mesh_test.cpp)
  add_executable(laplace_mpi simple_halo.cpp)
  add_definitions(-DHAVE_MPI=1)

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

  target_link_libraries(laplace_mpi ${LINKING_LIBRARIES})
endif()
