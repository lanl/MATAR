cmake_minimum_required(VERSION 3.1.3)

project (phasefield_mpi)


if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)

  find_package(Kokkos REQUIRED)
  find_package(MPI REQUIRED)  

  add_executable(phasefield_mpi main.cpp sim_parameters.cpp heffte_fft.cpp
                 global_arrays.cpp fourier_space.cpp complex_arrays.cpp
                 system.cpp profile.cpp vtk_writer_mpi_io)

  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
    target_link_libraries(phasefield_mpi matar Kokkos::kokkos Heffte MPI::MPI_CXX)

  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
    target_link_libraries(phasefield_mpi matar Kokkos::kokkos Heffte MPI::MPI_CXX)

  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    target_link_libraries(phasefield_mpi matar Kokkos::kokkos Heffte MPI::MPI_CXX)

  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
    target_link_libraries(phasefield_mpi matar Kokkos::kokkos Heffte MPI::MPI_CXX)
  
  else()
    target_link_libraries(phasefield_mpi matar Kokkos::kokkos Heffte MPI::MPI_CXX)
  
  endif()


  if (USE_FFT_SYMMETRY)
    add_definitions(-DUSE_FFT_SYMMETRY=1)
  endif()

  include_directories(heffte)
  add_subdirectory(heffte)
endif()
