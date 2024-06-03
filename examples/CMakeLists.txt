cmake_minimum_required(VERSION 3.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Matar REQUIRED)
if (MPI)
    find_package(MPI REQUIRED)
endif()

add_executable(mtest main.cpp)
target_link_libraries(mtest matar)

if (NOT KOKKOS)
   add_executable(test_for test_for.cpp)
   target_link_libraries(test_for matar)

   add_executable(test_shared_ptr test_shared_ptr.cpp)
   target_link_libraries(test_shared_ptr matar)
endif()

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  add_executable(mtestkokkos main_kokkos.cpp)
  target_link_libraries(mtestkokkos matar Kokkos::kokkos)

  add_executable(test_kokkos_for kokkos_for.cpp)
  target_link_libraries(test_kokkos_for matar Kokkos::kokkos)

  add_executable(test_dual_types test_dual_types.cpp)
  target_link_libraries(test_dual_types matar Kokkos::kokkos)

  add_definitions(-DHAVE_KOKKOS=1)

  add_executable(kokkos_csr CSRKokkos.cpp)
  target_link_libraries(kokkos_csr matar Kokkos::kokkos)

  add_executable(kokkos_csc CSCKokkos.cpp)
  target_link_libraries(kokkos_csc matar Kokkos::kokkos)

  add_executable(mtr_kokkos-simple mtr-kokkos-simple.cpp)
  target_link_libraries(mtr_kokkos-simple matar Kokkos::kokkos)


  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    add_executable(parallel_hello_world parallel_hello_world.cpp)
    target_link_libraries(parallel_hello_world matar Kokkos::kokkos)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
  endif()

endif()


### HIP Linking error, will add back in after fixed
if (NOT HIP)
    include_directories(virtualFcnKokkos)
    add_subdirectory(virtualFcnKokkos)
endif()

include_directories(virtualFcnMATAR)
add_subdirectory(virtualFcnMATAR)

include_directories(laplace)
add_subdirectory(laplace)

include_directories(halfspace_cooling)
add_subdirectory(halfspace_cooling)

include_directories(watt-graph)
add_subdirectory(watt-graph)

#include_directories(matar_fortran)
#add_subdirectory(matar_fortran)

include_directories(sparsetests)
add_subdirectory(sparsetests)

include_directories(test_rocm)
add_subdirectory(test_rocm)

include_directories(laplaceMPI)
add_subdirectory(laplaceMPI)

#include_directories(phaseField/srcKokkosVerbose)
#add_subdirectory(phaseField/srcKokkosVerbose)

#include_directories(phaseField/srcMacros)
#add_subdirectory(phaseField/srcMacros)

#include_directories(phaseFieldMPI)
#add_subdirectory(phaseFieldMPI)
