cmake_minimum_required(VERSION 3.8)

# --- custom targets: ---

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT TARGET distclean)
  # Only include distclean if it has not already been defined (by any
  # other package that defines distclean and uses MATAR as a submodule)
  INCLUDE(../cmake/Modules/TargetDistclean.cmake OPTIONAL)
endif (NOT TARGET distclean)

set(LINKING_LIBRARIES matar)

find_package(Matar REQUIRED)
if (MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DHAVE_MPI=1)
    list(APPEND LINKING_LIBRARIES MPI::MPI_CXX)
    if (GPU_AWARE_MPI)
        add_definitions(-DHAVE_GPU_AWARE_MPI)
#    elseif (GPU_SHARED_MEM)
#        add_definitions(-DHAVE_GPU_SHARED_MEM)
    endif()
endif()

if (NOT KOKKOS)
    add_executable(mtest main.cpp)
    target_link_libraries(mtest ${LINKING_LIBRARIES})

   add_executable(test_for test_for.cpp)
   target_link_libraries(test_for ${LINKING_LIBRARIES})

   add_executable(test_shared_ptr test_shared_ptr.cpp)
   target_link_libraries(test_shared_ptr ${LINKING_LIBRARIES})
endif()

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  list(APPEND LINKING_LIBRARIES Kokkos::kokkos)

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

  add_executable(mtestkokkos main_kokkos.cpp)
  target_link_libraries(mtestkokkos ${LINKING_LIBRARIES})

  add_executable(test_kokkos_for kokkos_for.cpp)
  target_link_libraries(test_kokkos_for ${LINKING_LIBRARIES})

  add_executable(test_dual_types test_dual_types.cpp)
  target_link_libraries(test_dual_types ${LINKING_LIBRARIES})

  add_executable(kokkos_csr CSRKokkos.cpp)
  target_link_libraries(kokkos_csr ${LINKING_LIBRARIES})

  add_executable(kokkos_csc CSCKokkos.cpp)
  target_link_libraries(kokkos_csc ${LINKING_LIBRARIES})

  add_executable(mtr_kokkos-simple mtr-kokkos-simple.cpp)
  target_link_libraries(mtr_kokkos-simple ${LINKING_LIBRARIES})

  if (OPENMP)
    add_executable(parallel_hello_world parallel_hello_world.cpp)
    target_link_libraries(parallel_hello_world ${LINKING_LIBRARIES})
  endif()

endif()

### HIP Linking error, will add back in after fixed
if (NOT HIP)
    include_directories(virtualFcnKokkos)
    add_subdirectory(virtualFcnKokkos)
endif()

# In testing, not working
#include_directories(gArrayofgArrays)
#add_subdirectory(gArrayofgArrays)

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
