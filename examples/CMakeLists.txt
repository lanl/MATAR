cmake_minimum_required(VERSION 3.8)

# --- custom targets: ---

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT TARGET distclean)
  # Only include distclean if it has not already been defined (by any
  # other package that defines distclean and uses MATAR as a submodule)
  INCLUDE(../cmake/Modules/TargetDistclean.cmake OPTIONAL)
endif (NOT TARGET distclean)

find_package(Matar REQUIRED)
set(LINKING_LIBRARIES matar)

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
  if (Matar_ENABLE_TRILINOS)
    find_package(Trilinos REQUIRED) #new
    # Assume if the CXX compiler exists, the rest do too.
    if (EXISTS ${Trilinos_CXX_COMPILER})
      set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
      set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
      set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})
    endif()
    if(NOT DISTRIBUTION)
      # Make sure to use same compilers and flags as Trilinos
      set(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
      set(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
      set(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")
    endif()

    message("\nFound Trilinos!  Here are the details: ")
    message("   Trilinos_DIR = ${Trilinos_DIR}")
    message("   Trilinos_VERSION = ${Trilinos_VERSION}")
    message("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
    message("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
    message("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
    message("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
    message("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
    message("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
    message("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
    message("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
    message("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
    message("End of Trilinos details\n")

    include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
    list(APPEND LINKING_LIBRARIES Trilinos::all_selected_libs)
    add_definitions(-DTRILINOS_INTERFACE=1)
  else()
    find_package(Kokkos REQUIRED) #new
    list(APPEND LINKING_LIBRARIES Kokkos::kokkos)
  endif()


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

  add_executable(testsetval test_set_values.cpp)
  target_link_libraries(testsetval ${LINKING_LIBRARIES})

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

  add_executable(annkokkos ann_kokkos.cpp)
  target_link_libraries(annkokkos ${LINKING_LIBRARIES})

  add_executable(annkokkos_compare ann_kokkos_compare.cpp)
  target_link_libraries(annkokkos_compare ${LINKING_LIBRARIES})

  add_executable(ompperftest ompperftest.cpp)
  target_link_libraries(ompperftest ${LINKING_LIBRARIES})

  if (Matar_ENABLE_TRILINOS)
    add_executable(anndistributed ann_distributed.cpp)
    target_link_libraries(anndistributed ${LINKING_LIBRARIES})
    
    add_executable(anndistributed_crs ann_distributed_crs.cpp)
    target_link_libraries(anndistributed_crs ${LINKING_LIBRARIES})

    add_executable(test_tpetra_farray test_tpetra_farray.cpp)
    target_link_libraries(test_tpetra_farray ${LINKING_LIBRARIES})

    add_executable(test_tpetra_carray test_tpetra_carray.cpp)
    target_link_libraries(test_tpetra_carray ${LINKING_LIBRARIES})

    add_executable(test_tpetra_mesh test_tpetra_mesh.cpp)
    target_link_libraries(test_tpetra_mesh ${LINKING_LIBRARIES})
  endif()

  if (OPENMP)
    add_executable(parallel_hello_world parallel_hello_world.cpp)
    target_link_libraries(parallel_hello_world ${LINKING_LIBRARIES})
  endif()

  if (MPI)
    include_directories(laplaceMPI)
    add_subdirectory(laplaceMPI)
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

#include_directories(phaseField/srcKokkosVerbose)
#add_subdirectory(phaseField/srcKokkosVerbose)

#include_directories(phaseField/srcMacros)
#add_subdirectory(phaseField/srcMacros)

#include_directories(phaseFieldMPI)
#add_subdirectory(phaseFieldMPI)
