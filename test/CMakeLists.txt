cmake_minimum_required(VERSION 3.1.3)

project (mtest)

add_executable(mtest main.cpp)

if (NOT KOKKOS)
   add_executable(test_for test_for.cpp)
   target_link_libraries(test_for matar)
endif()

if (KOKKOS)
  add_executable(mtestkokkos main_kokkos.cpp)
  add_executable(test_kokkos_for kokkos_for.cpp)
  add_executable(test_dual_types test_dual_types.cpp)

  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  endif()

  target_link_libraries(mtestkokkos matar Kokkos::kokkos)
  target_link_libraries(test_kokkos_for Kokkos::kokkos)
  target_link_libraries(test_dual_types matar Kokkos::kokkos)
endif()


target_link_libraries(mtest matar)


include_directories(virtualFcnKokkos)
add_subdirectory(virtualFcnKokkos)

include_directories(laplace)
add_subdirectory(laplace)

#include_directories(phaseField/srcKokkosVerbose)
#add_subdirectory(phaseField/srcKokkosVerbose)

#include_directories(phaseField/srcMacros)
#add_subdirectory(phaseField/srcMacros)
