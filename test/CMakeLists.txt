cmake_minimum_required(VERSION 3.1.3)

project (mtest)

add_executable(mtest main.cpp)


if (KOKKOS)
  add_executable(mtestkokkos main_kokkos.cpp)
  add_executable(test_kokkos_for kokkos_for.cpp)

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
endif()


target_link_libraries(mtest matar)

include_directories(virtualFcnKokkos)
add_subdirectory(virtualFcnKokkos)

