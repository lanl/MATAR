cmake_minimum_required(VERSION 3.1.3)

project (mtest)

#include_directories(MATAR)

add_executable(mtest main.cpp)

if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  endif()
endif()

#add_subdirectory(MATAR)

target_link_libraries(mtest matar)
target_link_libraries(mtest Kokkos::kokkos)
