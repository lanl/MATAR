cmake_minimum_required(VERSION 3.1)
project(mtest CXX C)

set (CMAKE_CXX_STANDARD 17)


include_directories(./)
include_directories(../)
include_directories(../src)


add_subdirectory(../src src/matarlib)

add_executable(rtest ragged_main.cpp)

if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  if (CUDA)
      add_definitions(-DHAVE_CUDA=1)
  elseif (OPENMP)
      add_definitions(-DHAVE_OPENMP=1)
  endif()
    target_link_libraries(rtest Kokkos::kokkos)
endif ()
target_link_libraries(rtest matar)
