cmake_minimum_required(VERSION 3.12)
#set (CMAKE_CXX_STANDARD 17)

if (KOKKOS)
  if (CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "")
  elseif (OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
  endif ()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
endif ()


if (KOKKOS)
    include_directories(Kokkos)
    add_subdirectory(Kokkos/kokkos)
    get_property(KOKKOS_INCLUDE_DIRS DIRECTORY Kokkos/kokkos/core/src PROPERTY INCLUDE_DIRECTORIES)
    include_directories(${KOKKOS_INCLUDE_DIRS})
endif()

#add_library(matar INTERFACE)

#target_sources(matar INTERFACE matar.h)

#if (KOKKOS)
#    target_include_directories(matar INTERFACE Kokkos)
#endif()


add_library(matar matar.cpp matar.h)

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

  #target_link_libraries(matar INTERFACE Kokkos::kokkos)
  target_link_libraries(matar  Kokkos::kokkos)
endif()

