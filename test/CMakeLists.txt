cmake_minimum_required(VERSION 3.1.3)

project (couette_flow)

if (NOT KOKKOS)
  add_executable(couette_flow_serial couette_flow.cpp)

  target_link_libraries(couette_flow matar)
endif()

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  add_executable(couette_flow couette_flow.cpp)

  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    #add_definitions(-DHAVE_CUDA=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (HIP)
    #add_definitions(-DHAVE_HIP=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_HIP)
  elseif (OPENMP)
    #add_definitions(-DHAVE_OPENMP=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_THREADS)
  endif()

  target_link_libraries(couette_flow matar Kokkos::kokkos)
endif()