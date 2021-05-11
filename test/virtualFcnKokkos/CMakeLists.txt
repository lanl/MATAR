cmake_minimum_required(VERSION 3.1.3)

project (virttestkokkos)

if (KOKKOS)
  add_executable(virttestkokkos child.cpp  child.hpp  inherited_inits.cpp  inherited_inits.hpp  kokkos_alias.h  main_kokkos_vfcn.cpp  parents.h)

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
  endif()

  target_link_libraries(virttestkokkos matar Kokkos::kokkos)
endif()
