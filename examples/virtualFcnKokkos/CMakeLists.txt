cmake_minimum_required(VERSION 3.1.3)

project (virttestkokkos)

find_package(Matar REQUIRED)

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  add_executable(virttestkokkos child.cpp  child.hpp  inherited_inits.cpp  inherited_inits.hpp  kokkos_alias.h  main_kokkos_vfcn.cpp  parents.h)

  add_definitions(-DHAVE_KOKKOS=1)

  target_link_libraries(virttestkokkos ${LINKING_LIBRARIES})
endif()
