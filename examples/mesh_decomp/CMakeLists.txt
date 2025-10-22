cmake_minimum_required(VERSION 3.1.3)

find_package(Matar REQUIRED)

if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new
  
  add_executable(mech_decomp mesh_decomp.cpp)

  add_definitions(-DHAVE_KOKKOS=1)

  target_link_libraries(mesh_decomp ${LINKING_LIBRARIES})
endif()
