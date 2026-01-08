cmake_minimum_required(VERSION 3.8)

find_package(Matar REQUIRED)

if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new
  
  add_executable(halfspace_cooling halfspace_cooling.cpp)

  add_definitions(-DHAVE_KOKKOS=1)

  target_link_libraries(halfspace_cooling ${LINKING_LIBRARIES})
endif()
