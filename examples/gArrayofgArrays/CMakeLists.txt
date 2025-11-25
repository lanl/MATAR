cmake_minimum_required(VERSION 3.5)

find_package(Matar REQUIRED)

if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  #find_package(Kokkos REQUIRED)

  add_definitions(-DHAVE_KOKKOS=1)
  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
  endif()

  add_executable(viewofview main.cpp)
  target_link_libraries(viewofview ${LINKING_LIBRARIES})

endif()
