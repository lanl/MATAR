cmake_minimum_required(VERSION 3.5)

find_package(Matar REQUIRED)

if (NOT KOKKOS)
  add_executable(carray_right main_carray_right.cpp)
  add_executable(carray_wrong main_carray_wrong.cpp)
  add_executable(farray_right main_farray_right.cpp)
  add_executable(farray_wrong main_farray_wrong.cpp)

  target_link_libraries(carray_right ${LINKING_LIBRARIES})
  target_link_libraries(carray_wrong ${LINKING_LIBRARIES})
  target_link_libraries(farray_right ${LINKING_LIBRARIES})
  target_link_libraries(farray_wrong ${LINKING_LIBRARIES})
endif()

if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new

  add_executable(kokkosview main_kokkosview.cpp)
  add_executable(carraykokkos_default_indexing main_carraykokkos_default_indexing.cpp)
  add_executable(carraykokkos_c_indexing main_carraykokkos_c_indexing.cpp)
  add_executable(farraykokkos_default_indexing main_farraykokkos_default_indexing.cpp)
  add_executable(farraykokkos_f_indexing main_farraykokkos_f_indexing.cpp)

  add_definitions(-DHAVE_KOKKOS=1)

  target_link_libraries(kokkosview ${LINKING_LIBRARIES})
  target_link_libraries(carraykokkos_default_indexing ${LINKING_LIBRARIES})
  target_link_libraries(carraykokkos_c_indexing ${LINKING_LIBRARIES})
  target_link_libraries(farraykokkos_default_indexing ${LINKING_LIBRARIES})
  target_link_libraries(farraykokkos_f_indexing ${LINKING_LIBRARIES})
endif()
