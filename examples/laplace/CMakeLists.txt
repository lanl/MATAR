cmake_minimum_required(VERSION 3.1.3)

find_package(Matar REQUIRED)

if (NOT KOKKOS)
  add_executable(carray_right main_carray_right.cpp)
  add_executable(carray_wrong main_carray_wrong.cpp)
  add_executable(farray_right main_farray_right.cpp)
  add_executable(farray_wrong main_farray_wrong.cpp)

  target_link_libraries(carray_right matar)
  target_link_libraries(carray_wrong matar)
  target_link_libraries(farray_right matar)
  target_link_libraries(farray_wrong matar)
endif()

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  add_executable(kokkosview main_kokkosview.cpp)
  add_executable(carraykokkos_default_indexing main_carraykokkos_default_indexing.cpp)
  add_executable(carraykokkos_c_indexing main_carraykokkos_c_indexing.cpp)
  add_executable(farraykokkos_default_indexing main_farraykokkos_default_indexing.cpp)
  add_executable(farraykokkos_f_indexing main_farraykokkos_f_indexing.cpp)

  add_definitions(-DHAVE_KOKKOS=1)

  target_link_libraries(kokkosview matar Kokkos::kokkos)
  target_link_libraries(carraykokkos_default_indexing matar Kokkos::kokkos)
  target_link_libraries(carraykokkos_c_indexing matar Kokkos::kokkos)
  target_link_libraries(farraykokkos_default_indexing matar Kokkos::kokkos)
  target_link_libraries(farraykokkos_f_indexing matar Kokkos::kokkos)
endif()
