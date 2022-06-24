cmake_minimum_required(VERSION 3.1.3)

add_executable(test_floyd floyd.cpp)


target_link_libraries(test_floyd matar)

if (KOKKOKS)
    add_executable(test_kokkos_floyd kokkos.cpp)
    target_link_libraries(test_kokkos_floyd matar Kokkos:kokkos)
endif (KOKKOS)
