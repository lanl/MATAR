if (NOT MATAR_ENABLE_KOKKOS)
    add_executable(test_floyd floyd.cpp)
    target_link_libraries(test_floyd ${LINKING_LIBRARIES})
endif()

if (MATAR_ENABLE_KOKKOS)
    add_executable(test_kokkos_floyd kokkos_floyd.cpp)
    target_link_libraries(test_kokkos_floyd ${LINKING_LIBRARIES})
endif()
