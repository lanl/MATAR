if (MATAR_ENABLE_KOKKOS)
  add_executable(test_rocm SomeClass.cpp main.cpp)
  target_link_libraries(test_rocm ${LINKING_LIBRARIES})
endif()
