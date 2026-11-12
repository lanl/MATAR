if (MATAR_ENABLE_KOKKOS)
  add_executable(virtual_fcn_test main.cpp classes.cpp)
  target_link_libraries(virtual_fcn_test ${LINKING_LIBRARIES})
endif()
