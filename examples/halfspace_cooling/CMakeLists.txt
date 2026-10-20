if (MATAR_ENABLE_KOKKOS)
  add_executable(halfspace_cooling halfspace_cooling.cpp)
  target_link_libraries(halfspace_cooling ${LINKING_LIBRARIES})
endif()
