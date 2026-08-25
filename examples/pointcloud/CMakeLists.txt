if (MATAR_ENABLE_KOKKOS)
  add_executable(pointcloud-gbl pointcloud-gbl.cpp)
  add_executable(pointcloud-rk pointcloud-rk.cpp)

  target_link_libraries(pointcloud-gbl ${LINKING_LIBRARIES})
  target_link_libraries(pointcloud-rk ${LINKING_LIBRARIES})
endif()
