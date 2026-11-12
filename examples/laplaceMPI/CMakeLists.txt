if (MATAR_ENABLE_KOKKOS AND MATAR_ENABLE_MPI)
  add_executable(laplace_mpi laplace_mpi.cpp)
  target_link_libraries(laplace_mpi ${LINKING_LIBRARIES})
endif()
