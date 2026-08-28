set(LINKING_LIBRARIES matar::matar)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../solvers)

if (NOT MATAR_ENABLE_KOKKOS)
    add_executable(mtest main.cpp)
    target_link_libraries(mtest ${LINKING_LIBRARIES})

    add_executable(test_for test_for.cpp)
    target_link_libraries(test_for ${LINKING_LIBRARIES})

    add_executable(test_shared_ptr test_shared_ptr.cpp)
    target_link_libraries(test_shared_ptr ${LINKING_LIBRARIES})
endif()

if (MATAR_ENABLE_KOKKOS)
  add_executable(testsetval test_set_values.cpp)
  target_link_libraries(testsetval ${LINKING_LIBRARIES})

  add_executable(mtestkokkos main_kokkos.cpp)
  target_link_libraries(mtestkokkos ${LINKING_LIBRARIES})

  add_executable(drrak_test test_drrak.cpp)
  target_link_libraries(drrak_test ${LINKING_LIBRARIES})

  add_executable(test_kokkos_for kokkos_for.cpp)
  target_link_libraries(test_kokkos_for ${LINKING_LIBRARIES})

  add_executable(test_dual_types test_dual_types.cpp)
  target_link_libraries(test_dual_types ${LINKING_LIBRARIES})

  add_executable(kokkos_csr CSRKokkos.cpp)
  target_link_libraries(kokkos_csr ${LINKING_LIBRARIES})

  add_executable(kokkos_csc CSCKokkos.cpp)
  target_link_libraries(kokkos_csc ${LINKING_LIBRARIES})

  add_executable(mtr_kokkos-simple mtr-kokkos-simple.cpp)
  target_link_libraries(mtr_kokkos-simple ${LINKING_LIBRARIES})

  add_executable(annkokkos ann_kokkos.cpp)
  target_link_libraries(annkokkos ${LINKING_LIBRARIES})

  add_executable(annkokkos_compare ann_kokkos_compare.cpp)
  target_link_libraries(annkokkos_compare ${LINKING_LIBRARIES})

  add_executable(lu_test test_lu_solve.cpp)
  target_link_libraries(lu_test ${LINKING_LIBRARIES})

  add_executable(qr_test test_qr_solve.cpp)
  target_link_libraries(qr_test ${LINKING_LIBRARIES})

  add_executable(cramers_rule_test test_cramers_rule.cpp)
  target_link_libraries(cramers_rule_test ${LINKING_LIBRARIES})

  add_subdirectory(pointcloud)

  if (Kokkos_ENABLE_OPENMP)
    add_executable(parallel_hello_world parallel_hello_world.cpp)
    target_link_libraries(parallel_hello_world ${LINKING_LIBRARIES})
  endif()

  if (MATAR_ENABLE_MPI)
    add_subdirectory(laplaceMPI)
    add_subdirectory(mesh_decomp)

    add_executable(matar_mpi matar_mpi.cpp)
    target_link_libraries(matar_mpi ${LINKING_LIBRARIES})
  endif()
endif()

### HIP Linking error, will add back in after fixed
if (NOT Kokkos_ENABLE_HIP)
    add_subdirectory(virtualFcnKokkos)
endif()

# In testing, not working
#add_subdirectory(gArrayofgArrays)

add_subdirectory(virtualFcnMATAR)
add_subdirectory(laplace)
add_subdirectory(halfspace_cooling)
add_subdirectory(watt-graph)
#add_subdirectory(matar_fortran)
add_subdirectory(sparsetests)
add_subdirectory(test_rocm)

# add_subdirectory(phaseField/srcKokkosVerbose)
# add_subdirectory(phaseField/srcMacros)
# add_subdirectory(phaseFieldMPI)
