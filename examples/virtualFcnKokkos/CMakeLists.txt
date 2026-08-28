if (MATAR_ENABLE_KOKKOS)
  add_executable(virttestkokkos child.cpp child.hpp inherited_inits.cpp inherited_inits.hpp kokkos_alias.h main_kokkos_vfcn.cpp parents.h)
  target_link_libraries(virttestkokkos ${LINKING_LIBRARIES})
endif()
