if (MATAR_ENABLE_KOKKOS AND MATAR_ENABLE_MPI)
  include(ExternalProject)

  # Build PT-Scotch at build time (not configure time). The clone/configure/build
  # only reruns when missing, so reconfigures are fast and offline.
  set(SCOTCH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/scotch/build)
  ExternalProject_Add(ptscotch_external
    GIT_REPOSITORY https://gitlab.inria.fr/scotch/scotch.git
    GIT_SHALLOW TRUE
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/scotch/src
    BINARY_DIR ${SCOTCH_BUILD_DIR}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_PTSCOTCH=ON
      -DMPI_C_COMPILER=mpicc
      -DMPI_Fortran_COMPILER=mpifort
      -DENABLE_TESTS=OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      ${SCOTCH_BUILD_DIR}/lib/libscotch.a
      ${SCOTCH_BUILD_DIR}/lib/libptscotch.a
      ${SCOTCH_BUILD_DIR}/lib/libptscotcherr.a
  )

  add_executable(mesh_decomp mesh_decomp.cpp)
  add_dependencies(mesh_decomp ptscotch_external)

  # Generated Scotch headers land in the external build tree
  target_include_directories(mesh_decomp PRIVATE ${SCOTCH_BUILD_DIR}/src/include)

  # Link libraries - order matters! libptscotch depends on libscotch
  # Use -Wl,--whole-archive to ensure all symbols are included from static libraries
  # Note: Only link libptscotcherr.a (not libscotcherr.a) to avoid multiple definitions
  target_link_libraries(mesh_decomp ${LINKING_LIBRARIES}
    -Wl,--whole-archive
    ${SCOTCH_BUILD_DIR}/lib/libscotch.a
    -Wl,--no-whole-archive
    -Wl,--whole-archive
    ${SCOTCH_BUILD_DIR}/lib/libptscotcherr.a
    ${SCOTCH_BUILD_DIR}/lib/libptscotch.a
    -Wl,--no-whole-archive
    -lz     # zlib for gzip compression
    -lbz2   # bzip2 library
    -llzma  # xz compression library
  )
endif()
