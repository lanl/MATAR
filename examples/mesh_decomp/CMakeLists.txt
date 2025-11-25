cmake_minimum_required(VERSION 3.5)

# Find MPI
find_package(MPI REQUIRED)
add_definitions(-DHAVE_MPI=1)

find_package(Matar REQUIRED)

execute_process(
  COMMAND ${CMAKE_CURRENT_LIST_DIR}/install_ptscotch.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE INSTALL_PTSCOTCH_RESULT
)

if(NOT INSTALL_PTSCOTCH_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to install PT-Scotch by running install_ptscotch.sh")
endif()


if (KOKKOS)
  #find_package(Kokkos REQUIRED) #new
  
  add_executable(mesh_decomp mesh_decomp.cpp)

  add_definitions(-DHAVE_KOKKOS=1)

  # Add include directories for MPI and Scotch/PT-Scotch
  target_include_directories(mesh_decomp PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_LIST_DIR}/lib/scotch/build/src/include)
  
  # Link libraries - order matters! libptscotch depends on libscotch
  # Use -Wl,--whole-archive to ensure all symbols are included from static libraries
  # Note: Only link libptscotcherr.a (not libscotcherr.a) to avoid multiple definitions
  target_link_libraries(mesh_decomp ${LINKING_LIBRARIES} MPI::MPI_CXX 
    -Wl,--whole-archive
    ${CMAKE_CURRENT_LIST_DIR}/lib/scotch/build/lib/libscotch.a
    -Wl,--no-whole-archive
    -Wl,--whole-archive
    ${CMAKE_CURRENT_LIST_DIR}/lib/scotch/build/lib/libptscotcherr.a
    ${CMAKE_CURRENT_LIST_DIR}/lib/scotch/build/lib/libptscotch.a
    -Wl,--no-whole-archive
    -lz     # zlib for gzip compression
    -lbz2   # bzip2 library
    -llzma  # xz compression library
  )
endif()
