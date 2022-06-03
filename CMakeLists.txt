# Usage of C++17 standard requires CMake version >= 3.8
cmake_minimum_required(VERSION 3.13)

project (MATAR)

# Current usage of shared_ptr in MATAR requires C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


# CMAKE_BUILD_TYPE:
# 1. Release: `-O3 -DNDEBUG`
# 2. Debug: `-O0 -g`
# 3. RelWithDebInfo: `-O2 -g -DNDEBUG`
# 4. MinSizeRel: `-Os -DNDEBUG`
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)


# Macros and packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# --- custom targets: ---
if (NOT TARGET distclean)
  # Only include distclean if it has not already been defined (by ELEMENTS or
  # any other package that defines distclean and uses MATAR as a submodule)
  INCLUDE(cmake/Modules/TargetDistclean.cmake OPTIONAL)
endif (NOT TARGET distclean)


# Add the matar library and setup targets, aliases etc

add_library(matar)

# Alias - this allows other projects that use Pkg as a subproject to
# find_package(Nmspc::Pkg) which does nothing because Pkg is already
# part of the project

add_library(matar::matar ALIAS matar)
set(MATAR_LIBRARIES matar::matar CACHE STRING "MATAR library target")


# Find our modules first
if (CMAKE_VERSIN GREATER_EQUAL 3.15)
  list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")
else ()
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules;$CMAKE_MODULE_PATH}")
endif ()


# Find vectorization flags
find_package(Vector)
if (CMAKE_VECTOR_NOVEC)
  set(VECTOR_C_FLAGS "${VECTOR_NOVEC_C_FLAGS}")
  set(VECTOR_CXX_FLAGS "${VECTOR_NOVEC_CXX_FLAGS}")
endif (CMAKE_VECTOR_NOVEC)
if (CMAKE_VECTOR_VERBOSE)
  set(VECTOR_C_FLAGS "${VECTOR_C_FLAGS} ${VECTOR_C_VERBOSE}")
  set(VECTOR_CXX_FLAGS "${VECTOR_CXX_FLAGS} ${VECTOR_CXX_VERBOSE}")
  set(VECTOR_Fortran_FLAGS "${VECTOR_Fortran_FLAGS} ${VECTOR_Fortran_VERBOSE}")
endif (CMAKE_VECTOR_VERBOSE)

# Add vectorization flags to compiler flags
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${VECTOR_Fortran_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")




# Find Kokkos and lower level packages depending on enabled options
set(MATAR_ENABLE_KOKKOS False CACHE BOOL "Is Kokkos enabled for MATAR?")
set(MATAR_ENABLE_KOKKOS_CUDA False CACHE BOOL "Is CUDA backend enabled for Kokkos?")
set(MATAR_ENABLE_KOKKOS_HIP False CACHE BOOL "Is HIP backend enabled for Kokkos?")
set(MATAR_ENABLE_KOKKOS_OPENMP False CACHE BOOL "Is OPENMP backend enabled for Kokkos?")
set(MATAR_ENABLE_KOKKOS_THREADS False CACHE BOOL "Is THREADS backend enabled for Kokkos?")
if (MATAR_ENABLE_KOKKOS)
  find_package(Kokkos REQUIRED) #new

  set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (MATAR_ENABLE_KOKKOS_CUDA)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (MATAR_ENABLE_KOKKOS_HIP)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_HIP)
  elseif (MATAR_ENABLE_KOKKOS_OPENMP)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  elseif (MATAR_ENABLE_KOKKOS_THREADS)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_THREADS)
  endif()

  target_link_libraries(matar PUBLIC Kokkos::kokkos)
endif()


# Process code directories

add_subdirectory(src)


set(ENABLE_UNIT_TESTS False CACHE BOOL "build and run MATAR tests")
if (ENABLE_UNIT_TESTS)
  add_subdirectory(test)
endif()



# MATAR targets for exporting

install(TARGETS matar
  EXPORT matar_LIBRARIES
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
  INCLUDES DESTINATION include
  )

# Write configuration files to be used by projects importing Wonton
# Prepare configuration file from template replacing only variables enclosed
# by the @ sign.

configure_file(${PROJECT_SOURCE_DIR}/cmake/matarConfig.cmake.in 
  matarConfig.cmake @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/matarConfig.cmake
  DESTINATION lib/cmake/matar)


# export targets

install(EXPORT matar_LIBRARIES
  FILE matarTargets.cmake
  NAMESPACE matar::
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION lib/cmake/matar)

# install the FindVector module needed by downstream packages when processing
# matarConfig.cmake

install(FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindVector.cmake
  DESTINATION lib/cmake/matar/modules)
