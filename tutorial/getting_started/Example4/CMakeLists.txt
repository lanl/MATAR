cmake_minimum_required(VERSION 3.21)
project(MATARExample4 LANGUAGES CXX Fortran)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Vectorization flags from the MATAR repository (optional performance boost).
# -DVECTOR_VERBOSE=ON adds the compiler's vectorization reports.
option(VECTOR_VERBOSE "Enable verbose vectorization output" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules)
find_package(Vector)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${VECTOR_Fortran_FLAGS}")
if(VECTOR_VERBOSE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_VERBOSE}")
endif()

# Pull in MATAR from the repository root. This builds the bundled Kokkos
# submodule with whatever standard Kokkos flags are passed at configure time
# (e.g. -DKokkos_ENABLE_OPENMP=ON, -DKokkos_ENABLE_CUDA=ON, ...).
# Linking matar::matar provides the headers, HAVE_KOKKOS/HAVE_* definitions,
# and the Kokkos link dependency automatically.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../.. matar)

add_executable(matar-fortran main.f90 matar_function.cpp)
target_link_libraries(matar-fortran matar::matar)
