cmake_minimum_required(VERSION 3.16)
project(MATARExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add FetchContent functionality to download Kokkos
include(FetchContent)

# Try to find the FindVector module, but don't fail if not found
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../../../cmake/Modules)
include(FindVector OPTIONAL)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

if(NOT VECTOR_CXX_FLAGS)
  # Define empty flags if not found
  set(VECTOR_CXX_FLAGS "")
  set(VECTOR_CXX_VERBOSE "")
  message(STATUS "FindVector.cmake not found, continuing without vectorization flags")
endif()

# Set backend options (same as in kokkos-install.sh)
set(Kokkos_BACKEND "cuda" CACHE STRING "Kokkos backend (serial, openmp, pthreads, cuda, hip)")
set_property(CACHE Kokkos_BACKEND PROPERTY STRINGS serial openmp pthreads cuda hip)

message(STATUS "Building with Kokkos backend: ${Kokkos_BACKEND}")

# Basic Kokkos options (always enabled)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Enable native architecture optimizations")
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Build Kokkos tests")
set(BUILD_TESTING OFF CACHE BOOL "Build tests")

# Backend-specific options
if(Kokkos_BACKEND STREQUAL "cuda")
    # Find CUDA
    find_package(CUDAToolkit REQUIRED)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA Lambda support")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Enable CUDA Constexpr support")
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable CUDA RDC")
elseif(Kokkos_BACKEND STREQUAL "hip")
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable HIP backend")
    set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable HIP RDC")
    set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE)
elseif(Kokkos_BACKEND STREQUAL "openmp")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend")
elseif(Kokkos_BACKEND STREQUAL "pthreads")
    set(Kokkos_ENABLE_THREADS ON CACHE BOOL "Enable Threads backend")
elseif(Kokkos_BACKEND STREQUAL "serial")
    # Serial is already enabled by default
    message(STATUS "Using Serial backend only")
else()
    message(FATAL_ERROR "Invalid Kokkos_BACKEND value: ${Kokkos_BACKEND}")
endif()

# Additional debug build option
option(DEBUG_BUILD "Enable debug build" OFF)
if(DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "Debug build enabled")
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Download and configure Kokkos
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master # You can specify a specific tag/release here
)

# Make Kokkos available
FetchContent_MakeAvailable(kokkos)

# Define that we're using Kokkos
add_definitions(-DHAVE_KOKKOS=1)


set(MATAR_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../src/include)
include_directories(${MATAR_INCLUDE_DIR})


message(STATUS "CMAKE_SOURCE_DIR absolute path: ${CMAKE_SOURCE_DIR}")
message(STATUS "SCRIPT_DIR absolute path: ${SCRIPT_DIR}")
message(STATUS "Primary MATAR include path: ${SCRIPT_DIR}/../MATAR")

# Set compiler flags for vectorization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_CXX_FLAGS} -O3 ") #-O3 -fstrict-aliasing -march=native -mtune=native -ftree-vectorize -mprefer-vector-width=512 -fno-trapping-math -fno-math-errno -fopenmp-simd

# Option for verbose vectorization output
option(VECTOR_VERBOSE "Enable verbose vectorization output" OFF)
if(VECTOR_VERBOSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_CXX_VERBOSE}")
    message(STATUS "Enabling vectorization verbose output")
endif()

# Display final compiler flags
message(STATUS "Final CXX compiler flags: ${CMAKE_CXX_FLAGS}")

# Create the executable
add_executable(hydro hydro.cpp)
target_link_libraries(hydro Kokkos::kokkos)