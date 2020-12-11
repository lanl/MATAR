#add_library(matar INTERFACE)

#target_sources(matar INTERFACE matar.h)

#target_include_directories(matar INTERFACE ./MATAR)


if (KOKKOS)
  if (CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "")
  elseif (OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
  endif ()
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
endif ()


# adds -g and -02
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# --- custom targets: ---
INCLUDE( cmake/Modules/TargetDistclean.cmake OPTIONAL)

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
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${VECTOR_Fortran_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")


if (KOKKOS)
    include_directories(Kokkos)
    add_subdirectory(Kokkos/kokkos)
    get_property(KOKKOS_INCLUDE_DIRS DIRECTORY Kokkos/kokkos/core/src PROPERTY INCLUDE_DIRECTORIES)
    include_directories(${KOKKOS_INCLUDE_DIRS})
endif()



add_library(matar matar.cpp matar.h)

if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  endif()

  #target_link_libraries(matar INTERFACE Kokkos::kokkos)
  target_link_libraries(matar  Kokkos::kokkos)
endif()

