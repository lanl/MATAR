cmake_minimum_required(VERSION 3.1.3)
project (MATAR)
set (CMAKE_CXX_STANDARD 17)


# adds -g and -02
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)


# Modules
# Check if this is an independent build of MATAR or if it is being built as a
# submodule in an ELEMENTS build
if(CMAKE_PROJECT_NAME STREQUAL "ELEMENTS")
  # If it's being built as part of ELEMENTS, don't need to include distclean,
  # since it's already done a by ELEMENTS and it'll just cause a conflict
else()
  # Presumably, if it's not being built as part of ELEMENTS, then it's being
  # built stand-alone. So then we need to include distclean and run
  # find_package(Vector).
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
endif()
 


# Compiler flags
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${VECTOR_Fortran_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")


include_directories(src)
add_subdirectory(src)

include_directories(test)
add_subdirectory(test)
