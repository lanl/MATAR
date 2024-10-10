# Usage of C++17 standard requires CMake version >= 3.8
cmake_minimum_required(VERSION 3.8)

# Current usage of shared_ptr in MATAR requires C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


project (MATAR)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CMAKE_BUILD_TYPE:
# 1. Release: `-O3 -DNDEBUG`
# 2. Debug: `-O0 -g`
# 3. RelWithDebInfo: `-O2 -g -DNDEBUG`
# 4. MinSizeRel: `-Os -DNDEBUG`
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "set default to Release")
endif(NOT CMAKE_BUILD_TYPE)


# Macros and packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

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


# Compiler flags
set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${VECTOR_Fortran_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")


add_library(matar INTERFACE)
target_include_directories(matar INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  
    $<INSTALL_INTERFACE:include>
)

if(Matar_ENABLE_KOKKOS)
    if("${Matar_KOKKOS_PACKAGE}" STREQUAL "Trilinos")
        find_package(Trilinos REQUIRED)
        add_definitions(-DTRILINOS_INTERFACE=1)
    elseif(Matar_ENABLE_TRILINOS)
        find_package(Trilinos REQUIRED)
        add_definitions(-DTRILINOS_INTERFACE=1)
    else()
        find_package(Kokkos REQUIRED)
    endif()
    if (Matar_ENABLE_MPI)
        find_package(MPI REQUIRED)
        add_definitions(-DHAVE_MPI=1)
        if(Matar_ENABLE_TRILINOS)
            target_link_libraries(matar INTERFACE Trilinos::all_selected_libs MPI::MPI_CXX)
        else()
            target_link_libraries(matar INTERFACE Kokkos::kokkos MPI::MPI_CXX)
        endif()
    else()
        target_link_libraries(matar INTERFACE Kokkos::kokkos)
    endif()
    add_definitions(-DHAVE_KOKKOS=1)
elseif(Matar_ENABLE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(matar INTERFACE MPI::MPI_CXX)
    add_definitions(-DHAVE_MPI=1)
endif()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/MatarConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS matar
    EXPORT MatarTargets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/MatarConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/MatarConfig.cmake"
    INSTALL_DESTINATION lib/cmake/matar
)

install(EXPORT MatarTargets DESTINATION lib/cmake/matar)
install(FILES "${PROJECT_BINARY_DIR}/MatarConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/MatarConfig.cmake"
        DESTINATION lib/cmake/matar)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include/ DESTINATION include)

