cmake_minimum_required(VERSION 3.21)

project(Matar VERSION 1.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(MATAR_ENABLE_KOKKOS       "Enable the Kokkos-backed device/dual MATAR types" ON)
option(MATAR_ENABLE_MPI          "Enable the MPI-aware MATAR types (MPICArrayKokkos, CommunicationPlan)" OFF)
option(MATAR_ENABLE_GPU_AWARE_MPI "Assume the MPI implementation is GPU-aware (device buffers passed to MPI)" OFF)
option(MATAR_USE_EXTERNAL_KOKKOS "Use an already-installed Kokkos (find_package) instead of building the bundled submodule" OFF)
option(MATAR_BUILD_EXAMPLES      "Build the MATAR example programs" OFF)
option(MATAR_BUILD_TESTS         "Build the MATAR unit tests" OFF)
option(MATAR_BUILD_BENCHMARKS    "Build the MATAR benchmarks" OFF)
option(MATAR_INSTALL             "Generate install/export rules" ${PROJECT_IS_TOP_LEVEL})

# Trilinos support has been removed; the old Trilinos-based build lives in legacy/.
if(DEFINED Matar_ENABLE_TRILINOS OR DEFINED Matar_KOKKOS_PACKAGE)
    message(FATAL_ERROR
        "Trilinos support has been removed from MATAR. "
        "See legacy/CMakeLists.txt for the retired Trilinos build.")
endif()

# Map the old option names used by existing consumers (ELEMENTS, Fierro).
if(DEFINED Matar_ENABLE_KOKKOS)
    message(DEPRECATION "Matar_ENABLE_KOKKOS is deprecated; use MATAR_ENABLE_KOKKOS")
    set(MATAR_ENABLE_KOKKOS ${Matar_ENABLE_KOKKOS})
endif()
if(DEFINED Matar_ENABLE_MPI)
    message(DEPRECATION "Matar_ENABLE_MPI is deprecated; use MATAR_ENABLE_MPI")
    set(MATAR_ENABLE_MPI ${Matar_ENABLE_MPI})
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
endif()

# -----------------------------------------------------------------------------
# Vectorization flags (cmake/Modules/FindVector.cmake)
#
# Detects per-compiler auto-vectorization flags and appends them to the Release
# flags for everything compiled in this tree (examples, tests, benchmarks, and
# the bundled Kokkos). Toggles:
#   -DCMAKE_VECTOR_NOVEC=ON    same flags but with vectorization disabled
#   -DCMAKE_VECTOR_VERBOSE=ON  compiler vectorization reports
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(Vector)
if(CMAKE_VECTOR_NOVEC)
    set(VECTOR_C_FLAGS "${VECTOR_NOVEC_C_FLAGS}")
    set(VECTOR_CXX_FLAGS "${VECTOR_NOVEC_CXX_FLAGS}")
endif()
if(CMAKE_VECTOR_VERBOSE)
    set(VECTOR_C_FLAGS "${VECTOR_C_FLAGS} ${VECTOR_C_VERBOSE}")
    set(VECTOR_CXX_FLAGS "${VECTOR_CXX_FLAGS} ${VECTOR_CXX_VERBOSE}")
endif()
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${VECTOR_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${VECTOR_CXX_FLAGS}")

# -----------------------------------------------------------------------------
# The matar target (header-only)
# -----------------------------------------------------------------------------
add_library(matar INTERFACE)
add_library(matar::matar ALIAS matar)

target_compile_features(matar INTERFACE cxx_std_17)
target_include_directories(matar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
# Kokkos
#
# Backend selection is done with the standard Kokkos cache variables
# (Kokkos_ENABLE_OPENMP, Kokkos_ENABLE_CUDA, Kokkos_ARCH_*, ...), which the
# bundled submodule build consumes directly. An external Kokkos is used only
# when explicitly requested, so higher-level projects get a Kokkos built to
# match the backend they asked for.
# -----------------------------------------------------------------------------
if(MATAR_ENABLE_KOKKOS)
    if(TARGET Kokkos::kokkos)
        # A parent project already provides Kokkos; adding the submodule again
        # would define duplicate targets.
        set(MATAR_KOKKOS_PROVIDER "parent")
    elseif(MATAR_USE_EXTERNAL_KOKKOS)
        find_package(Kokkos CONFIG REQUIRED)
        set(MATAR_KOKKOS_PROVIDER "external")
    else()
        if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/Kokkos/kokkos/CMakeLists.txt)
            message(FATAL_ERROR
                "The bundled Kokkos submodule is missing. Run:\n"
                "    git submodule update --init --recursive\n"
                "or point at an installed Kokkos with "
                "-DMATAR_USE_EXTERNAL_KOKKOS=ON -DKokkos_ROOT=<prefix>")
        endif()
        add_subdirectory(src/Kokkos/kokkos)
        set(MATAR_KOKKOS_PROVIDER "bundled")
    endif()

    target_link_libraries(matar INTERFACE Kokkos::kokkos)
    target_compile_definitions(matar INTERFACE HAVE_KOKKOS=1)

    if(Kokkos_ENABLE_CUDA)
        target_compile_definitions(matar INTERFACE HAVE_CUDA=1)
    elseif(Kokkos_ENABLE_HIP)
        target_compile_definitions(matar INTERFACE HAVE_HIP=1)
    elseif(Kokkos_ENABLE_OPENMP)
        target_compile_definitions(matar INTERFACE HAVE_OPENMP=1)
    elseif(Kokkos_ENABLE_THREADS)
        target_compile_definitions(matar INTERFACE HAVE_THREADS=1)
    endif()
endif()

# -----------------------------------------------------------------------------
# MPI
# -----------------------------------------------------------------------------
if(MATAR_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(matar INTERFACE MPI::MPI_CXX)
    target_compile_definitions(matar INTERFACE HAVE_MPI=1)
    if(MATAR_ENABLE_GPU_AWARE_MPI)
        target_compile_definitions(matar INTERFACE HAVE_GPU_AWARE_MPI)
    endif()
endif()

# -----------------------------------------------------------------------------
# Install / export
#
# When the bundled Kokkos is used, its own install rules run too, so Kokkos is
# installed into the same prefix and MatarConfig's find_dependency(Kokkos)
# resolves against the sibling KokkosConfig.cmake.
# -----------------------------------------------------------------------------
if(MATAR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS matar EXPORT MatarTargets)
    install(DIRECTORY src/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    install(EXPORT MatarTargets
            NAMESPACE matar::
            FILE MatarTargets.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Matar)

    configure_package_config_file(
        cmake/MatarConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/MatarConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Matar)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/MatarConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/MatarConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/MatarConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Matar)
endif()

# -----------------------------------------------------------------------------
# Examples / tests / benchmarks
# -----------------------------------------------------------------------------
if(MATAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
if(MATAR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(MATAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
