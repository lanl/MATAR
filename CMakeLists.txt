# 3.22 is the minimum required by the bundled Kokkos 5.x.
# (CUDA builds additionally need CMake >= 3.25.2 for C++20 in the CUDA language.)
cmake_minimum_required(VERSION 3.22)

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

# Kokkos 5 requires C++20; MATAR itself is C++17-clean but must match.
target_compile_features(matar INTERFACE cxx_std_20)
target_include_directories(matar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
# Floating-point precision tiers
#
# Three independently swappable tiers, fixed at configure time:
#   real_t       default working precision      (MATAR_REAL)
#   high_real_t  must-stay-accurate fields      (MATAR_HIGH_REAL)
#   low_real_t   tolerant / bulk-storage fields (MATAR_LOW_REAL)
# Values: double (default) | float | half | bfloat16 | quad.
# No-Kokkos builds support only double and float. quad (__float128) is
# host-backend-only and needs Kokkos_ENABLE_LIBQUADMATH.
# -----------------------------------------------------------------------------
set(MATAR_REAL      "double" CACHE STRING "Precision of the real_t tier")
set(MATAR_HIGH_REAL "double" CACHE STRING "Precision of the high_real_t tier")
set(MATAR_LOW_REAL  "double" CACHE STRING "Precision of the low_real_t tier")
set_property(CACHE MATAR_REAL      PROPERTY STRINGS double float half bfloat16 quad)
set_property(CACHE MATAR_HIGH_REAL PROPERTY STRINGS double float quad)
set_property(CACHE MATAR_LOW_REAL  PROPERTY STRINGS double float half bfloat16)

function(matar_precision_token tier value out_token)
    string(TOLOWER "${value}" v)
    set(map_double MATAR_FP64)
    set(map_float  MATAR_FP32)
    set(map_half   MATAR_FP16)
    set(map_bfloat16 MATAR_BF16)
    set(map_quad   MATAR_FP128)
    if(NOT DEFINED map_${v})
        message(FATAL_ERROR "MATAR_${tier}=${value} is not a recognized precision "
                            "(valid values: double, float, half, bfloat16, quad)")
    endif()
    if(NOT v MATCHES "^(double|float)$" AND NOT MATAR_ENABLE_KOKKOS)
        message(FATAL_ERROR "MATAR_${tier}=${value} requires a Kokkos build; "
                            "no-Kokkos builds support only double and float")
    endif()
    if(tier STREQUAL "HIGH_REAL" AND v MATCHES "^(half|bfloat16)$")
        message(FATAL_ERROR "MATAR_HIGH_REAL=${value}: the high-precision tier "
                            "must be double, float, or quad")
    endif()
    if(v STREQUAL "quad" AND (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL))
        message(FATAL_ERROR "MATAR_${tier}=quad: __float128 is host-only and "
                            "cannot be combined with a GPU backend")
    endif()
    if(v STREQUAL "bfloat16" AND Kokkos_ENABLE_HIP)
        message(WARNING "MATAR_${tier}=bfloat16 with the HIP backend may be "
                        "float-emulated depending on the Kokkos version "
                        "(check MATAR_BF16_IS_EMULATED at compile time)")
    endif()
    set(${out_token} ${map_${v}} PARENT_SCOPE)
endfunction()

matar_precision_token(REAL      "${MATAR_REAL}"      MATAR_REAL_TOKEN)
matar_precision_token(HIGH_REAL "${MATAR_HIGH_REAL}" MATAR_HIGH_REAL_TOKEN)
matar_precision_token(LOW_REAL  "${MATAR_LOW_REAL}"  MATAR_LOW_REAL_TOKEN)

# Emit definitions only for non-default tiers (existing consumers see
# unchanged command lines; precision.h defaults undefined tiers to double).
if(NOT MATAR_REAL STREQUAL "double")
    target_compile_definitions(matar INTERFACE MATAR_REAL_TYPE=${MATAR_REAL_TOKEN})
endif()
if(NOT MATAR_HIGH_REAL STREQUAL "double")
    target_compile_definitions(matar INTERFACE MATAR_HIGH_REAL_TYPE=${MATAR_HIGH_REAL_TOKEN})
endif()
if(NOT MATAR_LOW_REAL STREQUAL "double")
    target_compile_definitions(matar INTERFACE MATAR_LOW_REAL_TYPE=${MATAR_LOW_REAL_TOKEN})
endif()

if(MATAR_REAL STREQUAL "quad" OR MATAR_HIGH_REAL STREQUAL "quad" OR MATAR_LOW_REAL STREQUAL "quad")
    set(MATAR_ANY_QUAD ON)
    # Must be in the cache before add_subdirectory(src/Kokkos/kokkos) below
    set(Kokkos_ENABLE_LIBQUADMATH ON CACHE BOOL "Required for MATAR quad-precision tiers" FORCE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(matar INTERFACE quadmath)
    endif()
endif()

# -----------------------------------------------------------------------------
# Host / device backend selection
#
# The device backend runs the standard macros (FOR_ALL, DO_ALL, ...); the host
# backend runs the _HOST twins (FOR_ALL_HOST, ...). Choosing different
# backends for the two lets CPU work overlap in-flight GPU kernels.
#
# These are a convenience front door: leaving them empty keeps the historical
# behavior of setting Kokkos_ENABLE_* directly, which existing presets and
# downstream projects (ELEMENTS, Fierro) rely on.
# -----------------------------------------------------------------------------
set(MATAR_HOST_BACKEND   "" CACHE STRING "Host-side parallel backend: serial|openmp|pthreads")
set(MATAR_DEVICE_BACKEND "" CACHE STRING "Device-side parallel backend: serial|openmp|pthreads|cuda|hip|sycl")
set_property(CACHE MATAR_HOST_BACKEND   PROPERTY STRINGS serial openmp pthreads)
set_property(CACHE MATAR_DEVICE_BACKEND PROPERTY STRINGS serial openmp pthreads cuda hip sycl)

function(matar_backend_kokkos_var role value out_var)
    string(TOLOWER "${value}" v)
    set(map_serial   Kokkos_ENABLE_SERIAL)
    set(map_openmp   Kokkos_ENABLE_OPENMP)
    set(map_pthreads Kokkos_ENABLE_THREADS)
    set(map_cuda     Kokkos_ENABLE_CUDA)
    set(map_hip      Kokkos_ENABLE_HIP)
    set(map_sycl     Kokkos_ENABLE_SYCL)
    if(NOT DEFINED map_${v})
        message(FATAL_ERROR "MATAR_${role}_BACKEND=${value} is not a recognized backend "
                            "(valid values: serial, openmp, pthreads"
                            "${ARGN})")
    endif()
    if(role STREQUAL "HOST" AND v MATCHES "^(cuda|hip|sycl)$")
        message(FATAL_ERROR "MATAR_HOST_BACKEND=${value} is a device backend; the host "
                            "backend must be serial, openmp, or pthreads")
    endif()
    if(NOT MATAR_ENABLE_KOKKOS AND NOT v STREQUAL "serial")
        message(FATAL_ERROR "MATAR_${role}_BACKEND=${value} requires a Kokkos build "
                            "(-DMATAR_ENABLE_KOKKOS=ON)")
    endif()
    set(${out_var} ${map_${v}} PARENT_SCOPE)
endfunction()

if(MATAR_HOST_BACKEND OR MATAR_DEVICE_BACKEND)
    string(TOLOWER "${MATAR_HOST_BACKEND}" _matar_host)
    string(TOLOWER "${MATAR_DEVICE_BACKEND}" _matar_dev)

    # Kokkos permits only one non-Serial HOST-parallel space per build, so a
    # host/device pair drawn from {openmp, pthreads} cannot coexist. Catch it
    # here with an explanation rather than letting Kokkos error out later.
    if(_matar_host AND _matar_dev AND NOT _matar_host STREQUAL _matar_dev
       AND _matar_host MATCHES "^(openmp|pthreads)$" AND _matar_dev MATCHES "^(openmp|pthreads)$")
        message(FATAL_ERROR
            "MATAR_HOST_BACKEND=${_matar_host} with MATAR_DEVICE_BACKEND=${_matar_dev}: Kokkos "
            "allows only one host-parallel execution space per build, so openmp and pthreads "
            "cannot both be enabled. Pair a host backend with a GPU device backend "
            "(cuda/hip/sycl), or use the same backend for both.")
    endif()

    if(_matar_host)
        matar_backend_kokkos_var(HOST "${_matar_host}" _matar_host_var)
        set(${_matar_host_var} ON CACHE BOOL "Enabled by MATAR_HOST_BACKEND" FORCE)
    endif()
    if(_matar_dev)
        matar_backend_kokkos_var(DEVICE "${_matar_dev}" _matar_dev_var ", cuda, hip, sycl")
        set(${_matar_dev_var} ON CACHE BOOL "Enabled by MATAR_DEVICE_BACKEND" FORCE)
    endif()

    # Same space on both sides is valid and correct, but the _HOST macros then
    # contend with device work instead of overlapping it.
    if(_matar_host AND _matar_dev AND _matar_host STREQUAL _matar_dev)
        message(WARNING
            "MATAR_HOST_BACKEND and MATAR_DEVICE_BACKEND are both '${_matar_host}': the _HOST "
            "macros will run correctly but share one execution space with the device macros, "
            "so there is no host/device concurrency. Pair a host backend (openmp/pthreads) "
            "with a GPU device backend (cuda/hip/sycl) to overlap them.")
    endif()
endif()

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

    # Device space: exactly one HAVE_* here, since kokkos_types.h selects
    # DefaultExecSpace from this chain.
    if(Kokkos_ENABLE_CUDA)
        target_compile_definitions(matar INTERFACE HAVE_CUDA=1)
        # Silence CUDA compiler warnings
        target_compile_options(matar INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xcudafe --diag_suppress=177>"
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xcudafe --diag_suppress=550>")
    elseif(Kokkos_ENABLE_HIP)
        target_compile_definitions(matar INTERFACE HAVE_HIP=1)
    elseif(Kokkos_ENABLE_SYCL)
        target_compile_definitions(matar INTERFACE HAVE_SYCL=1)
    elseif(Kokkos_ENABLE_OPENMP)
        target_compile_definitions(matar INTERFACE HAVE_OPENMP=1)
    elseif(Kokkos_ENABLE_THREADS)
        target_compile_definitions(matar INTERFACE HAVE_THREADS=1)
    endif()

    # Host space, independent of the device chain above: on a GPU build this
    # is what the _HOST macros run in. The macros themselves resolve
    # Kokkos::DefaultHostExecutionSpace; these defines are for introspection
    # and tests.
    if(Kokkos_ENABLE_OPENMP)
        target_compile_definitions(matar INTERFACE HAVE_HOST_OPENMP=1)
    elseif(Kokkos_ENABLE_THREADS)
        target_compile_definitions(matar INTERFACE HAVE_HOST_THREADS=1)
    endif()

    # quad tiers force LIBQUADMATH on for the bundled build; an external or
    # parent-provided Kokkos must already have it.
    if(MATAR_ANY_QUAD AND NOT MATAR_KOKKOS_PROVIDER STREQUAL "bundled" AND NOT Kokkos_ENABLE_LIBQUADMATH)
        message(FATAL_ERROR
            "A MATAR quad-precision tier requires Kokkos built with "
            "Kokkos_ENABLE_LIBQUADMATH=ON; the ${MATAR_KOKKOS_PROVIDER} Kokkos lacks it")
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
