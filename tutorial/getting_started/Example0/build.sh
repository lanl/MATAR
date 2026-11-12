#!/bin/bash

# Build this MATAR tutorial example with a chosen Kokkos backend.
# MATAR and its bundled Kokkos submodule are built automatically by CMake;
# backends are selected with the standard Kokkos_ENABLE_* CMake flags.

# Guard against sourcing
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "This script should be executed, not sourced"
    echo "Please run: ./build.sh -t <build_type>"
    return 1
fi

usage() {
    echo "Usage: $0 [-t build_type] [-d] [-v]"
    echo "build_type options: all, serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    echo "  -v    Enable vectorization verbose output (optional)"
    exit 1
}

while getopts "t:dv" opt; do
    case ${opt} in
        t ) build_type=$OPTARG ;;
        d ) debug=true ;;
        v ) vector_verbose=true ;;
        \? ) usage ;;
    esac
done

if [ -z "$build_type" ]; then
    echo "Error: Build type (-t) is required"
    usage
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

build_backend() {
    local backend=$1
    local build_dir="${SCRIPT_DIR}/build_${backend}"

    echo "===================================================="
    echo "Building MATAR example with ${backend} backend"
    echo "===================================================="

    # Map the backend name to standard Kokkos CMake flags
    CMAKE_OPTIONS="-DKokkos_ARCH_NATIVE=ON"
    case ${backend} in
        serial )
            ;;  # Kokkos enables the Serial backend by default
        openmp )
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DKokkos_ENABLE_OPENMP=ON" ;;
        pthreads )
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DKokkos_ENABLE_THREADS=ON" ;;
        cuda )
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON" ;;
        hip )
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON" ;;
        * )
            echo "Invalid backend: ${backend}"; usage ;;
    esac

    if [ "$debug" = "true" ]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
        echo "Debug build enabled"
    fi

    if [ "$vector_verbose" = "true" ]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DVECTOR_VERBOSE=ON"
        echo "Vectorization verbose output enabled"
    fi

    echo "Running CMake with options: ${CMAKE_OPTIONS}"
    cmake -S "${SCRIPT_DIR}" -B "${build_dir}" ${CMAKE_OPTIONS}
    cmake --build "${build_dir}" -j"$(nproc)"

    echo "Build for ${backend} completed!"
    echo "Executable is in: ${build_dir}"
    echo ""
}

if [ "${build_type}" = "all" ]; then
    BACKENDS=("serial" "openmp" "pthreads")

    if command -v nvcc &> /dev/null; then
        BACKENDS+=("cuda")
    else
        echo "CUDA not found, skipping CUDA backend build"
    fi

    if command -v hipcc &> /dev/null; then
        BACKENDS+=("hip")
    else
        echo "HIP not found, skipping HIP backend build"
    fi

    for backend in "${BACKENDS[@]}"; do
        build_backend "${backend}"
    done
else
    build_backend "${build_type}"
fi
