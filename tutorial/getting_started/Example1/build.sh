#!/bin/bash

# Guard against sourcing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    : # continue with script
else
    echo "This script should be executed, not sourced"
    echo "Please run: ./build.sh -t <build_type>"
    return 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [-t build_type] [-d] [-v]"
    echo "build_type options: all, serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    echo "  -v    Enable vectorization verbose output (optional)"
    exit 1
}

# Parse command line arguments
while getopts "t:dv" opt; do
    case ${opt} in
        t )
            build_type=$OPTARG
            ;;
        d )
            debug=true
            ;;
        v )
            vector_verbose=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate build type
if [ -z "$build_type" ]; then
    echo "Error: Build type (-t) is required"
    usage
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to build a specific backend
build_backend() {
    local backend=$1
    local build_dir="${SCRIPT_DIR}/build_${backend}"
    
    echo "===================================================="
    echo "Building MATAR example with ${backend} backend"
    echo "===================================================="
    
    # Create build directory
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    
    # Set CMake options based on flags
    CMAKE_OPTIONS="-DKokkos_BACKEND=${backend}"
    
    if [ "$debug" = "true" ]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DDEBUG_BUILD=ON"
        echo "Debug build enabled"
    fi
    
    if [ "$vector_verbose" = "true" ]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DVECTOR_VERBOSE=ON"
        echo "Vectorization verbose output enabled"
    fi
    
    # Configure with CMake
    echo "Running CMake with options: ${CMAKE_OPTIONS}"
    cmake ${CMAKE_OPTIONS} ..
    
    # Build
    echo "Building with make..."
    make -j$(nproc)
    
    echo "Build for ${backend} completed!"
    echo "Executable: ${build_dir}/matmul"
    echo ""
}

# Build based on the selected option
if [ "${build_type}" = "all" ]; then
    # Build all available backends
    BACKENDS=("serial" "openmp" "pthreads")
    
    # Add CUDA if available
    if command -v nvcc &> /dev/null; then
        BACKENDS+=("cuda")
    else
        echo "CUDA not found, skipping CUDA backend build"
    fi
    
    # Add HIP if available
    if command -v hipcc &> /dev/null; then
        BACKENDS+=("hip")
    else
        echo "HIP not found, skipping HIP backend build"
    fi
    
    # Build each backend
    for backend in "${BACKENDS[@]}"; do
        build_backend "${backend}"
    done
    
    echo "All builds completed!"
else
    # Build a single backend
    build_backend "${build_type}"
fi

echo "To run an example:"
echo "cd build_<backend>"
echo "./matmul" 