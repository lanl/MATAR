#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [-t build_type] [-d] [-p install_prefix]"
    echo "build_type options: serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    echo "  -p    Installation prefix path (optional)"
    exit 1
}

# Parse command line arguments
while getopts "t:dp:" opt; do
    case ${opt} in
        t )
            build_type=$OPTARG
            ;;
        d )
            debug=true
            ;;
        p )
            INSTALL_PREFIX=$OPTARG
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

# Validate build type value
valid_types=("serial" "openmp" "pthreads" "cuda" "hip")
if [[ ! " ${valid_types[@]} " =~ " ${build_type} " ]]; then
    echo "Error: Invalid build type. Must be one of: ${valid_types[*]}"
    usage
fi

# Exit on error
#set -e

# Get the directory where the script is called from
CURRENT_DIR=$(pwd)
INSTALL_PREFIX="${INSTALL_PREFIX:-${CURRENT_DIR}/install}"
INSTALL_DIR="${CURRENT_DIR}/build_tmp"
BUILD_DIR="${INSTALL_DIR}/build"
KOKKOS_SOURCE_DIR="${INSTALL_DIR}/kokkos"
KOKKOS_BUILD_DIR="${BUILD_DIR}"
KOKKOS_INSTALL_DIR="${INSTALL_PREFIX}"

# Create directories
mkdir -p $INSTALL_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_PREFIX
cd $INSTALL_DIR

echo "Cloning Kokkos repository..."
git clone https://github.com/kokkos/kokkos.git
cd kokkos

echo "Creating build directory..."
cd $BUILD_DIR

# Kokkos flags for Cuda
CUDA_ADDITIONS=(
# -DKokkos_ARCH_PASCAL60=ON
-D Kokkos_ENABLE_CUDA=ON
-D Kokkos_ENABLE_CUDA_CONSTEXPR=ON
-D Kokkos_ENABLE_CUDA_LAMBDA=ON
-D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON
)

# Kokkos flags for Hip
HIP_ADDITIONS=(
-D Kokkos_ENABLE_HIP=ON
-D CMAKE_CXX_COMPILER=hipcc
-D Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON
)

# Kokkos flags for OpenMP
OPENMP_ADDITIONS=(
-D Kokkos_ENABLE_OPENMP=ON
)

# Kokkos flags for PThreads
PTHREADS_ADDITIONS=(
-D Kokkos_ENABLE_THREADS=ON
)

# Configure kokkos using CMake
cmake_options=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_DIR}"
    -D CMAKE_CXX_STANDARD=17
    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ARCH_NATIVE=ON
    -D Kokkos_ENABLE_TESTS=OFF
    -D BUILD_TESTING=OFF
)

if [ "$build_type" = "openmp" ]; then
    cmake_options+=(
        ${OPENMP_ADDITIONS[@]}
    )
elif [ "$build_type" = "pthreads" ]; then
    cmake_options+=(
        ${PTHREADS_ADDITIONS[@]}
    )
elif [ "$build_type" = "cuda" ]; then
    cmake_options+=(
        ${CUDA_ADDITIONS[@]}
    )
elif [ "$build_type" = "hip" ]; then
    cmake_options+=(
        ${HIP_ADDITIONS[@]}
    )
fi

if [ "$debug" = "true" ]; then
    echo "Setting debug to true for CMAKE build type"
    cmake_options+=(
        -DCMAKE_BUILD_TYPE=Debug
    )
fi

echo "Configuring Kokkos..."
# Configure kokkos
cmake "${cmake_options[@]}" -B "${KOKKOS_BUILD_DIR}" -S "${KOKKOS_SOURCE_DIR}"

echo "Building Kokkos..."
make -j$(nproc)

echo "Installing Kokkos..."
make install

echo "Cleaning up..."
cd $CURRENT_DIR
rm -rf $INSTALL_DIR

echo "Setting up environment..."
# Create a setup script instead of modifying .bashrc
cat > ${INSTALL_PREFIX}/setup_env.sh << EOF
#!/bin/bash
export CMAKE_PREFIX_PATH=\${CMAKE_PREFIX_PATH}:${INSTALL_PREFIX}
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${INSTALL_PREFIX}/lib64
EOF
chmod +x ${INSTALL_PREFIX}/setup_env.sh

echo "Kokkos installation completed!"
echo "Installation location: ${INSTALL_PREFIX}"
echo "To set up the environment variables, run:"
echo "source ${INSTALL_PREFIX}/setup_env.sh"