#!/bin/bash -e

kokkos_build_type="${1}"

cmake_options=(
    -D CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR}"
)

if [ "$kokkos_build_type" = "none" ]; then
    cmake_options+=(
        -D KOKKOS=OFF
    )
else
    cmake_options+=(
        -D KOKKOS=ON
    )
fi

# Print CMake options for reference
echo "CMake Options: ${cmake_options[@]}"

# Configure Examples
cmake "${cmake_options[@]}" -B "${EXAMPLE_BUILD_DIR}" -S "${EXAMPLE_SOURCE_DIR}"

# Build Examples
make -C "${EXAMPLE_BUILD_DIR}" -j${EXAMPLE_BUILD_CORES}

cd $basedir
