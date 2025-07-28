#!/bin/bash -e

kokkos_build_type="${1}"
trilinos="${2}"
debug="${3}"

if [ ! -d "${TEST_SOURCE_DIR}/googletest" ]
then
  echo "Missing googletest for testing, downloading...."
  git clone https://github.com/google/googletest.git ${TEST_SOURCE_DIR}/googletest
fi

if [ "$debug" = "enabled" ]; then
    cmake_options=(
        -D CMAKE_BUILD_TYPE=Debug
        -D CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR}"
    )
else
    cmake_options=(
        -D CMAKE_BUILD_TYPE=Release
        -D CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR}"
    )
fi

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

# Configure test
cmake "${cmake_options[@]}" -B "${TEST_BUILD_DIR}" -S "${TEST_SOURCE_DIR}"

# Build test
make -C "${TEST_BUILD_DIR}" -j${MATAR_BUILD_CORES}

cd $basedir
