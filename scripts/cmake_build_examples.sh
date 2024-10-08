#!/bin/bash -e

kokkos_build_type="${1}"
trilinos="${2}"

if [ ! -d "${EXAMPLE_SOURCE_DIR}/phaseFieldMPI/heffte" ]
then
  echo "Missing heffte for examples, downloading...."
  git clone https://bitbucket.org/icl/heffte.git ${EXAMPLE_SOURCE_DIR}/phaseFieldMPI/heffte
fi

cmake_options=(
    -D CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR}"
    -D CMAKE_BUILD_TYPE=Release
    #-D CMAKE_BUILD_TYPE=Debug
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

if [[ "$kokkos_build_type" = *"mpi"* ]]; then
    cmake_options+=(
        -D MPI=ON
    )
fi

# Print CMake options for reference
echo "CMake Options: ${cmake_options[@]}"

# Configure Examples
cmake "${cmake_options[@]}" -B "${EXAMPLE_BUILD_DIR}" -S "${EXAMPLE_SOURCE_DIR}"

# Build Examples
make -C "${EXAMPLE_BUILD_DIR}" -j${MATAR_BUILD_CORES}

cd $basedir
