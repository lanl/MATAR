#!/bin/bash -e

kokkos_build_type="${1}"

if [ ! -d "${BENCHMARK_SOURCE_DIR}/benchmark" ]
then
  echo "Missing googlebenchmark for benchmarking, downloading and installing...."
  git clone https://github.com/google/benchmark.git ${BENCHMARK_SOURCE_DIR}/benchmark
  cd ${BENCHMARK_SOURCE_DIR}/benchmark
  cmake -E make_directory "build"
  cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
  cmake --build  "build" --config Release -j${MATAR_BUILD_CORES}
  # Test install
  cmake -E chdir "build" ctest --build-config Release
fi

cmake_options=(
    -D CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR};${KOKKOS_INSTALL_DIR};${BENCHMARK_INSTALL_DIR}"
    -D BENCHMARK_DOWNLOAD_DEPENDENCIES=on
    -DCMAKE_BUILD_TYPE=Release
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

cmake "${cmake_options[@]}" -B "${BENCHMARK_BUILD_DIR}" -S "${BENCHMARK_SOURCE_DIR}"

# Build benchmark
make -C "${BENCHMARK_BUILD_DIR}" -j${MATAR_BUILD_CORES}

cd $basedir