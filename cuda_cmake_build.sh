#!/bin/bash -e

rm -rf ${MATAR_BUILD_DIR}
mkdir -p ${MATAR_BUILD_DIR}
cd ${MATAR_BUILD_DIR}

OPTIONS=(
-D KOKKOS=ON
-D CUDA=ON
-D CMAKE_BUILD_TYPE=Debug
-D CMAKE_CXX_COMPILER=hipcc
-D Kokkos_DIR=${KOKKOS_INSTALL_DIR}/lib64/cmake/Kokkos
)
set -x
cmake "${OPTIONS[@]}" "${MATAR_BASE_DIR:-../}"
set +x
make -j16 -l32

