#!/bin/bash -e

rm -rf ${MATAR_BUILD_DIR}
mkdir -p ${MATAR_BUILD_DIR}
cd ${MATAR_BUILD_DIR}

OPTIONS=(
-D MATAR_ENABLE_KOKKOS=ON
-D MATAR_ENABLE_KOKKOS_CUDA=ON
-D CMAKE_CXX_COMPILER=${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper
-D Kokkos_ROOT=${KOKKOS_INSTALL_DIR}
-D ENABLE_UNIT_TESTS=OFF
)
set -x
cmake "${OPTIONS[@]}" "${MATAR_BASE_DIR:-../}"
set +x
make -j16 -l32

cd $basedir
