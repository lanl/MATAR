#!/bin/bash -e

rm -rf ${MATAR_BUILD_DIR} ${MATAR_INSTALL_DIR}
mkdir -p ${MATAR_BUILD_DIR} 
cd ${MATAR_BUILD_DIR}

NUM_TASKS=32
if [ "$1" = "macos" ]
then
    NUM_TASKS=1
fi

# Kokkos flags for Cuda
CUDA_ADDITIONS=(
-D CUDA=ON
-D CMAKE_CXX_COMPILER=${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper
)

# Kokkos flags for Hip
HIP_ADDITIONS=(
-D HIP=ON
-D CMAKE_CXX_COMPILER=hipcc
)

# Kokkos flags for OpenMP
OPENMP_ADDITIONS=(
-D THREADS=ON
)

# Kokkos flags for PThreads
PTHREADS_ADDITIONS=(
-D THREADS=ON
)

# Empty those lists if not building
if [ "$2" = "cuda" ]
then
    HIP_ADDITIONS=() 
    PTHREADS_ADDITIONS=() 
    OPENMP_ADDITIONS=()
elif [ "$2" = "hip" ]
then
    CUDA_ADDITIONS=()
    PTHREADS_ADDITIONS=() 
    OPENMP_ADDITIONS=()
elif [ "$2" = "openmp" ]
then
    HIP_ADDITIONS=() 
    CUDA_ADDITIONS=()
    PTHREADS_ADDITIONS=() 
elif [ "$2" = "pthreads" ]
then
    HIP_ADDITIONS=() 
    CUDA_ADDITIONS=()
    OPENMP_ADDITIONS=()
else
    HIP_ADDITIONS=() 
    CUDA_ADDITIONS=()
    PTHREADS_ADDITIONS=() 
    OPENMP_ADDITIONS=()
fi

KOKKOS_ADDITIONS=(
-D KOKKOS=ON
-D Kokkos_DIR=${KOKKOS_INSTALL_DIR}/lib64/cmake/Kokkos
)
if [ "$2" = "none" ]
then
    KOKKOS_ADDITIONS=()
fi

ADDITIONS=(
${CUDA_ADDITIONS[@]}
${HIP_ADDITIONS[@]}
${OPENMP_ADDITIONS[@]}
${PTHREADS_ADDITIONS[@]}
${KOKKOS_ADDITIONS[@]}
)

OPTIONS=(
-D CMAKE_BUILD_TYPE=Release
-D CMAKE_INSTALL_PREFIX="${MATAR_INSTALL_DIR}"
-D CMAKE_CXX_STANDARD=17
${ADDITIONS[@]}
)
cmake "${OPTIONS[@]}" -S "${MATAR_SOURCE_DIR}"
make -j${NUM_TASKS}
make install

cd $scriptdir
