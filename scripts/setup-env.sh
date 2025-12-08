#!/bin/bash -e

# Initialize variables with default values
machine="$1"
kokkos_build_type="$2"
build_cores="$3"

my_build="build-matar-${kokkos_build_type}"

export scriptdir=`pwd`

cd ..
export topdir=`pwd`
export basedir=${topdir}
export srcdir=${basedir}/src
export libdir=${topdir}/lib
export builddir=${basedir}/${my_build}
export installdir=${basedir}/install

export EXAMPLE_SOURCE_DIR=${basedir}/examples
export EXAMPLE_BUILD_DIR=${builddir}

export TEST_SOURCE_DIR=${basedir}/test
export TEST_BUILD_DIR=${builddir}

export BENCHMARK_SOURCE_DIR=${basedir}/benchmark
export BENCHMARK_INSTALL_DIR=${basedir}/benchmark/build
export BENCHMARK_BUILD_DIR=${builddir}

export KOKKOS_SOURCE_DIR=${basedir}/src/Kokkos/kokkos
export KOKKOS_BUILD_DIR=${builddir}/kokkos
export KOKKOS_INSTALL_DIR=${installdir}/kokkos

# adding this for Windows, telling CMake where Kokkos is installed
export CMAKE_PREFIX_PATH="${KOKKOS_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"

export TRILINOS_SOURCE_DIR=${basedir}/lib/Trilinos
export TRILINOS_BUILD_DIR=${TRILINOS_SOURCE_DIR}/build-${kokkos_build_type}
export TRILINOS_INSTALL_DIR=${TRILINOS_BUILD_DIR}

export MATAR_SOURCE_DIR=${basedir}
export MATAR_BUILD_DIR=${builddir}/matar
export MATAR_INSTALL_DIR=${installdir}/matar

# adding this for Windows, I must tell cmake where MATAR is
export CMAKE_PREFIX_PATH="${MATAR_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"


export MATAR_BUILD_CORES=$build_cores

# Clean stale build and install(s)

cd $scriptdir

# Call the appropriate script to load modules based on the machine
source machines/$machine-env.sh ${2}



