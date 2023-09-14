#!/bin/bash -e
if [ "$1" != "hpc" ] && [ "$1" != "macos" ] && [ "$1" != "linux" ]
then
    echo "The first argument needs to be either hpc, macos, or linux"
    return 1
fi
if [ "$2" != "cuda" ] && [ "$2" != "hip" ] && [ "$2" != "openmp" ] && [ "$2" != "pthreads" ] && [ "$2" != "serial" ] && [ "$2" != "none" ]
then
    echo "The second argument needs to be either cuda, hip, openmp, pthreads, serial, or none"
    return 1
fi

SYSTEM=$1
PARALLEL=$2
CUSTOM_BUILD=$3

source setup-env.sh ${SYSTEM} ${PARALLEL} ${CUSTOM_BUILD}
if [ "$2" != "none" ]
then
    source kokkos-install.sh ${SYSTEM} ${PARALLEL}
fi
source cmake_build.sh ${SYSTEM} ${PARALLEL}
