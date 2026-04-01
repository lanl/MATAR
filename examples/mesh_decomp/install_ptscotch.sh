#!/bin/bash

# Install script for Scotch and PT-Scotch
set -e

# Configuration
LIB_DIR="lib"
# SCOTCH_VERSION="7.0.4"
# PTSCOTCH_VERSION="7.0.4"
# INSTALL_PREFIX="$(pwd)/${LIB_DIR}"

# echo "Installing Scotch and PT-Scotch to ${INSTALL_PREFIX}"

# Create lib directory if it doesn't exist
if [ ! -d "${LIB_DIR}" ]; then
    mkdir -p "${LIB_DIR}"
fi
cd ${LIB_DIR}
# Clone and build Scotch
echo "Cloning Scotch..."
if [ -d "scotch" ]; then
    rm -rf scotch
fi
git clone https://gitlab.inria.fr/scotch/scotch.git
cd scotch

echo "Building Scotch..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DSCOTCH_MPI=ON \
         -DMPI_C_COMPILER=mpicc \
         -DMPI_Fortran_COMPILER=mpifort
make

echo "Installation complete! Libraries installed in: ${INSTALL_PREFIX}"