#!/bin/bash

# Install script for Scotch and PT-Scotch
set -e

# Configuration
LIB_DIR="lib"
# SCOTCH_VERSION="7.0.4"
# PTSCOTCH_VERSION="7.0.4"
# INSTALL_PREFIX="$(pwd)/${LIB_DIR}"

# echo "Installing Scotch and PT-Scotch to ${INSTALL_PREFIX}"

# Create lib directory
mkdir -p "${LIB_DIR}"
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
cmake ..
make

echo "Installation complete! Libraries installed in: ${INSTALL_PREFIX}"