/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/
#include <stdio.h>
#include <iostream>

#include "matar.h"
#include "Kokkos_DualView.hpp"

using namespace mtr; // matar namespace

void DViewCArrayKokkosTwoDimensionExample();
void DCArrayKokkosTwoDimensionExample();

int main()
{
    Kokkos::initialize();
    {
        // Run DViewCArrayKokkos 2D example
        DViewCArrayKokkosTwoDimensionExample();

        // Run DCArrayKokkos 2D example
        DCArrayKokkosTwoDimensionExample();
    } // end of kokkos scope
    Kokkos::finalize();
}

void DViewCArrayKokkosTwoDimensionExample()
{
    printf("\n====================Running 2D DViewCArrayKokkos example====================\n");

    int nx = 2;
    int ny = 2;

    // CPU arr
    int arr[nx * ny];

    for (int i = 0; i < nx * ny; i++) {
        arr[i] = 1;
    }

    // Create A_2D
    auto A_2D = DViewCArrayKokkos<int>(&arr[0], nx, ny);

    // Print host copy of data
    printf("Printing host copy of data (should be all 1s):\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%d\n", A_2D.host(i, j));
        }
    }

    // Print device copy of data
    printf("Printing device copy of data (should be all 1s):\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        printf("%d\n", A_2D(i, j));
    });
    Kokkos::fence();

    // Manupulate data on device and update host
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        A_2D(i, j) = 2;
    });
    A_2D.update_host();
    Kokkos::fence();
    printf("---Data updated to 2 on device---\n");

    // Print host copy of data
    printf("Printing host copy of data (should be all 2s):\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%d\n", A_2D.host(i, j));
        }
    }

    // Print device copy of data
    printf("Printing device copy of data (should be all 2s):\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        printf("%d\n", A_2D(i, j));
    });
    Kokkos::fence();

    // Print pointer to data on host and device
    printf("\nPrinting pointer to data on host and device.\n");
    printf("Should be same address if using OpenMP backend.\n");
    printf("Should be different addresses if using GPU backend.\n");
    printf("Host data pointer: %p\n", A_2D.host_pointer());
    printf("Device data pointer: %p\n", A_2D.device_pointer());
}

void DCArrayKokkosTwoDimensionExample()
{
    printf("\n====================Running 2D DCArrayKokkos example====================\n");

    int nx = 2;
    int ny = 2;

    // Create A_2D
    auto A_2D = DCArrayKokkos<int>(nx, ny);

    // Set data to one on host and updata device
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            A_2D.host(i, j) = 1;
        }
    }
    A_2D.update_device();
    Kokkos::fence();

    // Print host copy of data
    printf("Printing host copy of data (should be all 1s):\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%d\n", A_2D.host(i, j));
        }
    }

    // Print device copy of data
    printf("Printing device copy of data (should be all 1s):\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        printf("%d\n", A_2D(i, j));
    });
    Kokkos::fence();

    // Manupulate data on device and update host
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        A_2D(i, j) = 2;
    });
    A_2D.update_host();
    Kokkos::fence();
    printf("---Data updated to 2 on device---\n");

    // Print host copy of data
    printf("Printing host copy of data (should be all 2s):\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%d\n", A_2D.host(i, j));
        }
    }

    // Print device copy of data
    printf("Printing device copy of data (should be all 2s):\n");
    FOR_ALL(i, 0, nx,
            j, 0, ny, {
        printf("%d\n", A_2D(i, j));
    });
    Kokkos::fence();

    // Print pointer to data on host and device
    printf("\nPrinting pointer to data on host and device.\n");
    printf("Should be same address if using OpenMP backend.\n");
    printf("Should be different addresses if using GPU backend.\n");
    printf("Host data pointer: %p\n", A_2D.host_pointer());
    printf("Device data pointer: %p\n", A_2D.device_pointer());
}
