/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
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

void TpetraCArrayOneDimensionExample();
void TpetraCArrayTwoDimensionExample();
void TpetraCArraySevenDimensionExample();

int main(int argc, char* argv[])
{   
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    Kokkos::initialize();
    {
        // Run TpetraFArray 1D example
        if(process_rank==0){
            printf("\n====================Running 1D TpetraCarray example====================\n");
        }
        TpetraCArrayOneDimensionExample();

        // Run TpetraFArray 2D example
        if(process_rank==0){
            printf("\n====================Running 2D TpetraCarray example====================\n");
        }
        TpetraCArrayTwoDimensionExample();

        // Run TpetraFArray 7D example
        if(process_rank==0){
            printf("\n====================Running 7D TpetraCarray example====================\n");
        }
        TpetraCArraySevenDimensionExample();
    } // end of kokkos scope
    Kokkos::finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    if(process_rank==0)
        printf("\nfinished\n\n");
    MPI_Finalize();
}

void TpetraCArrayOneDimensionExample()
{

    int n = 20; //global dimension

    //distributed dual array with layout left
    TpetraDCArray<double> myarray(n);

    //local size
    int nlocal = myarray.size();

    // set values on host copy of data
    printf("Printing host copy of data (should be global ids):\n");
    for (int i = 0; i < nlocal; i++) {
        //set each array element to the corresponding global index
        //we get global indices using a partition map member in the array
        myarray.host(i) = myarray.pmap.getGlobalIndex(i);
    }

    myarray.update_device();

    // Print host copy of data
    myarray.print();
    Kokkos::fence();

    // Manupulate data on device and update host
    FOR_ALL(i, 0, nlocal,{
        myarray(i) = 2*myarray(i);
    });
    myarray.update_host();
    Kokkos::fence();
    printf("---Data multiplied by 2 on device---\n");

    // Print host copy of data
    myarray.print();
    Kokkos::fence();
}

void TpetraCArrayTwoDimensionExample()
{

    int nx = 20; //global dimension
    int ny = 5;

    //distributed (first dimension gets distributed) dual array with layout left
    TpetraDCArray<double> myarray(nx, ny);

    //local size
    int nxlocal = myarray.dims(0);

    // set values on host copy of data
    printf("Printing host copy of data (should be global ids):\n");
    for (int i = 0; i < nxlocal; i++) {
        for (int j = 0; j < ny; j++){
            //set each array element to a computed global degree of freedom index
            //we get global indices for dim0 using a partition map member in the array
            myarray.host(i,j) = ny*myarray.pmap.getGlobalIndex(i) + j;
        }
    }

    myarray.update_device();

    // Print host copy of data
    myarray.print();
    Kokkos::fence();

    // Manupulate data on device and update host
    FOR_ALL(i, 0, nxlocal,
            j, 0, ny,{
        myarray(i,j) = 2*myarray(i,j);
    });
    myarray.update_host();
    Kokkos::fence();
    printf("---Data multiplied by 2 on device---\n");

    // Print host copy of data
    myarray.print();
    Kokkos::fence();
}

void TpetraCArraySevenDimensionExample()
{

    int nx = 20; //global dimension
    int ny = 3;
    int nz = 3;
    int nu = 3;
    int ns = 2;
    int nt = 2;
    int nw = 2;

    //distributed (first dimension gets distributed) dual array with layout left
    TpetraDCArray<double> myarray(nx, ny, nz, nu, ns, nt, nw);

    //local size
    int nxlocal = myarray.dims(0);

    // set values on host copy of data
    printf("Printing host copy of data (should be global ids):\n");
    for (int i = 0; i < nxlocal; i++) {
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz; k++){
                for (int u = 0; u < nu; u++){
                    for (int s = 0; s < ns; s++){
                        for (int t = 0; t < nt; t++){
                            for (int w = 0; w < nw; w++){
                                //set each array element to a computed global degree of freedom index
                                //we get global indices for dim0 using a partition map member in the array
                                myarray.host(i,j,k,u,s,t,w) = ny*nz*nu*ns*nt*nw*myarray.pmap.getGlobalIndex(i) +
                                                              nz*nu*ns*nt*nw*j + nu*ns*nt*nw*k + ns*nt*nw*u +
                                                              nt*nw*s + nw*t + w;
                            }
                        }
                    }
                }
            }
        }
    }

    myarray.update_device();

    // Print host copy of data
    myarray.print();
    Kokkos::fence();

    // Manupulate data on device and update host
    FOR_ALL(i, 0, nxlocal,
            j, 0, ny,
            k, 0, nz,{
            for (int u = 0; u < nu; u++){
                    for (int s = 0; s < ns; s++){
                        for (int t = 0; t < nt; t++){
                            for (int w = 0; w < nw; w++){
                                myarray(i,j,k,u,s,t,w) = 2*myarray(i,j,k,u,s,t,w);
                            }
                        }
                    }
            }
    });
    myarray.update_host();
    Kokkos::fence();
    printf("---Data multiplied by 2 on device---\n");

    // Print host copy of data
    myarray.print();
    Kokkos::fence();
}
