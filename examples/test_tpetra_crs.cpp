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

void TpetraCRSMatrixExample();

void TpetraCRSMatrixMultiplyExample();

int main(int argc, char* argv[])
{   
    MPI_Init(&argc, &argv);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    Kokkos::initialize();
    {
        // Run TpetraCRS example
        TpetraCRSMatrixExample();
        TpetraCRSMatrixMultiplyExample();
    } // end of kokkos scope
    Kokkos::finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    if(process_rank==0)
        printf("\nfinished\n\n");
    MPI_Finalize();
}

void TpetraCRSMatrixExample()
{
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    
    if(process_rank==0)
        printf("\n====================Running TpetraCRSMatrix example with map argument====================\n");
    
    //construct a row map over MPI ranks
    long long int n = 100; //global dimension
    TpetraPartitionMap<> input_pmap(n);
    int nlocal = input_pmap.size();

    //construct strides, index graph, and values arrays
    DCArrayKokkos<size_t, Kokkos::LayoutRight> matrix_strides(nlocal, "matrix_strides");
    //set strides; map is contiguous so Trilinos leaves device view of map empty (BE WARNED)
    const long long int min_global_index = input_pmap.getMinGlobalIndex();
    FOR_ALL(i, 0, nlocal,{
        matrix_strides(i) = (min_global_index+i) + 1;
    });

    //global indices array
    RaggedRightArrayKokkos<long long int, Kokkos::LayoutRight> input_crs(matrix_strides,"graph_indices");
    FOR_ALL(i, 0, nlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_crs(i,j) = j;
        }
    });

    //values array
    RaggedRightArrayKokkos<double, Kokkos::LayoutRight> input_values(matrix_strides,"ragged_values");
    FOR_ALL(i, 0, nlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_values(i,j) = 3*j;
        }
    });

    //NOTE! the constructor resorts the crs graph since Trilinos requires this; the values in each row are likely
    //in different places (in the same row) afterwards but the correspondence to global column ID is always preserved
    TpetraCRSMatrix<double, Kokkos::LayoutRight> mymatrix(input_pmap, matrix_strides, input_crs, input_values);
    //TpetraCRSMatrix<double, Kokkos::LayoutRight> mymatrix(input_pmap, matrix_strides);
    mymatrix.print();

    //test case that doesnt pass a map object; reset inputs since the graph and values array were resorted for Trilinos compatibility
    if(process_rank==0)
        printf("\n====================Running TpetraCRSMatrix example with local dim argument====================\n");

    //global indices array
    FOR_ALL(i, 0, nlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_crs(i,j) = j;
        }
    });

    //values array
    FOR_ALL(i, 0, nlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_values(i,j) = 3*j;
        }
    });
    TpetraCRSMatrix<double, Kokkos::LayoutRight> mymatrix2(nlocal, matrix_strides, input_crs, input_values);
    //TpetraCRSMatrix<double, Kokkos::LayoutRight> mymatrix(input_pmap, matrix_strides);
    mymatrix2.print();
}

void TpetraCRSMatrixMultiplyExample()
{   
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    if(process_rank==0)
        printf("\n====================Running Matrix Multiply example====================\n");

    int nx = 100; //global dimension
    int ny = 5;

    //distributed (first dimension gets distributed) dual array with layout left
    TpetraDFArray<double> myarray(nx, ny);

    //local size
    int nxlocal = myarray.dims(0);

    // set values on host copy of data
    for (int i = 0; i < nxlocal; i++) {
        for (int j = 0; j < ny; j++){
            //set each array element to a computed global degree of freedom index
            //we get global indices for dim0 using a partition map member in the array
            myarray.host(i,j) = ny*myarray.pmap.getGlobalIndex(i) + j;
        }
    }
    myarray.update_device();

    //construct matrix
    //construct strides, index graph, and values arrays
    DCArrayKokkos<size_t, Kokkos::LayoutRight> matrix_strides(nxlocal, "matrix_strides");
    //set strides; map is contiguous so Trilinos leaves device view of map empty (BE WARNED)
    const long long int min_global_index = myarray.pmap.getMinGlobalIndex();
    FOR_ALL(i, 0, nxlocal,{
        matrix_strides(i) = (min_global_index+i) + 1;
    });

    //global indices array
    RaggedRightArrayKokkos<long long int, Kokkos::LayoutRight> input_crs(matrix_strides,"graph_indices");
    FOR_ALL(i, 0, nxlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_crs(i,j) = j;
        }
    });

    //values array
    RaggedRightArrayKokkos<double, Kokkos::LayoutRight> input_values(matrix_strides,"ragged_values");
    FOR_ALL(i, 0, nxlocal,{
        for(int j = 0; j < matrix_strides(i); j++){
            input_values(i,j) = 3*j;
        }
    });

    TpetraCRSMatrix<double, Kokkos::LayoutRight> mymatrix(nxlocal, matrix_strides, input_crs, input_values);
    
    if(process_rank==0)
        printf("multiplication result:\n");

    //perform multiplication; currently the operator can allocate the result vector for you if it wasnt already
    TpetraDFArray<double> result = mymatrix*myarray;
    result.update_host();
    result.print();
}
