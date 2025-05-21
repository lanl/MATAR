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
#include <matar.h>

#include <algorithm>  // std::max, std::min, etc.

using namespace mtr; // matar namespace

// main
int main()
{

    Kokkos::initialize();
    {

    // =========================
    // Dense Data Types
    // =========================

    //  Dense Arrays (C-style)
    CArrayDevice<int> carr_dev_1D(10);
    CArrayDevice<int> carr_dev_2D(10, 10);
    CArrayDevice<int> carr_dev_3D(10, 10, 10);

    FOR_ALL(i, 0, 10, {
        carr_dev_1D(i) = i;
    });     

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        carr_dev_2D(i, j) = i+j;
    });
    
    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        carr_dev_3D(i, j, k) = i+j+k;
    });
    
    // Dense Arrays (F-style): First index varies fastest
    FArrayDevice<int> farr_dev_1D(10);
    FArrayDevice<int> farr_dev_2D(10, 10);
    FArrayDevice<int> farr_dev_3D(10, 10, 10);

    FOR_ALL(i, 0, 10, {
        farr_dev_1D(i) = i;
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        farr_dev_2D(j, i) = i+j;
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        farr_dev_3D(k, j, i) = i+j+k;
    }); 

    // Views of C-style MATAR arrays
    ViewCArrayDevice<int> view_carr_dev_1D(carr_dev_1D.pointer(), 10);
    ViewCArrayDevice<int> view_carr_dev_2D(carr_dev_2D.pointer(), 10, 10);
    ViewCArrayDevice<int> view_carr_dev_3D(carr_dev_3D.pointer(), 10, 10, 10);

    FOR_ALL(i, 0, 10, {
        view_carr_dev_1D(i) -= i;
        if (view_carr_dev_1D(i) != 0) {
            printf("view_carr_dev_1D(%d) = %d\n", i, view_carr_dev_1D(i));
        }
    });     

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        view_carr_dev_2D(i, j) -= i+j;
        if (view_carr_dev_2D(i, j) != 0) {
            printf("view_carr_dev_2D(%d, %d) = %d\n", i, j, view_carr_dev_2D(i, j));
        }
    });
    
    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        view_carr_dev_3D(i, j, k) -= i+j+k;
        if (view_carr_dev_3D(i, j, k) != 0) {
            printf("view_carr_dev_3D(%d, %d, %d) = %d\n", i, j, k, view_carr_dev_3D(i, j, k));
        }
    });

    // Views of C-style C++ arrays (requires pointer to first index of the array)
    int some_array[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    ViewCArrayDevice<int> view_some_array(&some_array[0], 9);

    FOR_ALL(i, 0, 9, {
        view_some_array(i) -= i;
        if (view_some_array(i) != 0) {
            printf("view_some_array(%d) = %d\n", i, view_some_array(i));
        }
    });

    // Using views to treat 1D arrays as N-dimensional arrays
    ViewCArrayDevice<int> view_some_array_2D(&some_array[0], 3, 3);

    FOR_ALL(i, 0, 3,
            j, 0, 3, {
        view_some_array_2D(i, j) += i+j;
        if (view_some_array_2D(i, j) != i+j) {
            printf("view_some_array_2D(%d, %d) = %d\n", i, j, view_some_array_2D(i, j));
        }
    });


    // Views of F-style MATAR arrays
    ViewFArrayDevice<int> view_farr_dev_1D(farr_dev_1D.pointer(), 10);
    ViewFArrayDevice<int> view_farr_dev_2D(farr_dev_2D.pointer(), 10, 10);
    ViewFArrayDevice<int> view_farr_dev_3D(farr_dev_3D.pointer(), 10, 10, 10);

    FOR_ALL(i, 0, 10, {
        view_farr_dev_1D(i) -= i;
        if (view_farr_dev_1D(i) != 0) {
            printf("view_farr_dev_1D(%d) = %d\n", i, view_farr_dev_1D(i));
        }
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        view_farr_dev_2D(j, i) -= i+j;
        if (view_farr_dev_2D(j, i) != 0) {
            printf("view_farr_dev_2D(%d, %d) = %d\n", j, i, view_farr_dev_2D(j, i));
        }
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        view_farr_dev_3D(k, j, i) -= i+j+k;
        if (view_farr_dev_3D(k, j, i) != 0) {
            printf("view_farr_dev_3D(%d, %d, %d) = %d\n", k, j, i, view_farr_dev_3D(k, j, i));
        }
    }); 

    CArrayKokkos<int> tmp_array;
    tmp_array = CArrayKokkos<int>(10, "temp_array");

    // Sparse Data Types
    // Compressed Sparse Column (CSC) format
    /*
    |1 2 2 0 0 0 0 0 0 0|
    |0 0 3 4 0 0 0 0 0 0|
    |0 0 0 0 5 6 0 0 0 14|
    */ 
    size_t nnz  = 8; // number of non-zero elements
    size_t dim1 = 3; // number of rows
    size_t dim2 = 10; // number of columns
    CArrayKokkos<size_t> starts(dim2 + 1); // 1d array that marks where the first element of each column starts
    CArrayKokkos<size_t> rows(nnz); // 1d array that marks what row each element is in
    CArrayKokkos<int>    values(nnz); // 1d array of data values in order as read top to bottom, left to right
    RUN({ 
        starts(0) = 0;
        starts(1) = 1;
        starts(2) = 2;
        starts(3) = 4;
        starts(4) = 5;
        starts(5) = 6;
        starts(6) = 7;
        starts(7) = 7;
        starts(8) = 7;
        starts(9) = 8;
        starts(10)= 8;

        rows(0) = 0;
        rows(1) = 0;
        rows(2) = 0;
        rows(3) = 1;
        rows(4) = 1;
        rows(5) = 2;
        rows(6) = 2;
        rows(7) = 1;
        
        values(0) = 1;
        values(1) = 2;
        values(2) = 2;
        values(3) = 3;
        values(4) = 4;
        values(5) = 5;
        values(6) = 6;
        values(7) = 14;
    });


    CSCArrayDevice<int>csc_dev(values, starts, rows, dim1, dim2, "CSC_Array");


    RUN({
        printf("This matix is %ld x %ld \n", csc_dev.dim1(), csc_dev.dim2());
    });

    RUN({
        printf("nnz : %ld \n", csc_dev.nnz());
    });

    // Print the matrix
    RUN({
        for (size_t i = 0; i < csc_dev.dim1(); i++) {
            for (size_t j = 0; j < csc_dev.dim2(); j++) {
                printf("%d ", csc_dev(i, j));
            }
        printf("\n");
        }
    });

    // Print the matric again, outside of a RUN block
    printf("\n");
    for (size_t i = 0; i < csc_dev.dim1(); i++) {
        for (size_t j = 0; j < csc_dev.dim2(); j++) {
            printf("%d ", csc_dev(i, j));
        }
        printf("\n");
    }







    // int loc_total = 0;
    // loc_total += 0; // Get rid of warning
    // FOR_REDUCE_SUM(i, 0, nnz,
    //                 loc_total, {
    //         loc_total += values[i];
    // }, total);
    // printf("Sum of nnz from pointer method %d\n", total);
    // total = 0;
    // FOR_REDUCE_SUM(i, 0, nnz,
    //                 loc_total, {
    //         loc_total += a_start[i];
    // }, total);
    // printf("Sum of start indices form .get_starts() %d\n", total);
    // total = 0;

    // FOR_REDUCE_SUM(i, 0, dim1,
    //                 j, 0, dim2 - 1,
    //     loc_total, {
    //         loc_total += A(i, j);
    // }, total);
    // printf("Sum of nnz in array notation %d\n", total);

    }
    Kokkos::finalize(); 

    std::cout << "MATAR data types demonstration completed successfully" << std::endl;
    return 0;
}
