/**********************************************************************************************
 2020. Triad National Security, LLC. All rights reserved.
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

/**
 * @file MATAR_data.cpp
 * @brief Example demonstrating MATAR's data-oriented design and performance portability features
 * 
 * This example showcases MATAR's core data structures and design philosophy:
 * 1. Data-oriented design: Structures are organized around data access patterns rather than objects
 * 2. Performance portability: Code runs efficiently across different architectures (CPU, GPU, etc.)
 * 3. Memory layout control: Support for both C-style (row-major) and F-style (column-major) layouts
 * 4. Unified interface: Consistent API for both dense and sparse data structures
 * 
 * Key concepts demonstrated:
 * - Dense arrays (C-style and F-style)
 * - Array views for flexible data access
 * - Dual arrays for host/device memory management
 * - Sparse matrix formats (CSC)
 * - Parallel operations using Kokkos
 */

#include <stdio.h>
#include <iostream>
#include <matar.h>
#include <algorithm>  // std::max, std::min, etc.

using namespace mtr; // matar namespace

int main()
{
    // Initialize Kokkos runtime for performance portability
    Kokkos::initialize();
    {
    // =========================
    // Dense Data Types
    // =========================
    
    /**
     * C-style Arrays (Row-major layout)
     * - First index varies slowest in memory
     * - Natural for C/C++ programmers
     * - Good for row-wise access patterns
     */
    CArrayDevice<int> carr_dev_1D(10);
    CArrayDevice<int> carr_dev_2D(10, 10);
    CArrayDevice<int> carr_dev_3D(10, 10, 10);


    // FOR_ALL is a MATAR macro that creates a parallel loop
    // It automatically handles device execution
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
    
    // set_values() provides a convenient way to initialize arrays

    carr_dev_1D.set_values(10);
    carr_dev_2D.set_values(10);
    carr_dev_3D.set_values(10);

    printf("CArrayDevice passes test\n");

    // CMatrixDevice is a row major data structure that is equivalent to a CArrayDevice
    // except that the indexing starts at 1 instead of 0, this is to make the mathmaticians happy 
    CMatrixDevice<int> cmat_dev_1D(10);

    FOR_ALL(i, 1, 10, {
        cmat_dev_1D(i) = i;
    });

    printf("CMatrixDevice passes test\n");
    

    /**
     * F-style Arrays (Column-major layout)
     * - First index varies fastest in memory
     * - Natural for Fortran/Matlab programmers
     * - Good for column-wise access patterns
     */
    FArrayDevice<int> farr_dev_1D(10);
    FArrayDevice<int> farr_dev_2D(10, 10);
    FArrayDevice<int> farr_dev_3D(10, 10, 10);

    // Note the different index order in F-style arrays
    FOR_ALL(i, 0, 10, {
        farr_dev_1D(i) = i;
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        farr_dev_2D(j, i) = i+j;  // Note: j,i instead of i,j
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        farr_dev_3D(k, j, i) = i+j+k;  // Note: k,j,i instead of i,j,k
    }); 

    printf("FArrayDevice passes test\n");

    /**
     * Array Views
     * - Provide flexible access to existing data
     * - No data copying, just different access patterns
     * - Can reinterpret 1D arrays as multi-dimensional
     */
    ViewCArrayDevice<int> view_carr_dev_1D(carr_dev_1D.pointer(), 10);
    view_carr_dev_1D.set_values(10);


    // Example of using views to modify data
    FOR_ALL(i, 0, 10, {
        view_carr_dev_1D(i) -= i;
        if (view_carr_dev_1D(i) != 10 - i) {
            printf("view_carr_dev_1D(%d) = %d\n", i, view_carr_dev_1D(i));
        }
    });     

    // Example of viewing a 1D array as 2D
    int some_array[9];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            some_array[i*3 + j] = i + j;
        }
    }

    ViewCArrayDevice<int> view_some_array_2D(&some_array[0], 3, 3);
    
    
    FOR_ALL(i, 0, 3,
            j, 0, 3, {
        if (view_some_array_2D(i, j) != i+j) {
            printf("view_some_array_2D(%d, %d) = %d\n", i, j, view_some_array_2D(i, j));
        }
    });

    printf("ViewCArrayDevice passes test\n");


    // Views of F-style MATAR arrays
    ViewFArrayDevice<int> view_farr_dev_1D(farr_dev_1D.pointer(), 10);
    ViewFArrayDevice<int> view_farr_dev_2D(farr_dev_2D.pointer(), 10, 10);
    ViewFArrayDevice<int> view_farr_dev_3D(farr_dev_3D.pointer(), 10, 10, 10);

    view_farr_dev_1D.set_values(10);
    view_farr_dev_2D.set_values(10);
    view_farr_dev_3D.set_values(10);

    FOR_ALL(i, 0, 10, {
        view_farr_dev_1D(i) -= i;
        if (view_farr_dev_1D(i) != 10-i) {
            printf("view_farr_dev_1D(%d) = %d\n", i, view_farr_dev_1D(i));
        }
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        view_farr_dev_2D(j, i) -= i+j;
        if (view_farr_dev_2D(j, i) != 10 - (i+j)) {
            printf("view_farr_dev_2D(%d, %d) = %d\n", j, i, view_farr_dev_2D(j, i));
        }
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        view_farr_dev_3D(k, j, i) -= i+j+k;
        if (view_farr_dev_3D(k, j, i) != 10 - (i+j+k)) {
            printf("view_farr_dev_3D(%d, %d, %d) = %d\n", k, j, i, view_farr_dev_3D(k, j, i));
        }
    }); 
    printf("ViewFArrayDevice passes test\n");
    
    /**
     * Dual Arrays
     * - Exist on both host (CPU) and device (GPU)
     * - Automatic memory management
     * - Explicit data transfer control
     */
    printf("Creating DCArrayKokkos 1D\n");
    DCArrayKokkos<int> d_carr_1D(10, "d_carr_1D");
    
    printf("Creating DCArrayKokkos 2D\n");

    auto d_carr_2D = DCArrayKokkos<int>(10, 10, "d_carr_2D");

    // DCArrayKokkos<int> d_carr_2D(10, 10, "d_carr_2D");
    
    printf("Creating DCArrayKokkos 3D\n");
    DCArrayKokkos<int> d_carr_3D(10, 10, 10, "d_carr_3D");
    
    // Initialize on host
    for (int i = 0; i < 10; i++) {
        printf("i = %d\n", i);
        //d_carr_1D.host(i) = i;

        for(int j = 0; j < 10; j++) {
            printf("j = %d\n", j);
            d_carr_2D.host(i, j) = i+j;
            
            for(int k = 0; k < 10; k++) {
                printf("k = %d\n", k);
                d_carr_3D.host(i, j, k) = i+j+k;
 
            }
        }
    }

    // Explicit data transfer to device
    // d_carr_1D.update_device();
    d_carr_2D.update_device();
    d_carr_3D.update_device();

    printf("CArrayDual passes test\n");

    /**
     * Reduction Operations
     * - Parallel sum reduction example
     * - FOR_REDUCE_SUM macro handles parallel reduction
     * - Results are automatically combined
     */
    // int loc_sum_1D = 0;
    // int sum_1D = 0;     
    // FOR_REDUCE_SUM(i, 0, 10,
    //                loc_sum_1D, {
    //     loc_sum_1D += d_carr_1D(i);
    // }, sum_1D); 

    //printf("Sum of d_carr_1D on the device: %d\n", sum_1D);

    int loc_sum_2D = 0;     
    int sum_2D = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   loc_sum_2D, {
        loc_sum_2D += d_carr_2D(i, j);
    }, sum_2D);

    printf("Sum of d_carr_2D on the device: %d\n", sum_2D);

    int loc_sum_3D = 0;     
    int sum_3D = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   k, 0, 10,    
                   loc_sum_3D, {
        loc_sum_3D += d_carr_3D(i, j, k);
    }, sum_3D);

    printf("Sum of d_carr_3D on the device: %d\n", sum_3D);


    // =========================
    // Sparse Data Types
    // =========================

    // =========================
    // Ragged Array Types
    // =========================

    /**
     * RaggedCArrayDevice
     * - A data structure for storing arrays of varying lengths
     * - Each row can have a different number of elements (stride)
     * - Memory efficient for irregular data structures
     * - Supports parallel operations across all rows
     * 
     * Common use cases:
     * - Storing variable-length lists
     * - Representing irregular grids
     * - Managing sparse data with varying row lengths
     * - Handling particle data where each cell has different numbers of particles
     */

    // Example: Creating a triangular number pattern
    /*
    |1            |  stride = 1
    |1 2          |  stride = 2
    |1 2 3        |  stride = 3
    |1 2 3 4      |  stride = 4
    |1 2 3 4 5    |  stride = 5
    */ 
    // Note: Stride sizes can be arbitrary - they don't need to follow a pattern
    //       This example uses increasing strides for demonstration
        
    int num_strides = 5;
    CArrayDevice<size_t> some_strides(num_strides);

    // Initialize the stride lengths
    FOR_ALL(i, 0, num_strides, {
        some_strides(i) = i+1;  // Each row will have i+1 elements
    });
    
    // Create the ragged array with the specified strides
    RaggedCArrayDevice<int> ragged_carr_dev(some_strides, "test_1D");
    
    // Fill the ragged array in parallel
    // Each thread handles one row, filling it with sequential numbers
    FOR_ALL(i, 0, num_strides, {
        for(int j = 0; j < ragged_carr_dev.stride(i); j++) {
            ragged_carr_dev(i, j) = j + 1;  // Fill each row with 1,2,3,...
        }
    });

    // Print the ragged array
    // Note: Printing is done inside a RUN block for device compatibility
    printf("RaggedCArrayDevice<int> ragged_carr_dev:\n");
    RUN({
        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < ragged_carr_dev.stride(i); j++) {    
                printf("%d ", ragged_carr_dev(i, j));
            }
            printf("\n");
        }
    });

    printf("RaggedCArrayDevice passes test\n");

    // Dual Ragged Arrays exist on both the host and device

    // Example: Creating a triangular number pattern
    /*
    |1            |  stride = 1
    |1 2          |  stride = 2
    |1 2 3        |  stride = 3
    |1 2 3 4      |  stride = 4
    |1 2 3 4 5    |  stride = 5
    */ 
    // Note: Stride sizes can be arbitrary - they don't need to follow a pattern
    //       This example uses increasing strides for demonstration
    printf("Creating RaggedCArrayDual\n");
    size_t new_strides[5] = {1, 2, 3, 4, 5};
    RaggedCArrayDual<int> ragged_carr_dual(new_strides, 5, "test_1D"); 
    // Note, 5 is required here because the size of the standard array is not encoded in the array itself, 
    // and the sizeof() function is not always available on the device

    printf("Filling RaggedCArrayDual\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < ragged_carr_dual.stride(i); j++) {
            ragged_carr_dual(i, j) = j + 1;
        }
    }

    printf("Copying RaggedCArrayDual to device\n");
    // Copy the data to the device (GPU)
    ragged_carr_dual.update_device();

    // Check the data on the device
    FOR_ALL(i, 0, num_strides, {
        for(int j = 0; j < ragged_carr_dev.stride(i); j++) {
            if (ragged_carr_dev(i, j) != j + 1) {
                printf("ragged_carr_dev(%d, %d) = %d\n", i, j, ragged_carr_dev(i, j));
            }
        }
    });

    printf("RaggedCArrayDual passes test\n");

    /**
     * Compressed Sparse Column (CSC) Format
     * - Efficient for column-wise operations
     * - Good for sparse matrices with many zeros
     * - Three arrays store the data:
     *   1. values: Non-zero elements
     *   2. rows: Row indices of non-zero elements
     *   3. starts: Starting index of each column
     */
    size_t nnz  = 8; // number of non-zero elements
    size_t dim1 = 3; // number of rows
    size_t dim2 = 10; // number of columns
    
    // Example sparse matrix:
    /*
    |1 2 2 0 0 0 0 0 0 0|
    |0 0 3 4 0 0 0 0 0 0|
    |0 0 0 0 5 6 0 0 0 14|
    */ 
    
    CArrayKokkos<size_t> starts(dim2 + 1); // Column start indices
    CArrayKokkos<size_t> rows(nnz);        // Row indices
    CArrayKokkos<int>    values(nnz);      // Non-zero values
    
    // Initialize the sparse matrix
    RUN({ 
        starts(0) = 0;  // First column starts at index 0
        starts(1) = 1;  // Second column starts at index 1
        starts(2) = 2;  // Third column starts at index 2
        starts(3) = 4;  // Fourth column starts at index 4
        starts(4) = 5;  // Fifth column starts at index 5
        starts(5) = 6;  // Sixth column starts at index 6
        starts(6) = 7;  // Seventh column starts at index 7
        starts(7) = 7;  // Eighth column starts at index 7
        starts(8) = 7;  // Ninth column starts at index 7
        starts(9) = 8;  // Tenth column starts at index 8
        starts(10)= 8;  // End of last column

        // Row indices for each non-zero element
        rows(0) = 0;  // First element is in row 0
        rows(1) = 0;  // Second element is in row 0
        rows(2) = 0;  // Third element is in row 0
        rows(3) = 1;  // Fourth element is in row 1
        rows(4) = 1;  // Fifth element is in row 1
        rows(5) = 2;  // Sixth element is in row 2
        rows(6) = 2;  // Seventh element is in row 2
        rows(7) = 1;  // Eighth element is in row 1
        
        // Values of non-zero elements
        values(0) = 1;
        values(1) = 2;
        values(2) = 2;
        values(3) = 3;
        values(4) = 4;
        values(5) = 5;
        values(6) = 6;
        values(7) = 14;
    });

    // Create CSC array from the components
    CSCArrayDevice<int> csc_dev(values, starts, rows, dim1, dim2, "CSC_Array");

    // Print matrix information
    RUN({
        printf("This matrix is %ld x %ld \n", csc_dev.dim1(), csc_dev.dim2());
        printf("nnz : %ld \n", csc_dev.nnz());
    });

    // Print the matrix in dense format
    // Note: this is done inside of a RUN block to ensure portability. 
    //       If you try to print the matrix outside of a RUN (or any other MATAR macro) block, it will not work for GPU builds 
    //       because the data is on the device.
    RUN({
        for (size_t i = 0; i < csc_dev.dim1(); i++) {
            for (size_t j = 0; j < csc_dev.dim2(); j++) {
                printf("%d ", csc_dev(i, j));
            }
            printf("\n");
        }
    });
    printf("CSCArrayDevice passes test\n");










    }
    Kokkos::finalize(); 

    std::cout << "MATAR data types demonstration completed successfully" << std::endl;
    return 0;
}
