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
 * @brief Comprehensive example demonstrating MATAR's data structures and performance portability
 * 
 * This example showcases MATAR's core data structures and design philosophy:
 * 1. Data-oriented design: Structures are organized around data access patterns rather than objects
 * 2. Performance portability: Code runs efficiently across different architectures (CPU, GPU, etc.)
 * 3. Memory layout control: Support for both C-style (row-major) and F-style (column-major) layouts
 * 4. Unified interface: Consistent API for both dense and sparse data structures
 * 
 * Key concepts demonstrated:
 * - Dense arrays (C-style and F-style)
 * - Array views for flexible data access without data copying
 * - Dual arrays for efficient host/device memory management
 * - Sparse matrix formats (CSC)
 * - Ragged arrays for irregular data structures
 * - Parallel operations using Kokkos
 * - Device synchronization and memory management
 */

#include <stdio.h>
#include <iostream>
#include <matar.h>
#include <algorithm>  // std::max, std::min, etc.

using namespace mtr; // matar namespace

int main()
{
    /**
     * Initialize Kokkos runtime
     * This is required before any Kokkos operations can be performed.
     * It sets up the execution environment based on the available hardware.
     */
    Kokkos::initialize();
    {
    // =========================
    // Dense Data Types
    // =========================
    
    /**
     * C-style Arrays (Row-major layout)
     * - First index varies slowest in memory (row-major order)
     * - Natural for C/C++ programmers
     * - Good for row-wise access patterns and cache utilization
     * - Example: For a 2D array A(i,j), elements A(0,0), A(0,1), A(0,2), ... are contiguous
     * 
     * CArrayDevice<T> allocates memory on the device (typically GPU)
     * The template parameter T specifies the data type (int, float, double, etc.)
     */
    CArrayDevice<int> carr_dev_1D(10);            // 1D array with 10 elements
    CArrayDevice<int> carr_dev_2D(10, 10);        // 2D array with 10x10 elements
    CArrayDevice<int> carr_dev_3D(10, 10, 10);    // 3D array with 10x10x10 elements


    /**
     * FOR_ALL is a MATAR macro that creates a parallel loop
     * - It automatically handles device execution
     * - When building for the GPU, this will launch a kernel
     * - The loop bounds are interpreted as [start, end)
     * - The code inside the braces is executed for each index in parallel
     * - When building in parallel on the CPU, this will launch a thread-based parallel loop
     * - using either OpenMP or Pthreads, depending on the build configuration
     */
    FOR_ALL(i, 0, 10, {
        carr_dev_1D(i) = i;    // Initialize array with index values
    });    

    /**
     * MATAR_FENCE() is a synchronization barrier
     * - Ensures all device operations are complete before proceeding
     * - Critical for correctness when there are data dependencies between operations
     * - Without it, CPU execution can continue before GPU operations finish
     */
    MATAR_FENCE();   

    // 2D parallel loop initializing a 2D array
    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        carr_dev_2D(i, j) = i+j;   // Initialize with sum of indices
    });
    MATAR_FENCE();
    
    // 3D parallel loop initializing a 3D array
    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        carr_dev_3D(i, j, k) = i+j+k;  // Initialize with sum of indices
    });
    MATAR_FENCE();
    
    /**
     * set_values() is a convenient method to initialize all elements to the same value
     * - More efficient than element-wise assignment in a loop
     * - Implemented as a parallel operation on the device
     */
    carr_dev_1D.set_values(10);   // Set all elements to 10
    carr_dev_2D.set_values(10);   // Set all elements to 10
    carr_dev_3D.set_values(10);   // Set all elements to 10

    printf("CArrayDevice passes test\n");

    /**
     * CMatrixDevice is similar to CArrayDevice but with 1-based indexing
     * - Designed for mathematical applications where 1-based indexing is preferred
     * - Functionally identical to CArrayDevice except for index offset
     * - Memory layout is the same (row-major)
     */
    CMatrixDevice<int> cmat_dev_1D(10);

    // Note: loop starts at 1 instead of 0 for CMatrixDevice
    FOR_ALL(i, 1, 10, {
        cmat_dev_1D(i) = i;
    });
    MATAR_FENCE();  
    printf("CMatrixDevice passes test\n");
    

    /**
     * F-style Arrays (Column-major layout)
     * - First index varies fastest in memory (column-major order)
     * - Natural for Fortran/Matlab programmers
     * - Good for column-wise access patterns
     * - Example: For a 2D array A(i,j), elements A(1,1), A(2,1), A(3,1), ... are contiguous
     * 
     * Memory layout is the key difference from C-style arrays:
     * - In row-major (C-style): A[row][col] = A[row*WIDTH + col]
     * - In column-major (F-style): A(row,col) = A[(col-1)*HEIGHT + (row-1)]
     */
    FArrayDevice<int> farr_dev_1D(10);
    FArrayDevice<int> farr_dev_2D(10, 10);
    FArrayDevice<int> farr_dev_3D(10, 10, 10);

    // 1D array initialization (works the same as C-style for 1D)
    FOR_ALL(i, 0, 10, {
        farr_dev_1D(i) = i;
    }); 

    /**
     * Note the index ordering in FArrayDevice:
     * - Parameters are in column-major order (j,i) instead of (i,j)
     * - This preserves the meaning of "first index varies fastest"
     * - Helps maintain consistency with Fortran-style array access
     */
    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        farr_dev_2D(j, i) = i+j;  // Note: j,i instead of i,j
    }); 

    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        farr_dev_3D(k, j, i) = i+j+k;  // Note: k,j,i instead of i,j,k
    }); 
    MATAR_FENCE();
    printf("FArrayDevice passes test\n");

    /**
     * Array Views
     * - Provide access to existing data without copying
     * - Enable flexible interpretations of memory layouts
     * - Useful for interfacing with external libraries or reinterpreting data
     * - Zero-cost abstraction (no runtime overhead)
     */
    ViewCArrayDevice<int> view_carr_dev_1D(carr_dev_1D.pointer(), 10);
    view_carr_dev_1D.set_values(10);
    MATAR_FENCE();  
    printf("ViewCArrayDevice created\n");

    // Verify view contents - should match the underlying array
    FOR_ALL(i, 0, 10, {
        view_carr_dev_1D(i) -= i;
        if (view_carr_dev_1D(i) != 10 - i) {
            printf("view_carr_dev_1D(%d) = %d\n", i, view_carr_dev_1D(i));
        }
    });     
    MATAR_FENCE();  

    // Views for F-style arrays work the same way
    ViewFArrayDevice<int> view_farr_dev_1D(farr_dev_1D.pointer(), 10);
    ViewFArrayDevice<int> view_farr_dev_2D(farr_dev_2D.pointer(), 10, 10);
    ViewFArrayDevice<int> view_farr_dev_3D(farr_dev_3D.pointer(), 10, 10, 10);

    // Initialize all views with the same value
    view_farr_dev_1D.set_values(10);
    view_farr_dev_2D.set_values(10);
    view_farr_dev_3D.set_values(10);

    // Verify view contents - modifying the view modifies the underlying array
    FOR_ALL(i, 0, 10, {
        view_farr_dev_1D(i) -= i;
        if (view_farr_dev_1D(i) != 10-i) {
            printf("view_farr_dev_1D(%d) = %d\n", i, view_farr_dev_1D(i));
        }
    }); 
    MATAR_FENCE();
    
    FOR_ALL(i, 0, 10,
            j, 0, 10, {
        view_farr_dev_2D(j, i) -= i+j;
        if (view_farr_dev_2D(j, i) != 10 - (i+j)) {
            printf("view_farr_dev_2D(%d, %d) = %d\n", j, i, view_farr_dev_2D(j, i));
        }
    }); 
    MATAR_FENCE();
    
    FOR_ALL(i, 0, 10,
            j, 0, 10,
            k, 0, 10, {
        view_farr_dev_3D(k, j, i) -= i+j+k;
        if (view_farr_dev_3D(k, j, i) != 10 - (i+j+k)) {
            printf("view_farr_dev_3D(%d, %d, %d) = %d\n", k, j, i, view_farr_dev_3D(k, j, i));
        }
    }); 
    MATAR_FENCE();
    printf("ViewFArrayDevice passes test\n");
    
    /**
     * Dual Arrays
     * - Exist on both host (CPU) and device (GPU) memory
     * - Provide separate accessors for host and device data
     * - Explicit data transfer between host and device via update methods
     * - Critical for heterogeneous computing patterns:
     *   1. Set up data on host
     *   2. Transfer to device
     *   3. Compute on device
     *   4. Transfer results back to host
     *
     * CArrayDual = Dual C-style Array using Kokkos backend
     */
    CArrayDual<int> d_carr_1D(10, "d_carr_1D");           // Name parameter helps with debugging
    CArrayDual<int> d_carr_2D(10, 10, "d_carr_2D");
    CArrayDual<int> d_carr_3D(10, 10, 10, "d_carr_3D");
    
    /**
     * Initialize on host using host() accessor
     * - Access is through regular sequential CPU code (not parallel)
     * - Changes only affect host memory until update_device() is called
     * - Useful for initialization or small changes when parallel execution isn't needed
     */
    for (int i = 0; i < 10; i++) {
        d_carr_1D.host(i) = i;
        for(int j = 0; j < 10; j++) {
            d_carr_2D.host(i, j) = i+j;
            for(int k = 0; k < 10; k++) {
                d_carr_3D.host(i, j, k) = i+j+k;
            }
        }
    }

    /**
     * Explicit data transfer to device
     * - update_device() copies data from host to device memory
     * - Must be called after host modifications and before device access
     * - Minimizing transfers is important for performance
     */
    d_carr_1D.update_device();
    d_carr_2D.update_device();
    d_carr_3D.update_device();

    printf("CArrayDual passes test\n");

    /**
     * Reduction Operations
     * - Parallel reductions combine values across all elements
     * - Common patterns: sum, min, max
     * - FOR_REDUCE_SUM macro handles the details of parallel reduction
     * - Each thread processes some elements and maintains a local sum
     * - Kokkos combines all local sums efficiently
     */
    
    // Local variable to hold partial sums during reduction
    int loc_sum_1D = 0;
    // Variable to store final sum
    int sum_1D = 0;     
    
    // Parallel reduction over 1D array
    // Note: local_sum_1D is used internally by Kokkos and needed for the pattern
    FOR_REDUCE_SUM(i, 0, 10,
                   loc_sum_1D, {
        loc_sum_1D += d_carr_1D(i);  // Add each element to local sum
    }, sum_1D);  // Final result goes to sum_1D

    printf("Sum of d_carr_1D on the device: %d\n", sum_1D); 

    // 2D reduction - operates over 2D index space
    int loc_sum_2D = 0;     
    int sum_2D = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   loc_sum_2D, {
        loc_sum_2D += d_carr_2D(i, j);
    }, sum_2D);

    printf("Sum of d_carr_2D on the device: %d\n", sum_2D);

    // 3D reduction - operates over 3D index space
    int loc_sum_3D = 0;     
    int sum_3D = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   k, 0, 10,    
                   loc_sum_3D, {
        loc_sum_3D += d_carr_3D(i, j, k);
    }, sum_3D);
    MATAR_FENCE();

    printf("Sum of d_carr_3D on the device: %d\n", sum_3D);

    /**
     * ViewCArrayDual combines the benefits of views and dual arrays
     * - Allows viewing existing data with dual host/device access
     * - Useful for operating on standard C++ arrays on the device
     * - No data copying - operates directly on the original memory
     */
    int some_array[9];
    // Initialize C array on host
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            some_array[i*3 + j] = i + j;
        }
    }

    // Create dual view of the array with 2D interpretation
    ViewCArrayDual<int> view_some_array_2D(&some_array[0], 3, 3);

    // Verify contents on device
    FOR_ALL(i, 0, 3,
            j, 0, 3, {
        if (view_some_array_2D(i, j) != i+j) {
            printf("view_some_array_2D(%d, %d) = %d\n", i, j, view_some_array_2D(i, j));
        }
    });
    MATAR_FENCE();
    printf("ViewCArrayDevice passes test\n");



    // =========================
    // Sparse Data Types
    // =========================

    // =========================
    // Ragged Array Types
    // =========================

    /**
     * RaggedCArrayDevice
     * - Data structure for arrays with variable-length rows
     * - Each row can have a different number of elements (stride)
     * - Memory efficient for irregular data structures
     * - Efficiently packed in memory - no wasted space
     * - Supports parallel operations across all rows
     * 
     * Common use cases:
     * - Storing variable-length lists (e.g., particles per cell)
     * - Representing irregular grids or sparse matrices
     * - Managing sparse data with varying row lengths
     * - Handling neighbor lists or connectivity data
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
    // First, create an array to hold the stride lengths
    CArrayDevice<size_t> some_strides(num_strides);

    // Initialize the stride lengths
    FOR_ALL(i, 0, num_strides, {
        some_strides(i) = i+1;  // Each row will have i+1 elements
    });
    
    // Create the ragged array with the specified strides
    // All memory management is handled automatically
    RaggedCArrayDevice<int> ragged_carr_dev(some_strides, "test_1D");
    
    // Fill the ragged array in parallel
    // Each thread handles one row, filling it with sequential numbers
    FOR_ALL(i, 0, num_strides, {
        // For each row, iterate up to its specific stride
        for(int j = 0; j < ragged_carr_dev.stride(i); j++) {
            ragged_carr_dev(i, j) = j + 1;  // Fill each row with 1,2,3,...
        }
    });
    MATAR_FENCE();

    // Print the ragged array
    // Note: Printing is done inside a RUN block for device compatibility
    // RUN executes the enclosed code once on the device in serial
    printf("RaggedCArrayDevice<int> ragged_carr_dev:\n");
    RUN({
        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < ragged_carr_dev.stride(i); j++) {    
                printf("%d ", ragged_carr_dev(i, j));
            }
            printf("\n");
        }
    });
    MATAR_FENCE();

    printf("RaggedCArrayDevice passes test\n");

    /**
     * RaggedCArrayDual
     * - Combines the ragged array structure with dual host/device functionality
     * - Initialize on host, compute on device
     * - Useful for complex or conditional initialization that's hard to parallelize
     */
    
    // Example: Creating another triangular pattern
    /*
    |1            |  stride = 1
    |1 2          |  stride = 2
    |1 2 3        |  stride = 3
    |1 2 3 4      |  stride = 4
    |1 2 3 4 5    |  stride = 5
    */ 
    printf("Creating RaggedCArrayDual\n");
    // Define strides on host
    size_t new_strides[5] = {1, 2, 3, 4, 5};
    
    // Create dual ragged array
    // Note: size (5) must be specified because C++ arrays don't carry their size
    RaggedCArrayDual<int> ragged_carr_dual(new_strides, 5, "test_1D"); 

    printf("Filling RaggedCArrayDual\n");
    // Fill on host using sequential code
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < ragged_carr_dual.stride_host(i); j++) {
            ragged_carr_dual.host(i, j) = j + 1;
        }
    }

    printf("Copying RaggedCArrayDual to device\n");
    // Copy the data to the device (GPU)
    ragged_carr_dual.update_device();

    // Verify the data on device using parallel execution
    FOR_ALL(i, 0, num_strides, {
        for(int j = 0; j < ragged_carr_dev.stride(i); j++) {
            if (ragged_carr_dev(i, j) != j + 1) {
                printf("ragged_carr_dev(%d, %d) = %d\n", i, j, ragged_carr_dev(i, j));
            }
        }
    });
    MATAR_FENCE();


    printf("RaggedCArrayDual passes test\n");

    /**
     * Compressed Sparse Column (CSC) Format
     * - Memory-efficient representation for sparse matrices
     * - Stores only non-zero elements plus indexing information
     * - Especially efficient for column-oriented operations
     * - Three key components:
     *   1. values: Non-zero element values (stored contiguously)
     *   2. rows: Row indices for each non-zero element
     *   3. starts: Starting index in values/rows for each column
     */
     
    // Define the sparse matrix dimensions
    size_t nnz  = 8;   // Number of non-zero elements
    size_t dim1 = 3;   // Number of rows
    size_t dim2 = 10;  // Number of columns
    
    // Example sparse matrix to represent:
    /*
    |1 2 2 0 0 0 0 0 0 0|
    |0 0 3 4 0 0 0 0 0 0|
    |0 0 0 0 5 6 0 0 0 14|
    */ 
    
    // Allocate arrays for CSC representation
    CArrayKokkos<size_t> starts(dim2 + 1); // Column start indices (length = #cols + 1)
    CArrayKokkos<size_t> rows(nnz);        // Row indices for each non-zero
    CArrayKokkos<int>    values(nnz);      // Values of non-zeros
    
    // Initialize the sparse matrix components
    // RUN executes once on device
    RUN({ 
        // starts[i] = index where column i begins in values/rows arrays
        // starts[dim2] = total number of non-zeros (end of last column)
        starts(0) = 0;  // First column starts at index 0
        starts(1) = 1;  // Second column starts at index 1
        starts(2) = 2;  // Third column starts at index 2
        starts(3) = 4;  // Fourth column starts at index 4
        starts(4) = 5;  // Fifth column starts at index 5
        starts(5) = 6;  // Sixth column starts at index 6
        starts(6) = 7;  // Seventh column starts at index 7
        starts(7) = 7;  // Eighth column starts at index 7 (empty column)
        starts(8) = 7;  // Ninth column starts at index 7 (empty column)
        starts(9) = 8;  // Tenth column starts at index 8
        starts(10)= 8;  // End of last column

        // Row indices for each non-zero element
        // Stores which row each value belongs to
        rows(0) = 0;  // First element is in row 0 (element at position 0,0)
        rows(1) = 0;  // Second element is in row 0 (element at position 0,1)
        rows(2) = 0;  // Third element is in row 0 (element at position 0,2)
        rows(3) = 1;  // Fourth element is in row 1 (element at position 1,2)
        rows(4) = 1;  // Fifth element is in row 1 (element at position 1,3)
        rows(5) = 2;  // Sixth element is in row 2 (element at position 2,4)
        rows(6) = 2;  // Seventh element is in row 2 (element at position 2,5)
        rows(7) = 1;  // Eighth element is in row 1 (element at position 1,9)
        
        // The actual values of non-zero elements
        // Stored in column-major order
        values(0) = 1;  // Value at (0,0)
        values(1) = 2;  // Value at (0,1)
        values(2) = 2;  // Value at (0,2)
        values(3) = 3;  // Value at (1,2)
        values(4) = 4;  // Value at (1,3)
        values(5) = 5;  // Value at (2,4)
        values(6) = 6;  // Value at (2,5)
        values(7) = 14; // Value at (1,9)
    });
    MATAR_FENCE();

    // Create CSC array from the components
    CSCArrayDevice<int> csc_dev(values, starts, rows, dim1, dim2, "CSC_Array");

    // Print matrix information
    RUN({
        printf("This matrix is %ld x %ld \n", csc_dev.dim1(), csc_dev.dim2());
        printf("nnz : %ld \n", csc_dev.nnz());
    });
    MATAR_FENCE();


    // Print the matrix in dense format
    // Note: Printing is done inside a RUN block for device compatibility
    // For GPU builds, data exists only on the device unless explicitly transferred
    RUN({
        for (size_t i = 0; i < csc_dev.dim1(); i++) {
            for (size_t j = 0; j < csc_dev.dim2(); j++) {
                printf("%d ", csc_dev(i, j));
            }
            printf("\n");
        }
    });
    MATAR_FENCE();
    printf("CSCArrayDevice passes test\n");

    } // End of Kokkos scope
    
    /**
     * Finalize Kokkos runtime
     * - Cleans up resources and ensures proper program termination
     * - Required after all Kokkos operations are complete
     * - Placed outside the Kokkos scope to ensure all operations finish first
     */
    Kokkos::finalize(); 

    std::cout << "MATAR data types demonstration completed successfully" << std::endl;
    return 0;
}
