#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Test constructor and basic initialization
TEST(CSRArrayKokkosTest, Constructor) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });

    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });
    
    // Create CSR array
    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");
    
    // Check dimensions
    EXPECT_EQ(csr.dim1(), dim1);
    EXPECT_EQ(csr.dim2(), dim2);
    EXPECT_EQ(csr.nnz(), nnz);
}

// Test value access and modification
TEST(CSRArrayKokkosTest, ValueAccess) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });
    
    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });

    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    //
    // Where:
    // data = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    // column = [0, 1, 2, 0, 1, 2] 
    // row = [0, 2, 4, 6]

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    
    // Test value access
    EXPECT_DOUBLE_EQ(csr(0, 0), 1.5);  
    EXPECT_DOUBLE_EQ(csr(0, 1), 2.5);  
    EXPECT_DOUBLE_EQ(csr(1, 2), 3.5); 
    EXPECT_DOUBLE_EQ(csr(1, 0), 4.5);  
    EXPECT_DOUBLE_EQ(csr(2, 1), 5.5);  
    EXPECT_DOUBLE_EQ(csr(2, 2), 6.5);
    
    // Test zero elements
    EXPECT_DOUBLE_EQ(csr(0, 2), 0.0);  // Zero element
    EXPECT_DOUBLE_EQ(csr(1, 1), 0.0);  // Zero element
    EXPECT_DOUBLE_EQ(csr(2, 0), 0.0);  // Zero element
}

// Test iterator functionality
TEST(CSRArrayKokkosTest, IteratorFunctions) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;
    });
    
    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });


    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    //
    // Where:
    // data = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    // column = [0, 1, 2, 0, 1, 2] 
    // row = [0, 2, 4, 6]
    
    // Test begin/end iterators
    EXPECT_EQ(csr.begin(0), &data(0));  // First row starts at beginning
    EXPECT_EQ(csr.end(0), &data(2));  // First row ends at index 2
    
    // Test begin_index/end_index
    EXPECT_EQ(csr.begin_index(0), 0);  // First row starts at index 0
    EXPECT_EQ(csr.end_index(0), 2);    // First row ends at index 2
    
    // Test nnz per row
    EXPECT_EQ(csr.nnz(0), 2);  // First row has 2 non-zero elements
    EXPECT_EQ(csr.nnz(1), 2);  // Second row has 2 non-zero elements
    EXPECT_EQ(csr.nnz(2), 2);  // Third row has 2 non-zero elements
}

// Test flat access functions
TEST(CSRArrayKokkosTest, FlatAccess) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });
    
    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });
    
    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    //
    // Where:
    // data = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    // column = [0, 1, 2, 0, 1, 2] 
    // row = [0, 2, 4, 6]
    
    
    // Test get_val_flat
    EXPECT_DOUBLE_EQ(csr.get_val_flat(0), 1.5);
    EXPECT_DOUBLE_EQ(csr.get_val_flat(1), 2.5);
    EXPECT_DOUBLE_EQ(csr.get_val_flat(2), 3.5);
    
    // Test get_col_flat
    EXPECT_EQ(csr.get_col_flat(0), 0);
    EXPECT_EQ(csr.get_col_flat(1), 1);
    EXPECT_EQ(csr.get_col_flat(2), 2);
    
    // Test flat_index
    EXPECT_EQ(csr.flat_index(0, 0), 0);  // First element
    EXPECT_EQ(csr.flat_index(0, 1), 1);  // Second element
    EXPECT_EQ(csr.flat_index(1, 0), 3);  // Third element
}

// Test conversion to dense format
TEST(CSRArrayKokkosTest, ToDense) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });
    
    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });
    
    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");
    
    // Convert to dense format
    CArrayKokkos<double> dense(dim1, dim2);
    csr.to_dense(dense);

    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    //
    // Where:
    // data = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    // column = [0, 1, 2, 0, 1, 2] 
    // row = [0, 2, 4, 6]
    
    // Check dense matrix values
    EXPECT_DOUBLE_EQ(dense(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(dense(0, 1), 2.5);
    EXPECT_DOUBLE_EQ(dense(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(dense(1, 0), 4.5);
    EXPECT_DOUBLE_EQ(dense(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(dense(1, 2), 3.5);
    EXPECT_DOUBLE_EQ(dense(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(dense(2, 1), 5.5);
    EXPECT_DOUBLE_EQ(dense(2, 2), 6.5);
}

// Test set_values functionality
TEST(CSRArrayKokkosTest, SetValues) {
    // Create test data
    size_t nnz = 6;
    size_t dim1 = 3;
    size_t dim2 = 3;
    
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> row(dim1 + 1);
    CArrayKokkos<size_t> column(nnz);
    
    // Initialize data
    FOR_ALL(i, 0, nnz,{
        data(i) = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });
    
    // Initialize row pointers
    FOR_ALL(i, 0, dim1 + 1,{
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });
    
    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    
    // Set all non-zero values to 42.0
    csr.set_values(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(csr(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(csr(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(csr(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(csr(1, 2), 42.0);
    EXPECT_DOUBLE_EQ(csr(2, 1), 42.0);
    EXPECT_DOUBLE_EQ(csr(2, 2), 42.0);
    
    // Zero elements should remain zero
    EXPECT_DOUBLE_EQ(csr(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(csr(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(csr(2, 0), 0.0);
}

// Test name management
// TEST(CSRArrayKokkosTest, NameManagement) {
//     // Create test data
//     size_t nnz = 6;
//     size_t dim1 = 3;
//     size_t dim2 = 3;
    
//     CArrayKokkos<double> data(nnz);
//     CArrayKokkos<size_t> row(dim1 + 1);
//     CArrayKokkos<size_t> column(nnz);
    
//     // Initialize data
//     FOR_ALL(i, 0, nnz,{
//         data(i) = i + 1.5;
//         column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
//     });
    
//     // Initialize row pointers
//     FOR_ALL(i, 0, dim1 + 1,{
//         row(i) = i * 2;  // Row pointers: 0,2,4,6
//     });
    
//     CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");
    
//     // Check name
//     EXPECT_EQ(csr.get_name(), "test_csr");
// }
