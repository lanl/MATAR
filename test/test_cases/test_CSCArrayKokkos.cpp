#include <gtest/gtest.h>
#include <matar.h>

using namespace mtr;

class CSCArrayKokkosTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code for all tests
        dim1 = 4;  // number of rows
        dim2 = 4;  // number of columns
        nnz = 6;   // number of non-zero elements
    }

    void TearDown() override {
        // Common cleanup code for all tests
    }

    size_t dim1, dim2, nnz;
};

TEST_F(CSCArrayKokkosTest, Constructor) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Verify dimensions
    EXPECT_EQ(csc.dim1(), dim1);
    EXPECT_EQ(csc.dim2(), dim2);
    EXPECT_EQ(csc.nnz(), nnz);
}

TEST_F(CSCArrayKokkosTest, ValueAccess) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Test value access
    EXPECT_DOUBLE_EQ(csc(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(csc(2, 0), 2.0);
    EXPECT_DOUBLE_EQ(csc(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(csc(2, 1), 4.0);
    EXPECT_DOUBLE_EQ(csc(0, 3), 5.0);
    EXPECT_DOUBLE_EQ(csc(3, 3), 6.0);

    // Test zero elements
    EXPECT_DOUBLE_EQ(csc(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(csc(3, 0), 0.0);
}

TEST_F(CSCArrayKokkosTest, IteratorFunctions) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Test begin/end functions
    EXPECT_EQ(csc.begin(0), &data(0));
    EXPECT_EQ(csc.end(0), &data(2));
    EXPECT_EQ(csc.begin(1), &data(2));
    EXPECT_EQ(csc.end(1), &data(3));

    // Test begin_index/end_index functions
    EXPECT_EQ(csc.begin_index(0), 0);
    EXPECT_EQ(csc.end_index(0), 2);
    EXPECT_EQ(csc.begin_index(1), 2);
    EXPECT_EQ(csc.end_index(1), 3);
}

TEST_F(CSCArrayKokkosTest, FlatAccess) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Test flat access functions
    EXPECT_DOUBLE_EQ(csc.get_val_flat(0), 1.0);
    EXPECT_DOUBLE_EQ(csc.get_val_flat(1), 2.0);
    EXPECT_EQ(csc.get_row_flat(0), 0);
    EXPECT_EQ(csc.get_row_flat(1), 2);

    // Test flat_index function
    EXPECT_EQ(csc.flat_index(0, 0), 0);
    EXPECT_EQ(csc.flat_index(2, 0), 1);
    EXPECT_EQ(csc.flat_index(1, 1), 2);
}

TEST_F(CSCArrayKokkosTest, SetValues) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Set all values to 1.0
    csc.set_values(1.0);

    // Verify values
    for (size_t i = 0; i < nnz; i++) {
        EXPECT_DOUBLE_EQ(csc.get_val_flat(i), 1.0);
    }
}

TEST_F(CSCArrayKokkosTest, NameManagement) {
    // Create arrays for CSC format
    CArrayKokkos<double> data(nnz);
    CArrayKokkos<size_t> start_index(dim2 + 1);
    CArrayKokkos<size_t> row_index(nnz);

    // Initialize data
    data(0) = 1.0; data(1) = 2.0; data(2) = 3.0;
    data(3) = 4.0; data(4) = 5.0; data(5) = 6.0;

    // Initialize column pointers (start_index)
    start_index(0) = 0; start_index(1) = 2;
    start_index(2) = 3; start_index(3) = 4;
    start_index(4) = 6;

    // Initialize row indices
    row_index(0) = 0; row_index(1) = 2;
    row_index(2) = 1; row_index(3) = 2;
    row_index(4) = 0; row_index(5) = 3;

    // Create CSC array with specific name
    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Test name management
    EXPECT_EQ(csc.get_name(), "test_csc");
}
