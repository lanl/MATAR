#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// FOR_ALL kernels cannot live inside TEST() — nvcc rejects KOKKOS_LAMBDA in
// the private TestBody().  All tests in this file share the same CSR
// initialization pattern, so one free function covers all of them.
inline void init_csr_data(CArrayKokkos<double>& data,
                          CArrayKokkos<size_t>& row,
                          CArrayKokkos<size_t>& column,
                          size_t nnz, size_t dim1) {
    FOR_ALL(i, 0, nnz, {
        data(i)   = i + 1.5;
        column(i) = i % 3;  // Column indices: 0,1,2,0,1,2
    });
    FOR_ALL(i, 0, dim1 + 1, {
        row(i) = i * 2;  // Row pointers: 0,2,4,6
    });
}
} // namespace

// Test constructor and basic initialization
TEST(CSRArrayKokkosTest, Constructor) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    EXPECT_EQ(csr.dim1(), dim1);
    EXPECT_EQ(csr.dim2(), dim2);
    EXPECT_EQ(csr.nnz(), nnz);
}

// Test value access and modification
TEST(CSRArrayKokkosTest, ValueAccess) {
    // The CSR matrix represents:
    // [1.5  2.5  0.0]
    // [4.5  0.0  3.5]
    // [0.0  5.5  6.5]
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    EXPECT_DOUBLE_EQ(csr(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(csr(0, 1), 2.5);
    EXPECT_DOUBLE_EQ(csr(1, 2), 3.5);
    EXPECT_DOUBLE_EQ(csr(1, 0), 4.5);
    EXPECT_DOUBLE_EQ(csr(2, 1), 5.5);
    EXPECT_DOUBLE_EQ(csr(2, 2), 6.5);

    // Zero elements
    EXPECT_DOUBLE_EQ(csr(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(csr(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(csr(2, 0), 0.0);
}

// Test iterator functionality
TEST(CSRArrayKokkosTest, IteratorFunctions) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    EXPECT_EQ(csr.begin(0), &data(0));
    EXPECT_EQ(csr.end(0),   &data(2));

    EXPECT_EQ(csr.begin_index(0), 0);
    EXPECT_EQ(csr.end_index(0),   2);

    EXPECT_EQ(csr.nnz(0), 2);
    EXPECT_EQ(csr.nnz(1), 2);
    EXPECT_EQ(csr.nnz(2), 2);
}

// Test flat access functions
TEST(CSRArrayKokkosTest, FlatAccess) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    EXPECT_DOUBLE_EQ(csr.get_val_flat(0), 1.5);
    EXPECT_DOUBLE_EQ(csr.get_val_flat(1), 2.5);
    EXPECT_DOUBLE_EQ(csr.get_val_flat(2), 3.5);

    EXPECT_EQ(csr.get_col_flat(0), 0);
    EXPECT_EQ(csr.get_col_flat(1), 1);
    EXPECT_EQ(csr.get_col_flat(2), 2);

    EXPECT_EQ(csr.flat_index(0, 0), 0);
    EXPECT_EQ(csr.flat_index(0, 1), 1);
    EXPECT_EQ(csr.flat_index(1, 0), 3);
}

// Test conversion to dense format
TEST(CSRArrayKokkosTest, ToDense) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    CArrayKokkos<double> dense(dim1, dim2);
    csr.to_dense(dense);

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
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    csr.set_values(42.0);

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
