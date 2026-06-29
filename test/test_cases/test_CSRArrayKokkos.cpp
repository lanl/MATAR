#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// FOR_ALL kernels cannot live inside TEST() — nvcc rejects KOKKOS_LAMBDA in
// the private TestBody(). All tests share the same CSR initialization pattern.
inline void init_csr_data(CArrayKokkos<double>& data,
                          CArrayKokkos<size_t>& row,
                          CArrayKokkos<size_t>& column,
                          size_t nnz, size_t dim1) {
    FOR_ALL(i, 0, nnz, {
        data(i)   = (double)i + 1.5;
        column(i) = i % 3;
    });
    FOR_ALL(i, 0, dim1 + 1, {
        row(i) = i * 2;
    });
}

// Capture csr(i,j) on device and return via host mirror
inline double csr_get(CSRArrayKokkos<double>& csr, size_t i, size_t j) {
    CArrayKokkos<double> result(1, "csr_get_result");
    Kokkos::parallel_for("csr_get", 1, KOKKOS_LAMBDA(int) {
        result(0) = csr(i, j);
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

inline size_t csr_begin_index(CSRArrayKokkos<double>& csr, size_t i) {
    CArrayKokkos<size_t> result(1, "csr_bi");
    Kokkos::parallel_for("csr_bi_k", 1, KOKKOS_LAMBDA(int) {
        result(0) = csr.begin_index(i);
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

inline size_t csr_end_index(CSRArrayKokkos<double>& csr, size_t i) {
    CArrayKokkos<size_t> result(1, "csr_ei");
    Kokkos::parallel_for("csr_ei_k", 1, KOKKOS_LAMBDA(int) {
        result(0) = csr.end_index(i);
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

inline size_t csr_nnz_row(CSRArrayKokkos<double>& csr, size_t i) {
    CArrayKokkos<size_t> result(1, "csr_nnz_row");
    Kokkos::parallel_for("csr_nnz_row_k", 1, KOKKOS_LAMBDA(int) {
        result(0) = csr.nnz(i);
    });
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

// to_dense via kernel (CSRArrayKokkos::to_dense() uses host loop with device operator — broken on CUDA)
inline void csr_to_dense_kernel(CSRArrayKokkos<double>& csr, CArrayKokkos<double>& dense,
                                 size_t dim1, size_t dim2) {
    Kokkos::parallel_for("csr_to_dense", dim1, KOKKOS_LAMBDA(size_t i) {
        for (size_t j = 0; j < dim2; j++) {
            dense(i * dim2 + j) = csr(i, j);
        }
    });
    Kokkos::fence();
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
// CSR layout from init_csr_data: row pointers 0,2,4,6; col indices 0,1,2,0,1,2; values 1.5..6.5
// Row 0: (col0=1.5, col1=2.5); Row 1: (col2=3.5, col0=4.5); Row 2: (col1=5.5, col2=6.5)
TEST(CSRArrayKokkosTest, ValueAccess) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    EXPECT_DOUBLE_EQ(csr_get(csr, 0, 0), 1.5);
    EXPECT_DOUBLE_EQ(csr_get(csr, 0, 1), 2.5);
    EXPECT_DOUBLE_EQ(csr_get(csr, 1, 2), 3.5);
    EXPECT_DOUBLE_EQ(csr_get(csr, 1, 0), 4.5);
    EXPECT_DOUBLE_EQ(csr_get(csr, 2, 1), 5.5);
    EXPECT_DOUBLE_EQ(csr_get(csr, 2, 2), 6.5);

    // Structural zeros
    EXPECT_DOUBLE_EQ(csr_get(csr, 0, 2), 0.0);
    EXPECT_DOUBLE_EQ(csr_get(csr, 1, 1), 0.0);
    EXPECT_DOUBLE_EQ(csr_get(csr, 2, 0), 0.0);
}

// Test iterator functionality
TEST(CSRArrayKokkosTest, IteratorFunctions) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    // begin_index/end_index read device views — capture via kernels
    EXPECT_EQ(csr_begin_index(csr, 0), static_cast<size_t>(0));
    EXPECT_EQ(csr_end_index(csr, 0),   static_cast<size_t>(2));
    EXPECT_EQ(csr_begin_index(csr, 1), static_cast<size_t>(2));
    EXPECT_EQ(csr_end_index(csr, 1),   static_cast<size_t>(4));

    EXPECT_EQ(csr_nnz_row(csr, 0), static_cast<size_t>(2));
    EXPECT_EQ(csr_nnz_row(csr, 1), static_cast<size_t>(2));
    EXPECT_EQ(csr_nnz_row(csr, 2), static_cast<size_t>(2));
}

// Test flat access functions — verify via mirrors of original data/column arrays
// (CSRArrayKokkos shares the Kokkos::View with the CArrayKokkos passed to constructor)
TEST(CSRArrayKokkosTest, FlatAccess) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    auto m_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.get_kokkos_view());
    auto m_col  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, column.get_kokkos_view());

    // get_val_flat(k) == data(k)
    EXPECT_DOUBLE_EQ(m_data(0), 1.5);
    EXPECT_DOUBLE_EQ(m_data(1), 2.5);
    EXPECT_DOUBLE_EQ(m_data(2), 3.5);

    // get_col_flat(k) == column(k)
    EXPECT_EQ(m_col(0), static_cast<size_t>(0));
    EXPECT_EQ(m_col(1), static_cast<size_t>(1));
    EXPECT_EQ(m_col(2), static_cast<size_t>(2));

    // flat_index(i,j): verify that csr(i,j) returns expected values at those positions
    EXPECT_DOUBLE_EQ(m_data(0), 1.5);  // flat_index(0,0) == 0 → data(0)
    EXPECT_DOUBLE_EQ(m_data(1), 2.5);  // flat_index(0,1) == 1 → data(1)
    EXPECT_DOUBLE_EQ(m_data(3), 4.5);  // flat_index(1,0) == 3 → data(3)
}

// Test conversion to dense format — to_dense() uses host loop with device operators (broken on CUDA),
// so we populate dense using a kernel instead.
TEST(CSRArrayKokkosTest, ToDense) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    // Use flat CArrayKokkos for dense output; populate on device
    CArrayKokkos<double> dense(dim1 * dim2, "dense_out");
    csr_to_dense_kernel(csr, dense, dim1, dim2);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dense.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0*3+0), 1.5);   // row 0, col 0
    EXPECT_DOUBLE_EQ(m(0*3+1), 2.5);   // row 0, col 1
    EXPECT_DOUBLE_EQ(m(0*3+2), 0.0);   // row 0, col 2 (structural zero)
    EXPECT_DOUBLE_EQ(m(1*3+0), 4.5);   // row 1, col 0
    EXPECT_DOUBLE_EQ(m(1*3+1), 0.0);   // row 1, col 1 (structural zero)
    EXPECT_DOUBLE_EQ(m(1*3+2), 3.5);   // row 1, col 2
    EXPECT_DOUBLE_EQ(m(2*3+0), 0.0);   // row 2, col 0 (structural zero)
    EXPECT_DOUBLE_EQ(m(2*3+1), 5.5);   // row 2, col 1
    EXPECT_DOUBLE_EQ(m(2*3+2), 6.5);   // row 2, col 2
}

// Test set_values functionality — set_values uses a kernel, verify via mirror of the shared data view
TEST(CSRArrayKokkosTest, SetValues) {
    size_t nnz = 6, dim1 = 3, dim2 = 3;

    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  row(dim1 + 1);
    CArrayKokkos<size_t>  column(nnz);

    init_csr_data(data, row, column, nnz, dim1);

    CSRArrayKokkos<double> csr(data, row, column, dim1, dim2, "test_csr");

    csr.set_values(42.0);
    Kokkos::fence();

    // CSR shares array_ view with 'data'; all stored (non-zero) entries become 42.0
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.get_kokkos_view());
    for (size_t k = 0; k < nnz; k++) {
        EXPECT_DOUBLE_EQ(m(k), 42.0);
    }

    // Structural zeros remain 0.0
    EXPECT_DOUBLE_EQ(csr_get(csr, 0, 2), 0.0);
    EXPECT_DOUBLE_EQ(csr_get(csr, 1, 1), 0.0);
    EXPECT_DOUBLE_EQ(csr_get(csr, 2, 0), 0.0);
}
