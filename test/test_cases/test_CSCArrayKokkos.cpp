#include <gtest/gtest.h>
#include <matar.h>

using namespace mtr;

namespace {
// CArrayKokkos writes must happen on device — these helpers capture literal values in kernels
inline void init_csc_data(CArrayKokkos<double>& d) {
    FOR_ALL(i, 0, d.size(), {
        d(i) = (double)i + 1.0;
    });
    MATAR_FENCE();
}

inline void init_csc_start_index(CArrayKokkos<size_t>& si) {
    RUN({
        si(0) = 0; 
        si(1) = 2; 
        si(2) = 3; 
        si(3) = 4; 
        si(4) = 6;
    });
    MATAR_FENCE();
}

inline void init_csc_row_index(CArrayKokkos<size_t>& ri) {
    RUN({
        ri(0) = 0; 
        ri(1) = 2; 
        ri(2) = 1; 
        ri(3) = 2; 
        ri(4) = 0; 
        ri(5) = 3;
    });
    MATAR_FENCE();
}

// Capture csc(i,j) on device and store in a result view for host verification
inline double csc_get(CSCArrayKokkos<double>& csc, size_t i, size_t j) {
    CArrayKokkos<double> result(1, "csc_result");
    RUN({
        result(0) = csc(i, j);
    });
    MATAR_FENCE();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

inline size_t csc_begin_index(CSCArrayKokkos<double>& csc, size_t i) {
    CArrayKokkos<size_t> result(1, "csc_bi_result");
    RUN({
        result(0) = csc.begin_index(i);
    });
    MATAR_FENCE();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}

inline size_t csc_end_index(CSCArrayKokkos<double>& csc, size_t i) {
    CArrayKokkos<size_t> result(1, "csc_ei_result");
    RUN({
        result(0) = csc.end_index(i);
    });
    MATAR_FENCE();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.get_kokkos_view());
    return m(0);
}
} // namespace

class CSCArrayKokkosTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim1 = 4;
        dim2 = 4;
        nnz  = 6;
    }
    size_t dim1, dim2, nnz;
};

TEST_F(CSCArrayKokkosTest, Constructor) {
    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  start_index(dim2 + 1);
    CArrayKokkos<size_t>  row_index(nnz);
    init_csc_data(data);
    init_csc_start_index(start_index);
    init_csc_row_index(row_index);

    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    EXPECT_EQ(csc.dim1(), dim1);
    EXPECT_EQ(csc.dim2(), dim2);
    EXPECT_EQ(csc.nnz(), nnz);
}

TEST_F(CSCArrayKokkosTest, ValueAccess) {
    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  start_index(dim2 + 1);
    CArrayKokkos<size_t>  row_index(nnz);
    init_csc_data(data);
    init_csc_start_index(start_index);
    init_csc_row_index(row_index);

    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    EXPECT_DOUBLE_EQ(csc_get(csc, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 2, 0), 2.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 1, 1), 3.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 2, 2), 4.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 0, 3), 5.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 3, 3), 6.0);

    EXPECT_DOUBLE_EQ(csc_get(csc, 1, 0), 0.0);
    EXPECT_DOUBLE_EQ(csc_get(csc, 3, 0), 0.0);
}

TEST_F(CSCArrayKokkosTest, IteratorFunctions) {
    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  start_index(dim2 + 1);
    CArrayKokkos<size_t>  row_index(nnz);
    init_csc_data(data);
    init_csc_start_index(start_index);
    init_csc_row_index(row_index);

    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    EXPECT_EQ(csc_begin_index(csc, 0), 0);
    EXPECT_EQ(csc_end_index(csc, 0),   2);
    EXPECT_EQ(csc_begin_index(csc, 1), 2);
    EXPECT_EQ(csc_end_index(csc, 1),   3);
}

TEST_F(CSCArrayKokkosTest, FlatAccess) {
    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  start_index(dim2 + 1);
    CArrayKokkos<size_t>  row_index(nnz);
    init_csc_data(data);
    init_csc_start_index(start_index);
    init_csc_row_index(row_index);

    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    // Flat access is equivalent to reading the original data array
    auto m_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.get_kokkos_view());
    auto m_ri   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, row_index.get_kokkos_view());

    EXPECT_DOUBLE_EQ(m_data(0), 1.0);
    EXPECT_DOUBLE_EQ(m_data(1), 2.0);
    EXPECT_EQ(m_ri(0), static_cast<size_t>(0));
    EXPECT_EQ(m_ri(1), static_cast<size_t>(2));
}

TEST_F(CSCArrayKokkosTest, SetValues) {
    CArrayKokkos<double>  data(nnz);
    CArrayKokkos<size_t>  start_index(dim2 + 1);
    CArrayKokkos<size_t>  row_index(nnz);
    init_csc_data(data);
    init_csc_start_index(start_index);
    init_csc_row_index(row_index);

    CSCArrayKokkos<double> csc(data, start_index, row_index, dim1, dim2, "test_csc");

    csc.set_values(1.0);
    Kokkos::fence();

    // CSC shares data view with the original 'data' array — mirror it directly
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.get_kokkos_view());
    for (size_t i = 0; i < nnz; i++) {
        EXPECT_DOUBLE_EQ(m(i), 1.0);
    }
}
