#include "matar.h"
#include "gtest/gtest.h"

using namespace mtr;

namespace {

// Different size per dimension to catch argument-order bugs in macros.
// If all dimensions were equal, a macro that silently swaps bounds would
// still produce valid indices and the test would not detect the error.
constexpr int N0 = 2;  // first dimension
constexpr int N1 = 3;  // second dimension
constexpr int N2 = 5;  // third dimension
constexpr int NH = 4;  // uniform size for hierarchical team-macro tests

// ---------------------------------------------------------------------------
// Fill patterns — "stored value equals flat storage index"
//
// CArrayKokkos (LayoutRight / C row-major):
//   arr(i,j,k)  stored at flat index  i*N1*N2 + j*N2 + k
//   => store value  i*N1*N2 + j*N2 + k  so that m(flat) == flat
//
// FArrayKokkos (LayoutLeft / column-major):
//   arr(i,j,k)  stored at flat index  i + j*N0 + k*N0*N1
//   => store value  i + j*N0 + k*N0*N1  so that m(flat) == flat
//
// Verification in both cases: for (int f = 0; f < total; f++) EXPECT_EQ(m(f), f)
// ---------------------------------------------------------------------------

// Sum of 0² + 1² + ... + (total-1)²
inline int sum_of_squares(int total)
{
    int s = 0;
    for (int i = 0; i < total; i++) s += i * i;
    return s;
}

// ---------------------------------------------------------------------------
// Host-side verification helpers (use host mirrors for CUDA compatibility)
// ---------------------------------------------------------------------------

inline void expect_carray_1d(const CArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int i = 0; i < N0; i++) EXPECT_EQ(m(i), i);
}

inline void expect_carray_2d(const CArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int f = 0; f < N0 * N1; f++) EXPECT_EQ(m(f), f);
}

inline void expect_carray_3d(const CArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int f = 0; f < N0 * N1 * N2; f++) EXPECT_EQ(m(f), f);
}

inline void expect_farray_1d(const FArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int i = 0; i < N0; i++) EXPECT_EQ(m(i), i);
}

inline void expect_farray_2d(const FArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int f = 0; f < N0 * N1; f++) EXPECT_EQ(m(f), f);
}

inline void expect_farray_3d(const FArrayKokkos<int>& arr)
{
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr.get_kokkos_view());
    for (int f = 0; f < N0 * N1 * N2; f++) EXPECT_EQ(m(f), f);
}

// ---------------------------------------------------------------------------
// Free functions wrapping FOR_ALL / DO_ALL kernels.
// KOKKOS_LAMBDA must not appear inside TEST()'s private TestBody.
// ---------------------------------------------------------------------------

inline void run_for_all_fill(CArrayKokkos<int>& arr1,
                              CArrayKokkos<int>& arr2,
                              CArrayKokkos<int>& arr3)
{
    FOR_ALL(i, 0, N0, { arr1(i) = i; });
    FOR_ALL(i, 0, N0, { arr1(i) = i; }, "FOR_ALL 1D");
    FOR_ALL(i, 0, N0, j, 0, N1, { arr2(i, j) = i * N1 + j; });
    FOR_ALL(i, 0, N0, j, 0, N1, { arr2(i, j) = i * N1 + j; }, "FOR_ALL 2D");
    FOR_ALL(i, 0, N0, j, 0, N1, k, 0, N2, { arr3(i, j, k) = i * N1 * N2 + j * N2 + k; });
    FOR_ALL(i, 0, N0, j, 0, N1, k, 0, N2, { arr3(i, j, k) = i * N1 * N2 + j * N2 + k; }, "FOR_ALL 3D");
}

// DO_ALL uses inclusive ranges: DO_ALL(i, 0, N0-1) loops i = 0 .. N0-1
inline void run_do_all_fill(FArrayKokkos<int>& arr1,
                             FArrayKokkos<int>& arr2,
                             FArrayKokkos<int>& arr3)
{
    DO_ALL(i, 0, N0 - 1, { arr1(i) = i; });
    DO_ALL(i, 0, N0 - 1, { arr1(i) = i; }, "DO_ALL 1D");
    DO_ALL(i, 0, N0 - 1, j, 0, N1 - 1, { arr2(i, j) = i + j * N0; });
    DO_ALL(i, 0, N0 - 1, j, 0, N1 - 1, { arr2(i, j) = i + j * N0; }, "DO_ALL 2D");
    DO_ALL(i, 0, N0 - 1, j, 0, N1 - 1, k, 0, N2 - 1, { arr3(i, j, k) = i + j * N0 + k * N0 * N1; });
    DO_ALL(i, 0, N0 - 1, j, 0, N1 - 1, k, 0, N2 - 1, { arr3(i, j, k) = i + j * N0 + k * N0 * N1; }, "DO_ALL 3D");
}

inline int reduce_sum_1d(CArrayKokkos<int>& arr)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, N0,
                   loc_sum, { loc_sum += arr(i) * arr(i); },
                   result, "FOR_REDUCE_SUM 1D");
    return result;
}

inline int reduce_sum_2d(CArrayKokkos<int>& arr)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, N0,
                   j, 0, N1,
                   loc_sum, { loc_sum += arr(i, j) * arr(i, j); },
                   result);
    return result;
}

inline int reduce_sum_3d(CArrayKokkos<int>& arr)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, N0,
                   j, 0, N1,
                   k, 0, N2,
                   loc_sum, { loc_sum += arr(i, j, k) * arr(i, j, k); },
                   result, "FOR_REDUCE_SUM 3D");
    return result;
}

inline void fill_3d_carray(CArrayKokkos<int>& arr)
{
    FOR_ALL(i, 0, N0, j, 0, N1, k, 0, N2,
            { arr(i, j, k) = i * N1 * N2 + j * N2 + k; });
}

inline int reduce_max_3d(CArrayKokkos<int>& arr)
{
    int loc_max = 0, result = 0;
    FOR_REDUCE_MAX(i, 0, N0,
                   j, 0, N1,
                   k, 0, N2,
                   loc_max, {
        if (loc_max < arr(i, j, k)) loc_max = arr(i, j, k);
    }, result);
    return result;
}

inline int reduce_min_3d(CArrayKokkos<int>& arr)
{
    int loc_min = 1000000, result = 0;
    FOR_REDUCE_MIN(i, 0, N0,
                   j, 0, N1,
                   k, 0, N2,
                   loc_min, {
        if (loc_min > arr(i, j, k)) loc_min = arr(i, j, k);
    }, result, "FOR_REDUCE_MIN 3D");
    return result;
}

inline void fill_constant_1d(CArrayKokkos<int>& arr, int val)
{
    FOR_ALL(i, 0, N0, { arr(i) = val; });
}

inline int reduce_product_1d(CArrayKokkos<int>& arr)
{
    int loc_prod = 1, result = 1;
    FOR_REDUCE_PRODUCT(i, 0, N0,
                       loc_prod, { loc_prod *= arr(i); },
                       result, "FOR_REDUCE_PRODUCT 1D");
    return result;
}

inline int do_reduce_sum_1d(FArrayKokkos<int>& arr)
{
    int loc_sum = 0, result = 0;
    DO_REDUCE_SUM(i, 0, N0 - 1, loc_sum, { loc_sum += arr(i); }, result);
    return result;
}

inline int do_reduce_max_2d(FArrayKokkos<int>& arr)
{
    int loc_max = 0, result = 0;
    DO_REDUCE_MAX(i, 0, N0 - 1,
                  j, 0, N1 - 1,
                  loc_max, {
        if (loc_max < arr(i, j)) loc_max = arr(i, j);
    }, result, "DO_REDUCE_MAX 2D");
    return result;
}

inline int do_reduce_min_3d(FArrayKokkos<int>& arr)
{
    int loc_min = 1000000, result = 0;
    DO_REDUCE_MIN(i, 0, N0 - 1,
                  j, 0, N1 - 1,
                  k, 0, N2 - 1,
                  loc_min, {
        if (loc_min > arr(i, j, k)) loc_min = arr(i, j, k);
    }, result);
    return result;
}

inline void run_set_flag(CArrayKokkos<int>& flag, int val)
{
    RUN({ flag(0) = val; }, "RUN test");
}

// Hierarchical tests use NH for all dimensions (these macros test team
// parallelism structure, not dimension ordering).
inline void fill_3d_nh(CArrayKokkos<int>& arr)
{
    FOR_ALL(i, 0, NH, j, 0, NH, k, 0, NH, {
        arr(i, j, k) = i * NH * NH + j * NH + k;
    });
}

inline void hierarchical_reduce_second(CArrayKokkos<int>& arr1,
                                        const CArrayKokkos<int>& arr3)
{
    FOR_FIRST(i, 0, NH, {
        int loc_sum = 0;
        int result  = 0;
        FOR_REDUCE_SUM_SECOND(j, i, NH, loc_sum, {
            loc_sum += arr3(i, j, 0);
        }, result);
        arr1(i) = result;
    });
}

inline void hierarchical_nested_write(CArrayKokkos<int>& arr)
{
    FOR_FIRST(i, 0, NH, {
        FOR_SECOND(j, i, NH, {
            FOR_THIRD(k, i, j, {
                arr(i, j, k) = i + j + k;
            });
        });
    });
}

} // namespace

// ---------------------------------------------------------------------------
// Class-based harness for _CLASS macro variants (already at class scope —
// KOKKOS_CLASS_LAMBDA is fine inside class methods).
// ---------------------------------------------------------------------------

class MacroClassHarness
{
public:
    CArrayKokkos<int> arr1_;
    CArrayKokkos<int> arr2_;
    CArrayKokkos<int> arr3_;
    CArrayKokkos<int> run_flag_;

    MacroClassHarness()
        : arr1_(N0)
        , arr2_(N0, N1)
        , arr3_(N0, N1, N2)
        , run_flag_(1)
    {}

    void fill_with_for_all_class()
    {
        FOR_ALL_CLASS(i, 0, N0, { arr1_(i) = i; });
        FOR_ALL_CLASS(i, 0, N0, j, 0, N1, { arr2_(i, j) = i * N1 + j; });
        FOR_ALL_CLASS(i, 0, N0, j, 0, N1, k, 0, N2, {
            arr3_(i, j, k) = i * N1 * N2 + j * N2 + k;
        }, "FOR_ALL_CLASS 3D");
    }

    int reduce_sum_class_1d() const
    {
        int loc_sum = 0, result = 0;
        FOR_REDUCE_SUM_CLASS(i, 0, N0,
                             loc_sum, { loc_sum += arr1_(i) * arr1_(i); },
                             result, "FOR_REDUCE_SUM_CLASS 1D");
        return result;
    }

    int reduce_max_class_3d() const
    {
        int loc_max = 0, result = 0;
        FOR_REDUCE_MAX_CLASS(i, 0, N0,
                             j, 0, N1,
                             k, 0, N2,
                             loc_max, {
            if (loc_max < arr3_(i, j, k)) loc_max = arr3_(i, j, k);
        }, result);
        return result;
    }

    int reduce_min_class_3d() const
    {
        int loc_min = 1000000, result = 0;
        FOR_REDUCE_MIN_CLASS(i, 0, N0,
                             j, 0, N1,
                             k, 0, N2,
                             loc_min, {
            if (loc_min > arr3_(i, j, k)) loc_min = arr3_(i, j, k);
        }, result);
        return result;
    }

    int reduce_product_class_1d() const
    {
        int loc_prod = 1, result = 1;
        FOR_REDUCE_PRODUCT_CLASS(i, 0, N0,
                                 loc_prod, { loc_prod *= arr1_(i); },
                                 result);
        return result;
    }

    void run_class_once()
    {
        run_flag_.set_values(0);
        RUN_CLASS({ run_flag_(0) = 99; }, "RUN_CLASS test");
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(TestMacros, FOR_ALL)
{
    CArrayKokkos<int> arr1(N0), arr2(N0, N1), arr3(N0, N1, N2);
    run_for_all_fill(arr1, arr2, arr3);
    MATAR_FENCE();
    expect_carray_1d(arr1);
    expect_carray_2d(arr2);
    expect_carray_3d(arr3);
}

TEST(TestMacros, DO_ALL)
{
    FArrayKokkos<int> arr1(N0), arr2(N0, N1), arr3(N0, N1, N2);
    run_do_all_fill(arr1, arr2, arr3);
    MATAR_FENCE();
    expect_farray_1d(arr1);
    expect_farray_2d(arr2);
    expect_farray_3d(arr3);
}

TEST(TestMacros, FOR_REDUCE_SUM)
{
    CArrayKokkos<int> arr1(N0), arr2(N0, N1), arr3(N0, N1, N2);
    run_for_all_fill(arr1, arr2, arr3);
    MATAR_FENCE();
    EXPECT_EQ(reduce_sum_1d(arr1), sum_of_squares(N0));
    EXPECT_EQ(reduce_sum_2d(arr2), sum_of_squares(N0 * N1));
    EXPECT_EQ(reduce_sum_3d(arr3), sum_of_squares(N0 * N1 * N2));
}

TEST(TestMacros, FOR_REDUCE_MAX_MIN)
{
    CArrayKokkos<int> arr3(N0, N1, N2);
    fill_3d_carray(arr3);
    MATAR_FENCE();
    EXPECT_EQ(reduce_max_3d(arr3), N0 * N1 * N2 - 1);
    EXPECT_EQ(reduce_min_3d(arr3), 0);
}

TEST(TestMacros, FOR_REDUCE_PRODUCT)
{
    CArrayKokkos<int> arr1(N0);
    fill_constant_1d(arr1, 2);
    MATAR_FENCE();
    int expected = 1;
    for (int i = 0; i < N0; i++) expected *= 2;
    EXPECT_EQ(reduce_product_1d(arr1), expected);
}

TEST(TestMacros, DO_REDUCE_SUM_MAX_MIN)
{
    FArrayKokkos<int> arr1(N0), arr2(N0, N1), arr3(N0, N1, N2);
    run_do_all_fill(arr1, arr2, arr3);
    MATAR_FENCE();
    // 1D: sum of flat indices 0..N0-1
    EXPECT_EQ(do_reduce_sum_1d(arr1), N0 * (N0 - 1) / 2);
    // 2D: max flat index = N0*N1 - 1
    EXPECT_EQ(do_reduce_max_2d(arr2), N0 * N1 - 1);
    // 3D: min flat index = 0
    EXPECT_EQ(do_reduce_min_3d(arr3), 0);
}

TEST(TestMacros, RUN)
{
    CArrayKokkos<int> flag(1);
    flag.set_values(0);
    run_set_flag(flag, 42);
    MATAR_FENCE();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flag.get_kokkos_view());
    EXPECT_EQ(m(0), 42);
}

TEST(TestMacros, CLASS_macros)
{
    MacroClassHarness harness;

    harness.fill_with_for_all_class();
    MATAR_FENCE();
    expect_carray_1d(harness.arr1_);
    expect_carray_2d(harness.arr2_);
    expect_carray_3d(harness.arr3_);

    EXPECT_EQ(harness.reduce_sum_class_1d(),  sum_of_squares(N0));
    EXPECT_EQ(harness.reduce_max_class_3d(),  N0 * N1 * N2 - 1);
    EXPECT_EQ(harness.reduce_min_class_3d(),  0);

    int expected_product = 1;
    for (int i = 0; i < N0; i++) expected_product *= i;
    EXPECT_EQ(harness.reduce_product_class_1d(), expected_product);

    harness.run_class_once();
    MATAR_FENCE();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, harness.run_flag_.get_kokkos_view());
        EXPECT_EQ(m(0), 99);
    }
}

TEST(TestMacros, Hierarchical_team_macros)
{
    CArrayKokkos<int> arr1(NH), arr3(NH, NH, NH);

    fill_3d_nh(arr3);
    MATAR_FENCE();

    hierarchical_reduce_second(arr1, arr3);
    MATAR_FENCE();

    {
        auto m1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr1.get_kokkos_view());
        for (int i = 0; i < NH; i++) {
            int expected = 0;
            for (int j = i; j < NH; j++) expected += i * NH * NH + j * NH;
            EXPECT_EQ(m1(i), expected);
        }
    }

    hierarchical_nested_write(arr3);
    MATAR_FENCE();

    {
        auto m3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, arr3.get_kokkos_view());
        for (int k = 0; k < NH; k++) {
            for (int j = 0; j < NH; j++) {
                for (int i = 0; i < NH; i++) {
                    if (j >= i && k >= i && k < j) {
                        int idx = i * NH * NH + j * NH + k;
                        EXPECT_EQ(m3(idx), i + j + k);
                    }
                }
            }
        }
    }
}
