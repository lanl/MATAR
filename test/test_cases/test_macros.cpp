#include "matar.h"
#include "gtest/gtest.h"

using namespace mtr;

namespace {

constexpr int N = 4;

// ---------------------------------------------------------------------------
// CPU-side index helpers
// ---------------------------------------------------------------------------

inline int carr_index_3d(int i, int j, int k)
{
    return k * N * N + j * N + i;
}

inline int expected_sum_squares_1d(int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++) sum += i * i;
    return sum;
}

inline int expected_sum_squares_2d(int n)
{
    int sum = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            const int v = j * n + i;
            sum += v * v;
        }
    return sum;
}

inline int expected_sum_squares_3d(int n)
{
    int sum = 0;
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                const int v = carr_index_3d(i, j, k);
                sum += v * v;
            }
    return sum;
}

inline int expected_max_3d(int n)
{
    return carr_index_3d(n - 1, n - 1, n - 1);
}

// ---------------------------------------------------------------------------
// CPU-side element-wise EXPECT helpers
// ---------------------------------------------------------------------------

inline void expect_carray_1d_pattern(const CArrayKokkos<int>& arr, int n)
{
    for (int i = 0; i < n; i++) EXPECT_EQ(arr(i), i);
}

inline void expect_carray_2d_pattern(const CArrayKokkos<int>& arr, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            EXPECT_EQ(arr(i, j), j * n + i);
}

inline void expect_carray_3d_pattern(const CArrayKokkos<int>& arr, int n)
{
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                EXPECT_EQ(arr(i, j, k), carr_index_3d(i, j, k));
}

inline void expect_farray_constant(const FArrayKokkos<int>& arr, int n, int value)
{
    for (int i = 0; i < n; i++) EXPECT_EQ(arr(i), value);
}

inline void expect_farray_2d_constant(const FArrayKokkos<int>& arr, int n, int value)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            EXPECT_EQ(arr(i, j), value);
}

inline void expect_farray_3d_constant(const FArrayKokkos<int>& arr, int n, int value)
{
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                EXPECT_EQ(arr(i, j, k), value);
}

// ---------------------------------------------------------------------------
// Free functions wrapping FOR_ALL / DO_ALL kernels
// (KOKKOS_LAMBDA must not appear inside TEST()'s private TestBody)
// ---------------------------------------------------------------------------

inline void run_for_all_fill(CArrayKokkos<int>& arr1,
                              CArrayKokkos<int>& arr2,
                              CArrayKokkos<int>& arr3, int n)
{
    FOR_ALL(i, 0, n, { arr1(i) = i; });
    FOR_ALL(i, 0, n, { arr1(i) = i; }, "FOR_ALL 1D");
    FOR_ALL(i, 0, n, j, 0, n, { arr2(i, j) = j * n + i; });
    FOR_ALL(i, 0, n, j, 0, n, { arr2(i, j) = j * n + i; }, "FOR_ALL 2D");
    FOR_ALL(i, 0, n, j, 0, n, k, 0, n, { arr3(i, j, k) = carr_index_3d(i, j, k); });
    FOR_ALL(i, 0, n, j, 0, n, k, 0, n, { arr3(i, j, k) = carr_index_3d(i, j, k); }, "FOR_ALL 3D");
}

inline void run_do_all_fill(FArrayKokkos<int>& arr1,
                             FArrayKokkos<int>& arr2,
                             FArrayKokkos<int>& arr3, int n)
{
    DO_ALL(i, 0, n - 1, { arr1(i) = 7; });
    DO_ALL(i, 0, n - 1, { arr1(i) = 7; }, "DO_ALL 1D");
    DO_ALL(i, 0, n - 1, j, 0, n - 1, { arr2(i, j) = 8; });
    DO_ALL(i, 0, n - 1, j, 0, n - 1, { arr2(i, j) = 8; }, "DO_ALL 2D");
    DO_ALL(i, 0, n - 1, j, 0, n - 1, k, 0, n - 1, { arr3(i, j, k) = 9; });
    DO_ALL(i, 0, n - 1, j, 0, n - 1, k, 0, n - 1, { arr3(i, j, k) = 9; }, "DO_ALL 3D");
}

inline int reduce_sum_1d(CArrayKokkos<int>& arr, int n)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, { loc_sum += arr(i) * arr(i); },
                   result, "FOR_REDUCE_SUM 1D");
    return result;
}

inline int reduce_sum_2d(CArrayKokkos<int>& arr, int n)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, n,
                   j, 0, n,
                   loc_sum, { loc_sum += arr(i, j) * arr(i, j); },
                   result);
    return result;
}

inline int reduce_sum_3d(CArrayKokkos<int>& arr, int n)
{
    int loc_sum = 0, result = 0;
    FOR_REDUCE_SUM(i, 0, n,
                   j, 0, n,
                   k, 0, n,
                   loc_sum, { loc_sum += arr(i, j, k) * arr(i, j, k); },
                   result, "FOR_REDUCE_SUM 3D");
    return result;
}

inline void fill_3d_carr_indexed(CArrayKokkos<int>& arr, int n)
{
    FOR_ALL(i, 0, n, j, 0, n, k, 0, n, { arr(i, j, k) = carr_index_3d(i, j, k); });
}

inline int reduce_max_3d(CArrayKokkos<int>& arr, int n)
{
    int loc_max = 0, result = 0;
    FOR_REDUCE_MAX(i, 0, n,
                   j, 0, n,
                   k, 0, n,
                   loc_max, {
        if (loc_max < arr(i, j, k)) loc_max = arr(i, j, k);
    }, result);
    return result;
}

inline int reduce_min_3d(CArrayKokkos<int>& arr, int n)
{
    int loc_min = 1000000, result = 0;
    FOR_REDUCE_MIN(i, 0, n,
                   j, 0, n,
                   k, 0, n,
                   loc_min, {
        if (loc_min > arr(i, j, k)) loc_min = arr(i, j, k);
    }, result, "FOR_REDUCE_MIN 3D");
    return result;
}

inline void fill_constant_1d(CArrayKokkos<int>& arr, int n, int val)
{
    FOR_ALL(i, 0, n, { arr(i) = val; });
}

inline int reduce_product_1d(CArrayKokkos<int>& arr, int n)
{
    int loc_prod = 1, result = 1;
    FOR_REDUCE_PRODUCT(i, 0, n,
                       loc_prod, { loc_prod *= arr(i); },
                       result, "FOR_REDUCE_PRODUCT 1D");
    return result;
}

inline void fill_do_all_constant(FArrayKokkos<int>& arr1,
                                  FArrayKokkos<int>& arr2,
                                  FArrayKokkos<int>& arr3, int n)
{
    DO_ALL(i, 0, n - 1, { arr1(i) = 2; }, "DO_ALL 1D");
    DO_ALL(i, 0, n - 1, j, 0, n - 1, { arr2(i, j) = 3; }, "DO_ALL 2D");
    DO_ALL(i, 0, n - 1, j, 0, n - 1, k, 0, n - 1, { arr3(i, j, k) = k + 1; }, "DO_ALL 3D");
}

inline int do_reduce_sum_1d(FArrayKokkos<int>& arr, int n)
{
    int loc_sum = 0, result = 0;
    DO_REDUCE_SUM(i, 0, n - 1, loc_sum, { loc_sum += arr(i); }, result);
    return result;
}

inline int do_reduce_max_2d(FArrayKokkos<int>& arr, int n)
{
    int loc_max = 0, result = 0;
    DO_REDUCE_MAX(i, 0, n - 1,
                  j, 0, n - 1,
                  loc_max, {
        if (loc_max < arr(i, j)) loc_max = arr(i, j);
    }, result, "DO_REDUCE_MAX 2D");
    return result;
}

inline int do_reduce_min_3d(FArrayKokkos<int>& arr, int n)
{
    int loc_min = 1000000, result = 0;
    DO_REDUCE_MIN(i, 0, n - 1,
                  j, 0, n - 1,
                  k, 0, n - 1,
                  loc_min, {
        if (loc_min > arr(i, j, k)) loc_min = arr(i, j, k);
    }, result);
    return result;
}

inline void run_set_flag(CArrayKokkos<int>& flag, int val)
{
    RUN({ flag(0) = val; }, "RUN test");
}

inline void fill_3d_linear_indexed(CArrayKokkos<int>& arr, int n)
{
    FOR_ALL(i, 0, n, j, 0, n, k, 0, n, {
        arr(i, j, k) = i * n * n + j * n + k;
    });
}

inline void hierarchical_reduce_second(CArrayKokkos<int>& arr1,
                                        const CArrayKokkos<int>& arr3, int n)
{
    FOR_FIRST(i, 0, n, {
        int loc_sum = 0;
        int result  = 0;
        FOR_REDUCE_SUM_SECOND(j, i, n, loc_sum, {
            loc_sum += arr3(i, j, 0);
        }, result);
        arr1(i) = result;
    });
}

inline void hierarchical_nested_write(CArrayKokkos<int>& arr, int n)
{
    FOR_FIRST(i, 0, n, {
        FOR_SECOND(j, i, n, {
            FOR_THIRD(k, i, j, {
                arr(i, j, k) = i + j + k;
            });
        });
    });
}

// ---------------------------------------------------------------------------
// Class-based harness for _CLASS macro variants (already at class scope —
// KOKKOS_CLASS_LAMBDA is fine inside class methods).
// ---------------------------------------------------------------------------

class MacroClassHarness
{
public:
    int n_;
    CArrayKokkos<int> arr1_;
    CArrayKokkos<int> arr2_;
    CArrayKokkos<int> arr3_;
    CArrayKokkos<int> run_flag_;

    explicit MacroClassHarness(int n)
        : n_(n)
        , arr1_(n)
        , arr2_(n, n)
        , arr3_(n, n, n)
        , run_flag_(1)
    {}

    void fill_with_for_all_class()
    {
        FOR_ALL_CLASS(i, 0, n_, { arr1_(i) = i; });
        FOR_ALL_CLASS(i, 0, n_, j, 0, n_, { arr2_(i, j) = j * n_ + i; });
        FOR_ALL_CLASS(i, 0, n_, j, 0, n_, k, 0, n_, {
            arr3_(i, j, k) = carr_index_3d(i, j, k);
        }, "FOR_ALL_CLASS 3D");
    }

    int reduce_sum_class_1d() const
    {
        int loc_sum = 0, result = 0;
        FOR_REDUCE_SUM_CLASS(i, 0, n_,
                             loc_sum, { loc_sum += arr1_(i) * arr1_(i); },
                             result, "FOR_REDUCE_SUM_CLASS 1D");
        return result;
    }

    int reduce_max_class_3d() const
    {
        int loc_max = 1000000, result = 0;
        FOR_REDUCE_MAX_CLASS(i, 0, n_,
                             j, 0, n_,
                             k, 0, n_,
                             loc_max, {
            if (loc_max < arr3_(i, j, k)) loc_max = arr3_(i, j, k);
        }, result);
        return result;
    }

    int reduce_min_class_3d() const
    {
        int loc_min = 1000000, result = 0;
        FOR_REDUCE_MIN_CLASS(i, 0, n_,
                             j, 0, n_,
                             k, 0, n_,
                             loc_min, {
            if (loc_min > arr3_(i, j, k)) loc_min = arr3_(i, j, k);
        }, result);
        return result;
    }

    int reduce_product_class_1d() const
    {
        int loc_prod = 1, result = 1;
        FOR_REDUCE_PRODUCT_CLASS(i, 0, n_,
                                 loc_prod, { loc_prod *= arr1_(i); },
                                 result);
        return result;
    }

    void run_class_once()
    {
        run_flag_(0) = 0;
        RUN_CLASS({ run_flag_(0) = 99; }, "RUN_CLASS test");
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(TestMacros, FOR_ALL)
{
    CArrayKokkos<int> arr1(N), arr2(N, N), arr3(N, N, N);
    run_for_all_fill(arr1, arr2, arr3, N);
    MATAR_FENCE();
    expect_carray_1d_pattern(arr1, N);
    expect_carray_2d_pattern(arr2, N);
    expect_carray_3d_pattern(arr3, N);
}

TEST(TestMacros, DO_ALL)
{
    FArrayKokkos<int> arr1(N), arr2(N, N), arr3(N, N, N);
    run_do_all_fill(arr1, arr2, arr3, N);
    MATAR_FENCE();
    expect_farray_constant(arr1, N, 7);
    expect_farray_2d_constant(arr2, N, 8);
    expect_farray_3d_constant(arr3, N, 9);
}

TEST(TestMacros, FOR_REDUCE_SUM)
{
    CArrayKokkos<int> arr1(N), arr2(N, N), arr3(N, N, N);
    run_for_all_fill(arr1, arr2, arr3, N);
    MATAR_FENCE();
    EXPECT_EQ(reduce_sum_1d(arr1, N), expected_sum_squares_1d(N));
    EXPECT_EQ(reduce_sum_2d(arr2, N), expected_sum_squares_2d(N));
    EXPECT_EQ(reduce_sum_3d(arr3, N), expected_sum_squares_3d(N));
}

TEST(TestMacros, FOR_REDUCE_MAX_MIN)
{
    CArrayKokkos<int> arr3(N, N, N);
    fill_3d_carr_indexed(arr3, N);
    MATAR_FENCE();
    EXPECT_EQ(reduce_max_3d(arr3, N), expected_max_3d(N));
    EXPECT_EQ(reduce_min_3d(arr3, N), 0);
}

TEST(TestMacros, FOR_REDUCE_PRODUCT)
{
    CArrayKokkos<int> arr1(N);
    fill_constant_1d(arr1, N, 2);
    MATAR_FENCE();
    int expected = 1;
    for (int i = 0; i < N; i++) expected *= 2;
    EXPECT_EQ(reduce_product_1d(arr1, N), expected);
}

TEST(TestMacros, DO_REDUCE_SUM_MAX_MIN)
{
    FArrayKokkos<int> arr1(N), arr2(N, N), arr3(N, N, N);
    fill_do_all_constant(arr1, arr2, arr3, N);
    MATAR_FENCE();
    EXPECT_EQ(do_reduce_sum_1d(arr1, N), 2 * N);
    EXPECT_EQ(do_reduce_max_2d(arr2, N), 3);
    EXPECT_EQ(do_reduce_min_3d(arr3, N), 1);
}

TEST(TestMacros, RUN)
{
    CArrayKokkos<int> flag(1);
    flag(0) = 0;
    run_set_flag(flag, 42);
    MATAR_FENCE();
    EXPECT_EQ(flag(0), 42);
}

TEST(TestMacros, CLASS_macros)
{
    MacroClassHarness harness(N);

    harness.fill_with_for_all_class();
    MATAR_FENCE();
    expect_carray_1d_pattern(harness.arr1_, N);
    expect_carray_2d_pattern(harness.arr2_, N);
    expect_carray_3d_pattern(harness.arr3_, N);

    EXPECT_EQ(harness.reduce_sum_class_1d(), expected_sum_squares_1d(N));
    EXPECT_EQ(harness.reduce_max_class_3d(), expected_max_3d(N));
    EXPECT_EQ(harness.reduce_min_class_3d(), 0);

    int expected_product = 1;
    for (int i = 0; i < N; i++) expected_product *= i;
    EXPECT_EQ(harness.reduce_product_class_1d(), expected_product);

    harness.run_class_once();
    MATAR_FENCE();
    EXPECT_EQ(harness.run_flag_(0), 99);
}

TEST(TestMacros, Hierarchical_team_macros)
{
    CArrayKokkos<int> arr1(N), arr3(N, N, N);

    fill_3d_linear_indexed(arr3, N);
    MATAR_FENCE();

    hierarchical_reduce_second(arr1, arr3, N);
    MATAR_FENCE();

    for (int i = 0; i < N; i++) {
        int expected = 0;
        for (int j = i; j < N; j++) expected += i * N * N + j * N;
        EXPECT_EQ(arr1(i), expected);
    }

    hierarchical_nested_write(arr3, N);
    MATAR_FENCE();

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (j >= i && k >= i && k < j) {
                    EXPECT_EQ(arr3(i, j, k), i + j + k);
                }
            }
        }
    }
}
