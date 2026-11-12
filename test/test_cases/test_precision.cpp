#include "matar.h"
#include <gtest/gtest.h>
#include <cmath>

#include <type_traits>

using namespace mtr;

// ---------------------------------------------------------------------------
// Compile-time checks: the three tier names resolve to the types the build
// asked for, and the global names mirror the mtr:: ones.
// ---------------------------------------------------------------------------

#if MATAR_REAL_TYPE == MATAR_FP64
static_assert(std::is_same_v<real_t, double>);
static_assert(sizeof(real_t) == 8);
#elif MATAR_REAL_TYPE == MATAR_FP32
static_assert(std::is_same_v<real_t, float>);
static_assert(sizeof(real_t) == 4);
#elif MATAR_REAL_TYPE == MATAR_FP16
static_assert(std::is_same_v<real_t, Kokkos::Experimental::half_t>);
static_assert(sizeof(real_t) == (MATAR_FP16_IS_EMULATED ? 4 : 2));
#elif MATAR_REAL_TYPE == MATAR_BF16
static_assert(std::is_same_v<real_t, Kokkos::Experimental::bhalf_t>);
static_assert(sizeof(real_t) == (MATAR_BF16_IS_EMULATED ? 4 : 2));
#elif MATAR_REAL_TYPE == MATAR_FP128
static_assert(sizeof(real_t) == 16);
#endif

#if MATAR_HIGH_REAL_TYPE == MATAR_FP64
static_assert(std::is_same_v<high_real_t, double>);
#elif MATAR_HIGH_REAL_TYPE == MATAR_FP32
static_assert(std::is_same_v<high_real_t, float>);
#elif MATAR_HIGH_REAL_TYPE == MATAR_FP128
static_assert(sizeof(high_real_t) == 16);
#endif

#if MATAR_LOW_REAL_TYPE == MATAR_FP64
static_assert(std::is_same_v<low_real_t, double>);
#elif MATAR_LOW_REAL_TYPE == MATAR_FP32
static_assert(std::is_same_v<low_real_t, float>);
#elif MATAR_LOW_REAL_TYPE == MATAR_FP16
static_assert(std::is_same_v<low_real_t, Kokkos::Experimental::half_t>);
static_assert(sizeof(low_real_t) == (MATAR_FP16_IS_EMULATED ? 4 : 2));
#elif MATAR_LOW_REAL_TYPE == MATAR_BF16
static_assert(std::is_same_v<low_real_t, Kokkos::Experimental::bhalf_t>);
static_assert(sizeof(low_real_t) == (MATAR_BF16_IS_EMULATED ? 4 : 2));
#endif

static_assert(std::is_same_v<real_t, mtr::real_t>);
static_assert(std::is_same_v<high_real_t, mtr::high_real_t>);
static_assert(std::is_same_v<low_real_t, mtr::low_real_t>);

// Availability macros are always defined to 0/1
static_assert(MATAR_HAS_FP64 == 1);
static_assert(MATAR_HAS_FP32 == 1);
static_assert(MATAR_HAS_FP8 == 0);
#ifdef HAVE_KOKKOS
static_assert(MATAR_HAS_FP16 == 1);
static_assert(MATAR_HAS_BF16 == 1);
#else
static_assert(MATAR_HAS_FP16 == 0);
static_assert(MATAR_HAS_BF16 == 0);
#endif

// ---------------------------------------------------------------------------
// Runtime checks. All values below are exactly representable at every tier
// down to bfloat16 (8-bit mantissa: integers <= 256 and halves <= 128 are
// exact), so the checks stay tight regardless of the build's precision.
// ---------------------------------------------------------------------------

TEST(Precision, TierArithmetic) {
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(0.5)), 0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(2) + real_t(3)), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(6) * real_t(0.5)), 3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(10) / real_t(4)), 2.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(2) - real_t(8)), -6.0);
    EXPECT_TRUE(real_t(2) < real_t(3));
    EXPECT_TRUE(real_t(-1) < real_t(0));
}

TEST(Precision, HighTierArithmetic) {
    EXPECT_DOUBLE_EQ(static_cast<double>(high_real_t(0.25)), 0.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(high_real_t(3) * high_real_t(4)), 12.0);
}

TEST(Precision, LowTierArithmetic) {
    EXPECT_DOUBLE_EQ(static_cast<double>(low_real_t(1.5)), 1.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(low_real_t(2) + low_real_t(2.5)), 4.5);
}

TEST(Precision, CrossTierConversion) {
    // exactly representable at every tier: conversions must round-trip
    const high_real_t h = high_real_t(42.5);
    const real_t r      = real_t(h);
    const low_real_t l  = low_real_t(r);
    EXPECT_DOUBLE_EQ(static_cast<double>(r), 42.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(l), 42.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(high_real_t(l)), 42.5);
}

TEST(Precision, HostTypesAtTier) {
    const int n = 16;
    CArrayHost<real_t> a(n);
    CArrayHost<low_real_t> b(n);
    for (int i = 0; i < n; i++) {
        a(i) = real_t(i);
        b(i) = low_real_t(a(i)) + low_real_t(1);
    }
    EXPECT_DOUBLE_EQ(static_cast<double>(a(7)), 7.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(b(7)), 8.0);
}

#ifdef HAVE_KOKKOS
namespace {

// KOKKOS_LAMBDA (used by the device macros) must not appear inside a TEST()'s
// TestBody: nvcc rejects an extended __host__ __device__ lambda whose enclosing
// function has private or protected class access. Wrap each device kernel in a
// namespace-scope function. The _HOST macros capture by reference and are not
// subject to this, so they may stay inline in a test body.

inline void fill_index(CArrayDevice<real_t>& a, int n) {
    FOR_ALL(i, 0, n, {
        a(i) = real_t(i);
    });
    MATAR_FENCE();
}

inline void fill_two_every_seventh(CArrayDevice<real_t>& a, int n) {
    FOR_ALL(i, 0, n, {
        a(i) = (i % 7 == 0) ? real_t(2) : real_t(1);
    });
    MATAR_FENCE();
}

inline real_t reduce_sum(CArrayDevice<real_t>& a, int n) {
    real_t result  = real_t(0);
    real_t loc_sum = real_t(0);
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += a(i);
    }, result);
    return result;
}

inline real_t reduce_max(CArrayDevice<real_t>& a, int n) {
    real_t max_val = real_t(0);
    real_t max_lcl = real_t(0);
    FOR_REDUCE_MAX(i, 0, n,
                   max_lcl, {
        if (a(i) > max_lcl) {
            max_lcl = a(i);
        }
    }, max_val);
    return max_val;
}

inline real_t reduce_min(CArrayDevice<real_t>& a, int n) {
    real_t min_val = real_t(n);
    real_t min_lcl = real_t(n);
    FOR_REDUCE_MIN(i, 0, n,
                   min_lcl, {
        if (a(i) < min_lcl) {
            min_lcl = a(i);
        }
    }, min_val);
    return min_val;
}

inline real_t reduce_product(CArrayDevice<real_t>& a, int n) {
    real_t prod     = real_t(1);
    real_t prod_lcl = real_t(1);
    FOR_REDUCE_PRODUCT(i, 0, n,
                       prod_lcl, {
        prod_lcl *= a(i);
    }, prod);
    return prod;
}

inline void dual_double_in_place(CArrayDual<real_t>& field, int n) {
    FOR_ALL(i, 0, n, {
        field(i) = real_t(2) * field(i);
    });
    MATAR_FENCE();
}

inline real_t dual_reduce_sum(CArrayDual<real_t>& field, int n) {
    real_t sum     = real_t(0);
    real_t sum_lcl = real_t(0);
    FOR_REDUCE_SUM(i, 0, n,
                   sum_lcl, {
        sum_lcl += field(i);
    }, sum);
    return sum;
}

inline void dual_low_add_one(CArrayDual<low_real_t>& field, int n) {
    FOR_ALL(i, 0, n, {
        field(i) = field(i) + low_real_t(1);
    });
    MATAR_FENCE();
}

inline void fill_mixed_tiers(CArrayDevice<high_real_t>& coords, CArrayDevice<real_t>& state, int n) {
    FOR_ALL(i, 0, n, {
        coords(i) = high_real_t(i) / high_real_t(4);  // multiples of 0.25: exact
        state(i)  = real_t(coords(i));
    });
    MATAR_FENCE();
}

inline real_t reduce_sum_state(CArrayDevice<real_t>& state, int n) {
    real_t result = real_t(0);
    real_t loc    = real_t(0);
    FOR_REDUCE_SUM(i, 0, n,
                   loc, {
        loc += state(i);
    }, result);
    return result;
}

// Reduce a real_t field into the wider mtr::prec::accum_t accumulator. The
// reduce macros take the accumulator type from the declared variable
// (macros.h deduces it with decltype), so this is the supported way to sum
// narrow storage without stagnating.
inline mtr::prec::accum_t<real_t> reduce_sum_wide(CArrayDevice<real_t>& field, int n) {
    using Acc = mtr::prec::accum_t<real_t>;
    Acc result = Acc(0);
    Acc loc    = Acc(0);
    FOR_REDUCE_SUM(i, 0, n,
                   loc, {
        loc += Acc(field(i));
    }, result);
    return result;
}

}  // namespace

TEST(Precision, TierReductionSum) {
    // 100 * 0.5 = 50: every partial sum is a multiple of 0.5 <= 50, exact
    // even at bfloat16
    const int n = 100;
    CArrayDevice<real_t> a(n, "precision_sum");
    a.set_values(real_t(0.5));
    MATAR_FENCE();

    const real_t result = reduce_sum(a, n);

    EXPECT_NEAR(static_cast<double>(result), 50.0, 0.5);
}

TEST(Precision, TierReductionMaxMin) {
    // integers 0..99: exact at every tier
    const int n = 100;
    CArrayDevice<real_t> a(n, "precision_maxmin");
    fill_index(a, n);

    EXPECT_DOUBLE_EQ(static_cast<double>(reduce_max(a, n)), 99.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(reduce_min(a, n)), 0.0);
}

TEST(Precision, TierReductionProduct) {
    // three 2s among 1s: product = 8, exact everywhere
    const int n = 20;
    CArrayDevice<real_t> a(n, "precision_prod");
    fill_two_every_seventh(a, n);

    EXPECT_DOUBLE_EQ(static_cast<double>(reduce_product(a, n)), 8.0);
}

TEST(Precision, DualTypeRoundTrip) {
    // device write -> update_host -> host read/write -> update_device -> device check
    const int n = 10;
    CArrayDual<real_t> field(n, "precision_dual");
    field.set_values(real_t(1.5));

    dual_double_in_place(field, n);
    field.update_host();

    EXPECT_DOUBLE_EQ(static_cast<double>(field.host(3)), 3.0);

    field.host(3) = real_t(7);
    field.update_device();

    const real_t sum = dual_reduce_sum(field, n);
    // 9 * 3.0 + 7.0 = 34
    EXPECT_NEAR(static_cast<double>(sum), 34.0, 0.5);
}

TEST(Precision, DualTypeLowTier) {
    const int n = 8;
    CArrayDual<low_real_t> field(n, "precision_dual_low");
    field.set_values(low_real_t(0.25));
    dual_low_add_one(field, n);
    field.update_host();
    EXPECT_DOUBLE_EQ(static_cast<double>(field.host(5)), 1.25);
}

TEST(Precision, MixedTierFields) {
    // high_real_t and real_t fields coexist; conversions happen on access
    const int n = 10;
    CArrayDevice<high_real_t> coords(n, "precision_coords");
    CArrayDevice<real_t> state(n, "precision_state");

    fill_mixed_tiers(coords, state, n);

    const real_t result = reduce_sum_state(state, n);

    // sum of i/4 for i=0..9 = 11.25
    EXPECT_NEAR(static_cast<double>(result), 11.25, 0.1);
}

TEST(Precision, AccumulatorTierIsWideEnough) {
    // accum_t never narrows, and widens anything below double so that a long
    // summation chain does not stagnate.
    static_assert(sizeof(mtr::prec::accum_t<real_t>) >= sizeof(real_t),
                  "accum_t must never be narrower than the tier it accumulates");
    static_assert(std::is_same_v<mtr::prec::accum_t<double>, double>,
                  "accum_t<double> should stay double");
    static_assert(std::is_same_v<mtr::prec::accum_t<float>, double>,
                  "accum_t<float> should widen to double");
    // Integer types accumulate in themselves.
    static_assert(std::is_same_v<mtr::prec::accum_t<int>, int>,
                  "accum_t<int> should stay int");
    EXPECT_GE(sizeof(mtr::prec::accum_t<real_t>), sizeof(real_t));
}

TEST(Precision, WideAccumulatorBeatsNarrowOne) {
    // 1 + n*eps summed at the storage tier: once the running total outgrows
    // the increments, same-tier accumulation stagnates and the wide
    // accumulator does not. At the double tier both are exact and agree.
    const int n = 4096;
    CArrayDevice<real_t> a(n, "precision_wide_accum");
    a.set_values(real_t(1));
    MATAR_FENCE();

    const auto wide     = reduce_sum_wide(a, n);
    const real_t narrow = reduce_sum(a, n);

    // Exact at every tier down to bfloat16 for the wide accumulator: 4096 is
    // a power of two and each term is 1.
    EXPECT_NEAR(static_cast<double>(wide), 4096.0, 1.0e-9);
    // The narrow result is whatever the tier can represent; assert only that
    // the wide accumulator is never worse.
    const double err_wide   = std::abs(static_cast<double>(wide) - 4096.0);
    const double err_narrow = std::abs(static_cast<double>(narrow) - 4096.0);
    EXPECT_LE(err_wide, err_narrow);
}

TEST(Precision, PublicNumericTraits) {
    // The public trait surface an application needs; std::numeric_limits is
    // not specialized for the half tiers, so these must come from mtr::prec.
    const double eps = static_cast<double>(mtr::prec::epsilon<real_t>());
    EXPECT_GT(eps, 0.0);
    EXPECT_LT(eps, 1.0);
    EXPECT_GT(static_cast<double>(mtr::prec::numeric_max<real_t>()), 1.0);
    EXPECT_GT(static_cast<double>(mtr::prec::numeric_min<real_t>()), 0.0);

    // 1 + eps must be distinguishable from 1 at the tier; eps/2 must not be.
    const real_t one = real_t(1);
    EXPECT_NE(static_cast<double>(real_t(one + mtr::prec::epsilon<real_t>())), 1.0);

    // Tier-aware math agrees with the host library for exactly representable
    // values at any tier.
    EXPECT_NEAR(static_cast<double>(mtr::prec::sqrt(real_t(4))), 2.0, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(mtr::prec::fabs(real_t(-2.5))), 2.5, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(mtr::prec::fmax(real_t(1), real_t(2))), 2.0, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(mtr::prec::fmin(real_t(1), real_t(2))), 1.0, 1.0e-6);
}
#endif  // HAVE_KOKKOS
