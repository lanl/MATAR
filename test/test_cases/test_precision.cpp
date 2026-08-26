#include "matar.h"
#include <gtest/gtest.h>

#include <type_traits>

using namespace mtr;

// ---------------------------------------------------------------------------
// Compile-time checks: the three tier names resolve to the types the build
// asked for, and the global names mirror the mtr:: ones.
// ---------------------------------------------------------------------------

#if MATAR_REAL_TYPE == MATAR_FP64
static_assert(std::is_same_v<real_t, double>);
#elif MATAR_REAL_TYPE == MATAR_FP32
static_assert(std::is_same_v<real_t, float>);
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
#endif

static_assert(std::is_same_v<real_t, mtr::real_t>);
static_assert(std::is_same_v<high_real_t, mtr::high_real_t>);
static_assert(std::is_same_v<low_real_t, mtr::low_real_t>);

// Availability macros are always defined to 0/1
static_assert(MATAR_HAS_FP64 == 1);
static_assert(MATAR_HAS_FP32 == 1);
static_assert(MATAR_HAS_FP8 == 0);

// ---------------------------------------------------------------------------
// Runtime checks at the active tiers
// ---------------------------------------------------------------------------

TEST(Precision, TierArithmetic) {
    const real_t half_val = real_t(0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(half_val), 0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(real_t(2) + real_t(3)), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(high_real_t(0.25)), 0.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(low_real_t(1.5)), 1.5);
}

#ifdef HAVE_KOKKOS
TEST(Precision, TierReductionSum) {
    // 100 * 0.5: representable exactly at every tier down to half
    const int n = 100;
    CArrayDevice<real_t> a(n, "precision_sum");
    a.set_values(real_t(0.5));
    MATAR_FENCE();

    real_t result  = real_t(0);
    real_t loc_sum = real_t(0);
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += a(i);
    }, result);

    EXPECT_NEAR(static_cast<double>(result), 50.0, 0.5);
}

TEST(Precision, MixedTierFields) {
    // high_real_t and real_t fields coexist; conversions happen on access
    const int n = 10;
    CArrayDevice<high_real_t> coords(n, "precision_coords");
    CArrayDevice<real_t> state(n, "precision_state");

    FOR_ALL(i, 0, n, {
        coords(i) = high_real_t(i) / high_real_t(n);
        state(i)  = real_t(coords(i));
    });
    MATAR_FENCE();

    real_t result = real_t(0);
    real_t loc    = real_t(0);
    FOR_REDUCE_SUM(i, 0, n,
                   loc, {
        loc += state(i);
    }, result);

    // sum of i/n for i=0..9 = 4.5
    EXPECT_NEAR(static_cast<double>(result), 4.5, 0.1);
}
#endif  // HAVE_KOKKOS
