#ifndef PRECISION_H
#define PRECISION_H
/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
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

// ---------------------------------------------------------------------------
// MATAR compile-time-swappable floating-point precision system.
//
// User code deals with exactly three type names, whose meaning is fixed for
// the whole build by CMake flags:
//
//     real_t       the default working precision       (-DMATAR_REAL=...)
//     high_real_t  fields that must stay accurate      (-DMATAR_HIGH_REAL=...)
//     low_real_t   tolerant / bulk-storage fields      (-DMATAR_LOW_REAL=...)
//
// Values: double (default) | float | half | bfloat16 | quad.
//
// Non-Kokkos builds support only double and float. half/bfloat16 map to the
// Kokkos types (native 16-bit on CUDA/HIP/SYCL; float-backed elsewhere,
// reported by MATAR_FP16_IS_EMULATED / MATAR_BF16_IS_EMULATED). quad is
// __float128, host backends only, requiring Kokkos_ENABLE_LIBQUADMATH.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

// ---------------------------------------------------------------------------
// Precision codes. CMake passes e.g. -DMATAR_REAL_TYPE=MATAR_FP32; the token
// expands against these definitions at the tier ladders below.
// MATAR_FP8_* codes are reserved for future storage-only FP8 support.
// ---------------------------------------------------------------------------
#define MATAR_FP64 64
#define MATAR_FP32 32
#define MATAR_FP16 16
#define MATAR_BF16 17
#define MATAR_FP128 128
#define MATAR_FP8_E4M3 8
#define MATAR_FP8_E5M2 9

// Tiers not set by the build default to double
#ifndef MATAR_REAL_TYPE
#define MATAR_REAL_TYPE MATAR_FP64
#endif
#ifndef MATAR_HIGH_REAL_TYPE
#define MATAR_HIGH_REAL_TYPE MATAR_FP64
#endif
#ifndef MATAR_LOW_REAL_TYPE
#define MATAR_LOW_REAL_TYPE MATAR_FP64
#endif

#ifdef HAVE_KOKKOS
#define MATAR_PRECISION_FUNC KOKKOS_INLINE_FUNCTION
#else
#define MATAR_PRECISION_FUNC inline
#endif

// ---------------------------------------------------------------------------
// Availability macros (always defined to 0 or 1)
// ---------------------------------------------------------------------------
#define MATAR_HAS_FP64 1
#define MATAR_HAS_FP32 1
#define MATAR_HAS_FP8 0  // reserved; no vendor-portable arithmetic exists

#ifdef HAVE_KOKKOS
#define MATAR_HAS_FP16 1
#define MATAR_HAS_BF16 1
#if KOKKOS_HALF_T_IS_FLOAT
#define MATAR_FP16_IS_EMULATED 1
#else
#define MATAR_FP16_IS_EMULATED 0
#endif
#if KOKKOS_BHALF_T_IS_FLOAT
#define MATAR_BF16_IS_EMULATED 1
#else
#define MATAR_BF16_IS_EMULATED 0
#endif
#else  // no Kokkos: only double and float
#define MATAR_HAS_FP16 0
#define MATAR_HAS_BF16 0
#endif

#if defined(HAVE_KOKKOS) && defined(KOKKOS_ENABLE_LIBQUADMATH) && defined(__SIZEOF_FLOAT128__) && !defined(HAVE_CUDA) && !defined(HAVE_HIP)
#include <quadmath.h>
#define MATAR_HAS_FP128 1
#else
#define MATAR_HAS_FP128 0
#endif

// ---------------------------------------------------------------------------
// Tier selection ladders
// ---------------------------------------------------------------------------
namespace mtr {

#if MATAR_REAL_TYPE == MATAR_FP64
using real_t = double;
#elif MATAR_REAL_TYPE == MATAR_FP32
using real_t = float;
#elif MATAR_REAL_TYPE == MATAR_FP16
#if !MATAR_HAS_FP16
#error "MATAR_REAL=half requires a Kokkos build (no-Kokkos builds support only double and float)"
#endif
using real_t = Kokkos::Experimental::half_t;
#elif MATAR_REAL_TYPE == MATAR_BF16
#if !MATAR_HAS_BF16
#error "MATAR_REAL=bfloat16 requires a Kokkos build (no-Kokkos builds support only double and float)"
#endif
using real_t = Kokkos::Experimental::bhalf_t;
#elif MATAR_REAL_TYPE == MATAR_FP128
#if !MATAR_HAS_FP128
#error "MATAR_REAL=quad requires __float128 on a host backend with Kokkos_ENABLE_LIBQUADMATH"
#endif
using real_t = __float128;
#else
#error "Unrecognized MATAR_REAL_TYPE (FP8 codes are reserved and not yet supported)"
#endif

#if MATAR_HIGH_REAL_TYPE == MATAR_FP64
using high_real_t = double;
#elif MATAR_HIGH_REAL_TYPE == MATAR_FP32
using high_real_t = float;
#elif MATAR_HIGH_REAL_TYPE == MATAR_FP128
#if !MATAR_HAS_FP128
#error "MATAR_HIGH_REAL=quad requires __float128 on a host backend with Kokkos_ENABLE_LIBQUADMATH"
#endif
using high_real_t = __float128;
#else
#error "MATAR_HIGH_REAL_TYPE must be double, float, or quad (the high-precision tier is never below float)"
#endif

#if MATAR_LOW_REAL_TYPE == MATAR_FP64
using low_real_t = double;
#elif MATAR_LOW_REAL_TYPE == MATAR_FP32
using low_real_t = float;
#elif MATAR_LOW_REAL_TYPE == MATAR_FP16
#if !MATAR_HAS_FP16
#error "MATAR_LOW_REAL=half requires a Kokkos build (no-Kokkos builds support only double and float)"
#endif
using low_real_t = Kokkos::Experimental::half_t;
#elif MATAR_LOW_REAL_TYPE == MATAR_BF16
#if !MATAR_HAS_BF16
#error "MATAR_LOW_REAL=bfloat16 requires a Kokkos build (no-Kokkos builds support only double and float)"
#endif
using low_real_t = Kokkos::Experimental::bhalf_t;
#else
#error "Unrecognized MATAR_LOW_REAL_TYPE (FP8 codes are reserved and not yet supported)"
#endif

// ---------------------------------------------------------------------------
// Internal helpers (MATAR implementation detail, not user-facing API):
// half_t-aware math and numeric traits used by the solvers. Bare ::fabs etc.
// have no half overloads and silently promote to double, and
// std::numeric_limits is not specialized for half_t/bhalf_t/__float128.
// ---------------------------------------------------------------------------
namespace impl {

#ifdef HAVE_KOKKOS
template <typename T>
MATAR_PRECISION_FUNC T fabs(T x) {
    return Kokkos::fabs(x);
}
template <typename T>
MATAR_PRECISION_FUNC T sqrt(T x) {
    return Kokkos::sqrt(x);
}
template <typename T>
MATAR_PRECISION_FUNC T fmax(T x, T y) {
    return Kokkos::fmax(x, y);
}
template <typename T>
MATAR_PRECISION_FUNC T fmin(T x, T y) {
    return Kokkos::fmin(x, y);
}
#else
template <typename T>
MATAR_PRECISION_FUNC T fabs(T x) {
    return std::fabs(x);
}
template <typename T>
MATAR_PRECISION_FUNC T sqrt(T x) {
    return std::sqrt(x);
}
template <typename T>
MATAR_PRECISION_FUNC T fmax(T x, T y) {
    return std::fmax(x, y);
}
template <typename T>
MATAR_PRECISION_FUNC T fmin(T x, T y) {
    return std::fmin(x, y);
}
#endif

// Kokkos (and std) math functions have no __float128 overloads; use quadmath.
// Non-template overloads win overload resolution over the templates above.
#if MATAR_HAS_FP128
MATAR_PRECISION_FUNC __float128 fabs(__float128 x) { return fabsq(x); }
MATAR_PRECISION_FUNC __float128 sqrt(__float128 x) { return sqrtq(x); }
MATAR_PRECISION_FUNC __float128 fmax(__float128 x, __float128 y) { return fmaxq(x, y); }
MATAR_PRECISION_FUNC __float128 fmin(__float128 x, __float128 y) { return fminq(x, y); }
#endif

template <typename T>
MATAR_PRECISION_FUNC T epsilon() {
    return std::numeric_limits<T>::epsilon();
}
#if MATAR_HAS_FP16 && !MATAR_FP16_IS_EMULATED
template <>
MATAR_PRECISION_FUNC Kokkos::Experimental::half_t epsilon<Kokkos::Experimental::half_t>() {
    return Kokkos::Experimental::epsilon_v<Kokkos::Experimental::half_t>;
}
#endif
#if MATAR_HAS_BF16 && !MATAR_BF16_IS_EMULATED
template <>
MATAR_PRECISION_FUNC Kokkos::Experimental::bhalf_t epsilon<Kokkos::Experimental::bhalf_t>() {
    return Kokkos::Experimental::epsilon_v<Kokkos::Experimental::bhalf_t>;
}
#endif
#if MATAR_HAS_FP128
template <>
MATAR_PRECISION_FUNC __float128 epsilon<__float128>() {
    return FLT128_EPSILON;
}
#endif

}  // namespace impl
}  // namespace mtr

// ---------------------------------------------------------------------------
// The tier names user code writes. Their meaning is fixed by the CMake flags.
// ---------------------------------------------------------------------------
using real_t      = mtr::real_t;
using high_real_t = mtr::high_real_t;
using low_real_t  = mtr::low_real_t;

// ---------------------------------------------------------------------------
// Kokkos < 5.2 does not ship reduction_identity for the native half types
// (5.2+ provides them in Kokkos_Half_ReductionIdentity.hpp, guarded the same
// way), so FOR_REDUCE_* over a half tier would not compile there.
// ---------------------------------------------------------------------------
#if defined(HAVE_KOKKOS) && (KOKKOS_VERSION < 50200)
#if MATAR_HAS_FP16 && !MATAR_FP16_IS_EMULATED
template <>
struct Kokkos::reduction_identity<Kokkos::Experimental::half_t> {
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::half_t sum() { return Kokkos::Experimental::half_t(0.0f); }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::half_t prod() { return Kokkos::Experimental::half_t(1.0f); }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::half_t max() { return Kokkos::Experimental::finite_min_v<Kokkos::Experimental::half_t>; }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::half_t min() { return Kokkos::Experimental::finite_max_v<Kokkos::Experimental::half_t>; }
};
#endif
#if MATAR_HAS_BF16 && !MATAR_BF16_IS_EMULATED
template <>
struct Kokkos::reduction_identity<Kokkos::Experimental::bhalf_t> {
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::bhalf_t sum() { return Kokkos::Experimental::bhalf_t(0.0f); }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::bhalf_t prod() { return Kokkos::Experimental::bhalf_t(1.0f); }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::bhalf_t max() {
        return Kokkos::Experimental::finite_min_v<Kokkos::Experimental::bhalf_t>;
    }
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Experimental::bhalf_t min() {
        return Kokkos::Experimental::finite_max_v<Kokkos::Experimental::bhalf_t>;
    }
};
#endif
#endif  // HAVE_KOKKOS && KOKKOS_VERSION < 50200

// Kokkos provides no reduction_identity for __float128 at all, so quad-tier
// FOR_REDUCE_* would not compile without this.
#if defined(HAVE_KOKKOS) && MATAR_HAS_FP128
template <>
struct Kokkos::reduction_identity<__float128> {
    KOKKOS_FORCEINLINE_FUNCTION static __float128 sum() { return __float128(0); }
    KOKKOS_FORCEINLINE_FUNCTION static __float128 prod() { return __float128(1); }
    KOKKOS_FORCEINLINE_FUNCTION static __float128 max() { return -FLT128_MAX; }
    KOKKOS_FORCEINLINE_FUNCTION static __float128 min() { return FLT128_MAX; }
};
#endif  // HAVE_KOKKOS && MATAR_HAS_FP128

#endif  // PRECISION_H
