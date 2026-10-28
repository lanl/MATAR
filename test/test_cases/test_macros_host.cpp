// Tests for the _HOST parallel macro family (FOR_ALL_HOST, DO_ALL_HOST,
// RUN_HOST, and the host reductions).
//
// Unlike the device macros, these use [&] capture rather than KOKKOS_LAMBDA,
// so they can be written directly inside a TEST() body -- no free-function
// wrapper is needed.
//
// Dimensions are deliberately unequal so an index-order bug in a macro
// cannot hide behind a square extent.

#include "matar.h"
#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using namespace mtr;

namespace {
constexpr int H0 = 2;
constexpr int H1 = 3;
constexpr int H2 = 5;
}  // namespace

TEST(TestMacrosHost, FOR_ALL_HOST_1D_2D_3D) {
    CArrayHost<int> a1(H0);
    CArrayHost<int> a2(H0, H1);
    CArrayHost<int> a3(H0, H1, H2);

    FOR_ALL_HOST(i, 0, H0, {
        a1(i) = i;
    });
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        a2(i, j) = i * H1 + j;
    });
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1,
                 k, 0, H2, {
        a3(i, j, k) = i * H1 * H2 + j * H2 + k;
    });

    EXPECT_EQ(a1(1), 1);
    EXPECT_EQ(a2(1, 2), 1 * H1 + 2);
    EXPECT_EQ(a3(1, 2, 3), 1 * H1 * H2 + 2 * H2 + 3);
}

TEST(TestMacrosHost, FOR_ALL_HOST_Named) {
    CArrayHost<int> a(H0, H1);
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        a(i, j) = 7;
    }, "host_named_fill");
    EXPECT_EQ(a(0, 0), 7);
    EXPECT_EQ(a(H0 - 1, H1 - 1), 7);
}

TEST(TestMacrosHost, DO_ALL_HOST_InclusiveBounds) {
    // DO_* bounds are inclusive: 1..H0 needs H0+1 slots
    CArrayHost<int> a(H0 + 1, H1 + 1);
    a(H0, H1) = 0;
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1, {
        a(i, j) = i * 100 + j;
    });
    EXPECT_EQ(a(1, 1), 101);
    EXPECT_EQ(a(H0, H1), H0 * 100 + H1);
}

TEST(TestMacrosHost, RUN_HOST) {
    int flag = 0;
    RUN_HOST({
        flag = 99;
    });
    EXPECT_EQ(flag, 99);
}

TEST(TestMacrosHost, Reductions) {
    CArrayHost<int> a(H0, H1);
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        a(i, j) = i * H1 + j;  // values 0..H0*H1-1
    });

    const int n = H0 * H1;

    int sum     = 0;
    int loc_sum = 0;
    FOR_REDUCE_SUM_HOST(i, 0, H0,
                        j, 0, H1,
                        loc_sum, {
        loc_sum += a(i, j);
    }, sum);
    EXPECT_EQ(sum, n * (n - 1) / 2);

    // Negative values so a wrong reducer (a bare scalar result defaults to
    // Sum, whose identity is 0) cannot pass by coincidence.
    CArrayHost<int> neg(H0, H1);
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        neg(i, j) = -(i * H1 + j) - 1;  // -1 .. -n
    });

    int max_val = 0;
    int loc_max = 0;
    FOR_REDUCE_MAX_HOST(i, 0, H0,
                        j, 0, H1,
                        loc_max, {
        if (neg(i, j) > loc_max) {
            loc_max = neg(i, j);
        }
    }, max_val);
    EXPECT_EQ(max_val, -1);

    // strictly positive values: a Sum-identity bug would yield 0, not 1
    int min_val = 0;
    int loc_min = 0;
    FOR_REDUCE_MIN_HOST(i, 0, H0,
                        j, 0, H1,
                        loc_min, {
        if (a(i, j) + 1 < loc_min) {
            loc_min = a(i, j) + 1;
        }
    }, min_val);
    EXPECT_EQ(min_val, 1);

    CArrayHost<int> twos(4);
    FOR_ALL_HOST(i, 0, 4, {
        twos(i) = 2;
    });
    int prod     = 1;
    int loc_prod = 1;
    FOR_REDUCE_PRODUCT_HOST(i, 0, 4,
                            loc_prod, {
        loc_prod *= twos(i);
    }, prod);
    EXPECT_EQ(prod, 16);
}

TEST(TestMacrosHost, DO_Reductions_InclusiveBounds) {
    CArrayHost<int> a(H0 + 1);
    DO_ALL_HOST(i, 1, H0, {
        a(i) = i;
    });

    int sum     = 0;
    int loc_sum = 0;
    DO_REDUCE_SUM_HOST(i, 1, H0,
                       loc_sum, {
        loc_sum += a(i);
    }, sum);
    EXPECT_EQ(sum, H0 * (H0 + 1) / 2);

    int max_val = 0;
    int loc_max = 0;
    DO_REDUCE_MAX_HOST(i, 1, H0,
                       loc_max, {
        if (a(i) > loc_max) {
            loc_max = a(i);
        }
    }, max_val);
    EXPECT_EQ(max_val, H0);

    int min_val = 1000;
    int loc_min = 1000;
    DO_REDUCE_MIN_HOST(i, 1, H0,
                       loc_min, {
        if (a(i) < loc_min) {
            loc_min = a(i);
        }
    }, min_val);
    EXPECT_EQ(min_val, 1);
}

// The motivating capability: [&] capture lets a host kernel touch std::
// objects that could never be captured into a device lambda. Each iteration
// writes its own slot, so there is no race.
TEST(TestMacrosHost, CapturesHostOnlyObjects) {
    const int n = 8;
    std::vector<std::string> lines(n);
    const std::string prefix = "row-";

    FOR_ALL_HOST(i, 0, n, {
        std::ostringstream os;
        os << prefix << i;
        lines[i] = os.str();
    });

    EXPECT_EQ(lines[0], "row-0");
    EXPECT_EQ(lines[n - 1], "row-7");
}

#ifdef HAVE_KOKKOS
namespace {

// Device macros expand to KOKKOS_LAMBDA, and nvcc rejects an extended
// __host__ __device__ lambda whose enclosing function has private or protected
// class access -- which gtest's TestBody() does. Wrap the device kernels in
// namespace-scope functions. The _HOST macros capture by reference and are not
// subject to this restriction, so they stay inline in the test bodies.

inline double dual_reduce_sum(CArrayDual<double>& field, int n) {
    double sum     = 0.0;
    double loc_sum = 0.0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += field(i);
    }, sum);
    MATAR_FENCE();
    return sum;
}

inline void device_fill_double(CArrayDevice<int>& a, int n) {
    FOR_ALL(i, 0, n, {
        a(i) = i * 2;
    });
    // deliberately no fence: the caller overlaps host work with this kernel
}

inline int device_reduce_sum(CArrayDevice<int>& a, int n) {
    int sum     = 0;
    int loc_sum = 0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += a(i);
    }, sum);
    MATAR_FENCE();
    return sum;
}

// slice each element's 3x3 block inside the kernel and fill it there
inline void view_fill_slices(CArrayDevice<double>& A, int n_elem) {
    FOR_ALL(e, 0, n_elem, {
        ViewCArrayKokkos<double> slice(&A(e, 0, 0), 3, 3);
        slice.set_values(static_cast<double>(e));
    });
    MATAR_FENCE();
}

inline double reduce_sum_3d(CArrayDevice<double>& A, int n_elem) {
    double sum     = 0.0;
    double loc_sum = 0.0;
    FOR_REDUCE_SUM(e, 0, n_elem,
                   i, 0, 3,
                   j, 0, 3,
                   loc_sum, {
        loc_sum += A(e, i, j);
    }, sum);
    MATAR_FENCE();
    return sum;
}

}  // namespace

// Documented interop path: host macros operate on the .host() side of a dual
// type, then the result is pushed to the device.
TEST(TestMacrosHost, DualTypeHostSide) {
    const int n = 16;
    CArrayDual<double> field(n, "host_macro_dual");

    FOR_ALL_HOST(i, 0, n, {
        field.host(i) = static_cast<double>(i) * 0.5;
    });
    field.update_device();

    const double sum = dual_reduce_sum(field, n);

    // sum of 0.5*i for i=0..15
    EXPECT_DOUBLE_EQ(sum, 0.5 * (n - 1) * n / 2.0);
}

// Device and host kernels launched back to back with no fence between them.
// On a GPU build the device kernel is still in flight while the host kernel
// runs (that is the point of the feature); everywhere else this at least
// verifies the two compose correctly.
TEST(TestMacrosHost, ConcurrentHostAndDeviceWork) {
    const int n = 1024;
    CArrayDevice<int> device_side(n, "concurrent_device");
    CArrayHost<int> host_side(n);

    device_fill_double(device_side, n);
    // no fence above on purpose: host work proceeds while the device runs
    FOR_ALL_HOST(i, 0, n, {
        host_side(i) = i * 3;
    });
    MATAR_FENCE_HOST();

    EXPECT_EQ(host_side(n - 1), (n - 1) * 3);

    EXPECT_EQ(device_reduce_sum(device_side, n), (n - 1) * n);
}

// ---------------------------------------------------------------------------
// View set_values: a View wraps a caller-supplied pointer, so filling it is a
// serial loop that works wherever the pointer is valid -- including from
// inside a kernel, which is the View types' main use case (issue #146).
// ---------------------------------------------------------------------------

TEST(TestMacrosHost, ViewSetValuesInsideKernel) {
    const int n_elem = 4;
    CArrayDevice<double> A(n_elem, 3, 3, "view_sv_A");
    A.set_values(-1.0);
    MATAR_FENCE();

    view_fill_slices(A, n_elem);

    const double sum = reduce_sum_3d(A, n_elem);

    // each 3x3 block holds its element index: 9 * (0+1+2+3)
    EXPECT_DOUBLE_EQ(sum, 9.0 * (0 + 1 + 2 + 3));
}

TEST(TestMacrosHost, ViewSetValuesOnHostMemory) {
    CArrayHost<double> backing(4, 4);
    ViewCArrayKokkos<double> view(&backing(0, 0), 4, 4);
    view.set_values(2.5);

    EXPECT_DOUBLE_EQ(backing(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(backing(3, 3), 2.5);

    // a View over a sub-block writes only that block
    ViewCArrayKokkos<double> row(&backing(1, 0), 4);
    row.set_values(9.0);
    EXPECT_DOUBLE_EQ(backing(1, 2), 9.0);
    EXPECT_DOUBLE_EQ(backing(2, 2), 2.5);
}
#endif  // HAVE_KOKKOS
