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
#include <type_traits>
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

// ===========================================================================
// Coverage for the _HOST argument forms not exercised above: the 1D and 3D
// arities of every macro, the named (trailing kernel-name string) spellings,
// the _CLASS aliases, and the host macros paired with the 1-based Matrix
// types, the F (column-major) types, the View types and a ragged type.
// ===========================================================================

TEST(TestMacrosHost, FOR_ALL_HOST_NamedAllArities) {
    CArrayHost<int> a1(H0);
    CArrayHost<int> a2(H0, H1);
    CArrayHost<int> a3(H0, H1, H2);

    FOR_ALL_HOST(i, 0, H0, {
        a1(i) = i + 1;
    }, "named_1d");
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        a2(i, j) = (i + 1) * (j + 1);
    }, "named_2d");
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1,
                 k, 0, H2, {
        a3(i, j, k) = (i + 1) * (j + 1) * (k + 1);
    }, "named_3d");

    EXPECT_EQ(a1(H0 - 1), H0);
    EXPECT_EQ(a2(H0 - 1, H1 - 1), H0 * H1);
    EXPECT_EQ(a3(H0 - 1, H1 - 1, H2 - 1), H0 * H1 * H2);
}

TEST(TestMacrosHost, DO_ALL_HOST_AllArities) {
    CArrayHost<int> a1(H0 + 1);
    CArrayHost<int> a3(H0 + 1, H1 + 1, H2 + 1);

    DO_ALL_HOST(i, 1, H0, {
        a1(i) = i;
    });
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1,
                k, 1, H2, {
        a3(i, j, k) = i * 10000 + j * 100 + k;
    });

    EXPECT_EQ(a1(1), 1);
    EXPECT_EQ(a1(H0), H0);
    // the inclusive upper bound must be written, not one short of it
    EXPECT_EQ(a3(1, 1, 1), 10101);
    EXPECT_EQ(a3(H0, H1, H2), H0 * 10000 + H1 * 100 + H2);
}

TEST(TestMacrosHost, DO_ALL_HOST_Named) {
    CArrayHost<int> a1(H0 + 1);
    CArrayHost<int> a2(H0 + 1, H1 + 1);
    CArrayHost<int> a3(H0 + 1, H1 + 1, H2 + 1);

    DO_ALL_HOST(i, 1, H0, {
        a1(i) = -i;
    }, "do_named_1d");
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1, {
        a2(i, j) = i + j;
    }, "do_named_2d");
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1,
                k, 1, H2, {
        a3(i, j, k) = i + j + k;
    }, "do_named_3d");

    EXPECT_EQ(a1(H0), -H0);
    EXPECT_EQ(a2(H0, H1), H0 + H1);
    EXPECT_EQ(a3(H0, H1, H2), H0 + H1 + H2);
}

TEST(TestMacrosHost, RUN_HOST_Named) {
    int flag = 0;
    RUN_HOST({
        flag = 7;
    }, "run_host_named");
    EXPECT_EQ(flag, 7);
}

TEST(TestMacrosHost, FOR_Reductions_1D_and_3D) {
    const int n = H0 * H1 * H2;

    CArrayHost<int> flat(n);
    FOR_ALL_HOST(i, 0, n, {
        flat(i) = i + 1;  // 1..n, strictly positive
    });

    // --- 1D forms, plain and named ---
    int sum     = 0;
    int loc_sum = 0;
    FOR_REDUCE_SUM_HOST(i, 0, n,
                        loc_sum, {
        loc_sum += flat(i);
    }, sum);
    EXPECT_EQ(sum, n * (n + 1) / 2);

    int sum_named     = 0;
    int loc_sum_named = 0;
    FOR_REDUCE_SUM_HOST(i, 0, n,
                        loc_sum_named, {
        loc_sum_named += flat(i);
    }, sum_named, "rsum_named_1d");
    EXPECT_EQ(sum_named, sum);

    int max_val = 0;
    int loc_max = 0;
    FOR_REDUCE_MAX_HOST(i, 0, n,
                        loc_max, {
        if (flat(i) > loc_max) {
            loc_max = flat(i);
        }
    }, max_val);
    EXPECT_EQ(max_val, n);

    int min_val = n + 1;
    int loc_min = n + 1;
    FOR_REDUCE_MIN_HOST(i, 0, n,
                        loc_min, {
        if (flat(i) < loc_min) {
            loc_min = flat(i);
        }
    }, min_val);
    EXPECT_EQ(min_val, 1);

    // --- 3D forms over the same values, so the totals must match ---
    CArrayHost<int> cube(H0, H1, H2);
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1,
                 k, 0, H2, {
        cube(i, j, k) = i * H1 * H2 + j * H2 + k + 1;  // same 1..n multiset
    });

    int sum3     = 0;
    int loc_sum3 = 0;
    FOR_REDUCE_SUM_HOST(i, 0, H0,
                        j, 0, H1,
                        k, 0, H2,
                        loc_sum3, {
        loc_sum3 += cube(i, j, k);
    }, sum3);
    EXPECT_EQ(sum3, n * (n + 1) / 2);

    int max3     = 0;
    int loc_max3 = 0;
    FOR_REDUCE_MAX_HOST(i, 0, H0,
                        j, 0, H1,
                        k, 0, H2,
                        loc_max3, {
        if (cube(i, j, k) > loc_max3) {
            loc_max3 = cube(i, j, k);
        }
    }, max3, "rmax_named_3d");
    EXPECT_EQ(max3, n);

    int min3     = n + 1;
    int loc_min3 = n + 1;
    FOR_REDUCE_MIN_HOST(i, 0, H0,
                        j, 0, H1,
                        k, 0, H2,
                        loc_min3, {
        if (cube(i, j, k) < loc_min3) {
            loc_min3 = cube(i, j, k);
        }
    }, min3);
    EXPECT_EQ(min3, 1);

    // --- product, 2D and 3D ---
    CArrayHost<int> twos(2, 3);
    FOR_ALL_HOST(i, 0, 2,
                 j, 0, 3, {
        twos(i, j) = 2;
    });
    int prod2     = 1;
    int loc_prod2 = 1;
    FOR_REDUCE_PRODUCT_HOST(i, 0, 2,
                            j, 0, 3,
                            loc_prod2, {
        loc_prod2 *= twos(i, j);
    }, prod2);
    EXPECT_EQ(prod2, 64);  // 2^6
}

TEST(TestMacrosHost, DO_Reductions_2D_and_3D) {
    CArrayHost<int> a(H0 + 1, H1 + 1, H2 + 1);
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1,
                k, 1, H2, {
        a(i, j, k) = 1;
    });

    // an inclusive 3D range covers H0*H1*H2 cells; summing ones counts them
    int count     = 0;
    int loc_count = 0;
    DO_REDUCE_SUM_HOST(i, 1, H0,
                       j, 1, H1,
                       k, 1, H2,
                       loc_count, {
        loc_count += a(i, j, k);
    }, count);
    EXPECT_EQ(count, H0 * H1 * H2);

    int count2     = 0;
    int loc_count2 = 0;
    DO_REDUCE_SUM_HOST(i, 1, H0,
                       j, 1, H1,
                       loc_count2, {
        loc_count2 += a(i, j, 1);
    }, count2, "do_rsum_named_2d");
    EXPECT_EQ(count2, H0 * H1);

    // max/min over i*j so the 2D index range itself is checked
    CArrayHost<int> prodij(H0 + 1, H1 + 1);
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1, {
        prodij(i, j) = i * j;
    });

    int max_val = 0;
    int loc_max = 0;
    DO_REDUCE_MAX_HOST(i, 1, H0,
                       j, 1, H1,
                       loc_max, {
        if (prodij(i, j) > loc_max) {
            loc_max = prodij(i, j);
        }
    }, max_val);
    EXPECT_EQ(max_val, H0 * H1);

    int min_val = 1000;
    int loc_min = 1000;
    DO_REDUCE_MIN_HOST(i, 1, H0,
                       j, 1, H1,
                       k, 1, H2,
                       loc_min, {
        if (a(i, j, k) < loc_min) {
            loc_min = a(i, j, k);
        }
    }, min_val);
    EXPECT_EQ(min_val, 1);
}

namespace {
// The _HOST macros capture by reference, so `this` comes along on its own and
// the _CLASS spellings are plain aliases. Both forms must compile and behave
// identically inside a member function.
class HostMember {
public:
    HostMember(int n) : n_(n), data_(n) {}

    void fill_plain() {
        FOR_ALL_HOST(i, 0, n_, {
            data_(i) = i;
        });
    }

    void fill_class() {
        FOR_ALL_HOST_CLASS(i, 0, n_, {
            data_(i) = 2 * i;
        }, "member_fill_class");
    }

    void run_class(int value) {
        RUN_HOST_CLASS({
            scalar_ = value;
        });
    }

    int sum_class() {
        int total     = 0;
        int loc_total = 0;
        FOR_REDUCE_SUM_HOST_CLASS(i, 0, n_,
                                  loc_total, {
            loc_total += data_(i);
        }, total);
        return total;
    }

    int max_class() {
        int result  = 0;
        int loc_max = 0;
        FOR_REDUCE_MAX_HOST_CLASS(i, 0, n_,
                                  loc_max, {
            if (data_(i) > loc_max) {
                loc_max = data_(i);
            }
        }, result);
        return result;
    }

    int min_class() {
        int result  = 1 << 30;
        int loc_min = 1 << 30;
        FOR_REDUCE_MIN_HOST_CLASS(i, 0, n_,
                                  loc_min, {
            if (data_(i) < loc_min) {
                loc_min = data_(i);
            }
        }, result);
        return result;
    }

    int scalar() const { return scalar_; }
    int at(int i) const { return data_(i); }

private:
    int n_;
    int scalar_ = 0;
    CArrayHost<int> data_;
};
}  // namespace

TEST(TestMacrosHost, ClassAliasesInsideMemberFunctions) {
    const int n = 8;
    HostMember obj(n);

    obj.fill_plain();
    EXPECT_EQ(obj.at(n - 1), n - 1);
    EXPECT_EQ(obj.sum_class(), (n - 1) * n / 2);

    obj.fill_class();
    EXPECT_EQ(obj.at(n - 1), 2 * (n - 1));
    EXPECT_EQ(obj.sum_class(), (n - 1) * n);
    EXPECT_EQ(obj.max_class(), 2 * (n - 1));
    EXPECT_EQ(obj.min_class(), 0);

    obj.run_class(11);
    EXPECT_EQ(obj.scalar(), 11);
}

TEST(TestMacrosHost, MatrixAndFTypes) {
    // Matrix types are 1-based, so they pair with DO_ALL_HOST
    CMatrixHost<int> cm(H0, H1);
    DO_ALL_HOST(i, 1, H0,
                j, 1, H1, {
        cm(i, j) = i * 1000 + j;
    });
    EXPECT_EQ(cm(1, 1), 1001);
    EXPECT_EQ(cm(H0, H1), H0 * 1000 + H1);

    FMatrixHost<int> fm(H0, H1);
    DO_ALL_HOST(j, 1, H1,
                i, 1, H0, {
        fm(i, j) = i * 1000 + j;
    });
    EXPECT_EQ(fm(H0, H1), H0 * 1000 + H1);

    // F Array types are 0-based and column-major: nest so the FIRST index is
    // innermost, which is what the macro runs fastest.
    FArrayHost<int> fa(H0, H1, H2);
    FOR_ALL_HOST(k, 0, H2,
                 j, 0, H1,
                 i, 0, H0, {
        fa(i, j, k) = i + 10 * j + 100 * k;
    });
    EXPECT_EQ(fa(0, 0, 0), 0);
    EXPECT_EQ(fa(H0 - 1, H1 - 1, H2 - 1), (H0 - 1) + 10 * (H1 - 1) + 100 * (H2 - 1));
}

TEST(TestMacrosHost, ViewAndRaggedHostTypes) {
    // a View over host memory is host-accessible, so a host kernel may fill it
    CArrayHost<double> backing(H0, H1);
    ViewCArrayHost<double> view(&backing(0, 0), H0, H1);
    FOR_ALL_HOST(i, 0, H0,
                 j, 0, H1, {
        view(i, j) = 0.5 * (double)(i * H1 + j);
    }, "fill_view_host");
    EXPECT_DOUBLE_EQ(backing(H0 - 1, H1 - 1), 0.5 * (H0 * H1 - 1));

    // ragged host type: the outer index is parallel, the inner stride is a
    // plain serial loop inside the body
    CArrayHost<size_t> strides(3);
    strides(0) = 1;
    strides(1) = 3;
    strides(2) = 2;
    RaggedCArrayHost<int> ragged(strides);

    FOR_ALL_HOST(i, 0, 3, {
        for (size_t j = 0; j < ragged.stride(i); j++) {
            ragged(i, j) = (int)(10 * i + j);
        }
    });
    EXPECT_EQ(ragged(0, 0), 0);
    EXPECT_EQ(ragged(1, 2), 12);
    EXPECT_EQ(ragged(2, 1), 21);

    int total     = 0;
    int loc_total = 0;
    FOR_REDUCE_SUM_HOST(i, 0, 3,
                        loc_total, {
        for (size_t j = 0; j < ragged.stride(i); j++) {
            loc_total += ragged(i, j);
        }
    }, total);
    EXPECT_EQ(total, 0 + (10 + 11 + 12) + (20 + 21));
}

TEST(TestMacrosHost, EmptyRange) {
    CArrayHost<int> a(4);
    a.set_values(-1);

    FOR_ALL_HOST(i, 0, 0, {
        a(i) = 99;  // never runs
    });
    EXPECT_EQ(a(0), -1);

    // a sum over an empty range is the reducer's identity, zero
    int sum     = 123;
    int loc_sum = 0;
    FOR_REDUCE_SUM_HOST(i, 0, 0,
                        loc_sum, {
        loc_sum += a(i);
    }, sum);
    EXPECT_EQ(sum, 0);
}

#ifdef HAVE_KOKKOS
// The host macros must name a host execution space, whatever the device
// backend is. If this ever stops holding, every "host" kernel above is
// silently running somewhere else.
TEST(TestMacrosHost, HostExecSpaceIsHostAccessible) {
    static_assert(Kokkos::SpaceAccessibility<DefaultHostExecSpace, Kokkos::HostSpace>::accessible,
                  "DefaultHostExecSpace must be able to reach HostSpace");
    EXPECT_TRUE(DefaultHostExecSpace().concurrency() >= 1);

#if defined(HAVE_CUDA) || defined(HAVE_HIP) || defined(HAVE_SYCL)
    // on a GPU build the two spaces must be distinct, or there is no overlap
    static_assert(!std::is_same<DefaultExecSpace, DefaultHostExecSpace>::value, "a GPU device space and the host space cannot be the same space");
#endif
}

namespace {

inline void device_fill_dual(CArrayDual<double>& field, int n) {
    FOR_ALL(i, 0, n, {
        field(i) = 3.0 * (double)i;
    });
}

inline void device_double_dual(CArrayDual<double>& field, int n) {
    FOR_ALL(i, 0, n, {
        field(i) = 2.0 * field(i);
    });
}

inline void device_fill_fmatrix_dual(FMatrixDual<double>& field, int n1, int n2) {
    DO_ALL(j, 1, n2,
           i, 1, n1, {
        field(i, j) = (double)(i * 100 + j);
    });
}

inline void device_add_one(CArrayDevice<double>& a, int n) {
    FOR_ALL(i, 0, n, {
        a(i) = a(i) + 1.0;
    });
    // left unfenced: the caller overlaps host work with this kernel
}

inline double device_sum_double(CArrayDevice<double>& a, int n) {
    double sum     = 0.0;
    double loc_sum = 0.0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += a(i);
    }, sum);
    return sum;
}

}  // namespace

// The other direction of the documented handoff: the device writes, the host
// macros read, with update_host() and a fence in between.
TEST(TestMacrosHost, DualTypeDeviceToHost) {
    const int n = 32;
    CArrayDual<double> field(n, "dual_dev_to_host");

    device_fill_dual(field, n);
    MATAR_FENCE();  // the kernel must finish before its data is copied back
    field.update_host();

    double sum     = 0.0;
    double loc_sum = 0.0;
    FOR_REDUCE_SUM_HOST(i, 0, n,
                        loc_sum, {
        loc_sum += field.host(i);
    }, sum);
    EXPECT_DOUBLE_EQ(sum, 3.0 * (n - 1) * n / 2.0);

    // host-side scaling, pushed back to the device, doubled there, and read
    // back again: a full round trip through both macro families
    FOR_ALL_HOST(i, 0, n, {
        field.host(i) = field.host(i) + 1.0;
    }, "host_bump");
    field.update_device();

    device_double_dual(field, n);
    MATAR_FENCE();
    field.update_host();

    double sum2     = 0.0;
    double loc_sum2 = 0.0;
    FOR_REDUCE_SUM_HOST(i, 0, n,
                        loc_sum2, {
        loc_sum2 += field.host(i);
    }, sum2);
    // 2 * sum(3i + 1) = 2 * (3*n*(n-1)/2 + n)
    EXPECT_DOUBLE_EQ(sum2, 2.0 * (3.0 * (n - 1) * n / 2.0 + n));
}

// A 1-based Dual type driven by DO_ALL on the device and DO_ALL_HOST on the
// host, to pin the inclusive-bound forms of both families against each other.
TEST(TestMacrosHost, DualMatrixBothFamilies) {
    const int n1 = 3;
    const int n2 = 4;
    FMatrixDual<double> field(n1, n2, "dual_fmatrix");

    device_fill_fmatrix_dual(field, n1, n2);
    MATAR_FENCE();
    field.update_host();

    double max_val = 0.0;
    double loc_max = 0.0;
    DO_REDUCE_MAX_HOST(j, 1, n2,
                       i, 1, n1,
                       loc_max, {
        if (field.host(i, j) > loc_max) {
            loc_max = field.host(i, j);
        }
    }, max_val);
    EXPECT_DOUBLE_EQ(max_val, (double)(n1 * 100 + n2));

    DO_ALL_HOST(j, 1, n2,
                i, 1, n1, {
        field.host(i, j) = -field.host(i, j);
    });

    double min_val = 0.0;
    double loc_min = 0.0;
    DO_REDUCE_MIN_HOST(j, 1, n2,
                       i, 1, n1,
                       loc_min, {
        if (field.host(i, j) < loc_min) {
            loc_min = field.host(i, j);
        }
    }, min_val);
    EXPECT_DOUBLE_EQ(min_val, -(double)(n1 * 100 + n2));
}

// Host kernels interleaved with unfenced device kernels: the host side must be
// unaffected by device work still in flight, and vice versa.
TEST(TestMacrosHost, InterleavedWithoutCrossTalk) {
    const int n     = 4096;
    const int iters = 4;

    CArrayDevice<double> dev(n, "interleave_dev");
    CArrayHost<double> host(n);
    dev.set_values(0.0);
    host.set_values(0.0);
    MATAR_FENCE();

    double expected_host = 0.0;
    for (int it = 0; it < iters; it++) {
        device_add_one(dev, n);  // unfenced on purpose

        FOR_ALL_HOST(i, 0, n, {
            host(i) = host(i) + 2.0;
        }, "interleave_host");
        expected_host += 2.0;
    }

    MATAR_FENCE_DEVICE();

    EXPECT_DOUBLE_EQ(host(0), expected_host);
    EXPECT_DOUBLE_EQ(host(n - 1), expected_host);
    EXPECT_DOUBLE_EQ(device_sum_double(dev, n), (double)iters * n);
}
#endif  // HAVE_KOKKOS
