// Tests pinning the set_values contract after the Kokkos::deep_copy refactor.
//
// The pre-existing per-type set_values tests all call update_host() before
// checking .host(i), so they cannot tell whether set_values itself populated
// the host side. These tests check the properties those cannot.

#include "matar.h"
#include <gtest/gtest.h>

using namespace mtr;

namespace {

// Free functions: KOKKOS_LAMBDA must not appear in a TEST()'s private TestBody.
template <typename ArrType>
double device_sum(ArrType& a, int n) {
    double sum     = 0.0;
    double loc_sum = 0.0;
    FOR_REDUCE_SUM(i, 0, n,
                   loc_sum, {
        loc_sum += a(i);
    }, sum);
    MATAR_FENCE();
    return sum;
}

// stride() is a device accessor, so strides are set inside a device kernel
inline void sparse_init_strides(DynamicRaggedRightArrayKokkos<double>& a, int dim1) {
    RUN({
        for (int i = 0; i < dim1; i++) {
            a.stride(i) = i + 1;  // row i holds i+1 live entries
        }
    });
    MATAR_FENCE();
}

}  // namespace

// ---------------------------------------------------------------------------
// Dual types: both sides hold the value WITHOUT an explicit update_host().
//
// NOTE: on a CPU-only build the DualView's host and device views alias the same
// allocation, so this passes trivially. It only distinguishes the two sides on a
// GPU build -- see the GPU checklist.
// ---------------------------------------------------------------------------

TEST(SetValuesDeepCopy, DualBothSidesWithoutSync) {
    const int n = 64;
    DCArrayKokkos<double> A(n, "dual_both_sides");
    A.set_values(3.5);
    // deliberately NO update_host() here
    for (int i = 0; i < n; i++) {
        EXPECT_DOUBLE_EQ(A.host(i), 3.5);
    }
    EXPECT_DOUBLE_EQ(device_sum(A, n), 3.5 * n);
}

TEST(SetValuesDeepCopy, DualMatrixBothSidesWithoutSync) {
    const int n = 8;
    DCMatrixKokkos<double> M(n, n, "dual_matrix_both_sides");
    M.set_values(2.25);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            EXPECT_DOUBLE_EQ(M.host(i, j), 2.25);
        }
    }
}

// After set_values both sides agree, so the DualView must report no pending sync.
TEST(SetValuesDeepCopy, DualLeavesNoPendingSync) {
    const int n = 32;
    DCArrayKokkos<double> A(n, "dual_sync_state");
    A.set_values(1.0);

    auto dv = A.get_kokkos_dual_view();
    EXPECT_FALSE(dv.need_sync_host());
    EXPECT_FALSE(dv.need_sync_device());
}

// Filling both sides must not leave flags in a state where the next update
// trips DualView's concurrent-modification abort.
TEST(SetValuesDeepCopy, DualUpdatesAfterSetValuesDoNotAbort) {
    const int n = 16;
    DCArrayKokkos<double> A(n, "dual_no_abort");
    A.set_values(7.0);
    A.update_host();    // must not abort
    A.update_device();  // must not abort
    EXPECT_DOUBLE_EQ(A.host(0), 7.0);
    EXPECT_DOUBLE_EQ(device_sum(A, n), 7.0 * n);
}

// ---------------------------------------------------------------------------
// DView* types: the host view wraps the CALLER's buffer, which set_values now
// writes directly.
// ---------------------------------------------------------------------------

TEST(SetValuesDeepCopy, DViewWritesCallerHostBuffer) {
    const int n = 16;
    std::vector<double> caller(n, -1.0);
    DViewCArrayKokkos<double> A(caller.data(), n);

    A.set_values(4.0);

    for (int i = 0; i < n; i++) {
        EXPECT_DOUBLE_EQ(caller[i], 4.0);  // caller's own memory was filled
    }
    EXPECT_DOUBLE_EQ(device_sum(A, n), 4.0 * n);
}

// ---------------------------------------------------------------------------
// Dynamic types: the fill now covers the whole allocation, and `count` still
// updates the logical size.
// ---------------------------------------------------------------------------

TEST(SetValuesDeepCopy, DynamicArrayFillsWholeAllocation) {
    const int cap = 32;
    DynamicArrayKokkos<double> A(cap, "dyn_whole_alloc");
    A.set_values(9.0, 4);  // logical size 4, but the whole capacity is filled

    // count still sets the logical size; the allocation (capacity) is unchanged
    EXPECT_EQ(A.dims(0), 4);
    EXPECT_EQ(A.dims_max(0), static_cast<size_t>(cap));

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.get_kokkos_view());
    for (int i = 0; i < cap; i++) {
        EXPECT_DOUBLE_EQ(m(i), 9.0) << "index " << i << " past the live range must still be filled";
    }
}

// ---------------------------------------------------------------------------
// set_values_sparse must touch only the strided entries -- this fails against
// the old implementation, which was a byte-identical copy of set_values.
// ---------------------------------------------------------------------------

TEST(SetValuesDeepCopy, SparseFillsOnlyStridedEntries) {
    const int dim1 = 3;
    const int dim2 = 5;
    DynamicRaggedRightArrayKokkos<double> A(dim1, dim2, "sparse_right");

    sparse_init_strides(A, dim1);

    A.set_values(0.0);         // whole buffer to a known baseline
    A.set_values_sparse(5.0);  // only the live entries
    MATAR_FENCE();

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.get_kokkos_view());
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            const double expect = (j < i + 1) ? 5.0 : 0.0;
            EXPECT_DOUBLE_EQ(m(j + i * dim2), expect) << "row " << i << " col " << j;
        }
    }
}
