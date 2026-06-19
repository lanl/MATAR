#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// RUN kernels cannot live inside TEST() — nvcc rejects KOKKOS_LAMBDA in the
// private TestBody().  Each free function here wraps one RUN block so the
// lambda is at namespace scope.
template<typename T>
inline void drr_init_strides(DynamicRaggedRightArrayKokkos<T>& array) {
    RUN({
        array.stride(0) = 1;
        array.stride(1) = 3;
        array.stride(2) = 2;
    });
}

template<typename T>
inline void drr_set_element_0_0(DynamicRaggedRightArrayKokkos<T>& array, T val) {
    Kokkos::parallel_for("set_elem", 1, KOKKOS_LAMBDA(int) {
        array(0, 0) = val;
    });
    Kokkos::fence();
}
} // namespace

//TO DO: Add following capability
// Test default constructor and basic initialization
// TEST(DynamicRaggedRightArrayKokkosTest, DefaultConstructor) { ... }
// TEST(DynamicRaggedRightArrayKokkosTest, PushBack) { ... }
// TEST(DynamicRaggedRightArrayKokkosTest, PopBack) { ... }

// Test set_values functionality
TEST(DynamicRaggedRightArrayKokkosTest, SetValues) {
    DynamicRaggedRightArrayKokkos<double> array(3, 5, "test_array");

    drr_init_strides(array);

    array.set_values(42.0);
    Kokkos::fence();

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    // flat layout: element [0,0] is at index 0; all elements should be 42.0
    EXPECT_DOUBLE_EQ(m(0), 42.0);
}

// Test set_values_sparse functionality
TEST(DynamicRaggedRightArrayKokkosTest, SetValuesSparse) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");

    drr_init_strides(array);

    array.set_values_sparse(42.0);
    Kokkos::fence();

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 42.0);
}

// Test name management
TEST(DynamicRaggedRightArrayKokkosTest, NameManagement) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");

    DynamicRaggedRightArrayKokkos<double> array2(3, 2, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST(DynamicRaggedRightArrayKokkosTest, DifferentDataTypes) {
    DynamicRaggedRightArrayKokkos<float> array_float(3, 2, "float_array");
    drr_init_strides(array_float);
    array_float.set_values(42.0f);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_float.get_kokkos_view());
        EXPECT_FLOAT_EQ(m(0), 42.0f);
    }

    DynamicRaggedRightArrayKokkos<int> array_int(3, 2, "int_array");
    drr_init_strides(array_int);
    array_int.set_values(42);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_int.get_kokkos_view());
        EXPECT_EQ(m(0), 42);
    }
}

#ifndef NDEBUG
// Test out-of-bounds access
TEST(DynamicRaggedRightArrayKokkosTest, OutOfBoundsAccess) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");

    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    EXPECT_DEATH(array(0, 2), ".*");  // Initial column size is 2
}
#endif

// Test get_kokkos_dual_view
TEST(DynamicRaggedRightArrayKokkosTest, GetKokkosDualView) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");

    auto view = array.get_kokkos_view();

    EXPECT_TRUE(view.data() != nullptr);
}
