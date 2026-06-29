#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// CArrayKokkos writes must happen on device; wrap in a kernel
inline void init_strides_2_3_1(CArrayKokkos<size_t>& strides) {
    Kokkos::parallel_for("init_rr_strides", 1, KOKKOS_LAMBDA(int) {
        strides(0) = 2;
        strides(1) = 3;
        strides(2) = 1;
    });
    Kokkos::fence();
}

// Set individual array elements on device
inline void rr_set_values_manual(RaggedRightArrayKokkos<double>& array) {
    Kokkos::parallel_for("set_rr_vals", 1, KOKKOS_LAMBDA(int) {
        array(0, 0) = 1.0;
        array(0, 1) = 2.0;
        array(1, 0) = 3.0;
        array(1, 1) = 4.0;
        array(1, 2) = 5.0;
        array(2, 0) = 6.0;
    });
    Kokkos::fence();
}
} // namespace

// Test constructor with strides array
TEST(RaggedRightArrayKokkosTest, ConstructorWithStrides) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test constructor with raw strides array
TEST(RaggedRightArrayKokkosTest, ConstructorWithRawStrides) {
    size_t strides[3] = {2, 3, 1};

    RaggedRightArrayKokkos<double> array(strides, 3, "test_array");

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test array access and modification
TEST(RaggedRightArrayKokkosTest, ArrayAccess) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");

    // Set values via device kernel
    rr_set_values_manual(array);

    // Verify via host mirror
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    // Flat layout: elements are laid out by row; row 0 has 2 elements (flat 0,1),
    // row 1 has 3 elements (flat 2,3,4), row 2 has 1 element (flat 5)
    EXPECT_DOUBLE_EQ(m(0), 1.0);
    EXPECT_DOUBLE_EQ(m(1), 2.0);
    EXPECT_DOUBLE_EQ(m(2), 3.0);
    EXPECT_DOUBLE_EQ(m(3), 4.0);
    EXPECT_DOUBLE_EQ(m(4), 5.0);
    EXPECT_DOUBLE_EQ(m(5), 6.0);
}

// Test set_values functionality
TEST(RaggedRightArrayKokkosTest, SetValues) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");
    array.set_values(42.0);
    Kokkos::fence();

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    for (size_t i = 0; i < array.size(); i++) {
        EXPECT_DOUBLE_EQ(m(i), 42.0);
    }
}

// Test stride management
TEST(RaggedRightArrayKokkosTest, StrideManagement) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");

    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test name management
TEST(RaggedRightArrayKokkosTest, NameManagement) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");

    RaggedRightArrayKokkos<double> array2(strides, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST(RaggedRightArrayKokkosTest, DifferentDataTypes) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<float> array_float(strides, "float_array");
    array_float.set_values(42.0f);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_float.get_kokkos_view());
        EXPECT_FLOAT_EQ(m(0), 42.0f);
    }

    RaggedRightArrayKokkos<int> array_int(strides, "int_array");
    array_int.set_values(42);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_int.get_kokkos_view());
        EXPECT_EQ(m(0), 42);
    }
}

#ifndef NDEBUG
// Test out-of-bounds access
TEST(RaggedRightArrayKokkosTest, OutOfBoundsAccess) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedRightArrayKokkos<double> array(strides, "test_array");

    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    EXPECT_DEATH(array(0, 2), ".*");  // Row 0 only has 2 columns
    EXPECT_DEATH(array(1, 3), ".*");  // Row 1 only has 3 columns
    EXPECT_DEATH(array(2, 1), ".*");  // Row 2 only has 1 column
}
#endif
