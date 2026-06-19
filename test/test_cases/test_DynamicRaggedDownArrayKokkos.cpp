#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// RUN kernels cannot live inside TEST() — nvcc rejects KOKKOS_LAMBDA in the
// private TestBody().  Each free function here wraps one RUN block so the
// lambda is at namespace scope.
template<typename T>
inline void drd_init_strides(DynamicRaggedDownArrayKokkos<T>& array) {
    RUN({
        array.stride(0) = 1;
        array.stride(1) = 3;
        array.stride(2) = 2;
    });
}
} // namespace

// Test default constructor and basic initialization
TEST(DynamicRaggedDownArrayKokkosTest, DefaultConstructor) {
    DynamicRaggedDownArrayKokkos<double> array(3, 4, "test_array");

    drd_init_strides(array);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 4);

    EXPECT_EQ(array.stride(0), 1);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 2);
}

// Test set_values functionality
TEST(DynamicRaggedDownArrayKokkosTest, SetValues) {
    DynamicRaggedDownArrayKokkos<double> array(3, 4, "test_array");

    drd_init_strides(array);

    // Set all values to 42.0
    array.set_values(42.0);

    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 2), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 2), 42.0);
}

// Test set_values_sparse functionality
TEST(DynamicRaggedDownArrayKokkosTest, SetValuesSparse) {
    DynamicRaggedDownArrayKokkos<double> array(3, 4, "test_array");

    drd_init_strides(array);

    // Set sparse values
    array.set_values_sparse(42.0);

    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 2), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 2), 42.0);
}

// Test name management
TEST(DynamicRaggedDownArrayKokkosTest, NameManagement) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");

    DynamicRaggedDownArrayKokkos<double> array2(3, 2, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST(DynamicRaggedDownArrayKokkosTest, DifferentDataTypes) {
    DynamicRaggedDownArrayKokkos<float> array_float(3, 4, "float_array");
    drd_init_strides(array_float);
    array_float(0, 0) = 42.0f;
    EXPECT_FLOAT_EQ(array_float(0, 0), 42.0f);

    DynamicRaggedDownArrayKokkos<int> array_int(3, 4, "int_array");
    drd_init_strides(array_int);
    array_int(0, 0) = 42;
    EXPECT_EQ(array_int(0, 0), 42);
}

#ifndef NDEBUG
// Test out-of-bounds access
TEST(DynamicRaggedDownArrayKokkosTest, OutOfBoundsAccess) {
    DynamicRaggedDownArrayKokkos<double> array(3, 4, "test_array");

    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    EXPECT_DEATH(array(0, 2), ".*");  // Initial column size is 2
}
#endif

// Test get_kokkos_view
TEST(DynamicRaggedDownArrayKokkosTest, GetKokkosDualView) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");

    auto view = array.get_kokkos_view();

    EXPECT_TRUE(view.data() != nullptr);
}
