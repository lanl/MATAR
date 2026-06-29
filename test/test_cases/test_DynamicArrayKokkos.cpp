#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// Helper: set two individual elements on device (KOKKOS_LAMBDA must not be in TEST body on NVCC)
inline void set_elements_0_4(DynamicArrayKokkos<double>& array, double v0, double v4) {
    Kokkos::parallel_for("set_elems", 1, KOKKOS_LAMBDA(int) {
        array(0) = v0;
        array(4) = v4;
    });
    Kokkos::fence();
}
} // namespace

// Test default constructor
TEST(DynamicArrayKokkosTest, DefaultConstructor) {
    DynamicArrayKokkos<double> array;
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims_max(0), 0);
    EXPECT_EQ(array.order(), 0);
}

// Test constructor with initial size
TEST(DynamicArrayKokkosTest, ConstructorWithSize) {
    DynamicArrayKokkos<double> array(10, "test_array");
    EXPECT_EQ(array.size(), 10);
    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims_max(0), 10);
    EXPECT_EQ(array.order(), 1);
    EXPECT_EQ(array.get_name(), "test_array");
}

// Test push_back functionality
TEST(DynamicArrayKokkosTest, PushBack) {
    DynamicArrayKokkos<double> array(5, "test_array");

    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.dims(0), 0);

    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);

    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.dims(0), 3);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 1.0);
    EXPECT_DOUBLE_EQ(m(1), 2.0);
    EXPECT_DOUBLE_EQ(m(2), 3.0);
}

// Test pop_back functionality
TEST(DynamicArrayKokkosTest, PopBack) {
    DynamicArrayKokkos<double> array(5, "test_array");

    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);

    EXPECT_EQ(array.size(), 5);

    array.pop_back();
    EXPECT_EQ(array.dims(0), 2);

    array.pop_back();
    EXPECT_EQ(array.dims(0), 1);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 1.0);
}

// Test set_values functionality
TEST(DynamicArrayKokkosTest, SetValues) {
    DynamicArrayKokkos<double> array(5, "test_array");

    array.set_values(42.0, 5);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    for (size_t i = 0; i < array.dims(0); i++) {
        EXPECT_DOUBLE_EQ(m(i), 42.0);
    }
}

// Test dimension management
TEST(DynamicArrayKokkosTest, DimensionManagement) {
    DynamicArrayKokkos<double> array(10, "test_array");

    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims_max(0), 10);
    EXPECT_EQ(array.order(), 1);

    array.push_back(1.0);
    array.push_back(2.0);

    EXPECT_EQ(array.dims(0), 2);
    EXPECT_EQ(array.dims_max(0), 10);

    array.pop_back();
    array.pop_back();

    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims_max(0), 10);
}

// Test name management
TEST(DynamicArrayKokkosTest, NameManagement) {
    DynamicArrayKokkos<double> array(5, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");

    DynamicArrayKokkos<double> array2(5, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test size and extent
TEST(DynamicArrayKokkosTest, SizeAndExtent) {
    DynamicArrayKokkos<double> array(5, "test_array");

    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.extent(), 5);

    array.push_back(1.0);
    EXPECT_EQ(array.dims(0), 1);
    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.extent(), 5);

    array.pop_back();
    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.extent(), 5);
}

// Test array access and modification
TEST(DynamicArrayKokkosTest, ArrayAccess) {
    DynamicArrayKokkos<double> array(5, "test_array");

    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);
    array.push_back(4.0);
    array.push_back(5.0);

    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
        EXPECT_DOUBLE_EQ(m(0), 1.0);
        EXPECT_DOUBLE_EQ(m(1), 2.0);
        EXPECT_DOUBLE_EQ(m(2), 3.0);
        EXPECT_DOUBLE_EQ(m(3), 4.0);
        EXPECT_DOUBLE_EQ(m(4), 5.0);
    }

    // Modify individual elements via device kernel
    set_elements_0_4(array, 10.0, 50.0);

    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
        EXPECT_DOUBLE_EQ(m(0), 10.0);
        EXPECT_DOUBLE_EQ(m(4), 50.0);
    }
}
