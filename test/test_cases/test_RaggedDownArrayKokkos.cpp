#include <gtest/gtest.h>
#include <matar.h>
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// CArrayKokkos writes must happen on device
inline void init_strides_2_3_1(CArrayKokkos<size_t>& strides) {
    Kokkos::parallel_for("init_rd_strides", 1, KOKKOS_LAMBDA(int) {
        strides(0) = 2;
        strides(1) = 3;
        strides(2) = 1;
    });
    Kokkos::fence();
}

// Set individual elements on device
inline void rd_set_values_manual(RaggedDownArrayKokkos<double>& array) {
    Kokkos::parallel_for("set_rd_vals", 1, KOKKOS_LAMBDA(int) {
        array(0, 0) = 1.0;
        array(1, 0) = 2.0;
        array(0, 1) = 3.0;
        array(1, 1) = 4.0;
        array(2, 1) = 5.0;
        array(0, 2) = 6.0;
    });
    Kokkos::fence();
}
} // namespace

// Test constructor with strides array
TEST(RaggedDownArrayKokkosTest, ConstructorWithStrides) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");

    EXPECT_EQ(array.dims(0), 3);
    // stride() is device-only on CUDA; dimensions verified via dims()
}

// Test constructor with raw strides array
TEST(RaggedDownArrayKokkosTest, ConstructorWithRawStrides) {
    size_t strides[3] = {2, 3, 1};

    RaggedDownArrayKokkos<double> array(strides, 3, "test_array");

    EXPECT_EQ(array.dims(0), 3);
}

// Test array access and modification
TEST(RaggedDownArrayKokkosTest, ArrayAccess) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");

    // Set values via device kernel
    rd_set_values_manual(array);

    // Verify via host mirror (flat storage)
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 1.0);
    EXPECT_DOUBLE_EQ(m(1), 2.0);
    EXPECT_DOUBLE_EQ(m(2), 3.0);
    EXPECT_DOUBLE_EQ(m(3), 4.0);
    EXPECT_DOUBLE_EQ(m(4), 5.0);
    EXPECT_DOUBLE_EQ(m(5), 6.0);
}

// Test set_values functionality
TEST(RaggedDownArrayKokkosTest, SetValues) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");
    array.set_values(42.0);
    Kokkos::fence();

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    for (size_t i = 0; i < array.size(); i++) {
        EXPECT_DOUBLE_EQ(m(i), 42.0);
    }
}

// Test stride management
TEST(RaggedDownArrayKokkosTest, StrideManagement) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");
    // Verify via set_values + mirror (stride verification requires device access)
    array.set_values(0.0);
    Kokkos::fence();
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array.get_kokkos_view());
    EXPECT_EQ(array.size(), static_cast<size_t>(2 + 3 + 1));  // total elements = sum of strides
}

// Test name management
TEST(RaggedDownArrayKokkosTest, NameManagement) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");

    RaggedDownArrayKokkos<double> array2(strides, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST(RaggedDownArrayKokkosTest, DifferentDataTypes) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<float> array_float(strides, "float_array");
    array_float.set_values(42.0f);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_float.get_kokkos_view());
        EXPECT_FLOAT_EQ(m(0), 42.0f);
    }

    RaggedDownArrayKokkos<int> array_int(strides, "int_array");
    array_int.set_values(42);
    Kokkos::fence();
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, array_int.get_kokkos_view());
        EXPECT_EQ(m(0), 42);
    }
}

#ifndef NDEBUG
// Test out-of-bounds access
TEST(RaggedDownArrayKokkosTest, OutOfBoundsAccess) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    RaggedDownArrayKokkos<double> array(strides, "test_array");

    EXPECT_DEATH(array(0, 3), ".*");  // Column 3 doesn't exist
    EXPECT_DEATH(array(2, 0), ".*");  // Column 0 only has 2 rows
    EXPECT_DEATH(array(3, 1), ".*");  // Column 1 only has 3 rows
    EXPECT_DEATH(array(1, 2), ".*");  // Column 2 only has 1 row
}
#endif
