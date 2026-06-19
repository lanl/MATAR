#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// Initialize CArrayKokkos strides on device from captured values
inline void init_strides_2_3_1(CArrayKokkos<size_t>& strides) {
    Kokkos::parallel_for("init_strides", 1, KOKKOS_LAMBDA(int) {
        strides(0) = 2;
        strides(1) = 3;
        strides(2) = 1;
    });
    Kokkos::fence();
}

// Set values on device via RUN kernel
inline void dragged_set_values(DRaggedRightArrayKokkos<double>& array,
                                int i0, int i1, double v00, double v01,
                                double v10, double v11, double v12, double v20) {
    RUN({
        array(i0, 0) = v00;
        array(i0, 1) = v01;
        array(i1, 0) = v10;
        array(i1, 1) = v11;
        array(i1, 2) = v12;
        array(2, 0)  = v20;
    });
}

inline void dragged_set_init_values(DRaggedRightArrayKokkos<double>& array) {
    RUN({
        array(0, 0) = 1.0;
        array(0, 1) = 2.0;
    });
}
} // namespace

// Test default constructor
TEST(DRaggedRightArrayKokkosTest, DefaultConstructor) {
    DRaggedRightArrayKokkos<double> array;
    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);
}

// Test constructor with CArrayKokkos strides
TEST(DRaggedRightArrayKokkosTest, ConstructorWithCArrayKokkos) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    DRaggedRightArrayKokkos<double> array(strides);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);

    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test constructor with DCArrayKokkos strides
TEST(DRaggedRightArrayKokkosTest, ConstructorWithDCArrayKokkos) {
    DCArrayKokkos<size_t> strides(3, "strides");
    strides.host(0) = 2;
    strides.host(1) = 3;
    strides.host(2) = 1;
    strides.update_device();

    DRaggedRightArrayKokkos<double> array(strides);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);

    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test constructor with raw array strides
TEST(DRaggedRightArrayKokkosTest, ConstructorWithRawArray) {
    size_t strides[3] = {2, 3, 1};

    DRaggedRightArrayKokkos<double> array(strides, 3);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);

    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test 2D array access
TEST(DRaggedRightArrayKokkosTest, ArrayAccess2D) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    DRaggedRightArrayKokkos<double> array(strides);
    array.set_values(0.0);

    // Set values on device via kernel
    dragged_set_values(array, 0, 1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    array.update_host();

    EXPECT_DOUBLE_EQ(array.host(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array.host(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array.host(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(array.host(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(array.host(2, 0), 6.0);
}

// Test host access
TEST(DRaggedRightArrayKokkosTest, HostAccess) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    DRaggedRightArrayKokkos<double> array(strides);
    array.set_values(0.0);

    // Set some test values on device
    dragged_set_init_values(array);

    // Update host
    array.update_host();

    // Check values on host
    EXPECT_DOUBLE_EQ(array.host(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1), 2.0);

    // Modify on host
    array.host(1, 0) = 3.0;
    array.host(1, 1) = 4.0;

    // Update device and round-trip back to verify
    array.update_device();
    array.update_host();

    EXPECT_DOUBLE_EQ(array.host(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array.host(1, 1), 4.0);
}

// Test vector constructor
TEST(DRaggedRightArrayKokkosTest, VectorConstructor) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    DRaggedRightArrayKokkos<double> array(strides, 2);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 2);
    EXPECT_EQ(array.dims(2), 0);

    array.set_values(0.0);

    // Set values on device
    RUN({
        array(0, 0, 0) = 1.0;
        array(0, 0, 1) = 2.0;
        array(0, 1, 0) = 3.0;
        array(0, 1, 1) = 4.0;
    });
    array.update_host();

    EXPECT_DOUBLE_EQ(array.host(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1, 1), 4.0);
}

// Test tensor constructor
TEST(DRaggedRightArrayKokkosTest, TensorConstructor) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);

    DRaggedRightArrayKokkos<double> array(strides, 2, 2);

    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 2);
    EXPECT_EQ(array.dims(2), 2);

    array.set_values(0.0);

    RUN({
        array(0, 0, 0, 0) = 1.0;
        array(0, 0, 0, 1) = 2.0;
        array(0, 0, 1, 0) = 3.0;
        array(0, 0, 1, 1) = 4.0;
    });
    array.update_host();

    EXPECT_DOUBLE_EQ(array.host(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array.host(0, 0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array.host(0, 0, 1, 1), 4.0);
}

// Test copy assignment
TEST(DRaggedRightArrayKokkosTest, CopyAssignment) {
    CArrayKokkos<size_t> strides1(3, "strides1");
    init_strides_2_3_1(strides1);
    DRaggedRightArrayKokkos<double> array1(strides1);
    array1.set_values(1.0);

    CArrayKokkos<size_t> strides2(3, "strides2");
    init_strides_2_3_1(strides2);
    DRaggedRightArrayKokkos<double> array2(strides2);
    array2.set_values(2.0);

    array1 = array2;
    array1.update_host();

    EXPECT_DOUBLE_EQ(array1.host(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(array1.host(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array1.host(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(array1.host(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(array1.host(1, 2), 2.0);
    EXPECT_DOUBLE_EQ(array1.host(2, 0), 2.0);
}

// Test get_name
TEST(DRaggedRightArrayKokkosTest, GetName) {
    CArrayKokkos<size_t> strides(3, "strides");
    DRaggedRightArrayKokkos<double> array(strides, "test_array");

    EXPECT_EQ(array.get_name(), "test_array");
}

// Test set_values
TEST(DRaggedRightArrayKokkosTest, SetValues) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);
    DRaggedRightArrayKokkos<double> array(strides);

    array.set_values(5.0);
    array.update_host();

    EXPECT_DOUBLE_EQ(array.host(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(array.host(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(array.host(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(array.host(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(array.host(2, 0), 5.0);
}

// Test stride_host
TEST(DRaggedRightArrayKokkosTest, StrideHost) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);
    DRaggedRightArrayKokkos<double> array(strides);

    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test device_pointer and host_pointer
TEST(DRaggedRightArrayKokkosTest, Pointers) {
    CArrayKokkos<size_t> strides(3, "strides");
    init_strides_2_3_1(strides);
    DRaggedRightArrayKokkos<double> array(strides);

    EXPECT_NE(array.device_pointer(), nullptr);
    EXPECT_NE(array.host_pointer(), nullptr);
}
