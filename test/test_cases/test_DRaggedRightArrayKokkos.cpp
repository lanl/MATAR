#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace


// Test fixture for DRaggedRightArrayKokkos tests
class DRaggedRightArrayKokkosTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Kokkos
        Kokkos::initialize();
    }

    void TearDown() override {
        // Finalize Kokkos
        Kokkos::finalize();
    }
};

// Test default constructor
TEST_F(DRaggedRightArrayKokkosTest, DefaultConstructor) {
    DRaggedRightArrayKokkos<double> array;
    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);
}

// Test constructor with CArrayKokkos strides
TEST_F(DRaggedRightArrayKokkosTest, ConstructorWithCArrayKokkos) {
    // Create strides array
    CArrayKokkos<size_t> strides( 3);
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;

    // Create array
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);
    
    // Check strides
    EXPECT_EQ(array.stride(0), 2);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 1);
}

// Test constructor with DCArrayKokkos strides
TEST_F(DRaggedRightArrayKokkosTest, ConstructorWithDCArrayKokkos) {
    // Create strides array
    DCArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    strides.update_device();

    // Create array
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);
    
    // Check strides
    EXPECT_EQ(array.stride(0), 2);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 1);
}

// Test constructor with raw array strides
TEST_F(DRaggedRightArrayKokkosTest, ConstructorWithRawArray) {
    // Create strides array
    size_t strides[3] = {2, 3, 1};
    
    // Create array
    DRaggedRightArrayKokkos<double> array(strides, 3);
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 0);
    EXPECT_EQ(array.dims(2), 0);
    
    // Check strides
    EXPECT_EQ(array.stride(0), 2);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 1);
}

// Test 2D array access
TEST_F(DRaggedRightArrayKokkosTest, ArrayAccess2D) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;

    // Create array
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Set values
    array.set_values(0.0);
    
    // Set some test values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(1, 1) = 4.0;
    array(1, 2) = 5.0;
    array(2, 0) = 6.0;
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(array(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 6.0);
}

// Test host access
TEST_F(DRaggedRightArrayKokkosTest, HostAccess) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;

    // Create array
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Set values
    array.set_values(0.0);
    
    // Set some test values on device
    RUN({
        array(0, 0) = 1.0;
        array(0, 1) = 2.0;
    });
    
    // Update host
    array.update_host();
    
    // Check values on host
    EXPECT_DOUBLE_EQ(array.host(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1), 2.0);
    
    // Modify on host
    array.host(1, 0) = 3.0;
    array.host(1, 1) = 4.0;
    
    // Update device
    array.update_device();
    
    // Check values on device
    EXPECT_DOUBLE_EQ(array(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 4.0);
}

// Test vector constructor
TEST_F(DRaggedRightArrayKokkosTest, VectorConstructor) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;

    // Create array with vector dimension
    DRaggedRightArrayKokkos<double> array(strides, 2);
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 2);
    EXPECT_EQ(array.dims(2), 0);
    
    // Set values
    array.set_values(0.0);
    
    // Set some test values
    array(0, 0, 0) = 1.0;
    array(0, 0, 1) = 2.0;
    array(0, 1, 0) = 3.0;
    array(0, 1, 1) = 4.0;
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array(0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(0, 1, 1), 4.0);
}

// Test tensor constructor
TEST_F(DRaggedRightArrayKokkosTest, TensorConstructor) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;

    // Create array with tensor dimensions
    DRaggedRightArrayKokkos<double> array(strides, 2, 2);
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.dims(1), 2);
    EXPECT_EQ(array.dims(2), 2);
    
    // Set values
    array.set_values(0.0);
    
    // Set some test values
    array(0, 0, 0, 0) = 1.0;
    array(0, 0, 0, 1) = 2.0;
    array(0, 0, 1, 0) = 3.0;
    array(0, 0, 1, 1) = 4.0;
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array(0, 0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(0, 0, 1, 1), 4.0);
}

// Test copy assignment
TEST_F(DRaggedRightArrayKokkosTest, CopyAssignment) {
    // Create first array
    CArrayKokkos<size_t> strides1(3, "strides1");
    strides1(0) = 2;
    strides1(1) = 3;
    strides1(2) = 1;
    DRaggedRightArrayKokkos<double> array1(strides1);
    array1.set_values(1.0);
    
    // Create second array
    CArrayKokkos<size_t> strides2(3, "strides2");
    strides2(0) = 2;
    strides2(1) = 3;
    strides2(2) = 1;
    DRaggedRightArrayKokkos<double> array2(strides2);
    array2.set_values(2.0);
    
    // Copy assign
    array1 = array2;
    
    // Check values
    EXPECT_DOUBLE_EQ(array1(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(array1(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array1(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(array1(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(array1(1, 2), 2.0);
    EXPECT_DOUBLE_EQ(array1(2, 0), 2.0);
}

// Test get_name
TEST_F(DRaggedRightArrayKokkosTest, GetName) {
    // Create array with custom name
    CArrayKokkos<size_t> strides(3, "strides");
    DRaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Check name
    EXPECT_EQ(array.get_name(), "test_arrayarray");
}

// Test set_values
TEST_F(DRaggedRightArrayKokkosTest, SetValues) {
    // Create array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Set values
    array.set_values(5.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(array(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 5.0);
}

// Test stride_host
TEST_F(DRaggedRightArrayKokkosTest, StrideHost) {
    // Create array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Check host strides
    EXPECT_EQ(array.stride_host(0), 2);
    EXPECT_EQ(array.stride_host(1), 3);
    EXPECT_EQ(array.stride_host(2), 1);
}

// Test device_pointer and host_pointer
TEST_F(DRaggedRightArrayKokkosTest, Pointers) {
    // Create array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    DRaggedRightArrayKokkos<double> array(strides);
    
    // Check pointers
    EXPECT_NE(array.device_pointer(), nullptr);
    EXPECT_NE(array.host_pointer(), nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 