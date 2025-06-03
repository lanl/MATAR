#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Test fixture for RaggedRightArrayKokkos tests
class RaggedRightArrayKokkosTest : public ::testing::Test {
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

// Test constructor with strides array
TEST_F(RaggedRightArrayKokkosTest, ConstructorWithStrides) {
    // Create a strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;  // First row has 2 elements
    strides(1) = 3;  // Second row has 3 elements
    strides(2) = 1;  // Third row has 1 element

    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);  // 3 rows
    EXPECT_EQ(array.stride(0), 2);  // First row stride
    EXPECT_EQ(array.stride(1), 3);  // Second row stride
    EXPECT_EQ(array.stride(2), 1);  // Third row stride
}

// Test constructor with raw strides array
TEST_F(RaggedRightArrayKokkosTest, ConstructorWithRawStrides) {
    // Create raw strides array
    size_t strides[3] = {2, 3, 1};
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, 3, "test_array");
    
    // Check dimensions
    EXPECT_EQ(array.dims(0), 3);
    EXPECT_EQ(array.stride(0), 2);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 1);
}

// Test array access and modification
TEST_F(RaggedRightArrayKokkosTest, ArrayAccess) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Set values
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

// Test set_values functionality
TEST_F(RaggedRightArrayKokkosTest, SetValues) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Set all values to 42.0
    array.set_values(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 2), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 42.0);
}

// Test stride management
TEST_F(RaggedRightArrayKokkosTest, StrideManagement) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Check strides
    EXPECT_EQ(array.stride(0), 2);
    EXPECT_EQ(array.stride(1), 3);
    EXPECT_EQ(array.stride(2), 1);
}

// Test name management
TEST_F(RaggedRightArrayKokkosTest, NameManagement) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");
    
    // Create another array with different name
    RaggedRightArrayKokkos<double> array2(strides, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST_F(RaggedRightArrayKokkosTest, DifferentDataTypes) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Test with float
    RaggedRightArrayKokkos<float> array_float(strides, "float_array");
    array_float.set_values(42.0f);
    EXPECT_FLOAT_EQ(array_float(0, 0), 42.0f);
    
    // Test with int
    RaggedRightArrayKokkos<int> array_int(strides, "int_array");
    array_int.set_values(42);
    EXPECT_EQ(array_int(0, 0), 42);
}

// Test out-of-bounds access
TEST_F(RaggedRightArrayKokkosTest, OutOfBoundsAccess) {
    // Create strides array
    CArrayKokkos<size_t> strides(3, "strides");
    strides(0) = 2;
    strides(1) = 3;
    strides(2) = 1;
    
    // Create ragged array
    RaggedRightArrayKokkos<double> array(strides, "test_array");
    
    // Test accessing beyond row bounds
    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    
    // Test accessing beyond column bounds
    EXPECT_DEATH(array(0, 2), ".*");  // Row 0 only has 2 columns
    EXPECT_DEATH(array(1, 3), ".*");  // Row 1 only has 3 columns
    EXPECT_DEATH(array(2, 1), ".*");  // Row 2 only has 1 column
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
