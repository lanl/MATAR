#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace
// Test fixture for DynamicRaggedRightArrayKokkos tests
class DynamicRaggedRightArrayKokkosTest : public ::testing::Test {
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

//TO DO: Add following capability
// Test default constructor and basic initialization
// TEST_F(DynamicRaggedRightArrayKokkosTest, DefaultConstructor) {
//     DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
//     // Check initial dimensions
//     EXPECT_EQ(array.dims(0), 3);  // 3 rows
//     EXPECT_EQ(array.dims(1), 2);  // Initial column size
    
//     // Check initial strides
//     EXPECT_EQ(array.stride(0), 0);
//     EXPECT_EQ(array.stride(1), 0);
//     EXPECT_EQ(array.stride(2), 0);
// }

// // Test push_back functionality
// TEST_F(DynamicRaggedRightArrayKokkosTest, PushBack) {
//     DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
//     // Push back values to first row
//     array.push_back(1.0);
//     array.push_back(2.0);
//     array.push_back(3.0);
    
//     // Check values
//     EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(array(0, 1), 2.0);
//     EXPECT_DOUBLE_EQ(array(0, 2), 3.0);
    
//     // Check stride
//     EXPECT_EQ(array.stride(0), 3);
// }

// // Test pop_back functionality
// TEST_F(DynamicRaggedRightArrayKokkosTest, PopBack) {
//     DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
//     // Push back values
//     array.push_back(1.0);
//     array.push_back(2.0);
//     array.push_back(3.0);
    
//     // Pop back
//     array.pop_back();
    
//     // Check values
//     EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(array(0, 1), 2.0);
    
//     // Check stride
//     EXPECT_EQ(array.stride(0), 2);
// }

// Test set_values functionality
TEST_F(DynamicRaggedRightArrayKokkosTest, SetValues) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
    // Set all values to 42.0
    array.set_values(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
}

// Test set_values_sparse functionality
TEST_F(DynamicRaggedRightArrayKokkosTest, SetValuesSparse) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
    // Set sparse values
    array.set_values_sparse(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
}


// Test name management
TEST_F(DynamicRaggedRightArrayKokkosTest, NameManagement) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");
    
    // Create another array with different name
    DynamicRaggedRightArrayKokkos<double> array2(3, 2, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST_F(DynamicRaggedRightArrayKokkosTest, DifferentDataTypes) {
    
    // Test with float
    DynamicRaggedRightArrayKokkos<float> array_float(3, 2, "float_array");
    array_float(0,0) = 42.0f;
    EXPECT_FLOAT_EQ(array_float(0, 0), 42.0f);
    
    
    // Test with int
    DynamicRaggedRightArrayKokkos<int> array_int(3, 2, "int_array");
    array_int(0,0) = 42;
    EXPECT_EQ(array_int(0, 0), 42);
}

// Test out-of-bounds access
TEST_F(DynamicRaggedRightArrayKokkosTest, OutOfBoundsAccess) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
    // Test accessing beyond row bounds
    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    
    // Test accessing beyond column bounds
    EXPECT_DEATH(array(0, 2), ".*");  // Initial column size is 2
}

// Test get_kokkos_dual_view
TEST_F(DynamicRaggedRightArrayKokkosTest, GetKokkosDualView) {
    DynamicRaggedRightArrayKokkos<double> array(3, 2, "test_array");
    
    // Get the dual view
    auto view = array.get_kokkos_view();
    
    // Check that the view is valid
    EXPECT_TRUE(view.data() != nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
