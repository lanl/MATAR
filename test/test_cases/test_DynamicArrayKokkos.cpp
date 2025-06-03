#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Test fixture for DynamicArrayKokkos tests
class DynamicArrayKokkosTest : public ::testing::Test {
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
TEST_F(DynamicArrayKokkosTest, DefaultConstructor) {
    DynamicArrayKokkos<double> array;
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.dims(0), 0);
    EXPECT_EQ(array.dims_max(0), 0);
    EXPECT_EQ(array.order(), 1);
}

// Test constructor with initial size
TEST_F(DynamicArrayKokkosTest, ConstructorWithSize) {
    DynamicArrayKokkos<double> array(10, "test_array");
    EXPECT_EQ(array.size(), 10);
    EXPECT_EQ(array.dims(0), 10);
    EXPECT_EQ(array.dims_max(0), 10);
    EXPECT_EQ(array.order(), 1);
    EXPECT_EQ(array.get_name(), "test_array");
}

// Test push_back functionality
TEST_F(DynamicArrayKokkosTest, PushBack) {
    DynamicArrayKokkos<double> array(5, "test_array");
    
    // Initial state
    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.dims(0), 5);
    
    // Push back some values
    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);
    
    // Check new size
    EXPECT_EQ(array.size(), 8);
    EXPECT_EQ(array.dims(0), 8);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(5), 1.0);
    EXPECT_DOUBLE_EQ(array(6), 2.0);
    EXPECT_DOUBLE_EQ(array(7), 3.0);
}

// Test pop_back functionality
TEST_F(DynamicArrayKokkosTest, PopBack) {
    DynamicArrayKokkos<double> array(5, "test_array");
    
    // Set some values
    array(0) = 1.0;
    array(1) = 2.0;
    array(2) = 3.0;
    
    // Initial state
    EXPECT_EQ(array.size(), 5);
    
    // Pop back
    array.pop_back();
    EXPECT_EQ(array.size(), 4);
    
    // Pop back again
    array.pop_back();
    EXPECT_EQ(array.size(), 3);
    
    // Check remaining values
    EXPECT_DOUBLE_EQ(array(0), 1.0);
    EXPECT_DOUBLE_EQ(array(1), 2.0);
    EXPECT_DOUBLE_EQ(array(2), 3.0);
}

// Test set_values functionality
TEST_F(DynamicArrayKokkosTest, SetValues) {
    DynamicArrayKokkos<double> array(5, "test_array");
    
    // Set all values to 42.0
    array.set_values(42.0);
    
    // Check values
    for (size_t i = 0; i < array.size(); i++) {
        EXPECT_DOUBLE_EQ(array(i), 42.0);
    }
}


// Test dimension management
TEST_F(DynamicArrayKokkosTest, DimensionManagement) {
    DynamicArrayKokkos<double> array(10, "test_array");
    
    // Check initial dimensions
    EXPECT_EQ(array.dims(0), 10);
    EXPECT_EQ(array.dims_max(0), 10);
    EXPECT_EQ(array.order(), 1);
    
    // Push back to increase size
    array.push_back(1.0);
    array.push_back(2.0);
    
    // Check updated dimensions
    EXPECT_EQ(array.dims(0), 12);
    EXPECT_EQ(array.dims_max(0), 12);
    
    // Pop back to decrease size
    array.pop_back();
    array.pop_back();
    
    // Check final dimensions
    EXPECT_EQ(array.dims(0), 10);
    EXPECT_EQ(array.dims_max(0), 12); // max dimension should not decrease
}

// Test name management
TEST_F(DynamicArrayKokkosTest, NameManagement) {
    DynamicArrayKokkos<double> array(5, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");
    
    // Create another array with different name
    DynamicArrayKokkos<double> array2(5, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test size and extent
TEST_F(DynamicArrayKokkosTest, SizeAndExtent) {
    DynamicArrayKokkos<double> array(5, "test_array");
    
    // Check initial size and extent
    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.extent(), 5);
    
    // Push back to increase size
    array.push_back(1.0);
    EXPECT_EQ(array.size(), 6);
    EXPECT_EQ(array.extent(), 6);
    
    // Pop back to decrease size
    array.pop_back();
    EXPECT_EQ(array.size(), 5);
    EXPECT_EQ(array.extent(), 5);
}

// Test array access and modification
TEST_F(DynamicArrayKokkosTest, ArrayAccess) {
    DynamicArrayKokkos<double> array(5, "test_array");
    
    // Set values
    array(0) = 1.0;
    array(1) = 2.0;
    array(2) = 3.0;
    array(3) = 4.0;
    array(4) = 5.0;
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0), 1.0);
    EXPECT_DOUBLE_EQ(array(1), 2.0);
    EXPECT_DOUBLE_EQ(array(2), 3.0);
    EXPECT_DOUBLE_EQ(array(3), 4.0);
    EXPECT_DOUBLE_EQ(array(4), 5.0);
    
    // Modify values
    array(0) = 10.0;
    array(4) = 50.0;
    
    // Check modified values
    EXPECT_DOUBLE_EQ(array(0), 10.0);
    EXPECT_DOUBLE_EQ(array(4), 50.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
