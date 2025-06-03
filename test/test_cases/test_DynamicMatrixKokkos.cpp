#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Test fixture for DynamicMatrixKokkos tests
class DynamicMatrixKokkosTest : public ::testing::Test {
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
TEST_F(DynamicMatrixKokkosTest, DefaultConstructor) {
    DynamicMatrixKokkos<double> matrix;
    EXPECT_EQ(matrix.size(), 0);
    EXPECT_EQ(matrix.dims(0), 0);
    EXPECT_EQ(matrix.dims_max(0), 0);
    EXPECT_EQ(matrix.order(), 1);
}

// Test constructor with initial size
TEST_F(DynamicMatrixKokkosTest, ConstructorWithSize) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    EXPECT_EQ(matrix.size(), 10);
    EXPECT_EQ(matrix.dims(0), 10);
    EXPECT_EQ(matrix.dims_max(0), 10);
    EXPECT_EQ(matrix.order(), 1);
    EXPECT_EQ(matrix.get_name(), "test_matrix");
}

// Test push_back functionality
TEST_F(DynamicMatrixKokkosTest, PushBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Initial state
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(0), 5);
    
    // Push back some values
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);
    
    // Check new size
    EXPECT_EQ(matrix.size(), 8);
    EXPECT_EQ(matrix.dims(0), 8);
    
    // Check values
    EXPECT_DOUBLE_EQ(matrix(5), 1.0);
    EXPECT_DOUBLE_EQ(matrix(6), 2.0);
    EXPECT_DOUBLE_EQ(matrix(7), 3.0);
}

// Test pop_back functionality
TEST_F(DynamicMatrixKokkosTest, PopBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Set some values
    matrix(0) = 1.0;
    matrix(1) = 2.0;
    matrix(2) = 3.0;
    
    // Initial state
    EXPECT_EQ(matrix.size(), 5);
    
    // Pop back
    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 4);
    
    // Pop back again
    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 3);
    
    // Check remaining values
    EXPECT_DOUBLE_EQ(matrix(0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(1), 2.0);
    EXPECT_DOUBLE_EQ(matrix(2), 3.0);
}

// Test set_values functionality
TEST_F(DynamicMatrixKokkosTest, SetValues) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Set all values to 42.0
    matrix.set_values(42.0);
    
    // Check values
    for (size_t i = 0; i < matrix.size(); i++) {
        EXPECT_DOUBLE_EQ(matrix(i), 42.0);
    }
}

// Test host/device synchronization
TEST_F(DynamicMatrixKokkosTest, HostDeviceSync) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Set values on device
    matrix(0) = 1.0;
    matrix(1) = 2.0;
    matrix(2) = 3.0;
    
    // Update host
    matrix.update_host();
    
    // Check values on host
    EXPECT_DOUBLE_EQ(matrix.host(0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.host(1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.host(2), 3.0);
    
    // Modify on host
    matrix.host(3) = 4.0;
    matrix.host(4) = 5.0;
    
    // Update device
    matrix.update_device();
    
    // Check values on device
    EXPECT_DOUBLE_EQ(matrix(3), 4.0);
    EXPECT_DOUBLE_EQ(matrix(4), 5.0);
}

// Test dimension management
TEST_F(DynamicMatrixKokkosTest, DimensionManagement) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    
    // Check initial dimensions
    EXPECT_EQ(matrix.dims(0), 10);
    EXPECT_EQ(matrix.dims_max(0), 10);
    EXPECT_EQ(matrix.order(), 1);
    
    // Push back to increase size
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    
    // Check updated dimensions
    EXPECT_EQ(matrix.dims(0), 12);
    EXPECT_EQ(matrix.dims_max(0), 12);
    
    // Pop back to decrease size
    matrix.pop_back();
    matrix.pop_back();
    
    // Check final dimensions
    EXPECT_EQ(matrix.dims(0), 10);
    EXPECT_EQ(matrix.dims_max(0), 12); // max dimension should not decrease
}

// Test name management
TEST_F(DynamicMatrixKokkosTest, NameManagement) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    EXPECT_EQ(matrix.get_name(), "test_matrix");
    
    // Create another matrix with different name
    DynamicMatrixKokkos<double> matrix2(5, "another_matrix");
    EXPECT_EQ(matrix2.get_name(), "another_matrix");
}

// Test size and extent
TEST_F(DynamicMatrixKokkosTest, SizeAndExtent) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Check initial size and extent
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
    
    // Push back to increase size
    matrix.push_back(1.0);
    EXPECT_EQ(matrix.size(), 6);
    EXPECT_EQ(matrix.extent(), 6);
    
    // Pop back to decrease size
    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
}

// Test matrix access and modification
TEST_F(DynamicMatrixKokkosTest, MatrixAccess) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Set values
    matrix(0) = 1.0;
    matrix(1) = 2.0;
    matrix(2) = 3.0;
    matrix(3) = 4.0;
    matrix(4) = 5.0;
    
    // Check values
    EXPECT_DOUBLE_EQ(matrix(0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(1), 2.0);
    EXPECT_DOUBLE_EQ(matrix(2), 3.0);
    EXPECT_DOUBLE_EQ(matrix(3), 4.0);
    EXPECT_DOUBLE_EQ(matrix(4), 5.0);
    
    // Modify values
    matrix(0) = 10.0;
    matrix(4) = 50.0;
    
    // Check modified values
    EXPECT_DOUBLE_EQ(matrix(0), 10.0);
    EXPECT_DOUBLE_EQ(matrix(4), 50.0);
}

// Test matrix operations with different data types
TEST_F(DynamicMatrixKokkosTest, DifferentDataTypes) {
    // Test with float
    DynamicMatrixKokkos<float> matrix_float(5, "float_matrix");
    matrix_float.set_values(42.0f);
    EXPECT_FLOAT_EQ(matrix_float(0), 42.0f);
    
    // Test with int
    DynamicMatrixKokkos<int> matrix_int(5, "int_matrix");
    matrix_int.set_values(42);
    EXPECT_EQ(matrix_int(0), 42);
}

// Test matrix operations with different layouts
TEST_F(DynamicMatrixKokkosTest, DifferentLayouts) {
    // Test with default layout
    DynamicMatrixKokkos<double> matrix_default(5, "default_matrix");
    matrix_default.set_values(42.0);
    EXPECT_DOUBLE_EQ(matrix_default(0), 42.0);
    
    // Test with row-major layout
    DynamicMatrixKokkos<double, Kokkos::LayoutRight> matrix_row(5, "row_matrix");
    matrix_row.set_values(42.0);
    EXPECT_DOUBLE_EQ(matrix_row(0), 42.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
