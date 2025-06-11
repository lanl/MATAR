#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Test default constructor
TEST(DynamicMatrixKokkosTest, DefaultConstructor) {
    DynamicMatrixKokkos<double> matrix;
    EXPECT_EQ(matrix.size(), 0);
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 0);
    EXPECT_EQ(matrix.order(), 0);
}

// Test constructor with initial size
TEST(DynamicMatrixKokkosTest, ConstructorWithSize) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    EXPECT_EQ(matrix.size(), 10);
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10);
    EXPECT_EQ(matrix.order(), 1);
    EXPECT_EQ(matrix.get_name(), "test_matrix");
}

// Test push_back functionality
TEST(DynamicMatrixKokkosTest, PushBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Initial state
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 0);
    
    // Push back some values
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);
    
    // Check new size
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 3);
    
    // Check values
    EXPECT_DOUBLE_EQ(matrix(1), 1.0);
    EXPECT_DOUBLE_EQ(matrix(2), 2.0);
    EXPECT_DOUBLE_EQ(matrix(3), 3.0);
}

// Test pop_back functionality
TEST(DynamicMatrixKokkosTest, PopBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // // Set some values
    // matrix(0) = 1.0;
    // matrix(1) = 2.0;
    // matrix(2) = 3.0;

    // Push back some values
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);
    
    // Initial state
    EXPECT_EQ(matrix.size(), 5);
    
    // Pop back
    matrix.pop_back();
    EXPECT_EQ(matrix.dims(1), 2);
    
    // Pop back again
    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 1);
    
    // Check remaining values
    EXPECT_DOUBLE_EQ(matrix(1), 1.0);
}

// Test set_values functionality
TEST(DynamicMatrixKokkosTest, SetValues) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    
    // Set all values to 42.0
    matrix.set_values(42.0, 10);

    EXPECT_DEATH(matrix.set_values(42.0, 11),"");
    
    // Check values
    for (size_t i = 1; i <= matrix.dims(1); i++) {
        EXPECT_DOUBLE_EQ(matrix(i), 42.0);
    }
}

// Test dimension management
TEST(DynamicMatrixKokkosTest, DimensionManagement) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    
    // Check initial dimensions
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10);
    EXPECT_EQ(matrix.order(), 1);
    
    // Push back to increase size
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    
    // Check updated dimensions
    EXPECT_EQ(matrix.dims(1), 2);
    EXPECT_EQ(matrix.dims_max(1), 10);
    
    // Pop back to decrease size
    matrix.pop_back();
    matrix.pop_back();
    
    // Check final dimensions
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10); // max dimension should not decrease
}

// Test name management
TEST(DynamicMatrixKokkosTest, NameManagement) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    EXPECT_EQ(matrix.get_name(), "test_matrix");
    
    // Create another matrix with different name
    DynamicMatrixKokkos<double> matrix2(5, "another_matrix");
    EXPECT_EQ(matrix2.get_name(), "another_matrix");
}

// Test size and extent
TEST(DynamicMatrixKokkosTest, SizeAndExtent) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Check initial size and extent
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
    
    // Push back to increase size
    matrix.push_back(1.0);
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
    
    // Pop back to decrease size
    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
}

// Test matrix access and modification
TEST(DynamicMatrixKokkosTest, MatrixAccess) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    
    // Set values
    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);
    matrix.push_back(4.0);
    matrix.push_back(5.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(matrix(1), 1.0);
    EXPECT_DOUBLE_EQ(matrix(2), 2.0);
    EXPECT_DOUBLE_EQ(matrix(3), 3.0);
    EXPECT_DOUBLE_EQ(matrix(4), 4.0);
    EXPECT_DOUBLE_EQ(matrix(5), 5.0);
    
    // Modify values
    matrix(1) = 10.0;
    matrix(4) = 50.0;
    
    // Check modified values
    EXPECT_DOUBLE_EQ(matrix(1), 10.0);
    EXPECT_DOUBLE_EQ(matrix(4), 50.0);
}

// Test matrix operations with different data types
TEST(DynamicMatrixKokkosTest, DifferentDataTypes) {
    // Test with float
    DynamicMatrixKokkos<float> matrix_float(5, "float_matrix");
    matrix_float.set_values(42.0f, 5);
    EXPECT_FLOAT_EQ(matrix_float(1), 42.0f);
    
    // Test with int
    DynamicMatrixKokkos<int> matrix_int(5, "int_matrix");
    matrix_int.set_values(42, 5);
    EXPECT_EQ(matrix_int(3), 42);
}
