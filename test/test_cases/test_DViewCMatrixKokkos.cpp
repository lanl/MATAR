#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <vector>

using namespace mtr; // matar namespace

// Helper function to create and return a DViewCMatrixKokkos object
DViewCMatrixKokkos<double> return_DViewCMatrixKokkos(int dims, std::vector<int> sizes, double* data, const std::string& tag_string = "test_matrix")
{
    switch(dims) {
        case 1:
            return DViewCMatrixKokkos<double>(data, sizes[0], tag_string);
        case 2:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], tag_string);
        case 3:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], tag_string);
        case 4:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], tag_string);
        case 5:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], tag_string);
        case 6:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], tag_string);
        case 7:
            return DViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], tag_string);
        default:
            return DViewCMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DViewCMatrixKokkos, default_constructor)
{
    DViewCMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test 1D constructor
TEST(Test_DViewCMatrixKokkos, constructor_1d)
{
    const int size = 10;
    double* data = new double[size];
    DViewCMatrixKokkos<double> A(data, size, "test_matrix");
    EXPECT_EQ(A.size(), size);
    EXPECT_EQ(A.extent(), size);
    EXPECT_EQ(A.order(), 1);
    EXPECT_EQ(A.dims(1), size);
    delete[] data;
}

// Test 2D constructor
TEST(Test_DViewCMatrixKokkos, constructor_2d)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    EXPECT_EQ(A.size(), size * size);
    EXPECT_EQ(A.extent(), size * size);
    EXPECT_EQ(A.order(), 2);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
    delete[] data;
}

// Test 3D constructor
TEST(Test_DViewCMatrixKokkos, constructor_3d)
{
    const int size = 10;
    double* data = new double[size * size * size];
    DViewCMatrixKokkos<double> A(data, size, size, size, "test_matrix");
    EXPECT_EQ(A.size(), size * size * size);
    EXPECT_EQ(A.extent(), size * size * size);
    EXPECT_EQ(A.order(), 3);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
    EXPECT_EQ(A.dims(3), size);
    delete[] data;
}

// Test get_name method
TEST(Test_DViewCMatrixKokkos, get_name)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    EXPECT_EQ(A.get_name(), "test_matrix");
    delete[] data;
}

// Test set_values method
TEST(Test_DViewCMatrixKokkos, set_values)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    A.set_values(42.0);
    
    // Check if all values are set correctly
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(A(i, j), 42.0);
        }
    }
    delete[] data;
}

// Test operator() access
TEST(Test_DViewCMatrixKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size * size * size];
    DViewCMatrixKokkos<double> A(data, size, size, size, "test_matrix");
    
    // Test 1D access
    EXPECT_DEATH(A(0), ".*");
    
    // Test 2D access
    EXPECT_DEATH(A(1, 1), ".*");
    
    // Test 3D access
    A(1, 1, 1) = 3.0;
    EXPECT_EQ(A(1, 1, 1), 3.0);
    
    // Test 5D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1), ".*");
    
    // Test 7D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1, 1, 1), ".*");
    
    delete[] data;
}

// Test bounds checking
TEST(Test_DViewCMatrixKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    
    // Test out of bounds access
    EXPECT_DEATH(A(0, 0), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
    
    delete[] data;
}

// Test different types
TEST(Test_DViewCMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test with int
    int* data_int = new int[size * size];
    DViewCMatrixKokkos<int> A(data_int, size, size, "test_matrix");
    A.set_values(42);
    EXPECT_EQ(A(1, 1), 42);
    delete[] data_int;
    
    // Test with float
    float* data_float = new float[size * size];
    DViewCMatrixKokkos<float> B(data_float, size, size, "test_matrix");
    B.set_values(42.0f);
    EXPECT_FLOAT_EQ(B(1, 1), 42.0f);
    delete[] data_float;
    
    // Test with bool
    bool* data_bool = new bool[size * size];
    DViewCMatrixKokkos<bool> C(data_bool, size, size, "test_matrix");
    C.set_values(true);
    EXPECT_EQ(C(1, 1), true);
    delete[] data_bool;
}

// Test RAII behavior
TEST(Test_DViewCMatrixKokkos, raii)
{
    const int size = 10;
    double* data = new double[size * size];
    {
        DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
        A.set_values(42.0);
        EXPECT_EQ(A(1, 1), 42.0);
    } // A goes out of scope here
    delete[] data;
}

// Test copy constructor
TEST(Test_DViewCMatrixKokkos, copy_constructor)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    A.set_values(42.0);
    
    DViewCMatrixKokkos<double> B(A);
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(1, 1), A(1, 1));
    
    delete[] data;
}

// Test assignment operator
TEST(Test_DViewCMatrixKokkos, assignment_operator)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    A.set_values(42.0);
    
    double* data2 = new double[size * size];
    DViewCMatrixKokkos<double> B(data2, size, size, "test_matrix");
    B = A;
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(1, 1), A(1, 1));
    
    delete[] data;
    delete[] data2;
}

// Test update_host method
TEST(Test_DViewCMatrixKokkos, update_host)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    A.set_values(42.0);
    A.update_host();
    // After update_host, host data should be synchronized
    EXPECT_EQ(A(1, 1), 42.0);
    delete[] data;
}

// Test update_device method
TEST(Test_DViewCMatrixKokkos, update_device)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    A.set_values(42.0);
    A.update_device();
    // After update_device, device data should be synchronized
    EXPECT_EQ(A(1, 1), 42.0);
    delete[] data;
}

// Test lock_update and unlock_update methods
// TEST(Test_DViewCMatrixKokkos, lock_unlock_update)
// {
//     const int size = 10;
//     double* data = new double[size * size];
//     DViewCMatrixKokkos<double> A(data, size, size, "test_matrix");
    
//     A.lock_update();
//     // After locking, updates should be prevented
//     A.set_values(42.0);
//     EXPECT_NE(A(0, 0), 42.0);
    
//     A.unlock_update();
//     // After unlocking, updates should work again
//     A.set_values(42.0);
//     EXPECT_EQ(A(0, 0), 42.0);
    
//     delete[] data;
// }
