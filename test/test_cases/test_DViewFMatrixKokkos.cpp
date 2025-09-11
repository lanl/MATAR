#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create matrices of different dimensions
DViewFMatrixKokkos<double> return_DViewFMatrixKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return DViewFMatrixKokkos<double>(data, sizes[0], "test_matrix_1d");
        case 2:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], "test_matrix_2d");
        case 3:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], "test_matrix_3d");
        case 4:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], "test_matrix_4d");
        case 5:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], "test_matrix_5d");
        case 6:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "test_matrix_6d");
        case 7:
            return DViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], "test_matrix_7d");
        default:
            return DViewFMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DViewFMatrixKokkos, default_constructor)
{
    DViewFMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.host_pointer(), nullptr);
}

// Test size function
TEST(Test_DViewFMatrixKokkos, size)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.size());
    delete[] data;
}

// Test extent function
TEST(Test_DViewFMatrixKokkos, extent)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.extent());
    delete[] data;
}

// Test dims function
TEST(Test_DViewFMatrixKokkos, dims)
{
    const int size = 100;
    double* data = new double[size*size*size];
    DViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    EXPECT_EQ(size, A.dims(3));
    delete[] data;
}

// Test order function
TEST(Test_DViewFMatrixKokkos, order)
{
    const int size = 100;
    double* data = new double[size*size*size];
    DViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(3, A.order());
    delete[] data;
}

// Test pointer function
TEST(Test_DViewFMatrixKokkos, pointer)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_NE(A.host_pointer(), nullptr);
    delete[] data;
}

// Test get_name function
// TEST(Test_DViewFMatrixKokkos, get_name)
// {
//     const int size = 100;
//     double* data = new double[size*size];
//     DViewFMatrixKokkos<double> A(data, size, size, "test_matrix");
//     EXPECT_EQ(A.get_name(), "test_matrix");
//     delete[] data;
// }

// Test set_values function
TEST(Test_DViewFMatrixKokkos, set_values)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    
    A.set_values(42.0);
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
    delete[] data;
}

// Test operator access
TEST(Test_DViewFMatrixKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size*size*size];
    DViewFMatrixKokkos<double> A(data, size, size, size);
    
    // Test 3D access
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            for(int k = 1; k <= size; k++) {
                A(i,j,k) = i*100 + j*10 + k;
            }
        }
    }
    
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            for(int k = 1; k <= size; k++) {
                EXPECT_EQ(i*100 + j*10 + k, A(i,j,k));
            }
        }
    }
    delete[] data;
}

#ifndef NDEBUG
// Test bounds checking
TEST(Test_DViewFMatrixKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size+1,size+1), ".*");
    delete[] data;
}
#endif

// Test different types
TEST(Test_DViewFMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    int* int_data = new int[size*size];
    DViewFMatrixKokkos<int> A(int_data, size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    delete[] int_data;
    
    // Test float
    float* float_data = new float[size*size];
    DViewFMatrixKokkos<float> B(float_data, size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    delete[] float_data;
    
    // Test bool
    bool* bool_data = new bool[size*size];
    DViewFMatrixKokkos<bool> C(bool_data, size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
    delete[] bool_data;
}

// Test host-device synchronization
TEST(Test_DViewFMatrixKokkos, host_device_sync)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFMatrixKokkos<double> A(data, size, size);
    
    // Set values on host
    A.set_values(42.0);

    // Update host
    A.update_host();
    
    // Check values on host
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
    delete[] data;
}

// Test lock/unlock update functionality
// TEST(Test_DViewFMatrixKokkos, lock_unlock_update)
// {
//     const int size = 100;
//     double* data = new double[size*size];
//     DViewFMatrixKokkos<double> A(data, size, size);
    
//     // Set initial values
//     A.set_values(42.0);
    
//     // Lock updates
//     A.lock_update();
    
//     // Try to modify values - should not affect the data
//     A.set_values(24.0);
    
//     // Check values remain unchanged
//     for(int i = 0; i < size; i++) {
//         for(int j = 0; j < size; j++) {
//             EXPECT_EQ(42.0, A(i,j));
//         }
//     }
    
//     // Unlock updates
//     A.unlock_update();
    
//     // Now modifications should work
//     A.set_values(24.0);
//     for(int i = 0; i < size; i++) {
//         for(int j = 0; j < size; j++) {
//             EXPECT_EQ(24.0, A(i,j));
//         }
//     }
//     delete[] data;
// }

// Test RAII behavior
TEST(Test_DViewFMatrixKokkos, raii)
{
    double* data = new double[100*100];
    {
        DViewFMatrixKokkos<double> A(data, 100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new matrix to verify memory was freed
    DViewFMatrixKokkos<double> B(data, 100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
    delete[] data;
}
