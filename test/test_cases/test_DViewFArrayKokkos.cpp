#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create arrays of different dimensions
DViewFArrayKokkos<double> return_DViewFArrayKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return DViewFArrayKokkos<double>(data, sizes[0]);
        case 2:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return DViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return DViewFArrayKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DViewFArrayKokkos, default_constructor)
{
    DViewFArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.host_pointer(), nullptr);
}

// Test size function
TEST(Test_DViewFArrayKokkos, size)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.size());
    delete[] data;
}

// Test extent function
TEST(Test_DViewFArrayKokkos, extent)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.extent());
    delete[] data;
}

// Test dims function
TEST(Test_DViewFArrayKokkos, dims)
{
    const int size = 100;
    double* data = new double[size*size*size];
    DViewFArrayKokkos<double> A(data, size, size, size);
    EXPECT_EQ(size, A.dims(0));
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    delete[] data;
}

// Test order function
TEST(Test_DViewFArrayKokkos, order)
{
    const int size = 100;
    double* data = new double[size*size*size];
    DViewFArrayKokkos<double> A(data, size, size, size);
    EXPECT_EQ(3, A.order());
    delete[] data;
}

// Test pointer function
TEST(Test_DViewFArrayKokkos, pointer)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    EXPECT_NE(A.host_pointer(), nullptr);
    delete[] data;
}

// Test get_name function
TEST(Test_DViewFArrayKokkos, get_name)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size, "test_array");
    EXPECT_EQ(A.get_name(), "test_array");
    delete[] data;
}

// Test set_values function
TEST(Test_DViewFArrayKokkos, set_values)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    
    A.set_values(42.0);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
    delete[] data;
}

// Test operator access
TEST(Test_DViewFArrayKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size*size*size];
    DViewFArrayKokkos<double> A(data, size, size, size);
    
    // Test 3D access
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            for(int k = 0; k < size; k++) {
                A(i,j,k) = i*100 + j*10 + k;
            }
        }
    }
    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            for(int k = 0; k < size; k++) {
                EXPECT_EQ(i*100 + j*10 + k, A(i,j,k));
            }
        }
    }
    delete[] data;
}

// Test bounds checking
TEST(Test_DViewFArrayKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size,size), ".*");
    delete[] data;
}

// Test different types
TEST(Test_DViewFArrayKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    int* int_data = new int[size*size];
    DViewFArrayKokkos<int> A(int_data, size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    delete[] int_data;
    
    // Test float
    float* float_data = new float[size*size];
    DViewFArrayKokkos<float> B(float_data, size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    delete[] float_data;
    
    // Test bool
    bool* bool_data = new bool[size*size];
    DViewFArrayKokkos<bool> C(bool_data, size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
    delete[] bool_data;
}

// Test host-device synchronization
TEST(Test_DViewFArrayKokkos, host_device_sync)
{
    const int size = 100;
    double* data = new double[size*size];
    DViewFArrayKokkos<double> A(data, size, size);
    
    // Set values on host
    A.set_values(42.0);
    
    // Update device
    A.update_device();
    
    // Modify on device
    Kokkos::parallel_for("ModifyDevice", size*size, KOKKOS_LAMBDA(const int i) {
        A(i) = 24.0;
    });
    
    // Update host
    A.update_host();
    
    // Check values on host
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(24.0, A(i,j));
        }
    }
    delete[] data;
}


// Test RAII behavior
TEST(Test_DViewFArrayKokkos, raii)
{
    double* data = new double[100*100];
    {
        DViewFArrayKokkos<double> A(data, 100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new array to verify memory was freed
    DViewFArrayKokkos<double> B(data, 100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
    delete[] data;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
