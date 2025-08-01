#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace


// Helper function to create arrays of different dimensions
DFArrayKokkos<double> return_DFArrayKokkos(int dims, std::vector<int> sizes)
{
    switch(dims) {
        case 1:
            return DFArrayKokkos<double>(sizes[0]);
        case 2:
            return DFArrayKokkos<double>(sizes[0], sizes[1]);
        case 3:
            return DFArrayKokkos<double>(sizes[0], sizes[1], sizes[2]);
        case 4:
            return DFArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return DFArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return DFArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return DFArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return DFArrayKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DFArrayKokkos, default_constructor)
{
    DFArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.host_pointer(), nullptr);
}

// Test size function
TEST(Test_DFArrayKokkos, size)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    EXPECT_EQ(size*size, A.size());
}

// Test extent function
TEST(Test_DFArrayKokkos, extent)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    EXPECT_EQ(size*size, A.extent());
}

// Test dims function
TEST(Test_DFArrayKokkos, dims)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size, size);
    EXPECT_EQ(size, A.dims(0));
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
}

// Test order function
TEST(Test_DFArrayKokkos, order)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size, size);
    EXPECT_EQ(3, A.order());
}

// Test pointer function
TEST(Test_DFArrayKokkos, pointer)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    EXPECT_NE(A.host_pointer(), nullptr);
}

// Test get_name function
TEST(Test_DFArrayKokkos, get_name)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size, "test_array");
    EXPECT_EQ(A.get_name(), "test_array");
}

// Test late initialization
TEST(Test_DFArrayKokkos, late_init)
{
    DFArrayKokkos<double> A;
    const int size = 100;
    
    // Initialize after construction
    A = DFArrayKokkos<double>(size, size);
    EXPECT_EQ(A.size(), size*size);
    EXPECT_EQ(A.order(), 2);
    EXPECT_EQ(A.dims(0), size);
    EXPECT_EQ(A.dims(1), size);
}

// Test assignment operator
TEST(Test_DFArrayKokkos, eq_overload)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    DFArrayKokkos<double> B(size, size);
    
    // Set values in A
    A.set_values(42.0);
    
    // Assign A to B
    B = A;
    
    // Check values in B
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(B(i,j), 42.0);
        }
    }
}

// Test set_values function
TEST(Test_DFArrayKokkos, set_values)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    
    A.set_values(42.0);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
}

// Test operator access
TEST(Test_DFArrayKokkos, operator_access)
{
    const int size = 10;
    DFArrayKokkos<double> A(size, size, size);
    
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
}

#ifndef NDEBUG
// Test bounds checking
TEST(Test_DFArrayKokkos, bounds_checking)
{
    const int size = 10;
    DFArrayKokkos<double> A(size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size,size), ".*");
}
#endif

// Test different types
TEST(Test_DFArrayKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    DFArrayKokkos<int> A(size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    
    // Test float
    DFArrayKokkos<float> B(size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    
    // Test bool
    DFArrayKokkos<bool> C(size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
}

// Test host-device synchronization
TEST(Test_DFArrayKokkos, host_device_sync)
{
    const int size = 100;
    DFArrayKokkos<double> A(size, size);
    
    // Set values on host
    A.set_values(42.0);
    
    // Update device
    A.update_device();
    
    // Modify on device
    A.set_values(24.0);

    // Update host
    A.update_host();
    
    // Check values on host
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(24.0, A(i,j));
        }
    }
}

// Test RAII behavior
TEST(Test_DFArrayKokkos, raii)
{
    {
        DFArrayKokkos<double> A(100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new array to verify memory was freed
    DFArrayKokkos<double> B(100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
}
