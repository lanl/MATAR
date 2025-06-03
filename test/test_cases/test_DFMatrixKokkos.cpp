#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace


// Helper function to create matrices of different dimensions
DFMatrixKokkos<double> return_DFMatrixKokkos(int dims, std::vector<int> sizes)
{
    switch(dims) {
        case 1:
            return DFMatrixKokkos<double>(sizes[0]);
        case 2:
            return DFMatrixKokkos<double>(sizes[0], sizes[1]);
        case 3:
            return DFMatrixKokkos<double>(sizes[0], sizes[1], sizes[2]);
        case 4:
            return DFMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return DFMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return DFMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return DFMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return DFMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DFMatrixKokkos, default_constructor)
{
    DFMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.host_pointer(), nullptr);
}

// Test size function
TEST(Test_DFMatrixKokkos, size)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    EXPECT_EQ(size*size, A.size());
}

// Test extent function
TEST(Test_DFMatrixKokkos, extent)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    EXPECT_EQ(size*size, A.extent());
}

// Test dims function
TEST(Test_DFMatrixKokkos, dims)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size, size);
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    EXPECT_EQ(size, A.dims(3));
}

// Test order function
TEST(Test_DFMatrixKokkos, order)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size, size);
    EXPECT_EQ(3, A.order());
}

// Test pointer function
TEST(Test_DFMatrixKokkos, pointer)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    EXPECT_NE(A.host_pointer(), nullptr);
}

// Test get_name function
TEST(Test_DFMatrixKokkos, get_name)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size, "test_matrix");
    EXPECT_EQ(A.get_name(), "test_matrix");
}

// Test late initialization
TEST(Test_DFMatrixKokkos, late_init)
{
    const int size = 100;
    DFMatrixKokkos<double> A;
    
    // Initialize after construction
    A = DFMatrixKokkos<double>(size, size);
    EXPECT_EQ(size*size, A.size());
    EXPECT_EQ(2, A.order());
    EXPECT_NE(A.host_pointer(), nullptr);
}

// Test assignment operator
TEST(Test_DFMatrixKokkos, eq_overload)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    DFMatrixKokkos<double> B(size, size);
    
    // Set values in A
    A.set_values(42.0);
    
    // Assign A to B
    B = A;
    
    // Check values in B
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(42.0, B(i,j));
        }
    }
}

// Test set_values function
TEST(Test_DFMatrixKokkos, set_values)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    
    A.set_values(42.0);
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
}

// Test operator access
TEST(Test_DFMatrixKokkos, operator_access)
{
    const int size = 10;
    DFMatrixKokkos<double> A(size, size, size);
    
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
}

// Test bounds checking
TEST(Test_DFMatrixKokkos, bounds_checking)
{
    const int size = 10;
    DFMatrixKokkos<double> A(size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size+1,size+1), ".*"); // Matrix indices go from 1 to size
}

// Test different types
TEST(Test_DFMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    DFMatrixKokkos<int> A(size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    
    // Test float
    DFMatrixKokkos<float> B(size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    
    // Test bool
    DFMatrixKokkos<bool> C(size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
}

// Test host-device synchronization
TEST(Test_DFMatrixKokkos, host_device_sync)
{
    const int size = 100;
    DFMatrixKokkos<double> A(size, size);
    
    // Set values on device
    A.set_values(42.0);
    
    // Update host
    A.update_host();
    
    // Check values on host
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(42.0, A(i,j));
        }
    }
}

// Test RAII behavior
TEST(Test_DFMatrixKokkos, raii)
{
    {
        DFMatrixKokkos<double> A(100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new matrix to verify memory was freed
    DFMatrixKokkos<double> B(100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {  
        int result = 0;
        testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
        return result;
    }
    Kokkos::finalize();
}
