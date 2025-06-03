#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <vector>

using namespace mtr; // matar namespace

// Helper function to create and return a DCMatrixKokkos object
DCMatrixKokkos<double> return_DCMatrixKokkos(int dims, std::vector<int> sizes, const std::string& tag_string = "test_matrix")
{
    switch(dims) {
        case 1:
            return DCMatrixKokkos<double>(sizes[0], tag_string);
        case 2:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], tag_string);
        case 3:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], tag_string);
        case 4:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], tag_string);
        case 5:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], tag_string);
        case 6:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], tag_string);
        case 7:
            return DCMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], tag_string);
        default:
            return DCMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DCMatrixKokkos, default_constructor)
{
    DCMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test 1D constructor
TEST(Test_DCMatrixKokkos, constructor_1d)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, "test_matrix");
    EXPECT_EQ(A.size(), size);
    EXPECT_EQ(A.extent(), size);
    EXPECT_EQ(A.order(), 1);
    EXPECT_EQ(A.dims(0), size);
}

// Test 2D constructor
TEST(Test_DCMatrixKokkos, constructor_2d)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    EXPECT_EQ(A.size(), size * size);
    EXPECT_EQ(A.extent(), size * size);
    EXPECT_EQ(A.order(), 2);
    EXPECT_EQ(A.dims(0), size);
    EXPECT_EQ(A.dims(1), size);
}

// Test 3D constructor
TEST(Test_DCMatrixKokkos, constructor_3d)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, size, "test_matrix");
    EXPECT_EQ(A.size(), size * size * size);
    EXPECT_EQ(A.extent(), size * size * size);
    EXPECT_EQ(A.order(), 3);
    EXPECT_EQ(A.dims(0), size);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
}

// Test get_name method
TEST(Test_DCMatrixKokkos, get_name)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    EXPECT_EQ(A.get_name(), "test_matrix");
}

// Test set_values method
TEST(Test_DCMatrixKokkos, set_values)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    // Check if all values are set correctly
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(A(i, j), 42.0);
        }
    }
}

// Test operator() access
TEST(Test_DCMatrixKokkos, operator_access)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, size, "test_matrix");
    
    // Test 1D access
    A(0) = 1.0;
    EXPECT_EQ(A(0), 1.0);
    
    // Test 2D access
    A(1, 1) = 2.0;
    EXPECT_EQ(A(1, 1), 2.0);
    
    // Test 3D access
    A(1, 1, 1) = 3.0;
    EXPECT_EQ(A(1, 1, 1), 3.0);
    
    // Test 5D access
    A(1, 1, 1, 1, 1) = 4.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1), 4.0);
    
    // Test 7D access
    A(1, 1, 1, 1, 1, 1, 1) = 5.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1, 1, 1), 5.0);
}

// Test bounds checking
TEST(Test_DCMatrixKokkos, bounds_checking)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    
    // Test out of bounds access
    EXPECT_DEATH(A(size, size), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
}

// Test different types
TEST(Test_DCMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test with int
    DCMatrixKokkos<int> A(size, size, "test_matrix");
    A.set_values(42);
    EXPECT_EQ(A(0, 0), 42);
    
    // Test with float
    DCMatrixKokkos<float> B(size, size, "test_matrix");
    B.set_values(42.0f);
    EXPECT_FLOAT_EQ(B(0, 0), 42.0f);
    
    // Test with bool
    DCMatrixKokkos<bool> C(size, size, "test_matrix");
    C.set_values(true);
    EXPECT_EQ(C(0, 0), true);
}

// Test RAII behavior
TEST(Test_DCMatrixKokkos, raii)
{
    const int size = 10;
    {
        DCMatrixKokkos<double> A(size, size, "test_matrix");
        A.set_values(42.0);
        EXPECT_EQ(A(0, 0), 42.0);
    } // A goes out of scope here
}

// Test copy constructor
TEST(Test_DCMatrixKokkos, copy_constructor)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    DCMatrixKokkos<double> B(A);
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
}

// Test assignment operator
TEST(Test_DCMatrixKokkos, assignment_operator)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    DCMatrixKokkos<double> B;
    B = A;
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
}

// Test update_host method
TEST(Test_DCMatrixKokkos, update_host)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    A.update_host();
    // After update_host, host data should be synchronized
    EXPECT_EQ(A(0, 0), 42.0);
}

// Test update_device method
TEST(Test_DCMatrixKokkos, update_device)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    A.update_device();
    // After update_device, device data should be synchronized
    EXPECT_EQ(A(0, 0), 42.0);
}

// Test lock_update and unlock_update methods
TEST(Test_DCMatrixKokkos, lock_unlock_update)
{
    const int size = 10;
    DCMatrixKokkos<double> A(size, size, "test_matrix");
    
    A.lock_update();
    // After locking, updates should be prevented
    A.set_values(42.0);
    EXPECT_NE(A(0, 0), 42.0);
    
    A.unlock_update();
    // After unlocking, updates should work again
    A.set_values(42.0);
    EXPECT_EQ(A(0, 0), 42.0);
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
