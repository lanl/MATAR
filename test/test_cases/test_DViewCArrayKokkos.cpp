#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <vector>

using namespace mtr; // matar namespace



// Helper function to create and return a DViewCArrayKokkos object
DViewCArrayKokkos<double> return_DViewCArrayKokkos(int dims, std::vector<int> sizes, double* data, const std::string& tag_string = "test_array")
{
    switch(dims) {
        case 1:
            return DViewCArrayKokkos<double>(data, sizes[0], tag_string);
        case 2:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], tag_string);
        case 3:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], tag_string);
        case 4:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], tag_string);
        case 5:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], tag_string);
        case 6:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], tag_string);
        case 7:
            return DViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], tag_string);
        default:
            return DViewCArrayKokkos<double>();
    }
}

// Test default constructor
TEST(Test_DViewCArrayKokkos, default_constructor)
{
    DViewCArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test size method
TEST(Test_DViewCArrayKokkos, size)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    EXPECT_EQ(A.size(), size * size);
    delete[] data;
}

// Test extent method
TEST(Test_DViewCArrayKokkos, extent)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    EXPECT_EQ(A.extent(), size * size);
    delete[] data;
}

// Test dims method
TEST(Test_DViewCArrayKokkos, dims)
{
    const int size = 10;
    double* data = new double[size * size * size];
    DViewCArrayKokkos<double> A(data, size, size, size, "test_array");
    EXPECT_EQ(A.dims(0), size);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
    delete[] data;
}

// Test order method
TEST(Test_DViewCArrayKokkos, order)
{
    const int size = 10;
    double* data = new double[size * size * size];
    DViewCArrayKokkos<double> A(data, size, size, size, "test_array");
    EXPECT_EQ(A.order(), 3);
    delete[] data;
}

// Test get_name method
TEST(Test_DViewCArrayKokkos, get_name)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    EXPECT_EQ(A.get_name(), "test_array");
    delete[] data;
}

// Test set_values method
TEST(Test_DViewCArrayKokkos, set_values)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    A.set_values(42.0);
    for(int i = 0; i < size * size; i++) {
        EXPECT_EQ(data[i], 42.0);
    }
    delete[] data;
}

// Test operator() access
TEST(Test_DViewCArrayKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size * size * size];
    DViewCArrayKokkos<double> A(data, size, size, size, "test_array");
    
    // Test 1D access
    data[0] = 1.0;
    EXPECT_EQ(A(0), 1.0);
    
    // Test 2D access
    data[size + 1] = 2.0;
    EXPECT_EQ(A(1, 1), 2.0);
    
    // Test 3D access
    data[size * size + size + 1] = 3.0;
    EXPECT_EQ(A(1, 1, 1), 3.0);
    
    // Test 5D access
    data[size * size * size * size + size * size * size + size * size + size + 1] = 4.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1), 4.0);
    
    // Test 7D access
    data[size * size * size * size * size * size + size * size * size * size * size + 
         size * size * size * size + size * size * size + size * size + size + 1] = 5.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1, 1, 1), 5.0);
    
    delete[] data;
}

// Test bounds checking
TEST(Test_DViewCArrayKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    
    // Test out of bounds access
    EXPECT_DEATH(A(size, size), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
    
    delete[] data;
}

// Test different types
TEST(Test_DViewCArrayKokkos, different_types)
{
    const int size = 10;
    
    // Test with int
    int* int_data = new int[size * size];
    DViewCArrayKokkos<int> A(int_data, size, size, "test_array");
    A.set_values(42);
    EXPECT_EQ(int_data[0], 42);
    delete[] int_data;
    
    // Test with float
    float* float_data = new float[size * size];
    DViewCArrayKokkos<float> B(float_data, size, size, "test_array");
    B.set_values(42.0f);
    EXPECT_FLOAT_EQ(float_data[0], 42.0f);
    delete[] float_data;
    
    // Test with bool
    bool* bool_data = new bool[size * size];
    DViewCArrayKokkos<bool> C(bool_data, size, size, "test_array");
    C.set_values(true);
    EXPECT_EQ(bool_data[0], true);
    delete[] bool_data;
}

// Test RAII behavior
TEST(Test_DViewCArrayKokkos, raii)
{
    const int size = 10;
    double* data = new double[size * size];
    {
        DViewCArrayKokkos<double> A(data, size, size, "test_array");
        A.set_values(42.0);
    } // A goes out of scope here
    // Data should still be accessible and unchanged
    EXPECT_EQ(data[0], 42.0);
    delete[] data;
}

// Test copy constructor
TEST(Test_DViewCArrayKokkos, copy_constructor)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    A.set_values(42.0);
    
    DViewCArrayKokkos<double> B(A);
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
    
    delete[] data;
}

// Test assignment operator
TEST(Test_DViewCArrayKokkos, assignment_operator)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    A.set_values(42.0);
    
    DViewCArrayKokkos<double> B;
    B = A;
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
    
    delete[] data;
}

// Test update_host method
TEST(Test_DViewCArrayKokkos, update_host)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    A.set_values(42.0);
    A.update_host();
    // After update_host, host data should be synchronized
    EXPECT_EQ(data[0], 42.0);
    delete[] data;
}

// Test update_device method
TEST(Test_DViewCArrayKokkos, update_device)
{
    const int size = 10;
    double* data = new double[size * size];
    DViewCArrayKokkos<double> A(data, size, size, "test_array");
    A.set_values(42.0);
    A.update_device();
    // After update_device, device data should be synchronized
    EXPECT_EQ(A(0, 0), 42.0);
    delete[] data;
}

// Test lock_update and unlock_update methods
// TEST(Test_DViewCArrayKokkos, lock_unlock_update)
// {
//     const int size = 10;
//     double* data = new double[size * size];
//     DViewCArrayKokkos<double> A(data, size, size, "test_array");
    
//     A.lock_update();
//     // After locking, updates should be prevented
//     A.set_values(42.0);
//     EXPECT_NE(data[0], 42.0);
    
//     A.unlock_update();
//     // After unlocking, updates should work again
//     A.set_values(42.0);
//     EXPECT_EQ(data[0], 42.0);
    
//     delete[] data;
// }

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
