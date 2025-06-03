#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <vector>

using namespace mtr; // matar namespace


// Helper function to create and return a ViewCArrayKokkos object
ViewCArrayKokkos<double> return_ViewCArrayKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return ViewCArrayKokkos<double>(data, sizes[0]);
        case 2:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return ViewCArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return ViewCArrayKokkos<double>();
    }
}

// Test default constructor
TEST(Test_ViewCArrayKokkos, default_constructor)
{
    ViewCArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test size method
TEST(Test_ViewCArrayKokkos, size)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    EXPECT_EQ(A.size(), size * size);
    delete[] data;
}

// Test extent method
TEST(Test_ViewCArrayKokkos, extent)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    EXPECT_EQ(A.extent(), size * size);
    delete[] data;
}

// Test dims method
TEST(Test_ViewCArrayKokkos, dims)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCArrayKokkos<double> A(data, size, size, size);
    EXPECT_EQ(A.dims(0), size);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
    delete[] data;
}

// Test order method
TEST(Test_ViewCArrayKokkos, order)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCArrayKokkos<double> A(data, size, size, size);
    EXPECT_EQ(A.order(), 3);
    delete[] data;
}

// Test pointer method
TEST(Test_ViewCArrayKokkos, pointer)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    EXPECT_EQ(A.pointer(), data);
    delete[] data;
}

// Test get_name method
TEST(Test_ViewCArrayKokkos, get_name)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    // Note: get_name() returns empty string by default
    EXPECT_EQ(A.get_name(), "");
    delete[] data;
}

// Test set_values method
TEST(Test_ViewCArrayKokkos, set_values)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    A.set_values(42.0);
    for(int i = 0; i < size * size; i++) {
        EXPECT_EQ(data[i], 42.0);
    }
    delete[] data;
}

// Test operator() access
TEST(Test_ViewCArrayKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCArrayKokkos<double> A(data, size, size, size);
    
    // Test 1D access
    data[0] = 1.0;
    EXPECT_EQ(A(0), 1.0);
    
    // Test 3D access
    data[size * size + size + 1] = 2.0;
    EXPECT_EQ(A(1, 1, 1), 2.0);
    
    // Test 5D access
    data[size * size * size * size + size * size * size + size * size + size + 1] = 3.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1), 3.0);
    
    // Test 7D access
    data[size * size * size * size * size * size + size * size * size * size * size + 
         size * size * size * size + size * size * size + size * size + size + 1] = 4.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1, 1, 1), 4.0);
    
    delete[] data;
}

// Test bounds checking
TEST(Test_ViewCArrayKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCArrayKokkos<double> A(data, size, size);
    
    // Test out of bounds access
    EXPECT_DEATH(A(size, size), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
    
    delete[] data;
}

// Test different types
TEST(Test_ViewCArrayKokkos, different_types)
{
    const int size = 10;
    
    // Test with int
    int* int_data = new int[size * size];
    ViewCArrayKokkos<int> A(int_data, size, size);
    A.set_values(42);
    EXPECT_EQ(int_data[0], 42);
    delete[] int_data;
    
    // Test with float
    float* float_data = new float[size * size];
    ViewCArrayKokkos<float> B(float_data, size, size);
    B.set_values(42.0f);
    EXPECT_FLOAT_EQ(float_data[0], 42.0f);
    delete[] float_data;
    
    // Test with bool
    bool* bool_data = new bool[size * size];
    ViewCArrayKokkos<bool> C(bool_data, size, size);
    C.set_values(true);
    EXPECT_EQ(bool_data[0], true);
    delete[] bool_data;
}

// Test RAII behavior
TEST(Test_ViewCArrayKokkos, raii)
{
    const int size = 10;
    double* data = new double[size * size];
    {
        ViewCArrayKokkos<double> A(data, size, size);
        A.set_values(42.0);
    } // A goes out of scope here
    // Data should still be accessible and unchanged
    EXPECT_EQ(data[0], 42.0);
    delete[] data;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
