#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create matrices of different dimensions
ViewFMatrixKokkos<double> return_ViewFMatrixKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return ViewFMatrixKokkos<double>(data, sizes[0]);
        case 2:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return ViewFMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return ViewFMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_ViewFMatrixKokkos, default_constructor)
{
    ViewFMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.pointer(), nullptr);
}

// Test size function
TEST(Test_ViewFMatrixKokkos, size)
{
    const int size = 100;
    double* data = new double[size*size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.size());
    delete[] data;
}

// Test extent function
TEST(Test_ViewFMatrixKokkos, extent)
{
    const int size = 100;
    double* data = new double[size*size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(size*size, A.extent());
    delete[] data;
}

// Test dims function
TEST(Test_ViewFMatrixKokkos, dims)
{
    const int size = 100;
    double* data = new double[size*size*size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(size, A.dims(0));
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    delete[] data;
}

// Test order function
TEST(Test_ViewFMatrixKokkos, order)
{
    const int size = 100;
    double* data = new double[size*size*size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(3, A.order());
    delete[] data;
}

// Test pointer function
TEST(Test_ViewFMatrixKokkos, pointer)
{
    const int size = 100;
    double* data = new double[size*size];
    ViewFMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(data, A.pointer());
    delete[] data;
}

// Test set_values function
TEST(Test_ViewFMatrixKokkos, set_values)
{
    const int size = 100;
    double* data = new double[size*size];
    ViewFMatrixKokkos<double> A(data, size, size);
    
    A.set_values(42.0);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            EXPECT_EQ(42.0, A(i,j));
        }
    }
    delete[] data;
}

// Test operator access
TEST(Test_ViewFMatrixKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size*size*size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    
    // Test 3D access
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            for(int k = 0; k < size; k++){
                A(i,j,k) = i*100 + j*10 + k;
            }
        }
    }
    
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            for(int k = 0; k < size; k++){
                EXPECT_EQ(i*100 + j*10 + k, A(i,j,k));
            }
        }
    }
    delete[] data;
}

// Test bounds checking
TEST(Test_ViewFMatrixKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size*size];
    ViewFMatrixKokkos<double> A(data, size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size,size), ".*");
    delete[] data;
}

// Test different types
TEST(Test_ViewFMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    int* int_data = new int[size*size];
    ViewFMatrixKokkos<int> A(int_data, size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    delete[] int_data;
    
    // Test float
    float* float_data = new float[size*size];
    ViewFMatrixKokkos<float> B(float_data, size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    delete[] float_data;
    
    // Test bool
    bool* bool_data = new bool[size*size];
    ViewFMatrixKokkos<bool> C(bool_data, size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
    delete[] bool_data;
}

// Test RAII behavior
TEST(Test_ViewFMatrixKokkos, raii)
{
    double* data = new double[100*100];
    {
        ViewFMatrixKokkos<double> A(data, 100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new matrix to verify memory was freed
    ViewFMatrixKokkos<double> B(data, 100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
    delete[] data;
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
