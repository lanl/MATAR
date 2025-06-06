#include <gtest/gtest.h>
#include <matar.h>
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create arrays of different dimensions
ViewFArrayKokkos<double> return_ViewFArrayKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return ViewFArrayKokkos<double>(data, sizes[0]);
        case 2:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return ViewFArrayKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return ViewFArrayKokkos<double>();
    }
}

// Test default constructor
TEST(Test_ViewFArrayKokkos, default_constructor)
{
    ViewFArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.pointer(), nullptr);
}

// Test size function
TEST(Test_ViewFArrayKokkos, size)
{
    std::vector<int> sizes;
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        val *= dims*2;
        
        // Allocate data array
        double* data = new double[val];
        ViewFArrayKokkos<double> A = return_ViewFArrayKokkos(dims, sizes, data);
        EXPECT_EQ(val, A.size());
        delete[] data;
    }
}

// Test extent function
TEST(Test_ViewFArrayKokkos, extent)
{
    std::vector<int> sizes;
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        val *= dims*2;
        
        // Allocate data array
        double* data = new double[val];
        ViewFArrayKokkos<double> A = return_ViewFArrayKokkos(dims, sizes, data);
        EXPECT_EQ(val, A.extent());
        delete[] data;
    }
}

// Test dims function
TEST(Test_ViewFArrayKokkos, dims)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        
        // Allocate data array
        double* data = new double[1]; // Size doesn't matter for this test
        ViewFArrayKokkos<double> A = return_ViewFArrayKokkos(dims, sizes, data);
        for(int j = 0; j < dims; j++){
            EXPECT_EQ(sizes[j], A.dims(j));
        }
        delete[] data;
    }
}

// Test order function
TEST(Test_ViewFArrayKokkos, order)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        
        // Allocate data array
        double* data = new double[1]; // Size doesn't matter for this test
        ViewFArrayKokkos<double> A = return_ViewFArrayKokkos(dims, sizes, data);
        EXPECT_EQ(dims, A.order());
        delete[] data;
    }
}

// Test pointer function
TEST(Test_ViewFArrayKokkos, pointer)
{
    const int size = 100;
    double* data = new double[size];
    ViewFArrayKokkos<double> A(data, size);
    EXPECT_EQ(data, A.pointer());
    delete[] data;
}

// Test set_values function
TEST(Test_ViewFArrayKokkos, set_values)
{
    const int size = 100;
    double* data = new double[size];
    ViewFArrayKokkos<double> A(data, size);
    
    A.set_values(42.0);
    for(int i = 0; i < size; i++){
        EXPECT_EQ(42.0, data[i]);
    }
    delete[] data;
}

// Test operator access
TEST(Test_ViewFArrayKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size*size*size];
    ViewFArrayKokkos<double> A(data, size, size, size);
    
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
TEST(Test_ViewFArrayKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size];
    ViewFArrayKokkos<double> A(data, size);
    
    // Test valid access
    A(5) = 42.0;
    EXPECT_EQ(42.0, A(5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(size), ".*");
    delete[] data;
}

// Test different types
TEST(Test_ViewFArrayKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    int* int_data = new int[size];
    ViewFArrayKokkos<int> A(int_data, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5));
    delete[] int_data;
    
    // Test float
    float* float_data = new float[size];
    ViewFArrayKokkos<float> B(float_data, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5));
    delete[] float_data;
    
    // Test bool
    bool* bool_data = new bool[size];
    ViewFArrayKokkos<bool> C(bool_data, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5));
    delete[] bool_data;
}
