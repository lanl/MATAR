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
    EXPECT_EQ(size, A.dims(1));
    EXPECT_EQ(size, A.dims(2));
    EXPECT_EQ(size, A.dims(3));
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
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewFMatrixKokkos<double> A(dev_data.data(), size, size);

    A.set_values(42.0);
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    for(int i = 0; i < size * size; i++){
        EXPECT_EQ(42.0, h(i));
    }
}

// Test operator access
TEST(Test_ViewFMatrixKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size*size*size];
    ViewFMatrixKokkos<double> A(data, size, size, size);
    
    // Test 3D access
    for(int i = 1; i <= size; i++){
        for(int j = 1; j <= size; j++){
            for(int k = 1; k <= size; k++){
                A(i,j,k) = i*100 + j*10 + k;
            }
        }
    }
    
    for(int i = 1; i <= size; i++){
        for(int j = 1; j <= size; j++){
            for(int k = 1; k <= size; k++){
                EXPECT_EQ(i*100 + j*10 + k, A(i,j,k));
            }
        }
    }
    delete[] data;
}

#ifndef NDEBUG

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
    EXPECT_DEATH(A(0,0), ".*");
    delete[] data;
}
#endif

// Test different types
TEST(Test_ViewFMatrixKokkos, different_types)
{
    const int size = 10;

    // Test int
    {
        Kokkos::View<int*> dev_data("int_data", size * size);
        ViewFMatrixKokkos<int> A(dev_data.data(), size, size);
        A.set_values(42);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(42, h(0));
    }

    // Test float
    {
        Kokkos::View<float*> dev_data("float_data", size * size);
        ViewFMatrixKokkos<float> B(dev_data.data(), size, size);
        B.set_values(42.0f);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(42.0f, h(0));
    }

    // Test bool
    {
        Kokkos::View<bool*> dev_data("bool_data", size * size);
        ViewFMatrixKokkos<bool> C(dev_data.data(), size, size);
        C.set_values(true);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(true, h(0));
    }
}

// Test RAII behavior
TEST(Test_ViewFMatrixKokkos, raii)
{
    Kokkos::View<double*> dev_data("dev_data", 100 * 100);
    {
        ViewFMatrixKokkos<double> A(dev_data.data(), 100, 100);
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }

    // Create new matrix using same backing memory
    ViewFMatrixKokkos<double> B(dev_data.data(), 100, 100);
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
}
