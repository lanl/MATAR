#include <gtest/gtest.h>
#include <matar.h>
#include <stdio.h>
#include <vector>

using namespace mtr; // matar namespace

// Helper function to create and return a ViewCMatrixKokkos object
ViewCMatrixKokkos<double> return_ViewCMatrixKokkos(int dims, std::vector<int> sizes, double* data)
{
    switch(dims) {
        case 1:
            return ViewCMatrixKokkos<double>(data, sizes[0]);
        case 2:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1]);
        case 3:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2]);
        case 4:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3]);
        case 5:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
        case 6:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]);
        case 7:
            return ViewCMatrixKokkos<double>(data, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        default:
            return ViewCMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_ViewCMatrixKokkos, default_constructor)
{
    ViewCMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test size method
TEST(Test_ViewCMatrixKokkos, size)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(A.size(), size * size);
    delete[] data;
}

// Test extent method
TEST(Test_ViewCMatrixKokkos, extent)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(A.extent(), size * size);
    delete[] data;
}

// Test dims method
TEST(Test_ViewCMatrixKokkos, dims)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(A.dims(1), size);
    EXPECT_EQ(A.dims(2), size);
    EXPECT_EQ(A.dims(3), size);
    delete[] data;
}

// Test order method
TEST(Test_ViewCMatrixKokkos, order)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCMatrixKokkos<double> A(data, size, size, size);
    EXPECT_EQ(A.order(), 3);
    delete[] data;
}

// Test pointer method
TEST(Test_ViewCMatrixKokkos, pointer)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCMatrixKokkos<double> A(data, size, size);
    EXPECT_EQ(A.pointer(), data);
    delete[] data;
}

// Test set_values method
TEST(Test_ViewCMatrixKokkos, set_values)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewCMatrixKokkos<double> A(dev_data.data(), size, size);
    A.set_values(42.0);
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    for(int i = 0; i < size * size; i++) {
        EXPECT_EQ(h(i), 42.0);
    }
}

#ifndef NDEBUG
// Test operator() access
TEST(Test_ViewCMatrixKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCMatrixKokkos<double> A(data, size, size, size);
    
    // Test 1D access
    EXPECT_DEATH(A(0), ".*");
    
    // Test 2D access
    EXPECT_DEATH(A(1, 1), ".*");
    
    // Test 3D access
    data[0] = 3.0;
    EXPECT_EQ(A(1, 1, 1), 3.0);
    
    // Test 5D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1), ".*");
    
    // Test 7D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1, 1, 1), ".*");
    
    delete[] data;
}

// Test bounds checking
TEST(Test_ViewCMatrixKokkos, bounds_checking)
{
    const int size = 10;
    double* data = new double[size * size];
    ViewCMatrixKokkos<double> A(data, size, size);
    
    // Test out of bounds access
    EXPECT_DEATH(A(0, 0), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
    
    delete[] data;
}
#endif

// Test different types
TEST(Test_ViewCMatrixKokkos, different_types)
{
    const int size = 10;

    // Test with int
    {
        Kokkos::View<int*> dev_data("int_data", size * size);
        ViewCMatrixKokkos<int> A(dev_data.data(), size, size);
        A.set_values(42);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(h(0), 42);
    }

    // Test with float
    {
        Kokkos::View<float*> dev_data("float_data", size * size);
        ViewCMatrixKokkos<float> B(dev_data.data(), size, size);
        B.set_values(42.0f);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_FLOAT_EQ(h(0), 42.0f);
    }

    // Test with bool
    {
        Kokkos::View<bool*> dev_data("bool_data", size * size);
        ViewCMatrixKokkos<bool> C(dev_data.data(), size, size);
        C.set_values(true);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(h(0), true);
    }
}

// Test RAII behavior
TEST(Test_ViewCMatrixKokkos, raii)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    {
        ViewCMatrixKokkos<double> A(dev_data.data(), size, size);
        A.set_values(42.0);
    } // A goes out of scope here
    // Data should still be accessible after A is destroyed
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    EXPECT_EQ(h(0), 42.0);
}

// Test copy constructor
TEST(Test_ViewCMatrixKokkos, copy_constructor)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewCMatrixKokkos<double> A(dev_data.data(), size, size);
    A.set_values(42.0);

    ViewCMatrixKokkos<double> B(A);
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    // Both A and B wrap the same pointer; verify values via mirror
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    EXPECT_EQ(h(0), 42.0);
}

// Test assignment operator
TEST(Test_ViewCMatrixKokkos, assignment_operator)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewCMatrixKokkos<double> A(dev_data.data(), size, size);
    A.set_values(42.0);

    ViewCMatrixKokkos<double> B;
    B = A;
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    EXPECT_EQ(h(0), 42.0);
}
