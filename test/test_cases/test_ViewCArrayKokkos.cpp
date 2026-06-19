#include <gtest/gtest.h>
#include <matar.h>
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

// Test set_values method
TEST(Test_ViewCArrayKokkos, set_values)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    ViewCArrayKokkos<double> A(dev_data.data(), size, size);
    A.set_values(42.0);
    Kokkos::fence();
    auto host_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    for(int i = 0; i < size * size; i++) {
        EXPECT_EQ(host_data(i), 42.0);
    }
}

#ifndef NDEBUG
// Test operator() access
TEST(Test_ViewCArrayKokkos, operator_access)
{
    const int size = 10;
    double* data = new double[size * size * size];
    ViewCArrayKokkos<double> A(data, size, size, size);
    
    // Test 1D access
    data[0] = 1.0;
    EXPECT_DEATH(A(0), ".*");
    
    // Test 3D access
    data[size * size + size + 1] = 2.0;
    EXPECT_EQ(A(1, 1, 1), 2.0);
    
    // Test 5D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1), ".*");
    
    // Test 7D access
    EXPECT_DEATH(A(1, 1, 1, 1, 1, 1, 1), ".*");
    
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
#endif

// Test different types
TEST(Test_ViewCArrayKokkos, different_types)
{
    const int size = 10;

    // Test with int
    {
        Kokkos::View<int*> dev_data("int_data", size * size);
        ViewCArrayKokkos<int> A(dev_data.data(), size, size);
        A.set_values(42);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(h(0), 42);
    }

    // Test with float
    {
        Kokkos::View<float*> dev_data("float_data", size * size);
        ViewCArrayKokkos<float> B(dev_data.data(), size, size);
        B.set_values(42.0f);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_FLOAT_EQ(h(0), 42.0f);
    }

    // Test with bool
    {
        Kokkos::View<bool*> dev_data("bool_data", size * size);
        ViewCArrayKokkos<bool> C(dev_data.data(), size, size);
        C.set_values(true);
        Kokkos::fence();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
        EXPECT_EQ(h(0), true);
    }
}

// Test RAII behavior
TEST(Test_ViewCArrayKokkos, raii)
{
    const int size = 10;
    Kokkos::View<double*> dev_data("dev_data", size * size);
    {
        ViewCArrayKokkos<double> A(dev_data.data(), size, size);
        A.set_values(42.0);
    } // A goes out of scope here
    // Data should still be accessible via mirror after A is destroyed
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dev_data);
    EXPECT_EQ(h(0), 42.0);
}
