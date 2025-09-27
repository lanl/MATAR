#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

FArrayKokkos<double> return_FArrayKokkos(int dims, std::vector<int> sizes)
{
    FArrayKokkos<double> A;

    if(dims == 1){
        A = FArrayKokkos<double>(sizes[0], "A_1D_FArrayKokkos");
    }
    else if(dims == 2){
        A = FArrayKokkos<double>(sizes[0], sizes[1], "A_2D_FArrayKokkos");
    }
    else if(dims == 3){
        A = FArrayKokkos<double>(sizes[0], sizes[1], sizes[2], "A_3D_FArrayKokkos");
    }
    else if(dims == 4){
        A = FArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], "A_4D_FArrayKokkos");
    }
    else if(dims == 5){
        A = FArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], "A_5D_FArrayKokkos");
    }
    else if(dims == 6){
        A = FArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "A_6D_FArrayKokkos");
    }
    else if(dims == 7){
        A = FArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], "A_7D_FArrayKokkos");
    }
    else{
        std::cout<<"Dims must be between 1 and 7 for FArrayKokkos" << std::endl;
    }
    return A;
}

// Test size function
TEST(Test_FArrayKokkos, size)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        val*= dims*2;
        EXPECT_EQ(val, A.size());
    }
}

// Test extent function
TEST(Test_FArrayKokkos, extent)
{
    std::vector<int> sizes; // Size of each dimension
    int val = 1; // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        val*= dims*2;
        EXPECT_EQ(val, A.extent());
    }
}

// Test dims function
TEST(Test_FArrayKokkos, dims)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        EXPECT_EQ(sizes[i], A.dims(i));
    }
}

// Test order function
TEST(Test_FArrayKokkos, order)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        EXPECT_EQ(dims, A.order());
    }
}

// Test pointer function
TEST(Test_FArrayKokkos, pointer)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        auto a = A.get_kokkos_view();
        EXPECT_EQ(&a[0], A.pointer());
    }
}

// Test get name function
TEST(Test_FArrayKokkos, names)
{
    std::vector<int> sizes;
    std::vector <std::string> names = {
        "A_1D_FArrayKokkos",
        "A_2D_FArrayKokkos",
        "A_3D_FArrayKokkos",
        "A_4D_FArrayKokkos",
        "A_5D_FArrayKokkos",
        "A_6D_FArrayKokkos",
        "A_7D_FArrayKokkos"
    };

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FArrayKokkos<double> A = return_FArrayKokkos(dims, sizes);
        EXPECT_EQ(names[i], A.get_name());
    }
}

// Add test for late initialization
TEST(Test_FArrayKokkos, late_init)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    FArrayKokkos<double> A;

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        A = return_FArrayKokkos(dims, sizes);
        val*= dims*2;
        EXPECT_EQ(val, A.size());
    }
}

// Add test for operator = overload
TEST(Test_FArrayKokkos, eq_overload)
{
    const int size = 100;
    FArrayKokkos<double> A(size, size);
    FArrayKokkos<double> B(size, size);

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            A(i,j) = (double)i + (double)j;
        }
    }

    B = A;

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            EXPECT_EQ(B(i,j), (double)i + (double)j);
        }
    }
}

// Test set_values function
TEST(Test_FArrayKokkos, set_values)
{
    const int size = 100;
    FArrayKokkos<double> A(size, "test_array");
    A.set_values(42.0);
    
    for(int i = 0; i < size; i++) {
        EXPECT_EQ(A(i), 42.0);
    }
}

// Test operator() overloads for different dimensions
TEST(Test_FArrayKokkos, operator_access)
{
    // Test 1D access
    FArrayKokkos<double> A1(10, "test_1d");
    A1(5) = 42.0;
    EXPECT_EQ(A1(5), 42.0);
    
    // Test 2D access
    FArrayKokkos<double> A2(10, 10, "test_2d");
    A2(5, 5) = 42.0;
    EXPECT_EQ(A2(5, 5), 42.0);
    
    // Test 3D access
    FArrayKokkos<double> A3(10, 10, 10, "test_3d");
    A3(5, 5, 5) = 42.0;
    EXPECT_EQ(A3(5, 5, 5), 42.0);
    
    // Test 4D access
    FArrayKokkos<double> A4(5, 5, 5, 5, "test_4d");
    A4(2, 2, 2, 2) = 42.0;
    EXPECT_EQ(A4(2, 2, 2, 2), 42.0);
    
    // Test 5D access
    FArrayKokkos<double> A5(3, 3, 3, 3, 3, "test_5d");
    A5(1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A5(1, 1, 1, 1, 1), 42.0);
    
    // Test 6D access
    FArrayKokkos<double> A6(2, 2, 2, 2, 2, 2, "test_6d");
    A6(1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A6(1, 1, 1, 1, 1, 1), 42.0);
    
    // Test 7D access
    FArrayKokkos<double> A7(2, 2, 2, 2, 2, 2, 2, "test_7d");
    A7(1, 1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A7(1, 1, 1, 1, 1, 1, 1), 42.0);
}

#ifndef NDEBUG
// Test bounds checking
TEST(Test_FArrayKokkos, bounds_checking)
{
    // Test 1D bounds
    FArrayKokkos<double> A1(10, "test_bounds_1d");
    EXPECT_DEATH(A1(-1), ".*");
    
    // Test 2D bounds
    FArrayKokkos<double> A2(10, 10, "test_bounds_2d");
    EXPECT_DEATH(A2(11, 5), ".*");
    EXPECT_DEATH(A2(0, 10), ".*");
    
    // Test 3D bounds
    FArrayKokkos<double> A3(5, 5, 5, "test_bounds_3d");
    EXPECT_DEATH(A3(6, 2, 2), ".*");
    EXPECT_DEATH(A3(6, 5, 2), ".*");
    EXPECT_DEATH(A3(6, 2, 5), ".*");
}
#endif

// Test different data types
TEST(Test_FArrayKokkos, different_types)
{
    // Test with int
    FArrayKokkos<int> A_int(10, "test_int");
    A_int.set_values(42);
    for(int i = 0; i < 10; i++) {
        EXPECT_EQ(A_int(i), 42);
    }
    
    // Test with float
    FArrayKokkos<float> A_float(10, "test_float");
    A_float.set_values(42.0f);
    for(int i = 0; i < 10; i++) {
        EXPECT_FLOAT_EQ(A_float(i), 42.0f);
    }
    
    // Test with bool
    FArrayKokkos<bool> A_bool(10, "test_bool");
    A_bool.set_values(true);
    for(int i = 0; i < 10; i++) {
        EXPECT_TRUE(A_bool(i));
    }
}

// Test default constructor
TEST(Test_FArrayKokkos, default_constructor)
{
    FArrayKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.pointer(), nullptr);
}

// Test Kokkos view access
TEST(Test_FArrayKokkos, kokkos_view)
{
    const int size = 100;
    FArrayKokkos<double> A(size, "test_view");
    
    // Test view access
    auto view = A.get_kokkos_view();
    EXPECT_EQ(view.size(), size);
    
    // Test view modification
    Kokkos::parallel_for("SetValues", size, KOKKOS_LAMBDA(const int i) {
        view(i) = 42.0;
    });
    
    // Verify values through array access
    for(int i = 0; i < size; i++) {
        EXPECT_EQ(A(i), 42.0);
    }
}

// Test RAII behavior
TEST(Test_FArrayKokkos, raii)
{
    {
        FArrayKokkos<double> A(100, "test_raii");
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 100);
        // A should be destroyed at end of scope
    }
    
    // Create new array to verify memory was freed
    FArrayKokkos<double> B(100, "test_raii_2");
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 100);
}
