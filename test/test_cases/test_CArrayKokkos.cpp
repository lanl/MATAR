#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

CArrayKokkos<double> return_CArrayKokkos(int dims, std::vector<int> sizes)
{

    CArrayKokkos<double> A;

    if(dims == 1){
        A = CArrayKokkos<double>(sizes[0], "A_1D_CArrayKokkos");
    }
    else if(dims == 2){
        A = CArrayKokkos<double>(sizes[0], sizes[1], "A_2D_CArrayKokkos");
    }
    else if(dims == 3){
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], "A_3D_CArrayKokkos");
    }
    else if(dims == 4){
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], "A_4D_CArrayKokkos");
    }
    else if(dims == 5){
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], "A_5D_CArrayKokkos");
    }
    else if(dims == 6){
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "A_6D_CArrayKokkos");
    }
    else if(dims == 7){
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], "A_7D_CArrayKokkos");
    }
    else{
        std::cout<<"Dims must be between 1 and 7 for CArrayKokkos" << std::endl;
    }
    return A;
}

// Test size function
TEST(Test_CArrayKokkos, size)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.size());
    }
}


// Test extent function
TEST(Test_CArrayKokkos, extent)
{
    std::vector<int> sizes; // Size of each dimension
    int val = 1; // Expected total length of data

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.extent());
    }
}

// Test dims function
TEST(Test_CArrayKokkos, dims)
{

    // Note: extend to other dims when initialized to zero

    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);

        EXPECT_EQ(sizes[i], A.dims(i));
    }
}

// Test order function
TEST(Test_CArrayKokkos, order)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){

        int dims = i+1;
        sizes.push_back(dims*2);
        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);
        EXPECT_EQ(dims, A.order());
    }
}


// Test pointer function
TEST(Test_CArrayKokkos, pointer)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){

        int dims = i+1;
        sizes.push_back(dims*2);
        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);
        auto a = A.get_kokkos_view();

        EXPECT_EQ(&a[0], A.pointer());
    }
}

// Test get name function
TEST(Test_CArrayKokkos, names)
{
    std::vector<int> sizes;
    std::vector <std::string> names = {
        "A_1D_CArrayKokkos",
        "A_2D_CArrayKokkos",
        "A_3D_CArrayKokkos",
        "A_4D_CArrayKokkos",
        "A_5D_CArrayKokkos",
        "A_6D_CArrayKokkos",
        "A_7D_CArrayKokkos"
    };

    for(int i = 0; i < 7; i++){

        int dims = i+1;
        sizes.push_back(dims*2);
        CArrayKokkos<double> A = return_CArrayKokkos(dims, sizes);
        EXPECT_EQ(names[i], A.get_name());
    }
}

// Add test for late initialization
TEST(Test_CArrayKokkos, late_init)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    CArrayKokkos<double> A;

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        A = return_CArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.size());
    }
}


// Add test for operator = overload
TEST(Test_CArrayKokkos, eq_overload)
{
    const int size = 100;
    CArrayKokkos<double> A(size, size);

    CArrayKokkos<double> B(size, size);

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
TEST(Test_CArrayKokkos, set_values)
{
    const int size = 100;
    CArrayKokkos<double> A(size, "test_array");
    A.set_values(42.0);
    
    for(int i = 0; i < size; i++) {
        EXPECT_EQ(A(i), 42.0);
    }
}

// Test operator() overloads for different dimensions
TEST(Test_CArrayKokkos, operator_access)
{
    // Test 1D access
    CArrayKokkos<double> A1(10, "test_1d");
    A1(5) = 42.0;
    EXPECT_EQ(A1(5), 42.0);
    
    // Test 2D access
    CArrayKokkos<double> A2(10, 10, "test_2d");
    A2(5, 5) = 42.0;
    EXPECT_EQ(A2(5, 5), 42.0);
    
    // Test 3D access
    CArrayKokkos<double> A3(10, 10, 10, "test_3d");
    A3(5, 5, 5) = 42.0;
    EXPECT_EQ(A3(5, 5, 5), 42.0);
    
    // Test 4D access
    CArrayKokkos<double> A4(5, 5, 5, 5, "test_4d");
    A4(2, 2, 2, 2) = 42.0;
    EXPECT_EQ(A4(2, 2, 2, 2), 42.0);
    
    // Test 5D access
    CArrayKokkos<double> A5(3, 3, 3, 3, 3, "test_5d");
    A5(1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A5(1, 1, 1, 1, 1), 42.0);
    
    // Test 6D access
    CArrayKokkos<double> A6(2, 2, 2, 2, 2, 2, "test_6d");
    A6(1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A6(1, 1, 1, 1, 1, 1), 42.0);
    
    // Test 7D access
    CArrayKokkos<double> A7(2, 2, 2, 2, 2, 2, 2, "test_7d");
    A7(1, 1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A7(1, 1, 1, 1, 1, 1, 1), 42.0);
}

// Test bounds checking
TEST(Test_CArrayKokkos, bounds_checking)
{
    // Test 1D bounds
    CArrayKokkos<double> A1(10, "test_bounds_1d");
    EXPECT_DEATH(A1(110), ".*");
    
    // Test 2D bounds
    CArrayKokkos<double> A2(10, 10, "test_bounds_2d");
    EXPECT_DEATH(A2(100, 5), ".*");
    EXPECT_DEATH(A2(50, 10), ".*");
    
    // Test 3D bounds
    CArrayKokkos<double> A3(5, 5, 5, "test_bounds_3d");
    EXPECT_DEATH(A3(50, 2, 2), ".*");
    EXPECT_DEATH(A3(20, 5, 2), ".*");
    EXPECT_DEATH(A3(26, 2, 5), ".*");
}

// Test different data types
TEST(Test_CArrayKokkos, different_types)
{
    // Test with int
    CArrayKokkos<int> A_int(10, "test_int");
    A_int.set_values(42);
    for(int i = 0; i < 10; i++) {
        EXPECT_EQ(A_int(i), 42);
    }
    
    // Test with float
    CArrayKokkos<float> A_float(10, "test_float");
    A_float.set_values(42.0f);
    for(int i = 0; i < 10; i++) {
        EXPECT_FLOAT_EQ(A_float(i), 42.0f);
    }
    
    // Test with bool
    CArrayKokkos<bool> A_bool(10, "test_bool");
    A_bool.set_values(true);
    for(int i = 0; i < 10; i++) {
        EXPECT_TRUE(A_bool(i));
    }
}
