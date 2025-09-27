#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

DCArrayKokkos<double> return_DCArrayKokkos(int dims, std::vector<int> sizes)
{

    DCArrayKokkos<double> A;

    if(dims == 1){
        A = DCArrayKokkos<double>(sizes[0], "A_1D_DCArrayKokkos");
    }
    else if(dims == 2){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], "A_2D_DCArrayKokkos");
    }
    else if(dims == 3){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], sizes[2], "A_3D_DCArrayKokkos");
    }
    else if(dims == 4){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], "A_4D_DCArrayKokkos");
    }
    else if(dims == 5){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], "A_5D_DCArrayKokkos");
    }
    else if(dims == 6){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "A_6D_DCArrayKokkos");
    }
    else if(dims == 7){
        A = DCArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], "A_7D_DCArrayKokkos");
    }
    else{
        std::cout<<"Dims must be between 1 and 7 for DCArrayKokkos" << std::endl;
    }
    return A;
}


// Test size function
TEST(Test_DCArrayKokkos, size)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        DCArrayKokkos<double> A = return_DCArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.size());
    }
}


// Test extent function
TEST(Test_DCArrayKokkos, extent)
{
    std::vector<int> sizes; // Size of each dimension
    int val = 1; // Expected total length of data

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        DCArrayKokkos<double> A = return_DCArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.extent());
    }
}

// Test dims function
TEST(Test_DCArrayKokkos, dims)
{

    // Note: extend to other dims when initialized to zero

    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        DCArrayKokkos<double> A = return_DCArrayKokkos(dims, sizes);

        EXPECT_EQ(sizes[i], A.dims(i));
    }
}

// Test order function
TEST(Test_DCArrayKokkos, order)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){

        int dims = i+1;
        sizes.push_back(dims*2);
        DCArrayKokkos<double> A = return_DCArrayKokkos(dims, sizes);
        EXPECT_EQ(dims, A.order());
    }
}


// Test get name function
TEST(Test_DCArrayKokkos, names)
{
    std::vector<int> sizes;
    std::vector <std::string> names = {
        "A_1D_DCArrayKokkos",
        "A_2D_DCArrayKokkos",
        "A_3D_DCArrayKokkos",
        "A_4D_DCArrayKokkos",
        "A_5D_DCArrayKokkos",
        "A_6D_DCArrayKokkos",
        "A_7D_DCArrayKokkos"
    };

    for(int i = 0; i < 7; i++){

        int dims = i+1;
        sizes.push_back(dims*2);
        DCArrayKokkos<double> A = return_DCArrayKokkos(dims, sizes);
        EXPECT_EQ(names[i], A.get_name());
    }
}

// Add test for late initialization
TEST(Test_DCArrayKokkos, late_init)
{
    std::vector<int> sizes;  // Size of each dimension
    int val = 1;  // Expected total length of data

    DCArrayKokkos<double> A;

    for(int i = 0; i < 7; i++){

        int dims = i+1;

        sizes.push_back(dims*2);

        A = return_DCArrayKokkos(dims, sizes);
        val*= dims*2;

        EXPECT_EQ(val, A.size());
    }
}


// Add test for operator = overload
TEST(Test_DCArrayKokkos, eq_overload)
{
    const int size = 100;
    DCArrayKokkos<double> A(size, size);

    DCArrayKokkos<double> B(size, size);

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            A.host(i,j) = (double)i + (double)j;
        }
    }

    A.update_device();

    B = A;

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){

            EXPECT_EQ(B.host(i,j), (double)i + (double)j);
        }
    }
}

// Test set_values function on host
TEST(Test_DCArrayKokkos, set_values)
{
    const int size = 100;
    DCArrayKokkos<double> A(size, "test_array");
    A.set_values(42.0);
    A.update_host();

    for(int i = 0; i < size; i++) {
        EXPECT_EQ(A.host(i), 42.0);
    }
}

// Test operator() overloads for different dimensions
TEST(Test_DCArrayKokkos, operator_access)
{
    // Test 1D access
    DCArrayKokkos<double> A1(10, "test_1d");
    A1.host(5) = 42.0;
    EXPECT_EQ(A1.host(5), 42.0);
    
    // Test 2D access
    DCArrayKokkos<double> A2(10, 10, "test_2d");
    A2.host(5, 5) = 42.0;
    EXPECT_EQ(A2.host(5, 5), 42.0);
    
    // Test 3D access
    DCArrayKokkos<double> A3(10, 10, 10, "test_3d");
    A3.host(5, 5, 5) = 42.0;
    EXPECT_EQ(A3.host(5, 5, 5), 42.0);
    
    // Test 4D access
    DCArrayKokkos<double> A4(5, 5, 5, 5, "test_4d");
    A4.host(2, 2, 2, 2) = 42.0;
    EXPECT_EQ(A4.host(2, 2, 2, 2), 42.0);
    
    // Test 5D access
    DCArrayKokkos<double> A5(3, 3, 3, 3, 3, "test_5d");
    A5.host(1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A5.host(1, 1, 1, 1, 1), 42.0);
    
    // Test 6D access
    DCArrayKokkos<double> A6(2, 2, 2, 2, 2, 2, "test_6d");
    A6.host(1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A6.host(1, 1, 1, 1, 1, 1), 42.0);
    
    // Test 7D access
    DCArrayKokkos<double> A7(2, 2, 2, 2, 2, 2, 2, "test_7d");
    A7.host(1, 1, 1, 1, 1, 1, 1) = 42.0;
    EXPECT_EQ(A7.host(1, 1, 1, 1, 1, 1, 1), 42.0);
}

// Test host and device updates
TEST(Test_DCArrayKokkos, host_device_updates)
{
    const int size = 100;
    DCArrayKokkos<double> A(size, "test_updates");
    
    // Set values on host
    for(int i = 0; i < size; i++) {
        A(i) = static_cast<double>(i);
    }
    
    // Update device
    A.update_device();
    
    // Modify values on device
    FOR_ALL(i, 0, size, {
        A(i) = A(i) * 2.0;
    });
    
    // Update host
    A.update_host();
    
    // Verify values on host
    for(int i = 0; i < size; i++) {
        EXPECT_EQ(A.host(i), static_cast<double>(i) * 2.0);
    }
}

#ifndef NDEBUG
// Test bounds checking
TEST(Test_DCArrayKokkos, bounds_checking)
{
    // Test 1D bounds
    DCArrayKokkos<double> A1(10, "test_bounds_1d");
    EXPECT_DEATH(A1.host(10), "");
    
    // Test 2D bounds
    DCArrayKokkos<double> A2(10, 10, "test_bounds_2d");
    EXPECT_DEATH(A2.host(10, 5), "");
    EXPECT_DEATH(A2.host(5, 10), "");
    
    // Test 3D bounds
    DCArrayKokkos<double> A3(5, 5, 5, "test_bounds_3d");
    EXPECT_DEATH(A3.host(5, 2, 2), "");
    EXPECT_DEATH(A3.host(2, 5, 2), "");
    EXPECT_DEATH(A3.host(2, 2, 5), "");
}
#endif

// Test different data types
TEST(Test_DCArrayKokkos, different_types)
{
    // Test with int
    DCArrayKokkos<int> A_int(10, "test_int");
    A_int.set_values(42);
    A_int.update_host();
    for(int i = 0; i < 10; i++) {
        EXPECT_EQ(A_int.host(i), 42);
    }
    
    // Test with float
    DCArrayKokkos<float> A_float(10, "test_float");
    A_float.set_values(42.0f);
    A_float.update_host();
    for(int i = 0; i < 10; i++) {
        EXPECT_FLOAT_EQ(A_float.host(i), 42.0f);
    }
    
    // Test with bool
    DCArrayKokkos<bool> A_bool(10, "test_bool");
    A_bool.set_values(true);
    A_bool.update_host();
    for(int i = 0; i < 10; i++) {
        EXPECT_TRUE(A_bool.host(i));
    }
}
