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