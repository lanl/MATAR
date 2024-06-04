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
        A = CArrayKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "A_7D_CArrayKokkos");
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


// Test order function
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
