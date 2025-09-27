#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create matrices of different dimensions
FMatrixKokkos<double> return_FMatrixKokkos(int dims, std::vector<int> sizes)
{
    switch(dims) {
        case 1:
            return FMatrixKokkos<double>(sizes[0], "A_1D_FMatrixKokkos");
        case 2:
            return FMatrixKokkos<double>(sizes[0], sizes[1], "A_2D_FMatrixKokkos");
        case 3:
            return FMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], "A_3D_FMatrixKokkos");
        case 4:
            return FMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], "A_4D_FMatrixKokkos");
        case 5:
            return FMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], "A_5D_FMatrixKokkos");
        case 6:
            return FMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], "A_6D_FMatrixKokkos");
        case 7:
            return FMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], "A_7D_FMatrixKokkos");
        default:
            return FMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_FMatrixKokkos, default_constructor)
{
    FMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.order(), 0);
    EXPECT_EQ(A.pointer(), nullptr);
}

// Test size function
TEST(Test_FMatrixKokkos, size)
{
    std::vector<int> sizes;
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        val *= dims*2;
        FMatrixKokkos<double> A = return_FMatrixKokkos(dims, sizes);
        EXPECT_EQ(val, A.size());
    }
}

// Test extent function
TEST(Test_FMatrixKokkos, extent)
{
    std::vector<int> sizes;
    int val = 1;  // Expected total length of data

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        val *= dims*2;
        FMatrixKokkos<double> A = return_FMatrixKokkos(dims, sizes);
        EXPECT_EQ(val, A.extent());
    }
}

// Test dims function
TEST(Test_FMatrixKokkos, dims)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FMatrixKokkos<double> A = return_FMatrixKokkos(dims, sizes);
        for(int j = 0; j < dims; j++){
            EXPECT_EQ(sizes[j], A.dims(j+1));
        }
    }
}

// Test order function
TEST(Test_FMatrixKokkos, order)
{
    std::vector<int> sizes;
    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FMatrixKokkos<double> A = return_FMatrixKokkos(dims, sizes);
        EXPECT_EQ(dims, A.order());
    }
}

// Test pointer function
TEST(Test_FMatrixKokkos, pointer)
{
    const int size = 100;
    FMatrixKokkos<double> A(size, "test_pointer");
    EXPECT_NE(nullptr, A.pointer());
}

// Test get_name function
TEST(Test_FMatrixKokkos, names)
{
    std::vector<int> sizes;
    std::vector<std::string> names = {
        "A_1D_FMatrixKokkos",
        "A_2D_FMatrixKokkos",
        "A_3D_FMatrixKokkos",
        "A_4D_FMatrixKokkos",
        "A_5D_FMatrixKokkos",
        "A_6D_FMatrixKokkos",
        "A_7D_FMatrixKokkos"
    };

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        FMatrixKokkos<double> A = return_FMatrixKokkos(dims, sizes);
        EXPECT_EQ(names[i], A.get_name());
    }
}

// Test late initialization
TEST(Test_FMatrixKokkos, late_init)
{
    std::vector<int> sizes;
    int val = 1;  // Expected total length of data

    FMatrixKokkos<double> A;

    for(int i = 0; i < 7; i++){
        int dims = i+1;
        sizes.push_back(dims*2);
        A = return_FMatrixKokkos(dims, sizes);
        val *= dims*2;
        EXPECT_EQ(val, A.size());
    }
}

// Test operator = overload
TEST(Test_FMatrixKokkos, eq_overload)
{
    const int size = 100;
    FMatrixKokkos<double> A(size, size);
    FMatrixKokkos<double> B(size, size);

    for(int i = 1; i <= size; i++){
        for(int j = 1; j <= size; j++){
            A(i,j) = i*size + j;
        }
    }

    B = A;

    for(int i = 1; i <= size; i++){
        for(int j = 1; j <= size; j++){
            EXPECT_EQ(i*size + j, B(i,j));
        }
    }
}

// Test set_values function
TEST(Test_FMatrixKokkos, set_values)
{
    const int size = 100;
    FMatrixKokkos<double> A(size, size);
    
    A.set_values(42.0);
    for(int i = 1; i <= size; i++){
        for(int j = 1; j <= size; j++){
            EXPECT_EQ(42.0, A(i,j));
        }
    }
}

// Test operator access
TEST(Test_FMatrixKokkos, operator_access)
{
    const int size = 10;
    FMatrixKokkos<double> A(size, size, size);
    
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
}

#ifndef NDEBUG
// Test bounds checking
TEST(Test_FMatrixKokkos, bounds_checking)
{
    const int size = 10;
    FMatrixKokkos<double> A(size, size);
    
    // Test valid access
    A(5,5) = 42.0;
    EXPECT_EQ(42.0, A(5,5));
    
    // Test invalid access - should throw
    EXPECT_DEATH(A(0,0), ".*");
}
#endif

// Test different types
TEST(Test_FMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test int
    FMatrixKokkos<int> A(size, size);
    A.set_values(42);
    EXPECT_EQ(42, A(5,5));
    
    // Test float
    FMatrixKokkos<float> B(size, size);
    B.set_values(42.0f);
    EXPECT_EQ(42.0f, B(5,5));
    
    // Test bool
    FMatrixKokkos<bool> C(size, size);
    C.set_values(true);
    EXPECT_EQ(true, C(5,5));
}

// Test Kokkos view access
TEST(Test_FMatrixKokkos, kokkos_view)
{
    const int size = 100;
    FMatrixKokkos<double> A(size, size);
    
    // Test view access
    auto view = A.get_kokkos_view();
    EXPECT_EQ(view.size(), size*size);

    A.set_values(42.0);
    
    // Verify values through array access
    for(int i = 1; i <= size; i++) {
        for(int j = 1; j <= size; j++) {
            EXPECT_EQ(A(i,j), 42.0);
        }
    }
}

// Test RAII behavior
TEST(Test_FMatrixKokkos, raii)
{
    {
        FMatrixKokkos<double> A(100, 100, "test_raii");
        A.set_values(42.0);
        EXPECT_EQ(A.size(), 10000);
        // A should be destroyed at end of scope
    }
    
    // Create new matrix to verify memory was freed
    FMatrixKokkos<double> B(100, 100, "test_raii_2");
    B.set_values(0.0);
    EXPECT_EQ(B.size(), 10000);
}
