#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

// Helper function to create and return a CMatrixKokkos object
CMatrixKokkos<double> return_CMatrixKokkos(int dims, std::vector<int> sizes, const std::string& tag_string = "test_matrix")
{
    switch(dims) {
        case 1:
            return CMatrixKokkos<double>(sizes[0], tag_string);
        case 2:
            return CMatrixKokkos<double>(sizes[0], sizes[1], tag_string);
        case 3:
            return CMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], tag_string);
        case 4:
            return CMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], tag_string);
        case 5:
            return CMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], tag_string);
        case 6:
            return CMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], tag_string);
        case 7:
            return CMatrixKokkos<double>(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6], tag_string);
        default:
            return CMatrixKokkos<double>();
    }
}

// Test default constructor
TEST(Test_CMatrixKokkos, default_constructor)
{
    CMatrixKokkos<double> A;
    EXPECT_EQ(A.size(), 0);
    EXPECT_EQ(A.extent(), 0);
    EXPECT_EQ(A.order(), 0);
}

// Test 1D constructor
TEST(Test_CMatrixKokkos, constructor_1d)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, "test_matrix");
    EXPECT_EQ(A.size(), size);
    EXPECT_EQ(A.extent(), size);
    EXPECT_EQ(A.order(), 1);
    EXPECT_EQ(A.dims(0), size);
}

// Test 2D constructor
TEST(Test_CMatrixKokkos, constructor_2d)
{
    const int size1 = 10;
    const int size2 = 20;
    CMatrixKokkos<double> A(size1, size2, "test_matrix");
    EXPECT_EQ(A.size(), size1 * size2);
    EXPECT_EQ(A.extent(), size1 * size2);
    EXPECT_EQ(A.order(), 2);
    EXPECT_EQ(A.dims(0), size1);
    EXPECT_EQ(A.dims(1), size2);
}

// Test 3D constructor
TEST(Test_CMatrixKokkos, constructor_3d)
{
    const int size1 = 10;
    const int size2 = 20;
    const int size3 = 30;
    CMatrixKokkos<double> A(size1, size2, size3, "test_matrix");
    EXPECT_EQ(A.size(), size1 * size2 * size3);
    EXPECT_EQ(A.extent(), size1 * size2 * size3);
    EXPECT_EQ(A.order(), 3);
    EXPECT_EQ(A.dims(0), size1);
    EXPECT_EQ(A.dims(1), size2);
    EXPECT_EQ(A.dims(2), size3);
}

// Test get_name method
TEST(Test_CMatrixKokkos, get_name)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, "test_matrix");
    EXPECT_EQ(A.get_name(), "test_matrix");
}

// Test set_values method
TEST(Test_CMatrixKokkos, set_values)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    // Check values on host
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            EXPECT_EQ(A(i, j), 42.0);
        }
    }
}

// Test operator() access
TEST(Test_CMatrixKokkos, operator_access)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, size, size, "test_matrix");
    
    // Test 1D access
    A(0) = 1.0;
    EXPECT_EQ(A(0), 1.0);
    
    // Test 2D access
    A(1, 1) = 2.0;
    EXPECT_EQ(A(1, 1), 2.0);
    
    // Test 3D access
    A(1, 1, 1) = 3.0;
    EXPECT_EQ(A(1, 1, 1), 3.0);
    
    // Test 5D access
    A(1, 1, 1, 1, 1) = 4.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1), 4.0);
    
    // Test 7D access
    A(1, 1, 1, 1, 1, 1, 1) = 5.0;
    EXPECT_EQ(A(1, 1, 1, 1, 1, 1, 1), 5.0);
}

// Test bounds checking
TEST(Test_CMatrixKokkos, bounds_checking)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, size, "test_matrix");
    
    // Test out of bounds access
    EXPECT_DEATH(A(size, size), ".*");
    EXPECT_DEATH(A(10000, 10000), ".*");
}

// Test different types
TEST(Test_CMatrixKokkos, different_types)
{
    const int size = 10;
    
    // Test with int
    CMatrixKokkos<int> A(size, size, "test_matrix_int");
    A.set_values(42);
    EXPECT_EQ(A(0, 0), 42);
    
    // Test with float
    CMatrixKokkos<float> B(size, size, "test_matrix_float");
    B.set_values(42.0f);
    EXPECT_FLOAT_EQ(B(0, 0), 42.0f);
    
    // Test with bool
    CMatrixKokkos<bool> C(size, size, "test_matrix_bool");
    C.set_values(true);
    EXPECT_EQ(C(0, 0), true);
}

// Test RAII behavior
TEST(Test_CMatrixKokkos, raii)
{
    const int size = 10;
    {
        CMatrixKokkos<double> A(size, size, "test_matrix");
        A.set_values(42.0);
        EXPECT_EQ(A(0, 0), 42.0);
    } // A goes out of scope here

    
}

// Test copy constructor
TEST(Test_CMatrixKokkos, copy_constructor)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    CMatrixKokkos<double> B(A);
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
}

// Test assignment operator
TEST(Test_CMatrixKokkos, assignment_operator)
{
    const int size = 10;
    CMatrixKokkos<double> A(size, size, "test_matrix");
    A.set_values(42.0);
    
    CMatrixKokkos<double> B;
    B = A;
    EXPECT_EQ(B.size(), A.size());
    EXPECT_EQ(B.extent(), A.extent());
    EXPECT_EQ(B.order(), A.order());
    EXPECT_EQ(B(0, 0), A(0, 0));
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
