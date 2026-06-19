#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace

namespace {
// Helper: set two elements on device (KOKKOS_LAMBDA must not be in TEST body on NVCC)
inline void set_matrix_elements_1_4(DynamicMatrixKokkos<double>& matrix, double v1, double v4) {
    Kokkos::parallel_for("set_matrix_elems", 1, KOKKOS_LAMBDA(int) {
        matrix(1) = v1;
        matrix(4) = v4;
    });
    Kokkos::fence();
}
} // namespace

// Test default constructor
TEST(DynamicMatrixKokkosTest, DefaultConstructor) {
    DynamicMatrixKokkos<double> matrix;
    EXPECT_EQ(matrix.size(), 0);
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 0);
    EXPECT_EQ(matrix.order(), 0);
}

// Test constructor with initial size
TEST(DynamicMatrixKokkosTest, ConstructorWithSize) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");
    EXPECT_EQ(matrix.size(), 10);
    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10);
    EXPECT_EQ(matrix.order(), 1);
    EXPECT_EQ(matrix.get_name(), "test_matrix");
}

// Test push_back functionality
TEST(DynamicMatrixKokkosTest, PushBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");

    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 0);

    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);

    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 3);

    // DynamicMatrixKokkos is 1-indexed; flat view is 0-indexed
    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 1.0);
    EXPECT_DOUBLE_EQ(m(1), 2.0);
    EXPECT_DOUBLE_EQ(m(2), 3.0);
}

// Test pop_back functionality
TEST(DynamicMatrixKokkosTest, PopBack) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");

    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);

    EXPECT_EQ(matrix.size(), 5);

    matrix.pop_back();
    EXPECT_EQ(matrix.dims(1), 2);

    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.dims(1), 1);

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix.get_kokkos_view());
    EXPECT_DOUBLE_EQ(m(0), 1.0);
}

// Test set_values functionality
TEST(DynamicMatrixKokkosTest, SetValues) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");

    matrix.set_values(42.0, 10);
#ifndef NDEBUG
    EXPECT_DEATH(matrix.set_values(42.0, 11),"");
#endif

    auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix.get_kokkos_view());
    for (size_t i = 0; i < matrix.dims(1); i++) {
        EXPECT_DOUBLE_EQ(m(i), 42.0);
    }
}

// Test dimension management
TEST(DynamicMatrixKokkosTest, DimensionManagement) {
    DynamicMatrixKokkos<double> matrix(10, "test_matrix");

    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10);
    EXPECT_EQ(matrix.order(), 1);

    matrix.push_back(1.0);
    matrix.push_back(2.0);

    EXPECT_EQ(matrix.dims(1), 2);
    EXPECT_EQ(matrix.dims_max(1), 10);

    matrix.pop_back();
    matrix.pop_back();

    EXPECT_EQ(matrix.dims(1), 0);
    EXPECT_EQ(matrix.dims_max(1), 10);
}

// Test name management
TEST(DynamicMatrixKokkosTest, NameManagement) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");
    EXPECT_EQ(matrix.get_name(), "test_matrix");

    DynamicMatrixKokkos<double> matrix2(5, "another_matrix");
    EXPECT_EQ(matrix2.get_name(), "another_matrix");
}

// Test size and extent
TEST(DynamicMatrixKokkosTest, SizeAndExtent) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");

    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);

    matrix.push_back(1.0);
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);

    matrix.pop_back();
    EXPECT_EQ(matrix.size(), 5);
    EXPECT_EQ(matrix.extent(), 5);
}

// Test matrix access and modification
TEST(DynamicMatrixKokkosTest, MatrixAccess) {
    DynamicMatrixKokkos<double> matrix(5, "test_matrix");

    matrix.push_back(1.0);
    matrix.push_back(2.0);
    matrix.push_back(3.0);
    matrix.push_back(4.0);
    matrix.push_back(5.0);

    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix.get_kokkos_view());
        EXPECT_DOUBLE_EQ(m(0), 1.0);
        EXPECT_DOUBLE_EQ(m(1), 2.0);
        EXPECT_DOUBLE_EQ(m(2), 3.0);
        EXPECT_DOUBLE_EQ(m(3), 4.0);
        EXPECT_DOUBLE_EQ(m(4), 5.0);
    }

    // Modify individual elements via device kernel
    set_matrix_elements_1_4(matrix, 10.0, 50.0);

    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix.get_kokkos_view());
        EXPECT_DOUBLE_EQ(m(0), 10.0);
        EXPECT_DOUBLE_EQ(m(3), 50.0);
    }
}

// Test matrix operations with different data types
TEST(DynamicMatrixKokkosTest, DifferentDataTypes) {
    DynamicMatrixKokkos<float> matrix_float(5, "float_matrix");
    matrix_float.set_values(42.0f, 5);
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix_float.get_kokkos_view());
        EXPECT_FLOAT_EQ(m(0), 42.0f);
    }

    DynamicMatrixKokkos<int> matrix_int(5, "int_matrix");
    matrix_int.set_values(42, 5);
    {
        auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, matrix_int.get_kokkos_view());
        EXPECT_EQ(m(2), 42);
    }
}
