#include "matar.h"
#include "gtest/gtest.h"
#include <stdio.h>

using namespace mtr; // matar namespace


class DDynamicRaggedRightArrayKokkosTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code for all tests
        dim1 = 4;  // number of rows
        dim2 = 4;  // maximum number of columns per row
    }

    void TearDown() override {
        // Common cleanup code for all tests
    }

    size_t dim1, dim2;
};

TEST_F(DDynamicRaggedRightArrayKokkosTest, Constructor) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Verify dimensions
    EXPECT_EQ(array.dim1(), dim1);
    EXPECT_EQ(array.dim2(), dim2);
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, StrideManagement) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Test stride access
    for (size_t i = 0; i < dim1; i++) {
        EXPECT_EQ(array.stride(i), 0);
    }

    // Test stride_host access
    for (size_t i = 0; i < dim1; i++) {
        EXPECT_EQ(array.stride_host(i), 0);
    }
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, ValueAccess) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Set some values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(2, 0) = 4.0;
    array(2, 1) = 5.0;
    array(3, 0) = 6.0;

    // Test value access
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 4.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(array(3, 0), 6.0);

    // Test host value access
    EXPECT_DOUBLE_EQ(array.host(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array.host(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array.host(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array.host(2, 0), 4.0);
    EXPECT_DOUBLE_EQ(array.host(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(array.host(3, 0), 6.0);
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, SetValues) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Set some initial values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(2, 0) = 4.0;
    array(2, 1) = 5.0;
    array(3, 0) = 6.0;

    // Set all values to 1.0
    array.set_values(1.0);

    // Verify values
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 1.0);
    EXPECT_DOUBLE_EQ(array(3, 0), 1.0);
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, SetValuesSparse) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Set some initial values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(2, 0) = 4.0;
    array(2, 1) = 5.0;
    array(3, 0) = 6.0;

    // Set values to 1.0 using sparse method
    array.set_values_sparse(1.0);

    // Verify values
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 1.0);
    EXPECT_DOUBLE_EQ(array(3, 0), 1.0);
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, UpdateFunctions) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Set some values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(2, 0) = 4.0;
    array(2, 1) = 5.0;
    array(3, 0) = 6.0;

    // Test update functions
    array.update_host();
    array.update_device();
    array.update_strides_host();
    array.update_strides_device();

    // Verify values after updates
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 4.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(array(3, 0), 6.0);
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, NameManagement) {
    // Create DDynamicRaggedRightArrayKokkos with specific name
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Test name management
    EXPECT_EQ(array.get_name(), "test_array");
}

TEST_F(DDynamicRaggedRightArrayKokkosTest, KokkosViewAccess) {
    // Create DDynamicRaggedRightArrayKokkos
    DDynamicRaggedRightArrayKokkos<double> array(dim1, dim2, "test_array");

    // Set some values
    array(0, 0) = 1.0;
    array(0, 1) = 2.0;
    array(1, 0) = 3.0;
    array(2, 0) = 4.0;
    array(2, 1) = 5.0;
    array(3, 0) = 6.0;

    // Get Kokkos view
    auto view = array.get_kokkos_dual_view();

    // Verify view is not null
    EXPECT_NE(view.h_view.data(), nullptr);
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
