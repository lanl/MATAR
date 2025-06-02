#include <gtest/gtest.h>
#include "kokkos_types.h"
#include <Kokkos_Core.hpp>

// Test fixture for DynamicRaggedDownArrayKokkos tests
class DynamicRaggedDownArrayKokkosTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Kokkos
        Kokkos::initialize();
    }

    void TearDown() override {
        // Finalize Kokkos
        Kokkos::finalize();
    }
};

// Test default constructor and basic initialization
TEST_F(DynamicRaggedDownArrayKokkosTest, DefaultConstructor) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Check initial dimensions
    EXPECT_EQ(array.dims(0), 3);  // 3 rows
    EXPECT_EQ(array.dims(1), 2);  // Initial column size
    
    // Check initial strides
    EXPECT_EQ(array.stride(0), 0);
    EXPECT_EQ(array.stride(1), 0);
    EXPECT_EQ(array.stride(2), 0);
}

// Test push_back functionality
TEST_F(DynamicRaggedDownArrayKokkosTest, PushBack) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Push back values to first column
    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 3.0);
    
    // Check stride
    EXPECT_EQ(array.stride(0), 3);
}

// Test pop_back functionality
TEST_F(DynamicRaggedDownArrayKokkosTest, PopBack) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Push back values
    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);
    
    // Pop back
    array.pop_back();
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 2.0);
    
    // Check stride
    EXPECT_EQ(array.stride(0), 2);
}

// Test set_values functionality
TEST_F(DynamicRaggedDownArrayKokkosTest, SetValues) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Set all values to 42.0
    array.set_values(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
}

// Test set_values_sparse functionality
TEST_F(DynamicRaggedDownArrayKokkosTest, SetValuesSparse) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Set sparse values
    array.set_values_sparse(42.0);
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(0, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(1, 1), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 0), 42.0);
    EXPECT_DOUBLE_EQ(array(2, 1), 42.0);
}

// Test host/device synchronization
TEST_F(DynamicRaggedDownArrayKokkosTest, HostDeviceSync) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Set values on host
    array(0, 0) = 1.0;
    array(1, 0) = 2.0;
    
    // Update device
    array.update_device();
    
    // Modify on device
    Kokkos::parallel_for("ModifyOnDevice", 1, KOKKOS_LAMBDA(const int i) {
        array(0, 0) = 3.0;
    });
    
    // Update host
    array.update_host();
    
    // Check values
    EXPECT_DOUBLE_EQ(array(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(array(1, 0), 2.0);
}

// Test stride management
TEST_F(DynamicRaggedDownArrayKokkosTest, StrideManagement) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Push back values to first column
    array.push_back(1.0);
    array.push_back(2.0);
    array.push_back(3.0);
    
    // Check strides
    EXPECT_EQ(array.stride(0), 3);
    EXPECT_EQ(array.stride(1), 0);
    EXPECT_EQ(array.stride(2), 0);
    
    // Check host strides
    EXPECT_EQ(array.stride_host(0), 3);
    EXPECT_EQ(array.stride_host(1), 0);
    EXPECT_EQ(array.stride_host(2), 0);
}

// Test name management
TEST_F(DynamicRaggedDownArrayKokkosTest, NameManagement) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    EXPECT_EQ(array.get_name(), "test_array");
    
    // Create another array with different name
    DynamicRaggedDownArrayKokkos<double> array2(3, 2, "another_array");
    EXPECT_EQ(array2.get_name(), "another_array");
}

// Test different data types
TEST_F(DynamicRaggedDownArrayKokkosTest, DifferentDataTypes) {
    // Test with float
    DynamicRaggedDownArrayKokkos<float> array_float(3, 2, "float_array");
    array_float.push_back(42.0f);
    EXPECT_FLOAT_EQ(array_float(0, 0), 42.0f);
    
    // Test with int
    DynamicRaggedDownArrayKokkos<int> array_int(3, 2, "int_array");
    array_int.push_back(42);
    EXPECT_EQ(array_int(0, 0), 42);
}

// Test different layouts
TEST_F(DynamicRaggedDownArrayKokkosTest, DifferentLayouts) {
    // Test with default layout
    DynamicRaggedDownArrayKokkos<double> array_default(3, 2, "default_array");
    array_default.push_back(42.0);
    EXPECT_DOUBLE_EQ(array_default(0, 0), 42.0);
    
    // Test with column-major layout
    DynamicRaggedDownArrayKokkos<double, Kokkos::LayoutLeft> array_col(3, 2, "col_array");
    array_col.push_back(42.0);
    EXPECT_DOUBLE_EQ(array_col(0, 0), 42.0);
}

// Test out-of-bounds access
TEST_F(DynamicRaggedDownArrayKokkosTest, OutOfBoundsAccess) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Test accessing beyond row bounds
    EXPECT_DEATH(array(3, 0), ".*");  // Row 3 doesn't exist
    
    // Test accessing beyond column bounds
    EXPECT_DEATH(array(0, 2), ".*");  // Initial column size is 2
}

// Test get_kokkos_dual_view
TEST_F(DynamicRaggedDownArrayKokkosTest, GetKokkosDualView) {
    DynamicRaggedDownArrayKokkos<double> array(3, 2, "test_array");
    
    // Get the dual view
    auto view = array.get_kokkos_dual_view();
    
    // Check that the view is valid
    EXPECT_TRUE(view.data() != nullptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
