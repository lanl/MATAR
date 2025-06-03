#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        int result = RUN_ALL_TESTS();
        Kokkos::finalize();
        return result;
    }
} 