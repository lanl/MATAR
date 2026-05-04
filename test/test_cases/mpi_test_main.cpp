#include <gtest/gtest.h>
#include <mpi.h>

#include <matar.h>

// MPI must initialize before Kokkos (MATAR_INITIALIZE).
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MATAR_FINALIZE();
    MPI_Finalize();
    return result;
}
