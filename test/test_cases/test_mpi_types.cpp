// Google tests for MPICArrayKokkos / CommunicationPlan (HAVE_MPI + HAVE_KOKKOS).
// Run the mpi_test_main executable under mpirun (see test_cases/CMakeLists.txt).

#if !defined(HAVE_MPI) || !defined(HAVE_KOKKOS)

#include <gtest/gtest.h>

TEST(MPI_Types, SkippedWithoutMpiKokkos) {
    GTEST_SKIP() << "Build MATAR tests with MPI and Kokkos enabled.";
}

#else

#include <cmath>
#include <mpi.h>

#include <gtest/gtest.h>
#include <matar.h>

using namespace mtr;

namespace {

void mpi_rank_size(int* rank, int* size) {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

} // namespace

TEST(MPICArrayKokkos, AllReduce_Sum_1D) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const int num_values = 100;
    MPICArrayKokkos<double> locals(num_values, "ut_values");
    locals.initialize_comm_plan(comm_plan);
    locals.set_values(1.0);
    locals.update_device();

    const double global_sum = locals.all_reduce(operation::sum);
    const double expected = static_cast<double>(num_values * size);
    EXPECT_DOUBLE_EQ(global_sum, expected);
}

TEST(MPICArrayKokkos, AllReduce_Sum_VariableLengthPerRank) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const int num_values_per_rank = 10 * (1 + rank);
    MPICArrayKokkos<double> rank_locals(num_values_per_rank, "ut_varlen");
    rank_locals.initialize_comm_plan(comm_plan);
    rank_locals.set_values(1.0);
    rank_locals.update_device();

    const double global_sum = rank_locals.all_reduce(operation::sum);
    const double expected =
        10.0 * static_cast<double>(size * (size + 1) / 2);
    EXPECT_DOUBLE_EQ(global_sum, expected);
}

TEST(MPICArrayKokkos, AllReduce_MinMax_1D) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const int num_values_per_rank = 10 * (1 + rank);
    MPICArrayKokkos<float> vals(num_values_per_rank, "ut_minmax");
    vals.initialize_comm_plan(comm_plan);

    FOR_ALL(i, 0, num_values_per_rank, {
        vals(i) = static_cast<float>(10 * rank + i);
    });
    MATAR_FENCE();
    vals.update_device();

    const float global_min = vals.all_reduce(operation::min);
    const float global_max = vals.all_reduce(operation::max);
    EXPECT_FLOAT_EQ(global_min, 0.0F);
    const float expected_max =
        static_cast<float>(10 * (size - 1) + (10 * size - 1));
    EXPECT_FLOAT_EQ(global_max, expected_max);
}

TEST(MPICArrayKokkos, AllReduce_Product) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    MPICArrayKokkos<double> prod_locals(4, "ut_prod");
    prod_locals.initialize_comm_plan(comm_plan);
    prod_locals.set_values(2.0);
    prod_locals.update_device();

    const double global_product = prod_locals.all_reduce(operation::product);
    const double expected = std::pow(2.0, 4 * size);
    EXPECT_DOUBLE_EQ(global_product, expected);
}

TEST(MPICArrayKokkos, AllReduce_Rank2_CentroidXYZ) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const size_t n_elem = 3;
    constexpr int num_coords = 3;

    MPICArrayKokkos<double> elem_centroids(n_elem, static_cast<size_t>(num_coords),
                                           "ut_centroids");
    elem_centroids.initialize_comm_plan(comm_plan);
    FOR_ALL(elem_id, 0, n_elem, elem_position, 0, num_coords, {
        const double base = 1000.0 * rank + 100.0 * elem_id;
        elem_centroids(elem_id, elem_position) =
            base + 10.0 * static_cast<double>(elem_position);
    });
    MATAR_FENCE();
    elem_centroids.update_device();

    const double max_x = elem_centroids.all_reduce(operation::max, 0U);
    const double max_y = elem_centroids.all_reduce(operation::max, 1U);
    const double max_z = elem_centroids.all_reduce(operation::max, 2U);

    const double base_rank = 1000.0 * static_cast<double>(size - 1);
    const double base_elem = 100.0 * static_cast<double>(n_elem - 1);
    EXPECT_DOUBLE_EQ(max_x, base_rank + base_elem + 0.0);
    EXPECT_DOUBLE_EQ(max_y, base_rank + base_elem + 10.0);
    EXPECT_DOUBLE_EQ(max_z, base_rank + base_elem + 20.0);
}

TEST(MPICArrayKokkos, AllReduce_Rank3_StressComponent) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const size_t n_elem = 3;
    MPICArrayKokkos<double> stress(n_elem, 3, 3, "ut_stress");
    stress.initialize_comm_plan(comm_plan);
    FOR_ALL(e, 0, n_elem, r, 0, 3, c, 0, 3, {
        stress(e, r, c) = 10000.0 * rank + 1000.0 * e + 100.0 * r + c;
    });
    MATAR_FENCE();
    stress.update_device();

    const double max_comp =
        stress.all_reduce(operation::max, static_cast<size_t>(0),
                          static_cast<size_t>(1));
    const double expected = 10000.0 * static_cast<double>(size - 1) +
                            1000.0 * static_cast<double>(n_elem - 1) + 1.0;
    EXPECT_DOUBLE_EQ(max_comp, expected);
}

TEST(MPICArrayKokkos, AllReduce_Rank4_GaussStressComponent) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const size_t n_elem = 3;
    const size_t n_gauss = 2;

    MPICArrayKokkos<double> s4(n_elem, n_gauss, 3, 3, "ut_s4");
    s4.initialize_comm_plan(comm_plan);
    for (size_t e = 0; e < n_elem; ++e) {
        for (size_t g = 0; g < n_gauss; ++g) {
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    s4.host(e, g, r, c) =
                        100000.0 * rank + 1000.0 * static_cast<double>(e) +
                        100.0 * static_cast<double>(g) +
                        10.0 * static_cast<double>(r) +
                        static_cast<double>(c);
                }
            }
        }
    }
    s4.update_device();

    const double max_qp =
        s4.all_reduce(operation::max, static_cast<size_t>(1),
                      static_cast<size_t>(0), static_cast<size_t>(1));
    const double expected = 100000.0 * static_cast<double>(size - 1) +
                            1000.0 * static_cast<double>(n_elem - 1) + 101.0;
    EXPECT_DOUBLE_EQ(max_qp, expected);
}

#endif // HAVE_MPI && HAVE_KOKKOS
