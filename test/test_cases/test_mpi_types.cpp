// Google tests for MPICArrayKokkos / CommunicationPlan (HAVE_MPI + HAVE_KOKKOS).
// Run the mpi_test_main executable under mpirun (see test_cases/CMakeLists.txt).

#if !defined(HAVE_MPI) || !defined(HAVE_KOKKOS)

#include <gtest/gtest.h>

TEST(MPI_Types, SkippedWithoutMpiKokkos) { GTEST_SKIP() << "Build MATAR tests with MPI and Kokkos enabled."; }

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

// ---------------------------------------------------------------------------
// Free functions wrapping FOR_ALL kernels.
// KOKKOS_LAMBDA must not appear inside TEST()'s private TestBody (nvcc rejects
// extended __device__ lambdas in functions with internal/private linkage).
// ---------------------------------------------------------------------------

inline void fill_minmax_1d(MPICArrayKokkos<float>& vals, int rank, int num_values_per_rank) {
    FOR_ALL(i, 0, num_values_per_rank, {
        vals(i) = static_cast<float>(10 * rank + i);
    });
    MATAR_FENCE();
}

inline void fill_centroids_rank2(MPICArrayKokkos<double>& elem_centroids, int rank, int n_elem, int num_coords) {
    FOR_ALL(elem_id, 0, n_elem,
            elem_position, 0, num_coords, {
        const double base                      = 1000.0 * rank + 100.0 * elem_id;
        elem_centroids(elem_id, elem_position) = base + 10.0 * static_cast<double>(elem_position);
    });
    MATAR_FENCE();
}

inline void fill_stress_rank3(MPICArrayKokkos<double>& stress, int rank, int n_elem) {
    FOR_ALL(e, 0, n_elem,
            r, 0, 3,
            c, 0, 3, {
        stress(e, r, c) = 10000.0 * rank + 1000.0 * e + 100.0 * r + c;
    });
    MATAR_FENCE();
}

}  // namespace

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

    const double global_sum = locals.all_reduce(operation::sum);
    const double expected   = static_cast<double>(num_values * size);
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

    const double global_sum = rank_locals.all_reduce(operation::sum);
    const double expected   = 10.0 * static_cast<double>(size * (size + 1) / 2);
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

    fill_minmax_1d(vals, rank, num_values_per_rank);

    const float global_min = vals.all_reduce(operation::min);
    const float global_max = vals.all_reduce(operation::max);
    EXPECT_FLOAT_EQ(global_min, 0.0F);
    const float expected_max = static_cast<float>(10 * (size - 1) + (10 * size - 1));
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

    const double global_product = prod_locals.all_reduce(operation::product);
    const double expected       = std::pow(2.0, 4 * size);
    EXPECT_DOUBLE_EQ(global_product, expected);
}

TEST(MPICArrayKokkos, AllReduce_Rank2_CentroidXYZ) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const size_t n_elem      = 3;
    constexpr int num_coords = 3;

    MPICArrayKokkos<double> elem_centroids(n_elem, static_cast<size_t>(num_coords), "ut_centroids");
    elem_centroids.initialize_comm_plan(comm_plan);
    fill_centroids_rank2(elem_centroids, rank, static_cast<int>(n_elem), num_coords);

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
    fill_stress_rank3(stress, rank, static_cast<int>(n_elem));

    const double max_comp = stress.all_reduce(operation::max, static_cast<size_t>(0), static_cast<size_t>(1));
    const double expected = 10000.0 * static_cast<double>(size - 1) + 1000.0 * static_cast<double>(n_elem - 1) + 1.0;
    EXPECT_DOUBLE_EQ(max_comp, expected);
}

TEST(MPICArrayKokkos, AllReduce_Rank4_GaussStressComponent) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const size_t n_elem  = 3;
    const size_t n_gauss = 2;

    MPICArrayKokkos<double> s4(n_elem, n_gauss, 3, 3, "ut_s4");
    s4.initialize_comm_plan(comm_plan);
    for (size_t e = 0; e < n_elem; ++e) {
        for (size_t g = 0; g < n_gauss; ++g) {
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    s4.host(e, g, r, c) = 100000.0 * rank + 1000.0 * static_cast<double>(e) + 100.0 * static_cast<double>(g) +
                                          10.0 * static_cast<double>(r) + static_cast<double>(c);
                }
            }
        }
    }
    s4.update_device();

    const double max_qp   = s4.all_reduce(operation::max, static_cast<size_t>(1), static_cast<size_t>(0), static_cast<size_t>(1));
    const double expected = 100000.0 * static_cast<double>(size - 1) + 1000.0 * static_cast<double>(n_elem - 1) + 101.0;
    EXPECT_DOUBLE_EQ(max_qp, expected);
}

// ---------------------------------------------------------------------------
// Precision-tier coverage: the same operations at the build's active real_t
// (double | float | half | bfloat16 | quad). Values are exact at bfloat16
// (integers <= 256, halves in range) so checks stay tight at every tier.
// On CPU builds half/bfloat16 run float-backed; the quad build exercises the
// custom MPI datatype + custom MPI_Op paths.
// ---------------------------------------------------------------------------

TEST(MPICArrayKokkos, AllReduce_Tier_Sum) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const int num_values = 100;
    MPICArrayKokkos<real_t> locals(num_values, "ut_tier_sum");
    locals.initialize_comm_plan(comm_plan);
    locals.set_values(real_t(0.5));

    const real_t global_sum = locals.all_reduce(operation::sum);
    // 100 * 0.5 * size: every partial is a multiple of 0.5 well inside range
    EXPECT_NEAR(static_cast<double>(global_sum), 50.0 * size, 0.5);
}

TEST(MPICArrayKokkos, AllReduce_Tier_MinMax) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    const int num_values = 10;
    MPICArrayKokkos<real_t> vals(num_values, "ut_tier_minmax");
    vals.initialize_comm_plan(comm_plan);
    for (int i = 0; i < num_values; i++) {
        vals.host(i) = real_t(10 * rank + i);  // integers < 256: exact at bf16
    }
    vals.update_device();

    const real_t global_min = vals.all_reduce(operation::min);
    const real_t global_max = vals.all_reduce(operation::max);
    EXPECT_DOUBLE_EQ(static_cast<double>(global_min), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(global_max), static_cast<double>(10 * (size - 1) + 9));
}

TEST(MPICArrayKokkos, AllReduce_Tier_Product) {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    // two 2s per rank: 2^(2*size) = 256 at 4 ranks — inside half's 65504 max
    MPICArrayKokkos<real_t> prod_locals(2, "ut_tier_prod");
    prod_locals.initialize_comm_plan(comm_plan);
    prod_locals.set_values(real_t(2));

    const real_t global_product = prod_locals.all_reduce(operation::product);
    EXPECT_DOUBLE_EQ(static_cast<double>(global_product), std::pow(2.0, 2 * size));
}

// ---------------------------------------------------------------------------
// Halo exchange (communicate() / MPI_Neighbor_alltoallv) over a ring:
// each rank owns 4 values (indices 0-3) plus one ghost slot (index 4);
// it sends its last owned value to the next rank and receives the previous
// rank's last owned value into the ghost slot.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
void run_halo_ring_test() {
    int rank = 0;
    int size = 1;
    mpi_rank_size(&rank, &size);
    if (size < 2) {
        GTEST_SKIP() << "halo test needs at least 2 ranks";
    }

    const int next = (rank + 1) % size;
    const int prev = (rank + size - 1) % size;

    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);
    int send_ranks[1] = {next};
    int recv_ranks[1] = {prev};
    comm_plan.initialize_graph_communicator(1, send_ranks, 1, recv_ranks);

    // one entry sent to the single send-neighbor: local index 3;
    // one entry received from the single recv-neighbor into local index 4
    DCArrayKokkos<size_t> send_strides(1, "ut_halo_send_strides");
    DCArrayKokkos<size_t> recv_strides(1, "ut_halo_recv_strides");
    send_strides.host(0) = 1;
    recv_strides.host(0) = 1;
    send_strides.update_device();
    recv_strides.update_device();

    DRaggedRightArrayKokkos<int> send_ids(send_strides, "ut_halo_send_ids");
    DRaggedRightArrayKokkos<int> recv_ids(recv_strides, "ut_halo_recv_ids");
    send_ids.host(0, 0) = 3;
    recv_ids.host(0, 0) = 4;
    send_ids.update_device();
    recv_ids.update_device();

    comm_plan.setup_send_recv(send_ids, recv_ids);

    const int num_owned = 4;
    const int num_total = num_owned + 1;  // + one ghost slot
    MPICArrayKokkos<T> field(num_total, "ut_halo_field");
    field.initialize_comm_plan(comm_plan);
    for (int i = 0; i < num_owned; i++) {
        field.host(i) = T(10 * rank + i);  // exact at bf16 for small rank counts
    }
    field.host(num_owned) = T(-1);  // ghost sentinel
    field.update_device();

    field.communicate();

    // ghost slot must now hold the previous rank's last owned value
    EXPECT_DOUBLE_EQ(static_cast<double>(field.host(num_owned)), static_cast<double>(10 * prev + 3));
    // owned values must be untouched
    EXPECT_DOUBLE_EQ(static_cast<double>(field.host(0)), static_cast<double>(10 * rank));
    EXPECT_DOUBLE_EQ(static_cast<double>(field.host(3)), static_cast<double>(10 * rank + 3));
}

}  // namespace

TEST(MPICArrayKokkos, Communicate_HaloRing_Double) { run_halo_ring_test<double>(); }

TEST(MPICArrayKokkos, Communicate_HaloRing_Tier) { run_halo_ring_test<real_t>(); }

#endif  // HAVE_MPI && HAVE_KOKKOS
