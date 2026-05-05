// Example: basic MPI features in MATAR (requires HAVE_MPI and HAVE_KOKKOS).
// - MPICArrayKokkos: distributed array with optional halo exchange via communicate()
// - CommunicationPlan: MPI distributed graph + per-neighbor index lists
// - MPICArrayKokkos::all_reduce: 1D, and fixed trailing indices for rank 2/3/4
//   (mpi_type_map + mpi_op_for in mpi_types.h / communication_plan.h)

#if !defined(HAVE_MPI) || !defined(HAVE_KOKKOS)

#include <iostream>

int main() {
    std::cerr
        << "This example requires MATAR built with HAVE_MPI and HAVE_KOKKOS.\n";
    return 0;
}

#else

#include <cmath>
#include <iostream>

#include <mpi.h>
#include <matar.h>

using namespace mtr;

int main(int argc, char* argv[]) {

MPI_Init(&argc, &argv);
MATAR_INITIALIZE(argc, argv);
{

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a basic communication plan, this handles things like storing MPI world comms and 
    // allows for basic reduction operations.  For more complex communication patterns, you can
    // use the CommunicationPlan to build a more complex communication plan.
    CommunicationPlan comm_plan;
    comm_plan.initialize(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // MPICArrayKokkos::all_reduce — uses mpi_type_map<T> for MPI_Datatype and
    // mpi_op_for(operation) for MPI_Op (no CommunicationPlan required).
    // -------------------------------------------------------------------------
    
    // Same size arrays on each rank, just do a simple reduction.
    const int num_values = 100;
    MPICArrayKokkos<double> locals(num_values, "values");
    locals.initialize_comm_plan(comm_plan);
    locals.set_values(1.0);
    locals.update_device();

    double global_sum = locals.all_reduce(operation::sum);
    
    double expected_sum = static_cast<double>(num_values * size);
    if (rank == 0) {
        std::cout << "all_reduce(sum): " << global_sum << " (expect "
                  << expected_sum << ")\n";
    }

    // Different size arrays on each rank, do a simple reduction.
    const int num_values_per_rank = 10*(1+rank);
    MPICArrayKokkos<double> rank_locals(num_values_per_rank, "values");
    rank_locals.initialize_comm_plan(comm_plan);
    rank_locals.set_values(1.0);
    rank_locals.update_device();

    global_sum = rank_locals.all_reduce(operation::sum);
    // Rank r has 10*(r+1) entries of 1 → total = 10 * (1 + … + size).
    expected_sum = 10.0 * static_cast<double>(size * (size + 1) / 2);


    if (rank == 0) {
        std::cout << "all_reduce(sum) with different size arrays on each rank (10*rank_id): " << global_sum << " (expect "
                  << expected_sum << ")\n";
    }


    // Different size arrays on each rank, find the minimum value
    MPICArrayKokkos<float> min_locals(num_values_per_rank, "values");
    min_locals.initialize_comm_plan(comm_plan);

    FOR_ALL(i, 0, num_values_per_rank, {
        min_locals(i) = static_cast<float>(10*rank + i);
    });

    min_locals.update_device();
    float global_min = min_locals.all_reduce(operation::min);
    float expected_min = 0.0F;
    if (rank == 0) {
        std::cout << "all_reduce(min) with different size arrays on each rank (10*rank_id): " << global_min << " (expect "
                  << expected_min << ")\n";
    }

    float global_max = min_locals.all_reduce(operation::max);
    // Largest entry is on rank (size-1) at i = 10*size - 1.
    const float expected_max = static_cast<float>(10 * (size - 1) + (10 * size - 1));
    if (rank == 0) {
        std::cout << "all_reduce(max) with different size arrays on each rank (10*rank_id): " << global_max << " (expect "
                  << expected_max << ")\n";
    }


    
    // Example: all_reduce with product
    // Initialize a MPICArrayKokkos<double> with all values 2.0
    MPICArrayKokkos<double> prod_locals(4, "prod_values");
    prod_locals.initialize_comm_plan(comm_plan);
    prod_locals.set_values(2.0);
    prod_locals.update_device();

    // Compute the product across all ranks and all values
    double global_product = prod_locals.all_reduce(operation::product);

    // The expected product is pow(2, 4*size), i.e., each rank contributes 4 twos
    double expected_product = std::pow(2.0, 4 * size);
    if (rank == 0) {
        std::cout << "all_reduce(product): " << global_product << " (expect "
                  << expected_product << ")\n";
    }

    // -------------------------------------------------------------------------
    // all_reduce with fixed trailing indices (multi-dimensional arrays).
    // -------------------------------------------------------------------------
    const size_t n_elem = size * 10;

    // Rank-2: element centroid coordinates — elem_centroids(elem_id, elem_position)
    // with elem_position ∈ {0,1,2} as x, y, z. Reduce over elem_id for each axis.
    {
        size_t n_elem = 3;
        size_t num_coords = 3;
        MPICArrayKokkos<double> elem_centroids(n_elem, num_coords,"elem_centroids");
        elem_centroids.initialize_comm_plan(comm_plan);
        FOR_ALL(elem_id, 0, n_elem,
                elem_position, 0, num_coords, {
            const double base = 1000.0 * rank + 100.0 * elem_id;
            elem_centroids(elem_id, elem_position) =
                base + 10.0 * static_cast<double>(elem_position);
        });
        MATAR_FENCE();
        elem_centroids.update_device();

        const double max_x = elem_centroids.all_reduce(operation::max, 0);
        const double max_y = elem_centroids.all_reduce(operation::max, 1);
        const double max_z = elem_centroids.all_reduce(operation::max, 2);

        const double base_rank = 1000.0 * static_cast<double>(size - 1);
        const double base_elem = 100.0 * static_cast<double>(n_elem - 1);
        const double expect_max_x = base_rank + base_elem + 0.0;
        const double expect_max_y = base_rank + base_elem + 10.0;
        const double expect_max_z = base_rank + base_elem + 20.0;

        if (rank == 0) {
            std::cout << "all_reduce(max, coord) rank-2 centroids — max x: " << max_x
                      << " (expect " << expect_max_x << ")\n";
            std::cout << "all_reduce(max, coord) rank-2 centroids — max y: " << max_y
                      << " (expect " << expect_max_y << ")\n";
            std::cout << "all_reduce(max, coord) rank-2 centroids — max z: " << max_z
                      << " (expect " << expect_max_z << ")\n";
        }
    }

    // Rank-3: reduce over e at fixed tensor component — e.g. stress(e, 0, 1).
    {
        // Rank-3: reduce over e at fixed tensor component — e.g. stress(e, 0, 1).
        MPICArrayKokkos<double> stress(n_elem, 3, 3, "stress");
        stress.initialize_comm_plan(comm_plan);
        FOR_ALL(e, 0, n_elem, r, 0, 3, c, 0, 3, {
            stress(e, r, c) =
                10000.0 * rank + 1000.0 * e + 100.0 * r + c;
        });
        MATAR_FENCE();
        stress.update_device();
        const double max_comp = stress.all_reduce(operation::max,  0, 1);
        
        const double expect_3d =
            10000.0 * static_cast<double>(size - 1) +
            1000.0 * static_cast<double>(n_elem - 1) + 1.0;
        if (rank == 0) {
            std::cout << "all_reduce(max, i, j) rank-3 s(e,i,j): " << max_comp
                      << " (expect " << expect_3d << ")\n";
        }
    }

    // Rank-4: reduce over element at fixed Gauss point and tensor component.
    {
        const size_t n_gauss = 2;
        MPICArrayKokkos<double> s4(n_elem, n_gauss, 3, 3, "s4");
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
        const size_t g_fix = 1;
        const size_t ti_fix = 0;
        const size_t tj_fix = 1;
        const double max_qp =
            s4.all_reduce(operation::max, g_fix, ti_fix, tj_fix);
        const double expect_4d =
            100000.0 * static_cast<double>(size - 1) +
            1000.0 * static_cast<double>(n_elem - 1) + 101.0;
        if (rank == 0) {
            std::cout << "all_reduce(max, g, i, j) rank-4 stress(e,g,i,j): "
                      << max_qp << " (expect " << expect_4d << ")\n";
        }
    }

    // -------------------------------------------------------------------------
    // CommunicationPlan + communicate(): periodic 1D halo (needs 2+ ranks).
    // Layout per rank: index 0 = left ghost, 1..L = owned, L+1 = right ghost.
    // -------------------------------------------------------------------------
    // if (size < 2) {
    //     if (rank == 0) {
    //         std::cout
    //             << "Re-run with 2+ MPI processes to run the halo exchange demo.\n";
    //     }
    // } else {
    //     const int L = 4;
    //     const int left = (rank + size - 1) % size;
    //     const int right = (rank + 1) % size;

    //     CommunicationPlan comm_plan;
    //     comm_plan.initialize(MPI_COMM_WORLD);

    //     int send_ranks[2] = { left, right };
    //     int recv_ranks[2] = { left, right };
    //     comm_plan.initialize_graph_communicator(2, send_ranks, 2, recv_ranks);

    //     DCArrayKokkos<size_t> send_strides(2, "send_strides");
    //     send_strides.host(0) = 1;
    //     send_strides.host(1) = 1;
    //     send_strides.update_device();
    //     DRaggedRightArrayKokkos<int> rank_send_ids(send_strides,
    //                                                "rank_send_ids");
    //     // Send first owned cell to the left neighbor; last owned to the right.
    //     rank_send_ids.host(0, 0) = 1;
    //     rank_send_ids.host(1, 0) = L;
    //     rank_send_ids.update_device();

    //     DCArrayKokkos<size_t> recv_strides(2, "recv_strides");
    //     recv_strides.host(0) = 1;
    //     recv_strides.host(1) = 1;
    //     recv_strides.update_device();
    //     DRaggedRightArrayKokkos<int> rank_recv_ids(recv_strides,
    //                                                "rank_recv_ids");
    //     rank_recv_ids.host(0, 0) = 0;
    //     rank_recv_ids.host(1, 0) = L + 1;
    //     rank_recv_ids.update_device();

    //     MATAR_FENCE();
    //     comm_plan.setup_send_recv(rank_send_ids, rank_recv_ids);

    //     MPICArrayKokkos<double> field(static_cast<size_t>(L + 2), "field");
    //     field.initialize_comm_plan(comm_plan);

    //     FOR_ALL(i, 0, L + 2, {
    //         field(i) = -1.0;
    //     });
    //     FOR_ALL(i, 1, L + 1, { field(i) = static_cast<double>(rank); });
    //     MATAR_FENCE();

    //     field.communicate();

    //     field.update_host();
    //     MATAR_FENCE();

    //     const double gl = field.host(0);
    //     const double gr = field.host(L + 1);
    //     const bool halo_ok =
    //         (std::fabs(gl - static_cast<double>(left)) < 1.0e-14) &&
    //         (std::fabs(gr - static_cast<double>(right)) < 1.0e-14);

    //     if (rank == 0) {
    //         std::cout << "After halo exchange (periodic 1D): ";
    //         if (halo_ok) {
    //             std::cout << "ghost values match neighbor ranks.\n";
    //         } else {
    //             std::cout << "verification failed.\n";
    //         }
    //     }
    // }
}
MATAR_FINALIZE();
MPI_Finalize();
return 0;
}

#endif
