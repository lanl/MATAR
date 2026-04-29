// Example: basic MPI features in MATAR (requires HAVE_MPI and HAVE_KOKKOS).
// - MPICArrayKokkos: distributed array with optional halo exchange via communicate()
// - CommunicationPlan: MPI distributed graph + per-neighbor index lists
// - Reductions use mpi_type_map (in mpi_types.h) and ::operation / mpi_op_for
//   (in communication_plan.h) internally for MPI_Allreduce.

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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -------------------------------------------------------------------------
    // MPICArrayKokkos::all_reduce — uses mpi_type_map<T> for MPI_Datatype and
    // mpi_op_for(operation) for MPI_Op (no CommunicationPlan required).
    // -------------------------------------------------------------------------
    const int num_values = 100;
    MPICArrayKokkos<double> locals(num_values, "values");
    locals.set_values(1.0);
    

    const double global_sum = locals.all_reduce(operation::sum);
    const double expected_sum = static_cast<double>(num_values * size);

    if (rank == 0) {
        std::cout << "all_reduce(sum): " << global_sum << " (expect "
                  << expected_sum << ")\n";
    }

    // -------------------------------------------------------------------------
    // CommunicationPlan + communicate(): periodic 1D halo (needs 2+ ranks).
    // Layout per rank: index 0 = left ghost, 1..L = owned, L+1 = right ghost.
    // -------------------------------------------------------------------------
    if (size < 2) {
        if (rank == 0) {
            std::cout
                << "Re-run with 2+ MPI processes to run the halo exchange demo.\n";
        }
    } else {
        const int L = 4;
        const int left = (rank + size - 1) % size;
        const int right = (rank + 1) % size;

        CommunicationPlan comm_plan;
        comm_plan.initialize(MPI_COMM_WORLD);

        int send_ranks[2] = { left, right };
        int recv_ranks[2] = { left, right };
        comm_plan.initialize_graph_communicator(2, send_ranks, 2, recv_ranks);

        DCArrayKokkos<size_t> send_strides(2, "send_strides");
        send_strides.host(0) = 1;
        send_strides.host(1) = 1;
        send_strides.update_device();
        DRaggedRightArrayKokkos<int> rank_send_ids(send_strides,
                                                   "rank_send_ids");
        // Send first owned cell to the left neighbor; last owned to the right.
        rank_send_ids.host(0, 0) = 1;
        rank_send_ids.host(1, 0) = L;
        rank_send_ids.update_device();

        DCArrayKokkos<size_t> recv_strides(2, "recv_strides");
        recv_strides.host(0) = 1;
        recv_strides.host(1) = 1;
        recv_strides.update_device();
        DRaggedRightArrayKokkos<int> rank_recv_ids(recv_strides,
                                                   "rank_recv_ids");
        rank_recv_ids.host(0, 0) = 0;
        rank_recv_ids.host(1, 0) = L + 1;
        rank_recv_ids.update_device();

        MATAR_FENCE();
        comm_plan.setup_send_recv(rank_send_ids, rank_recv_ids);

        MPICArrayKokkos<double> field(static_cast<size_t>(L + 2), "field");
        field.initialize_comm_plan(comm_plan);

        FOR_ALL(i, 0, L + 2, {
            field(i) = -1.0;
        });
        FOR_ALL(i, 1, L + 1, { field(i) = static_cast<double>(rank); });
        MATAR_FENCE();

        field.communicate();

        field.update_host();
        MATAR_FENCE();

        const double gl = field.host(0);
        const double gr = field.host(L + 1);
        const bool halo_ok =
            (std::fabs(gl - static_cast<double>(left)) < 1.0e-14) &&
            (std::fabs(gr - static_cast<double>(right)) < 1.0e-14);

        if (rank == 0) {
            std::cout << "After halo exchange (periodic 1D): ";
            if (halo_ok) {
                std::cout << "ghost values match neighbor ranks.\n";
            } else {
                std::cout << "verification failed.\n";
            }
        }
    }

    MATAR_FINALIZE();
    MPI_Finalize();
    return 0;
}

#endif
