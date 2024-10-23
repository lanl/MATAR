#include <mpi.h>
#include <matar.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <string>

// Dont change ROOT
#define ROOT  0
//----------------
// For overlapping async calls with setup
#define FAST

using namespace mtr; // matar namespace

int width = 4;
int height = 4;
int max_num_iterations = 1000;
double temp_tolerance = 0.01;

// Example of DENSE halo sending and receiving
// uses a map for global to local, to know what needs to be sent
// simple directional comp grid
// n = north (j-1), s = south (j+1), w = west (i-1), e = east (i+1)
void example_comms_with_map(int world_size, int rank) {
    // neighbor calcs
    int n = sqrt(world_size);
    int world_i = rank % n;
    int world_j = rank / n;
    int neighbors = 0;
    int j_n = world_j - 1;
    int j_s = world_j + 1;
    int i_w = world_i - 1;
    int i_e = world_i + 1;
    if (j_n >= 0) neighbors++;
    if (j_s < n) neighbors++;
    if (i_w >= 0) neighbors++;
    if (i_e < n) neighbors++;

    int rank_n = j_n * n + world_i;
    int rank_s = j_s * n + world_i;
    int rank_w = world_j * n + i_w;
    int rank_e = world_j * n + i_e;

/*
    int stag_n = rank * 10 + rank_n;
    int stag_s = rank * 10 + rank_s;
    int stag_w = rank * 10 + rank_w;
    int stag_e = rank * 10 + rank_e;
    int rtag_n = rank_n * 10 + rank;
    int rtag_s = rank_s * 10 + rank;
    int rtag_w = rank_w * 10 + rank;
    int rtag_e = rank_e * 10 + rank;
*/

    // data setup
    //int width_loc = width / n;
    //int height_loc = height / n;
    int width_loc = simple_decomp_row_size(world_size, rank, width);
    int height_loc = simple_decomp_col_size(world_size, rank, height);
    //printf("rank %d width %d height %d\n", rank, width_loc, height_loc);
    int halo = 1;
    int arr_size_i = width_loc + halo * 2; //both sides of halo
    int arr_size_j = height_loc + halo * 2;
    MPIArrayKokkos <double> velocity = MPIArrayKokkos <double> (arr_size_i, arr_size_j, "velo");
    DCArrayKokkos <double> varX = DCArrayKokkos <double> (arr_size_i, arr_size_j, "varX");

    // new function
    velocity.mpi_decomp(world_size, rank, halo, MPI_COMM_WORLD);

    // halos
    /*
    MPIArrayKokkos <double> send_n = MPIArrayKokkos <double> (width_loc);
    MPIArrayKokkos <double> send_s = MPIArrayKokkos <double> (width_loc);
    MPIArrayKokkos <double> send_w = MPIArrayKokkos <double> (height_loc);
    MPIArrayKokkos <double> send_e = MPIArrayKokkos <double> (height_loc);
    MPIArrayKokkos <double> recv_n = MPIArrayKokkos <double> (width_loc);
    MPIArrayKokkos <double> recv_s = MPIArrayKokkos <double> (width_loc);
    MPIArrayKokkos <double> recv_w = MPIArrayKokkos <double> (height_loc);
    MPIArrayKokkos <double> recv_e = MPIArrayKokkos <double> (height_loc);
    */
    // setup basic
    for (int ii = 0; ii < arr_size_i; ii++) {
        for (int jj = 0; jj < arr_size_j; jj++) {
            // if halo -1, else owned so rank
            varX.host(ii, jj) = rank * rank;
            if (ii < halo || jj < halo || ii >= width_loc + halo || jj >= height_loc + halo)
                velocity.host(ii, jj) = -1.0;
            else
                velocity.host(ii, jj) = rank;
        }
    }
    varX.update_device();
    velocity.update_device();

/*
    // setup outer boundary (probably more efficient ways, but this is easy
    for (int ii = 0; ii < arr_size_i; ii++) {
        for (int jj = 0; jj < arr_size_j; jj++) {
            // left side
            if (ii < halo && i_w < 0)
                velocity.host(ii, jj) = rank;
            // right side
            if (ii >= width_loc + halo && i_e >= n)
                velocity.host(ii, jj) = rank;
            // top side
            if (jj < halo && j_n < 0)
                velocity.host(ii, jj) = rank;
            // bot side
            if (jj >= height_loc + halo && j_s >= n)
                velocity.host(ii, jj) = rank;
        }
    }
    velocity.update_device();
*/

    // setup halo (probably more efficient ways, but this is easy
    for (int ii = 0; ii < arr_size_i; ii++) {
        for (int jj = 0; jj < arr_size_j; jj++) {
            // right neighbor halo
            if (ii < halo && i_w >= 0)
                velocity.host(ii, jj) = rank_w;
            // right side
            if (ii >= width_loc + halo && i_e < n)
                velocity.host(ii, jj) = rank_e;
            // top side
            if (jj < halo && j_n >= 0)
                velocity.host(ii, jj) = rank_n;
            // bot side
            if (jj >= height_loc + halo && j_s < n)
                velocity.host(ii, jj) = rank_s;
        }
    }
    velocity.update_device();

    FOR_ALL(ii, 1, arr_size_i-1,
             jj, 1, arr_size_j-1, {
              velocity(ii, jj) = varX(ii-1, jj) + varX(ii+1, jj) + varX(ii, jj-1) + varX(ii, jj+1);  
    });
    Kokkos::fence();
    FOR_ALL(ii, 1, arr_size_i-1,
             jj, 1, arr_size_j-1, {
                if (rank == 0) {
                    printf("(%d,%d) %f\n", ii, jj, velocity(ii, jj));
                }
    });
    Kokkos::fence();

    for (int ts = 0; ts < 1; ts++) {

        // "update variable"
        FOR_ALL(ii, 0, arr_size_i,
                 jj, 0, arr_size_j, {
                  varX(ii, jj) = velocity(ii, jj) * 2; 
        });
        Kokkos::fence();

        //printf("in example %d\n", velocity.extent());
        velocity.mpi_halo_update();
        if (rank == 0) printf("\n\n--------------\n\n\n");

        FOR_ALL(ii, 1, arr_size_i-1,
                jj, 1, arr_size_j-1, {
                    if (rank == 0) {
                        printf("(%d,%d) %f\n", ii, jj, velocity(ii, jj));
                    }
        });
        Kokkos::fence();
    
        // "update velocity"
        FOR_ALL(ii, 1, arr_size_i-1,
             jj, 1, arr_size_j-1, {
              velocity(ii, jj) = varX(ii-1, jj) + varX(ii+1, jj) + varX(ii, jj-1) + varX(ii, jj+1);  
    });
    Kokkos::fence();
/*

        if (j_n >= 0) {
            FOR_ALL(hh, 0, width_loc, {
                send_n(hh) = velocity(hh+halo, halo+1); // third from top
            }); 
            Kokkos::fence();
            send_n.isend(width_loc, rank_n, stag_n, MPI_COMM_WORLD);
            recv_n.irecv(width_loc, rank_n, rtag_n, MPI_COMM_WORLD);
        }
        if (j_s < n) {
            FOR_ALL(hh, 0, width_loc, {
                send_s(hh) = velocity(hh+halo, height_loc); // third from bot
            }); 
            Kokkos::fence();
            send_s.isend(width_loc, rank_s, stag_s, MPI_COMM_WORLD);
            recv_s.irecv(width_loc, rank_s, rtag_s, MPI_COMM_WORLD);
        }
        if (i_w >= 0) {
            FOR_ALL(hh, 0, height_loc, {
                send_n(hh) = velocity(halo+1, hh+halo); // third from left
            }); 
            Kokkos::fence();
            send_w.isend(width_loc, rank_w, stag_w, MPI_COMM_WORLD);
            recv_w.irecv(width_loc, rank_w, rtag_w, MPI_COMM_WORLD);
        }
        if (i_e < n) {
            FOR_ALL(hh, 0, height_loc, {
                send_n(hh) = velocity(halo+1, width_loc); // third from left
            }); 
            Kokkos::fence();
            send_e.isend(width_loc, rank_e, stag_e, MPI_COMM_WORLD);
            recv_e.irecv(width_loc, rank_e, rtag_e, MPI_COMM_WORLD);
        }

        if (j_n >= 0) {
            send_n.wait_send();
            recv_n.wait_recv();
        }
        if (j_s < n) {
            send_s.wait_send();
            recv_s.wait_recv();
        }
        if (i_w >= 0) {
            send_w.wait_send();
            recv_w.wait_recv();
        }
        if (i_e < n) {
            send_e.wait_send();
            recv_e.wait_recv();
        }
*/
    }
}
/*

// Example for DENSE halo sending and receiving
// Multiple, non uniform halo communication - assumes 4 ranks for this example
// Ranks + (size-1) = # of sends 
// 0 sends to 1,2,3 and receives from no one - 1 sends to 2,3 and receives from 0 - 2 sends to 3 and receives from 0,1, 3 sends to no one receives from all
// Done asynchronously
void example_nonuniform_halo_comms(int world_size, int rank) {
    int send_rank = rank + 1;
    int recv_rank = ROOT;
    int tag = 99 + 3 * (rank + send_rank);
    int r_tag = 99 + 3 * (recv_rank + rank);
    int size = 5 + rank;
    int in_size = 5 + recv_rank;

    MPIArrayKokkos <int> myhalo;
    MPIArrayKokkos <int> halo0;
    MPIArrayKokkos <int> halo1;
    MPIArrayKokkos <int> halo2;

    myhalo = MPIArrayKokkos <int> (size);
    if (rank == 0) {
        // send rank needs to be updated for each rank
        myhalo.mpi_setup(send_rank, tag, MPI_COMM_WORLD);
#ifdef FAST
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
        // Waits
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
#endif
    }
    else if (rank == 1) {
        myhalo.mpi_setup(send_rank, tag, MPI_COMM_WORLD);
#ifdef FAST
        // Sends
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
#endif
        halo0 = MPIArrayKokkos <int> (in_size);
        halo0.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        // Recvs
        halo0.halo_irecv();
        // Waits
        halo0.wait_recv();
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
#endif
    }
    else if (rank == 2) {
        myhalo.mpi_setup(send_rank, tag, MPI_COMM_WORLD);
#ifdef FAST
        myhalo.halo_isend();
#endif
        halo0 = MPIArrayKokkos <int> (in_size);
        halo0.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo0.halo_irecv();
#endif
        halo1 = MPIArrayKokkos <int> (in_size+1);
        r_tag = 99 + 3 * (++recv_rank + rank);
        halo1.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        // Recvs
        halo1.halo_irecv();
        // Waits
        halo0.wait_recv();
        halo1.wait_recv();
        myhalo.wait_send();
#endif
    }
    else if (rank == 3) {
        myhalo.mpi_setup(send_rank, tag, MPI_COMM_WORLD);
        halo0 = MPIArrayKokkos <int> (in_size);
        halo0.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo0.halo_irecv();
#endif
        halo1 = MPIArrayKokkos <int> (in_size+1);
        r_tag = 99 + 3 * (++recv_rank + rank);
        halo1.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo1.halo_irecv();
#endif
        halo2 = MPIArrayKokkos <int> (in_size+2);
        r_tag = 99 + 3 * (++recv_rank + rank);
        halo2.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo2.halo_irecv();
        // Waits
        halo0.wait_recv();
        halo1.wait_recv();
        halo2.wait_recv();
#endif
    }
    else {
        printf("size is too big, rank %d will not do work\n", rank);
    }

    FOR_ALL(idx, 0, size, { 
        myhalo(idx) = idx * rank;
    });  
    Kokkos::fence();

#ifndef FAST
    if (rank == 0) {
        // Sends
        //printf("Rank %d sending message to %d with tag %d\n", rank, myhalo.get_rank(), myhalo.get_tag());
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
        // Waits
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
    }
    else if (rank == 1) {
        // Sends
        myhalo.halo_isend();
        tag = 99 + 3 * (++send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.halo_isend();
        // Recvs
        halo0.halo_irecv();
        // Waits
        halo0.wait_recv();
        myhalo.wait_send();
        tag = 99 + 3 * (--send_rank + rank);
        myhalo.mpi_set_rank(send_rank);
        myhalo.mpi_set_tag(tag);
        myhalo.wait_send();
    }
    else if (rank == 2) {
        // Sends
        myhalo.halo_isend();
        // Recvs
        halo0.halo_irecv();
        halo1.halo_irecv();
        // Waits
        halo0.wait_recv();
        halo1.wait_recv();
        myhalo.wait_send();
    }
    else if (rank == 3) {
        // Recvs
        halo0.halo_irecv();
        halo1.halo_irecv();
        halo2.halo_irecv();
        // Waits
        halo0.wait_recv();
        halo1.wait_recv();
        halo2.wait_recv();
    }
    else {
        printf("size is too big, rank %d will not do work\n", rank);
    }
#endif

    if (rank > 1) {
        // Print statement just to show it copied up to GPU.
        FOR_ALL(idx, 0, in_size+1, {
            printf("Rank %d) idx %d with value %d\n", rank, idx, halo1(idx));
        });
        Kokkos::fence();
    }

}

// Example for DENSE halo sending and receiving
// Only 1 halo communication
// Done asynchrnously
void example_halo_comms(int world_size, int rank) {
    int size = 10;
    int s_neighbor = (rank + 1) % world_size; 
    int r_neighbor = (rank + (world_size - 1)) % world_size; 
    int s_tag = 10 * rank + 20 * s_neighbor;
    int r_tag = 20 * rank + 10 * r_neighbor;
    
    MPIArrayKokkos <int> myhalo = MPIArrayKokkos <int> (size);
    MPIArrayKokkos <int> neibhalo = MPIArrayKokkos <int> (size);

    //printf("Rank %d rneighb %d rtag %d sneighb %d stag %d\n", rank, r_neighbor, r_tag, s_neighbor, s_tag);

    myhalo.mpi_setup(s_neighbor, s_tag, MPI_COMM_WORLD);
    neibhalo.mpi_setup(r_neighbor, r_tag, MPI_COMM_WORLD);
    FOR_ALL(idx, 0, size, { 
        myhalo(idx) = idx * rank;
    });  
    Kokkos::fence();

    // Send halos
    myhalo.halo_isend();
    // Receiv halos
    //neibhalo.halo_recv();
    neibhalo.halo_irecv();
    // Block until I have received
    neibhalo.wait_recv();
    // Block until that send has been received
    myhalo.wait_send();
    // The asynch receive is not mandatory
    // Could simply asynch send, then have a blocking receive
    // Asynch send is necessary to not get a communication block


    FOR_ALL(idx, 0, size, {
        printf("Rank %d) idx %d with value %d\n", rank, idx, neibhalo(idx));
    });
    Kokkos::fence();

}
*/

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  { // kokkos scope

  double begin_time_total = MPI_Wtime();

  int world_size,
      rank;

  // get world_size and rank
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //example_halo_comms(world_size, rank);
  //example_nonuniform_halo_comms(world_size, rank);
  example_comms_with_map(world_size, rank);

  // stop timing
  double end_time = MPI_Wtime();

  if (rank == ROOT) {
    printf("\n");
    printf("Number of MPI processes = %d\n", world_size);
    printf("Total code time was %10.6e seconds.\n", end_time-begin_time_total);
    //printf("Main loop time was %10.6e seconds.\n", end_time-begin_time_main_loop);
  }


  } // end kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}


