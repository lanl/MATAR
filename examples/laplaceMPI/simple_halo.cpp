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

int width = 1000;
int height = 1000;
int max_num_iterations = 1000;
double temp_tolerance = 0.01;


// Example for DENSE halo sending and receiving
// Only multiple, non uniform halo communication - assumes 4 ranks for this example
// Ranks + (size-1) = # of sends (0 sends to 1,2,3 - 1 sends to 2,3 - 2 sends to 3, 3 sends to no one)
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

    myhalo = MPIArrayKokkos <int> (MPI_INT, size);
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
        halo0 = MPIArrayKokkos <int> (MPI_INT, in_size);
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
        halo0 = MPIArrayKokkos <int> (MPI_INT, in_size);
        halo0.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo0.halo_irecv();
#endif
        halo1 = MPIArrayKokkos <int> (MPI_INT, in_size+1);
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
        halo0 = MPIArrayKokkos <int> (MPI_INT, in_size);
        halo0.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo0.halo_irecv();
#endif
        halo1 = MPIArrayKokkos <int> (MPI_INT, in_size+1);
        r_tag = 99 + 3 * (++recv_rank + rank);
        halo1.mpi_setup(recv_rank, r_tag, MPI_COMM_WORLD);
#ifdef FAST
        halo1.halo_irecv();
#endif
        halo2 = MPIArrayKokkos <int> (MPI_INT, in_size+2);
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
    
    MPIArrayKokkos <int> myhalo = MPIArrayKokkos <int> (MPI_INT, size);
    MPIArrayKokkos <int> neibhalo = MPIArrayKokkos <int> (MPI_INT, size);

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

///*
    FOR_ALL(idx, 0, size, {
        printf("Rank %d) idx %d with value %d\n", rank, idx, neibhalo(idx));
    });
    Kokkos::fence();
//*/
}

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
  example_nonuniform_halo_comms(world_size, rank);

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


