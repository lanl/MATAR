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

// Change to 0 or 1 as needed
#define TRACK_PROGRESS  0

#if defined HAVE_CUDA || defined HAVE_HIP
  #define GPU 1
#else
  #define GPU 0
#endif

using namespace mtr; // matar namespace

int width = 1000;
int height = 1000;
int max_num_iterations = 1000;
double temp_tolerance = 0.01;

// Example for the scatter and gather functions
// All it does is take an array, split it up,
// half the value of each chunk, and send those
// back to be grouped again by the root
void example_gather(int world_size, int rank) {
    int size = 12;
    int sub_size = size / world_size;
    MPIDCArrayKokkos <float> arrAll = MPIDCArrayKokkos <float> (MPI_FLOAT, size);
    MPIDCArrayKokkos <float> arrSub = MPIDCArrayKokkos <float> (MPI_FLOAT, sub_size);

    if (rank == ROOT) {    
        FOR_ALL(idx, 0, size, { 
            arrAll(idx) = idx+1;
            printf("%d) %f\n", idx, arrAll(idx));   
        });  
        Kokkos::fence();
        printf("------------\n");
    }

    arrAll.scatter(sub_size, arrSub, sub_size, ROOT, MPI_COMM_WORLD); 

    FOR_ALL(idx, 0, sub_size, {
        arrSub(idx) = arrSub(idx) / 2; 
    });
    Kokkos::fence();

    // Gather
    /*
    arrSub.gather(sub_size, arrAll, sub_size, ROOT, MPI_COMM_WORLD);
    
    if (rank == ROOT) {
        FOR_ALL(idx, 0, size, {
            printf("%d) %f\n", idx, arrAll(idx));   
        });
        Kokkos::fence();
    }
    */
    // Allgather
    arrSub.allgather(sub_size, arrAll, sub_size, MPI_COMM_WORLD);
    
    // A rank other than Root
    if (rank == 1) {
        FOR_ALL(idx, 0, size, {
            printf("%d) %f\n", idx, arrAll(idx));   
        });
        Kokkos::fence();
    }
}

// Example for sending, receiving, and broadcasting
void example_simple_comms(int world_size, int rank) {
    int size = 10;
    int tag = 99;
    MPIDCArrayKokkos <int> mca_s = MPIDCArrayKokkos <int> (MPI_INT, size);
    MPIDCArrayKokkos <int> mca_r = MPIDCArrayKokkos <int> (MPI_INT, size);

    if (rank == ROOT) {
        FOR_ALL(idx, 0, size, { 
            mca_s(idx) = idx+1;
        });  
        Kokkos::fence();
    }

    if (rank == ROOT) {
        mca_s.isend(size, 1, tag, MPI_COMM_WORLD);
        mca_s.wait_send();
    }
    else {
        mca_r.irecv(size, 0, tag, MPI_COMM_WORLD);
        mca_r.wait_recv();
    }
    mca_r.barrier(MPI_COMM_WORLD);  

    if (rank != ROOT) {
        FOR_ALL(idx, 0, size, {
            printf("idx %d with value %d\n", idx, mca_r(idx));
        });
        Kokkos::fence();
    }

    if (rank == ROOT) {
        FOR_ALL(idx, 0, size, {
            mca_s(idx) = mca_s(idx) * 2;       
        });
        Kokkos::fence();
    }

    mca_s.broadcast(size, ROOT, MPI_COMM_WORLD);

    if (rank != ROOT) {
        FOR_ALL(idx, 0, size, {
            printf("idx %d with value %d\n", idx, mca_s(idx));
        });
        Kokkos::fence();
    }
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

  //example_gather(world_size, rank);
  example_simple_comms(world_size, rank);

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


