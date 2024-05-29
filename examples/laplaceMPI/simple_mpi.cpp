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

void example_gather(int world_size, int rank) {
    int size = 9;
    int root = 0;  
    MPIDCArrayKokkos <float> arrA = MPIDCArrayKokkos <float> (MPI_INT, 10);
    MPIDCArrayKokkos <float> arrB = MPIDCArrayKokkos <float> (MPI_INT, 10);

    if (rank == root) {    
        FOR_ALL(idx, 0, size, { 
            arrA(idx) = idx+1;
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

  // main loop
  MPIDCArrayKokkos <int> mca_s = MPIDCArrayKokkos <int> (MPI_INT, 10);
  MPIDCArrayKokkos <int> mca_r = MPIDCArrayKokkos <int> (MPI_INT, 10);

  if (rank == 0) {
    FOR_ALL(idx, 0, 10, { 
        mca_s(idx) = idx+1;
    });  
    Kokkos::fence();
  }

  if (rank == 0) {
    mca_s.isend(10, 1, 99, MPI_COMM_WORLD);
    mca_s.wait_send();
  }
  else {
    mca_r.irecv(10, 0, 99, MPI_COMM_WORLD);
    mca_r.wait_recv();
  }
  mca_r.barrier(MPI_COMM_WORLD);  

  if (rank != 0) {
    FOR_ALL(idx, 0, 10, {
        printf("idx %d with value %d\n", idx, mca_r(idx));
    });
    Kokkos::fence();
  }


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


