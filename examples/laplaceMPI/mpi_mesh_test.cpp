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
#define N 20

using namespace mtr; // matar namespace
    int get_id(int i, int j, int k, int num_i, int num_j)
    {
        return i + j * num_i + k * num_i * num_j;
    }

    void build_3d_box(int world_size, int rank)
    {
        printf("Rank %d Creating a 3D box mesh \n", rank);

        const int num_dim = 3;

        const double lx = pow(2.0, (double) N);
        const double ly = pow(2.0, (double) N);
        const double lz = pow(2.0, (double) N);

        const double num_elems_i = N;
        const double num_elems_j = N;
        const double num_elems_k = N;

        const int num_points_i = num_elems_i + 1; // num points in x
        const int num_points_j = num_elems_j + 1; // num points in y
        const int num_points_k = num_elems_k + 1; // num points in y

        const int num_nodes = num_points_i * num_points_j * num_points_k;

        const double dx = lx / ((double)num_elems_i);  // len/(num_elems_i)
        const double dy = ly / ((double)num_elems_j);  // len/(num_elems_j)
        const double dz = lz / ((double)num_elems_k);  // len/(num_elems_k)

        const int num_elems = num_elems_i * num_elems_j * num_elems_k;
        std::vector<double> origin(num_dim);
        for (int i = 0; i < num_dim; i++) { origin[i] = 0; }

        // --- 3D parameters ---
        // const int num_faces_in_elem  = 6;  // number of faces in elem
        // const int num_points_in_elem = 8;  // number of points in elem
        // const int num_points_in_face = 4;  // number of points in a face
        // const int num_edges_in_elem  = 12; // number of edges in a elem

        // --- mesh node ordering ---
        // Convert ijk index system to the finite element numbering convention
        // for vertices in elem
        auto convert_point_number_in_Hex = CArray<int>(8);
        convert_point_number_in_Hex(0) = 0;
        convert_point_number_in_Hex(1) = 1;
        convert_point_number_in_Hex(2) = 3;
        convert_point_number_in_Hex(3) = 2;
        convert_point_number_in_Hex(4) = 4;
        convert_point_number_in_Hex(5) = 5;
        convert_point_number_in_Hex(6) = 7;
        convert_point_number_in_Hex(7) = 6;

        int rk_num_bins = 1;

        // intialize node variables

        // extra math for mpi decomposition
        int grid_world_size, krem, jrem, col, row, kadd, jadd, ksub, jsub, kchunk, jchunk, kstart, kend, jstart, jend;
        grid_world_size = sqrt(world_size);
        krem = num_points_k % grid_world_size; // how many extra points we have after a division of ranks
        jrem = num_points_j % grid_world_size;
        col = rank % grid_world_size;
        row = rank / grid_world_size;
        kadd = col / (grid_world_size - krem); // 1 if you have an extra piece, 0 otherwise
        jadd = row / (grid_world_size - jrem); // 1 if you have an extra piece, 0 otherwise
        ksub = kadd * (grid_world_size - krem); // subtraction from your start based on how many other ranks have an extra piece
        jsub = jadd * (grid_world_size - jrem);
        kchunk = num_points_k / grid_world_size + kadd;
        jchunk = num_points_j / grid_world_size + jadd;
        kstart = col * kchunk - ksub;
        kend = kstart +  kchunk;
        jstart = row * jchunk - jsub;
        jend = jstart +  jchunk;

        // --- Build nodes ---
        auto coords = MPIArrayKokkos <double> (MPI_DOUBLE, rk_num_bins, num_points_i * kchunk * jchunk, num_dim);
        // populate the point data structures
        for (int k = kstart, k_loc = 0; k < kend; k++, k_loc++) {
            for (int j = jstart, j_loc = 0; j < jend; j++, j_loc++) {
                for (int i = 0; i < num_points_i; i++) {
                    int node_gid = get_id(i, j_loc, k_loc, num_points_i, jchunk);

                    // store the point coordinates
                    coords.host(0, node_gid, 0) = origin[0] + (double)i * dx;
                    coords.host(0, node_gid, 1) = origin[1] + (double)j * dy;
                    coords.host(0, node_gid, 2) = origin[2] + (double)k * dz;
                } // end for i
            } // end for j
        } // end for k

        for (int rk_level = 1; rk_level < rk_num_bins; rk_level++) {
            for (int node_gid = 0; node_gid < num_points_i * kchunk * jchunk; node_gid++) {
                coords.host(rk_level, node_gid, 0) = coords.host(0, node_gid, 0);
                coords.host(rk_level, node_gid, 1) = coords.host(0, node_gid, 1);
                coords.host(rk_level, node_gid, 2) = coords.host(0, node_gid, 2);
            }
        }
        coords.update_device();

        // intialize elem variables

        krem = (int) num_elems_k % grid_world_size; // how many extra points we have after a division of ranks
        jrem = (int) num_elems_j % grid_world_size;
        col = rank % grid_world_size;
        row = rank / grid_world_size;
        kadd = col / (grid_world_size - krem); // 1 if you have an extra piece, 0 otherwise
        jadd = row / (grid_world_size - jrem); // 1 if you have an extra piece, 0 otherwise
        ksub = kadd * (grid_world_size - krem); // subtraction from your start based on how many other ranks have an extra piece
        jsub = jadd * (grid_world_size - jrem);
        kchunk = (int) num_elems_k / grid_world_size + kadd;
        jchunk = (int) num_elems_j / grid_world_size + jadd;
        kstart = col * kchunk - ksub;
        kend = kstart +  kchunk;
        jstart = row * jchunk - jsub;
        jend = jstart +  jchunk;
        // --- Build elems  ---

        auto nodes_in_elem = MPIArrayKokkos <double> (MPI_DOUBLE, num_elems_i * kchunk * jchunk, 8);
        // populate the elem center data structures
        for (int k = kstart, k_loc = 0; k < kend / world_size; k++, k_loc++) {
            for (int j = kstart, j_loc = 0; j < jend; j++, j_loc++) {
                for (int i = 0; i < num_elems_i; i++) {
                    int elem_gid = get_id(i, j_loc, k_loc, num_elems_i, jchunk);

                    // store the point IDs for this elem where the range is
                    // (i:i+1, j:j+1, k:k+1) for a linear hexahedron
                    int this_point = 0;
                    for (int kcount = k; kcount <= k + 1; kcount++) {
                        for (int jcount = j; jcount <= j + 1; jcount++) {
                            for (int icount = i; icount <= i + 1; icount++) {
                                int node_gid = get_id(icount, jcount, kcount,
                                                  num_points_i, num_points_j);

                                // convert this_point index to the FE index convention
                                int this_index = convert_point_number_in_Hex(this_point);

                                // store the points in this elem according the the finite
                                // element numbering convention
                                nodes_in_elem.host(elem_gid, this_index) = node_gid;

                                // increment the point counting index
                                this_point = this_point + 1;
                            } // end for icount
                        } // end for jcount
                    }  // end for kcount
                } // end for i
            } // end for j
        } // end for k
        // update device side
        nodes_in_elem.update_device();
        // intialize corner variables
        //corner.initialize(num_corners, num_dim);

        // Build connectivity
        //mesh.build_connectivity();
}

int main(int argc, char *argv[])
{

  //MPI_Init(&argc, &argv);
  //Kokkos::initialize(argc, argv);
  MATAR_MPI_INIT
  MATAR_KOKKOS_INIT
  { // kokkos scope

  //double begin_time_total = MPI_Wtime();
  double begin_time_total = MATAR_MPI_TIME

  int world_size,
      rank;

  // get world_size and rank
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //build_3d_box(world_size, rank);

  auto typetestb = MPIArrayKokkos <bool> (MPI_C_BOOL, 8, 8);
  auto typetesti = MPIArrayKokkos <int> (MPI_INT, 8, 8);
  auto typetestd = MPIArrayKokkos <double> (MPI_DOUBLE, 8, 8);

  typetestb.host(0) = false;
  typetesti.host(0) = 8;
  typetestd.host(0) = 8.0;

  // stop timing
  //double end_time = MPI_Wtime();
  double end_time = MATAR_MPI_TIME

  if (rank == ROOT) {
    printf("\n");
    printf("Number of MPI processes = %d\n", world_size);
    printf("Total code time was %10.6e seconds.\n", end_time-begin_time_total);
    //printf("Main loop time was %10.6e seconds.\n", end_time-begin_time_main_loop);
  }


  } // end kokkos scope
  MATAR_KOKKOS_FINALIZE
  MATAR_MPI_FINALIZE
  //Kokkos::finalize();
  //MPI_Finalize();
  return 0;
}
