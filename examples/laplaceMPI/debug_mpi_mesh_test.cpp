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

    //void build_3d_box(mesh_t& mesh, elem_t& elem, node_t& node, corner_t& corner, simulation_parameters_t& sim_param) const
    void build_3d_box(int world_size, int rank)
    {
        //printf(" Creating a 3D box mesh \n");

        const int num_dim = 3;

        const double lx = pow(2.0, (double) N);
        const double ly = pow(2.0, (double) N);
        const double lz = pow(2.0, (double) N);

        //const double num_elems_i = pow(2.0, (double) N);
        //const double num_elems_j = pow(2.0, (double) N);
        //const double num_elems_k = pow(2.0, (double) N);
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
        //DAN
        //for (int i = 0; i < num_dim; i++) { origin[i] = sim_param.mesh_input.origin.host(i); }
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

        //DAN
        //int rk_num_bins = sim_param.dynamic_options.rk_num_bins;
        int rk_num_bins = 1;

        // intialize node variables
        //mesh.initialize_nodes(num_nodes);
        //node.initialize(rk_num_bins, num_nodes, num_dim);

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
        //printf("Rank %d grid_world_size %d col %d row %d kchunk %d jchunk %d kstart %d jstart%d\n", rank, grid_world_size, col, row, kchunk, jchunk, kstart, jstart);
        //printf("Rank %d jsub %d jadd %d\n", rank, jsub, jadd);

        // --- Build nodes ---
        //DAN
        //DANauto coords = MPIArrayKokkos <double> (MPI_DOUBLE, rk_num_bins, num_nodes, num_dim);
        auto coords = MPIArrayKokkos <double> (MPI_DOUBLE, rk_num_bins, num_points_i * kchunk * jchunk, num_dim);
        // populate the point data structures
        for (int k = kstart, k_loc = 0; k < kend; k++, k_loc++) {
            for (int j = jstart, j_loc = 0; j < jend; j++, j_loc++) {
                for (int i = 0; i < num_points_i; i++) {
                    // global id for the point
                    //DAN
                    //int global_k = num_points_k / world_size * rank + k;
                    //printf("Rank %d size %d global k %d local k\n", rank, kchunk * jchunk, k, k_loc);
                    int node_gid = get_id(i, j_loc, k_loc, num_points_i, jchunk);
                    //if (node_gid < 0 || node_gid > num_points_i * kchunk * jchunk) printf("Rank %d %d,%d,%d\n", rank, i, j, k);

                    // store the point coordinates
                    coords.host(0, node_gid, 0) = origin[0] + (double)i * dx;
                    //printf("%f\n", coords.host(0, node_gid, 0));
                    coords.host(0, node_gid, 1) = origin[1] + (double)j * dy;
                    coords.host(0, node_gid, 2) = origin[2] + (double)k * dz;
                    //printf("Rank %d, Coords %d,%d,%d - %d,%d,%d\n", rank, i, j, k, coords.host(0, node_gid, 0), coords.host(0, node_gid, 1), coords.host(0, node_gid, 2));
                    //printf("rank %d node_gid %d j val %f\n", rank, get_id(i, j, k, num_points_i, num_points_j), coords.host(0, node_gid, 1));
                } // end for i
            } // end for j
        } // end for k

        for (int rk_level = 1; rk_level < rk_num_bins; rk_level++) {
            //MPIfor (int node_gid = 0; node_gid < num_nodes / world_size; node_gid++) {
            for (int node_gid = 0; node_gid < num_points_i * kchunk * jchunk; node_gid++) {
                //int a_node_gid = num_nodes / world_size * rank + node_gid
                coords.host(rk_level, node_gid, 0) = coords.host(0, node_gid, 0);
                coords.host(rk_level, node_gid, 1) = coords.host(0, node_gid, 1);
                coords.host(rk_level, node_gid, 2) = coords.host(0, node_gid, 2);
            }
        }
        coords.update_device();

        // intialize elem variables
        //mesh.initialize_elems(num_elems, num_dim);
        //elem.initialize(rk_num_bins, num_nodes, 3); // always 3D here, even for 2D


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

        //DAN
        //MPIauto nodes_in_elem = MPIArrayKokkos <double> (MPI_DOUBLE, num_elems, 8);
        auto nodes_in_elem = MPIArrayKokkos <double> (MPI_DOUBLE, num_elems_i * kchunk * jchunk, 8);
        // populate the elem center data structures
        for (int k = kstart, k_loc = 0; k < kend / world_size; k++, k_loc++) {
            for (int j = kstart, j_loc = 0; j < jend; j++, j_loc++) {
                for (int i = 0; i < num_elems_i; i++) {
                    // global id for the elem
                    //DAN
                    int elem_gid = get_id(i, j_loc, k_loc, num_elems_i, jchunk);

                    // store the point IDs for this elem where the range is
                    // (i:i+1, j:j+1, k:k+1) for a linear hexahedron
                    int this_point = 0;
                    for (int kcount = k; kcount <= k + 1; kcount++) {
                        for (int jcount = j; jcount <= j + 1; jcount++) {
                            for (int icount = i; icount <= i + 1; icount++) {
                                // global id for the points
                    //DAN
                                //int kcount_global = num_elems_k / world_size * rank + kcount;
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
        //int num_corners = num_elems * mesh.num_nodes_in_elem;
        //mesh.initialize_corners(num_corners);
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

  build_3d_box(world_size, rank);

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
