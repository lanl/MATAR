// #include <iostream>
// #include <cstdlib>
// #include <cstring>
// #include <vector>
// #include <memory>
// #include <mpi.h>
// #include <set>
// #include <map>


// #include "mesh.h"
// #include "state.h"
// #include "mesh_io.h"

#include "decomp_utils.h"

// Include Scotch headers
#include "scotch.h"
#include "ptscotch.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope

    int world_size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t_main_start = MPI_Wtime();

    // Mesh size
    double origin[3] = {0.0, 0.0, 0.0};
    double length[3] = {1.0, 1.0, 1.0};
    int num_elems_dim[3] = {20, 20, 20};

    // Initial mesh built on rank zero
    Mesh_t initial_mesh;
    node_t initial_node;

    // Mesh partitioned by pt-scotch, including ghost
    Mesh_t final_mesh;
    node_t final_node;

    GaussPoint_t gauss_point;

// ********************************************************  
//              Build the initial mesh
// ********************************************************  

    double t_init_mesh_start = MPI_Wtime();
    if (rank == 0) {
        std::cout<<"World size: "<<world_size<<std::endl;
        std::cout<<"Rank "<<rank<<" Building initial mesh"<<std::endl;

        std::cout<<"Initializing mesh"<<std::endl;
        build_3d_box(initial_mesh,  initial_node, origin, length, num_elems_dim);

        // Read the mesh from a file
        // read_vtk_mesh(initial_mesh, initial_node, 3, "/home/jacobmoore/Desktop/repos/MATAR/meshes/impellerOpt.vtk");

        double t_init_mesh_end = MPI_Wtime();
        std::cout << "Initial mesh build time: " << (t_init_mesh_end - t_init_mesh_start) << " seconds" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
// ********************************************************  
//             Partition and balance the mesh
// ********************************************************  
    double t_partition_start = MPI_Wtime();
    partition_mesh(initial_mesh, final_mesh, initial_node, final_node, gauss_point, world_size, rank);
    double t_partition_end = MPI_Wtime();
    
    
    
    if(rank == 0) {
        printf("Mesh partitioning time: %.2f seconds\n", t_partition_end - t_partition_start);
    }

    // write_vtk(intermediate_mesh, intermediate_node, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    write_vtu(final_mesh, final_node, gauss_point, rank, MPI_COMM_WORLD);
    // write_vtk(final_mesh, final_node, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // Stop timer and get execution time
    double t_main_end = MPI_Wtime();
    
    if(rank == 0) {
        printf("Total execution time: %.2f seconds\n", t_main_end - t_main_start);
    }

    } // end MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

    return 0;
}