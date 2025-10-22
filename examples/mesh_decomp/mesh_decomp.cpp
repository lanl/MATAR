#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <mpi.h>


#include "mesh.h"
#include "state.h"
#include "mesh_io.h"

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



    // Create mesh, gauss points, and node data structures on each rank
    Mesh_t mesh;
    GaussPoint_t GaussPoints;
    node_t node;


    if (rank == 0) {
        std::cout<<"Rank "<<rank<<" Building initial mesh"<<std::endl;
        std::cout<<"World size: "<<world_size<<std::endl;

        Mesh_t initial_mesh;
        GaussPoint_t initial_GaussPoints;
        node_t initial_node;
    
        double origin[3] = {0.0, 0.0, 0.0};
        double length[3] = {1.0, 1.0, 1.0};
        int num_elems[3] = {10, 10, 10};
    
        std::cout<<"Initializing mesh"<<std::endl;
        build_3d_box(initial_mesh, initial_GaussPoints, initial_node, origin, length, num_elems);




    }


    




    if (rank == 0) std::cout<<"Finished decomposition"<<std::endl;
    
    } // end MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();
    return 0;
}