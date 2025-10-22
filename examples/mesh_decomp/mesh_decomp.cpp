#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <mpi.h>

// Include Scotch headers
#include "scotch.h"
#include "ptscotch.h"


struct initial_mesh_t {
    int num_elems;                    // Number of elements
    
    std::vector<SCOTCH_Num> nodes_in_elem;  // Nodes in an element
    std::vector<SCOTCH_Num> elems_in_elem;  // Elements in an element
    
    std::vector<SCOTCH_Num> verttab;  // Start index in edgetab for each element (size num_elems+1)
    std::vector<SCOTCH_Num> edgetab;  // Adjacency info: neighboring element indices
};


int main(int argc, char** argv) {

    initial_mesh_t initial_mesh;




    return 0;
}