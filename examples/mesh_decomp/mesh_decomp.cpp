#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <mpi.h>
#include <set>


#include "mesh.h"
#include "state.h"
#include "mesh_io.h"

// Include Scotch headers
#include "scotch.h"
#include "ptscotch.h"

// Timer class for timing the execution of the matrix multiplication
class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        bool is_running;
    
    public:
        Timer() : is_running(false) {}
        
        void start() {
            start_time = std::chrono::high_resolution_clock::now();
            is_running = true;
        }
        
        double stop() {
            if (!is_running) {
                std::cerr << "Timer was not running!" << std::endl;
                return 0.0;
            }
            end_time = std::chrono::high_resolution_clock::now();
            is_running = false;
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000.0; // Convert to milliseconds
        }
};

void print_rank_mesh_info(Mesh_t& mesh, int rank) {

    std::cout<<std::endl;
    std::cout<<"Rank "<<rank<<" printing mesh info"<<std::endl;
    std::cout<<"Mesh has "<<mesh.num_elems<<" elements"<<std::endl;
    std::cout<<"Mesh has "<<mesh.num_nodes<<" nodes"<<std::endl;

    for (int i = 0; i < mesh.num_elems; i++) {
        std::cout<<"Element "<<i<<" has nodes: ";
        for (int j = 0; j < mesh.num_nodes_in_elem; j++) {
            std::cout<<mesh.nodes_in_elem.host(i, j)<<" ";
        }
        std::cout<<std::endl;
        std::cout<<"Which have global indices of : ";
        for (int k = 0; k < mesh.num_nodes_in_elem; k++) {
            std::cout<<mesh.local_to_global_node_mapping.host(mesh.nodes_in_elem.host(i, k))<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void calc_elements_per_rank(std::vector<int>& elems_per_rank, int num_elems, int world_size){
    // Compute elements to send to each rank; handle remainders for non-even distribution
    std::fill(elems_per_rank.begin(), elems_per_rank.end(), num_elems / world_size);
    int remainder = num_elems % world_size;
    for (int i = 0; i < remainder; ++i) {
        elems_per_rank[i] += 1;
    }
}

void print_mesh_info(Mesh_t& mesh){
    std::cout<<"Mesh has "<<mesh.num_elems<<" elements"<<std::endl;
    std::cout<<"Mesh has "<<mesh.num_nodes<<" nodes"<<std::endl;

    for (int i = 0; i < mesh.num_elems; i++) {
        std::cout<<"Element "<<i<<" has nodes: ";
        for (int j = 0; j < mesh.num_nodes_in_elem; j++) {
            std::cout<<mesh.nodes_in_elem.host(i, j)<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}


int main(int argc, char** argv) {

    // Create and start timer
    Timer timer;
    timer.start();

    bool print_info = true;
    bool print_vtk = false;


    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope

    int world_size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Initial mesh size
    double origin[3] = {0.0, 0.0, 0.0};
    double length[3] = {1.0, 1.0, 1.0};
    int num_elems_dim[3] = {2, 2, 2};

    Mesh_t initial_mesh;
    GaussPoint_t initial_GaussPoints;
    node_t initial_node;

    // Create mesh, gauss points, and node data structures on each rank
    Mesh_t mesh;
    GaussPoint_t GaussPoints;
    node_t node;

    int num_elements_on_rank = 0;
    int num_nodes_on_rank = 0;

    int num_nodes_per_elem = 0;

    std::vector<int> elements_on_rank;  
    std::vector<int> nodes_on_rank;


    std::vector<int> elems_per_rank(world_size); // number of elements to send to each rank size(world_size)
    std::vector<int> nodes_per_rank(world_size); // number of nodes to send to each rank size(world_size)

    // create a 2D vector of elements to send to each rank
    std::vector<std::vector<int>> elements_to_send(world_size);

    // create a 2D vector of nodes to send to each rank
    std::vector<std::vector<int>> nodes_to_send(world_size);

    // Create a 2D vector to hold the nodal positions on each rank
    std::vector<std::vector<double>> node_pos_to_send(world_size);

    // create a 2D vector to hold the node positions on each rank
    std::vector<std::vector<double>> node_pos_on_rank(world_size);


// ********************************************************  
//              Build the initial mesh
// ********************************************************  
    if (rank == 0) {
        std::cout<<"World size: "<<world_size<<std::endl;
        std::cout<<"Rank "<<rank<<" Building initial mesh"<<std::endl;
    
        std::cout<<"Initializing mesh"<<std::endl;
        build_3d_box(initial_mesh, initial_GaussPoints, initial_node, origin, length, num_elems_dim);

        num_nodes_per_elem = initial_mesh.num_nodes_in_elem;

        // print out the nodes associated with each element in the initial mesh
        if (print_info) {
            print_mesh_info(initial_mesh);
        }

        // Compute elements to send to each rank; handle remainders for non-even distribution
        calc_elements_per_rank(elems_per_rank, initial_mesh.num_elems, world_size);
    }

    // int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
    MPI_Bcast(&num_nodes_per_elem, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);
    
// ********************************************************  
//        Scatter the number of elements to each rank
// ******************************************************** 
    // All ranks participate in the scatter operation
    // MPI_Scatter signature:
    // MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //             void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //             int root, MPI_Comm comm)
    MPI_Scatter(elems_per_rank.data(), 1, MPI_INT, 
                &num_elements_on_rank, 1, MPI_INT, 
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Resize the elements_on_rank vector to hold the received data
    elements_on_rank.resize(num_elements_on_rank);
    



// ********************************************************  
//     Scatter the actual element global ids to each rank
// ******************************************************** 
    if (rank == 0) {

        //print elements per rank
        std::cout<<std::endl;
        int elem_gid = 0;
        for (int i = 0; i < world_size; i++) {

            for (int j = 0; j < elems_per_rank[i]; j++) {
                elements_to_send[i].push_back(elem_gid);
                elem_gid++;
            }
        }

        if (print_info) {
            for (int i = 0; i < world_size; i++) {
                std::cout<<std::endl;
                std::cout<<"Rank "<<i<<" will get "<<elems_per_rank[i]<<" elements: ";
                for (int j = 0; j < elems_per_rank[i]; j++) {
                    std::cout<<elements_to_send[i][j]<<" ";
                }
            }
            std::cout<<std::endl;
        }

        // Prepare data for MPI_Scatterv (scatter with variable counts)
        // Flatten the 2D elements_to_send into a 1D array
        std::vector<int> all_elements;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = elems_per_rank[i];
            displs[i] = displacement;
            // Copy elements for rank i to the flattened array
            for (int j = 0; j < elems_per_rank[i]; j++) {
                all_elements.push_back(elements_to_send[i][j]);
            }
            displacement += elems_per_rank[i];
        }

        // Send the elements to each rank
        MPI_Scatterv(all_elements.data(), sendcounts.data(), displs.data(), MPI_INT,
                     elements_on_rank.data(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    } 
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     elements_on_rank.data(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    

    MPI_Barrier(MPI_COMM_WORLD);

    if (print_info) {
        std::cout << "Rank " << rank << " received elements: ";
        for (int i = 0; i < num_elements_on_rank; i++) {
            std::cout << elements_on_rank[i] << " ";
        }
        std::cout << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);


// ****************************************************************************************** 
//     Scatter the number of nodes to each rank and compute which nodes to send to each rank
// ****************************************************************************************** 

    if (rank == 0) {

        // Populate the nodes_to_send array by finding all nodes in the elements in elements_to_send and removing duplicates    
        for (int i = 0; i < world_size; i++) {      
            std::set<int> nodes_set;
            for (int j = 0; j < elems_per_rank[i]; j++) {
                for (int k = 0; k < num_nodes_per_elem; k++) {
                    nodes_set.insert(initial_mesh.nodes_in_elem.host(elements_to_send[i][j], k));
                }
            }
            nodes_to_send[i] = std::vector<int>(nodes_set.begin(), nodes_set.end());
        } 

        if (print_info) {

            for (int i = 0; i < world_size; i++) {
                nodes_per_rank[i] = nodes_to_send[i].size();
            }
            std::cout<<std::endl;
            // print the nodes_to_send array
            for (int i = 0; i < world_size; i++) {

                std::cout<<std::endl;
                std::cout<<"Rank "<<i<<" will get "<<nodes_to_send[i].size()<<" nodes: ";

                for (int j = 0; j < nodes_to_send[i].size(); j++) {
                    std::cout<<nodes_to_send[i][j]<<" ";
        }
        std::cout<<std::endl;
            }
        }
    }

    // Send the number of nodes to each rank using MPI_scatter
    MPI_Scatter(nodes_per_rank.data(), 1, MPI_INT, &num_nodes_on_rank, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    MPI_Barrier(MPI_COMM_WORLD);

    if (print_info) {
        std::cout << "Rank " << rank << " received " << num_nodes_on_rank << " nodes" << std::endl;
    }

    // resize the nodes_on_rank vector to hold the received data
    nodes_on_rank.resize(num_nodes_on_rank);


// ****************************************************************************************** 
//     Scatter the actual node global ids to each rank
// ****************************************************************************************** 
    if (rank == 0) {

        // Prepare data for MPI_Scatterv (scatter with variable counts)
        // Flatten the 2D nodes_to_send into a 1D array
        std::vector<int> all_nodes;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = nodes_to_send[i].size();
            displs[i] = displacement;
            // Copy nodes for rank i to the flattened array
            for (int j = 0; j < nodes_to_send[i].size(); j++) {
                all_nodes.push_back(nodes_to_send[i][j]);
            }
            displacement += nodes_to_send[i].size();
        }
        // Send the nodes to each rank
        // all_nodes.data(): Pointer to the flattened array of all nodes to be sent to each rank
        // sendcounts.data(): Array with the number of nodes to send to each rank
        // displs.data(): Array with the displacement for each rank in the flattened array
        // MPI_INT: Data type of the nodes (integer)
        // nodes_on_rank.data(): Pointer to the buffer where each rank will receive its nodes
        // num_nodes_on_rank: Number of nodes that the receiving rank expects to receive
        // MPI_INT: Data type of the receive buffer (integer)
        // 0: The root rank (rank 0) that is performing the scatter
        // MPI_COMM_WORLD: The communicator
        MPI_Scatterv(all_nodes.data(), sendcounts.data(), displs.data(), MPI_INT,
                     nodes_on_rank.data(), num_nodes_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    else {

        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
            nodes_on_rank.data(), num_nodes_on_rank, MPI_INT,
            0, MPI_COMM_WORLD);

    }

    MPI_Barrier(MPI_COMM_WORLD);

// ****************************************************************************************** 
//     Scatter the node positions to each rank
// ****************************************************************************************** 
    // Create a flat 1D vector for node positions (3 coordinates per node)
    std::vector<double> node_pos_on_rank_flat(num_nodes_on_rank * 3);

    if(rank == 0)
    {
        for (int i = 0; i < world_size; i++) {
            for(int node_gid = 0; node_gid < nodes_to_send[i].size(); node_gid++)
            {
                node_pos_to_send[i].push_back(initial_node.coords.host(nodes_to_send[i][node_gid], 0));
                node_pos_to_send[i].push_back(initial_node.coords.host(nodes_to_send[i][node_gid], 1));
                node_pos_to_send[i].push_back(initial_node.coords.host(nodes_to_send[i][node_gid], 2));
            }
        }

        // Prepare data for MPI_Scatterv (scatter with variable counts)
        // Flatten the 2D node_pos_to_send into a 1D array
        std::vector<double> all_node_pos;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = nodes_to_send[i].size() * 3;
            displs[i] = displacement; // displacement is the starting index of the nodes for the current rank in the flattened array
            // Copy node positions for rank i to the flattened array
            for(int j = 0; j < nodes_to_send[i].size(); j++) {
                for(int k = 0; k < 3; k++) {
                    all_node_pos.push_back(node_pos_to_send[i][j * 3 + k]);
                }
            }
            displacement += nodes_to_send[i].size() * 3;
        }   

        // Send the node positions to each rank
        MPI_Scatterv(all_node_pos.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     node_pos_on_rank_flat.data(), num_nodes_on_rank * 3, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     node_pos_on_rank_flat.data(), num_nodes_on_rank * 3, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && print_info) {
        // Print out the node positions on this rank
        std::cout << "Rank " << rank << " received node positions: ";
        for (int i = 0; i < num_nodes_on_rank; i++) {
            std::cout << "(" << node_pos_on_rank_flat[i*3] << ", " 
                      << node_pos_on_rank_flat[i*3+1] << ", " 
                      << node_pos_on_rank_flat[i*3+2] << ") ";
        }
        std::cout << std::endl;
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1 && print_info) {
        // Print out the node positions on this rank
        std::cout << "Rank " << rank << " received node positions: ";
        for (int i = 0; i < num_nodes_on_rank; i++) {
            std::cout << "(" << node_pos_on_rank_flat[i*3] << ", " 
                      << node_pos_on_rank_flat[i*3+1] << ", " 
                      << node_pos_on_rank_flat[i*3+2] << ") ";
        }
        std::cout << std::endl;
    }

// ****************************************************************************************** 
//     Initialize the node state variables
// ****************************************************************************************** 

    // initialize node state variables, for now, we just need coordinates, the rest will be initialize by the respective solvers
    std::vector<node_state> required_node_state = { node_state::coords };
    node.initialize(num_nodes_on_rank, 3, required_node_state);

    for(int i = 0; i < num_nodes_on_rank; i++) {
        node.coords.host(i, 0) = node_pos_on_rank_flat[i*3];
        node.coords.host(i, 1) = node_pos_on_rank_flat[i*3+1];
        node.coords.host(i, 2) = node_pos_on_rank_flat[i*3+2];
    }

    node.coords.update_device();

// ****************************************************************************************** 
//     Send the element-node connectivity data from the initial mesh to each rank
// ****************************************************************************************** 

    // Send the element-node connectivity data from the initial mesh to each rank
    std::vector<int> nodes_in_elem_on_rank(num_elements_on_rank * num_nodes_per_elem);
    
    if (rank == 0) {
        // Prepare element-node connectivity data for each rank
        std::vector<int> all_nodes_in_elem;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        for(int i = 0; i < world_size; i++) {
            int num_connectivity_entries = elements_to_send[i].size() * num_nodes_per_elem; // num_nodes_per_elem nodes per element
            sendcounts[i] = num_connectivity_entries;
            displs[i] = displacement;
            
            // Copy element-node connectivity for rank i
            for(int j = 0; j < elements_to_send[i].size(); j++) {
                for(int k = 0; k < num_nodes_per_elem; k++) {
                    all_nodes_in_elem.push_back(initial_mesh.nodes_in_elem.host(elements_to_send[i][j], k));
                }
            }
            displacement += num_connectivity_entries;
        }
        // Send the connectivity data to each rank
        MPI_Scatterv(all_nodes_in_elem.data(), sendcounts.data(), displs.data(), MPI_INT,
                     nodes_in_elem_on_rank.data(), num_elements_on_rank * num_nodes_per_elem, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     nodes_in_elem_on_rank.data(), num_elements_on_rank * num_nodes_per_elem, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && print_info) {

        std::cout << "Rank " << rank << " received element-node connectivity (" 
                << num_elements_on_rank << " elements, " << nodes_in_elem_on_rank.size() << " entries):" << std::endl;
        for (int elem = 0; elem < num_elements_on_rank; elem++) {
            std::cout << "  Element " << elem << " nodes: ";
            for (int node = 0; node < num_nodes_per_elem; node++) {
                int idx = elem * num_nodes_per_elem + node;
                std::cout << nodes_in_elem_on_rank[idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);


// ****************************************************************************************** 
//     Send the element-element connectivity data from the initial mesh to each rank
// ****************************************************************************************** 

    // First, rank 0 computes how many connectivity entries each rank will receive
    // and scatters that information
    std::vector<int> elem_elem_counts(world_size);
    int total_elem_elem_entries = 0;
    
    
    if (rank == 0){
        // Calculate total number of connectivity entries for each rank
        for(int i = 0; i < world_size; i++) {
            elem_elem_counts[i] = 0;
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                elem_elem_counts[i] += initial_mesh.num_elems_in_elem(elements_to_send[i][k]);
            }

            std::cout << "Rank " << i << " will receive " << elem_elem_counts[i] << " element-element connectivity entries" << std::endl;
        }

        // Print element-element connectivity entries for each rank in the initial mesh
        for(int i = 0; i < world_size; i++) {
            std::cout << std::endl;
            std::cout << "Rank " << i << " will receive element-element connectivity entries for the following elements: "<<std::endl;
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                std::cout << "Element " << elements_to_send[i][k] << " has " << initial_mesh.num_elems_in_elem(elements_to_send[i][k]) << " element-element connectivity entries: ";
                for(int l = 0; l < initial_mesh.num_elems_in_elem(elements_to_send[i][k]); l++) {
                    std::cout << initial_mesh.elems_in_elem(elements_to_send[i][k], l) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    

    // Define total_elem_elem_entries to be the sum of the elem_elem_counts
    // Scatter the counts to each rank
    MPI_Scatter(elem_elem_counts.data(), 1, MPI_INT,
                &total_elem_elem_entries, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    
    std::vector<int> elems_in_elem_on_rank(total_elem_elem_entries);
    
    // Now scatter the num_elems_in_elem for each element on each rank
    std::vector<int> num_elems_in_elem_per_rank(num_elements_on_rank);
    
    if (rank == 0) {
        std::vector<int> all_num_elems_in_elem;
        std::vector<int> displs_ee(world_size);
        int displacement = 0;
        
        for(int i = 0; i < world_size; i++) {
            displs_ee[i] = displacement;
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                all_num_elems_in_elem.push_back(initial_mesh.num_elems_in_elem(elements_to_send[i][k]));
            }
            displacement += elements_to_send[i].size();
        }
        
        MPI_Scatterv(all_num_elems_in_elem.data(), elems_per_rank.data(), displs_ee.data(), MPI_INT,
                     num_elems_in_elem_per_rank.data(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     num_elems_in_elem_per_rank.data(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    
    if (rank == 0){
        // Prepare the element-element connectivity data for each rank
        std::vector<int> all_elems_in_elem;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        
        for(int i = 0; i < world_size; i++) {
            sendcounts[i] = elem_elem_counts[i];
            displs[i] = displacement;
            
            // Copy element-element connectivity for rank i
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                for(int l = 0; l < initial_mesh.num_elems_in_elem(elements_to_send[i][k]); l++) {
                    all_elems_in_elem.push_back(initial_mesh.elems_in_elem(elements_to_send[i][k], l));
                }
            }
            displacement += elem_elem_counts[i];
        }

        // Send the element-element connectivity data to each rank using MPI_Scatterv
        MPI_Scatterv(all_elems_in_elem.data(), sendcounts.data(), displs.data(), MPI_INT,
                     elems_in_elem_on_rank.data(), total_elem_elem_entries, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     elems_in_elem_on_rank.data(), total_elem_elem_entries, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0 && print_info) {
        std::cout << "Rank " << rank << " received element-element connectivity (" 
                << num_elements_on_rank << " elements, " << elems_in_elem_on_rank.size() << " entries):" << std::endl;
        
        int offset = 0;
        for (int elem = 0; elem < num_elements_on_rank; elem++) {
            std::cout << "  Element " << elem << " has neighbors: ";
            int num_neighbors = num_elems_in_elem_per_rank[elem];
            for (int j = 0; j < num_neighbors; j++) {
                std::cout << elems_in_elem_on_rank[offset + j] << " ";
            }
            offset += num_neighbors;
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1 && print_info) {
        std::cout << "Rank " << rank << " received element-element connectivity (" 
                << num_elements_on_rank << " elements, " << elems_in_elem_on_rank.size() << " entries):" << std::endl;
        
        int offset = 0;
        for (int elem = 0; elem < num_elements_on_rank; elem++) {
            std::cout << "  Element " << elem << " has neighbors: ";
            int num_neighbors = num_elems_in_elem_per_rank[elem];
            for (int j = 0; j < num_neighbors; j++) {
                std::cout << elems_in_elem_on_rank[offset + j] << " ";
            }
            offset += num_neighbors;
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

// ****************************************************************************************** 
//     Initialize the mesh data structures for each rank
// ****************************************************************************************** 
    mesh.initialize_nodes(num_nodes_on_rank);
    mesh.initialize_elems(num_elements_on_rank, 3);

    mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(num_nodes_on_rank, "mesh.local_to_global_node_mapping");
    mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(num_elements_on_rank, "mesh.local_to_global_elem_mapping");

    for(int i = 0; i < num_nodes_on_rank; i++) {
        mesh.local_to_global_node_mapping.host(i) = nodes_on_rank[i];
    }   

    for(int i = 0; i < num_elements_on_rank; i++) {
        mesh.local_to_global_elem_mapping.host(i) = elements_on_rank[i];
    }

    mesh.local_to_global_node_mapping.update_device();
    mesh.local_to_global_elem_mapping.update_device();

    // rebuild the local element-node connectivity using the local node ids
    for(int i = 0; i < num_elements_on_rank; i++) {
        for(int j = 0; j < num_nodes_per_elem; j++) {

            int node_gid = nodes_in_elem_on_rank[i * num_nodes_per_elem + j];

            int node_lid = -1;

            // Search through the local to global mapp to find the equivalent local index
            for(int k = 0; k < num_nodes_on_rank; k++){

                if(node_gid == mesh.local_to_global_node_mapping.host(k)) {
                    node_lid = k;
                    break;
                }
            }

            mesh.nodes_in_elem.host(i, j) = node_lid;
        }
    }


    mesh.nodes_in_elem.update_device();

// ****************************************************************************************** 
//     Build the connectivity for the local mesh
// ****************************************************************************************** 

    mesh.build_connectivity();
    MPI_Barrier(MPI_COMM_WORLD);


    if(rank == 0 && print_info) {
        print_rank_mesh_info(mesh, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 1 && print_info) {
        print_rank_mesh_info(mesh, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (print_vtk) {
        write_vtk(mesh, node, rank);
    }


// ****************************************************************************************** 
//     Repartition the mesh using pt-scotch
// ****************************************************************************************** 



    // --- Simple compact CSR build using global neighbor GIDs (recommended) ---
    SCOTCH_Dgraph dgraph;
    if (SCOTCH_dgraphInit(&dgraph, MPI_COMM_WORLD) != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphInit failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const SCOTCH_Num baseval = 0;                       // 0-based
    const SCOTCH_Num vertlocnbr = static_cast<SCOTCH_Num>(mesh.num_elems);
    const SCOTCH_Num vertlocmax = vertlocnbr;           // no holes

    // Build compact CSR: vertloctab (size vertlocnbr+1) and edgeloctab (neighbors as GLOBAL elem GIDs)
    std::vector<SCOTCH_Num> vertloctab(vertlocnbr + 1);
    std::vector<SCOTCH_Num> edgeloctab;
    edgeloctab.reserve(vertlocnbr * 6); // heuristic reserve

    // Build the graph from elems_in_elem_on_rank which contains global neighbor IDs
    // First, create a map from element GID to its position in elems_in_elem_on_rank
    std::map<int, size_t> elem_gid_to_offset;
    size_t current_offset = 0;
    for (size_t k = 0; k < num_elements_on_rank; k++) {
        elem_gid_to_offset[elements_on_rank[k]] = current_offset;
        current_offset += num_elems_in_elem_per_rank[k];
    }
    
    SCOTCH_Num offset = 0;
    for (size_t lid = 0; lid < mesh.num_elems; ++lid) {
        vertloctab[lid] = offset;

        // Get local element's global ID
        int elem_gid = mesh.local_to_global_elem_mapping.host(lid);
        
        // Get the offset in elems_in_elem_on_rank for this element
        size_t elems_in_elem_offset = elem_gid_to_offset[elem_gid];
        
        // Get neighbor count - need to find the right index in elements_on_rank
        size_t idx = 0;
        for (size_t k = 0; k < num_elements_on_rank; k++) {
            if (elements_on_rank[k] == elem_gid) {
                idx = k;
                break;
            }
        }
        size_t num_nbrs = num_elems_in_elem_per_rank[idx];
        
        for (size_t j = 0; j < num_nbrs; ++j) {
            // Get global neighbor ID from elems_in_elem_on_rank
            size_t neighbor_gid = elems_in_elem_on_rank[elems_in_elem_offset + j];
            edgeloctab.push_back(static_cast<SCOTCH_Num>(neighbor_gid));
            ++offset;
        }
    }
    vertloctab[vertlocnbr] = offset;
    const SCOTCH_Num edgelocnbr = offset;
    const SCOTCH_Num edgelocsiz = edgelocnbr;

    // Debug: print graph structure
    if (print_info) {
        std::cout << "Rank " << rank << ": vertlocnbr=" << vertlocnbr << ", edgelocnbr=" << edgelocnbr << std::endl;
        std::cout << "vertloctab: ";
        for (size_t i = 0; i <= vertlocnbr; i++) {
            std::cout << vertloctab[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "edgeloctab (first 20): ";
        for (size_t i = 0; i < std::min((size_t)20, edgeloctab.size()); i++) {
            std::cout << edgeloctab[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // NOTE: Using compact CSR => pass vendloctab = nullptr, vlblloctab = nullptr.
    //       edgeloctab contains GLOBAL neighbor IDs; SCOTCH will discover remote vertices itself.
    int rc = SCOTCH_dgraphBuild(&dgraph,
                                baseval,
                                vertlocnbr,
                                vertlocmax,
                                vertloctab.data(),   // compact offsets
                                /*vendloctab*/ nullptr,
                                /*veloloctab*/ nullptr,
                                /*vlblloctab*/ nullptr,
                                edgelocnbr,
                                edgelocsiz,
                                edgeloctab.data(),
                                /*edgegsttab*/ nullptr,
                                /*edloloctab*/ nullptr);
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphBuild failed rc=" << rc << "\n";
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Print graph info after build but before check
    if (print_info) {
        SCOTCH_Num vertlocnbr_out, vertloctab_size;
        SCOTCH_dgraphSize(&dgraph, &vertlocnbr_out, nullptr, nullptr, nullptr);
        std::cout << "Rank " << rank << ": After dgraphBuild, vertlocnbr=" << vertlocnbr_out << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Sanity check
    rc = SCOTCH_dgraphCheck(&dgraph);
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphCheck failed rc=" << rc << "\n";
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Partition the mesh using pt-scotch
    // Partition into world_size parts
    // Note: Since we already have a distributed mesh, we're asking for a repartition
    SCOTCH_Arch archdat;
    SCOTCH_archInit(&archdat);
    SCOTCH_archCmplt(&archdat, static_cast<SCOTCH_Num>(world_size));
    
    SCOTCH_Strat stratdat;
    SCOTCH_stratInit(&stratdat);
    
    std::vector<SCOTCH_Num> partloctab(vertlocnbr);
    rc = SCOTCH_dgraphMap(&dgraph, &archdat, &stratdat, partloctab.data());
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphMap failed rc=" << rc << "\n";
        SCOTCH_stratExit(&stratdat);
        SCOTCH_archExit(&archdat);
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    
    SCOTCH_stratExit(&stratdat);
    SCOTCH_archExit(&archdat);

    // Print partition assignment (optional)
    for (size_t lid = 0; lid < mesh.num_elems; ++lid) {
        size_t gid = mesh.local_to_global_elem_mapping.host(lid);
        std::cout << "[rank " << rank << "] elem_local=" << lid << " gid=" << gid
                << " -> part=" << partloctab[lid] << "\n";
    }
    


    MPI_Barrier(MPI_COMM_WORLD);









    } // end MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

     // Stop timer and get execution time
    double time_ms = timer.stop();
     
    printf("Execution time: %.2f ms\n", time_ms);

    return 0;
}