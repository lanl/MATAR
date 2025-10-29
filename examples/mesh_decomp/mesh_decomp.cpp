#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <mpi.h>
#include <set>
#include <map>


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
        std::cout<<"Element "<<i<<" has nodes global id: "<<mesh.local_to_global_elem_mapping.host(i)<<" and local nodes:";
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

    bool print_info = false;
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
    // This is the initial partitioned mesh
    Mesh_t mesh;
    GaussPoint_t GaussPoints;
    node_t node;

    // Mesh partitioned by pt-scotch
    Mesh_t final_mesh; 
    node_t final_node;

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
    double t_init_mesh_start = MPI_Wtime();

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

    double t_init_mesh_end = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Initial mesh generation + broadcast took " << (t_init_mesh_end - t_init_mesh_start) << " seconds." << std::endl;
    }
    
// ********************************************************  
//        Scatter the number of elements to each rank
// ******************************************************** 
    // All ranks participate in the scatter operation
    // MPI_Scatter signature:
    // MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //             void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //             int root, MPI_Comm comm)
    double t_scatter_start = MPI_Wtime();
    MPI_Scatter(elems_per_rank.data(), 1, MPI_INT, 
                &num_elements_on_rank, 1, MPI_INT, 
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Resize the elements_on_rank vector to hold the received data
    elements_on_rank.resize(num_elements_on_rank);
    

    MPI_Barrier(MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();
    if(rank == 0) {
        std::cout<<" Finished scattering the number of elements to each rank"<<std::endl;
        std::cout << " Scatter operation took " << (t_scatter_end - t_scatter_start) << " seconds." << std::endl;
    }

// ********************************************************  
//     Scatter the actual element global ids to each rank
// ******************************************************** 
    double t_scatter_gids_start = MPI_Wtime();

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
    double t_scatter_gids_end = MPI_Wtime();
    if(rank == 0) {
        std::cout<<" Finished scattering the actual element global ids to each rank"<<std::endl;
        std::cout << " Scattering the actual element global ids to each rank took " 
                  << (t_scatter_gids_end - t_scatter_gids_start) << " seconds." << std::endl;
    }
    

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

    // Timer: Start measuring time for node scattering
    double t_scatter_nodes_start = MPI_Wtime();

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

        for (int i = 0; i < world_size; i++) {
            nodes_per_rank[i] = nodes_to_send[i].size();
        }

        if (print_info) {

            
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

    // resize the nodes_on_rank vector to hold the received data
    nodes_on_rank.resize(num_nodes_on_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (print_info) {
        std::cout << "Rank " << rank << " received " << num_nodes_on_rank << " nodes" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Timer: End measuring time for node scattering
    double t_scatter_nodes_end = MPI_Wtime();

    if(rank == 0) {
        std::cout<<" Finished scattering the number of nodes to each rank"<<std::endl;
        std::cout << " Scattering the number of nodes to each rank took " 
                  << (t_scatter_nodes_end - t_scatter_nodes_start) << " seconds." << std::endl;
    }


// ****************************************************************************************** 
//     Scatter the actual node global ids to each rank
// ****************************************************************************************** 
    // Timer: Start measuring time for scattering node global ids
    double t_scatter_nodeids_start = MPI_Wtime();

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

    // Timer: End measuring time for scattering node global ids
    double t_scatter_nodeids_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
        std::cout<<" Finished scattering the actual node global ids to each rank"<<std::endl;
        std::cout << " Scattering node global ids took "
                  << (t_scatter_nodeids_end - t_scatter_nodeids_start) << " seconds." << std::endl;
    }

// ****************************************************************************************** 
//     Scatter the node positions to each rank
// ****************************************************************************************** 
    // Create a flat 1D vector for node positions (3 coordinates per node)
    std::vector<double> node_pos_on_rank_flat(num_nodes_on_rank * 3);

    // Timer for scattering node positions
    double t_scatter_nodepos_start = MPI_Wtime();

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

    MPI_Barrier(MPI_COMM_WORLD);

    double t_scatter_nodepos_end = MPI_Wtime();
    if(rank == 0) {
        std::cout<<" Finished scattering the node positions to each rank"<<std::endl;
        std::cout << " Scattering node positions took "
                  << (t_scatter_nodepos_end - t_scatter_nodepos_start) << " seconds." << std::endl;
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
    
    double t_scatter_elemnode_start = MPI_Wtime();

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

    double t_scatter_elemnode_end = MPI_Wtime();
    if(rank == 0) {
        std::cout << " Finished scattering the element-node connectivity data from the initial mesh to each rank" << std::endl;
        std::cout << " Scattering element-node connectivity took "
                  << (t_scatter_elemnode_end - t_scatter_elemnode_start) << " seconds." << std::endl;
    }

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
    if(rank == 0) std::cout<<" Finished scattering the element-node connectivity data from the initial mesh to each rank"<<std::endl;


// ****************************************************************************************** 
//     Send the element-element connectivity data from the initial mesh to each rank
// ****************************************************************************************** 

    // First, rank 0 computes how many connectivity entries each rank will receive
    // and scatters that information
    std::vector<int> elem_elem_counts(world_size);
    int total_elem_elem_entries = 0;
    
    
    double t_scatter_elem_elem_start = MPI_Wtime();

    if (rank == 0){
        // Calculate total number of connectivity entries for each rank
        for(int i = 0; i < world_size; i++) {
            elem_elem_counts[i] = 0;
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                elem_elem_counts[i] += initial_mesh.num_elems_in_elem(elements_to_send[i][k]);
            }

            if(print_info) std::cout << "Rank " << i << " will receive " << elem_elem_counts[i] << " element-element connectivity entries" << std::endl;
        }

        // Print element-element connectivity entries for each rank in the initial mesh
        if(print_info) {
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
    }
    

    // Define total_elem_elem_entries to be the sum of the elem_elem_counts
    // Scatter the counts to each rank
    MPI_Scatter(elem_elem_counts.data(), 1, MPI_INT,
                &total_elem_elem_entries, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_scatter_elem_elem_end = MPI_Wtime();
    if(rank == 0) {
        std::cout<<" Finished scattering the number of element-element connectivity entries to each rank"<<std::endl;
        std::cout<<" Scattering element-element connectivity counts took "
                 << (t_scatter_elem_elem_end - t_scatter_elem_elem_start) << " seconds." << std::endl;
    }

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
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished scattering the actual element-element connectivity counts per element to each rank"<<std::endl;

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

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished receiving the actual element-element connectivity entries to each rank"<<std::endl;

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

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Starting reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;

    // Timer for reverse mapping of element-node connectivity
    double t_reverse_map_start = MPI_Wtime();

    // rebuild the local element-node connectivity using the local node ids
    for(int i = 0; i < num_elements_on_rank; i++) {
        for(int j = 0; j < num_nodes_per_elem; j++) {

            int node_gid = nodes_in_elem_on_rank[i * num_nodes_per_elem + j];

            int node_lid = -1;

            // Use binary search to find the local node index for node_gid
            int left = 0, right = num_nodes_on_rank - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                size_t mid_gid = mesh.local_to_global_node_mapping.host(mid);
                if (node_gid == mid_gid) {
                    node_lid = mid;
                    break;
                } else if (node_gid < mid_gid) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            mesh.nodes_in_elem.host(i, j) = node_lid;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t_reverse_map_end = MPI_Wtime();
    if(rank == 0) {
        std::cout<<" Finished reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;
        std::cout<<" Reverse mapping time: " << (t_reverse_map_end - t_reverse_map_start) << " seconds." << std::endl;
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
//     Compute a repartition of the mesh using pt-scotch
// ****************************************************************************************** 



    /**********************************************************************************
     * Build PT-Scotch distributed graph representation of the mesh for repartitioning *
     **********************************************************************************
     *
     * This section constructs the distributed graph (SCOTCH_Dgraph) needed by PT-Scotch
     * for mesh repartitioning. In this graph, each mesh element is a vertex, and edges
     * correspond to mesh-neighbor relationships (i.e., elements that share a face or are
     * otherwise neighbors per your mesh definition).
     *
     * We use the compact CSR (Compressed Sparse Row) representation, passing only the
     * essential information required by PT-Scotch.
     * 
     * Variables and structures used:
     *   - SCOTCH_Dgraph dgraph:
     *       The distributed graph instance managed by PT-Scotch. Each MPI rank creates
     *       and fills in its portion of the global graph.
     * 
     *   - const SCOTCH_Num baseval:
     *       The base value for vertex and edge numbering. Set to 0 for C-style zero-based
     *       arrays. Always use 0 unless you are using Fortran style 1-based arrays.
     * 
     *   - const SCOTCH_Num vertlocnbr:
     *       The *number of local vertices* (mesh elements) defined on this MPI rank.
     *       In our mesh, this is mesh.num_elems. PT-Scotch expects each rank to specify
     *       its own local vertex count.
     *
     *   - const SCOTCH_Num vertlocmax:
     *       The *maximum number of local vertices* that could be stored (capacity). We
     *       allocate with no unused holes, so vertlocmax = vertlocnbr.
     *
     *   - std::vector<SCOTCH_Num> vertloctab:
     *       CSR array [size vertlocnbr+1]: for each local vertex i, vertloctab[i]
     *       gives the index in edgeloctab where the neighbor list of vertex i begins.
     *       PT-Scotch expects this array to be of size vertlocnbr+1, where the difference
     *       vertloctab[i+1] - vertloctab[i] gives the number of edges for vertex i.
     *
     *   - std::vector<SCOTCH_Num> edgeloctab:
     *       CSR array [variable size]: a flattened list of *neighboring element global IDs*,
     *       in no particular order. For vertex i, its neighbors are located at
     *       edgeloctab[vertloctab[i]...vertloctab[i+1]-1].
     *       In this compact CSR, these are global IDs (GIDs), enabling PT-Scotch to
     *       recognize edges both within and across ranks.
     *
     *   - std::map<int, size_t> elem_gid_to_offset:
     *       Helper map: For a given element global ID, gives the starting offset in 
     *       the flattened neighbor array (elems_in_elem_on_rank) where this element's
     *       list of neighbor GIDs begins. This allows efficient neighbor list lookup.
     *
     *   - (other arrays used, from mesh setup and communication phase)
     *       - elements_on_rank: vector of global element IDs owned by this rank.
     *       - num_elements_on_rank: number of owned elements.
     *       - num_elems_in_elem_per_rank: array, for each owned element, how many
     *         neighbors it has.
     *       - elems_in_elem_on_rank: flattened array of global neighbor IDs for all local elements.
     *
     **********************************************************************************/

    // --- Step 1: Initialize the PT-Scotch distributed graph object on this MPI rank ---
    SCOTCH_Dgraph dgraph;
    if (SCOTCH_dgraphInit(&dgraph, MPI_COMM_WORLD) != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphInit failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Set base value for numbering (0 for C-style arrays)
    const SCOTCH_Num baseval = 0;

    // vertlocnbr: Number of elements (vertices) that are local to this MPI rank
    const SCOTCH_Num vertlocnbr = static_cast<SCOTCH_Num>(mesh.num_elems);

    // vertlocmax: Maximum possible local vertices (no holes, so identical to vertlocnbr)
    const SCOTCH_Num vertlocmax = vertlocnbr;

    // --- Step 2: Build compact CSR arrays for PT-Scotch (vertloctab, edgeloctab) ---
    // vertloctab: for each local mesh element [vertex], gives index in edgeloctab where its neighbor list begins
    std::vector<SCOTCH_Num> vertloctab(vertlocnbr + 1);

    // edgeloctab: flat array of neighbor global IDs for all local elements, built in order
    std::vector<SCOTCH_Num> edgeloctab;
    edgeloctab.reserve(vertlocnbr * 6); // heuristic: assume typical mesh degree is ~6, for performance

    // Construct a map from element GID to its offset into elems_in_elem_on_rank (the array of neighbor GIDs)
    // This allows, for a given element GID, quick lookup of where its neighbor list starts in the flat array.
    std::map<int, size_t> elem_gid_to_offset;
    size_t current_offset = 0;
    for (size_t k = 0; k < num_elements_on_rank; k++) {
        elem_gid_to_offset[elements_on_rank[k]] = current_offset;
        current_offset += num_elems_in_elem_per_rank[k];
    }

    // --- Step 3: Fill in the CSR arrays, looping over each locally-owned element ---
    SCOTCH_Num offset = 0; // running count of edges encountered

    for (size_t lid = 0; lid < mesh.num_elems; ++lid) {

        // Record current edge offset for vertex lid in vertloctab
        vertloctab[lid] = offset;

        // Obtain this local element's global ID (from mapping)
        int elem_gid = mesh.local_to_global_elem_mapping.host(lid);

        // Find offset in the flattened neighbor array for this element's neighbor list
        size_t elems_in_elem_offset = elem_gid_to_offset[elem_gid];

        // For this element, find the count of its neighbors
        // This requires finding its index in the elements_on_rank array
        size_t idx = 0;
        for (size_t k = 0; k < num_elements_on_rank; k++) {
            if (elements_on_rank[k] == elem_gid) {
                idx = k;
                break;
            }
        }
        size_t num_nbrs = num_elems_in_elem_per_rank[idx];

        // Append each neighbor (by its GLOBAL elem GID) to edgeloctab
        for (size_t j = 0; j < num_nbrs; ++j) {
            size_t neighbor_gid = elems_in_elem_on_rank[elems_in_elem_offset + j]; // This is a global element ID!
            edgeloctab.push_back(static_cast<SCOTCH_Num>(neighbor_gid));
            ++offset; // Increment running edge count
        }
    }

    // vertloctab[vertlocnbr] stores total number of edges written, finalizes the CSR structure
    vertloctab[vertlocnbr] = offset;

    // edgelocnbr/edgelocsiz: Number of edge endpoints defined locally
    // (PT-Scotch's distributed graphs allow edges to be replicated or owned by either endpoint)
    const SCOTCH_Num edgelocnbr = offset; // total number of edge endpoints (sum of all local neighbor degrees)
    const SCOTCH_Num edgelocsiz = edgelocnbr; // allocated size matches number of endpoints

    // Optionally print graph structure for debugging/validation
    if (print_info) {
        std::cout << "Rank " << rank << ": vertlocnbr = # of local elements(vertices) = " << vertlocnbr
                  << ", edgelocnbr = # of local edge endpoints = " << edgelocnbr << std::endl;
        std::cout << "vertloctab (CSR row offsets): ";
        for (size_t i = 0; i <= vertlocnbr; i++) {
            std::cout << vertloctab[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "edgeloctab (first 20 neighbor GIDs): ";
        for (size_t i = 0; i < std::min((size_t)20, edgeloctab.size()); i++) {
            std::cout << edgeloctab[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /**************************************************************************
     * Step 4: Build the distributed graph using PT-Scotch's SCOTCH_dgraphBuild
     *
     *   - PT-Scotch will use our CSR arrays. Since we use compact representation,
     *     most optional arrays ("veloloctab", "vlblloctab", "edgegsttab", "edloloctab")
     *     can be passed as nullptr.
     *   - edgeloctab contains *GLOBAL element GIDs* of neighbors. PT-Scotch uses this
     *     to discover connections across processor boundaries, so you do not have to
     *     encode ownership or partition information yourself.
     **************************************************************************/
    int rc = SCOTCH_dgraphBuild(
                &dgraph,
                baseval,                // start index (0)
                vertlocnbr,             // local vertex count (local elements)
                vertlocmax,             // local vertex max (no holes)
                vertloctab.data(),      // row offsets in edgeloctab
                /*vendloctab*/ nullptr, // end of row offsets (compact CSR => nullptr)
                /*veloloctab*/ nullptr, // vertex weights, not used
                /*vlblloctab*/ nullptr, // vertex global labels (we use GIDs in edgeloctab)
                edgelocnbr,             // local edge endpoints count
                edgelocsiz,             // size of edge array
                edgeloctab.data(),      // global neighbor IDs for each local node
                /*edgegsttab*/ nullptr, // ghost edge array, not used
                /*edloloctab*/ nullptr  // edge weights, not used
    );
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphBuild failed rc=" << rc << "\n";
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Optionally, print rank summary after graph build for further validation
    if (print_info) {
        SCOTCH_Num vertlocnbr_out;
        SCOTCH_dgraphSize(&dgraph, &vertlocnbr_out, nullptr, nullptr, nullptr);
        std::cout << "Rank " << rank << ": After dgraphBuild, vertlocnbr = " << vertlocnbr_out << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished building the distributed graph using PT-Scotch"<<std::endl;

    /********************************************************
     * Step 5: Validate the graph using SCOTCH_dgraphCheck
     ********************************************************/
    rc = SCOTCH_dgraphCheck(&dgraph);
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphCheck failed rc=" << rc << "\n";
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    /**************************************************************
     * Step 6: Partition (repartition) the mesh using PT-Scotch
     * - Each vertex (mesh element) will be assigned a part (mesh chunk).
     * - Arch is initialized for a complete graph of world_size parts (one per rank).
     * - Loki
     **************************************************************/
    SCOTCH_Arch archdat;        // PT-Scotch architecture structure: describes desired partition topology
    SCOTCH_archInit(&archdat);
    SCOTCH_archCmplt(&archdat, static_cast<SCOTCH_Num>(world_size)); // Partition into world_size complete nodes

    SCOTCH_Strat stratdat;      // PT-Scotch strategy object: holds partitioning options/settings
    SCOTCH_stratInit(&stratdat);
    SCOTCH_stratDgraphMapBuild(&stratdat, SCOTCH_STRATQUALITY, world_size, 0, 0.01); // zero is recursion count, 0=automatic

    // partloctab: output array mapping each local element (vertex) to a *target partition number*
    // After partitioning, partloctab[i] gives the part-assignment (in [0,world_size-1]) for local element i.
    std::vector<SCOTCH_Num> partloctab(vertlocnbr);
    rc = SCOTCH_dgraphMap(&dgraph, &archdat, &stratdat, partloctab.data());
    if (rc != 0) {
        std::cerr << "[rank " << rank << "] SCOTCH_dgraphMap failed rc=" << rc << "\n";
        SCOTCH_stratExit(&stratdat);
        SCOTCH_archExit(&archdat);
        SCOTCH_dgraphFree(&dgraph);
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Clean up PT-Scotch strategy and architecture objects
    SCOTCH_stratExit(&stratdat);
    SCOTCH_archExit(&archdat);
    
    // Free the graph now that we have the partition assignments
    SCOTCH_dgraphFree(&dgraph);

    /***************************************************************************
     * Step 7 (Optional): Print out the partitioning assignment per element
     * - Each local element's local index lid and global ID (gid) are listed with the
     *   part to which PT-Scotch has assigned them.
     ***************************************************************************/
    print_info = true;
    for(int rank_id = 0; rank_id < world_size; rank_id++) {
        if(rank_id == rank && print_info) {
            for (size_t lid = 0; lid < mesh.num_elems; ++lid) {
                size_t gid = mesh.local_to_global_elem_mapping.host(lid);
                std::cout << "[rank " << rank_id << "] elem_local=" << lid << " gid=" << gid
                        << " -> part=" << partloctab[lid] << "\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    print_info = false;



// ****************************************************************************************** 
//     Build the final mesh from the repartition
// ****************************************************************************************** 



    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\n=== Starting Mesh Redistribution Phase ===\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // -------------- Phase 1: Determine elements to send to each rank --------------
    std::vector<std::vector<int>> elems_to_send(world_size);
    for (int lid = 0; lid < mesh.num_elems; ++lid) {
        int dest = static_cast<int>(partloctab[lid]);
        int elem_gid = static_cast<int>(mesh.local_to_global_elem_mapping.host(lid));
        elems_to_send[dest].push_back(elem_gid);
    }

    // -------------- Phase 2: Exchange element GIDs --------------
    std::vector<int> sendcounts(world_size), recvcounts(world_size);
    for (int r = 0; r < world_size; ++r)
        sendcounts[r] = static_cast<int>(elems_to_send[r].size());

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> sdispls(world_size), rdispls(world_size);
    int send_total = 0, recv_total = 0;
    for (int r = 0; r < world_size; ++r) {
        sdispls[r] = send_total;
        rdispls[r] = recv_total;
        send_total += sendcounts[r];
        recv_total += recvcounts[r];
    }


    // Flatten send buffer
    std::vector<int> sendbuf;
    sendbuf.reserve(send_total);
    for (int r = 0; r < world_size; ++r)
        sendbuf.insert(sendbuf.end(), elems_to_send[r].begin(), elems_to_send[r].end());

    // Receive new local element GIDs
    std::vector<int> recvbuf(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging element GIDs"<<std::endl;

    // New elements owned by this rank
    std::vector<int> new_elem_gids = recvbuf;
    int num_new_elems = static_cast<int>(new_elem_gids.size());
    
    
    if (print_info) {
        std::cout << "[rank " << rank << "] new elems: " << num_new_elems << std::endl;
    }

    // -------------- Phase 3: Send element–node connectivity --------------
    int nodes_per_elem = mesh.num_nodes_in_elem;

    // Flatten element-node connectivity by global node IDs
    std::vector<int> conn_sendbuf;
    for (int r = 0; r < world_size; ++r) {
        for (int gid : elems_to_send[r]) {
            // find local element lid from gid
            int lid = -1;
            for (int i = 0; i < mesh.num_elems; ++i)
                if (mesh.local_to_global_elem_mapping.host(i) == gid) { lid = i; break; }

            for (int j = 0; j < nodes_per_elem; ++j) {
                int node_lid = mesh.nodes_in_elem.host(lid, j);
                int node_gid = mesh.local_to_global_node_mapping.host(node_lid);
                conn_sendbuf.push_back(node_gid);
            }
        }
    }

    // element-node connectivity counts (ints per dest rank)
    std::vector<int> conn_sendcounts(world_size), conn_recvcounts(world_size);
    for (int r = 0; r < world_size; ++r)
        conn_sendcounts[r] = sendcounts[r] * nodes_per_elem;

    MPI_Alltoall(conn_sendcounts.data(), 1, MPI_INT, conn_recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging element–node connectivity counts"<<std::endl;

    std::vector<int> conn_sdispls(world_size), conn_rdispls(world_size);
    int conn_send_total = 0, conn_recv_total = 0;
    for (int r = 0; r < world_size; ++r) {
        conn_sdispls[r] = conn_send_total;
        conn_rdispls[r] = conn_recv_total;
        conn_send_total += conn_sendcounts[r];
        conn_recv_total += conn_recvcounts[r];
    }

    std::vector<int> conn_recvbuf(conn_recv_total);
    MPI_Alltoallv(conn_sendbuf.data(), conn_sendcounts.data(), conn_sdispls.data(), MPI_INT,
                conn_recvbuf.data(), conn_recvcounts.data(), conn_rdispls.data(), MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging element–node connectivity"<<std::endl;

    // -------------- Phase 4: Build new node list (unique GIDs) --------------
    std::set<int> node_gid_set(conn_recvbuf.begin(), conn_recvbuf.end());
    std::vector<int> new_node_gids(node_gid_set.begin(), node_gid_set.end());
    int num_new_nodes = static_cast<int>(new_node_gids.size());

    // Build map gid→lid
    std::unordered_map<int,int> node_gid_to_lid;
    for (int i = 0; i < num_new_nodes; ++i)
        node_gid_to_lid[new_node_gids[i]] = i;

    if (print_info)
        std::cout << "[rank " << rank << "] owns " << num_new_nodes << " unique nodes\n";


    // -------------- Phase 5: Request node coordinates --------------
    std::vector<double> node_coords_sendbuf;
    for (int r = 0; r < world_size; ++r) {
        for (int gid : elems_to_send[r]) {
            int lid = -1;
            for (int i = 0; i < mesh.num_elems; ++i)
                if (mesh.local_to_global_elem_mapping.host(i) == gid) { lid = i; break; }

            for (int j = 0; j < nodes_per_elem; ++j) {
                int node_lid = mesh.nodes_in_elem.host(lid, j);
                int node_gid = mesh.local_to_global_node_mapping.host(node_lid);

                node_coords_sendbuf.push_back(node.coords.host(node_lid, 0));
                node_coords_sendbuf.push_back(node.coords.host(node_lid, 1));
                node_coords_sendbuf.push_back(node.coords.host(node_lid, 2));
            }
        }
    }

    // Each node is 3 doubles; same sendcounts scaling applies
    std::vector<int> coord_sendcounts(world_size), coord_recvcounts(world_size);
    for (int r = 0; r < world_size; ++r)
        coord_sendcounts[r] = sendcounts[r] * nodes_per_elem * 3;

    MPI_Alltoall(coord_sendcounts.data(), 1, MPI_INT, coord_recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging node coordinates counts"<<std::endl;

    std::vector<int> coord_sdispls(world_size), coord_rdispls(world_size);
    int coord_send_total = 0, coord_recv_total = 0;
    for (int r = 0; r < world_size; ++r) {
        coord_sdispls[r] = coord_send_total;
        coord_rdispls[r] = coord_recv_total;
        coord_send_total += coord_sendcounts[r];
        coord_recv_total += coord_recvcounts[r];
    }

    std::vector<double> coord_recvbuf(coord_recv_total);
    MPI_Alltoallv(node_coords_sendbuf.data(), coord_sendcounts.data(), coord_sdispls.data(), MPI_DOUBLE,
                coord_recvbuf.data(), coord_recvcounts.data(), coord_rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging node coordinates"<<std::endl;

    // -------------- Phase 6: Build the final_mesh --------------
    final_mesh.initialize_nodes(num_new_nodes);
    final_mesh.initialize_elems(num_new_elems, mesh.num_dims);
    final_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(num_new_nodes);
    final_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(num_new_elems);

    // Fill global mappings
    for (int i = 0; i < num_new_nodes; ++i)
        final_mesh.local_to_global_node_mapping.host(i) = new_node_gids[i];
    for (int i = 0; i < num_new_elems; ++i)
        final_mesh.local_to_global_elem_mapping.host(i) = new_elem_gids[i];

    final_mesh.local_to_global_node_mapping.update_device();
    final_mesh.local_to_global_elem_mapping.update_device();


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Starting reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;
    // rebuild the local element-node connectivity using the local node ids
    for(int i = 0; i < num_new_elems; i++) {
        for(int j = 0; j < nodes_per_elem; j++) {

            int node_gid = conn_recvbuf[i * nodes_per_elem + j];

            int node_lid = -1;

            // Binary search through local_to_global_node_mapping to find the equivalent local index
            int left = 0, right = num_new_nodes - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                size_t mid_gid = final_mesh.local_to_global_node_mapping.host(mid);
                if (node_gid == mid_gid) {
                    node_lid = mid;
                    break;
                } else if (node_gid < mid_gid) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            final_mesh.nodes_in_elem.host(i, j) = node_lid;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;

    final_mesh.nodes_in_elem.update_device();

    // Fill node coordinates
    // coord_recvbuf contains coords in element-node order, but we need them in node order
    // Build a map from node GID to coordinates
    std::map<int, std::array<double, 3>> node_gid_to_coords;
    int coord_idx = 0;
    for (int e = 0; e < num_new_elems; ++e) {
        for (int j = 0; j < nodes_per_elem; ++j) {
            int node_gid = conn_recvbuf[e * nodes_per_elem + j];
            if (node_gid_to_coords.find(node_gid) == node_gid_to_coords.end()) {
                node_gid_to_coords[node_gid] = {
                    coord_recvbuf[coord_idx*3 + 0],
                    coord_recvbuf[coord_idx*3 + 1],
                    coord_recvbuf[coord_idx*3 + 2]
                };
            }
            coord_idx++;
        }
    }
    
    // Now fill coordinates in node order
    final_node.initialize(num_new_nodes, 3, {node_state::coords});
    for (int i = 0; i < num_new_nodes; ++i) {
        int node_gid = new_node_gids[i];
        auto it = node_gid_to_coords.find(node_gid);
        if (it != node_gid_to_coords.end()) {
            final_node.coords.host(i, 0) = it->second[0];
            final_node.coords.host(i, 1) = it->second[1];
            final_node.coords.host(i, 2) = it->second[2];
        }
    }
    final_node.coords.update_device();

    // Connectivity rebuild
    final_mesh.build_connectivity();
    MPI_Barrier(MPI_COMM_WORLD);



// ****************************************************************************************** 
//     Build the ghost elements
// ****************************************************************************************** 

    double t_ghost_start = MPI_Wtime();
    
    // Update host arrays for ghost detection
    final_mesh.local_to_global_elem_mapping.update_host();
    final_mesh.local_to_global_node_mapping.update_host();
    final_mesh.nodes_in_elem.update_host();
    Kokkos::fence();
    
    // Build a set of locally-owned element global IDs for fast lookup
    std::set<size_t> local_elem_gids;
    for (int i = 0; i < num_new_elems; ++i) {
        local_elem_gids.insert(final_mesh.local_to_global_elem_mapping.host(i));
    }
    
    // Exchange element GIDs with all ranks to know who owns what
    // Collect all locally-owned element global IDs to send to other ranks
    std::vector<size_t> local_elem_gids_vec(local_elem_gids.begin(), local_elem_gids.end());
    
    // First, gather the number of elements each rank owns
    std::vector<int> elem_counts(world_size);
    int local_elem_count = static_cast<int>(local_elem_gids_vec.size());
    
    MPI_Allgather(&local_elem_count, 1, MPI_INT, elem_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Compute displacements
    std::vector<int> elem_displs(world_size);
    int total_elems = 0;
    for (int r = 0; r < world_size; ++r) {
        elem_displs[r] = total_elems;
        total_elems += elem_counts[r];
    }
    
    // Gather all element GIDs from all ranks
    std::vector<size_t> all_elem_gids(total_elems);
    MPI_Allgatherv(local_elem_gids_vec.data(), local_elem_count, MPI_UNSIGNED_LONG_LONG,
                   all_elem_gids.data(), elem_counts.data(), elem_displs.data(), 
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    
    // Build a map: element GID -> owning rank
    std::map<size_t, int> elem_gid_to_rank;
    for (int r = 0; r < world_size; ++r) {
        for (int i = 0; i < elem_counts[r]; ++i) {
            size_t gid = all_elem_gids[elem_displs[r] + i];
            elem_gid_to_rank[gid] = r;
        }
    }
    
    // Strategy: Find ghost elements by checking neighbors of our boundary elements.
    // A boundary element is one that has a neighbor owned by another rank.
    // However, since build_connectivity() only includes locally-owned elements,
    // we need to use a different approach: find elements on other ranks that share
    // nodes with our locally-owned elements.
    
    // First, collect all nodes that belong to our locally-owned elements
    std::set<size_t> local_elem_nodes;
    for (int lid = 0; lid < num_new_elems; ++lid) {
        for (int j = 0; j < nodes_per_elem; ++j) {
            size_t node_lid = final_mesh.nodes_in_elem.host(lid, j);
            size_t node_gid = final_mesh.local_to_global_node_mapping.host(node_lid);
            local_elem_nodes.insert(node_gid);
        }
    }
    
    // Now collect element-to-node connectivity to send to all ranks
    // Format: for each element, list its node GIDs (each entry is a pair: elem_gid, node_gid)
    std::vector<size_t> elem_node_conn;
    int local_conn_size = 0;
    
    for (int lid = 0; lid < num_new_elems; ++lid) {
        size_t elem_gid = final_mesh.local_to_global_elem_mapping.host(lid);
        for (int j = 0; j < nodes_per_elem; ++j) {
            size_t node_lid = final_mesh.nodes_in_elem.host(lid, j);
            size_t node_gid = final_mesh.local_to_global_node_mapping.host(node_lid);
            elem_node_conn.push_back(elem_gid);
            elem_node_conn.push_back(node_gid);
        }
        local_conn_size += nodes_per_elem * 2;  // Each pair is 2 size_ts
    }
    
    // Exchange element-node connectivity with all ranks using Allgather
    // First, gather the sizes from each rank
    std::vector<int> conn_sizes(world_size);
    MPI_Allgather(&local_conn_size, 1, MPI_INT, conn_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Compute displacements
    std::vector<int> conn_displs(world_size);
    int total_conn = 0;
    for (int r = 0; r < world_size; ++r) {
        conn_displs[r] = total_conn;
        total_conn += conn_sizes[r];
    }
    
    // Gather all element-node pairs from all ranks
    std::vector<size_t> all_conn(total_conn);
    MPI_Allgatherv(elem_node_conn.data(), local_conn_size, MPI_UNSIGNED_LONG_LONG,
                   all_conn.data(), conn_sizes.data(), conn_displs.data(),
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    
    // Build a map: node GID -> set of element GIDs that contain it (from other ranks)
    std::map<size_t, std::set<size_t>> node_to_ext_elem;
    for (int r = 0; r < world_size; ++r) {
        if (r == rank) continue;  // Skip our own data
        // Process pairs from rank r: conn_sizes[r] is in units of size_ts, so num_pairs = conn_sizes[r] / 2
        int num_pairs = conn_sizes[r] / 2;
        for (int i = 0; i < num_pairs; ++i) {
            // Each pair is 2 size_ts, starting at conn_displs[r]
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // If this node is in one of our elements, then the element is a potential ghost
            if (local_elem_nodes.find(node_gid) != local_elem_nodes.end()) {
                // Check if this element is not owned by us
                if (local_elem_gids.find(elem_gid) == local_elem_gids.end()) {
                    node_to_ext_elem[node_gid].insert(elem_gid);
                }
            }
        }
    }
    
    // Collect all unique ghost element GIDs
    std::set<size_t> ghost_elem_gids;
    for (const auto& pair : node_to_ext_elem) {
        for (size_t elem_gid : pair.second) {
            ghost_elem_gids.insert(elem_gid);
        }
    }
    
    // Additional check: elements that are neighbors of our locally-owned elements
    // but are owned by other ranks (these might already be in ghost_elem_gids, but check connectivity)
    
    for (int lid = 0; lid < num_new_elems; ++lid) {
        size_t num_neighbors = final_mesh.num_elems_in_elem(lid);
        
        for (size_t nbr_idx = 0; nbr_idx < num_neighbors; ++nbr_idx) {
            size_t neighbor_lid = final_mesh.elems_in_elem(lid, nbr_idx);
            
            if (neighbor_lid < static_cast<size_t>(num_new_elems)) {
                size_t neighbor_gid = final_mesh.local_to_global_elem_mapping(neighbor_lid);
                
                // Check if neighbor is owned by this rank
                auto it = elem_gid_to_rank.find(neighbor_gid);
                if (it != elem_gid_to_rank.end() && it->second != rank) {
                    // Neighbor is owned by another rank - it's a ghost for us
                    ghost_elem_gids.insert(neighbor_gid);
                }
            }
        }
    }
    
    // Count unique ghost elements
    final_mesh.num_ghost_elems = ghost_elem_gids.size();
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_ghost_end = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << " Finished calculating ghost elements" << std::endl;
        std::cout << " Ghost element calculation took " << (t_ghost_end - t_ghost_start) << " seconds." << std::endl;
    }
    
    // Print ghost element info if requested
    print_info = true;
    for(int i = 0; i < world_size; i++) {
        if(rank == i && print_info) {
            std::cout << "[rank " << rank << "] owns " << num_new_elems 
                  << " elements and has " << final_mesh.num_ghost_elems << " ghost elements" << std::endl;
            std::cout << "[rank " << rank << "] owned element global IDs: ";
            for (int j = 0; j < num_new_elems; ++j) {
                std::cout << final_mesh.local_to_global_elem_mapping(j) << " ";
            }
            std::cout << std::endl;
            
            
            
            std::cout << "[rank " << rank << "] ghost element GIDs: ";
            for (size_t gid : ghost_elem_gids) {
                std::cout << gid << " ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    

    
    MPI_Barrier(MPI_COMM_WORLD);














    for(int i = 0; i < world_size; i++) {
        if(rank == i && print_info) {
            print_rank_mesh_info(final_mesh, i);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    write_vtk(final_mesh, final_node, rank);

    } // end MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

     // Stop timer and get execution time
    double time_ms = timer.stop();
     
    printf("Execution time: %.2f ms\n", time_ms);

    return 0;
}