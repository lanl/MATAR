#ifndef DECOMP_UTILS_H
#define DECOMP_UTILS_H

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
#include "communication_plan.h"


// Include Scotch headers
#include "scotch.h"
#include "ptscotch.h"


void naive_partition_mesh(
    Mesh_t& initial_mesh,
    node_t& initial_node,
    Mesh_t& naive_mesh,
    node_t& naive_node,
    std::vector<int>& elems_in_elem_on_rank,
    std::vector<int>& num_elems_in_elem_per_rank,
    int world_size,
    int rank)
{

    bool print_info = false;

    int num_elements_on_rank = 0;
    int num_nodes_on_rank = 0;

    int num_nodes_per_elem = 0;

    
    std::vector<int> nodes_on_rank;


    std::vector<int> elems_per_rank(world_size); // number of elements to send to each rank size(world_size)
    std::vector<int> nodes_per_rank(world_size); // number of nodes to send to each rank size(world_size)

    // create a 2D vector of elements to send to each rank
    std::vector<std::vector<int>> elements_to_send(world_size);

    // create a 2D vector of nodes to send to each rank
    std::vector<std::vector<int>> nodes_to_send(world_size);

    // Create a 2D vector to hold the nodal positions on each rank
    std::vector<std::vector<double>> node_pos_to_send(world_size);

    if (rank == 0) {

        num_nodes_per_elem = initial_mesh.num_nodes_in_elem;

        // Compute elements to send to each rank; handle remainders for non-even distribution
        std::fill(elems_per_rank.begin(), elems_per_rank.end(), initial_mesh.num_elems / world_size);
        int remainder = initial_mesh.num_elems % world_size;
        for (int i = 0; i < remainder; i++) {
            elems_per_rank[i] += 1;
        }
    }

    // Broadcasts the value of num_nodes_per_elem from the root rank (0) to all other ranks in MPI_COMM_WORLD.
    // After this call, all ranks will have the same value for num_nodes_per_elem.
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
    double t_scatter_start = MPI_Wtime();
    MPI_Scatter(elems_per_rank.data(), 1, MPI_INT, 
                &num_elements_on_rank, 1, MPI_INT, 
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);


    // Vector of element to send to each rank using a naive partitioning (0-m, m-n, n-o, etc.)
    std::vector<int> elements_on_rank(num_elements_on_rank);  
    MPI_Barrier(MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();

    // ********************************************************  
    //     Scatter the actual element global ids to each rank
    // ******************************************************** 
    double t_scatter_gids_start = MPI_Wtime();

    if (rank == 0) {

        // Populate the elements_to_send array by finding all elements in the elements_per_rank array and adding them to the elements_to_send array
    
        int elem_gid = 0;
        for (int rank = 0; rank < world_size; rank++) {

            for (int j = 0; j < elems_per_rank[rank]; j++) {
                elements_to_send[rank].push_back(elem_gid);
                elem_gid++;
            }
        }

        // Prepare data for MPI_Scatterv (scatter with variable counts)
        // Flatten the 2D elements_to_send into a 1D array
        std::vector<int> all_elements; // array of all elements to be sent to each rank
        std::vector<int> sendcounts(world_size); // array of the number of elements to send to each rank
        std::vector<int> displs(world_size); // array of the displacement for each rank in the flattened array
        
        int displacement = 0; // displacement is the starting index of the elements for the current rank in the flattened array
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = elems_per_rank[i]; // number of elements to send to each rank
            displs[i] = displacement; // displacement for each rank in the flattened array
            // Copy elements for rank i to the flattened array
            for (int j = 0; j < elems_per_rank[i]; j++) {
                all_elements.push_back(elements_to_send[i][j]); // add the elements to the flattened array
            }
            displacement += elems_per_rank[i]; // increment the displacement by the number of elements to send to the next rank
        }

        // Send the elements to each rank
        // all_elements.data(): Pointer to the flattened array of all elements to be sent to each rank
        // sendcounts.data(): Array with the number of elements to send to each rank
        // displs.data(): Array with the displacement for each rank in the flattened array
        // MPI_INT: Data type of the elements (integer)
        // elements_on_rank.data(): Pointer to the buffer where each rank will receive its elements
        // num_elements_on_rank: Number of elements that the receiving rank expects to receive
        // MPI_INT: Data type of the receive buffer (integer)
        // 0: The root rank (rank 0) that is performing the scatter
        // MPI_COMM_WORLD: The communicator
        MPI_Scatterv(all_elements.data(), sendcounts.data(), displs.data(), MPI_INT,
                    elements_on_rank.data(), num_elements_on_rank, MPI_INT,
                    0, MPI_COMM_WORLD);
    } 
    else {
        // If the rank is not the root rank, it will receive nullptr for the sendbuf, sendcounts, and displs arrays
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                    elements_on_rank.data(), num_elements_on_rank, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    // Wait for all ranks to complete the scatter operation
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

    // ****************************************************************************************** 
    //     Initialize the node state variables
    // ****************************************************************************************** 

    // initialize node state variables, for now, we just need coordinates, the rest will be initialize by the respective solvers
    std::vector<node_state> required_node_state = { node_state::coords };
    naive_node.initialize(num_nodes_on_rank, 3, required_node_state);

    for(int i = 0; i < num_nodes_on_rank; i++) {
        naive_node.coords.host(i, 0) = node_pos_on_rank_flat[i*3];
        naive_node.coords.host(i, 1) = node_pos_on_rank_flat[i*3+1];
        naive_node.coords.host(i, 2) = node_pos_on_rank_flat[i*3+2];
    }

    naive_node.coords.update_device();


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
        }
    }
    
    // Define total_elem_elem_entries to be the sum of the elem_elem_counts
    // Scatter the counts to each rank
    MPI_Scatter(elem_elem_counts.data(), 1, MPI_INT,
                &total_elem_elem_entries, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    elems_in_elem_on_rank.resize(total_elem_elem_entries);
    
    // Now scatter the num_elems_in_elem for each element on each rank
    num_elems_in_elem_per_rank.resize(num_elements_on_rank);
    
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


    // ****************************************************************************************** 
    //     Initialize the naive_mesh data structures for each rank
    // ****************************************************************************************** 
    naive_mesh.initialize_nodes(num_nodes_on_rank);
    naive_mesh.initialize_elems(num_elements_on_rank, 3);

    naive_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(num_nodes_on_rank, "naive_mesh.local_to_global_node_mapping");
    naive_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(num_elements_on_rank, "naive_mesh.local_to_global_elem_mapping");

    for(int i = 0; i < num_nodes_on_rank; i++) {
        naive_mesh.local_to_global_node_mapping.host(i) = nodes_on_rank[i];
    }   

    for(int i = 0; i < num_elements_on_rank; i++) {
        naive_mesh.local_to_global_elem_mapping.host(i) = elements_on_rank[i];
    }

    naive_mesh.local_to_global_node_mapping.update_device();
    naive_mesh.local_to_global_elem_mapping.update_device();

    MPI_Barrier(MPI_COMM_WORLD);

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
                size_t mid_gid = naive_mesh.local_to_global_node_mapping.host(mid);
                if (node_gid == mid_gid) {
                    node_lid = mid;
                    break;
                } else if (node_gid < mid_gid) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            naive_mesh.nodes_in_elem.host(i, j) = node_lid;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t_reverse_map_end = MPI_Wtime();
    if(rank == 0 && print_info) {
        std::cout<<" Finished reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;
        std::cout<<" Reverse mapping time: " << (t_reverse_map_end - t_reverse_map_start) << " seconds." << std::endl;
    }

    naive_mesh.nodes_in_elem.update_device();

    // ****************************************************************************************** 
    //     Build the connectivity for the local naive_mesh
    // ****************************************************************************************** 

    naive_mesh.build_connectivity();
    MPI_Barrier(MPI_COMM_WORLD);
    

    return;
}



void partition_mesh(
    Mesh_t& initial_mesh,
    Mesh_t& final_mesh,
    node_t& initial_node,
    node_t& final_node,
    GaussPoint_t& gauss_point,
    int world_size,
    int rank){

    bool print_info = false;
    bool print_vtk = false;

    // Create mesh, gauss points, and node data structures on each rank
    // This is the initial partitioned mesh
    Mesh_t naive_mesh;
    node_t naive_node;

    // Mesh partitioned by pt-scotch, not including ghost
    Mesh_t intermediate_mesh; 
    node_t intermediate_node;


    // Helper arrays to hold element-element connectivity for naive partitioning that include what would be ghost, without having to build the full mesh
    std::vector<int> elems_in_elem_on_rank;
    std::vector<int> num_elems_in_elem_per_rank;

    naive_partition_mesh(initial_mesh, initial_node, naive_mesh, naive_node, elems_in_elem_on_rank, num_elems_in_elem_per_rank, world_size, rank);


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
    const SCOTCH_Num vertlocnbr = static_cast<SCOTCH_Num>(naive_mesh.num_elems);

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
    for (size_t k = 0; k < naive_mesh.num_elems; k++) {
        int elem_gid_on_rank = naive_mesh.local_to_global_elem_mapping.host(k);
        elem_gid_to_offset[elem_gid_on_rank] = current_offset;
        current_offset += num_elems_in_elem_per_rank[k]; // WARNING< THIS MUST INCLUDE GHOST< WHICH DONT EXISTS ON THE NAIVE MESH
    }

    // --- Step 3: Fill in the CSR arrays, looping over each locally-owned element ---
    SCOTCH_Num offset = 0; // running count of edges encountered

    for (size_t lid = 0; lid < naive_mesh.num_elems; lid++) {

        // Record current edge offset for vertex lid in vertloctab
        vertloctab[lid] = offset;

        // Obtain this local element's global ID (from mapping)
        int elem_gid = naive_mesh.local_to_global_elem_mapping.host(lid);

        // Find offset in the flattened neighbor array for this element's neighbor list
        size_t elems_in_elem_offset = elem_gid_to_offset[elem_gid];

        // For this element, find the count of its neighbors
        // This requires finding its index in the elements_on_rank array
        size_t idx = 0;
        for (size_t k = 0; k < naive_mesh.num_elems; k++) {
            int elem_gid_on_rank = naive_mesh.local_to_global_elem_mapping.host(k);
            if (elem_gid_on_rank == elem_gid) {
                idx = k;
                break;
            }
        }
        size_t num_nbrs = num_elems_in_elem_per_rank[idx];

        // Append each neighbor (by its GLOBAL elem GID) to edgeloctab
        for (size_t j = 0; j < num_nbrs; j++) {
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
     **************************************************************/
    // SCOTCH_Arch controls the "architecture" for partitioning: the topology
    // (number and connectivity of parts) to which the graph will be mapped.
    // The archdat variable encodes this. Below are common options:
    //
    // - SCOTCH_archCmplt(&archdat, nbparts)
    //     * Creates a "complete graph" architecture with nbparts nodes (fully connected).
    //       Every part is equally distant from every other part.
    //       This is typically used when minimizing only *balance* and *edge cut*,
    //       not considering any underlying machine topology.
    //
    // - SCOTCH_archHcub(&archdat, dimension)
    //     * Hypercube architecture (rare in modern use).
    //       Sets up a hypercube of given dimension.
    //
    // - SCOTCH_archTleaf / SCOTCH_archTleafX
    //     * Tree architectures, for hierarchically structured architectures.
    //
    // - SCOTCH_archMesh2 / SCOTCH_archMesh3
    //     * 2D or 3D mesh topology architectures (useful for grid/matrix machines).
    //
    // - SCOTCH_archBuild
    //     * General: builds any architecture from a descriptor string.
    //
    // For distributed mesh partitioning to MPI ranks (where all ranks are equal),
    // the most common and appropriate is "complete graph" (Cmplt): each part (rank)
    // is equally reachable from any other (no communication topology bias).
    SCOTCH_Arch archdat;        // PT-Scotch architecture structure: describes desired partition topology
    SCOTCH_archInit(&archdat);
    // Partition into 'world_size' equally connected parts (each MPI rank is a "node")
    // Other topology options could be substituted above according to your needs (see docs).
    SCOTCH_archCmplt(&archdat, static_cast<SCOTCH_Num>(world_size)); 



    
    // ===================== PT-Scotch Strategy Selection and Documentation ======================
    // The PT-Scotch "strategy" (stratdat here) controls the algorithms and heuristics used for partitioning.
    // You can specify a string or build a strategy using functions that adjust speed, quality, and recursion.
    //
    // Common strategy flags (see "scotch.h", "ptscotch.h", and PT-Scotch documentation):
    //
    // - SCOTCH_STRATDEFAULT:     Use the default (fast, reasonable quality) partitioning strategy.
    //                            Useful for quick, generic partitions where quality is not critical.
    //
    // - SCOTCH_STRATSPEED:       Aggressively maximizes speed (at the cost of cut quality).
    //                            For large runs or test runs where speed is more important than minimizing edgecut.
    //
    // - SCOTCH_STRATQUALITY:     Prioritizes partition *quality* (minimizing edge cuts, maximizing load balance).
    //                            Slower than the default. Use when high-quality partitioning is desired.
    //
    // - SCOTCH_STRATBALANCE:     Tradeoff between speed and quality for balanced workload across partitions.
    //                            Use if load balance is more critical than cut size.
    //
    // Additional Options:
    // - Strategy can also be specified as a string (see Scotch manual, e.g., "b{sep=m{...} ...}").
    // - Recursion count parameter (here, set to 0) controls strategy recursion depth (0 = automatic).
    // - Imbalance ratio (here, 0.01) allows minor imbalance in part weight for better cut quality.
    //
    // Example usage:
    //   SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY, nparts, 0, 0.01);
    //      ^ quality-focused, nparts=number of parts/ranks
    //   SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSPEED, nparts, 0, 0.05);
    //      ^ speed-focused, allow 5% imbalance
    //
    // Reference:
    // - https://gitlab.inria.fr/scotch/scotch/-/blob/master/doc/libptscotch.pdf
    // - SCOTCH_stratDgraphMapBuild() and related "strategy" documentation.
    //
    // --------------- Set up the desired partitioning strategy here: ---------------
    SCOTCH_Strat stratdat;      // PT-Scotch strategy object: holds partitioning options/settings
    SCOTCH_stratInit(&stratdat);

    // Select partitioning strategy for this run:
    // Use SCOTCH_STRATQUALITY for best cut quality.
    // To change: replace with SCOTCH_STRATDEFAULT, SCOTCH_STRATSPEED, or SCOTCH_STRATBALANCE as discussed above.
    // Arguments: (strategy object, strategy flag, #parts, recursion (0=auto), imbalance ratio)
    SCOTCH_stratDgraphMapBuild(&stratdat, SCOTCH_STRATQUALITY, world_size, 0, 0.01);

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
    print_info = false;
    for(int rank_id = 0; rank_id < world_size; rank_id++) {
        if(rank_id == rank && print_info) {
            for (size_t lid = 0; lid < naive_mesh.num_elems; lid++) {
                size_t gid = naive_mesh.local_to_global_elem_mapping.host(lid);
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
    for (int lid = 0; lid < naive_mesh.num_elems; lid++) {
        int dest = static_cast<int>(partloctab[lid]);
        int elem_gid = static_cast<int>(naive_mesh.local_to_global_elem_mapping.host(lid));
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
    // send_elems: flattened list of element global IDs (GIDs) that this rank is sending to all other ranks.
    // For each rank r, elems_to_send[r] contains the element GIDs that should be owned by rank r after repartitioning.
    std::vector<int> send_elems;
    send_elems.reserve(send_total);
    for (int r = 0; r < world_size; ++r)
        send_elems.insert(send_elems.end(), elems_to_send[r].begin(), elems_to_send[r].end());

    // new_elem_gids: receives the list of new element global IDs this rank will own after the exchange.
    // It is filled after MPI_Alltoallv completes, and contains the GIDs for the elements new to (or remained on) this rank.
    std::vector<int> new_elem_gids(recv_total);
    MPI_Alltoallv(send_elems.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                new_elem_gids.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // New elements owned by this rank
    int num_new_elems = static_cast<int>(new_elem_gids.size());
    
    if (print_info) {
        std::cout << "[rank " << rank << "] new elems: " << num_new_elems << std::endl;
    }

    // -------------- Phase 3: Send element–node connectivity --------------
    int nodes_per_elem = naive_mesh.num_nodes_in_elem;

    // Flatten element-node connectivity by global node IDs
    std::vector<int> conn_sendbuf;
    for (int r = 0; r < world_size; r++) {
        for (int elem_gid : elems_to_send[r]) {
            // find local element lid from elem_gid
            int lid = -1;
            for (int i = 0; i < naive_mesh.num_elems; i++)
                if (naive_mesh.local_to_global_elem_mapping.host(i) == elem_gid) { lid = i; break; }

            for (int j = 0; j < nodes_per_elem; j++) {
                int node_lid = naive_mesh.nodes_in_elem.host(lid, j);
                int node_gid = naive_mesh.local_to_global_node_mapping.host(node_lid);
                conn_sendbuf.push_back(node_gid);
            }
        }
    }

    // element-node connectivity counts (ints per dest rank)
    std::vector<int> conn_sendcounts(world_size), conn_recvcounts(world_size);
    for (int r = 0; r < world_size; r++)
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
    for (int i = 0; i < num_new_nodes; i++)
        node_gid_to_lid[new_node_gids[i]] = i;

    if (print_info)
        std::cout << "[rank " << rank << "] owns " << num_new_nodes << " unique nodes\n";


    // -------------- Phase 5: Request node coordinates --------------
    std::vector<double> node_coords_sendbuf;
    for (int r = 0; r < world_size; ++r) {
        for (int gid : elems_to_send[r]) {
            int lid = -1;
            for (int i = 0; i < naive_mesh.num_elems; i++)
                if (naive_mesh.local_to_global_elem_mapping.host(i) == gid) { lid = i; break; }

            for (int j = 0; j < nodes_per_elem; j++) {
                int node_lid = naive_mesh.nodes_in_elem.host(lid, j);
                int node_gid = naive_mesh.local_to_global_node_mapping.host(node_lid);

                node_coords_sendbuf.push_back(naive_node.coords.host(node_lid, 0));
                node_coords_sendbuf.push_back(naive_node.coords.host(node_lid, 1));
                node_coords_sendbuf.push_back(naive_node.coords.host(node_lid, 2));
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

    // -------------- Phase 6: Build the intermediate_mesh --------------
    intermediate_mesh.initialize_nodes(num_new_nodes);
    intermediate_mesh.initialize_elems(num_new_elems, naive_mesh.num_dims);
    intermediate_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(num_new_nodes);
    intermediate_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(num_new_elems);

    // Fill global mappings
    for (int i = 0; i < num_new_nodes; i++)
        intermediate_mesh.local_to_global_node_mapping.host(i) = new_node_gids[i];
    for (int i = 0; i < num_new_elems; i++)
        intermediate_mesh.local_to_global_elem_mapping.host(i) = new_elem_gids[i];

    intermediate_mesh.local_to_global_node_mapping.update_device();
    intermediate_mesh.local_to_global_elem_mapping.update_device();


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Starting reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;
    // rebuild the local element-node connectivity using the local node ids
    for(int i = 0; i < intermediate_mesh.num_elems; i++) {
        for(int j = 0; j < intermediate_mesh.num_nodes_in_elem; j++) {

            int node_gid = conn_recvbuf[i * intermediate_mesh.num_nodes_in_elem + j];

            int node_lid = -1;

            // Binary search through local_to_global_node_mapping to find the equivalent local index
            int left = 0, right = num_new_nodes - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                size_t mid_gid = intermediate_mesh.local_to_global_node_mapping.host(mid);
                if (node_gid == mid_gid) {
                    node_lid = mid;
                    break;
                } else if (node_gid < mid_gid) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            intermediate_mesh.nodes_in_elem.host(i, j) = node_lid;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished reverse mapping of the element-node connectivity from the global node ids to the local node ids"<<std::endl;

    intermediate_mesh.nodes_in_elem.update_device();

    // Fill node coordinates
    // coord_recvbuf contains coords in element-node order, but we need them in node order
    // Build a map from node GID to coordinates
    std::map<int, std::array<double, 3>> node_gid_to_coords;
    int coord_idx = 0;
    for (int e = 0; e < intermediate_mesh.num_elems; ++e) {
        for (int j = 0; j < intermediate_mesh.num_nodes_in_elem; j++) {
            int node_gid = conn_recvbuf[e * intermediate_mesh.num_nodes_in_elem + j];
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
    intermediate_node.initialize(num_new_nodes, 3, {node_state::coords});
    for (int i = 0; i < num_new_nodes; i++) {
        int node_gid = new_node_gids[i];
        auto it = node_gid_to_coords.find(node_gid);
        if (it != node_gid_to_coords.end()) {
            intermediate_node.coords.host(i, 0) = it->second[0];
            intermediate_node.coords.host(i, 1) = it->second[1];
            intermediate_node.coords.host(i, 2) = it->second[2];
        }
    }
    intermediate_node.coords.update_device();

    // Connectivity rebuild
    intermediate_mesh.build_connectivity();
    MPI_Barrier(MPI_COMM_WORLD);



// ****************************************************************************************** 
//     Build the ghost elements and nodes
// ****************************************************************************************** 

    double t_ghost_start = MPI_Wtime();
    
    // First, gather the number of elements each rank owns
    std::vector<int> elem_counts(world_size);
    MPI_Allgather(&intermediate_mesh.num_elems, 1, MPI_INT, elem_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Compute displacements
    std::vector<int> elem_displs(world_size);
    int total_elems = 0;
    for (int r = 0; r < world_size; ++r) {
        elem_displs[r] = total_elems;
        total_elems += elem_counts[r];
    }
    
    // Gather all element GIDs from all ranks
    std::vector<size_t> all_elem_gids(total_elems);
    MPI_Allgatherv(intermediate_mesh.local_to_global_elem_mapping.host_pointer(), intermediate_mesh.num_elems, MPI_UNSIGNED_LONG_LONG,
                   all_elem_gids.data(), elem_counts.data(), elem_displs.data(), 
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // Build a map: element GID -> owning rank
    std::map<size_t, int> elem_gid_to_rank;
    for (int r = 0; r < world_size; ++r) {
        for (int i = 0; i < elem_counts[r]; i++) {
            size_t gid = all_elem_gids[elem_displs[r] + i];
            elem_gid_to_rank[gid] = r;
        }
    }
    
    // Strategy: Find elements on other ranks that share
    // nodes with our locally-owned elements.
    
    // First, collect all nodes that belong to our locally-owned elements
    std::set<size_t> local_elem_nodes;

    for(int node_rid = 0; node_rid < intermediate_mesh.num_nodes; node_rid++) {
        size_t node_gid = intermediate_mesh.local_to_global_node_mapping.host(node_rid);
        local_elem_nodes.insert(node_gid);
    }
    
    
    // Now collect element-to-node connectivity to send to all ranks
    // Format: for each element, list its node GIDs (each entry is a pair: elem_gid, node_gid)
    std::vector<size_t> elem_node_conn;
    int local_conn_size = 0;
    
    for (int lid = 0; lid < intermediate_mesh.num_elems; lid++) {
        size_t elem_gid = intermediate_mesh.local_to_global_elem_mapping.host(lid);
        for (int j = 0; j < intermediate_mesh.num_nodes_in_elem; j++) {
            size_t node_lid = intermediate_mesh.nodes_in_elem.host(lid, j);
            size_t node_gid = intermediate_mesh.local_to_global_node_mapping.host(node_lid);
            elem_node_conn.push_back(elem_gid);
            elem_node_conn.push_back(node_gid);
        }
        local_conn_size += nodes_per_elem * 2;  // Each pair is 2 size_ts
    }

   
    
    // Exchange element-node connectivity with all ranks using Allgather
    // First, gather the sizes from each rank
    std::vector<int> conn_sizes(world_size);
    MPI_Allgather(&local_conn_size, 1, MPI_INT, conn_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
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
    MPI_Barrier(MPI_COMM_WORLD);

    
    DCArrayKokkos<size_t> local_nodes_in_elem(intermediate_mesh.num_elems, intermediate_mesh.num_nodes_in_elem);
    DCArrayKokkos<size_t> all_nodes_in_elem(total_elems, intermediate_mesh.num_nodes_in_elem);

    std::vector<int> mtr_conn_sizes(world_size);
    

    local_nodes_in_elem = intermediate_mesh.nodes_in_elem;
    int mtr_size = intermediate_mesh.num_elems * intermediate_mesh.num_nodes_in_elem;

    MPI_Allgather(&mtr_size, 1, MPI_INT, mtr_conn_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> mtr_conn_displs(world_size);
    int total_mtr_conn = 0;
    for (int r = 0; r < world_size; ++r) {
        mtr_conn_displs[r] = total_mtr_conn;
        total_mtr_conn += mtr_conn_sizes[r];
    }


    MPI_Allgatherv(local_nodes_in_elem.host_pointer(), mtr_size, MPI_UNSIGNED_LONG_LONG,
                   all_nodes_in_elem.host_pointer(), mtr_conn_sizes.data(), mtr_conn_displs.data(),
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);



    
    // create a set for local_elem_gids
    std::set<size_t> local_elem_gids;
    for (int i = 0; i < intermediate_mesh.num_elems; i++) {
        local_elem_gids.insert(intermediate_mesh.local_to_global_elem_mapping.host(i));
    }
    
    // Build a map: node GID -> set of element GIDs that contain it (from other ranks)
    std::map<size_t, std::set<size_t>> node_to_ext_elem;
    for (int r = 0; r < world_size; ++r) {
        if (r == rank) continue;  // Skip our own data
        // Process pairs from rank r: conn_sizes[r] is in units of size_ts, so num_pairs = conn_sizes[r] / 2
        int num_pairs = conn_sizes[r] / 2;
        for (int i = 0; i < num_pairs; i++) {
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
    
    // for (int lid = 0; lid < num_new_elems; lid++) {
    //     size_t num_neighbors = intermediate_mesh.num_elems_in_elem(lid);
        
    //     for (size_t nbr_idx = 0; nbr_idx < num_neighbors; ++nbr_idx) {
    //         size_t neighbor_lid = intermediate_mesh.elems_in_elem(lid, nbr_idx);
            
    //         if (neighbor_lid < static_cast<size_t>(num_new_elems)) {
    //             size_t neighbor_gid = intermediate_mesh.local_to_global_elem_mapping(neighbor_lid);
                
    //             // Check if neighbor is owned by this rank
    //             auto it = elem_gid_to_rank.find(neighbor_gid);
    //             if (it != elem_gid_to_rank.end() && it->second != rank) {
    //                 // Neighbor is owned by another rank - it's a ghost for us
    //                 std::cout << "[rank " << rank << "] found ghost element " << neighbor_gid << std::endl;
    //                 ghost_elem_gids.insert(neighbor_gid);
    //             }
    //         }
    //     }
    // }
    
    // Count unique ghost elements
    intermediate_mesh.num_ghost_elems = ghost_elem_gids.size();
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_ghost_end = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << " Finished calculating ghost elements" << std::endl;
        std::cout << " Ghost element calculation took " << (t_ghost_end - t_ghost_start) << " seconds." << std::endl;
    }
    // Build the connectivity that includes ghost elements
    // Create an extended mesh with owned elements first, then ghost elements appended
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout << " Starting to build extended mesh with ghost elements" << std::endl;
    
    // Step 1: Extract ghost element-node connectivity from all_conn
    // Build a map: ghost_elem_gid -> vector of node_gids (ordered as in all_conn)
    std::map<size_t, std::vector<size_t>> ghost_elem_to_nodes;
    for (const size_t& ghost_gid : ghost_elem_gids) {
        ghost_elem_to_nodes[ghost_gid].reserve(intermediate_mesh.num_nodes_in_elem);
    }
    
    // Extract nodes for each ghost element from all_conn
    // The all_conn array has pairs (elem_gid, node_gid) for each rank's elements
    for (int r = 0; r < world_size; ++r) {
        if (r == rank) continue;  // Skip our own data (we already have owned element connectivity)
        
        int num_pairs = conn_sizes[r] / 2;
        
        // Process pairs in order - each element's nodes are contiguous
        for (int i = 0; i < num_pairs; i++) {
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // If this is one of our ghost elements, record its node (in order)
            auto it = ghost_elem_to_nodes.find(elem_gid);
            if (it != ghost_elem_to_nodes.end()) {
                it->second.push_back(node_gid);
            }
        }
    }
    
    // Verify each ghost element has the correct number of nodes
    for (auto& pair : ghost_elem_to_nodes) {
        if (pair.second.size() != static_cast<size_t>(intermediate_mesh.num_nodes_in_elem)) {
            std::cerr << "[rank " << rank << "] ERROR: Ghost element " << pair.first 
                      << " has " << pair.second.size() << " nodes, expected " << intermediate_mesh.num_nodes_in_elem << std::endl;
        }
    }
    
    // Step 2: Build extended node list (owned nodes first, then ghost-only nodes)
    // Start with owned nodes
    std::map<size_t, int> node_gid_to_extended_lid;
    int extended_node_lid = 0;
    
    // Add all owned nodes
    for (int i = 0; i < intermediate_mesh.num_nodes; i++) {
        size_t node_gid = intermediate_mesh.local_to_global_node_mapping.host(i);
        node_gid_to_extended_lid[node_gid] = extended_node_lid++;
    }
    
    // Add ghost-only nodes (nodes that belong to ghost elements but not to owned elements)
    std::set<size_t> ghost_only_nodes;
    for (const auto& pair : ghost_elem_to_nodes) {
        for (size_t node_gid : pair.second) {
            // Check if we already have this node
            if (node_gid_to_extended_lid.find(node_gid) == node_gid_to_extended_lid.end()) {
                ghost_only_nodes.insert(node_gid);
            }
        }
    }
    
    // Assign extended local IDs to ghost-only nodes
    for (size_t node_gid : ghost_only_nodes) {
        node_gid_to_extended_lid[node_gid] = extended_node_lid++;
    }
    
    int total_extended_nodes = extended_node_lid;
    
    // Step 3: Prepare requests for ghost node coordinates from owning ranks (if needed later)
    // Build request list: for each ghost node, find an owning rank via any ghost element that contains it
    std::map<int, std::vector<size_t>> rank_to_ghost_node_requests;
    for (size_t node_gid : ghost_only_nodes) {
        // Find which rank owns an element containing this node
        // Look through ghost elements
        for (const auto& pair : ghost_elem_to_nodes) {
            size_t ghost_elem_gid = pair.first;
            const std::vector<size_t>& nodes = pair.second;
            bool found = false;
            for (size_t ngid : nodes) {
                if (ngid == node_gid) {
                    found = true;
                    break;
                }
            }
            if (found) {
                auto owner_it = elem_gid_to_rank.find(ghost_elem_gid);
                if (owner_it != elem_gid_to_rank.end()) {
                    rank_to_ghost_node_requests[owner_it->second].push_back(node_gid);
                    break;
                }
            }
        }
    }
    
    // Step 4: Build extended element list and node connectivity
    // Owned elements: 0 to num_new_elems-1 (already have these)
    // Ghost elements: num_new_elems to num_new_elems + num_ghost_elems - 1
    
    // Create extended element-node connectivity array
    int total_extended_elems = intermediate_mesh.num_elems + intermediate_mesh.num_ghost_elems;
    std::vector<std::vector<int>> extended_nodes_in_elem(total_extended_elems);
    
    // Copy owned element connectivity (convert to extended node LIDs)
    for (int lid = 0; lid < intermediate_mesh.num_elems; lid++) {
        extended_nodes_in_elem[lid].reserve(nodes_per_elem);
        for (int j = 0; j < nodes_per_elem; j++) {
            size_t node_lid = intermediate_mesh.nodes_in_elem.host(lid, j);
            size_t node_gid = intermediate_mesh.local_to_global_node_mapping.host(node_lid);
            int ext_lid = node_gid_to_extended_lid[node_gid];
            extended_nodes_in_elem[lid].push_back(ext_lid);
        }
    }
    
    // Add ghost element connectivity (map ghost node GIDs to extended node LIDs)
    int ghost_elem_ext_lid = intermediate_mesh.num_elems;
    std::vector<size_t> ghost_elem_gids_ordered(ghost_elem_gids.begin(), ghost_elem_gids.end());
    std::sort(ghost_elem_gids_ordered.begin(), ghost_elem_gids_ordered.end());
    
    for (size_t ghost_gid : ghost_elem_gids_ordered) {
        auto it = ghost_elem_to_nodes.find(ghost_gid);
        if (it == ghost_elem_to_nodes.end()) continue;
        
        extended_nodes_in_elem[ghost_elem_ext_lid].reserve(nodes_per_elem);
        for (size_t node_gid : it->second) {
            int ext_lid = node_gid_to_extended_lid[node_gid];
            extended_nodes_in_elem[ghost_elem_ext_lid].push_back(ext_lid);
        }
        ghost_elem_ext_lid++;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // Sequential rank-wise printing of extended mesh structure info
    if(print_info) {
        for (int r = 0; r < world_size; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                std::cout << "[rank " << rank << "] Finished building extended mesh structure" << std::endl;
                std::cout << "[rank " << rank << "]   - Owned elements: " << intermediate_mesh.num_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Ghost elements: " << ghost_elem_gids.size() << std::endl;
                std::cout << "[rank " << rank << "]   - Total extended elements: " << total_extended_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Owned nodes: " << intermediate_mesh.num_nodes << std::endl;
                std::cout << "[rank " << rank << "]   - Ghost-only nodes: " << ghost_only_nodes.size() << std::endl;
                std::cout << "[rank " << rank << "]   - Total extended nodes: " << total_extended_nodes << std::endl;
                std::cout << std::flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // The extended_nodes_in_elem vector now contains the connectivity for both owned and ghost elements
    // Each element's nodes are stored using extended local node IDs (0-based, contiguous)
    
    // Build reverse maps: extended_lid -> gid for nodes and elements
    std::vector<size_t> extended_lid_to_node_gid(total_extended_nodes);
    for (const auto& pair : node_gid_to_extended_lid) {
        extended_lid_to_node_gid[pair.second] = pair.first;
    }
    
    // Build extended element GID list: owned first, then ghost
    std::vector<size_t> extended_lid_to_elem_gid(total_extended_elems);
    // Owned elements
    for (int i = 0; i < intermediate_mesh.num_elems; i++) {
        extended_lid_to_elem_gid[i] = intermediate_mesh.local_to_global_elem_mapping.host(i);
    }
    // Ghost elements (in sorted order)
    for (size_t i = 0; i < ghost_elem_gids_ordered.size(); i++) {
        extended_lid_to_elem_gid[intermediate_mesh.num_elems + i] = ghost_elem_gids_ordered[i];
    }

    // Build array: for each ghost element, store which rank owns it (where to receive data from)
    std::vector<int> ghost_elem_owner_ranks(ghost_elem_gids_ordered.size());
    for (size_t i = 0; i < ghost_elem_gids_ordered.size(); i++) {
        size_t ghost_gid = ghost_elem_gids_ordered[i];
        auto it = elem_gid_to_rank.find(ghost_gid);
        if (it != elem_gid_to_rank.end()) {
            ghost_elem_owner_ranks[i] = it->second;
        } else {
            std::cerr << "[rank " << rank << "] ERROR: Ghost element GID " << ghost_gid 
                      << " not found in elem_gid_to_rank map!" << std::endl;
            ghost_elem_owner_ranks[i] = -1; // Invalid rank as error indicator
        }
    }

    // Create a std::set of all the ranks this rank will receive data from
    std::set<int> ghost_elem_receive_ranks;
    for (size_t i = 0; i < ghost_elem_gids_ordered.size(); i++) {
        ghost_elem_receive_ranks.insert(ghost_elem_owner_ranks[i]);
    }

// ****************************************************************************************** 
//     Build the final partitioned mesh
// ****************************************************************************************** 


    final_mesh.initialize_nodes(total_extended_nodes);
    final_mesh.initialize_elems(total_extended_elems, 3);
    final_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(total_extended_nodes);
    final_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(total_extended_elems);
    for (int i = 0; i < total_extended_nodes; i++) {
        final_mesh.local_to_global_node_mapping.host(i) = extended_lid_to_node_gid[i];
    }
    for (int i = 0; i < total_extended_elems; i++) {
        final_mesh.local_to_global_elem_mapping.host(i) = extended_lid_to_elem_gid[i];
    }
    final_mesh.local_to_global_node_mapping.update_device();
    final_mesh.local_to_global_elem_mapping.update_device();

    final_mesh.num_ghost_elems = ghost_elem_gids.size();
    final_mesh.num_ghost_nodes = ghost_only_nodes.size();
    

    final_mesh.num_owned_elems = intermediate_mesh.num_elems;
    final_mesh.num_owned_nodes = intermediate_mesh.num_nodes;

    MPI_Barrier(MPI_COMM_WORLD);
    // rebuild the local element-node connectivity using the local node ids
    // extended_nodes_in_elem already contains extended local node IDs, so we can use them directly
    for(int i = 0; i < total_extended_elems; i++) {
        for(int j = 0; j < nodes_per_elem; j++) {
            final_mesh.nodes_in_elem.host(i, j) = extended_nodes_in_elem[i][j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    final_mesh.nodes_in_elem.update_device();
    final_mesh.build_connectivity();

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0) std::cout << " Finished building final mesh structure with ghost nodes and elements" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

// ****************************************************************************************** 
//     Build the final nodes that include ghost
// ****************************************************************************************** 


    final_node.initialize(total_extended_nodes, 3, {node_state::coords});
    
    // The goal here is to populate final_node.coords using globally gathered ghost node coordinates,
    // since intermediate_node does not contain ghost node coordinates.
    //
    // Each rank will:
    //  1. Gather coordinates of its owned nodes (from intermediate_node).
    //  2. Use MPI to gather all coordinates for all required (owned + ghost) global node IDs
    //     into a structure mapping global ID -> coordinate.
    //  3. Use this map to fill final_node.coords.

    // 1. Build list of all global node IDs needed on this rank (owned + ghosts)
    std::vector<size_t> all_needed_node_gids(total_extended_nodes);
    for (int i = 0; i < total_extended_nodes; i++) {
        all_needed_node_gids[i] = final_mesh.local_to_global_node_mapping.host(i);
    }

    // 2. Build owned node GIDs and their coordinates
    std::vector<size_t> owned_gids(final_mesh.num_owned_nodes);
    for (int i = 0; i < final_mesh.num_owned_nodes; i++)
        owned_gids[i] = final_mesh.local_to_global_node_mapping.host(i);

     // 3. Gather all GIDs in the world that are needed anywhere (owned or ghosted, by any rank)
     //    so we can distribute the needed coordinate data.
     // The easiest is to Allgather everyone's "owned_gids" and coords
 
     int local_owned_count = static_cast<int>(owned_gids.size());
     std::vector<int> owned_counts(world_size, 0);
     if (local_owned_count < 0) local_owned_count = 0; // Clean up possibility of -1

    // a) Gather counts
    owned_counts.resize(world_size, 0);
    MPI_Allgather(&local_owned_count, 1, MPI_INT, owned_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // b) Displacements and total
    std::vector<int> owned_displs(world_size,0);
    int total_owned = 0;
    for (int r=0; r<world_size; ++r) {
        owned_displs[r] = total_owned;
        total_owned += owned_counts[r];
    }

    // c) Global GIDs (size: total_owned)
    std::vector<size_t> all_owned_gids(total_owned);
    MPI_Allgatherv(owned_gids.data(), local_owned_count, MPI_UNSIGNED_LONG_LONG,
                   all_owned_gids.data(), owned_counts.data(), owned_displs.data(),
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);


    // d) Global coords (size: total_owned x 3)
    std::vector<double> owned_coords_send(3*local_owned_count, 0.0);
    for (int i=0; i<local_owned_count; i++) {
        owned_coords_send[3*i+0] = intermediate_node.coords.host(i,0);
        owned_coords_send[3*i+1] = intermediate_node.coords.host(i,1);
        owned_coords_send[3*i+2] = intermediate_node.coords.host(i,2);
    }
    std::vector<double> all_owned_coords(3 * total_owned, 0.0);

    // Create coordinate-specific counts and displacements (in units of doubles, not nodes)
    std::vector<int> coord_counts(world_size);
    std::vector<int> coord_displs(world_size);
    for (int r=0; r<world_size; ++r) {
        coord_counts[r] = 3 * owned_counts[r];  // Each node has 3 doubles
        coord_displs[r] = 3 * owned_displs[r];  // Displacement in doubles
    }

    MPI_Allgatherv(owned_coords_send.data(), 3*local_owned_count, MPI_DOUBLE,
                   all_owned_coords.data(), coord_counts.data(), coord_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // e) Build map: gid -> coord[3]
    std::unordered_map<size_t, std::array<double,3>> gid_to_coord;
    for (int i=0; i<total_owned; i++) {
        std::array<double,3> xyz = {
            all_owned_coords[3*i+0],
            all_owned_coords[3*i+1],
            all_owned_coords[3*i+2]
        };
         gid_to_coord[all_owned_gids[i]] = xyz;
    }

    // 4. Finally, fill final_node.coords with correct coordinates.
    for (int i = 0; i < total_extended_nodes; i++) {
        size_t gid = final_mesh.local_to_global_node_mapping.host(i);
        auto it = gid_to_coord.find(gid);
        if (it != gid_to_coord.end()) {
            final_node.coords.host(i,0) = it->second[0];
            final_node.coords.host(i,1) = it->second[1];
            final_node.coords.host(i,2) = it->second[2];
        } else {
            // Could happen if there's a bug: fill with zeros for safety
            final_node.coords.host(i,0) = 0.0;
            final_node.coords.host(i,1) = 0.0;
            final_node.coords.host(i,2) = 0.0;
        }
    }
    final_node.coords.update_device();


    // --------------------------------------------------------------------------------------
// Build the send patterns for elements
    // Build reverse map via global IDs: for each local element gid, find ranks that ghost it.
    // Steps:
    // 1) Each rank contributes its ghost element GIDs.
    // 2) Allgatherv ghost GIDs to build gid -> [ranks that ghost it].
    // 3) For each locally-owned element gid, lookup ranks that ghost it and record targets.
    // --------------------------------------------------------------------------------------
    std::vector<std::vector<std::pair<int, size_t>>> boundary_elem_targets(final_mesh.num_owned_elems);

    // Prepare local ghost list as vector
    std::vector<size_t> ghost_gids_vec;
    ghost_gids_vec.reserve(final_mesh.num_ghost_elems);
    for (int i = 0; i < final_mesh.num_ghost_elems; i++) {
        ghost_gids_vec.push_back(final_mesh.local_to_global_elem_mapping.host(final_mesh.num_owned_elems + i)); // Ghost elements are after the owned elements in the global element mapping
    }

    // Exchange counts
    std::vector<int> ghost_counts(world_size, 0);
    int local_ghost_count = final_mesh.num_ghost_elems;
    MPI_Allgather(&local_ghost_count, 1, MPI_INT, ghost_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Displacements and recv buffer
    std::vector<int> ghost_displs(world_size, 0);
    int total_ghosts = 0;
    for (int r = 0; r < world_size; ++r) {
        ghost_displs[r] = total_ghosts;
        total_ghosts += ghost_counts[r];
    }
    std::vector<size_t> all_ghost_gids(total_ghosts);

    // Gather ghost gids
    MPI_Allgatherv(ghost_gids_vec.data(), local_ghost_count, MPI_UNSIGNED_LONG_LONG,
                   all_ghost_gids.data(), ghost_counts.data(), ghost_displs.data(),
                   MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    
    // Build map gid -> ranks that ghost it
    std::unordered_map<size_t, std::vector<int>> gid_to_ghosting_ranks;
    gid_to_ghosting_ranks.reserve(static_cast<size_t>(total_ghosts));
    for (int r = 0; r < world_size; ++r) {
        int cnt = ghost_counts[r];
        int off = ghost_displs[r];
        for (int i = 0; i < cnt; i++) {
            size_t g = all_ghost_gids[off + i];
            gid_to_ghosting_ranks[g].push_back(r);
        }
    }

    // For each local element, list destinations: ranks that ghost our gid
    for (int elem_lid = 0; elem_lid < final_mesh.num_owned_elems; elem_lid++) {
        size_t local_elem_gid = final_mesh.local_to_global_elem_mapping.host(elem_lid);
        auto it = gid_to_ghosting_ranks.find(local_elem_gid);
        if (it == gid_to_ghosting_ranks.end()) continue;
        const std::vector<int> &dest_ranks = it->second;
        for (int rr : dest_ranks) {
            if (rr == rank) continue;
            boundary_elem_targets[elem_lid].push_back(std::make_pair(rr, local_elem_gid));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Add a vector to store boundary element local_ids (those who have ghost destinations across ranks)
    std::vector<int> boundary_elem_local_ids;
    std::vector<std::vector<int>> boundary_to_ghost_ranks;  // ragged array dimensions (num_boundary_elems, num_ghost_ranks)

    std::set<int> ghost_comm_ranks; // set of ranks that this rank communicates with
    

    for (int elem_lid = 0; elem_lid < final_mesh.num_owned_elems; elem_lid++) {

        int local_elem_gid = final_mesh.local_to_global_elem_mapping.host(elem_lid);
        if (boundary_elem_targets[elem_lid].empty()) 
        {
            continue;
        }
        else
        {
            // Fill in vector of boundary local_ids
            boundary_elem_local_ids.push_back(elem_lid);
            std::vector<int> ghost_ranks_for_this_boundary_elem;
            for (const auto &pr : boundary_elem_targets[elem_lid]) {
                ghost_ranks_for_this_boundary_elem.push_back(pr.first);
                ghost_comm_ranks.insert(pr.first);
            }
            boundary_to_ghost_ranks.push_back(ghost_ranks_for_this_boundary_elem);
        }
    }

    int num_ghost_comm_ranks = ghost_comm_ranks.size();
    std::vector<int> ghost_comm_ranks_vec(num_ghost_comm_ranks);
    int i = 0;
    for (const auto &r : ghost_comm_ranks) {
        ghost_comm_ranks_vec[i] = r;
        i++;
    }


    MPI_Barrier(MPI_COMM_WORLD);

    final_mesh.num_boundary_elems = boundary_elem_local_ids.size();
    final_mesh.boundary_elem_local_ids = DCArrayKokkos<size_t>(final_mesh.num_boundary_elems);
    for (int i = 0; i < final_mesh.num_boundary_elems; i++) {
        final_mesh.boundary_elem_local_ids.host(i) = boundary_elem_local_ids[i];
    }
    final_mesh.boundary_elem_local_ids.update_device();

    print_info = false;

    
    MPI_Barrier(MPI_COMM_WORLD);


// ****************************************************************************************** 
//     Create Communication Plan for element communication
// ****************************************************************************************** 


    CommunicationPlan element_communication_plan;
    element_communication_plan.initialize(MPI_COMM_WORLD);
    // MPI_Dist_graph_create_adjacent creates a distributed graph topology communicator
    // that efficiently represents the communication pattern between ranks.
    // This allows MPI to optimize communication based on the actual connectivity pattern.
    
    
    // ---------- Prepare INCOMING edges (sources) ----------
    // indegree: Number of ranks from which this rank will RECEIVE data
    // These are the ranks that own elements which are ghosted on this rank
    std::vector<int> ghost_elem_receive_ranks_vec(ghost_elem_receive_ranks.begin(), 
                                                    ghost_elem_receive_ranks.end());
    // The number of ranks from which this rank will receive data (incoming neighbors)
    int indegree = static_cast<int>(ghost_elem_receive_ranks_vec.size());
    
    // sources: Array of source rank IDs (ranks we receive from)
    // Each element corresponds to a rank that owns elements we ghost
    int* sources = (indegree > 0) ? ghost_elem_receive_ranks_vec.data() : MPI_UNWEIGHTED;

    
    // sourceweights: Weights on incoming edges (not used here, set to MPI_UNWEIGHTED)
    // Could be used to specify communication volume if needed for optimization
    int* sourceweights = MPI_UNWEIGHTED;
    
    // ---------- Prepare OUTGOING edges (destinations) ----------
    // outdegree: Number of ranks to which this rank will SEND data
    // These are the ranks that ghost elements owned by this rank
    int outdegree = num_ghost_comm_ranks;
    
    // destinations: Array of destination rank IDs (ranks we send to)
    // Each element corresponds to a rank that ghosts our owned elements
    int* destinations = (outdegree > 0) ? ghost_comm_ranks_vec.data() : MPI_UNWEIGHTED;

    // Initialize the graph communicator for element communication
    element_communication_plan.initialize_graph_communicator(outdegree, ghost_comm_ranks_vec.data(), indegree, ghost_elem_receive_ranks_vec.data());
    MPI_Barrier(MPI_COMM_WORLD);
    // Optional: Verify the graph communicator was created successfully
    if(print_info) element_communication_plan.verify_graph_communicator();


// ****************************************************************************************** 
//     Build send counts and displacements for element communication
// ****************************************************************************************** 

 // ========== Build send counts and displacements for OUTGOING neighbors (destinations) ==========
    // For MPI_Neighbor_alltoallv with graph communicator:
    //   - elem_sendcounts[i] = number of elements to send to i-th outgoing neighbor (destinations_out[i])
    //   - elem_sdispls[i] = starting position in send buffer for i-th outgoing neighbor
    
    // std::vector<int> elem_sendcounts(element_communication_plan.num_send_ranks, 0);
    // std::vector<int> elem_sdispls(element_communication_plan.num_send_ranks, 0);
    
    // Count how many boundary elements go to each destination rank
    // boundary_elem_targets[elem_lid] contains pairs (dest_rank, elem_gid) for each boundary element
    std::map<int, std::vector<int>> elems_to_send_by_rank;  // rank -> list of boundary element local IDs
    
    for (int elem_lid = 0; elem_lid < intermediate_mesh.num_elems; elem_lid++) {
        if (!boundary_elem_targets[elem_lid].empty()) {
            for (const auto &pr : boundary_elem_targets[elem_lid]) {
                int dest_rank = pr.first;
                elems_to_send_by_rank[dest_rank].push_back(elem_lid);
            }
        }
    }

    // Serialize into a DRaggedRightArrayKokkos
    CArrayKokkos<size_t> strides_array(element_communication_plan.num_send_ranks);
    for (int i = 0; i < element_communication_plan.num_send_ranks; i++) {
        int dest_rank = element_communication_plan.send_rank_ids.host(i);
        strides_array(i) = elems_to_send_by_rank[dest_rank].size();
    }
    DRaggedRightArrayKokkos<int> elems_to_send_by_rank_rr(strides_array, "elems_to_send_by_rank");

    // Fill in the data
    for (int i = 0; i < element_communication_plan.num_send_ranks; i++) {
        int dest_rank = element_communication_plan.send_rank_ids.host(i);
        for (int j = 0; j < elems_to_send_by_rank[dest_rank].size(); j++) {
            elems_to_send_by_rank_rr.host(i, j) = elems_to_send_by_rank[dest_rank][j];
        }
    }
    elems_to_send_by_rank_rr.update_device();

    
    // Count how many ghost elements come from each source rank
    // ghost_elem_owner_ranks[i] tells us which rank owns the i-th ghost element
    std::map<int, std::vector<int>> elems_to_recv_by_rank;  // rank -> list of ghost element indices
    
    for (size_t i = 0; i < ghost_elem_owner_ranks.size(); i++) {
        int source_rank = ghost_elem_owner_ranks[i];
        int ghost_elem_local_id = final_mesh.num_owned_elems + i;
        elems_to_recv_by_rank[source_rank].push_back(ghost_elem_local_id);
    }

    // ========== Serialize into a DRaggedRightArrayKokkos ==========
    CArrayKokkos<size_t> elem_recv_strides_array(element_communication_plan.num_recv_ranks);
    for (int i = 0; i < element_communication_plan.num_recv_ranks; i++) {
        int source_rank = element_communication_plan.recv_rank_ids.host(i);
        elem_recv_strides_array(i) = elems_to_recv_by_rank[source_rank].size();
       
    }
    DRaggedRightArrayKokkos<int> elems_to_recv_by_rank_rr(elem_recv_strides_array, "elems_to_recv_by_rank");
    // Fill in the data
    for (int i = 0; i < element_communication_plan.num_recv_ranks; i++) {
        int source_rank = element_communication_plan.recv_rank_ids.host(i);
        for (int j = 0; j < elems_to_recv_by_rank[source_rank].size(); j++) {
            elems_to_recv_by_rank_rr.host(i, j) = elems_to_recv_by_rank[source_rank][j];
        }
    }
    elems_to_recv_by_rank_rr.update_device();
    element_communication_plan.setup_send_recv(elems_to_send_by_rank_rr, elems_to_recv_by_rank_rr);

    MPI_Barrier(MPI_COMM_WORLD);
    
// ****************************************************************************************** 
//     Test element communication using MPI_Neighbor_alltoallv
// ****************************************************************************************** 
    // Gauss points share the same communication plan as elements.
    // This test initializes gauss point fields on owned elements and exchanges them with ghost elements.

    std::vector<gauss_pt_state> gauss_pt_states = {gauss_pt_state::fields, gauss_pt_state::fields_vec};

    gauss_point.initialize(final_mesh.num_elems, final_mesh.num_dims, gauss_pt_states, element_communication_plan); // , &element_communication_plan

    // Initialize the gauss point fields on each rank
    // Set owned elements to rank number, ghost elements to -1 (to verify communication)
    for (int i = 0; i < final_mesh.num_owned_elems; i++) {
        gauss_point.fields.host(i) = static_cast<double>(rank);
        gauss_point.fields_vec.host(i, 0) = static_cast<double>(rank);
        gauss_point.fields_vec.host(i, 1) = static_cast<double>(rank);
        gauss_point.fields_vec.host(i, 2) = static_cast<double>(rank);
    }
    for (int i = final_mesh.num_owned_elems; i < final_mesh.num_elems; i++) {
        gauss_point.fields.host(i) = -1.0;  // Ghost elements should be updated
        gauss_point.fields_vec.host(i, 0) = -1.0;
        gauss_point.fields_vec.host(i, 1) = -1.0;
        gauss_point.fields_vec.host(i, 2) = -1.0;
    }
    gauss_point.fields.update_device();
    gauss_point.fields_vec.update_device();
    
    gauss_point.fields.communicate();
    gauss_point.fields_vec.communicate();
    
    // Loop over all elements and average the values of elements connected to that element
    for (int i = 0; i < final_mesh.num_elems; i++) {
        double value = 0.0;
        for (int j = 0; j < final_mesh.num_elems_in_elem(i); j++) {
            value += gauss_point.fields.host(final_mesh.elems_in_elem(i, j));
        }
        value /= final_mesh.num_elems_in_elem(i);
        gauss_point.fields.host(i) = value;
    }
    for (int i = 0; i < final_mesh.num_elems; i++) {
        double value = 0.0;
        for (int j = 0; j < final_mesh.num_elems_in_elem(i); j++) {
            value += gauss_point.fields_vec.host(final_mesh.elems_in_elem(i, j), 0);
        }
        value /= final_mesh.num_elems_in_elem(i);
        gauss_point.fields_vec.host(i, 0) = value;
        gauss_point.fields_vec.host(i, 1) = value;
        gauss_point.fields_vec.host(i, 2) = value;
    }
    gauss_point.fields_vec.update_device();



    // --------------------------------------------------------------------------------------
// Build the send pattern for nodes
    // Build reverse map via global IDs: for each local node gid, find ranks that ghost it.
    // Steps:
    // 1) Each rank contributes its ghost node GIDs.
    // 2) Allgatherv ghost node GIDs to build gid -> [ranks that ghost it].
    // 3) For each locally-owned node gid, lookup ranks that ghost it and record targets.
    // --------------------------------------------------------------------------------------
    
    // std::vector<std::vector<std::pair<int, size_t>>> boundary_node_targets(intermediate_mesh.num_nodes);
    
    // // Prepare local ghost node list as vector
    // std::vector<size_t> ghost_node_gids_vec;
    // ghost_node_gids_vec.reserve(ghost_only_nodes.size());
    // for (const auto &g : ghost_only_nodes) ghost_node_gids_vec.push_back(g);
    
    // // Exchange counts
    // std::vector<int> ghost_node_counts(world_size, 0);
    // int local_ghost_node_count = static_cast<int>(ghost_node_gids_vec.size());
    // MPI_Allgather(&local_ghost_node_count, 1, MPI_INT, ghost_node_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // // Displacements and recv buffer
    // std::vector<int> ghost_node_displs(world_size, 0);
    // int total_ghost_nodes = 0;
    // for (int r = 0; r < world_size; ++r) {
    //     ghost_node_displs[r] = total_ghost_nodes;
    //     total_ghost_nodes += ghost_node_counts[r];
    // }
    // std::vector<size_t> all_ghost_node_gids(total_ghost_nodes);
    
    // // Gather ghost node gids
    // MPI_Allgatherv(ghost_node_gids_vec.data(), local_ghost_node_count, MPI_UNSIGNED_LONG_LONG,
    //                all_ghost_node_gids.data(), ghost_node_counts.data(), ghost_node_displs.data(),
    //                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    
    // MPI_Barrier(MPI_COMM_WORLD);
    // if(rank == 0) std::cout << " Finished gathering ghost node GIDs" << std::endl;
    
    
    // MPI_Barrier(MPI_COMM_WORLD);
    // if(rank == 0) std::cout << " Starting to build the reverse map for node communication" << std::endl;
    
    // // Build map node_gid -> ranks that ghost it
    // std::unordered_map<size_t, std::vector<int>> node_gid_to_ghosting_ranks;
    // node_gid_to_ghosting_ranks.reserve(static_cast<size_t>(total_ghost_nodes));
    // for (int r = 0; r < world_size; ++r) {
    //     int cnt = ghost_node_counts[r];
    //     int off = ghost_node_displs[r];
    //     for (int i = 0; i < cnt; i++) {
    //         size_t g = all_ghost_node_gids[off + i];
    //         node_gid_to_ghosting_ranks[g].push_back(r);
    //     }
    // }
    
    // // For each local node, list destinations: ranks that ghost our node gid
    // for (int node_lid = 0; node_lid < intermediate_mesh.num_nodes; node_lid++) {
    //     size_t local_node_gid = intermediate_mesh.local_to_global_node_mapping.host(node_lid);
    //     auto it = node_gid_to_ghosting_ranks.find(local_node_gid);
    //     if (it == node_gid_to_ghosting_ranks.end()) continue;
    //     const std::vector<int> &dest_ranks = it->second;
    //     for (int rr : dest_ranks) {
    //         if (rr == rank) continue;
    //         boundary_node_targets[node_lid].push_back(std::make_pair(rr, local_node_gid));
    //     }
    // }
    
    // std::cout.flush();
    // MPI_Barrier(MPI_COMM_WORLD);
    // print_info = false;
    
    // // Optional: print a compact summary of node reverse map for verification (limited output)
    // for(int i = 0; i < world_size; i++) {
    //     if (rank == i && print_info) {
    //         std::cout << std::endl;
    //         for (int node_lid = 0; node_lid < intermediate_mesh.num_nodes; node_lid++) {
                
    //             size_t local_node_gid = intermediate_mesh.local_to_global_node_mapping.host(node_lid);
    //             if (boundary_node_targets[node_lid].empty()) 
    //             {
    //                 std::cout << "[rank " << rank << "] " << "node_lid: "<< node_lid <<" -  node_gid: " << local_node_gid << " sends to: no ghost nodes" << std::endl;
    //             }
    //             else
    //             {
    //                 std::cout << "[rank " << rank << "] " << "node_lid: "<< node_lid <<" -  node_gid: " << local_node_gid << " sends to: ";
    //                 int shown = 0;
    //                 for (const auto &pr : boundary_node_targets[node_lid]) {
    //                     if (shown >= 12) { std::cout << " ..."; break; }
    //                     std::cout << "(r" << pr.first << ":gid " << pr.second << ") ";
    //                     shown++;
    //                 }
    //                 std::cout << std::endl;
    //             }
    //         }
    //         std::cout.flush();
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    
    // print_info = false;
    
    // MPI_Barrier(MPI_COMM_WORLD);
    // if(rank == 0) std::cout << " Finished building node communication reverse map" << std::endl;




}




#endif