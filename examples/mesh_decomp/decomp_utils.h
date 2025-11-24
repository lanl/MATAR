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
#include <unordered_set>


#include "mesh.h"
#include "state.h"
#include "mesh_io.h"
#include "communication_plan.h"


// Include Scotch headers
#include "scotch.h"
#include "ptscotch.h"

/**
 * @brief Partitions the input mesh into a naive element-based decomposition across MPI ranks.
 *
 * This function splits the input mesh (and its associated node information) evenly among the given number of MPI ranks.
 * It assigns contiguous blocks of elements (and the corresponding nodes and nodal data) to each rank.
 * 
 * The function constructs:
 * - The sub-mesh (naive_mesh) and its nodes (naive_node) for the local rank.
 * - Maps and vectors indicating elements and nodes present on each rank.
 * - Auxiliary arrays (elems_in_elem_on_rank, num_elems_in_elem_per_rank) for local element connectivity and neighbor look-ups.
 *
 * The decomposition is "naive" in that it uses a simple contiguous block assignment, without regard to mesh topology or quality of partitioning.
 * This function is generally used as the preliminary step before repartitioning with tools like PT-Scotch or for algorithm prototyping.
 *
 * @param initial_mesh[in]         The input mesh containing all elements/nodes on rank 0.
 * @param initial_node[in]         The nodal data for the input mesh on rank 0.
 * @param naive_mesh[out]          The mesh on this rank after naive partitioning.
 * @param naive_node[out]          The nodal data on this rank after naive partitioning.
 * @param elems_in_elem_on_rank[out]   Vector of element-to-element connectivity for this rank's local mesh.
 * @param num_elems_in_elem_per_rank[out] Vector of counts for element neighbors for each local element.
 * @param world_size[in]           Number of MPI ranks (world size).
 * @param rank[in]                 This MPI rank's id.
 */

void naive_partition_mesh(
    Mesh_t& initial_mesh,
    node_t& initial_node,
    Mesh_t& naive_mesh,
    node_t& naive_node,
    CArrayDual<int>& elems_in_elem_on_rank,
    CArrayDual<int>& num_elems_in_elem_per_rank,
    int world_size,
    int rank)
{

    bool print_info = false;

    int num_elements_on_rank = 0;
    int num_nodes_on_rank = 0;
    int num_nodes_per_elem = 0;
    int num_dim = initial_mesh.num_dims;


    // Compute the number of elements to send to each rank and num_nodes_per_elem
    std::vector<int> elems_per_rank(world_size); // number of elements to send to each rank size(world_size)
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

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(elems_per_rank.data(), 1, MPI_INT, 
                &num_elements_on_rank, 1, MPI_INT, 
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);


    // Vector of element to send to each rank using a naive partitioning (0-m, m-n, n-o, etc.)
    std::vector<int> elements_on_rank(num_elements_on_rank);  


    // ********************************************************  
    //     Scatter the actual element global ids to each rank
    // ******************************************************** 

    // create a 2D vector of elements to send to each rank
    std::vector<std::vector<int>> elements_to_send(world_size);
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
    std::vector<int> nodes_per_rank(world_size); // number of nodes to send to each rank size(world_size)
    std::vector<int> nodes_on_rank; // node gids the current rank
    std::vector<std::vector<int>> nodes_to_send(world_size); // nodes to send to each rank

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
    }

    // Send the number of nodes to each rank using MPI_scatter
    MPI_Scatter(nodes_per_rank.data(), 1, MPI_INT, &num_nodes_on_rank, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);

    // resize the nodes_on_rank vector to hold the received data
    nodes_on_rank.resize(num_nodes_on_rank);

    MPI_Barrier(MPI_COMM_WORLD);

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
    // Create a flat 1D vector for node positions (num_dim coordinates per node)
    std::vector<double> node_pos_on_rank_flat(num_nodes_on_rank * num_dim);
    CArrayDual<double> node_pos_on_rank(num_nodes_on_rank, num_dim, "node_pos_on_rank_decomp");
    
    if(rank == 0){

        // Prepare data for MPI_Scatterv (scatter with variable counts)
        // Flatten the 2D node_pos_to_send into a 1D array
        std::vector<double> all_node_pos;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = nodes_to_send[i].size() * num_dim;
            displs[i] = displacement; // displacement is the starting index of the nodes for the current rank in the flattened array
            // Copy node positions for rank i to the flattened array
            for(int node_gid = 0; node_gid < nodes_to_send[i].size(); node_gid++) {
                for(int dim = 0; dim < num_dim; dim++) {
                    all_node_pos.push_back(initial_node.coords.host(nodes_to_send[i][node_gid], dim));
                }
            }
            displacement += nodes_to_send[i].size() * num_dim;
        }   

        // Send the node positions to each rank
        MPI_Scatterv(all_node_pos.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                     node_pos_on_rank.host_pointer(), num_nodes_on_rank * num_dim, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     node_pos_on_rank.host_pointer(), num_nodes_on_rank * num_dim, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    node_pos_on_rank.update_device();

    // ****************************************************************************************** 
    //     Initialize the node state variables
    // ****************************************************************************************** 

    // initialize node state variables, for now, we just need coordinates, the rest will be initialize by the respective solvers
    std::vector<node_state> required_node_state = { node_state::coords };
    naive_node.initialize(num_nodes_on_rank, num_dim, required_node_state);

    FOR_ALL(node_id, 0, num_nodes_on_rank,
            dim, 0, num_dim,{
        naive_node.coords(node_id, dim) = node_pos_on_rank(node_id, dim);
    });
    MATAR_FENCE();

    naive_node.coords.update_host();

    // ****************************************************************************************** 
    //     Send the element-node connectivity data from the initial mesh to each rank
    // ****************************************************************************************** 

    // Send the element-node connectivity data from the initial mesh to each rank
    std::vector<int> nodes_in_elem_on_rank(num_elements_on_rank * num_nodes_per_elem);

    MPI_Barrier(MPI_COMM_WORLD);
  

    // Instead of staging a full copy of the connectivity data per-rank, compute the
    // scatter counts/displacements directly from the contiguous global array.
    std::vector<int> conn_sendcounts(world_size);
    std::vector<int> conn_displs(world_size);
    int conn_displacement = 0;
    for (int i = 0; i < world_size; i++) {
        conn_sendcounts[i] = elems_per_rank[i] * num_nodes_per_elem;
        conn_displs[i] = conn_displacement;
        conn_displacement += conn_sendcounts[i];
    }

    // Scatter using the native storage type (size_t) and then convert locally to int
    size_t* global_nodes_in_elem = nullptr;
    if (rank == 0) {
        global_nodes_in_elem = initial_mesh.nodes_in_elem.host_pointer();
    }
    MPI_Barrier(MPI_COMM_WORLD);


    { //scope to free memory for tmp vector
        std::vector<size_t> nodes_in_elem_on_rank_size_t(num_elements_on_rank * num_nodes_per_elem);

        MPI_Scatterv(global_nodes_in_elem, conn_sendcounts.data(), conn_displs.data(), MPI_UNSIGNED_LONG_LONG,
                    nodes_in_elem_on_rank_size_t.data(), nodes_in_elem_on_rank_size_t.size(), MPI_UNSIGNED_LONG_LONG,
                    0, MPI_COMM_WORLD);

        for (size_t idx = 0; idx < nodes_in_elem_on_rank_size_t.size(); ++idx) {
            nodes_in_elem_on_rank[idx] = static_cast<int>(nodes_in_elem_on_rank_size_t[idx]);
        }
    }

    // ****************************************************************************************** 
    //     Send the element-element connectivity data from the initial mesh to each rank
    // ****************************************************************************************** 

    // First, rank 0 computes how many connectivity entries each rank will receive
    // and scatters that information
    int total_elem_elem_entries = 0;

    std::vector<int> elem_elem_counts(world_size);
    
    if (rank == 0){

        DCArrayKokkos<size_t> tmp_num_elems_in_elem(initial_mesh.num_elems, "tmp_elems_in_elem"); 
        FOR_ALL(i, 0, initial_mesh.num_elems, {
            tmp_num_elems_in_elem(i) = initial_mesh.num_elems_in_elem(i);
        });
        tmp_num_elems_in_elem.update_host();
        MATAR_FENCE();
        // Calculate total number of connectivity entries for each rank
        for(int i = 0; i < world_size; i++) {
            elem_elem_counts[i] = 0;
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                elem_elem_counts[i] += tmp_num_elems_in_elem.host(elements_to_send[i][k]);
            }
        }
    }
    
    // Define total_elem_elem_entries to be the sum of the elem_elem_counts
    // Scatter the counts to each rank
    MPI_Scatter(elem_elem_counts.data(), 1, MPI_INT,
                &total_elem_elem_entries, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout<< " Finished scatter" <<std::endl;

    elems_in_elem_on_rank = CArrayDual<int>(total_elem_elem_entries, "elems_in_elem_on_rank");

    // Now scatter the num_elems_in_elem for each element on each rank
    num_elems_in_elem_per_rank = CArrayDual<int>(num_elements_on_rank, "num_elems_in_elem_per_rank");
    
    if (rank == 0) {
        std::vector<int> all_num_elems_in_elem;
        std::vector<int> displs_ee(world_size);
        int displacement = 0;

        DCArrayKokkos<size_t> tmp_num_elems_in_elem(initial_mesh.num_elems, "tmp_elems_in_elem"); 
        FOR_ALL(i, 0, initial_mesh.num_elems, {
            tmp_num_elems_in_elem(i) = initial_mesh.num_elems_in_elem(i);
        });
        tmp_num_elems_in_elem.update_host();
        MATAR_FENCE();
        
        for(int i = 0; i < world_size; i++) {
            displs_ee[i] = displacement;

            std::cout<< "Rank = "<< i <<std::endl;

            for(int k = 0; k < elements_to_send[i].size(); k++) {
                all_num_elems_in_elem.push_back(tmp_num_elems_in_elem.host(elements_to_send[i][k]));
            }

            std::cout<< " Finished all_num_elem_elem" <<std::endl;
            displacement += elements_to_send[i].size();
        }
        
        MPI_Scatterv(all_num_elems_in_elem.data(), elems_per_rank.data(), displs_ee.data(), MPI_INT,
                     num_elems_in_elem_per_rank.host_pointer(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     num_elems_in_elem_per_rank.host_pointer(), num_elements_on_rank, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    num_elems_in_elem_per_rank.update_device();

    if (rank == 0){

        std::cout<<"Sending connectivity"<<std::endl;
        // Prepare the element-element connectivity data for each rank
        std::vector<int> all_elems_in_elem;
        std::vector<int> sendcounts(world_size);
        std::vector<int> displs(world_size);
        
        int displacement = 0;

        DRaggedRightArrayKokkos<size_t> tmp_elems_in_elem(initial_mesh.num_elems_in_elem, "temp_elem_in_elem");

        FOR_ALL(elem_gid, 0, initial_mesh.num_elems, {
            for (size_t i = 0; i < initial_mesh.num_elems_in_elem(elem_gid); i++) {
                tmp_elems_in_elem(elem_gid, i) = initial_mesh.elems_in_elem(elem_gid, i);
            } // end for i
        });  // end FOR_ALL elems
        MATAR_FENCE();
        tmp_elems_in_elem.update_host();



        DCArrayKokkos<size_t> tmp_num_elems_in_elem(initial_mesh.num_elems, "tmp_elems_in_elem"); 
        FOR_ALL(i, 0, initial_mesh.num_elems, {
            tmp_num_elems_in_elem(i) = initial_mesh.num_elems_in_elem(i);
        });
        MATAR_FENCE();
        tmp_num_elems_in_elem.update_host();
        
        
        for(int i = 0; i < world_size; i++) {
            sendcounts[i] = elem_elem_counts[i];
            displs[i] = displacement;
            
            // Copy element-element connectivity for rank i
            for(int k = 0; k < elements_to_send[i].size(); k++) {
                for(int l = 0; l < tmp_num_elems_in_elem.host(elements_to_send[i][k]); l++) {
                    all_elems_in_elem.push_back(tmp_elems_in_elem.host(elements_to_send[i][k], l));
                }
            }
            displacement += elem_elem_counts[i];
        }

        // Send the element-element connectivity data to each rank using MPI_Scatterv
        MPI_Scatterv(all_elems_in_elem.data(), sendcounts.data(), displs.data(), MPI_INT,
                     elems_in_elem_on_rank.host_pointer(), total_elem_elem_entries, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
                     elems_in_elem_on_rank.host_pointer(), total_elem_elem_entries, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    elems_in_elem_on_rank.update_device();

    MPI_Barrier(MPI_COMM_WORLD);

    // ****************************************************************************************** 
    //     Initialize the naive_mesh data structures for each rank
    // ****************************************************************************************** 
    naive_mesh.initialize_nodes(num_nodes_on_rank);
    naive_mesh.initialize_elems(num_elements_on_rank, num_dim);

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

            // Use binary search to find the local node index for node_gid, local_to_global_node_mapping is sorted
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

/// @brief Builds ghost elements and nodes for distributed mesh decomposition.
///
/// In distributed memory parallel computing with MPI, each rank owns a subset of the mesh.
/// Ghost elements and nodes are copies of elements/nodes from neighboring ranks that share
/// nodes with the locally-owned elements. This function identifies and extracts these ghost
/// entities to enable inter-rank communication and maintain consistency at domain boundaries.
///
/// The algorithm operates in 5 primary steps:
///  1. Gather element ownership information from all ranks using MPI_Allgatherv
///  2. Collect local element-node connectivity for distribution
///  3. Broadcast connectivity information to all ranks via MPI collective operations
///  4. Identify which remote elements touch local elements (by shared nodes)
///  5. Extract the full connectivity data for identified ghost elements and their nodes
///
/// @param[in] input_mesh The locally-owned mesh on this rank containing local elements/nodes
/// @param[out] output_mesh The enriched mesh with ghost elements and nodes added to local mesh
/// @param[in] input_node Node data associated with the input mesh
/// @param[out] output_node Node data extended with ghost nodes
/// @param[in,out] element_communication_plan MPI communication plan specifying which ranks
///                                            exchange element data (populated by this function)
/// @param[in] world_size Total number of MPI ranks
/// @param[in] rank Current MPI rank (process ID)
///
/// @note This is a collective MPI operation - all ranks must call this function together.
/// @note Uses data-oriented programming patterns with device-accessible arrays (MATAR containers)
/// @note Performance: O(n_local_elements * n_nodes_per_element) for local operations,
///                    plus O(n_global_elements) for global MPI collective operations
void build_ghost(
    Mesh_t& input_mesh,
    Mesh_t& output_mesh,
    node_t& input_node,
    node_t& output_node,
    CommunicationPlan& element_communication_plan,
    CommunicationPlan& node_communication_plan,
    int world_size,
    int rank)
{
    bool print_info = false;
    // ****************************************************************************************** 
    //     Build the ghost elements and nodes
    // ================================================================================================**
    //
    // OVERVIEW OF GHOST ELEMENT IDENTIFICATION:
    // ==========================================
    // In distributed memory parallel computing with MPI, each processor (rank) owns a subset of mesh
    // elements. However, to perform computations that depend on element neighbors or to maintain
    // consistency at domain boundaries, we need ghost elements: copies of elements from neighboring
    // ranks that share nodes with our locally-owned elements.
    //
    // This algorithm identifies and extracts ghost element data in 5 steps:
    //  1. Gather ownership information: Which rank owns which elements (via MPI_Allgatherv)
    //  2. Collect local element-node connectivity for distribution
    //  3. Broadcast connectivity to all ranks (via MPI_Allgatherv)
    //  4. Identify which remote elements touch our local elements
    //  5. Extract the full connectivity data for identified ghost elements

    // ========================================================================
    // STEP 1: Gather element ownership information from all ranks
    // ========================================================================
    // In a distributed mesh, each rank owns a subset of elements. To identify
    // ghost elements (elements from other ranks needed by this rank), we need
    // to know which rank owns each element. This section uses MPI collective
    // operations to gather element GID ownership information.
    //
    // MPI COLLECTIVE OPERATIONS EXPLAINED:
    // ====================================
    // - MPI_Barrier: Synchronizes all ranks; waits until all ranks reach this point
    // - MPI_Allgather: Each rank sends one item of data; each rank receives one item from each rank
    //   Input: Each rank provides local data
    //   Output: Every rank has data from every rank in order (rank 0's data, rank 1's data, ...)
    // - MPI_Allgatherv: Like MPI_Allgather but for variable-sized data
    //   Input: Each rank provides data of potentially different sizes
    //   Output: Every rank has all data from all ranks, with displacement arrays specifying where each rank's data goes
    //
    // COMMUNICATION PATTERN VISUALIZATION:
    // Rank 0: elem_count[0] ----> All ranks receive: [elem_count[0], elem_count[1], elem_count[2], ...]
    // Rank 1: elem_count[1] /
    // Rank 2: elem_count[2] /

    int num_dim = input_mesh.num_dims;

    int nodes_per_elem = input_mesh.num_nodes_in_elem;

    // MPI_Allgather: Each rank sends its element count, every rank receives
    // the count from every other rank. Result: elem_counts[r] = number of
    // elements owned by rank r.
    std::vector<int> elem_counts(world_size);
    MPI_Allgather(&input_mesh.num_elems, 1, MPI_INT, elem_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all ranks before proceeding

    // Compute displacements: offset into the global array for each rank's data
    // Example: if elem_counts = [100, 150, 120], then
    // elem_displs = [0, 100, 250] (where each rank's data starts in all_elem_gids)
    std::vector<int> elem_displs(world_size);
    int total_elems = 0;
    for (int r = 0; r < world_size; r++) {
        elem_displs[r] = total_elems;
        total_elems += elem_counts[r];
    }

    // MPI_Allgatherv: Gather variable-sized data from all ranks into one array
    // Each rank contributes its local_to_global_elem_mapping, which maps
    // local element indices to global element GIDs. After this call,
    // all_elem_gids contains ALL element GIDs from all ranks, organized by rank.
    std::vector<size_t> all_elem_gids(total_elems);
    MPI_Allgatherv(input_mesh.local_to_global_elem_mapping.host_pointer(), input_mesh.num_elems, MPI_UNSIGNED_LONG_LONG,
                all_elem_gids.data(), elem_counts.data(), elem_displs.data(), 
                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Build a lookup map: element GID -> owning rank
    // This allows O(log n) lookups to determine which rank owns any given element.
    std::map<size_t, int> elem_gid_to_rank;
    for (int rank_id = 0; rank_id < world_size; rank_id++) {
        for (int i = 0; i < elem_counts[rank_id]; i++) {
            size_t gid = all_elem_gids[elem_displs[rank_id] + i];
            elem_gid_to_rank[gid] = rank_id;
        }
    }

    // ========================================================================
    // STEP 2: Build index sets for local elements and nodes
    // ========================================================================
    std::set<size_t> local_node_gids;
    std::map<size_t, int> global_to_local_node_mapping;  // GID -> local index mapping
    for(int node_rid = 0; node_rid < input_mesh.num_nodes; node_rid++) {
        size_t node_gid = input_mesh.local_to_global_node_mapping.host(node_rid);
        local_node_gids.insert(node_gid);
        global_to_local_node_mapping[node_gid] = node_rid;
    }

    // Build a set of locally-owned element GIDs for quick lookup
    std::set<size_t> local_elem_gids;
    for (int i = 0; i < input_mesh.num_elems; i++) {
        local_elem_gids.insert(input_mesh.local_to_global_elem_mapping.host(i));
    }

    // ========================================================================
    // STEP 3: Exchange element-to-node connectivity via MPI_Allgatherv
    // ========================================================================
    // Build a flattened connectivity array: pairs of (elem_gid, node_gid)
    // Example for 2 elements with 8 nodes each:
    //   elem_node_conn = [elem0_gid, node0, elem0_gid, node1, ..., elem1_gid, node0, ...]
    //
    // This format is chosen because it's easy to serialize and deserialize over MPI,
    // and allows us to reconstruct the full element-node relationships.
    std::vector<size_t> elem_node_conn;
    int local_conn_size = 0;

    // For each locally-owned element, record its GID and all its node GIDs
    for (int lid = 0; lid < input_mesh.num_elems; lid++) {
        size_t elem_gid = input_mesh.local_to_global_elem_mapping.host(lid);
        
        // Access nodes_in_elem[lid][*] to get all nodes in this element
        for (int j = 0; j < input_mesh.num_nodes_in_elem; j++) {
            size_t node_lid = input_mesh.nodes_in_elem.host(lid, j);  // Local index
            size_t node_gid = input_mesh.local_to_global_node_mapping.host(node_lid);  // Global index
            
            elem_node_conn.push_back(elem_gid);
            elem_node_conn.push_back(node_gid);
        }
        local_conn_size += nodes_per_elem * 2;  // Each element contributes (num_nodes_in_elem * 2) size_ts
    }



    // ========================================================================
    // Perform MPI communication to gather connectivity from all ranks
    // ========================================================================
    // Similar to Step 1, we use MPI_Allgatherv to collect all element-node
    // connectivity pairs. This is a two-stage process:
    // 1) Gather the size of each rank's connectivity data
    // 2) Gather the actual connectivity data with proper offsets

    // Stage 1: Gather connectivity sizes from each rank
    // conn_sizes[r] = number of size_t values that rank r will send
    std::vector<int> conn_sizes(world_size);
    MPI_Allgather(&local_conn_size, 1, MPI_INT, conn_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute displacements for the second MPI_Allgatherv call
    // Displcements tell each rank where its data should be placed in the global array
    std::vector<int> conn_displs(world_size);
    int total_conn = 0;
    for (int r = 0; r < world_size; r++) {
        conn_displs[r] = total_conn;
        total_conn += conn_sizes[r];
    }

    // Stage 2: Gather all element-node connectivity data
    // After this call, all_conn contains the flattened connectivity from every rank,
    // organized by rank. Access data from rank r using indices [conn_displs[r], conn_displs[r] + conn_sizes[r])
    std::vector<size_t> all_conn(total_conn);
    MPI_Allgatherv(elem_node_conn.data(), local_conn_size, MPI_UNSIGNED_LONG_LONG,
                all_conn.data(), conn_sizes.data(), conn_displs.data(),
                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // STEP 4: Identify ghost elements
    // ========================================================================
    // A ghost element is an element owned by another rank that shares at least
    // one node with our locally-owned elements. This step identifies all such elements.

    
    // We use a set to eliminate duplicates (same ghost element might share multiple nodes with us)
    std::set<size_t> ghost_elem_gids;
    std::set<size_t> ghost_node_gids;

    std::map<size_t, int> ghost_node_recv_rank;

    // Iterate through connectivity data from each rank (except ourselves)
    for (int r = 0; r < world_size; r++) {
        if (r == rank) continue;  // Skip our own data - we already know our elements
        
        // Parse the connectivity data for rank r
        // Data format: [elem0_gid, node0, elem0_gid, node1, ..., elem1_gid, node0, ...]
        // Each pair is 2 size_ts, so num_pairs = conn_sizes[r] / 2
        int num_pairs = conn_sizes[r] / 2;
        
        for (int i = 0; i < num_pairs; i++) {
            // Offset into all_conn for this pair (elem_gid, node_gid)
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // Check if this node belongs to one of our locally-owned elements
            if (local_node_gids.find(node_gid) != local_node_gids.end()) {
                
                // Check if this element is NOT owned by us (i.e., it's from another rank)
                if (local_elem_gids.find(elem_gid) == local_elem_gids.end()) {
                    // This is a ghost element for us
                    ghost_elem_gids.insert(elem_gid);
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::map<int, std::set<size_t>> ghost_nodes_from_ranks;

    // Iterate through connectivity data from each rank (except ourselves)
    for (int r = 0; r < world_size; r++) {
        if (r == rank) continue;  // Skip our own data - we already know our elements
        
        // Parse the connectivity data for rank r
        // Data format: [elem0_gid, node0, elem0_gid, node1, ..., elem1_gid, node0, ...]
        // Each pair is 2 size_ts, so num_pairs = conn_sizes[r] / 2
        int num_pairs = conn_sizes[r] / 2;
        
        for (int i = 0; i < num_pairs; i++) {
            // Offset into all_conn for this pair (elem_gid, node_gid)
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // Check if this element belongs to one of our ghost elements
            if (ghost_elem_gids.find(elem_gid) != ghost_elem_gids.end()) {
                
                // Check if this node is NOT owned by us (i.e., it's from another rank)
                if (local_node_gids.find(node_gid) == local_node_gids.end()) {
                    // This is a ghost node for us
                    ghost_node_gids.insert(node_gid);
                    ghost_node_recv_rank[node_gid] = r;
                    ghost_nodes_from_ranks[r].insert(node_gid);
                }
            }
        }
    }

    std::set<size_t> shared_nodes; // nodes on MPI rank boundaries
    // Iterate through connectivity data from each rank (except ourselves) to find shared nodes
    for (int r = 0; r < world_size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank) continue;  // Skip our own data - we already know our elements
        
        // Parse the connectivity data for rank r
        // Data format: [elem0_gid, node0, elem0_gid, node1, ..., elem1_gid, node0, ...]
        // Each pair is 2 size_ts, so num_pairs = conn_sizes[r] / 2
        int num_pairs = conn_sizes[r] / 2;
        
        for (int i = 0; i < num_pairs; i++) {
            // Offset into all_conn for this pair (elem_gid, node_gid)
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // Check if this element belongs to one of our ghost elements
            if (ghost_elem_gids.find(elem_gid) != ghost_elem_gids.end()) {
                // If another rank references a node that is also owned by us, it is a shared node
                if (local_node_gids.find(node_gid) != local_node_gids.end()) {
                    shared_nodes.insert(node_gid);
                    
                }
            }
        }
    }

    // Create a vecor of the ranks that this rank will receive data from for ghost nodes
    std::set<int> ghost_node_receive_ranks;
    for (const auto& pair : ghost_node_recv_rank) {
        ghost_node_receive_ranks.insert(pair.second);
    }

    std::vector<int> ghost_node_receive_ranks_vec(ghost_node_receive_ranks.begin(), ghost_node_receive_ranks.end());

    
    // Find which nodes *we own* are ghosted on other ranks, and on which ranks
    // We want: for each of our local nodes, the list of ranks that ghost it
    
    // Map: local_node_gid -> set of remote ranks that ghost this node
    std::map<size_t, std::set<int>> local_node_gid_to_ghosting_ranks;

    std::vector<std::set<size_t>> shared_nodes_on_ranks(world_size);
    
    // Iterate through connectivity from all ranks except ourselves
    for (int r = 0; r < world_size; r++) {
        if (r == rank) continue; // skip our own rank
        
        int num_pairs = conn_sizes[r] / 2;
        for (int i = 0; i < num_pairs; i++) {
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // If this node is owned by us, and remote rank references it, they are ghosting it
            if (local_node_gids.find(node_gid) != local_node_gids.end()) {
                local_node_gid_to_ghosting_ranks[node_gid].insert(r);
                shared_nodes_on_ranks[r].insert(node_gid);
            }
        }
    }

    // Use the map to create a vector of the ranks that this rank will receive data from for ghost nodes
    std::set<int> ghost_node_send_ranks;
    for (const auto& pair : local_node_gid_to_ghosting_ranks) {
        ghost_node_send_ranks.insert(pair.second.begin(), pair.second.end());
    }
    std::vector<int> ghost_node_send_ranks_vec(ghost_node_send_ranks.begin(), ghost_node_send_ranks.end());

    // Store the count of ghost elements for later use
    input_mesh.num_ghost_elems = ghost_elem_gids.size();
    input_mesh.num_ghost_nodes = ghost_node_gids.size();
    MPI_Barrier(MPI_COMM_WORLD);


    // ========================================================================
    // STEP 5: Extract ghost element connectivity
    // ========================================================================
    // Now that we know which elements are ghosts, we need to extract their
    // full node connectivity from all_conn. This allows us to properly construct
    // the extended mesh with ghost elements included.

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout << " Starting to build extended mesh with ghost elements" << std::endl;

    // Build a map: ghost_elem_gid -> vector of node_gids
    // We pre-allocate the vector size to avoid repeated reallocations
    std::map<size_t, std::vector<size_t>> ghost_elem_to_nodes;
    for (const size_t& ghost_gid : ghost_elem_gids) {
        ghost_elem_to_nodes[ghost_gid].reserve(input_mesh.num_nodes_in_elem);
    }

    // ========================================================================
    // Extract nodes for each ghost element from the globally-collected all_conn
    // ========================================================================
    // The all_conn array was populated by MPI_Allgatherv and contains connectivity
    // pairs (elem_gid, node_gid) for all elements from all ranks. We now parse
    // this data to extract the nodes for each ghost element.
    for (int r = 0; r < world_size; r++) {
        if (r == rank) continue;  // Skip our own data - we already have owned element connectivity
        
        // Parse connectivity data for rank r
        int num_pairs = conn_sizes[r] / 2;
        
        for (int i = 0; i < num_pairs; i++) {
            // Calculate offset for this pair: displacement + (pair_index * 2)
            int offset = conn_displs[r] + i * 2;
            size_t elem_gid = all_conn[offset];
            size_t node_gid = all_conn[offset + 1];
            
            // If this element is one of our identified ghost elements, record its node
            auto it = ghost_elem_to_nodes.find(elem_gid);
            if (it != ghost_elem_to_nodes.end()) {
                it->second.push_back(node_gid);
            }
        }
    }

    // ========================================================================
    // Validation: Verify each ghost element has the correct number of nodes
    // ========================================================================
    // This catch detects issues in the MPI communication or parsing logic
    for (auto& pair : ghost_elem_to_nodes) {
        if (pair.second.size() != static_cast<size_t>(input_mesh.num_nodes_in_elem)) {
            std::cerr << "[rank " << rank << "] ERROR: Ghost element " << pair.first 
                    << " has " << pair.second.size() << " nodes, expected " << input_mesh.num_nodes_in_elem << std::endl;
        }
    }

    // Step 2: Build extended node list (owned nodes first, then ghost-only nodes)
    // Start with owned nodes
    std::map<size_t, int> node_gid_to_extended_lid;
    int extended_node_lid = 0;

    // Add all owned nodes
    for (int i = 0; i < input_mesh.num_nodes; i++) {
        size_t node_gid = input_mesh.local_to_global_node_mapping.host(i);
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

    MPI_Barrier(MPI_COMM_WORLD);
    
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
    int total_extended_elems = input_mesh.num_elems + input_mesh.num_ghost_elems;
    std::vector<std::vector<int>> extended_nodes_in_elem(total_extended_elems);

    // Copy owned element connectivity (convert to extended node LIDs)
    for (int lid = 0; lid < input_mesh.num_elems; lid++) {
        extended_nodes_in_elem[lid].reserve(nodes_per_elem);
        for (int j = 0; j < nodes_per_elem; j++) {
            size_t node_lid = input_mesh.nodes_in_elem.host(lid, j);
            size_t node_gid = input_mesh.local_to_global_node_mapping.host(node_lid);
            int ext_lid = node_gid_to_extended_lid[node_gid];
            extended_nodes_in_elem[lid].push_back(ext_lid);
        }
    }

    // Add ghost element connectivity (map ghost node GIDs to extended node LIDs)
    int ghost_elem_ext_lid = input_mesh.num_elems;
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
        for (int r = 0; r < world_size; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                std::cout << "[rank " << rank << "] Finished building extended mesh structure" << std::endl;
                std::cout << "[rank " << rank << "]   - Owned elements: " << input_mesh.num_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Ghost elements: " << ghost_elem_gids.size() << std::endl;
                std::cout << "[rank " << rank << "]   - Total extended elements: " << total_extended_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Owned nodes: " << input_mesh.num_nodes << std::endl;
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
    for (int i = 0; i < input_mesh.num_elems; i++) {
        extended_lid_to_elem_gid[i] = input_mesh.local_to_global_elem_mapping.host(i);
    }

    // Ghost elements (in sorted order)
    for (size_t i = 0; i < ghost_elem_gids_ordered.size(); i++) {
        extended_lid_to_elem_gid[input_mesh.num_elems + i] = ghost_elem_gids_ordered[i];
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


    output_mesh.initialize_nodes(total_extended_nodes);
    output_mesh.initialize_elems(total_extended_elems, 3);
    output_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(total_extended_nodes);
    output_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(total_extended_elems);
    for (int i = 0; i < total_extended_nodes; i++) {
        output_mesh.local_to_global_node_mapping.host(i) = extended_lid_to_node_gid[i];
    }
    for (int i = 0; i < total_extended_elems; i++) {
        output_mesh.local_to_global_elem_mapping.host(i) = extended_lid_to_elem_gid[i];
    }
    output_mesh.local_to_global_node_mapping.update_device();
    output_mesh.local_to_global_elem_mapping.update_device();

    output_mesh.num_ghost_elems = ghost_elem_gids.size();
    output_mesh.num_ghost_nodes = ghost_only_nodes.size();

    output_mesh.num_owned_elems = input_mesh.num_elems;
    output_mesh.num_owned_nodes = input_mesh.num_nodes;

    MPI_Barrier(MPI_COMM_WORLD);
    // rebuild the local element-node connectivity using the local node ids
    // extended_nodes_in_elem already contains extended local node IDs, so we can use them directly
    for(int i = 0; i < total_extended_elems; i++) {
        for(int j = 0; j < nodes_per_elem; j++) {
            output_mesh.nodes_in_elem.host(i, j) = extended_nodes_in_elem[i][j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    output_mesh.nodes_in_elem.update_device();
    output_mesh.build_connectivity();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout << " Finished building final mesh structure" << std::endl;


    // ****************************************************************************************** 
    //     Build the final nodes that include ghost
    // ****************************************************************************************** 


    output_node.initialize(total_extended_nodes, num_dim, {node_state::coords}, node_communication_plan);
    MPI_Barrier(MPI_COMM_WORLD);

    // The goal here is to populate output_node.coords using globally gathered ghost node coordinates,
    // since input_node does not contain ghost node coordinates.
    //
    // Each rank will:
    //  1. Gather coordinates of its owned nodes (from input_node).
    //  2. Use MPI to gather all coordinates for all required (owned + ghost) global node IDs
    //     into a structure mapping global ID -> coordinate.
    //  3. Use this map to fill output_node.coords.

    // 1. Build list of all global node IDs needed on this rank (owned + ghosts)
    std::vector<size_t> all_needed_node_gids(total_extended_nodes);
    for (int i = 0; i < total_extended_nodes; i++) {
        all_needed_node_gids[i] = output_mesh.local_to_global_node_mapping.host(i);
    }

    // 2. Build owned node GIDs and their coordinates
    std::vector<size_t> owned_gids(output_mesh.num_owned_nodes);
    for (int i = 0; i < output_mesh.num_owned_nodes; i++)
        owned_gids[i] = output_mesh.local_to_global_node_mapping.host(i);

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
    for (int r = 0; r < world_size; r++) {
        owned_displs[r] = total_owned;
        total_owned += owned_counts[r];
    }

    // c) Global GIDs (size: total_owned)
    std::vector<size_t> all_owned_gids(total_owned);
    MPI_Allgatherv(owned_gids.data(), local_owned_count, MPI_UNSIGNED_LONG_LONG,
                all_owned_gids.data(), owned_counts.data(), owned_displs.data(),
                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    // Map node gid -> owning rank
    std::unordered_map<size_t, int> node_gid_to_owner_rank;
    int owner_offset = 0;
    for (int r = 0; r < world_size; r++) {
        for (int i = 0; i < owned_counts[r]; i++) {
            node_gid_to_owner_rank[all_owned_gids[owner_offset + i]] = r;
        }
        owner_offset += owned_counts[r];
    }


    // d) Global coords (size: total_owned x 3)
    std::vector<double> owned_coords_send(num_dim*local_owned_count, 0.0);
    for (int i = 0; i < local_owned_count; i++) {
        for(int dim = 0; dim < num_dim; dim++){
            owned_coords_send[num_dim*i+dim] = input_node.coords.host(i,dim);
        }
    }
    std::vector<double> all_owned_coords(num_dim * total_owned, 0.0);

    // Create coordinate-specific counts and displacements (in units of doubles, not nodes)
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout << " Getting coord_counts" << std::endl;

    std::vector<int> coord_counts(world_size);
    std::vector<int> coord_displs(world_size);
    for (int r = 0; r < world_size; r++) {
        coord_counts[r] = num_dim * owned_counts[r];  // Each node has num_dim doubles
        coord_displs[r] = num_dim * owned_displs[r];  // Displacement in doubles
    }

    MPI_Allgatherv(owned_coords_send.data(), num_dim*local_owned_count, MPI_DOUBLE,
                all_owned_coords.data(), coord_counts.data(), coord_displs.data(),
                MPI_DOUBLE, MPI_COMM_WORLD);

    // e) Build map: gid -> coord[3]
    std::unordered_map<size_t, std::vector<double>> gid_to_coord;
    for (int i = 0; i < total_owned; i++) {
        std::vector<double> xyz(num_dim);  // size is runtime-dependent
        for (int dim = 0; dim < num_dim; dim++) {
            xyz[dim] = all_owned_coords[num_dim * i + dim];
        }
        gid_to_coord[all_owned_gids[i]] = std::move(xyz);
    }

    // 4. Finally, fill output_node.coords with correct coordinates.
    for (int i = 0; i < total_extended_nodes; i++) {
        size_t gid = output_mesh.local_to_global_node_mapping.host(i);
        auto it = gid_to_coord.find(gid);
        if (it != gid_to_coord.end()) {
            for (int dim = 0; dim < num_dim; dim++) {
                output_node.coords.host(i,dim) = it->second[dim];
            }
        } else {
            // Could happen if there's a bug: fill with zeros for safety
            for (int dim = 0; dim < num_dim; dim++) {
                output_node.coords.host(i,dim) = 0.0;
            }
        }
    }
    output_node.coords.update_device();


    // --------------------------------------------------------------------------------------
    // Build the send patterns for elements
    // Build reverse map via global IDs: for each local element gid, find ranks that ghost it.
    // Steps:
    // 1) Each rank contributes its ghost element GIDs.
    // 2) Allgatherv ghost GIDs to build gid -> [ranks that ghost it].
    // 3) For each locally-owned element gid, lookup ranks that ghost it and record targets.
    // --------------------------------------------------------------------------------------
    std::vector<std::vector<std::pair<int, size_t>>> boundary_elem_targets(output_mesh.num_owned_elems);

    // Prepare local ghost list as vector
    std::vector<size_t> ghost_gids_vec;
    ghost_gids_vec.reserve(output_mesh.num_ghost_elems);
    for (int i = 0; i < output_mesh.num_ghost_elems; i++) {
        ghost_gids_vec.push_back(output_mesh.local_to_global_elem_mapping.host(output_mesh.num_owned_elems + i)); // Ghost elements are after the owned elements in the global element mapping
    }

    // Exchange counts
    std::vector<int> ghost_counts(world_size, 0);
    int local_ghost_count = output_mesh.num_ghost_elems;
    MPI_Allgather(&local_ghost_count, 1, MPI_INT, ghost_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Displacements and recv buffer
    std::vector<int> ghost_displs(world_size, 0);
    int total_ghosts = 0;
    for (int r = 0; r < world_size; r++) {
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
    for (int r = 0; r < world_size; r++) {
        int cnt = ghost_counts[r];
        int off = ghost_displs[r];
        for (int i = 0; i < cnt; i++) {
            size_t g = all_ghost_gids[off + i];
            gid_to_ghosting_ranks[g].push_back(r);
        }
    }

    // For each local element, list destinations: ranks that ghost our gid
    for (int elem_lid = 0; elem_lid < output_mesh.num_owned_elems; elem_lid++) {
        size_t local_elem_gid = output_mesh.local_to_global_elem_mapping.host(elem_lid);
        auto it = gid_to_ghosting_ranks.find(local_elem_gid);
        if (it == gid_to_ghosting_ranks.end()) continue;
        const std::vector<int> &dest_ranks = it->second;
        for (int rr : dest_ranks) {
            if (rr == rank) continue;
            boundary_elem_targets[elem_lid].push_back(std::make_pair(rr, local_elem_gid));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<"After boundary_elem_targets"<<std::endl;

    // Add a vector to store boundary element local_ids (those who have ghost destinations across ranks)
    std::vector<int> boundary_elem_local_ids;
    std::vector<std::vector<int>> boundary_to_ghost_ranks;  // ragged array dimensions (num_boundary_elems, num_ghost_ranks)

    std::set<int> ghost_comm_ranks; // set of ranks that this rank communicates with


    for (int elem_lid = 0; elem_lid < output_mesh.num_owned_elems; elem_lid++) {

        int local_elem_gid = output_mesh.local_to_global_elem_mapping.host(elem_lid);
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

    output_mesh.num_boundary_elems = boundary_elem_local_ids.size();
    output_mesh.boundary_elem_local_ids = DCArrayKokkos<size_t>(output_mesh.num_boundary_elems, "boundary_elem_local_ids");
    for (int i = 0; i < output_mesh.num_boundary_elems; i++) {
        output_mesh.boundary_elem_local_ids.host(i) = boundary_elem_local_ids[i];
    }
    output_mesh.boundary_elem_local_ids.update_device();

    print_info = false;


    MPI_Barrier(MPI_COMM_WORLD);

    std::map<int, std::set<size_t>> node_set_to_send_by_rank;

    // For each owned element that will be ghosted on other ranks,
    // collect the nodes that need to be sent to those ranks
    // boundary_elem_targets[elem_lid] contains pairs (rank, elem_gid) for ranks that ghost this element
    for (int elem_lid = 0; elem_lid < input_mesh.num_elems; elem_lid++) {
        // Get ranks that will ghost this element
        for (const auto& pair : boundary_elem_targets[elem_lid]) {
            int ghosting_rank = pair.first;
            
            // For each node in this element
            for (int j = 0; j < nodes_per_elem; j++) {
                size_t node_lid = input_mesh.nodes_in_elem.host(elem_lid, j);
                size_t node_gid = input_mesh.local_to_global_node_mapping.host(node_lid);
                
                // Only send nodes that are NOT shared (not on MPI rank boundary)
                // Shared nodes are already known to both ranks
                if (shared_nodes_on_ranks[ghosting_rank].find(node_gid) == shared_nodes_on_ranks[ghosting_rank].end()) { // WARNING: THIS SHOULD BE MOFIFIED TO ONLY FILTER SHARED NODES WITH THIS SPECIFIC RANK
                    node_set_to_send_by_rank[ghosting_rank].insert(node_gid);
                }
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    std::map<int, std::vector<int>> nodes_to_send_by_rank;  // rank -> list of global node indices

    // Copy the node_set_to_send_by_rank map to nodes_to_send_by_rank
    for (const auto& [dest_rank, node_gids] : node_set_to_send_by_rank) {
        for (size_t node_gid : node_gids) {
            nodes_to_send_by_rank[dest_rank].push_back(node_gid);
        }
    }

    // Initialize graph comms for elements    
    // MPI_Dist_graph_create_adjacent creates a distributed graph topology communicator
    // that efficiently represents the communication pattern between ranks.
    // This allows MPI to optimize communication based on the actual connectivity pattern.
    
    
    // ---------- Prepare INCOMING edges (sources) ----------
    // indegree: Number of ranks from which this rank will RECEIVE data
    // These are the ranks that own elements which are ghosted on this rank
    std::vector<int> ghost_elem_receive_ranks_vec(ghost_elem_receive_ranks.begin(), 
                                                    ghost_elem_receive_ranks.end());
    // The number of ranks from which this rank will receive data (incoming neighbors)
    int elem_indegree = static_cast<int>(ghost_elem_receive_ranks_vec.size());
    
    // sources: Array of source rank IDs (ranks we receive from)
    // Each element corresponds to a rank that owns elements we ghost
    int* sources = (elem_indegree > 0) ? ghost_elem_receive_ranks_vec.data() : MPI_UNWEIGHTED;

    
    // sourceweights: Weights on incoming edges (not used here, set to MPI_UNWEIGHTED)
    // Could be used to specify communication volume if needed for optimization
    // int* sourceweights = MPI_UNWEIGHTED;
    
    // ---------- Prepare OUTGOING edges (destinations) ----------
    // outdegree: Number of ranks to which this rank will SEND data
    // These are the ranks that ghost elements owned by this rank
    int outdegree = num_ghost_comm_ranks;
    
    // destinations: Array of destination rank IDs (ranks we send to)
    // Each element corresponds to a rank that ghosts our owned elements
    int* destinations = (outdegree > 0) ? ghost_comm_ranks_vec.data() : MPI_UNWEIGHTED;

    // Initialize the graph communicator for element communication
    element_communication_plan.initialize_graph_communicator(outdegree, ghost_comm_ranks_vec.data(), elem_indegree, ghost_elem_receive_ranks_vec.data());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Optional: Verify the graph communicator was created successfully
    // if(print_info) element_communication_plan.verify_graph_communicator();


    // Initialize graph comms for nodes    
    // ---------- Prepare INCOMING edges (sources) ----------
    // indegree: Number of ranks from which this rank will RECEIVE data
    // These are the ranks that own nodes which are ghosted on this rank
    int node_indegree = static_cast<int>(ghost_node_receive_ranks.size());
    int* node_sources = (node_indegree > 0) ? ghost_node_receive_ranks_vec.data() : MPI_UNWEIGHTED;
    
    // sourceweights: Weights on incoming edges (not used here, set to MPI_UNWEIGHTED)
    //int* node_sourceweights = MPI_UNWEIGHTED;   

    // ---------- Prepare OUTGOING edges (destinations) ----------
    // outdegree: Number of ranks to which this rank will SEND data
    // These are the ranks that ghost nodes owned by this rank
    int node_outdegree = static_cast<int>(ghost_node_send_ranks.size());
    int* node_destinations = (node_outdegree > 0) ? ghost_node_send_ranks_vec.data() : MPI_UNWEIGHTED;

    // destinationweights: Weights on outgoing edges (not used here, set to MPI_UNWEIGHTED)
    // int* node_destinationweights = MPI_UNWEIGHTED;

    // Initialize the graph communicator for node communication
    node_communication_plan.initialize_graph_communicator(node_outdegree, node_destinations, node_indegree, node_sources);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout<<"After node graph communicator"<<std::endl;

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
    
    for (int elem_lid = 0; elem_lid < input_mesh.num_elems; elem_lid++) {
        if (!boundary_elem_targets[elem_lid].empty()) {
            for (const auto &pr : boundary_elem_targets[elem_lid]) {
                int dest_rank = pr.first;
                elems_to_send_by_rank[dest_rank].push_back(elem_lid);
            }
        }
    }

    // Serialize into a DRaggedRightArrayKokkos
    DCArrayKokkos<size_t> strides_array(element_communication_plan.num_send_ranks, "strides_for_elems_to_send");
    for (int i = 0; i < element_communication_plan.num_send_ranks; i++) {
        int dest_rank = element_communication_plan.send_rank_ids.host(i);
        strides_array.host(i) = elems_to_send_by_rank[dest_rank].size();
    }
    strides_array.update_device();
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
        int ghost_elem_local_id = output_mesh.num_owned_elems + i;
        elems_to_recv_by_rank[source_rank].push_back(ghost_elem_local_id);
    }

    // ========== Serialize into a DRaggedRightArrayKokkos ==========
    DCArrayKokkos<size_t> elem_recv_strides_array(element_communication_plan.num_recv_ranks, "elem_recv_strides_array");
    for (int i = 0; i < element_communication_plan.num_recv_ranks; i++) {
        int source_rank = element_communication_plan.recv_rank_ids.host(i);
        elem_recv_strides_array.host(i) = elems_to_recv_by_rank[source_rank].size();
       
    }
    elem_recv_strides_array.update_device();
    DRaggedRightArrayKokkos<int> elems_to_recv_by_rank_rr(elem_recv_strides_array, "elems_to_recv_by_rank");
    // Fill in the data
    for (int i = 0; i < element_communication_plan.num_recv_ranks; i++) {
        int source_rank = element_communication_plan.recv_rank_ids.host(i);
        for (int j = 0; j < elems_to_recv_by_rank[source_rank].size(); j++) {
            elems_to_recv_by_rank_rr.host(i, j) = elems_to_recv_by_rank[source_rank][j];
        }
    }
    elems_to_recv_by_rank_rr.update_device();
    MATAR_FENCE();
    element_communication_plan.setup_send_recv(elems_to_send_by_rank_rr, elems_to_recv_by_rank_rr);

    MPI_Barrier(MPI_COMM_WORLD);

    // --------------------------------------------------------------------------------------
    // Build the send pattern for nodes
    // --------------------------------------------------------------------------------------
    // Build reverse map via global IDs: for each local node gid, find ranks that ghost it.
    // Steps:
    // 1) Each rank contributes its ghost node GIDs.
    // 2) Allgatherv ghost node GIDs to build gid -> [ranks that ghost it].
    // 3) For each locally-owned node gid, lookup ranks that ghost it and record targets.
    // --------------------------------------------------------------------------------------

    // Serialize into a DRaggedRightArrayKokkos
    DCArrayKokkos<size_t> node_send_strides_array(node_communication_plan.num_send_ranks,"node_send_strides_array");
    for (int i = 0; i < node_communication_plan.num_send_ranks; i++) {
        int dest_rank = node_communication_plan.send_rank_ids.host(i);
        node_send_strides_array.host(i) = nodes_to_send_by_rank[dest_rank].size();
    }
    node_send_strides_array.update_device();
    DRaggedRightArrayKokkos<int> nodes_to_send_by_rank_rr(node_send_strides_array, "nodes_to_send_by_rank");

    // Fill in the data
    for (int i = 0; i < node_communication_plan.num_send_ranks; i++) {
        int dest_rank = node_communication_plan.send_rank_ids.host(i);
        for (int j = 0; j < nodes_to_send_by_rank[dest_rank].size(); j++) {
            int node_gid = nodes_to_send_by_rank[dest_rank][j];
            int node_lid = node_gid_to_extended_lid[node_gid];
            nodes_to_send_by_rank_rr.host(i, j) = node_lid;
        }
    }
    nodes_to_send_by_rank_rr.update_device();

    // For each ghost element, determine which nodes need to be received from the owning rank
    // Build the receive list based on ghost element nodes, not on ghost_node_gids
    // This ensures we receive all nodes needed by ghost elements
    std::map<int, std::set<size_t>> node_set_to_recv_by_rank;  // rank -> set of node GIDs to receive
    
    for (int i = 0; i < output_mesh.num_ghost_elems; i++) {
        int ghost_elem_lid = output_mesh.num_owned_elems + i;
        size_t ghost_elem_gid = output_mesh.local_to_global_elem_mapping.host(ghost_elem_lid);
        int owning_rank = elem_gid_to_rank.at(ghost_elem_gid);
        
        // Collect all nodes in this ghost element
        for (int j = 0; j < nodes_per_elem; j++) {
            size_t node_lid = output_mesh.nodes_in_elem.host(ghost_elem_lid, j);
            size_t node_gid = output_mesh.local_to_global_node_mapping.host(node_lid);
            
            // Only receive nodes that:
            // 1. We don't own (not in local_node_gids)
            // 2. Are NOT shared (not on MPI rank boundary)
            // Shared nodes are already known to both ranks via element connectivity
            if (local_node_gids.find(node_gid) == local_node_gids.end() && 
                shared_nodes_on_ranks[owning_rank].find(node_gid) == shared_nodes_on_ranks[owning_rank].end()) {
                node_set_to_recv_by_rank[owning_rank].insert(node_gid);
            }
        }
    }
    
    // Convert node GIDs to local indices and build nodes_to_recv_by_rank
    std::map<int, std::vector<int>> nodes_to_recv_by_rank;  // rank -> list of ghost node local indices
    std::map<size_t, int> node_gid_to_ghost_lid;  // map ghost node GID to its local index in output_mesh
    
    // Build the GID->local index mapping for ALL ghost nodes in output_mesh
    // Ghost nodes are those with local IDs >= num_owned_nodes
    for (int i = output_mesh.num_owned_nodes; i < output_mesh.num_nodes; i++) {
        size_t node_gid = output_mesh.local_to_global_node_mapping.host(i);
        node_gid_to_ghost_lid[node_gid] = i;
    }
    
    // Now convert the GID sets to local index vectors
    for (const auto& pair : node_set_to_recv_by_rank) {
        int source_rank = pair.first;
        const std::set<size_t>& node_gids = pair.second;
        
        for (size_t node_gid : node_gids) {
            auto it = node_gid_to_ghost_lid.find(node_gid);
            if (it != node_gid_to_ghost_lid.end()) {
                nodes_to_recv_by_rank[source_rank].push_back(it->second);
            }
        }
    }
    
    // Serialize into a DRaggedRightArrayKokkos
    DCArrayKokkos<size_t> nodes_recv_strides_array(node_communication_plan.num_recv_ranks, "nodes_recv_strides_array");
    for (int i = 0; i < node_communication_plan.num_recv_ranks; i++) {
        int source_rank = node_communication_plan.recv_rank_ids.host(i);
        nodes_recv_strides_array.host(i) = nodes_to_recv_by_rank[source_rank].size();
    }
    nodes_recv_strides_array.update_device();
    DRaggedRightArrayKokkos<int> nodes_to_recv_by_rank_rr(nodes_recv_strides_array, "nodes_to_recv_by_rank");
    // Fill in the data
    for (int i = 0; i < node_communication_plan.num_recv_ranks; i++) {
        int source_rank = node_communication_plan.recv_rank_ids.host(i);
        for (int j = 0; j < nodes_to_recv_by_rank[source_rank].size(); j++) {
            size_t node_gid = nodes_to_recv_by_rank[source_rank][j];
            size_t local_id = node_gid_to_extended_lid[node_gid];

            nodes_to_recv_by_rank_rr.host(i, j) = nodes_to_recv_by_rank[source_rank][j];
        }
    }
    nodes_to_recv_by_rank_rr.update_device();

    MPI_Barrier(MPI_COMM_WORLD);

    node_communication_plan.setup_send_recv(nodes_to_send_by_rank_rr, nodes_to_recv_by_rank_rr);
    MPI_Barrier(MPI_COMM_WORLD);

    // node_communication_plan.verify_send_recv();

}


/**
 * @brief Partitions the input mesh using PT-Scotch and constructs the final distributed mesh.
 *
 * This function performs parallel mesh partitioning using a two-stage approach:
 *   1. A naive partition is first constructed (simple assignment of mesh elements/nodes across ranks).
 *   2. PT-Scotch is then used to repartition the mesh for load balancing and improved connectivity.
 *
 * The partitioned mesh, nodal data, and associated connectivity/gauss point information
 * are distributed among MPI ranks as a result. The procedure ensures that each rank receives
 * its assigned portion of the mesh and associated data in the final (target) decomposition.
 *
 * @param initial_mesh[in]  The input (global) mesh, present on rank 0 or all ranks at start.
 * @param final_mesh[out]   The mesh assigned to this rank after PT-Scotch decomposition.
 * @param initial_node[in]  Nodal data for the input (global) mesh; must match initial_mesh.
 * @param final_node[out]   Nodal data for this rank after decomposition (corresponds to final_mesh).
 * @param gauss_point[out]  Gauss point data structure, filled out for this rank's mesh.
 * @param world_size[in]    Number of MPI ranks in use (the total number of partitions).
 * @param rank[in]          This process's MPI rank ID.
 *
 * Internals:
 * - The routine uses a naive_partition_mesh() helper to create an initial contiguous mesh partition.
 * - It then uses PT-Scotch distributed graph routines to compute an improved partition and create the final mesh layout.
 * - Both element-to-element and node-to-element connectivity, as well as mapping and ghosting information,
 *   are managed and exchanged across ranks.
 * - MPI routines synchronize and exchange the relevant mesh and nodal data following the computed partition.
 */

void partition_mesh(
    Mesh_t& initial_mesh,
    Mesh_t& final_mesh,
    node_t& initial_node,
    node_t& final_node,
    GaussPoint_t& gauss_point,
    int world_size,
    int rank){

    bool print_info = false;
    // bool print_vtk = false;

    int num_dim = initial_mesh.num_dims;

    // Create mesh, gauss points, and node data structures on each rank
    // This is the initial partitioned mesh
    Mesh_t naive_mesh;
    node_t naive_node;

    // Mesh partitioned by pt-scotch, not including ghost
    Mesh_t intermediate_mesh; 
    node_t intermediate_node;

    // Helper arrays to hold element-element connectivity for naive partitioning that include what would be ghost, without having to build the full mesh
    CArrayDual<int> elems_in_elem_on_rank;
    CArrayDual<int> num_elems_in_elem_per_rank;


    // Perform the naive partitioning of the mesh
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Performing the naive partitioning of the mesh" << std::endl;
    naive_partition_mesh(initial_mesh, initial_node, naive_mesh, naive_node, elems_in_elem_on_rank, num_elems_in_elem_per_rank, world_size, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Begin repartitioning using PT-Scotch" << std::endl;

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
    // edgeloctab holds the flattened list of all neighbors (edges) for all local elements,
    // in a compact CSR (Compressed Sparse Row) format expected by PT-Scotch. Each entry is a global element ID
    // of a neighbor. The edgeloctab array is built incrementally with one entry per element neighbor edge,
    // so we reserve its capacity up front for efficiency.
    //
    // Heuristic: For unstructured 3D hexahedral meshes, a single element can have significantly more neighbors 
    // than in 2D cases. In a fully structured 3D grid, each hexahedral element can have up to 26 neighbors 
    // (since it may touch all surrounding elements along all axes). In unstructured grids, it's possible for some 
    // elements to have even more neighbors due to mesh irregularities and refinements. 
    // 
    // For most practical unstructured hexahedral meshes, values in the low 20s are common, but extreme cases 
    // (e.g., high-order connectivity, pathological splits, or meshes with "hanging nodes") may see higher counts. 
    // Using vertlocnbr * 26 as an upper limit is a reasonable estimate for fully connected (structured) cases, 
    // but consider increasing this if working with highly unstructured or pathological meshes. For safety and 
    // to avoid repeated reallocations during construction, we use 26 here as a conservative guess.
    edgeloctab.reserve(vertlocnbr * 26);

    // Construct a map from element GID to its offset into elems_in_elem_on_rank (the array of neighbor GIDs)
    // This allows, for a given element GID, quick lookup of where its neighbor list starts in the flat array.
    std::map<int, size_t> elem_gid_to_offset;
    size_t current_offset = 0;
    for (size_t k = 0; k < naive_mesh.num_elems; k++) {
        int elem_gid_on_rank = naive_mesh.local_to_global_elem_mapping.host(k);
        elem_gid_to_offset[elem_gid_on_rank] = current_offset;
        current_offset += num_elems_in_elem_per_rank.host(k); 
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
        size_t num_nbrs = num_elems_in_elem_per_rank.host(idx);

        // Append each neighbor (by its GLOBAL elem GID) to edgeloctab
        for (size_t j = 0; j < num_nbrs; j++) {
            size_t neighbor_gid = elems_in_elem_on_rank.host(elems_in_elem_offset + j); // This is a global element ID!
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
    SCOTCH_stratDgraphMapBuild(&stratdat, SCOTCH_STRATQUALITY, world_size, 0, 0.001);

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
//     Build the intermediate mesh (without ghost nodes and elements) from the repartition
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
    for (int r = 0; r < world_size; r++)
        sendcounts[r] = static_cast<int>(elems_to_send[r].size());

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> sdispls(world_size), rdispls(world_size);
    int send_total = 0, recv_total = 0;
    for (int r = 0; r < world_size; r++) {
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
    for (int r = 0; r < world_size; r++)
        send_elems.insert(send_elems.end(), elems_to_send[r].begin(), elems_to_send[r].end());

    // new_elem_gids: receives the list of new element global IDs this rank will own after the exchange.
    // It is filled after MPI_Alltoallv completes, and contains the GIDs for the elements new to (or remained on) this rank.
    std::vector<int> new_elem_gids(recv_total);
    MPI_Alltoallv(send_elems.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                new_elem_gids.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // New elements owned by this rank
    int num_new_elems = static_cast<int>(new_elem_gids.size());
    
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
    for (int r = 0; r < world_size; r++) {
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
    for (int r = 0; r < world_size; r++) {
        for (int gid : elems_to_send[r]) {
            int lid = -1;
            for (int i = 0; i < naive_mesh.num_elems; i++)
                if (naive_mesh.local_to_global_elem_mapping.host(i) == gid) { lid = i; break; }

            for (int j = 0; j < nodes_per_elem; j++) {
                int node_lid = naive_mesh.nodes_in_elem.host(lid, j);
                int node_gid = naive_mesh.local_to_global_node_mapping.host(node_lid);

                for(int dim = 0; dim < num_dim; dim++) {
                    node_coords_sendbuf.push_back(naive_node.coords.host(node_lid, dim));
                }
            }
        }
    }

    // Each node is 3 doubles; same sendcounts scaling applies
    std::vector<int> coord_sendcounts(world_size), coord_recvcounts(world_size);
    for (int r = 0; r < world_size; r++)
        coord_sendcounts[r] = sendcounts[r] * nodes_per_elem * 3;

    MPI_Alltoall(coord_sendcounts.data(), 1, MPI_INT, coord_recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished exchanging node coordinates counts"<<std::endl;

    std::vector<int> coord_sdispls(world_size), coord_rdispls(world_size);
    int coord_send_total = 0, coord_recv_total = 0;
    for (int r = 0; r < world_size; r++) {
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
    intermediate_mesh.local_to_global_node_mapping = DCArrayKokkos<size_t>(num_new_nodes, "intermediate_mesh.local_to_global_node_mapping");
    intermediate_mesh.local_to_global_elem_mapping = DCArrayKokkos<size_t>(num_new_elems, "intermediate_mesh.local_to_global_elem_mapping");

    // Fill global mappings
    for (int i = 0; i < num_new_nodes; i++)
        intermediate_mesh.local_to_global_node_mapping.host(i) = new_node_gids[i];
    for (int i = 0; i < num_new_elems; i++)
        intermediate_mesh.local_to_global_elem_mapping.host(i) = new_elem_gids[i];

    intermediate_mesh.local_to_global_node_mapping.update_device();
    intermediate_mesh.local_to_global_elem_mapping.update_device();

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

    CommunicationPlan element_communication_plan;
    element_communication_plan.initialize(MPI_COMM_WORLD);
    
    CommunicationPlan node_communication_plan;
    node_communication_plan.initialize(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Starting the ghost element and node construction"<<std::endl;

    build_ghost(intermediate_mesh, final_mesh, intermediate_node, final_node, element_communication_plan, node_communication_plan, world_size, rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<" Finished the ghost element and node construction"<<std::endl;
    

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
        gauss_point.fields_vec.host(i, 0) = -100.0;
        gauss_point.fields_vec.host(i, 1) = -100.0;
        gauss_point.fields_vec.host(i, 2) = -100.0;
    }
    gauss_point.fields.update_device();
    gauss_point.fields_vec.update_device();

    MPI_Barrier(MPI_COMM_WORLD);
    
    gauss_point.fields.communicate();
    gauss_point.fields_vec.communicate();

    MPI_Barrier(MPI_COMM_WORLD);

    CArrayKokkos <double> tmp(final_mesh.num_elems);
    
    // Loop over all elements and average the values of elements connected to that element
    FOR_ALL(i, 0, final_mesh.num_elems, {
        double value = 0.0;
        for (int j = 0; j < final_mesh.num_elems_in_elem(i); j++) {
            value += gauss_point.fields(final_mesh.elems_in_elem(i, j));
        }
        value /= final_mesh.num_elems_in_elem(i);

        tmp(i) = value;
        

        value = 0.0;
        for (int j = 0; j < final_mesh.num_elems_in_elem(i); j++) {
            value += gauss_point.fields_vec(final_mesh.elems_in_elem(i, j), 0);
        }
        value /= final_mesh.num_elems_in_elem(i);
        gauss_point.fields_vec(i, 0) = value;
        gauss_point.fields_vec(i, 1) = value;
        gauss_point.fields_vec(i, 2) = value;
    });
    MATAR_FENCE();

    FOR_ALL(i, 0, final_mesh.num_elems, {
        gauss_point.fields(i) = tmp(i);
    });
    MATAR_FENCE();

    gauss_point.fields.update_host();
    gauss_point.fields_vec.update_host();



    // Test node communication using MPI_Neighbor_alltoallv
    std::vector<node_state> node_states = {node_state::coords, node_state::scalar_field, node_state::vector_field};
    final_node.initialize(final_mesh.num_nodes, 3, node_states, node_communication_plan);

    for (int r = 0; r < world_size; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                std::cout << "[rank " << rank << "] Finished building extended mesh structure" << std::endl;
                std::cout << "[rank " << rank << "]   - Owned elements: " << final_mesh.num_owned_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Ghost elements: " << final_mesh.num_elems - final_mesh.num_owned_elems << std::endl;
                std::cout << "[rank " << rank << "]   - Owned nodes: " << final_mesh.num_owned_nodes << std::endl;
                std::cout << "[rank " << rank << "]   - Ghost-only nodes: " << final_mesh.num_nodes - final_mesh.num_owned_nodes << std::endl;
                std::cout << std::flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    
    for (int i = 0; i < final_mesh.num_owned_nodes; i++) {
        final_node.scalar_field.host(i) = static_cast<double>(rank);
        final_node.vector_field.host(i, 0) = static_cast<double>(rank);
        final_node.vector_field.host(i, 1) = static_cast<double>(rank);
        final_node.vector_field.host(i, 2) = static_cast<double>(rank);
    }
    for (int i = final_mesh.num_owned_nodes; i < final_mesh.num_nodes; i++) {
        final_node.scalar_field.host(i) = -100.0;
        final_node.vector_field.host(i, 0) = -100.0;
        final_node.vector_field.host(i, 1) = -100.0;
        final_node.vector_field.host(i, 2) = -100.0;
    }

    final_node.coords.update_device();
    final_node.scalar_field.update_device();
    final_node.vector_field.update_device();
    MATAR_FENCE();
    MPI_Barrier(MPI_COMM_WORLD);

    node_communication_plan.verify_graph_communicator();

    final_node.scalar_field.communicate();
    // final_node.vector_field.communicate();
    MPI_Barrier(MPI_COMM_WORLD);


    // Update scalar field to visualize the communication

    CArrayKokkos <double> tmp_too(final_mesh.num_elems);
    FOR_ALL(i, 0, final_mesh.num_elems, {

        double value = 0.0;
        for(int j = 0; j < final_mesh.num_nodes_in_elem; j++) {
            value += final_node.scalar_field(final_mesh.nodes_in_elem(i, j));
        }
        value /= final_mesh.num_nodes_in_elem;
        tmp_too(i) = value;
    });
    MATAR_FENCE();

    FOR_ALL(i, 0, final_mesh.num_elems, {
        for(int j = 0; j < final_mesh.num_nodes_in_elem; j++) {
            final_node.scalar_field(final_mesh.nodes_in_elem(i, j)) = tmp_too(i);
        }
    });
    MATAR_FENCE();

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)std::cout<<"Print from rank 0"<<std::endl;
    if(rank == 1)std::cout<<"Print from rank 1"<<std::endl;

    MATAR_FENCE();
    final_node.scalar_field.update_host();
    MATAR_FENCE();
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // DECOMP_UTILS_H