#ifndef COMMUNICATION_PLAN_H
#define COMMUNICATION_PLAN_H

#ifdef HAVE_MPI
#include <mpi.h>
#include "matar.h"

#include <set>

using namespace mtr;

/**
 * @struct CommunicationPlan
 * @brief Manages efficient MPI communication for ghost element and node data exchange
 * 
 * Pure data-oriented design with only flat, contiguous arrays for maximum cache efficiency.
 * Designed to be embedded in distributed data structures for automatic ghost synchronization.
 * 
 * Usage pattern in distributed structures:
 *   node.velocity.comm()  -> automatically syncs ghost nodes
 *   elem.density.comm()   -> automatically syncs ghost elements
 * 
 */
enum class communication_plan_type {
    no_communication,
    all_to_all_graph
};


 struct CommunicationPlan {
    
    // ========================================================================
    // Metadata for MPI neighbor graph communication 
    // ========================================================================

    communication_plan_type comm_type = communication_plan_type::no_communication;

    // MPI world communicator
    MPI_Comm mpi_comm_world;
    bool has_comm_world = false;
    int world_size = -1;

    // MPI graph communicator
    MPI_Comm mpi_comm_graph;
    bool has_comm_graph = false;

    // Number of send and recv ranks
    int num_send_ranks;  // In MPI language, this is the outdegree of the graph communicator
    int num_recv_ranks;  // In MPI language, this is the indegree of the graph communicator

    // Rank IDs for send and recv ranks
    DCArrayKokkos<int> send_rank_ids;  // [size: num_send_ranks] Destination rank IDs
    DCArrayKokkos<int> recv_rank_ids;  // [size: num_recv_ranks] Source rank IDs

    // recv_weights: Weights on incoming edges (not used here, set to MPI_UNWEIGHTED)
    // Could be used to specify communication volume if needed for optimization
    int* recv_weights = MPI_UNWEIGHTED; // [size: num_recv_ranks] Weights on incoming edges, set to MPI_UNWEIGHTED if not used
    
    // send_weights: Weights on outgoing edges (not used here, set to MPI_UNWEIGHTED)
    // Could be used to specify communication volume if needed for optimization
    int* send_weights = MPI_UNWEIGHTED; // [size: num_send_ranks] Weights on outgoing edges, set to MPI_UNWEIGHTED if not used
    
    // info: Hints for optimization (MPI_INFO_NULL means use defaults)
    MPI_Info info = MPI_INFO_NULL;
    
    // reorder: Whether to allow MPI to reorder ranks for optimization (0=no reordering)
    // Setting to 0 preserves original rank numbering
    // Note: In the future, we may want to allow MPI to reorder ranks for optimization by setting to 1, 
    // this would allow MPI to reorder the ranks to make them physically closer on the hardware. 
    // This is a good optimization for large meshes, but will require maps from MPI_comm_world rank IDs to the new reordered rank IDs.
    int reorder = 0; 

    DRaggedRightArrayKokkos<int> send_indices_; // [size: num_send_ranks, num_items_to_send_per_rank] Indices of items to send to each rank
    DRaggedRightArrayKokkos<int> recv_indices_; // [size: num_recv_ranks, num_items_to_recv_per_rank] Indices of items to receive from each rank

    DCArrayKokkos<int> send_counts_; // [size: num_send_ranks] Number of items to send to each rank
    DCArrayKokkos<int> recv_counts_; // [size: num_recv_ranks] Number of items to receive from each rank
    
    
    DCArrayKokkos<int> send_displs_; // [size: num_send_ranks] Starting index of items to send to each rank
    DCArrayKokkos<int> recv_displs_; // [size: num_recv_ranks] Starting index of items to receive from each rank

    int total_send_count;   // Total number of items to send
    int total_recv_count;   // Total number of items to receive

    // ========================================================================
    // CONSTRUCTOR / INITIALIZATION
    // ========================================================================
    
    CommunicationPlan() 
        : num_send_ranks(0), num_recv_ranks(0),
          has_comm_graph(false) {}
    
    
    // Destructor to free MPI resources
    ~CommunicationPlan() {
        // Free graph communicator
        if (has_comm_graph && mpi_comm_graph != MPI_COMM_NULL) {
            MPI_Comm_free(&mpi_comm_graph);
        }
    }
    
    
    void initialize(MPI_Comm comm_world){
        this->mpi_comm_world = comm_world;
        has_comm_world = true;
        MPI_Comm_size(comm_world, &world_size);
    }
    
    /**
     * @brief Initialize an MPI distributed graph communicator for sparse neighbor communication.
     *
     * This function creates an MPI "dist graph communicator" tailored to the sparse data exchange
     * patterns typical in mesh-based parallel applications. It establishes direct knowledge for MPI
     * about which processes (ranks) each process will communicate with. This improves the efficiency 
     * and clarity of later communication (for example, with MPI_Neighbor_alltoallv).
     *
     * This function is especially useful when the communication pattern is not all-to-all, but rather
     * a sparse subset: for instance, where each process only exchanges data with a few neighbors.
     *
     * ==== Key Concepts ====
     * - MPI Communicator:  An MPI object representing a group of processes that can communicate with each other.
     *   For context, "MPI_COMM_WORLD" is a communicator including all processes, but a graph communicator
     *   customizes direct process connections.
     * - Rank:              Integer ID identifying a process in a communicator.
     * - Distributed Graph: MPI can represent communication as a directed sparse graph, with edges from
     *   this rank to those it needs to send to, and from those it will receive from.
     *
     * ==== Parameters ====
     * @param num_send_ranks   [in] Number of ranks this process will send data to (out-neighbors).
     * @param send_rank_ids    [in] Array of size num_send_ranks; each entry is the rank of a process to send to.
     * @param num_recv_ranks   [in] Number of ranks this process will receive data from (in-neighbors).
     * @param recv_rank_ids    [in] Array of size num_recv_ranks; each entry is the rank of a process to receive from.
     *
     * ==== Steps ====
     *
     * 1. Checks if the basic communicator has been initialized.
     *    Throws an error if it has not.
     *
     * 2. Stores the send/receive neighbor counts and rank lists internally.
     *    Copies the IDs into the internal device-host arrays.
     *      - send_rank_ids: process IDs that will be destinations for outgoing messages.
     *      - recv_rank_ids: process IDs that will provide incoming messages.
     *
     * 3. Calls MPI_Dist_graph_create_adjacent:
     *    This constructs a new MPI communicator ("mpi_comm_graph") that encodes this process's
     *    inbound and outbound neighbors. MPI uses this to optimize and route messages directly
     *    and efficiently during later neighbor collectives.
     *
     *    - Note: The 'recv_weights' and 'send_weights' arguments are set to NULL here;
     *            this means we are not giving extra weighting or priorities to any connection.
     *    - The 'reorder' argument (set to 0 in this class) disables rank reordering;
     *      this ensures the assignment of process ranks is preserved, which is often needed
     *      for mapping data or results back to physical entities.
     *    - On return, 'mpi_comm_graph' will allow use of "neighbor" collectives (MPI_Neighbor_alltoall[v], etc.),
     *      which automatically use the provided topology to send/receive to only neighbors efficiently.
     *
     * 4. Marks the internal flag indicating that the graph communicator has been set up ("has_comm_graph").
     *
     * ==== Example Usage ====
     * Suppose rank 0 will send to ranks 1 and 2, and receive from rank 3 only:
     *    int send_ranks[2] = {1, 2};
     *    int recv_ranks[1] = {3};
     *    initialize_graph_communicator(2, send_ranks, 1, recv_ranks);
     *
     * ==== Why Use This? ====
     * - This avoids the need to do manual pairwise MPI_Send/MPI_Recv in your code, 
     *   and enables the use of neighbor collectives -- concise, scalable, and hard-to-get-wrong.
     * - It explicitly tells MPI only about your neighbors, so it can optimize routes and memory.
     * - If you have a large number of processes or a mesh/network with only local coupling,
     *   this approach scales much better than using global/all-to-all communication.
     *
     * @throws std::runtime_error if the base communicator has not been initialized.
     */
    void initialize_graph_communicator(int num_send_ranks, int* send_rank_ids, int num_recv_ranks, int* recv_rank_ids){
        
        this->comm_type = communication_plan_type::all_to_all_graph;
        // Check if the MPI_COMM_WORLD communicator has been initialized.
        if(!has_comm_world){
            throw std::runtime_error("MPI communicator for the world has not been initialized");
        }
        
        // Store the number of outbound and inbound neighbors
        this->num_send_ranks = num_send_ranks;
        this->num_recv_ranks = num_recv_ranks;
        
        // Copy and store send neighbor IDs (out-bound neighbors: where we will send data to)
        this->send_rank_ids = DCArrayKokkos<int>(num_send_ranks, "send_rank_ids");
        for(int i = 0; i < num_send_ranks; i++){
            this->send_rank_ids(i) = send_rank_ids[i];
        }

        // Copy and store receive neighbor IDs (in-bound neighbors: where we will receive data from)
        this->recv_rank_ids = DCArrayKokkos<int>(num_recv_ranks, "recv_rank_ids");
        for(int i = 0; i < num_recv_ranks; i++){
            this->recv_rank_ids(i) = recv_rank_ids[i];
        }
        
        // Create the distributed graph communicator.
        // This call links this process to its explicit send and receive neighbors.
        // See https://www.open-mpi.org/doc/v4.0/man3/MPI_Dist_graph_create_adjacent.3.php for more details.
        MPI_Dist_graph_create_adjacent(
            mpi_comm_world,                                       // Existing communicator (usually MPI_COMM_WORLD)
            num_recv_ranks,                                       // Number of in-neighbors (recv)
            this->recv_rank_ids.host_pointer(),                   // Array of in-neighbor ranks (who we receive from)
            recv_weights,                                         // Edge weights (NULL = unweighted)
            num_send_ranks,                                       // Number of out-neighbors (send)
            this->send_rank_ids.host_pointer(),                   // Array of out-neighbor ranks (who we send to)
            send_weights,                                         // Edge weights (NULL = unweighted)
            info,                                                 // Additional info for MPI (not used, set to MPI_INFO_NULL)
            reorder,                                              // Allow MPI to reorder ranks for performance (0 disables)
            &mpi_comm_graph                                       // [out] New graph communicator
        );

        // Set the internal flag indicating that we have created the MPI distributed graph communicator.
        has_comm_graph = true;
    }

    void verify_graph_communicator(){
        if(!has_comm_graph){
            throw std::runtime_error("MPI graph communicator has not been initialized");
        }

        // ============================================================================
        // Verify the distributed graph communicator
        // ============================================================================
        // Query the graph to verify it matches what we specified
        int indegree_out, outdegree_out, weighted;
        MPI_Dist_graph_neighbors_count(mpi_comm_graph, &indegree_out, &outdegree_out, &weighted);
        
        // Allocate arrays to receive neighbor information
        std::vector<int> sources_out(indegree_out);
        std::vector<int> sourceweights_out(indegree_out);
        std::vector<int> destinations_out(outdegree_out);
        std::vector<int> destweights_out(outdegree_out);
        
        // Retrieve the actual neighbors from the graph communicator
        MPI_Dist_graph_neighbors(mpi_comm_graph, 
                                indegree_out, sources_out.data(), sourceweights_out.data(),
                                outdegree_out, destinations_out.data(), destweights_out.data());
        
        int rank = -1;
        MPI_Comm_rank(mpi_comm_world, &rank);

        // Additional verification: Check if the queried values match our input
        bool verification_passed = true;
        
        // Print verification information for each rank sequentially
        for (int r = 0; r < world_size; ++r) {
            MPI_Barrier(mpi_comm_world);
            if (rank == r) {
                std::cout << "\n[rank " << rank << "] Graph Communicator Verification:" << std::endl;
                std::cout << "  Indegree (receives from " << indegree_out << " ranks): ";
                for (int i = 0; i < indegree_out; ++i) {
                    std::cout << sources_out[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "  Outdegree (sends to " << outdegree_out << " ranks): ";
                for (int i = 0; i < outdegree_out; ++i) {
                    std::cout << destinations_out[i] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "  Weighted: " << (weighted ? "yes" : "no") << std::endl;
            }
            MPI_Barrier(mpi_comm_world);
        }
        
        // Check if the counts match our stored values
        if (indegree_out != num_recv_ranks) {
            std::cerr << "[rank " << rank << "] ERROR: indegree mismatch! "
                      << "Expected " << num_recv_ranks << ", got " << indegree_out << std::endl;
            verification_passed = false;
        }
        if (outdegree_out != num_send_ranks) {
            std::cerr << "[rank " << rank << "] ERROR: outdegree mismatch! "
                      << "Expected " << num_send_ranks << ", got " << outdegree_out << std::endl;
            verification_passed = false;
        }
        
        // Check if source ranks match (build set from our stored recv_rank_ids)
        std::set<int> sources_set_in;
        for (int i = 0; i < num_recv_ranks; ++i) {
            sources_set_in.insert(recv_rank_ids.host(i));
        }
        std::set<int> sources_set_out(sources_out.begin(), sources_out.end());
        if (sources_set_in != sources_set_out) {
            std::cerr << "[rank " << rank << "] ERROR: source ranks mismatch!" << std::endl;
            verification_passed = false;
        }
        
        // Check if destination ranks match (build set from our stored send_rank_ids)
        std::set<int> dests_set_in;
        for (int i = 0; i < num_send_ranks; ++i) {
            dests_set_in.insert(send_rank_ids.host(i));
        }
        std::set<int> dests_set_out(destinations_out.begin(), destinations_out.end());
        if (dests_set_in != dests_set_out) {
            std::cerr << "[rank " << rank << "] ERROR: destination ranks mismatch!" << std::endl;
            verification_passed = false;
        }
        
        // Global verification check
        int local_passed = verification_passed ? 1 : 0;
        int global_passed = 0;
        MPI_Allreduce(&local_passed, &global_passed, 1, MPI_INT, MPI_MIN, mpi_comm_world);
        MPI_Barrier(mpi_comm_world);
        if (rank == 0) {
            if (global_passed) {
                std::cout << "\n✓ Graph communicator verification PASSED on all ranks\n" << std::endl;
            } else {
                std::cout << "\n✗ Graph communicator verification FAILED on one or more ranks\n" << std::endl;
            }
        }
        MPI_Barrier(mpi_comm_world);
    }

    void setup_send_recv(DRaggedRightArrayKokkos<int> &rank_send_ids, DRaggedRightArrayKokkos<int> &rank_recv_ids){

        this->send_indices_ = rank_send_ids; // indices of element data to send to each rank
        this->recv_indices_ = rank_recv_ids; // indices of element data to receive from each rank

        // Setup send data
        this->send_counts_ = DCArrayKokkos<int>(num_send_ranks, "send_counts");
        this->total_send_count = 0;
        for(int i = 0; i < num_send_ranks; i++){
            this->send_counts_.host(i) = rank_send_ids.stride_host(i);
            this->total_send_count += this->send_counts_.host(i);
        }
        this->send_counts_.update_device();

        this->send_displs_ = DCArrayKokkos<int>(num_send_ranks, "send_displs");
        for(int i = 0; i < num_send_ranks; i++){
            this->send_displs_.host(i) = 0;
            for(int j = 0; j < i; j++){
                this->send_displs_.host(i) += this->send_counts_.host(j);
            }
        }
        this->send_displs_.update_device();

        // Setup recv data
        this->recv_counts_ = DCArrayKokkos<int>(num_recv_ranks, "recv_counts");
        this->total_recv_count = 0;
        for(int i = 0; i < num_recv_ranks; i++){
            this->recv_counts_.host(i) = rank_recv_ids.stride_host(i);
            this->total_recv_count += this->recv_counts_.host(i);
        }
        this->recv_counts_.update_device();

        this->recv_displs_ = DCArrayKokkos<int>(num_recv_ranks, "recv_displs");
        for(int i = 0; i < num_recv_ranks; i++){
            this->recv_displs_.host(i) = 0;
            for(int j = 0; j < i; j++){
                this->recv_displs_.host(i) += this->recv_counts_.host(j);
            }
        }
        this->recv_displs_.update_device();

        MPI_Barrier(mpi_comm_world);
    }
}; // End of CommunicationPlan

#endif // end if HAVE_MPI
#endif // end if COMMUNICATION_PLAN_H


