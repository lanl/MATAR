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
 struct CommunicationPlan {
    
    // ========================================================================
    // Metadata for MPI neighbor graph communication 
    // ========================================================================

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
    
    void initialize_graph_communicator(int num_send_ranks, int* send_rank_ids, int num_recv_ranks, int* recv_rank_ids){
        
        if(!has_comm_world){
            throw std::runtime_error("MPI communicator for the world has not been initialized");
        }
        
        this->num_send_ranks = num_send_ranks;
        this->num_recv_ranks = num_recv_ranks;

        this->send_rank_ids = DCArrayKokkos<int>(num_send_ranks, "send_rank_ids");
        for(int i = 0; i < num_send_ranks; i++){
            this->send_rank_ids(i) = send_rank_ids[i];
        }


        this->recv_rank_ids = DCArrayKokkos<int>(num_recv_ranks, "recv_rank_ids");
        for(int i = 0; i < num_recv_ranks; i++){
            this->recv_rank_ids(i) = recv_rank_ids[i];
        }

        MPI_Dist_graph_create_adjacent(
            mpi_comm_world,
            num_recv_ranks,
            this->recv_rank_ids.host_pointer(),
            recv_weights,
            num_send_ranks,
            this->send_rank_ids.host_pointer(),
            send_weights,
            info,
            reorder,
            &mpi_comm_graph
        );

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


};
