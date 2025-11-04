// /**
//  * @struct CommunicationPlan
//  * @brief Manages efficient MPI communication for ghost element and node data exchange
//  * 
//  * Pure data-oriented design with only flat, contiguous arrays for maximum cache efficiency.
//  * Designed to be embedded in distributed data structures for automatic ghost synchronization.
//  * 
//  * Usage pattern in distributed structures:
//  *   node.velocity.comm()  -> automatically syncs ghost nodes
//  *   elem.density.comm()   -> automatically syncs ghost elements
//  * 
//  */
//  struct CommunicationPlan {
    
//     // ========================================================================
//     // CORE DATA STRUCTURES - FLAT ARRAYS ONLY
//     // ========================================================================


//     // --- Ghost Send Plan: Owned elements/nodes -> destination ranks --- (Works for both elements and nodes)
//     int num_send_ranks;                            // Number of destination ranks
//     DCArrayKokkos<size_t> send_rank_ids;                // [size: num_send_ranks] Destination rank IDs
//     DCArrayKokkos<size_t> send_ghost_offsets;            // [size: num_send_ranks+1] CSR offsets into send_ghost_lids
//     DCArrayKokkos<size_t> send_ghost_lids;               // [size: total_send_ghosts] Local IDs of owned elements/nodes to send
//     std::vector<size_t> send_ghost_gids;            // [size: total_send_ghosts] Global IDs (for debug/validation)
    
//     // --- Ghost Receive Plan: Ghost elements/nodes <- source ranks --- (Works for both elements and nodes)
//     int num_recv_ranks;                            // Number of source ranks
//     DCArrayKokkos<size_t> recv_rank_ids;                // [size: num_recv_ranks] Source rank IDs
//     DCArrayKokkos<size_t> recv_ghost_offsets;            // [size: num_recv_ranks+1] CSR offsets into recv_ghost_lids
//     DCArrayKokkos<size_t> recv_ghost_lids;               // [size: total_recv_ghosts] Local IDs of ghost elements/nodes (>= num_owned)
//     std::vector<size_t> recv_ghost_gids;            // [size: total_recv_ghosts] Global IDs

    
//     DCArrayKokkos<MPI_Request> send_requests;        // Request handles for sends
//     DCArrayKokkos<MPI_Request> recv_requests;        // Request handles for receives
//     DCArrayKokkos<MPI_Status> mpi_statuses;          // Status array for MPI_Waitall
    
//     // --- Persistent communication (optional optimization) ---
//     DCArrayKokkos<MPI_Request> persistent_send_requests;
//     DCArrayKokkos<MPI_Request> persistent_recv_requests;
//     bool has_persistent_comm;
    
    
//     // --- Distributed Graph Topology for Neighborhood Collectives ---
//     MPI_Comm graph_comm;                           // Graph communicator encoding sparse communication pattern
//     bool has_graph_comm;                            // Whether graph communicator is initialized
    
//     // Counts and displacements for MPI_Neighbor_alltoallv
//     DCArrayKokkos<size_t> send_counts;                   // [num_send_ranks] Number of items to send per neighbor
//     DCArrayKokkos<size_t> send_displs;                   // [num_send_ranks] Displacements in send buffer
//     DCArrayKokkos<size_t> recv_counts;                   // [num_recv_ranks] Number of items to recv per neighbor
//     DCArrayKokkos<size_t> recv_displs;                   // [num_recv_ranks] Displacements in recv buffer
    
//     // --- Persistent Neighborhood Collectives (MPI-4.0+) ---
//     MPI_Request persistent_neighbor_request;        // Persistent request for neighborhood collective
//     bool has_persistent_neighbor;                   // Whether persistent neighborhood is initialized
//     int persistent_num_fields;                      // Fields per item for persistent request
    
    
//     // ========================================================================
//     // CONSTRUCTOR / INITIALIZATION
//     // ========================================================================
    
//     CommunicationPlan() 
//         : num_send_ranks(0), num_recv_ranks(0),
//           has_persistent_comm(false),
//           has_graph_comm(false),
//           has_persistent_neighbor(false),
//           graph_comm(MPI_COMM_NULL),
//           persistent_neighbor_request(MPI_REQUEST_NULL),
//           persistent_num_fields(0) {}
    
    
//     // Destructor to free MPI resources
//     ~CommunicationPlan() {
//         // Free persistent neighborhood collective
//         if (has_persistent_neighbor && persistent_neighbor_request != MPI_REQUEST_NULL) {
//             MPI_Request_free(&persistent_neighbor_request);
//         }
        
//         // Free graph communicator
//         if (has_graph_comm && graph_comm != MPI_COMM_NULL) {
//             MPI_Comm_free(&graph_comm);
//         }
//     }
    
    
//     void initialize(int num_send_ranks, int num_recv_ranks){
//         this->num_send_ranks = num_send_ranks;
//         this->num_recv_ranks = num_recv_ranks;
        
//         send_rank_ids = DCArrayKokkos<size_t>(num_send_ranks, "send_rank_ids");
//         recv_rank_ids = DCArrayKokkos<size_t>(num_recv_ranks, "recv_rank_ids");
//         send_ghost_offsets = DCArrayKokkos<size_t>(num_send_ranks + 1, "send_ghost_offsets");
//         recv_ghost_offsets = DCArrayKokkos<size_t>(num_recv_ranks + 1, "recv_ghost_offsets");
//         send_ghost_lids = DCArrayKokkos<size_t>(total_send_ghosts, "send_ghost_lids");
//         recv_ghost_lids = DCArrayKokkos<size_t>(total_recv_ghosts, "recv_ghost_lids");
//         send_ghost_gids = std::vector<size_t>(total_send_ghosts, "send_ghost_gids");
//         recv_ghost_gids = std::vector<size_t>(total_recv_ghosts, "recv_ghost_gids");
//         send_requests = DCArrayKokkos<MPI_Request>(total_send_ghosts, "send_requests");
//         recv_requests = DCArrayKokkos<MPI_Request>(total_recv_ghosts, "recv_requests");
//         mpi_statuses = DCArrayKokkos<MPI_Status>(total_send_ghosts + total_recv_ghosts, "mpi_statuses");
//         persistent_send_requests = DCArrayKokkos<MPI_Request>(total_send_ghosts, "persistent_send_requests");
//         persistent_recv_requests = DCArrayKokkos<MPI_Request>(total_recv_ghosts, "persistent_recv_requests");
//         send_counts = DCArrayKokkos<size_t>(num_send_ranks, "send_counts");
//         send_displs = DCArrayKokkos<size_t>(num_send_ranks, "send_displs");
//         recv_counts = DCArrayKokkos<size_t>(num_recv_ranks, "recv_counts");
//         recv_displs = DCArrayKokkos<size_t>(num_recv_ranks, "recv_displs");
        
//     }
    

    
//     // ========================================================================
//     // INLINE IMPLEMENTATIONS - NEIGHBORHOOD COLLECTIVES
//     // ========================================================================
    
//     /**
//      * @brief Create distributed graph communicator from communication pattern
//      */
//     inline void create_graph_communicator(MPI_Comm base_comm) {
        
//         if (has_graph_comm) {
//             std::cerr << "Warning: Graph communicator already created, skipping." << std::endl;
//             return;
//         }
        
//         int indegree = num_recv_ranks;   // Number of ranks we receive FROM
//         int outdegree = num_send_ranks;  // Number of ranks we send TO
        
//         // Create the distributed graph communicator
//         // MPI_Dist_graph_create_adjacent signature:
//         //   (comm_old, indegree, sources[], sourceweights, outdegree, dests[], destweights,
//         //    info, reorder, comm_dist_graph)
//         int reorder = 0;  // Don't reorder ranks (keep same as base_comm)
        
//         MPI_Dist_graph_create_adjacent(
//             base_comm,                    // Base communicator
//             indegree,                     // We receive from num_recv_ranks neighbors
//             recv_rank_ids.data(),         // Source ranks (we receive from these)
//             MPI_UNWEIGHTED,               // No edge weights for sources
//             outdegree,                    // We send to num_send_ranks neighbors
//             send_rank_ids.data(),         // Destination ranks (we send to these)
//             MPI_UNWEIGHTED,               // No edge weights for destinations
//             MPI_INFO_NULL,                // No special hints
//             reorder,                      // Don't reorder ranks
//             &graph_comm                   // Output: new graph communicator
//         );
        
//         has_graph_comm = true;
        
//         // Pre-allocate counts and displacements arrays
//         send_counts.resize(num_send_ranks);
//         send_displs.resize(num_send_ranks);
//         recv_counts.resize(num_recv_ranks);
//         recv_displs.resize(num_recv_ranks);
//     }
    
    
//     /**
//      * @brief Exchange ghost data using MPI_Neighbor_alltoallv
//      */
//     inline void exchange_ghosts_neighborhood(double* data_ptr, int num_fields) {
        
//         if (!has_graph_comm) {
//             std::cerr << "Error: Must call create_graph_communicator() first!" << std::endl;
//             return;
//         }
        
//         // 1. Pack send buffer from owned items
//         int total_send = send_ghost_lids.size();
//         ghost_send_buffer.resize(total_send * num_fields);
        
//         for (size_t i = 0; i < send_ghost_lids.size(); i++) {
//             int local_id = send_ghost_lids[i];
//             for (int f = 0; f < num_fields; f++) {
//                 ghost_send_buffer[i * num_fields + f] = data_ptr[local_id * num_fields + f];
//             }
//         }
        
//         // 2. Update counts and displacements for this num_fields
//         for (int i = 0; i < num_send_ranks; i++) {
//             int start_idx = send_ghost_offsets[i];
//             int end_idx = send_ghost_offsets[i + 1];
//             send_counts[i] = (end_idx - start_idx) * num_fields;
//             send_displs[i] = start_idx * num_fields;
//         }
        
//         int total_recv = recv_ghost_lids.size();
//         ghost_recv_buffer.resize(total_recv * num_fields);
        
//         for (int i = 0; i < num_recv_ranks; i++) {
//             int start_idx = recv_ghost_offsets[i];
//             int end_idx = recv_ghost_offsets[i + 1];
//             recv_counts[i] = (end_idx - start_idx) * num_fields;
//             recv_displs[i] = start_idx * num_fields;
//         }
        
//         // 3. Execute neighborhood collective (BLOCKING but fast with graph_comm)
//         // MPI_Neighbor_alltoallv signature:
//         //   (sendbuf, sendcounts[], sdispls[], sendtype,
//         //    recvbuf, recvcounts[], rdispls[], recvtype, comm)
//         MPI_Neighbor_alltoallv(
//             ghost_send_buffer.data(),    // Send buffer
//             send_counts.data(),          // Send counts per neighbor
//             send_displs.data(),          // Send displacements
//             MPI_DOUBLE,                  // Send type
//             ghost_recv_buffer.data(),    // Receive buffer
//             recv_counts.data(),          // Receive counts per neighbor
//             recv_displs.data(),          // Receive displacements
//             MPI_DOUBLE,                  // Receive type
//             graph_comm                   // Graph communicator (NOT MPI_COMM_WORLD!)
//         );
        
//         // 4. Unpack receive buffer into ghost items
//         for (size_t i = 0; i < recv_ghost_lids.size(); i++) {
//             int ghost_local_id = recv_ghost_lids[i];
//             for (int f = 0; f < num_fields; f++) {
//                 data_ptr[ghost_local_id * num_fields + f] = ghost_recv_buffer[i * num_fields + f];
//             }
//         }
//     }
    
    
//     /**
//      * @brief Initialize persistent neighborhood collective (MPI-4.0+)
//      */
//     inline void init_persistent_neighborhood(int num_fields) {
        
//         if (!has_graph_comm) {
//             std::cerr << "Error: Must call create_graph_communicator() first!" << std::endl;
//             return;
//         }
        
//         if (has_persistent_neighbor) {
//             std::cerr << "Warning: Persistent neighborhood already initialized, freeing and re-creating." << std::endl;
//             free_persistent_neighborhood();
//         }
        
//         persistent_num_fields = num_fields;
        
//         // Allocate buffers
//         int total_send = send_ghost_lids.size();
//         int total_recv = recv_ghost_lids.size();
//         ghost_send_buffer.resize(total_send * num_fields);
//         ghost_recv_buffer.resize(total_recv * num_fields);
        
//         // Setup counts and displacements for persistent request
//         for (int i = 0; i < num_send_ranks; i++) {
//             int start_idx = send_ghost_offsets[i];
//             int end_idx = send_ghost_offsets[i + 1];
//             send_counts[i] = (end_idx - start_idx) * num_fields;
//             send_displs[i] = start_idx * num_fields;
//         }
        
//         for (int i = 0; i < num_recv_ranks; i++) {
//             int start_idx = recv_ghost_offsets[i];
//             int end_idx = recv_ghost_offsets[i + 1];
//             recv_counts[i] = (end_idx - start_idx) * num_fields;
//             recv_displs[i] = start_idx * num_fields;
//         }
        
// #if MPI_VERSION >= 4
//         // MPI-4.0+ persistent neighborhood collective
//         // MPI_Neighbor_alltoallv_init signature (similar to MPI_Neighbor_alltoallv but creates request):
//         //   (sendbuf, sendcounts[], sdispls[], sendtype,
//         //    recvbuf, recvcounts[], rdispls[], recvtype, comm, info, request)
//         MPI_Neighbor_alltoallv_init(
//             ghost_send_buffer.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
//             ghost_recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
//             graph_comm,
//             MPI_INFO_NULL,
//             &persistent_neighbor_request
//         );
//         has_persistent_neighbor = true;
// #else
//         int rank;
//         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//         if (rank == 0) {
//             std::cerr << "Warning: MPI-4.0 required for persistent neighborhood collectives" << std::endl;
//             std::cerr << "         Detected MPI version: " << MPI_VERSION << "." << MPI_SUBVERSION << std::endl;
//             std::cerr << "         Will fall back to standard neighborhood collective" << std::endl;
//         }
//         has_persistent_neighbor = false;
// #endif
//     }
    
    
//     /**
//      * @brief Exchange ghosts using persistent neighborhood collective (FASTEST)
//      */
//     inline void exchange_ghosts_persistent(double* data_ptr) {
        
// #if MPI_VERSION >= 4
//         if (!has_persistent_neighbor) {
//             std::cerr << "Error: Must call init_persistent_neighborhood() first!" << std::endl;
//             std::cerr << "       Falling back to standard neighborhood collective..." << std::endl;
//             exchange_ghosts_neighborhood(data_ptr, persistent_num_fields);
//             return;
//         }
        
//         // 1. Pack send buffer (same memory location as during init)
//         for (size_t i = 0; i < send_ghost_lids.size(); i++) {
//             int local_id = send_ghost_lids[i];
//             for (int f = 0; f < persistent_num_fields; f++) {
//                 ghost_send_buffer[i * persistent_num_fields + f] = 
//                     data_ptr[local_id * persistent_num_fields + f];
//             }
//         }
        
//         // 2. Start persistent request (VERY fast - no setup overhead)
//         MPI_Start(&persistent_neighbor_request);
        
//         // 3. Wait for completion
//         MPI_Wait(&persistent_neighbor_request, MPI_STATUS_IGNORE);
        
//         // 4. Unpack receive buffer
//         for (size_t i = 0; i < recv_ghost_lids.size(); i++) {
//             int ghost_id = recv_ghost_lids[i];
//             for (int f = 0; f < persistent_num_fields; f++) {
//                 data_ptr[ghost_id * persistent_num_fields + f] = 
//                     ghost_recv_buffer[i * persistent_num_fields + f];
//             }
//         }
// #else
//         // Fallback to standard method if MPI-4 not available
//         exchange_ghosts_neighborhood(data_ptr, persistent_num_fields);
// #endif
//     }
    
    
//     /**
//      * @brief Free persistent neighborhood collective resources
//      */
//     inline void free_persistent_neighborhood() {
// #if MPI_VERSION >= 4
//         if (has_persistent_neighbor && persistent_neighbor_request != MPI_REQUEST_NULL) {
//             MPI_Request_free(&persistent_neighbor_request);
//             persistent_neighbor_request = MPI_REQUEST_NULL;
//             has_persistent_neighbor = false;
//         }
// #endif
//     }
    
// };


