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
 * Memory layout philosophy:
 * - Only std::vector<POD types> (int, size_t, double)
 * - CSR-style indexing for variable-length per-rank data
 * - No std::map, std::set, std::pair, or nested containers
 * - Pre-allocated MPI buffers to avoid repeated allocations
 * - Separate element and node communication plans
 */
 struct CommunicationPlan {
    
    // ========================================================================
    // CORE DATA STRUCTURES - FLAT ARRAYS ONLY
    // ========================================================================


    // --- Ghost Send Plan: Owned elements/nodes -> destination ranks --- (Works for both elements and nodes)
    int num_send_ranks;                            // Number of destination ranks
    std::vector<int> send_rank_ids;                // [size: num_send_ranks] Destination rank IDs
    std::vector<int> send_ghost_offsets;            // [size: num_send_ranks+1] CSR offsets into send_ghost_lids
    std::vector<int> send_ghost_lids;               // [size: total_send_ghosts] Local IDs of owned elements/nodes to send
    std::vector<size_t> send_ghost_gids;            // [size: total_send_ghosts] Global IDs (for debug/validation)
    
    // --- Ghost Receive Plan: Ghost elements/nodes <- source ranks --- (Works for both elements and nodes)
    int num_recv_ranks;                            // Number of source ranks
    std::vector<int> recv_rank_ids;                // [size: num_recv_ranks] Source rank IDs
    std::vector<int> recv_ghost_offsets;            // [size: num_recv_ranks+1] CSR offsets into recv_ghost_lids
    std::vector<int> recv_ghost_lids;               // [size: total_recv_ghosts] Local IDs of ghost elements/nodes (>= num_owned)
    std::vector<size_t> recv_ghost_gids;            // [size: total_recv_ghosts] Global IDs

    
    // --- MPI Communication Buffers (pre-allocated, reusable) ---
    std::vector<double> ghost_send_buffer;          // Flat buffer for ghost data
    std::vector<double> ghost_recv_buffer;          // Flat buffer for ghost data
    
    std::vector<MPI_Request> send_requests;        // Request handles for sends
    std::vector<MPI_Request> recv_requests;        // Request handles for receives
    std::vector<MPI_Status> mpi_statuses;          // Status array for MPI_Waitall
    
    // --- Persistent communication (optional optimization) ---
    std::vector<MPI_Request> persistent_send_requests;
    std::vector<MPI_Request> persistent_recv_requests;
    bool has_persistent_comm;
    
    
    // --- Distributed Graph Topology for Neighborhood Collectives ---
    MPI_Comm graph_comm;                           // Graph communicator encoding sparse communication pattern
    bool has_graph_comm;                            // Whether graph communicator is initialized
    
    // Counts and displacements for MPI_Neighbor_alltoallv
    std::vector<int> send_counts;                   // [num_send_ranks] Number of items to send per neighbor
    std::vector<int> send_displs;                   // [num_send_ranks] Displacements in send buffer
    std::vector<int> recv_counts;                   // [num_recv_ranks] Number of items to recv per neighbor
    std::vector<int> recv_displs;                   // [num_recv_ranks] Displacements in recv buffer
    
    // --- Persistent Neighborhood Collectives (MPI-4.0+) ---
    MPI_Request persistent_neighbor_request;        // Persistent request for neighborhood collective
    bool has_persistent_neighbor;                   // Whether persistent neighborhood is initialized
    int persistent_num_fields;                      // Fields per item for persistent request
    
    
    // ========================================================================
    // CONSTRUCTOR / INITIALIZATION
    // ========================================================================
    
    CommunicationPlan() 
        : num_send_ranks(0), num_recv_ranks(0),
          has_persistent_comm(false),
          has_graph_comm(false),
          has_persistent_neighbor(false),
          graph_comm(MPI_COMM_NULL),
          persistent_neighbor_request(MPI_REQUEST_NULL),
          persistent_num_fields(0) {}
    
    
    // Destructor to free MPI resources
    ~CommunicationPlan() {
        // Free persistent neighborhood collective
        if (has_persistent_neighbor && persistent_neighbor_request != MPI_REQUEST_NULL) {
            MPI_Request_free(&persistent_neighbor_request);
        }
        
        // Free graph communicator
        if (has_graph_comm && graph_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&graph_comm);
        }
    }
    
    
    /**
     * @brief Build communication plan from mesh with flat array inputs
     * @param mesh Reference to partitioned mesh (with ghost elements/nodes)
     * @param world_size Number of MPI ranks
     * @param my_rank Current MPI rank ID
     * @param boundary_ghost_dest_ranks Flat array of destination ranks for boundary elements [size: sum of neighbors]
     * @param boundary_ghsot_dest_offsets CSR offsets: boundary_ghost_dest_offsets[elem_lid] = start index in boundary_ghost_dest_ranks
     * @param boundary_ghost_dest_gids Flat array of global ghost IDs to send [size: sum of neighbors]
     * @param all_ghost_gids All ghost global IDs across all ranks
     * @param all_ghost_owner_ranks Owner rank for each ghost GID
     * 
     * This build() function takes only flat arrays as input (no std::map, std::set, std::pair).
     * The caller must pre-process the mesh data into flat CSR-style arrays.
     * 
     * Implementation:
     * 1. Group sends/receives by rank using flat arrays and CSR indexing
     * 2. Pre-allocate all MPI buffers
     * 3. Store everything in contiguous memory
     */
    void build(
        const Mesh_t& mesh,
        int world_size,
        int my_rank,
        const int* boundary_ghost_dest_ranks,      // Flat array of dest ranks
        const int* boundary_ghost_dest_offsets,    // CSR offsets [size: num_owned_ghosts+1]
        const size_t* boundary_ghost_dest_gids,    // Flat array of ghost GIDs
        const size_t* all_ghost_gids,              // All ghost GIDs
        const int* all_ghost_owner_ranks,          // Owner ranks indexed by GID
    );
    
    
    // ========================================================================
    // COMMUNICATION INTERFACE - FOR DISTRIBUTED DATA STRUCTURES
    // ========================================================================
    
    /**
     * @brief Pack and exchange data with automatic ghost synchronization
     * @param data_ptr Pointer to data array [size: num_total_items * stride]
     * @param num_fields Number of fields per item (stride)
     * @param item_type 0=elements, 1=nodes
     * @param comm MPI communicator
     * @param blocking If true, waits for completion before returning
     * 
     * This is the main interface for distributed structures like:
     *   node.velocity.comm()  internally calls:
     *     comm_plan.communicate(node.velocity.data(), 3, 1, MPI_COMM_WORLD, true)
     */
    void communicate(double* data_ptr, int num_fields, int item_type, 
                    MPI_Comm comm = MPI_COMM_WORLD, bool blocking = true);
    
    
    /**
     * @brief Non-blocking version: initiate communication
     * Returns immediately; user must call wait_communication()
     */
    void communicate_begin(double* data_ptr, int num_fields, int item_type,
                          MPI_Comm comm = MPI_COMM_WORLD);
    
    
    /**
     * @brief Wait for non-blocking communication to complete
     */
    void wait_communication(double* data_ptr, int num_fields, int item_type);
    
    
    // ========================================================================
    // LOW-LEVEL PACK/UNPACK (for manual control)
    // ========================================================================
    
    /**
     * @brief Pack element data from contiguous array into send buffer
     * @param data_ptr Pointer to element data [size: num_total_elems * num_fields]
     * @param num_fields Stride (fields per element)
     * 
     * Packs data in layout: [elem0_field0, elem0_field1, ..., elem1_field0, ...]
     */
    void pack_ghosts(const double* data_ptr, int num_fields, int field_dimension);
    
    
    /**
     * @brief Unpack received element data into ghost elements
     */
    void unpack_ghosts(double* data_ptr, int num_fields, int field_dimension);
    
    
    
    // ========================================================================
    // MPI EXCHANGE PRIMITIVES
    // ========================================================================
    
    /**
     * @brief Execute MPI_Isend/Irecv for elements
     */
    void exchange_ghosts_begin(int num_fields, int field_dimension, MPI_Comm comm = MPI_COMM_WORLD);
    
    
    /**
     * @brief Wait for element exchange to complete
     */
    void exchange_ghosts_wait();
    
    
    
    // ========================================================================
    // PERSISTENT COMMUNICATION (OPTIMIZATION)
    // ========================================================================
    
    /**
     * @brief Setup persistent MPI communication handles (one-time setup)
     * Call once after build(), then use start_persistent/wait_persistent
     */
    void init_persistent(int elem_fields, int node_fields, MPI_Comm comm = MPI_COMM_WORLD);
    
    
    /**
     * @brief Start persistent send/recv (must call pack_* first)
     */
    void start_persistent();
    
    
    /**
     * @brief Wait for persistent communication (then call unpack_*)
     */
    void wait_persistent();
    
    
    /**
     * @brief Free persistent communication handles
     */
    void free_persistent();
    
    
    // ========================================================================
    // NEIGHBORHOOD COLLECTIVES (MPI-3.0+)
    // ========================================================================
    
    /**
     * @brief Create distributed graph communicator from communication pattern
     * 
     * Call this ONCE after populating send_rank_ids and recv_rank_ids.
     * The graph communicator encodes the sparse communication topology and is
     * reused for all subsequent neighborhood collective calls.
     * 
     * @param base_comm Base communicator (usually MPI_COMM_WORLD)
     * 
     * Example from your output:
     *   rank 0 sends to: {2, 3, 4, 10, 11}
     *   rank 0 receives from: {computed from ghost ownership}
     * 
     * This creates a directed graph where edges represent communication channels.
     * MPI can optimize routing and minimize network contention.
     * 
     * Requirements: MPI-3.0+ (2012)
     */
    void create_graph_communicator(MPI_Comm base_comm = MPI_COMM_WORLD);
    
    
    /**
     * @brief Exchange ghost data using MPI_Neighbor_alltoallv
     * 
     * Uses the pre-created graph communicator for efficient sparse communication.
     * This is cleaner than manual Isend/Irecv loops and allows MPI to optimize.
     * 
     * @param data_ptr Pointer to data array [size: num_total_items * num_fields]
     * @param num_fields Number of fields per item (e.g., 3 for velocity)
     * 
     * Workflow:
     * 1. Pack owned items into send buffer
     * 2. Call MPI_Neighbor_alltoallv (blocking but fast with graph_comm)
     * 3. Unpack ghost items from receive buffer
     * 
     * The graph_comm is reused each call - only pack/unpack overhead per timestep.
     * 
     * Requirements: Must call create_graph_communicator() once before using this.
     */
    void exchange_ghosts_neighborhood(double* data_ptr, int num_fields);
    
    
    /**
     * @brief Initialize persistent neighborhood collective (MPI-4.0+)
     * 
     * Creates a persistent MPI request that pre-allocates all internal buffers
     * and communication paths. Provides maximum performance for repeated exchanges
     * with the same num_fields.
     * 
     * @param num_fields Number of fields per item (must be same for all timesteps)
     * 
     * Call once during setup:
     *   comm_plan.create_graph_communicator(MPI_COMM_WORLD);
     *   comm_plan.init_persistent_neighborhood(3);  // For 3D velocity
     * 
     * Then use exchange_ghosts_persistent() each timestep.
     * 
     * Requirements: MPI-4.0+ (2021). Check with: mpirun --version
     */
    void init_persistent_neighborhood(int num_fields);
    
    
    /**
     * @brief Exchange ghosts using persistent neighborhood collective (FASTEST)
     * 
     * Must call init_persistent_neighborhood() once before using this.
     * This is the fastest ghost exchange method for fixed communication patterns.
     * 
     * @param data_ptr Pointer to data array [size: num_total_items * num_fields]
     * 
     * Workflow:
     * 1. Pack data into same send buffer used during init
     * 2. MPI_Start() - extremely fast, no setup overhead
     * 3. MPI_Wait() - wait for completion
     * 4. Unpack from receive buffer
     * 
     * Typical speedup vs standard neighborhood: 1.2-1.5x
     * 
     * Note: Falls back to exchange_ghosts_neighborhood() if MPI-4 unavailable.
     */
    void exchange_ghosts_persistent(double* data_ptr);
    
    
    /**
     * @brief Free persistent neighborhood collective resources
     * 
     * Call at end of simulation to release MPI resources.
     * Automatically called by destructor if not explicitly freed.
     */
    void free_persistent_neighborhood();
    
    
    // ========================================================================
    // UTILITIES
    // ========================================================================
    
    void print_summary(int rank) const;
    bool validate(MPI_Comm comm = MPI_COMM_WORLD) const;
    size_t send_volume(int elem_fields, int node_fields) const;
    size_t recv_volume(int elem_fields, int node_fields) const;
    bool needs_communication() const;
    int num_neighbor_ranks() const;
    
    
    // ========================================================================
    // INLINE IMPLEMENTATIONS - NEIGHBORHOOD COLLECTIVES
    // ========================================================================
    
    /**
     * @brief Create distributed graph communicator from communication pattern
     */
    inline void create_graph_communicator(MPI_Comm base_comm) {
        
        if (has_graph_comm) {
            std::cerr << "Warning: Graph communicator already created, skipping." << std::endl;
            return;
        }
        
        int indegree = num_recv_ranks;   // Number of ranks we receive FROM
        int outdegree = num_send_ranks;  // Number of ranks we send TO
        
        // Create the distributed graph communicator
        // MPI_Dist_graph_create_adjacent signature:
        //   (comm_old, indegree, sources[], sourceweights, outdegree, dests[], destweights,
        //    info, reorder, comm_dist_graph)
        int reorder = 0;  // Don't reorder ranks (keep same as base_comm)
        
        MPI_Dist_graph_create_adjacent(
            base_comm,                    // Base communicator
            indegree,                     // We receive from num_recv_ranks neighbors
            recv_rank_ids.data(),         // Source ranks (we receive from these)
            MPI_UNWEIGHTED,               // No edge weights for sources
            outdegree,                    // We send to num_send_ranks neighbors
            send_rank_ids.data(),         // Destination ranks (we send to these)
            MPI_UNWEIGHTED,               // No edge weights for destinations
            MPI_INFO_NULL,                // No special hints
            reorder,                      // Don't reorder ranks
            &graph_comm                   // Output: new graph communicator
        );
        
        has_graph_comm = true;
        
        // Pre-allocate counts and displacements arrays
        send_counts.resize(num_send_ranks);
        send_displs.resize(num_send_ranks);
        recv_counts.resize(num_recv_ranks);
        recv_displs.resize(num_recv_ranks);
    }
    
    
    /**
     * @brief Exchange ghost data using MPI_Neighbor_alltoallv
     */
    inline void exchange_ghosts_neighborhood(double* data_ptr, int num_fields) {
        
        if (!has_graph_comm) {
            std::cerr << "Error: Must call create_graph_communicator() first!" << std::endl;
            return;
        }
        
        // 1. Pack send buffer from owned items
        int total_send = send_ghost_lids.size();
        ghost_send_buffer.resize(total_send * num_fields);
        
        for (size_t i = 0; i < send_ghost_lids.size(); i++) {
            int local_id = send_ghost_lids[i];
            for (int f = 0; f < num_fields; f++) {
                ghost_send_buffer[i * num_fields + f] = data_ptr[local_id * num_fields + f];
            }
        }
        
        // 2. Update counts and displacements for this num_fields
        for (int i = 0; i < num_send_ranks; i++) {
            int start_idx = send_ghost_offsets[i];
            int end_idx = send_ghost_offsets[i + 1];
            send_counts[i] = (end_idx - start_idx) * num_fields;
            send_displs[i] = start_idx * num_fields;
        }
        
        int total_recv = recv_ghost_lids.size();
        ghost_recv_buffer.resize(total_recv * num_fields);
        
        for (int i = 0; i < num_recv_ranks; i++) {
            int start_idx = recv_ghost_offsets[i];
            int end_idx = recv_ghost_offsets[i + 1];
            recv_counts[i] = (end_idx - start_idx) * num_fields;
            recv_displs[i] = start_idx * num_fields;
        }
        
        // 3. Execute neighborhood collective (BLOCKING but fast with graph_comm)
        // MPI_Neighbor_alltoallv signature:
        //   (sendbuf, sendcounts[], sdispls[], sendtype,
        //    recvbuf, recvcounts[], rdispls[], recvtype, comm)
        MPI_Neighbor_alltoallv(
            ghost_send_buffer.data(),    // Send buffer
            send_counts.data(),          // Send counts per neighbor
            send_displs.data(),          // Send displacements
            MPI_DOUBLE,                  // Send type
            ghost_recv_buffer.data(),    // Receive buffer
            recv_counts.data(),          // Receive counts per neighbor
            recv_displs.data(),          // Receive displacements
            MPI_DOUBLE,                  // Receive type
            graph_comm                   // Graph communicator (NOT MPI_COMM_WORLD!)
        );
        
        // 4. Unpack receive buffer into ghost items
        for (size_t i = 0; i < recv_ghost_lids.size(); i++) {
            int ghost_local_id = recv_ghost_lids[i];
            for (int f = 0; f < num_fields; f++) {
                data_ptr[ghost_local_id * num_fields + f] = ghost_recv_buffer[i * num_fields + f];
            }
        }
    }
    
    
    /**
     * @brief Initialize persistent neighborhood collective (MPI-4.0+)
     */
    inline void init_persistent_neighborhood(int num_fields) {
        
        if (!has_graph_comm) {
            std::cerr << "Error: Must call create_graph_communicator() first!" << std::endl;
            return;
        }
        
        if (has_persistent_neighbor) {
            std::cerr << "Warning: Persistent neighborhood already initialized, freeing and re-creating." << std::endl;
            free_persistent_neighborhood();
        }
        
        persistent_num_fields = num_fields;
        
        // Allocate buffers
        int total_send = send_ghost_lids.size();
        int total_recv = recv_ghost_lids.size();
        ghost_send_buffer.resize(total_send * num_fields);
        ghost_recv_buffer.resize(total_recv * num_fields);
        
        // Setup counts and displacements for persistent request
        for (int i = 0; i < num_send_ranks; i++) {
            int start_idx = send_ghost_offsets[i];
            int end_idx = send_ghost_offsets[i + 1];
            send_counts[i] = (end_idx - start_idx) * num_fields;
            send_displs[i] = start_idx * num_fields;
        }
        
        for (int i = 0; i < num_recv_ranks; i++) {
            int start_idx = recv_ghost_offsets[i];
            int end_idx = recv_ghost_offsets[i + 1];
            recv_counts[i] = (end_idx - start_idx) * num_fields;
            recv_displs[i] = start_idx * num_fields;
        }
        
#if MPI_VERSION >= 4
        // MPI-4.0+ persistent neighborhood collective
        // MPI_Neighbor_alltoallv_init signature (similar to MPI_Neighbor_alltoallv but creates request):
        //   (sendbuf, sendcounts[], sdispls[], sendtype,
        //    recvbuf, recvcounts[], rdispls[], recvtype, comm, info, request)
        MPI_Neighbor_alltoallv_init(
            ghost_send_buffer.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
            ghost_recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
            graph_comm,
            MPI_INFO_NULL,
            &persistent_neighbor_request
        );
        has_persistent_neighbor = true;
#else
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Warning: MPI-4.0 required for persistent neighborhood collectives" << std::endl;
            std::cerr << "         Detected MPI version: " << MPI_VERSION << "." << MPI_SUBVERSION << std::endl;
            std::cerr << "         Will fall back to standard neighborhood collective" << std::endl;
        }
        has_persistent_neighbor = false;
#endif
    }
    
    
    /**
     * @brief Exchange ghosts using persistent neighborhood collective (FASTEST)
     */
    inline void exchange_ghosts_persistent(double* data_ptr) {
        
#if MPI_VERSION >= 4
        if (!has_persistent_neighbor) {
            std::cerr << "Error: Must call init_persistent_neighborhood() first!" << std::endl;
            std::cerr << "       Falling back to standard neighborhood collective..." << std::endl;
            exchange_ghosts_neighborhood(data_ptr, persistent_num_fields);
            return;
        }
        
        // 1. Pack send buffer (same memory location as during init)
        for (size_t i = 0; i < send_ghost_lids.size(); i++) {
            int local_id = send_ghost_lids[i];
            for (int f = 0; f < persistent_num_fields; f++) {
                ghost_send_buffer[i * persistent_num_fields + f] = 
                    data_ptr[local_id * persistent_num_fields + f];
            }
        }
        
        // 2. Start persistent request (VERY fast - no setup overhead)
        MPI_Start(&persistent_neighbor_request);
        
        // 3. Wait for completion
        MPI_Wait(&persistent_neighbor_request, MPI_STATUS_IGNORE);
        
        // 4. Unpack receive buffer
        for (size_t i = 0; i < recv_ghost_lids.size(); i++) {
            int ghost_id = recv_ghost_lids[i];
            for (int f = 0; f < persistent_num_fields; f++) {
                data_ptr[ghost_id * persistent_num_fields + f] = 
                    ghost_recv_buffer[i * persistent_num_fields + f];
            }
        }
#else
        // Fallback to standard method if MPI-4 not available
        exchange_ghosts_neighborhood(data_ptr, persistent_num_fields);
#endif
    }
    
    
    /**
     * @brief Free persistent neighborhood collective resources
     */
    inline void free_persistent_neighborhood() {
#if MPI_VERSION >= 4
        if (has_persistent_neighbor && persistent_neighbor_request != MPI_REQUEST_NULL) {
            MPI_Request_free(&persistent_neighbor_request);
            persistent_neighbor_request = MPI_REQUEST_NULL;
            has_persistent_neighbor = false;
        }
#endif
    }
    
};


