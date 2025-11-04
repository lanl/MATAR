// #ifndef MPIDARRAYKOKKOS_H
// #define MPIDARRAYKOKKOS_H

// #include "matar.h"
// #include "communication_plan.h"

// using namespace mtr;

// /////////////////////////
// // MPIDArrayKokkos:  Dual type for managing distributed data on both CPU and GPU.
// // 
// // Enhanced with automatic ghost synchronization via CommunicationPlan.
// // Allocates space for owned + ghost items and provides communicate() method.
// //
// // Usage:
// //   node.coords.communicate()  -> syncs ghost nodes automatically
// //   elem.density.communicate() -> syncs ghost elements automatically
// /////////////////////////
// template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
// class MPIDArrayKokkos {

//     // this is manage
//     using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
// protected:
//     size_t dims_[7];
//     size_t length_;
//     size_t order_;  // tensor order (rank)
//     int mpi_recv_rank_;
//     int mpi_tag_;
//     MPI_Comm mpi_comm_;
//     MPI_Status mpi_status_;
//     MPI_Datatype mpi_datatype_;
//     MPI_Request mpi_request_;
//     TArray1D this_array_;
    
//     // --- Ghost Communication Support ---
//     CommunicationPlan* comm_plan_;      // Pointer to shared communication plan
//     size_t num_owned_items_;            // Number of owned items (nodes/elements)
//     size_t num_total_items_;            // Total items including ghosts (owned + ghost)
//     size_t num_fields_;                 // Fields per item (e.g., 3 for 3D coordinates)
    
//     void set_mpi_type();

// public:
//     // Data member to access host view
//     ViewCArray <T> host;

//     MPIDArrayKokkos();
    
//     MPIDArrayKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
//                  size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
//                  size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
//                  size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

//     MPIDArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
//                  size_t dim3, size_t dim4, size_t dim5,
//                  size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    
//     // ========================================================================
//     // DISTRIBUTED COMMUNICATION METHODS (NEW)
//     // ========================================================================
    
//     /**
//      * @brief Set communication plan and ghost metadata
//      * 
//      * Call this ONCE after allocating the array to enable ghost communication.
//      * Multiple fields can share the same CommunicationPlan pointer.
//      * 
//      * @param plan Pointer to shared CommunicationPlan (node or element plan)
//      * @param num_owned Number of owned items on this rank
//      * @param num_total Total items including ghosts (owned + ghost)
//      * 
//      * Example:
//      *   node.coords = MPIDArrayKokkos<double>(num_total_nodes, 3);
//      *   node.coords.set_communication_plan(&node_comm_plan, num_owned_nodes, num_total_nodes);
//      */
//     void set_communication_plan(CommunicationPlan* plan, size_t num_owned, size_t num_total);
    
    
//     /**
//      * @brief Synchronize ghost data using neighborhood collectives
//      * 
//      * Automatically exchanges boundary → ghost data for this field.
//      * Uses the CommunicationPlan provided via set_communication_plan().
//      * 
//      * Workflow:
//      * 1. Updates host data from device (if needed)
//      * 2. Packs owned boundary items
//      * 3. Calls MPI_Neighbor_alltoallv (via comm_plan)
//      * 4. Unpacks into ghost items
//      * 5. Updates device with new ghost data
//      * 
//      * Example usage:
//      *   // Update owned nodes
//      *   for (int i = 0; i < num_owned_nodes; i++) {
//      *       node.coords(i, 0) += dt * velocity(i, 0);
//      *   }
//      *   
//      *   // Sync ghosts
//      *   node.coords.communicate();
//      *   
//      *   // Now ghost data is current
//      */
//     void communicate();
    
    
//     /**
//      * @brief Non-blocking version: start ghost exchange
//      * 
//      * For advanced users who want to overlap computation with communication.
//      * Must call communicate_wait() before accessing ghost data.
//      */
//     void communicate_begin();
    
    
//     /**
//      * @brief Wait for non-blocking ghost exchange to complete
//      */
//     void communicate_wait();
    
    
//     /**
//      * @brief Get number of owned items (excludes ghosts)
//      */
//     KOKKOS_INLINE_FUNCTION
//     size_t num_owned() const { return num_owned_items_; }
    
    
//     /**
//      * @brief Get total items including ghosts
//      */
//     KOKKOS_INLINE_FUNCTION
//     size_t num_total() const { return num_total_items_; }
    
    
//     /**
//      * @brief Check if ghost communication is configured
//      */
//     bool has_communication_plan() const { return comm_plan_ != nullptr; }
    
//     // These functions can setup the data needed for halo send/receives
//     // Not necessary for standard MPI comms
//     void mpi_setup();

//     void mpi_setup(int recv_rank);

//     void mpi_setup(int recv_rank, int tag);

//     void mpi_setup(int recv_rank, int tag, MPI_Comm comm);

//     void mpi_set_rank(int recv_rank);

//     void mpi_set_tag(int tag);

//     void mpi_set_comm(MPI_Comm comm);

//     int get_rank();

//     int get_tag();

//     MPI_Comm get_comm();

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j, size_t k) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j, size_t k, size_t l) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
//                   size_t n) const;

//     KOKKOS_INLINE_FUNCTION
//     T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
//                   size_t n, size_t o) const;
    
//     KOKKOS_INLINE_FUNCTION
//     MPIDArrayKokkos& operator=(const MPIDArrayKokkos& temp);

//     // GPU Method
//     // Method that returns size
//     KOKKOS_INLINE_FUNCTION
//     size_t size() const;

//     // Host Method
//     // Method that returns size
//     KOKKOS_INLINE_FUNCTION
//     size_t extent() const;

//     KOKKOS_INLINE_FUNCTION
//     size_t dims(size_t i) const;

//     KOKKOS_INLINE_FUNCTION
//     size_t order() const;
 
//     // Method returns the raw device pointer of the Kokkos DualView
//     KOKKOS_INLINE_FUNCTION
//     T* device_pointer() const;

//     // Method returns the raw host pointer of the Kokkos DualView
//     KOKKOS_INLINE_FUNCTION
//     T* host_pointer() const;

//     // Method returns kokkos dual view
//     KOKKOS_INLINE_FUNCTION
//     TArray1D get_kokkos_dual_view() const;

//     // Method that update host view
//     void update_host();

//     // Method that update device view
//     void update_device();

    

//     // Deconstructor
//     virtual KOKKOS_INLINE_FUNCTION
//     ~MPIDArrayKokkos ();
// }; // End of MPIDArrayKokkos


// // ============================================================================
// // INLINE IMPLEMENTATIONS - DISTRIBUTED COMMUNICATION
// // ============================================================================

// /**
//  * @brief Default constructor - initialize ghost communication members
//  */
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// KOKKOS_INLINE_FUNCTION
// MPIDArrayKokkos<T, Layout, ExecSpace, MemoryTraits>::MPIDArrayKokkos() 
//     : comm_plan_(nullptr), 
//       num_owned_items_(0), 
//       num_total_items_(0), 
//       num_fields_(0) 
// {
//     // Base constructor handles array initialization
// }


// /**
//  * @brief Set communication plan and ghost metadata
//  */
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// inline void MPIDArrayKokkos<T, Layout, ExecSpace, MemoryTraits>::set_communication_plan(
//     CommunicationPlan* plan, 
//     size_t num_owned, 
//     size_t num_total)
// {
//     comm_plan_ = plan;
//     num_owned_items_ = num_owned;
//     num_total_items_ = num_total;
    
//     // Infer number of fields from array dimensions
//     // Assumption: dim0 = num_items, dim1+ = fields
//     if (order_ == 1) {
//         num_fields_ = 1;  // Scalar field
//     } else if (order_ == 2) {
//         num_fields_ = dims_[1];  // Vector field (e.g., coords[num_nodes, 3])
//     } else {
//         // For higher order tensors, treat everything after dim0 as fields
//         num_fields_ = 1;
//         for (size_t i = 1; i < order_; i++) {
//             num_fields_ *= dims_[i];
//         }
//     }
    
//     // Validate dimensions match total items
//     if (dims_[0] != num_total) {
//         std::cerr << "Error: Array dim0 (" << dims_[0] << ") does not match num_total (" 
//                   << num_total << ")" << std::endl;
//         std::cerr << "       Array must be allocated with size = num_owned + num_ghost" << std::endl;
//     }
// }


// /**
//  * @brief Synchronize ghost data using neighborhood collectives
//  */
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// inline void MPIDArrayKokkos<T, Layout, ExecSpace, MemoryTraits>::communicate()
// {
//     if (!comm_plan_) {
//         std::cerr << "Error: CommunicationPlan not set. Call set_communication_plan() first." << std::endl;
//         return;
//     }
    
//     if (!comm_plan_->has_graph_comm) {
//         std::cerr << "Error: Graph communicator not initialized in CommunicationPlan." << std::endl;
//         std::cerr << "       Call comm_plan.create_graph_communicator() first." << std::endl;
//         return;
//     }
    
//     // 1. Update host from device (ensure data is current on CPU for MPI)
//     this->update_host();
    
//     // 2. Get raw pointer to data
//     T* data_ptr = this->host_pointer();
    
//     // 3. Convert to double* for MPI communication
//     // TODO: Support other types (int, float, etc.) with template specialization
//     static_assert(std::is_same<T, double>::value, 
//                   "Currently only double supported for ghost communication");
    
//     double* double_ptr = reinterpret_cast<double*>(data_ptr);
    
//     // 4. Call neighborhood collective exchange
//     comm_plan_->exchange_ghosts_neighborhood(double_ptr, static_cast<int>(num_fields_));
    
//     // 5. Update device with new ghost data
//     this->update_device();
// }


// /**
//  * @brief Non-blocking version: start ghost exchange
//  */
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// inline void MPIDArrayKokkos<T, Layout, ExecSpace, MemoryTraits>::communicate_begin()
// {
//     // TODO: Implement non-blocking version using Isend/Irecv
//     // For now, just call blocking version
//     std::cerr << "Warning: communicate_begin() not yet implemented, using blocking communicate()" << std::endl;
//     communicate();
// }


// /**
//  * @brief Wait for non-blocking ghost exchange to complete
//  */
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// inline void MPIDArrayKokkos<T, Layout, ExecSpace, MemoryTraits>::communicate_wait()
// {
//     // TODO: Implement non-blocking version
//     // For now, this is a no-op since communicate_begin() is blocking
// }


// #endif // MPIDARRAYKOKKOS_H
