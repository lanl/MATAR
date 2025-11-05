#ifndef MPICARRAYKOKKOS_H
#define MPICARRAYKOKKOS_H

#include "matar.h"
#include "communication_plan.h"

using namespace mtr;

/////////////////////////
// MPICArrayKokkos:  Dual type for managing distributed data on both CPU and GPU.
// 
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPICArrayKokkos {

    // Dual view for managing data on both CPU and GPU
    DCArrayKokkos<T> this_array_;

    DCArrayKokkos<T> send_buffer_;
    DCArrayKokkos<T> recv_buffer_;
    
protected:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)

    MPI_Comm mpi_comm_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;

    
    // --- Ghost Communication Support ---
    CommunicationPlan* comm_plan_;      // Pointer to shared communication plan


    DCArrayKokkos<int> send_counts_; // [size: num_send_ranks] Number of items to send to each rank
    DCArrayKokkos<int> recv_counts_; // [size: num_recv_ranks] Number of items to receive from each rank
    DCArrayKokkos<int> send_displs_; // [size: num_send_ranks] Starting index of items to send to each rank
    DCArrayKokkos<int> recv_displs_; // [size: num_recv_ranks] Starting index of items to receive from each rank


    DRaggedRightArrayKokkos<int> send_indices_; // [size: num_send_ranks, num_items_to_send_by_rank] Indices of items to send to each rank
    DRaggedRightArrayKokkos<int> recv_indices_; // [size: num_recv_ranks, num_items_to_recv_by_rank] Indices of items to receive from each rank
    
    
    size_t num_owned_;            // Number of owned items (nodes/elements)
    size_t num_ghost_;            // Number of ghost items (nodes/elements)
    
    void set_mpi_type();

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPICArrayKokkos();
    
    MPICArrayKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5,
                 size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    


    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    KOKKOS_INLINE_FUNCTION
    MPICArrayKokkos& operator=(const MPICArrayKokkos& temp);


    // Method to set comm plan
    KOKKOS_INLINE_FUNCTION
    void initialize_comm_plan(CommunicationPlan& comm_plan){
        comm_plan_ = &comm_plan;
        send_buffer_ = DCArrayKokkos<T>(comm_plan_->total_send_count, "send_buffer");
        recv_buffer_ = DCArrayKokkos<T>(comm_plan_->total_recv_count, "recv_buffer");
    };



    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    size_t dims(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    size_t order() const;
 
    // Method returns the raw device pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* device_pointer() const;

    // Method returns the raw host pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* host_pointer() const;

    // Method returns kokkos dual view
    KOKKOS_INLINE_FUNCTION
    Kokkos::DualView<T*, Layout, ExecSpace, MemoryTraits> get_kokkos_dual_view() const;

    // Method that update host view
    void update_host();

    // Method that update device view
    void update_device();

    // Method that builds the send buffer
    void fill_send_buffer(){

        int rank;
        MPI_Comm_rank(comm_plan_->mpi_comm_world, &rank);

        // this_array_.update_host();
        int send_idx = 0;
        for(int i = 0; i < comm_plan_->num_send_ranks; i++){
            for(int j = 0; j < comm_plan_->send_counts_.host(i); j++){
                int src_idx = comm_plan_->send_indices_.host(i, j);
                send_buffer_.host(send_idx) = this_array_.host(src_idx);
                if(rank == 0) std::cout << "MPICArrayKokkos::fill_send_buffer() - send_buffer(" << send_idx << ") = " << this_array_.host(src_idx) << std::endl;
                send_idx++;
            }
        }
    };

    // Method that copies the recv buffer
    void copy_recv_buffer(){
        int rank;
        MPI_Comm_rank(comm_plan_->mpi_comm_world, &rank);

        // NOTE: Do NOT call recv_buffer_.update_host() here!
        // MPI already wrote directly to host memory, so calling update_host()
        // would overwrite the received data by copying stale device data
        int recv_idx = 0;
        for(int i = 0; i < comm_plan_->num_recv_ranks; i++){
            for(int j = 0; j < comm_plan_->recv_counts_.host(i); j++){
                int dest_idx = comm_plan_->recv_indices_.host(i, j);
                this_array_.host(dest_idx) = recv_buffer_.host(recv_idx);
                //if(rank == 0) std::cout << "MPICArrayKokkos::copy_recv_buffer() - this_array(" << dest_idx << ") = " << recv_buffer_.host(recv_idx) << std::endl;
                recv_idx++;
            }
        }
    };

    void communicate(){
        int rank;
        MPI_Comm_rank(comm_plan_->mpi_comm_world, &rank);
        
        if(rank == 0) {
            std::cout << "MPICArrayKokkos::communicate() - this_array size: " << this_array_.size() << std::endl;
            std::cout << "MPICArrayKokkos::communicate() - send_buffer size: " << send_buffer_.size() 
                      << ", recv_buffer size: " << recv_buffer_.size() << std::endl;
            std::cout << "MPICArrayKokkos::communicate() - total_send_count: " << comm_plan_->total_send_count 
                      << ", total_recv_count: " << comm_plan_->total_recv_count << std::endl;
        }
        
        fill_send_buffer();

        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() - Starting MPI_Neighbor_alltoallv" << std::endl;


        MPI_Barrier(comm_plan_->mpi_comm_world);
        
        // Verify buffer sizes match expected
        if(rank == 0) {
            std::cout << "Send buffer size check: " << send_buffer_.size() << " vs expected " << comm_plan_->total_send_count << std::endl;
            std::cout << "Recv buffer size check: " << recv_buffer_.size() << " vs expected " << comm_plan_->total_recv_count << std::endl;
            
            // Print first few send values
            std::cout << "MPICArrayKokkos::communicate() - send_buffer values: ";
            for(int i = 0; i < 10 && i < send_buffer_.size(); i++) {
                std::cout << send_buffer_.host(i) << " ";
            }
            std::cout << std::endl;
            
            // Print send counts and displs
            std::cout << "Send counts: ";
            int total_send = 0;
            for(int i = 0; i < comm_plan_->num_send_ranks; i++) {
                int count = comm_plan_->send_counts_.host(i);
                std::cout << count << " ";
                total_send += count;
            }
            std::cout << "(total=" << total_send << ")" << std::endl;
            
            std::cout << "Send displs: ";
            for(int i = 0; i < comm_plan_->num_send_ranks; i++) {
                std::cout << comm_plan_->send_displs_.host(i) << " ";
            }
            std::cout << std::endl;
            
            // Print recv counts and displs
            std::cout << "Recv counts: ";
            int total_recv = 0;
            for(int i = 0; i < comm_plan_->num_recv_ranks; i++) {
                int count = comm_plan_->recv_counts_.host(i);
                std::cout << count << " ";
                total_recv += count;
            }
            std::cout << "(total=" << total_recv << ")" << std::endl;
            
            std::cout << "Recv displs: ";
            for(int i = 0; i < comm_plan_->num_recv_ranks; i++) {
                std::cout << comm_plan_->recv_displs_.host(i) << " ";
            }
            std::cout << std::endl;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() calling MPI_Neighbor_alltoallv"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        // CRITICAL: Get all pointers BEFORE the MPI call and store them in local stack variables
        // This prevents Kokkos from deallocating during the MPI call
        // Use nullptr for empty arrays to avoid accessing element 0 of 0-sized array (undefined behavior)
        T* send_buf_ptr = (send_buffer_.size() > 0) ? &send_buffer_.host(0) : nullptr;
        T* recv_buf_ptr = (recv_buffer_.size() > 0) ? &recv_buffer_.host(0) : nullptr;
        int* send_cnt_ptr = (comm_plan_->num_send_ranks > 0) ? &comm_plan_->send_counts_.host(0) : nullptr;
        int* send_dsp_ptr = (comm_plan_->num_send_ranks > 0) ? &comm_plan_->send_displs_.host(0) : nullptr;
        int* recv_cnt_ptr = (comm_plan_->num_recv_ranks > 0) ? &comm_plan_->recv_counts_.host(0) : nullptr;
        int* recv_dsp_ptr = (comm_plan_->num_recv_ranks > 0) ? &comm_plan_->recv_displs_.host(0) : nullptr;
        
        if(rank == 0) {
            std::cout << "Pointer addresses:" << std::endl;
            std::cout << "  send_buf_ptr = " << (void*)send_buf_ptr << std::endl;
            std::cout << "  send_cnt_ptr = " << (void*)send_cnt_ptr << std::endl;
            std::cout << "  send_dsp_ptr = " << (void*)send_dsp_ptr << std::endl;
            std::cout << "  recv_buf_ptr = " << (void*)recv_buf_ptr << std::endl;
            std::cout << "  recv_cnt_ptr = " << (void*)recv_cnt_ptr << std::endl;
            std::cout << "  recv_dsp_ptr = " << (void*)recv_dsp_ptr << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Neighbor_alltoallv(
            &send_buffer_.host(0),
            &comm_plan_->send_counts_.host(0),
            &comm_plan_->send_displs_.host(0),
            MPI_DOUBLE,
            &recv_buffer_.host(0),
            &comm_plan_->recv_counts_.host(0),
            &comm_plan_->recv_displs_.host(0), 
            MPI_DOUBLE, 
            comm_plan_->mpi_comm_graph);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() finished MPI_Neighbor_alltoallv"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() about to copy recv buffer"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        copy_recv_buffer();

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() finished copying recv buffer"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() about to update device"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        //this_array_.update_device();  // Commented out - not needed since nothing runs on device

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() finished updating device (skipped)"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(rank == 0) std::cout << "MPICArrayKokkos::communicate() about to return"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    };

    

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPICArrayKokkos ();
}; // End of MPIDArrayKokkos



// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos()
    : this_array_() { }

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1);
}

// Overloaded 3D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2);
}

// Overloaded 4D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, dim4, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3, dim4);
}

// Overloaded 6D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, dim4, dim5, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3, dim4, dim5);
}

// Overloaded 7D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3, dim4, dim5, dim6);
}


template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 1D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 1D!");
    return this_array_(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 2D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 2D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 2D!");
    return this_array_(i, j);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 3D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 3D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 3D!");
    assert(k < dims_[2] && "k is out of bounds in MPICArrayKokkos 3D!");
    return this_array_(i, j, k);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 4D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 4D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 4D!");
    assert(k < dims_[2] && "k is out of bounds in MPICArrayKokkos 4D!");
    assert(l < dims_[3] && "l is out of bounds in MPICArrayKokkos 4D!");
    return this_array_(i, j, k, l);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 5D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 5D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 5D!");
    assert(k < dims_[2] && "k is out of bounds in MPICArrayKokkos 5D!");
    assert(l < dims_[3] && "l is out of bounds in MPICArrayKokkos 5D!");
    assert(m < dims_[4] && "m is out of bounds in MPICArrayKokkos 5D!");
    return this_array_(i, j, k, l, m);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 6D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 6D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 6D!");
    assert(k < dims_[2] && "k is out of bounds in MPICArrayKokkos 6D!");
    assert(l < dims_[3] && "l is out of bounds in MPICArrayKokkos 6D!");
    assert(m < dims_[4] && "m is out of bounds in MPICArrayKokkos 6D!");
    assert(n < dims_[5] && "n is out of bounds in MPICArrayKokkos 6D!");
    return this_array_(i, j, k, l, m, n);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 7D!");
    assert(i < dims_[0] && "i is out of bounds in MPICArrayKokkos 7D!");
    assert(j < dims_[1] && "j is out of bounds in MPICArrayKokkos 7D!");
    assert(k < dims_[2] && "k is out of bounds in MPICArrayKokkos 7D!");
    assert(l < dims_[3] && "l is out of bounds in MPICArrayKokkos 7D!");
    assert(m < dims_[4] && "m is out of bounds in MPICArrayKokkos 7D!");
    assert(n < dims_[5] && "n is out of bounds in MPICArrayKokkos 7D!");
    assert(o < dims_[6] && "o is out of bounds in MPICArrayKokkos 7D!");
    return this_array_(i, j, k, l, m, n, o);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator=(const MPICArrayKokkos& temp) {
    this_array_ = temp.this_array_;
    host = temp.host;  // Also copy the host ViewCArray
    comm_plan_ = temp.comm_plan_;
    send_buffer_ = temp.send_buffer_;
    recv_buffer_ = temp.recv_buffer_;
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return this_array_.size();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return this_array_.extent();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPICArrayKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(dims_[i]>0 && "Access to MPICArrayKokkos dims is out of bounds!");
    return this_array_.dims(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return this_array_.order();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.device_pointer();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.host_pointer();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
    return this_array_.get_kokkos_dual_view();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {
    this_array_.update_host();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {
    this_array_.update_device();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPICArrayKokkos() {
    // Member variables (this_array_, send_buffer_, recv_buffer_) are automatically
    // destroyed by the compiler - no explicit cleanup needed
}

#endif