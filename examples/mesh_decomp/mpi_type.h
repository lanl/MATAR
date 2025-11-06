#ifndef MPICARRAYKOKKOS_H
#define MPICARRAYKOKKOS_H

#include "matar.h"
#include "communication_plan.h"

using namespace mtr;

// Add this before the MPICArrayKokkos class definition

// Type trait to map C++ types to MPI_Datatype
template <typename T>
struct mpi_type_map {
    static MPI_Datatype value() {
        static_assert(sizeof(T) == 0, "Unsupported type for MPI communication");
        return MPI_DATATYPE_NULL;
    }
};

// Specializations for common types
template <>
struct mpi_type_map<int> {
    static MPI_Datatype value() { return MPI_INT; }
};

template <>
struct mpi_type_map<long> {
    static MPI_Datatype value() { return MPI_LONG; }
};

template <>
struct mpi_type_map<long long> {
    static MPI_Datatype value() { return MPI_LONG_LONG; }
};

template <>
struct mpi_type_map<unsigned int> {
    static MPI_Datatype value() { return MPI_UNSIGNED; }
};

template <>
struct mpi_type_map<unsigned long> {
    static MPI_Datatype value() { return MPI_UNSIGNED_LONG; }
};

template <>
struct mpi_type_map<float> {
    static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct mpi_type_map<double> {
    static MPI_Datatype value() { return MPI_DOUBLE; }
};

template <>
struct mpi_type_map<char> {
    static MPI_Datatype value() { return MPI_CHAR; }
};

template <>
struct mpi_type_map<unsigned char> {
    static MPI_Datatype value() { return MPI_UNSIGNED_CHAR; }
};

template <>
struct mpi_type_map<bool> {
    static MPI_Datatype value() { return MPI_C_BOOL; }
};


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

    size_t stride_; // [size: num_dims] Number of contiguous values per first index element


    DRaggedRightArrayKokkos<int> send_indices_; // [size: num_send_ranks, num_items_to_send_by_rank] Indices of items to send to each rank
    DRaggedRightArrayKokkos<int> recv_indices_; // [size: num_recv_ranks, num_items_to_recv_by_rank] Indices of items to receive from each rank
    
    
    size_t num_owned_;            // Number of owned items (nodes/elements)
    size_t num_ghost_;            // Number of ghost items (nodes/elements)

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
    void initialize_comm_plan(CommunicationPlan& comm_plan){
        comm_plan_ = &comm_plan;
        
        size_t send_size = comm_plan_->total_send_count * stride_;
        size_t recv_size = comm_plan_->total_recv_count * stride_;
        
        if (send_size > 0) {
            send_buffer_ = DCArrayKokkos<T>(send_size, "send_buffer");
        }
        if (recv_size > 0) {
            recv_buffer_ = DCArrayKokkos<T>(recv_size, "recv_buffer");
        }

        if (comm_plan_->num_send_ranks > 0) {
            send_counts_ = DCArrayKokkos<int>(comm_plan_->num_send_ranks, "send_counts");
            send_displs_ = DCArrayKokkos<int>(comm_plan_->num_send_ranks, "send_displs");
            
            for(int i = 0; i < comm_plan_->num_send_ranks; i++){
                send_counts_.host(i) = comm_plan_->send_counts_.host(i) * stride_;
                send_displs_.host(i) = comm_plan_->send_displs_.host(i) * stride_;
            }
            send_counts_.update_device();
            send_displs_.update_device();
        }
        
        if (comm_plan_->num_recv_ranks > 0) {
            recv_counts_ = DCArrayKokkos<int>(comm_plan_->num_recv_ranks, "recv_counts");
            recv_displs_ = DCArrayKokkos<int>(comm_plan_->num_recv_ranks, "recv_displs");
            
            for(int i = 0; i < comm_plan_->num_recv_ranks; i++){
                recv_counts_.host(i) = comm_plan_->recv_counts_.host(i) * stride_;
                recv_displs_.host(i) = comm_plan_->recv_displs_.host(i) * stride_;
            }
            recv_counts_.update_device();
            recv_displs_.update_device();
        }
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

    // Method that builds the send buffer, note, this has to be ordered
    // Such that all the boundary elements going to a given rank are contiguous in the send buffer.
    void fill_send_buffer(){


      
        T* src_ptr = this_array_.host_pointer();

        
        size_t send_idx = 0;
        for(int i = 0; i < comm_plan_->num_send_ranks; i++){
            for(int j = 0; j < comm_plan_->send_counts_.host(i); j++){
                size_t src_idx = comm_plan_->send_indices_.host(i, j); // index of the element to send
                
                // Copy all values associated with this element (handles multi-dimensional arrays)
                for(size_t k = 0; k < stride_; k++){
                    send_buffer_.host(send_idx + k) = src_ptr[src_idx * stride_ + k];
                }
                send_idx += stride_;
            }
        }
    };

    // Method that copies the recv buffer into the this_array
    void copy_recv_buffer(){
        
        T* dest_ptr = this_array_.host_pointer();
        
        size_t recv_idx = 0;
        for(int i = 0; i < comm_plan_->num_recv_ranks; i++){
            for(int j = 0; j < comm_plan_->recv_counts_.host(i); j++){
                size_t dest_idx = comm_plan_->recv_indices_.host(i, j);
                
                // Copy all values associated with this element (handles multi-dimensional arrays)
                for(size_t k = 0; k < stride_; k++){
                    dest_ptr[dest_idx * stride_ + k] = recv_buffer_.host(recv_idx + k);
                }
                
                recv_idx += stride_;
            }
        }
        this_array_.update_device();
    };


    // Note: This "may" be needed, im not sure.  Currently, it works....
        // Use nullptr for empty arrays to avoid accessing element 0 of 0-sized array (undefined behavior)
        // T* send_buf_ptr = (send_buffer_.size() > 0) ? &send_buffer_.host(0) : nullptr;
        // T* recv_buf_ptr = (recv_buffer_.size() > 0) ? &recv_buffer_.host(0) : nullptr;
        // int* send_cnt_ptr = (comm_plan_->num_send_ranks > 0) ? &comm_plan_->send_counts_.host(0) : nullptr;
        // int* send_dsp_ptr = (comm_plan_->num_send_ranks > 0) ? &comm_plan_->send_displs_.host(0) : nullptr;
        // int* recv_cnt_ptr = (comm_plan_->num_recv_ranks > 0) ? &comm_plan_->recv_counts_.host(0) : nullptr;
        // int* recv_dsp_ptr = (comm_plan_->num_recv_ranks > 0) ? &comm_plan_->recv_displs_.host(0) : nullptr;

    // Method that communicates the data between the ranks
    void communicate(){

        this_array_.update_host();
       
        fill_send_buffer();
        
        MPI_Neighbor_alltoallv(
            send_buffer_.host_pointer(),
            send_counts_.host_pointer(),
            send_displs_.host_pointer(),
            mpi_type_map<T>::value(),  // MPI_TYPE
            recv_buffer_.host_pointer(),
            recv_counts_.host_pointer(),
            recv_displs_.host_pointer(), 
            mpi_type_map<T>::value(),  // MPI_TYPE
            comm_plan_->mpi_comm_graph);
        
        copy_recv_buffer();

        this_array_.update_device();
    };

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPICArrayKokkos ();
}; // End of MPIDArrayKokkos

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos()
    : this_array_(), stride_(1) { }

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, const std::string& tag_string) 
    : stride_(1) {
    this_array_ = DCArrayKokkos<T>(dim0, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string) 
    : stride_(dim1) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1);
}

// Overloaded 3D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string) 
    : stride_(dim1 * dim2) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2);
}

// Overloaded 4D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) 
    : stride_(dim1 * dim2 * dim3) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) 
    : stride_(dim1 * dim2 * dim3 * dim4) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, dim4, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3, dim4);
}

// Overloaded 6D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string) 
    : stride_(dim1 * dim2 * dim3 * dim4 * dim5) {
    this_array_ = DCArrayKokkos<T>(dim0, dim1, dim2, dim3, dim4, dim5, tag_string);
    host = ViewCArray <T> (this_array_.host_pointer(), dim0, dim1, dim2, dim3, dim4, dim5);
}

// Overloaded 7D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string) 
    : stride_(dim1 * dim2 * dim3 * dim4 * dim5 * dim6) {
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
    stride_ = temp.stride_;
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