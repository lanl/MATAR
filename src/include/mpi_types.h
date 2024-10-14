#ifndef MPI_TYPES_H
#define MPI_TYPES_H
/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/

#include "host_types.h"
#include "kokkos_types.h"
#include <typeinfo>
#ifdef HAVE_MPI
#include <mpi.h>

namespace mtr
{

/////////////////////////
// MPIArrayKokkos:  Dual type for managing distributed data on both CPU and GPU.
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIArrayKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[3];
    size_t length_;
    size_t order_;  // tensor order (rank)
    int rank_;
    int world_size_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;
    TArray1D this_array_;
    DCArrayKokkos <T> send_n_, recv_n_, send_s_, recv_s_, send_w_, recv_w_, send_e_, recv_e_;
    
    void set_mpi_type();

    void mpi_decomp(int world_size, int rank, int halos, MPI_Comm comm);

    void mpi_halo_update();

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPIArrayKokkos();
    
    MPIArrayKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    MPIArrayKokkos& operator=(const MPIArrayKokkos& temp);

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
    TArray1D get_kokkos_dual_view() const;

    // Method that update host view
    void update_host();

    // Method that update device view
    void update_device();

    // MPI send wrapper
    void send(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void recv(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI broadcast wrapper
    void broadcast(size_t count, int root, MPI_Comm comm);

    // MPI scatter wrapper
    void scatter(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm);

    // MPI gather wrapper
    void gather(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm);

    // MPI allgather wrapper
    void allgather(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, MPI_Comm comm);

    // MPI send wrapper
    void isend(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void irecv(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI wait wrapper for sender
    void wait_send();

    // MPI wait wrapper for receiver
    void wait_recv();

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPIArrayKokkos ();
}; // End of MPIArrayKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIArrayKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 3; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIArrayKokkos(size_t dim0, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    set_mpi_type();
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIArrayKokkos(size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
    if (typeid(T).name() == typeid(bool).name()) {
        mpi_datatype_ = MPI_C_BOOL;
    }
    else if (typeid(T).name() == typeid(int).name()) {
        mpi_datatype_ = MPI_INT;
    }
    else if (typeid(T).name() == typeid(long int).name()) {
        mpi_datatype_ = MPI_LONG;
    }
    else if (typeid(T).name() == typeid(long long int).name()) {
        mpi_datatype_ = MPI_LONG_LONG_INT;
    }
    else if (typeid(T).name() == typeid(float).name()) {
        mpi_datatype_ = MPI_FLOAT;
    }
    else if (typeid(T).name() == typeid(double).name()) {
        mpi_datatype_ = MPI_DOUBLE;
    }
    else {
        printf("Your entered MPIArrayKokkos type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIArrayKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIArrayKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPIArrayKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIArrayKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIArrayKokkos 2D!");
    return this_array_.d_view(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPIArrayKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIArrayKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIArrayKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIArrayKokkos 3D!");
    return this_array_.d_view(k + (j * dims_[2])
                                + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIArrayKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        send_n_ = temp.send_n_;
        recv_n_ = temp.recv_n_;
        send_s_ = temp.send_s_;
        recv_s_ = temp.recv_s_;
        send_w_ = temp.send_w_;
        recv_w_ = temp.recv_w_;
        send_e_ = temp.send_e_;
        recv_e_ = temp.recv_e_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPIArrayKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to MPIArrayKokkos dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

//MPI_Send wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::send(size_t count, int dest, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Send(device_pointer(), count, mpi_datatype_, dest, tag, comm); 
#else
    update_host();
    MPI_Send(host_pointer(), count, mpi_datatype_, dest, tag, comm); 
#endif
}

//MPI_Recv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::recv(size_t count, int source, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Recv(device_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
#else
    MPI_Recv(host_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
    update_device();
#endif
}

//MPI_Bcast wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::broadcast(size_t count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Bcast(device_pointer(), count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Bcast(host_pointer(), count, mpi_datatype_, root, comm); 
    update_device();
#endif
}

//MPI_Scatter wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::scatter(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Scatter(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Scatter(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, root, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_Gather wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::gather(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Gather(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Gather(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, root, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_AllGather wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::allgather(size_t send_count, MPIArrayKokkos recv_buffer, size_t recv_count, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Allgather(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, comm); 
#else
    update_host();
    MPI_Allgather(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_Isend wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::isend(size_t count, int dest, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Isend(device_pointer(), count, mpi_datatype_, dest, tag, comm, &mpi_request_); 
#else
    update_host();
    MPI_Isend(host_pointer(), count, mpi_datatype_, dest, tag, comm, &mpi_request_); 
#endif
}

//MPI_Irecv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::irecv(size_t count, int source, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Irecv(device_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_request_); 
#else
    MPI_Irecv(host_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_request_); 
#endif
}

//MPI_Wait wrapper for the sender
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_send() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
}

//MPI_Wait wrapper for the receiver
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_recv() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
#ifndef HAVE_GPU_AWARE_MPI
    update_device();
#endif
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_decomp(int world_size, int rank, int halos, MPI_Comm comm) {
    int neighbors = 0;
    int neighb, tag;
    rank_ = rank;
    world_size_ = world_size;
    if (order_ == 1) {

    }
    if (order_ == 2) {
        int mpi_dim_size = sqrt(world_size);
        int world_i = rank % mpi_dim_size;
        int world_j = rank / mpi_dim_size;
        int j_n = world_j - 1;
        int j_s = world_j + 1;
        int i_w = world_i - 1;
        int i_e = world_i + 1;

        // setup north
        send_n_ = DCArrayKokkos <T> (dims_[0]);
        recv_n_ = DCArrayKokkos <T> (dims_[0]);
        neighb = j_n * mpi_dim_size + world_i;
        tag = rank * 10 + neighb;
        send_n_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        recv_n_.mpi_setup(neighb, tag, halos, comm);
        // setup south
        send_s_ = DCArrayKokkos <T> (dims_[0]);
        recv_s_ = DCArrayKokkos <T> (dims_[0]);
        neighb = j_s * mpi_dim_size + world_i;
        tag = rank * 10 + neighb;
        send_s_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        recv_s_.mpi_setup(neighb, tag, halos, comm);
        // setup west
        send_w_ = DCArrayKokkos <T> (dims_[1]);
        recv_w_ = DCArrayKokkos <T> (dims_[1]);
        neighb = world_j * mpi_dim_size + i_w;
        tag = rank * 10 + neighb;
        send_w_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        recv_w_.mpi_setup(neighb, tag, halos, comm);
        // setup east
        send_s_ = DCArrayKokkos <T> (dims_[1]);
        recv_s_ = DCArrayKokkos <T> (dims_[1]);
        neighb = world_j * mpi_dim_size + i_e;
        tag = rank * 10 + neighb;
        send_s_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        recv_s_.mpi_setup(neighb, tag, halos, comm);
    }
    if (order_ == 3) {

    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_halo_update() {
    if (order_ == 1) {

    }
    if (order_ == 2) {
        int halo = send_n_.get_halo(); // just need one of them, all have same number of halos
        if (send_n_.get_neighbor() >= 0) {
            Kokkos::parallel_for("haloupdatenorth", dims_[0], KOKKOS_CLASS_LAMBDA(const int hh) {
                send_n_(hh) = this_array_(hh+halo, halo+(halo-1)); // row depending on # of halos
            }); 
            Kokkos::fence();
            send_n_.halo_isend();
            recv_n_.halo_irecv();
        }
        if (send_s_.get_neighbors() >= 0) {
            Kokkos::parallel_for("haloupdatesouth", dims_[0], KOKKOS_CLASS_LAMBDA(const int hh) {
                send_s_(hh) = this_array_(hh+halo, dims_[1]); // row depending on # of halos
            }); 
            Kokkos::fence();
            send_s_.halo_isend();
            recv_s_.halo_irecv();
        }
        if (send_w_.get_neighbors() >= 0) {
            Kokkos::parallel_for("haloupdatewest", dims_[1], KOKKOS_CLASS_LAMBDA(const int hh) {
                send_w_(hh) = this_array_(hh+halo, halo+(halo-1)); // column depending on # of halos
            }); 
            Kokkos::fence();
            send_w_.halo_isend();
            recv_w_.halo_irecv();
        }
        if (send_e_.get_neighbors() >= 0) {
            Kokkos::parallel_for("haloupdateeast", dims_[1], KOKKOS_CLASS_LAMBDA(const int hh) {
                send_e_(hh) = this_array_(hh+halo, dims_[0]); // column depending on # of halos
            }); 
            Kokkos::fence();
            send_e_.halo_isend();
            recv_e_.halo_irecv();
        }
    }
    if (order_ == 3) {

    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPIArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// MPIHaloKokkos:  Really only used for internal comms in the original MPIHaloKokkos class
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIHaloKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    // mpi arrays below are all order north, south, west, east
    int mpi_neighbor_;
    int mpi_tag_;
    int mpi_halos_;
    MPI_Comm mpi_comm_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;
    TArray1D this_array_;
    
    void set_mpi_type();

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPIHaloKokkos();
    
    MPIHaloKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5,
                 size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    // These functions can setup the data needed for halo send/receives
    // Not necessary for standard MPI comms
    void mpi_setup(int neighbor, int tag, int halos, MPI_Comm comm);

    int get_neighbor();

    int get_tag();

    int get_halos();

    MPI_Comm get_comm();

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
    MPIHaloKokkos& operator=(const MPIHaloKokkos& temp);

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
    TArray1D get_kokkos_dual_view() const;

    // Method that update host view
    void update_host();

    // Method that update device view
    void update_device();

    // MPI send wrapper
    void halo_send();

    // MPI recieve wrapper
    void halo_recv();

    // MPI send wrapper
    void halo_isend();

    // MPI recieve wrapper
    void halo_irecv();

    // MPI wait wrapper for sender
    void wait_send();

    // MPI wait wrapper for receiver
    void wait_recv();

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPIHaloKokkos ();
}; // End of MPIHaloKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = (dim0 * dim1 * dim2 * dim3);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4, dim5);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5,
                              size_t dim6, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4, dim5, dim6);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
    if (typeid(T).name() == typeid(bool).name()) {
        mpi_datatype_ = MPI_C_BOOL;
    }
    else if (typeid(T).name() == typeid(int).name()) {
        mpi_datatype_ = MPI_INT;
    }
    else if (typeid(T).name() == typeid(long int).name()) {
        mpi_datatype_ = MPI_LONG;
    }
    else if (typeid(T).name() == typeid(long long int).name()) {
        mpi_datatype_ = MPI_LONG_LONG_INT;
    }
    else if (typeid(T).name() == typeid(float).name()) {
        mpi_datatype_ = MPI_FLOAT;
    }
    else if (typeid(T).name() == typeid(double).name()) {
        mpi_datatype_ = MPI_DOUBLE;
    }
    else {
        printf("Your entered MPIHaloKokkos type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 2D!");
    return this_array_.d_view(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 3D!");
    return this_array_.d_view(k + (j * dims_[2])
                                + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 4D!");
    return this_array_.d_view(l + (k * dims_[3])
                                + (j * dims_[3] * dims_[2])
                                + (i * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 5D!");
    return this_array_.d_view(m + (l * dims_[4])
                                + (k * dims_[4] * dims_[3])
                                + (j * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIHaloKokkos 6D!");
    return this_array_.d_view(n + (m * dims_[5])
                                + (l * dims_[5] * dims_[4])
                                + (k * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIHaloKokkos 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in MPIHaloKokkos 7D!");
    return this_array_.d_view(o + (n * dims_[6])
                                + (m * dims_[6] * dims_[5])
                                + (l * dims_[6] * dims_[5] * dims_[4])
                                + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIHaloKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        mpi_neighbor_ = temp.mpi_neighbor_;
        mpi_tag_ = temp.mpi_tag_;
        mpi_comm_ = temp.mpi_comm_;
        mpi_status_ = temp.mpi_status_;
        mpi_datatype_ = temp.mpi_datatype_;
        mpi_request_ = temp.mpi_request_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPIHaloKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to MPIHaloKokkos dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_setup(int neighbor, int tag, int halos, MPI_Comm comm) {
    mpi_neighbor_ = neighbor;
    mpi_tag_ = tag;
    mpi_halos_ = halos;
    mpi_comm_ = comm;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_neighbor() {
    return mpi_neighbor_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_tag() {
    return mpi_tag_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_halos() {
    return mpi_halos_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPI_Comm MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_comm() {
    return mpi_comm_;
}

//MPI_iSend halo wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::halo_isend() {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Isend(device_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#else
    update_host();
    MPI_Isend(host_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#endif
}

//MPI_iRecv halo wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::halo_irecv() {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Irecv(device_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#else
    MPI_Irecv(host_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#endif
}

//MPI_Wait wrapper for the sender
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_send() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
}

//MPI_Wait wrapper for the receiver
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_recv() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
#ifndef HAVE_GPU_AWARE_MPI
    update_device();
#endif
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIHaloKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPIHaloKokkos
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI

#endif // MPI_TYPES_H

